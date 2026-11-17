#!/usr/bin/env python3
"""rough_cut.py 的自动测试：不需要任何真实素材。

合成一段「类语音」音频（带音节起伏和词内微停的调制噪声 + -60dB 底噪），
按已知长度插入停顿，配上 30fps 测试画面。跑 analyze → cut 后再对成片 analyze，
逐个量停顿是否压到规则值（误差 ≤ 0.05s），并检查 drops 删除和音画时长一致。

用法：python tests/test_rough_cut.py
"""
import json, os, shutil, subprocess, sys, tempfile, wave
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")   # Windows 控制台/管道默认 GBK
except (AttributeError, ValueError):
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "skills", "ai-talking-head-edit", "scripts", "rough_cut.py")
SR = 48000
TOL = 0.05
rng = np.random.default_rng(7)


def speech_chunk(dur):
    """调制噪声模拟说话：4~6Hz 音节包络，中间夹 0.05s 词内微停。"""
    n = int(dur * SR)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    # 简单带通：两次差分 + 平滑，偏向人声频段
    noise = np.convolve(np.diff(noise, prepend=0), np.ones(6) / 6, "same")
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 5 * t) ** 2
    x = 0.25 * noise * env
    mid = n // 2
    x[mid: mid + int(0.05 * SR)] *= 0.02          # 词内微停，不应被当成停顿
    fade = int(0.01 * SR)
    x[:fade] *= np.linspace(0, 1, fade); x[-fade:] *= np.linspace(1, 0, fade)
    return x


def build(path_wav):
    chunks = [1.2, 0.9, 1.6, 1.1, 2.0, 0.8, 1.3, 1.5, 1.0, 1.4]
    pauses = [0.20, 0.30, 0.50, 0.70, 1.00, 1.20, 2.00, 0.60, 1.40]
    parts = [np.zeros(int(0.5 * SR))]
    truth = []
    t = 0.5
    for k, c in enumerate(chunks):
        parts.append(speech_chunk(c)); t += c
        if k < len(pauses):
            truth.append(pauses[k])
            parts.append(np.zeros(int(pauses[k] * SR))); t += pauses[k]
    parts.append(np.zeros(int(0.6 * SR)))
    x = np.concatenate(parts)
    x += 10 ** (-60 / 20) * rng.standard_normal(len(x))   # 底噪
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path_wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
    return truth, len(x) / SR


def expected(p):
    if p <= 0.35: return p
    if p < 0.9: return 0.40
    if p < 1.6: return 0.55
    return 0.70


def analyze(src, out):
    subprocess.run([sys.executable, SCRIPT, "analyze", src, "-o", out], check=True)
    return json.load(open(out, encoding="utf-8"))


def stream_dur(path, kind):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", kind, "-show_entries",
                        "stream=duration", "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    return float(r.stdout.strip())


def main():
    fails = []
    d = tempfile.mkdtemp(prefix="rc_test_")
    wav = os.path.join(d, "a.wav"); src = os.path.join(d, "src.mp4")
    truth, total = build(wav)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=s=640x360:r=30:d={total:.3f}",
                    "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", src], check=True)

    # 1) analyze 是否量准原停顿、没有把词内微停当停顿
    info = analyze(src, os.path.join(d, "p.json"))
    got = [p["dur"] for p in info["pauses"]]
    print("源停顿 真值:", truth)
    print("源停顿 实测:", got)
    if len(got) != len(truth):
        fails.append(f"停顿个数 {len(got)} ≠ {len(truth)}")
    else:
        for g, t in zip(got, truth):
            if abs(g - t) > TOL: fails.append(f"源停顿测量 {g:.3f} vs 真值 {t:.3f}")

    # 2) cut 后成片的停顿是否符合规则
    out = os.path.join(d, "cut.mp4")
    subprocess.run([sys.executable, SCRIPT, "cut", src, os.path.join(d, "p.json"), "-o", out], check=True)
    info2 = analyze(out, os.path.join(d, "p2.json"))
    got2 = [p["dur"] for p in info2["pauses"]]
    exp2 = [expected(t) for t in truth]
    print("成片停顿 期望:", exp2)
    print("成片停顿 实测:", got2)
    if len(got2) != len(exp2):
        fails.append(f"成片停顿个数 {len(got2)} ≠ {len(exp2)}")
    else:
        for g, e in zip(got2, exp2):
            if abs(g - e) > TOL: fails.append(f"成片停顿 {g:.3f} vs 期望 {e:.3f}")
    lead = info2["speech"][0][0]; tail = stream_dur(out, "a:0") - info2["speech"][-1][1]
    print(f"开口前留 {lead:.3f}s（期望 0.15），说完后留 {tail:.3f}s（期望 0.20）")
    if abs(lead - 0.15) > TOL: fails.append(f"开口前留白 {lead:.3f}")
    if abs(tail - 0.20) > 0.06: fails.append(f"说完后留白 {tail:.3f}")
    av = abs(stream_dur(out, "v:0") - stream_dur(out, "a:0"))
    print(f"音画时长差 {av:.3f}s")
    if av > 0.05: fails.append(f"音画时长差 {av:.3f}s")

    # 3) drops：删掉第 4 个有声段（模拟重复句）。用户最自然的填法 = 该段的精确起止时间。
    #    检查：有声段少 1 个；拼接处停顿按「被保留那句之后的停顿」的标签重算，不能两段停顿叠加
    sp = info["speech"]
    info["drops"] = [[sp[3][0], sp[3][1], "测试：重复句"]]
    json.dump(info, open(os.path.join(d, "p3.json"), "w", encoding="utf-8"), ensure_ascii=False)
    out3 = os.path.join(d, "cut3.mp4")
    subprocess.run([sys.executable, SCRIPT, "cut", src, os.path.join(d, "p3.json"), "-o", out3], check=True)
    info4 = analyze(out3, os.path.join(d, "p4.json"))
    n3 = len(info4["speech"])
    print(f"drops 后有声段 {n3}（期望 {len(sp) - 1}）")
    if n3 != len(sp) - 1: fails.append(f"drops 后有声段 {n3}")
    else:
        j = info4["pauses"][2]["dur"]          # 原第 3 段和原第 5 段之间的拼接处
        e = expected(truth[2])
        print(f"拼接处停顿 {j:.3f}s（期望 {e:.2f}，即第 3 句之后原停顿的规则值）")
        if abs(j - e) > TOL: fails.append(f"拼接处停顿 {j:.3f} vs {e:.2f}")

    # 4) 音轨起始时间不为 0（手机素材常见）：测量结果应整体后移，剪出来的停顿仍正确
    off = os.path.join(d, "offset.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-itsoffset", "0.4", "-i", src,
                    "-map", "0:v", "-map", "1:a", "-c", "copy", off], check=True)
    io = analyze(off, os.path.join(d, "po.json"))
    shift = io["speech"][0][0] - info["speech"][0][0]
    print(f"音轨延后 0.4s：有声段起点后移 {shift:.3f}s")
    if abs(shift - 0.4) > TOL: fails.append(f"音轨偏移测量 {shift:.3f}")
    outo = os.path.join(d, "cut_off.mp4")
    subprocess.run([sys.executable, SCRIPT, "cut", off, os.path.join(d, "po.json"), "-o", outo], check=True)
    go = [p["dur"] for p in analyze(outo, os.path.join(d, "po2.json"))["pauses"]]
    if len(go) != len(exp2) or any(abs(g - e) > TOL for g, e in zip(go, exp2)):
        fails.append(f"音轨偏移素材剪后停顿不对：{go}")

    # 5) 带封面图的音频：不能被当成视频；输出 mp3
    cov = os.path.join(d, "cover.m4a"); png = os.path.join(d, "c.png")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "color=red:s=64x64", "-frames:v", "1", png], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", wav, "-i", png, "-map", "0:a", "-map", "1:v",
                    "-c:a", "aac", "-c:v", "png", "-disposition:v", "attached_pic", cov], check=True)
    ic = analyze(cov, os.path.join(d, "pc.json"))
    outc = os.path.join(d, "cut_cover.mp3")
    r = subprocess.run([sys.executable, SCRIPT, "cut", cov, os.path.join(d, "pc.json"), "-o", outc])
    ok = r.returncode == 0 and os.path.exists(outc) and os.path.getsize(outc) > 1000
    print("带封面音频 → mp3：", "成功" if ok else "失败")
    if not ok: fails.append("带封面音频输出 mp3 失败")

    print("\n结果：", "全部通过" if not fails else "失败")
    for f in fails: print("  -", f)
    if fails or "--keep" in sys.argv:
        print("临时文件：", d)
    else:
        shutil.rmtree(d, ignore_errors=True)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
