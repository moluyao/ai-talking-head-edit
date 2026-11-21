#!/usr/bin/env python3
"""口播粗剪：按实测有声边界压缩停顿（实验性，整理者补充，非原视频内容）。

停顿规则来自「大师的AI小灶」剪气口提示词的【留白与停顿】一节：
  开口前留 0.15s，说完后留 0.2s；0.35s 以内的自然停顿保留；
  超过 0.35s：句内犹豫≈0.4s，逗号 0.35~0.45s（取 0.4），句号 0.55s，换段 0.7s；
  剪掉口癖后的句内拼接处只留 0.04s。

分工：脚本只做测量和机械剪切；哪段是重复/重说、每个停顿属于逗号还是句号，
由 Claude 读转写后改 pauses.json 里的 label / drops 决定。

用法：
  python rough_cut.py analyze 口播.mp4 -o pauses.json     # 测量有声段和停顿，按时长给默认标签
  （Claude / 人工检查并修改 pauses.json）
  python rough_cut.py cut 口播.mp4 pauses.json -o 粗剪.mp4  # 按标签压缩停顿、删除 drops，输出视频 + 交付说明

依赖：Python 3.9+、numpy、ffmpeg/ffprobe（在 PATH 中）。
"""
import argparse, json, os, re, subprocess, sys, tempfile
import numpy as np

SR = 16000
HOP = 0.01          # 10ms 一帧
LEAD, TAIL = 0.15, 0.20
KEEP_MAX = 0.35
TARGET = {"hesitation": 0.40, "comma": 0.40, "period": 0.55, "paragraph": 0.70, "splice": 0.04}
LABELS = set(TARGET) | {"keep"}
AUDIO_CODEC = {".m4a": ["-c:a", "aac", "-b:a", "192k"], ".aac": ["-c:a", "aac", "-b:a", "192k"],
               ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"], ".wav": ["-c:a", "pcm_s16le"],
               ".flac": ["-c:a", "flac"], ".ogg": ["-c:a", "libvorbis", "-q:a", "5"]}
AUDIO_ONLY_EXT = set(AUDIO_CODEC)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def load_audio(path):
    # aresample first_pts=0：音轨起始时间不为 0（手机/相机素材常见）时在前面补静音，
    # 让测量时间和 cut 里 trim/atrim 用的时间戳对齐
    out = run(["ffmpeg", "-v", "error", "-i", path, "-vn", "-map", "0:a:0",
               "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", str(SR),
               "-f", "s16le", "-"], stdout=subprocess.PIPE).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0


def _rate(r):
    try:
        n, d = r.split("/")
        return float(n) / float(d) if float(d) else None
    except (ValueError, AttributeError):
        return None


def probe(path):
    j = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                        "-show_format", path], stdout=subprocess.PIPE).stdout)
    streams = j.get("streams", [])
    # 跳过 mp3/m4a 里的封面图（attached_pic），它不是真正的视频
    v = next((s for s in streams if s["codec_type"] == "video"
              and not s.get("disposition", {}).get("attached_pic")), None)
    has_audio = any(s["codec_type"] == "audio" for s in streams)
    fps = vfr = None
    if v:
        fps = _rate(v.get("avg_frame_rate"))
        r = _rate(v.get("r_frame_rate"))
        vfr = bool(fps and r and abs(fps - r) / r > 0.01)
    dur = j.get("format", {}).get("duration")
    if dur is None:                         # 部分 mkv/ts 容器没有 format.duration
        ds = [float(s["duration"]) for s in streams if s.get("duration")]
        if not ds:
            sys.exit("读不出素材时长：" + path)
        dur = max(ds)
    return {"duration": float(dur), "fps": fps, "vfr": vfr,
            "has_video": v is not None, "has_audio": has_audio}


def frame_db(x):
    hop = int(SR * HOP)
    n = len(x) // hop
    fr = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
    return 20 * np.log10(rms + 1e-9)


def detect_speech(db, margin_db=10.0, min_speech=0.08, min_gap=0.12, fixed_thr=None):
    """自适应阈值。返回 [(s,e)] 秒、阈值、噪底。

    用 Otsu 法在 dB 直方图上把帧分成「静音 / 有声」两类，阈值不随静音占比漂移
    （分位数法在长停顿多时会把阈值拉低，把底噪当成人声）。
    再保证阈值至少比噪底高 margin_db，防止整段几乎全是人声时阈值落进人声里。
    """
    lo, hi = np.percentile(db, 1), np.percentile(db, 99.5)
    v = np.clip(db, lo, hi)
    hist, edges = np.histogram(v, bins=120, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(hist); w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers) / np.maximum(w0, 1)
    m1 = (np.sum(hist * centers) - np.cumsum(hist * centers)) / np.maximum(w1, 1)
    otsu = centers[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))]
    floor = float(np.median(db[db <= otsu])) if np.any(db <= otsu) else float(lo)
    thr = max(float(otsu), floor + margin_db)
    if fixed_thr is not None:            # 手动锁定阈值（同批素材 / 复测时保持一致）
        thr = float(fixed_thr)
    on = db > thr
    segs, i, n = [], 0, len(on)
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            segs.append([i * HOP, j * HOP])
            i = j
        else:
            i += 1
    merged = []
    for s, e in segs:                       # 合并过短的间隙（词内微停）
        if merged and s - merged[-1][1] < min_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [seg for seg in merged if seg[1] - seg[0] >= min_speech]
    # 双阈值：边界向外延伸到电平跌破低阈值，保住轻声起音和尾音余韵
    low = max(thr - 6.0, floor + 6.0)      # 低阈值也要明显高于噪底，否则会顺着底噪/背景声一路延伸
    n = len(db)
    QUIET = 3                               # 连续 3 帧(30ms)低于低阈值才算到头，单帧凹陷不截断
    for k, seg in enumerate(merged):
        lim = int(round(merged[k - 1][1] / HOP)) if k else 0
        i = int(round(seg[0] / HOP)); best = i; q = 0
        while i - 1 > lim and q < QUIET:
            i -= 1
            if db[i] > low:
                best, q = i, 0
            else:
                q += 1
        lim2 = int(round(merged[k + 1][0] / HOP)) if k + 1 < len(merged) else n
        j = int(round(seg[1] / HOP)); bestj = j; q = 0
        while j < lim2 - 1 and q < QUIET:
            if db[j] > low:
                bestj, q = j + 1, 0
            else:
                q += 1
            j += 1
        seg[0], seg[1] = best * HOP, bestj * HOP
    final = []
    for s, e in merged:                     # 延伸后可能相接，再合并一次
        if final and s - final[-1][1] < min_gap:
            final[-1][1] = e
        else:
            final.append([s, e])
    return final, float(thr), float(floor)


def default_label(d):
    if d <= KEEP_MAX:
        return "keep"
    if d < 0.9:
        return "comma"
    if d < 1.6:
        return "period"
    return "paragraph"


def cmd_analyze(a):
    if not probe(a.input)["has_audio"]:
        sys.exit("素材里没有音轨，无法按声音判断停顿：" + a.input)
    x = load_audio(a.input)
    db = frame_db(x)
    segs, thr, floor = detect_speech(db, a.margin_db, fixed_thr=a.threshold_db)
    if not segs:
        sys.exit("没有检测到有声段：检查音轨，或调小 --margin-db")
    pauses = []
    for k in range(len(segs) - 1):
        s, e = segs[k][1], segs[k + 1][0]
        d = round(e - s, 3)
        pauses.append({"id": k, "start": round(s, 3), "end": round(e, 3), "dur": d,
                       "label": default_label(d)})
    out = {
        "source": os.path.basename(a.input),   # 只存文件名，分享 json 时不暴露本机目录
        "threshold_db": round(thr, 1), "noise_floor_db": round(floor, 1),
        "rules": {"lead": LEAD, "tail": TAIL, "keep_max": KEEP_MAX, "targets": TARGET},
        "note": "label 可选 keep/hesitation/comma/period/paragraph/splice；默认标签只按时长猜，"
                "请对照转写改成真实标点。drops 填要整段删掉的源时间区间（重复、说一半重来、口癖）。"
                "texts 可选：与 speech 一一对应的转写文字，填了交付说明会算语速。",
        "speech": [[round(s, 3), round(e, 3)] for s, e in segs],
        "texts": [],
        "pauses": pauses,
        "drops": [],
    }
    with open(a.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    long_n = sum(p["dur"] > KEEP_MAX for p in pauses)
    print(f"有声段 {len(segs)} 个，停顿 {len(pauses)} 个（>{KEEP_MAX}s 的 {long_n} 个），阈值 {thr:.1f} dB → {a.output}")


def subtract(ranges, drops):
    out = []
    for s, e in ranges:
        pieces = [[s, e]]
        for ds, de in drops:
            nxt = []
            for ps, pe in pieces:
                if de <= ps or ds >= pe:
                    nxt.append([ps, pe])
                else:
                    if ps < ds:
                        nxt.append([ps, ds])
                    if de < pe:
                        nxt.append([de, pe])
            pieces = nxt
        out += pieces
    return out


def build_keep(info, dur):
    """先从有声段里减掉 drops，再在「剩下的相邻有声段」之间按规则重算停顿。

    拼接处（中间删过东西）的标签：
      - 删除切进了某个有声段内部（删口癖、删半个词）→ splice（只留 0.04s）
      - 删掉的是整段 → 用被保留那句话之后原本那个停顿的标签（它反映这句话的句末标点）
    停顿里保留的「空气」只从原本的静音里取，不会把被删的内容带回来。
    """
    speech = info["speech"]; pauses = info["pauses"]
    drops = info.get("drops_ranges", [])
    for p in pauses:
        if p.get("label", "keep") not in LABELS:
            sys.exit(f"pause {p.get('id')} 的 label 不认识：{p.get('label')}")
    pieces = []
    for k, (s, e) in enumerate(speech):
        for ps, pe in subtract([[s, e]], drops):
            if pe - ps >= 0.03:
                pieces.append({"s": ps, "e": pe, "k": k,
                               "s_orig": abs(ps - s) < 1e-6, "e_orig": abs(pe - e) < 1e-6})
    if not pieces:
        sys.exit("drops 删光了全部有声内容，没有可输出的部分")

    def pause_len(k):                       # speech k 和 k+1 之间原本的静音长度
        return speech[k + 1][0] - speech[k][1] if 0 <= k < len(speech) - 1 else 0.0

    first, last = pieces[0], pieces[-1]
    before = first["s"] - (speech[first["k"] - 1][1] if first["k"] > 0 else 0.0)
    lead_avail = min(before, first["s"]) if first["s_orig"] else 0.0
    after = (speech[last["k"] + 1][0] if last["k"] + 1 < len(speech) else dur) - last["e"]
    tail_avail = max(0.0, min(after, dur - last["e"])) if last["e_orig"] else 0.0
    # 开头/结尾的留白也不能伸进被删除的区间
    lead_s = max(first["s"] - min(LEAD, lead_avail), 0.0)
    for ds, de in drops:
        if ds < first["s"] and de > lead_s:
            lead_s = max(lead_s, min(de, first["s"]))
    tail_e = last["e"] + min(TAIL, tail_avail)
    for ds, de in drops:
        if de > last["e"] and ds < tail_e:
            tail_e = min(tail_e, max(ds, last["e"]))
    keep = [[lead_s, first["s"]]]
    plan = []
    for i, pc in enumerate(pieces):
        keep.append([pc["s"], pc["e"]])
        if i == len(pieces) - 1:
            break
        nx = pieces[i + 1]
        g = nx["s"] - pc["e"]
        joined = nx["k"] != pc["k"] + 1 or not pc["e_orig"] or not nx["s_orig"] or any(
            ds < nx["s"] and de > pc["e"] for ds, de in drops)
        if nx["k"] == pc["k"] or not pc["e_orig"] or not nx["s_orig"]:
            lab = "splice"
        else:
            lab = pauses[pc["k"]].get("label", "keep")
        plen = pause_len(pc["k"])
        if lab == "splice":
            t = TARGET["splice"]
        elif lab == "keep" or plen <= KEEP_MAX:
            t = plen
        else:
            t = TARGET[lab]
        t = min(t, g)
        # 空气只从真实静音里取：前段来自这句之后的停顿，后段来自下一句之前的停顿
        f_av = plen if pc["e_orig"] else t
        b_av = pause_len(nx["k"] - 1) if nx["s_orig"] else t
        for ds, de in drops:                # 不能伸进被删除的区间
            if pc["e"] <= ds < pc["e"] + f_av:
                f_av = ds - pc["e"]
            if nx["s"] - b_av < de <= nx["s"]:
                b_av = nx["s"] - de
        a_ = min(t * 0.6, max(f_av, 0.0))   # 前段多留，保住尾音余韵
        b_ = min(t - a_, max(b_av, 0.0))
        a_ = min(t - b_, max(f_av, 0.0))
        if a_ + b_ >= g - 1e-6:
            keep.append([pc["e"], nx["s"]])
        else:
            if a_ > 0: keep.append([pc["e"], pc["e"] + a_])
            if b_ > 0: keep.append([nx["s"] - b_, nx["s"]])
        plan.append({"id": pc["k"], "at": round(pc["e"], 3), "orig": round(g, 3),
                     "new": round(min(a_ + b_, g), 3), "label": lab, "joined": joined})
    keep.append([last["e"], tail_e])
    keep.sort()
    merged = []
    for s, e in keep:                       # 合并相接的区间，减少切点
        if merged and s - merged[-1][1] < 1e-3:
            merged[-1][1] = max(merged[-1][1], e)
        elif e - s > 1e-3:
            merged.append([s, e])
    return merged, plan


def snap(t, fps):
    return round(t * fps) / fps if fps else t


def fmt(t):
    m, s = divmod(round(t, 3), 60)
    return f"{int(m):02d}:{s:06.3f}"


def cmd_cut(a):
    info = json.load(open(a.pauses, encoding="utf-8"))
    meta = probe(a.input)
    if not meta["has_audio"]:
        sys.exit("素材里没有音轨：" + a.input)
    ext = os.path.splitext(a.output)[1].lower()
    if ext in AUDIO_ONLY_EXT:               # 输出纯音频格式时不带画面
        meta["has_video"] = False
    info["drops_ranges"] = [[float(d[0]), float(d[1])] for d in info.get("drops", [])]
    if len(info["pauses"]) != len(info["speech"]) - 1:
        sys.exit("pauses 数量应等于 speech 数量 - 1；请不要增删 pauses 条目，只改 label")
    keep, plan = build_keep(info, meta["duration"])
    fps = meta["fps"]
    if meta["has_video"] and fps and not meta["vfr"]:   # 恒定帧率：切点对齐到帧，防止音画漂移
        keep = [[snap(s, fps), snap(e, fps)] for s, e in keep]
        keep = [k for k in keep if k[1] - k[0] >= 1.0 / fps]
    if not keep:
        sys.exit("剪完没有剩下任何内容（drops 是否删光了？）")
    fade = 0.004
    lines = []
    for i, (s, e) in enumerate(keep):
        if meta["has_video"]:
            lines.append(f"[0:v]trim=start={s:.6f}:end={e:.6f},setpts=PTS-STARTPTS[v{i}];")
        lines.append(f"[0:a]atrim=start={s:.6f}:end={e:.6f},asetpts=PTS-STARTPTS,"
                     f"afade=t=in:d={fade},afade=t=out:st={max(0, e - s - fade):.6f}:d={fade}[a{i}];")
    ins = "".join((f"[v{i}][a{i}]" if meta["has_video"] else f"[a{i}]") for i in range(len(keep)))
    lines.append(f"{ins}concat=n={len(keep)}:v={1 if meta['has_video'] else 0}:a=1"
                 + ("[v][a]" if meta["has_video"] else "[a]"))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines)); script = f.name
    tail = []
    if meta["has_video"]:
        tail += ["-map", "[v]", "-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium", "-pix_fmt", "yuv420p"]
        if meta["vfr"]:
            tail += ["-fps_mode", "vfr"]
    tail += ["-map", "[a]"] + AUDIO_CODEC.get(ext, ["-c:a", "aac", "-b:a", "192k"]) + [a.output]
    try:
        # ffmpeg ≥7 用 -/filter_complex <文件>；旧版只认 -filter_complex_script
        errs = []
        for opt in (["-/filter_complex", script], ["-filter_complex_script", script]):
            r = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.input] + opt + tail,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 0:
                break
            errs.append(r.stderr[-1500:])
            # 只有「不认识这个参数」才换旧写法重试；滤镜本身出错直接报，不误导
            if not re.search(r"[Uu]nrecognized option '-?/?filter_complex", r.stderr):
                sys.exit("ffmpeg 失败：\n" + r.stderr[-2000:])
        else:
            sys.exit("ffmpeg 不支持滤镜脚本参数：\n" + "\n---\n".join(errs))
    finally:
        os.unlink(script)
    # 源时间 → 成片时间 映射，写交付说明
    def to_out(t):
        acc = 0.0
        for s, e in keep:
            if t < s:
                return acc
            if t <= e:
                return acc + t - s
            acc += e - s
        return acc
    total = sum(e - s for s, e in keep)
    rep = [f"# 粗剪交付说明\n", f"- 源文件：`{a.input}`（{meta['duration']:.2f}s）",
           f"- 成片：`{a.output}`（{total:.2f}s，缩短 {meta['duration'] - total:.2f}s）",
           (f"- 切点 {len(keep) - 1} 处，已对齐到 {fps:.3f}fps 帧边界" if meta["has_video"] and fps and not meta["vfr"]
            else f"- 切点 {len(keep) - 1} 处（可变帧率素材，未对齐帧）" if meta["has_video"] else "- 纯音频输出"),
           "\n## 删除的区间（drops）\n", "| 源时间 | 成片位置 | 原因 |", "|---|---|---|"]
    for d in info.get("drops", []):
        rep.append(f"| {fmt(float(d[0]))}–{fmt(float(d[1]))} | {fmt(to_out(float(d[0])))} | {d[2] if len(d) > 2 else ''} |")
    rep += ["\n## 被压缩的停顿 / 拼接点\n", "| id | 源时间 | 成片时间 | 标签 | 原长→新长 | 备注 |", "|---|---|---|---|---|---|"]
    for p in plan:
        if abs(p["orig"] - p["new"]) > 1e-3 or p["joined"]:
            rep.append(f"| {p['id']} | {fmt(p['at'])} | {fmt(to_out(p['at']))} | {p['label']} | "
                       f"{p['orig']:.2f}→{p['new']:.2f}s | {'删除后拼接，请听一下是否自然' if p['joined'] else ''} |")
    review = []
    texts = info.get("texts") or []
    if texts:                               # 语速：原提示词【交付】要求列出
        if len(texts) != len(info["speech"]):
            sys.exit("texts 的条数应与 speech 相同（每个有声段一条，没有就填空字符串）")
        rows = []
        for k, ((s, e), tx) in enumerate(zip(info["speech"], texts)):
            kept = sum(max(0.0, min(e, ke) - max(s, ks)) for ks, ke in keep)
            n = len(re.sub(r"[\s\W_]", "", tx or "", flags=re.UNICODE))
            if n and kept > 0.2:
                rows.append((k, s, n, kept, n / kept))
        if rows:
            med = float(np.median([r[4] for r in rows]))
            tot_n = sum(r[2] for r in rows); tot_t = sum(r[3] for r in rows)
            rep += ["\n## 语速\n", f"- 有声部分平均 {tot_n / tot_t:.2f} 字/秒（中位数 {med:.2f}），"
                    f"含停顿整片 {tot_n / total:.2f} 字/秒",
                    "\n| 段 | 成片时间 | 字数 | 时长 | 字/秒 | 备注 |", "|---|---|---|---|---|---|"]
            for k, s, n, t, r in rows:
                flag = "偏快" if r > med * 1.35 else "偏慢" if r < med * 0.65 else ""
                if flag:
                    review.append(f"- 第 {k} 段（成片 {fmt(to_out(s))}）语速{flag}：{r:.2f} 字/秒，"
                                  "可能是吞音、抢话，或转写文字和有声段没对齐")
                rep.append(f"| {k} | {fmt(to_out(s))} | {n} | {t:.2f}s | {r:.2f} | {flag} |")
    for p in plan:
        if p["joined"]:
            review.append(f"- 拼接点（成片 {fmt(to_out(p['at']))}）：听一下前后语气和气口是否连贯")
    rep += ["\n## 需复查清单\n"] + (review or ["- （无自动发现的问题）"])
    rep += ["\n## 需要人工确认\n", "- 默认标签只按停顿时长猜测，未对照转写修改的停顿可能把逗号当句号。",
            "- 吸气、咳嗽如果响度接近人声，会被当成有声段保留。",
            "- 挑 Take（及原因）、重复/重说判断、跨 Take 拼接不在本脚本范围，由 Claude 按 prompts/03 处理并补进本说明。"]
    rp = os.path.splitext(a.output)[0] + "_交付说明.md"
    open(rp, "w", encoding="utf-8").write("\n".join(rep) + "\n")
    print(f"{len(keep)} 段 → {a.output}（{total:.2f}s），交付说明 → {rp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("analyze"); p1.add_argument("input"); p1.add_argument("-o", "--output", default="pauses.json")
    p1.add_argument("--margin-db", type=float, default=10.0, help="阈值至少高于噪底多少 dB（默认 10）")
    p1.add_argument("--threshold-db", type=float, default=None, help="手动指定有声阈值 dBFS，覆盖自动阈值")
    p2 = sub.add_parser("cut"); p2.add_argument("input"); p2.add_argument("pauses"); p2.add_argument("-o", "--output", default="rough_cut.mp4")
    p2.add_argument("--crf", type=int, default=18)
    try:                                    # Windows 管道默认 GBK，路径里有生僻字时 print 会崩
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    a = ap.parse_args()
    {"analyze": cmd_analyze, "cut": cmd_cut}[a.cmd](a)


if __name__ == "__main__":
    main()
