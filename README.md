# AI 口播剪辑工作流 Skill（ai-talking-head-edit）

一份 Claude Code skill：用 Claude 把真人口播、数字人或 AI 配音素材，剪成完整的知识类视频。

## 来源与原创声明

**本仓库的工作流、方法和「剪气口 / 挑 Take」提示词，均为「大师的AI小灶」原创。**

- 原视频：[《全网最深度的 AI 剪辑教程》](https://x.com/dashiAIxz/status/2122492466490433678)（X，2026-11-17）
- 原作者：大师的AI小灶，X [@dashiAIxz](https://x.com/dashiAIxz) · GitHub [chuspeeism](https://github.com/chuspeeism) · 抖音 / 小红书 / B站同名
- 原作者的分镜动画 skill：[chuspeeism/dashi-animationstyle](https://github.com/chuspeeism/dashi-animationstyle)

本仓库只是看完视频后整理的学习笔记，结构化成可以直接调用的 skill，方便大家照着做。
整理时补充的内容（替代方案、实现提示、版权提醒）都在文中单独标了出来。
视频里被字幕遮挡的一行提示词是整理者推断补齐的，文中已注明。

如果原作者希望修改署名，或者删除本仓库，请提 Issue，我们会立即处理。
想看完整讲解和演示，请去原视频支持原作者。

## 内容

```
skills/ai-talking-head-edit/SKILL.md
```

流程：素材准备（脚本表 + 口播）→ 视频生成（调色参考图、动画分镜、BGM、音效、先做开头 30 秒）→ 剪气口粗剪 → 表格化反馈循环 → 导出剪映工程交付。

## 使用

把 `skills/ai-talking-head-edit` 文件夹复制到项目的 `.claude/skills/` 下，或者复制到 `~/.claude/skills/` 下。
然后对 Claude 说「帮我把这段口播剪成片」即可。

## 许可

见 [LICENSE](LICENSE)。原创内容版权归原作者所有。
