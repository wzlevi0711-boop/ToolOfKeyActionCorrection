# ToolOfKeyActionCorrection

CT 读片眼动实验的辅助工具集，包含两套工具：

## 1. 标注动作校正播放器（annotation_player.py）

图形界面播放器：加载已检测出的「标注 / 删除」动作事件 + 眼动录屏视频，逐条播放核对、标记对错、补充漏检关键帧，最后导出正式的核对表格。

- 运行：`python annotation_player.py`
- 依赖：PySide6、numpy、pandas、opencv-python
- 详细说明见 [annotation_segments/README_视频插件.md](annotation_segments/README_视频插件.md)

## 2. 病例自动切分工具（case切分工具/）

读医生操作鼠标的键鼠日志，利用「双击打开病例」+「列表滚动 y 坐标规律」自动把整段录屏切成 60 个病例片段（3 组 × 20），并生成带截图的 HTML 报告供人工核对。

- 启动：双击 `case切分工具/启动切分工具.bat`（图形界面）
- 详细说明见 [annotation_segments/case切分工具/README.md](annotation_segments/case切分工具/README.md)

## 目录结构

```
annotation_segments/
├── annotation_player.py         # 标注动作校正播放器
├── extract_events.py            # 三类事件提取脚本
├── README_视频插件.md            # 播放器操作说明
├── events/
│   └── annotation_events.csv    # 待核对的标注/删除动作事件
├── verification/                # 核对结果与关键帧
└── case切分工具/                 # 病例自动切分工具（脚本+界面+结果+说明）
```

## 数据说明

- 原始数据（键鼠日志 `*.jsonl`、录屏视频 `raw.mp4/overlay.mp4`、眼动 `raw.csv`）体积很大，保存在本地 `AI读片原数据\`，不入库。
- 视频与眼动数据已通过 `.gitignore` 排除。
