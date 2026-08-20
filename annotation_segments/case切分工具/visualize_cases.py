"""
切分结果可视化: 生成自包含 HTML 报告 (每个 case 边界带视频截图)。

用法:
  python visualize_cases.py <参与者目录> [--out out.html] [--gap-sec 4]
  python visualize_cases.py --all <数据根目录> [--gap-sec 4]
"""
import argparse
import base64
import os
import sys
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_cases as sc


def load_events(path):
    ev = []
    for line in open(path, encoding='utf-8'):
        line = line.strip()
        if not line:
            continue
        try:
            ev.append(__import__('json').loads(line))
        except Exception:
            pass
    return ev


def find_video(participant_dir):
    eye_dir = os.path.join(participant_dir, "眼动")
    if not os.path.isdir(eye_dir):
        return None
    for root, dirs, files in os.walk(eye_dir):
        for f in files:
            if f.lower() == "raw.mp4":
                return os.path.join(root, f)
    return None


def frame_to_b64(frame, max_w=560):
    h, w = frame.shape[:2]
    if w > max_w:
        h = int(h * max_w / w)
        w = max_w
        frame = cv2.resize(frame, (w, h))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode()


def generate_report(participant_dir, out=None, gap_sec=0.0, video=None):
    pdir = os.path.abspath(participant_dir)
    name = os.path.basename(pdir.rstrip("\\/"))
    jl = sc.find_keymouse_jsonl(pdir)
    if not jl:
        return None

    dbl = sc.detect_double_clicks(load_events(jl))
    dbl = sc.merge_reclicks(dbl)
    band, chain, bands = sc.best_sawtooth_band(dbl, gap=80, step_range=(10, 80))
    groups = sc.chain_groups(chain)
    video_start = sc.find_eye_video_start(pdir)
    cases = sc.build_cases(chain, video_start=video_start)

    video = video or find_video(pdir)
    cap = None
    fps = 30.0
    if video and os.path.exists(video):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    def grab_frame(video_sec):
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, video_sec * fps)))
        ok, frame = cap.read()
        return frame if ok else None

    cards = []
    for c in cases:
        vs = c.get("start_video_sec")
        img = None
        if vs is not None and vs >= 0:
            f = grab_frame(vs + gap_sec)
            if f is not None:
                img = frame_to_b64(f)
        cards.append({
            "order": c["order"],
            "start_ts": c["start_ts"],
            "video_sec": vs,
            "duration": c.get("duration_sec"),
            "open_x": c["open_x"],
            "open_y": c["open_y"],
            "img": img,
        })

    if cap is not None:
        cap.release()

    gcounts = [len(g) for g in groups]
    vid_info = f"视频: {os.path.basename(video)}" if video else "视频: (未找到)"
    band_info = (f"x带: [{min(c['x'] for c in band):.0f},{max(c['x'] for c in band):.0f}]"
                 if band else "")
    vs_start = (datetime.fromtimestamp(video_start).strftime('%H:%M:%S')
                if video_start else "?")
    html = [f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>{name} 切分结果</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#16161a;color:#ddd;margin:20px}}
h1{{color:#fff}} .meta{{color:#999;margin:6px 0}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}}
.card{{background:#22222a;border:1px solid #333;border-radius:8px;padding:8px;width:560px}}
.card img{{width:100%;display:block;border-radius:4px}}
.card .info{{font-size:13px;margin-top:6px;line-height:1.5}}
.card .order{{color:#4da3ff;font-weight:bold}}
</style></head><body>
<h1>{name} — case 切分 (共 {len(cases)} 个, 分组 {gcounts})</h1>
<div class="meta">{vid_info} | 病例列表{band_info} | 视频起始 {vs_start}</div>
<div class="grid">"""]
    for c in cards:
        dur_s = f"{c['duration']:.0f}s" if c['duration'] is not None else "末"
        vs = f"{c['video_sec']:.1f}s" if c['video_sec'] is not None else "?"
        img_tag = (f'<img src="data:image/jpeg;base64,{c["img"]}">'
                   if c["img"] else '<div style="height:200px;color:#666;display:flex;align-items:center;justify-content:center">无帧</div>')
        html.append(f"""<div class="card">
{img_tag}
<div class="info"><span class="order">#{c['order']}</span> &nbsp; {c['start_ts']}<br>
video {vs} · 时长 {dur_s} · 双击点 ({c['open_x']:.0f},{c['open_y']:.0f})</div>
</div>""")
    html.append("</div></body></html>")

    out = out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果", f"{name}_report.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(html))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("participant_dir", help="参与者目录, 或配合 --all 用数据根目录")
    ap.add_argument("--all", action="store_true", help="处理根目录下所有参与者")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gap-sec", type=float, default=0.0,
                    help="截图相对双击时刻的偏移秒 (默认0=双击瞬间)")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    if args.all:
        base = os.path.abspath(args.participant_dir)
        dirs = sorted(d for d in os.listdir(base)
                      if os.path.isdir(os.path.join(base, d)))
        for d in dirs:
            out = generate_report(os.path.join(base, d), gap_sec=args.gap_sec)
            print(f"{'OK ' if out else '跳过'} {d} -> {out}")
    else:
        out = generate_report(args.participant_dir, out=args.out,
                              gap_sec=args.gap_sec, video=args.video)
        print(f"已生成: {out}")


if __name__ == "__main__":
    main()
