"""
双击检测切分脚本 (case segmentation via double-click detection)

思路:
  医生在病例列表中双击条目来打开一个 case, 每次双击的 y 坐标随列表滚动单调递增。
  解析键鼠 jsonl, 提取鼠标左键点击, 检测双击, 过滤出落在病例列表方框内的双击,
  每个方框内双击 = 一个 case 的打开时刻 = case 边界, 进而得到每个 case 的起止时间与顺序。

用法:
  python segment_cases.py <参与者目录> [--box X1 Y1 X2 Y2] [--dt-max 500] [--dist-max 15]
                              [--out-dir OUT] [--show] [--video-root ROOT]
示例:
  python segment_cases.py "C:\\Users\\wangzhi\\Desktop\\AI读片原数据\\260808_BBY_W_AI"
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# 双击判定参数默认值 (ms / px)
DT_MAX_MS = 500
# 位移阈值放宽到 50px: 医生双击时鼠标常移动几十像素 ("马虎双击")
DIST_MAX_PX = 50

# 病例列表方框默认值 (屏幕坐标, 左面板)
BOX_DEFAULT = (150, 150, 900, 650)

TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})")


def find_keymouse_jsonl(participant_dir):
    """在参与者目录下找原始键鼠 jsonl (排除 keymouse_log.json 等衍生文件)。

    键鼠目录名可能为 "键鼠" 或 "键盘", 故先查这两个目录, 再递归兜底。"""
    for km_name in ("键鼠", "键盘"):
        km_dir = os.path.join(participant_dir, km_name)
        if os.path.isdir(km_dir):
            candidates = []
            for f in os.listdir(km_dir):
                if f.lower().endswith(".jsonl") and "log" not in f.lower():
                    candidates.append(os.path.join(km_dir, f))
            if candidates:
                return max(candidates, key=os.path.getsize)
    # 递归兜底: 找全目录下所有 .jsonl, 排除派生日志
    best = None
    for root, dirs, files in os.walk(participant_dir):
        for f in files:
            if not f.lower().endswith(".jsonl"):
                continue
            if "log" in f.lower():
                continue
            p = os.path.join(root, f)
            if best is None or os.path.getsize(p) > os.path.getsize(best):
                best = p
    return best


def find_eye_video_start(participant_dir):
    """解析视频起始 unix 时间戳 (精确到毫秒)。

    优先读 眼动/<ts>/raw.csv 第一行 date_time (眼动样本首帧 = 视频第 0 帧),
    目录名只精确到秒, 会偏 1~2 秒导致截帧落到"已打开 CT"而不是"列表"。
    """
    eye_dir = os.path.join(participant_dir, "眼动")
    if not os.path.isdir(eye_dir):
        return None
    for d in os.listdir(eye_dir):
        sub = os.path.join(eye_dir, d)
        if not os.path.isdir(sub):
            continue
        m = TS_RE.match(d)
        if not m:
            continue
        # 精确: 读 raw.csv 首行 date_time
        csv = os.path.join(sub, "raw.csv")
        if os.path.exists(csv):
            try:
                with open(csv, encoding="utf-8") as f:
                    header = f.readline()
                    line = f.readline()
                # 找 date_time 列
                cols = header.strip().split(",")
                if "date_time" in cols:
                    ci = cols.index("date_time")
                    val = line.strip().split(",")[ci]
                    ts = datetime.fromisoformat(val).timestamp()
                    return ts
            except (ValueError, IndexError, OSError):
                pass
        # 回退: 目录名 (秒级)
        try:
            ts = datetime.strptime(
                f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
                "%Y-%m-%d %H:%M:%S").timestamp()
            return ts
        except ValueError:
            continue
    return None


def load_events(path):
    ev = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            ev.append(d)
    return ev


def detect_double_clicks(events, dt_max_ms=DT_MAX_MS, dist_max_px=DIST_MAX_PX):
    """检测左键双击 (按"爆发"分组, 兼容三连击)。

    返回列表, 每项:
      {t, ts, x, y, n_clicks, dt_ms, dist_px}
    """
    downs = []
    for d in events:
        if d.get("type") == "mouse_down" and d.get("button") == "left":
            ts = datetime.fromisoformat(d["ts"])
            downs.append((ts.timestamp() * 1000, d["x"], d["y"], d["ts"]))
    downs.sort(key=lambda e: e[0])

    bursts = []  # 每组爆发 = 连续间隔 < dt_max 的点击
    cur = []
    for t, x, y, ts in downs:
        if cur and (t - cur[-1][0]) <= dt_max_ms:
            cur.append((t, x, y, ts))
        else:
            if cur:
                bursts.append(cur)
            cur = [(t, x, y, ts)]
    if cur:
        bursts.append(cur)

    dbl = []
    for b in bursts:
        if len(b) < 2:
            continue
        xs = [p[1] for p in b]
        ys = [p[2] for p in b]
        span_x = max(xs) - min(xs)
        span_y = max(ys) - min(ys)
        dist = (span_x ** 2 + span_y ** 2) ** 0.5
        if dist <= dist_max_px:
            t0, x0, y0, ts0 = b[0]
            t1 = b[-1][0]
            dbl.append({
                "t": (t0 + t1) / 2 / 1000.0,          # 秒 (unix)
                "ts": b[0][3],                        # 字符串 (第一击)
                "x": sum(xs) / len(xs),
                "y": sum(ys) / len(ys),
                "n_clicks": len(b),
                "dt_ms": round(t1 - t0, 1),
                "dist_px": round(dist, 2),
            })
    return dbl


def in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def merge_reclicks(dbl_clicks, merge_sec=3.0, merge_dy=25.0):
    """合并"补双击"：同一行在短时间内重复双击 (第一次未打开再点一次)。

    返回合并后的双击列表 (保留框内/框外原始位置)。"""
    merged = []
    for c in dbl_clicks:
        if merged:
            p = merged[-1]
            dt = c["t"] - p["t"]
            dy = abs(c["y"] - p["y"])
            if dt <= merge_sec and dy <= merge_dy:
                # 视为同一 case 的补双击, 用后一次的时间作为实际打开时刻
                p["t"] = c["t"]
                p["ts"] = c["ts"]
                p["x"] = c["x"]
                p["y"] = c["y"]
                p["n_clicks"] += c["n_clicks"]
                p["reclick"] = True
                continue
        merged.append(dict(c))
    return merged


def auto_detect_box(dbl_clicks, x_gap=150, margin=30):
    """自动检测病例列表方框。

    原理: 病例列表是一个竖直面板, 双击位置在 x 上聚成一条竖直带,
          而图像区/工具区的双击分散在其它 x 位置。按 x 聚簇,
          取"点击数多且 y 跨度大"的簇作为列表区域。
    返回 (box, bands), bands 用于透明化调试。
    """
    if not dbl_clicks:
        return BOX_DEFAULT, []

    clicks = sorted(dbl_clicks, key=lambda c: c["x"])
    bands = []
    cur = []
    for c in clicks:
        if cur and c["x"] - cur[-1]["x"] > x_gap:
            bands.append(cur)
            cur = []
        cur.append(c)
    if cur:
        bands.append(cur)

    scored = []
    for b in bands:
        xs = [c["x"] for c in b]
        ys = [c["y"] for c in b]
        y_span = max(ys) - min(ys)
        # 分数: 点击数为主, y 跨度作为乘数 (竖直列表 y 跨度大)
        score = len(b) * (1.0 + y_span / 500.0)
        scored.append({
            "n": len(b),
            "y_span": round(y_span, 1),
            "x1": min(xs),
            "x2": max(xs),
            "y1": min(ys),
            "y2": max(ys),
            "score": round(score, 1),
        })
    scored.sort(key=lambda s: s["score"], reverse=True)
    best = scored[0]
    box = (best["x1"] - margin, best["y1"] - 20,
           best["x2"] + margin, best["y2"] + 20)
    return box, scored


def sawtooth_chain(dbl_clicks, step_range=(0, 80), reset_max=-60, max_gap_sec=300):
    """提取 case 打开双击 (y 锯齿链)。

    病例列表逐条下滚: 连续 case 的 y 小步递增 (步进), 或翻页/滚回顶部时大幅回跳 (reset)。
    落入中间、不符合锯齿规律的孤立双击 (图像放大/其它列) 被跳过。
    长时间无操作 (> max_gap_sec) 视为新起点, 重新开始链。
    """
    clicks = sorted(dbl_clicks, key=lambda c: c["t"])
    chain = []
    for c in clicks:
        if not chain:
            chain.append(c)
            continue
        last = chain[-1]
        if c["t"] - last["t"] > max_gap_sec:
            chain.append(c)
            continue
        dy = c["y"] - last["y"]
        if step_range[0] <= dy <= step_range[1] or dy <= reset_max:
            chain.append(c)
    return chain


def x_bands(clicks, gap=80):
    """按 x 聚簇双击 (用于找病例列表所在的竖直列)。"""
    cs = sorted(clicks, key=lambda c: c["x"])
    bands = []
    cur = []
    for c in cs:
        if cur and c["x"] - cur[-1]["x"] > gap:
            bands.append(cur)
            cur = []
        cur.append(c)
    if cur:
        bands.append(cur)
    return bands


def best_sawtooth_band(clicks, gap=80, step_range=(0, 80), reset_max=-60, max_gap_sec=300):
    """找 y 锯齿链最长的 x 带 (即病例列表所在列, 可能在左也可能在右)。"""
    bands = x_bands(clicks, gap)
    best_band, best_chain = None, []
    for b in bands:
        c = sawtooth_chain(b, step_range=step_range, reset_max=reset_max,
                           max_gap_sec=max_gap_sec)
        if len(c) > len(best_chain):
            best_band, best_chain = b, c
    return best_band, best_chain, bands


def chain_groups(chain, reset_max=-60):
    """按 y 大回跳把链切成组 (每组 = 一种病, 应为 20 个 case)。"""
    chain = sorted(chain, key=lambda c: c["t"])
    groups = []
    cur = []
    prev_y = None
    for c in chain:
        if prev_y is not None and (c["y"] - prev_y) <= reset_max:
            groups.append(cur)
            cur = []
        cur.append(c)
        prev_y = c["y"]
    if cur:
        groups.append(cur)
    return groups


def build_cases(case_clicks, video_start=None):
    """根据已提取的 case 打开双击构建 case 边界 (起止时间与顺序)。"""
    cases = []
    prev = None
    for i, c in enumerate(case_clicks):
        order = len(cases) + 1
        case = {
            "order": order,
            "start_ts": c["ts"],
            "start_t": c["t"],
            "open_x": round(c["x"], 1),
            "open_y": round(c["y"], 1),
            "n_clicks": c["n_clicks"],
            "dt_ms": c["dt_ms"],
        }
        if prev is not None:
            case["prev_y_delta"] = round(c["y"] - prev["open_y"], 1)
            case["prev_dt_sec"] = round(c["t"] - prev["start_t"], 2)
        else:
            case["prev_y_delta"] = None
            case["prev_dt_sec"] = None
        if video_start is not None:
            case["start_video_sec"] = round(c["t"] - video_start, 3)
        cases.append(case)
        if prev is not None:
            prev["end_ts"] = c["ts"]
            prev["end_t"] = c["t"]
            prev["duration_sec"] = round(c["t"] - prev["start_t"], 3)
            if video_start is not None:
                prev["end_video_sec"] = round(c["t"] - video_start, 3)
        prev = case

    if prev is not None:
        # 最后一个 case 的结束时间留空 (或 = 视频结束, 未知)
        prev["end_ts"] = None
        prev["end_t"] = None
        prev["duration_sec"] = None
        if video_start is not None:
            prev["end_video_sec"] = None

    return cases


def main():
    ap = argparse.ArgumentParser(description="双击检测切分 case")
    ap.add_argument("participant_dir", help="参与者目录 (含 键鼠/ 和 眼动/)")
    ap.add_argument("--box", nargs=4, type=float, default=None,
                    help="可选: 先用方框 X1 Y1 X2 Y2 预过滤双击 (默认不预过滤, 用 y 锯齿链)")
    ap.add_argument("--step-lo", type=float, default=10.0,
                    help="y 锯齿链: 相邻 case 的 y 递增下限 px (排除同行重开)")
    ap.add_argument("--step-hi", type=float, default=80.0,
                    help="y 锯齿链: 相邻 case 的 y 递增上限 px")
    ap.add_argument("--reset-max", type=float, default=-60.0,
                    help="y 锯齿链: 翻页回跳的 y 阈值 (<= 此值视为回跳)")
    ap.add_argument("--max-gap-sec", type=float, default=300.0,
                    help="y 锯齿链: 长时间无操作视为新起点 (s)")
    ap.add_argument("--no-xband", action="store_true",
                    help="不做 x 带自动筛选 (直接对全部双击做锯齿链)")
    ap.add_argument("--x-gap", type=float, default=80.0,
                    help="x 带聚簇的间隔阈值 px")
    ap.add_argument("--dt-max", type=float, default=DT_MAX_MS, help="双击间隔上限 ms")
    ap.add_argument("--dist-max", type=float, default=DIST_MAX_PX, help="双击位移上限 px")
    ap.add_argument("--out-dir", default=None, help="输出目录 (默认=参与者目录/case_segments)")
    ap.add_argument("--show", action="store_true", help="打印全部双击明细")
    ap.add_argument("--merge-sec", type=float, default=3.0, help="补双击合并时间窗 s")
    ap.add_argument("--merge-dy", type=float, default=25.0, help="补双击合并 y 位移 px")
    args = ap.parse_args()

    pdir = os.path.abspath(args.participant_dir)
    if not os.path.isdir(pdir):
        print(f"错误: 目录不存在 {pdir}")
        sys.exit(1)

    jl = find_keymouse_jsonl(pdir)
    if not jl:
        print(f"错误: 未找到键鼠 jsonl (在 {pdir}\\键鼠 下)")
        sys.exit(1)
    print(f"键鼠文件: {jl}")

    events = load_events(jl)
    print(f"事件总数: {len(events)}")

    dbl = detect_double_clicks(events, dt_max_ms=args.dt_max, dist_max_px=args.dist_max)
    print(f"检测到双击/多击: {len(dbl)} 次")

    dbl = merge_reclicks(dbl, merge_sec=args.merge_sec, merge_dy=args.merge_dy)
    print(f"补双击合并后: {len(dbl)} 次")

    # 可选: 方框预过滤
    box = None
    if args.box:
        box = tuple(args.box)
        n0 = len(dbl)
        dbl = [c for c in dbl if in_box(c["x"], c["y"], box)]
        print(f"方框预过滤 {box}: {n0} -> {len(dbl)} 次")

    step_range = (args.step_lo, args.step_hi)

    # x 带自动筛选 (病例列表可能在左也可能在右), 找锯齿链最长的 x 带
    if args.no_xband:
        chain = sawtooth_chain(dbl, step_range=step_range,
                               reset_max=args.reset_max, max_gap_sec=args.max_gap_sec)
        print(f"y 锯齿链提取 case (无 x 带筛选): {len(chain)} 个")
    else:
        band, chain, bands = best_sawtooth_band(
            dbl, gap=args.x_gap, step_range=step_range,
            reset_max=args.reset_max, max_gap_sec=args.max_gap_sec)
        if band:
            xr = (min(c["x"] for c in band), max(c["x"] for c in band))
            print(f"x 带聚簇 ({len(bands)} 带), 选中病例列表带 x=[{xr[0]:.0f},{xr[1]:.0f}]")
            for b in sorted(bands, key=lambda b: -len(b)):
                bx = (min(c["x"] for c in b), max(c["x"] for c in b))
                print(f"    n={len(b):3d}  x=[{bx[0]:5.0f},{bx[1]:5.0f}]")
        print(f"y 锯齿链提取 case: {len(chain)} 个")

    # 透明化: 被跳过 (非 case) 的双击
    chain_ids = {id(c) for c in chain}
    skipped = [c for c in dbl if id(c) not in chain_ids]
    print(f"跳过的非 case 双击: {len(skipped)} 个")

    # 组结构 (每组应 = 20, 3 组 = 60)
    groups = chain_groups(chain, reset_max=args.reset_max)
    gcounts = [len(g) for g in groups]
    print(f"分组结构 (目标 [20,20,20]): {gcounts}  (共 {sum(gcounts)})")

    video_start = find_eye_video_start(pdir)
    if video_start is not None:
        print(f"眼动视频起始: {datetime.fromtimestamp(video_start).strftime('%Y-%m-%d %H:%M:%S')}")

    cases = build_cases(chain, video_start=video_start)
    print(f"case 边界: {len(cases)} 个")

    if args.show:
        print("\n=== 全部双击 (链内/链外) ===")
        for i, c in enumerate(sorted(dbl, key=lambda x: x["t"])):
            flag = "链内" if id(c) in chain_ids else "链外"
            print(f"  [{i:3d}] {c['ts']}  x={c['x']:7.1f} y={c['y']:7.1f}  "
                  f"n={c['n_clicks']} dt={c['dt_ms']:5.0f}ms  {flag}")

    # 输出
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(pdir.rstrip("\\/"))

    # CSV
    rows = []
    for c in cases:
        rows.append({
            "order": c["order"],
            "start_ts": c["start_ts"],
            "start_t": c["start_t"],
            "start_video_sec": c.get("start_video_sec"),
            "end_ts": c.get("end_ts"),
            "end_t": c.get("end_t"),
            "end_video_sec": c.get("end_video_sec"),
            "duration_sec": c.get("duration_sec"),
            "open_x": c["open_x"],
            "open_y": c["open_y"],
            "prev_y_delta": c["prev_y_delta"],
            "prev_dt_sec": c["prev_dt_sec"],
            "n_clicks": c["n_clicks"],
        })
    csv_path = os.path.join(out_dir, f"{name}_cases.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # JSON
    json_path = os.path.join(out_dir, f"{name}_cases.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "participant": name,
            "keymouse_file": jl,
            "box": list(box) if box else None,
            "video_start": video_start,
            "dt_max_ms": args.dt_max,
            "dist_max_px": args.dist_max,
            "step_lo": args.step_lo,
            "step_hi": args.step_hi,
            "reset_max": args.reset_max,
            "max_gap_sec": args.max_gap_sec,
            "no_xband": args.no_xband,
            "x_gap": args.x_gap,
            "n_double_clicks": len(dbl),
            "n_cases": len(cases),
            "cases": cases,
            "skipped_double_clicks": [
                {"ts": c["ts"], "x": c["x"], "y": c["y"]} for c in skipped
            ],
        }, f, ensure_ascii=False, indent=2)

    print(f"\n已保存:")
    print(f"  {csv_path}")
    print(f"  {json_path}")

    # 简洁预览
    print("\n=== case 边界预览 ===")
    for c in cases:
        v = c.get("start_video_sec")
        vstr = f"  video={v:8.2f}s" if v is not None else ""
        dy = c["prev_y_delta"]
        dystr = f"  Δy={dy:+.1f}" if dy is not None else ""
        print(f"  #{c['order']:3d}  {c['start_ts']}{vstr}  "
              f"x={c['open_x']:6.1f} y={c['open_y']:6.1f}{dystr}")


if __name__ == "__main__":
    main()
