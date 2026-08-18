import os
import re
import sys
import csv
from collections import defaultdict

BASE = r"C:\Users\wangzhi\Desktop"
DOCTOR_BASE = os.path.join(BASE, "AI读片原数据")
GT_BASE = os.path.join(BASE, "实验原数据_实验_0724_1620-jw AIvsNOAI_bone")

PT_RE = re.compile(r"\{(-?\d+(?:\.\d+)?(?:[eE]-?\d+)?), (-?\d+(?:\.\d+)?(?:[eE]-?\d+)?)\}")
XM_RE = re.compile(r"XM(\d+)")
DOCTOR_FILE_RE = re.compile(r"^a\d+_XM(\d+)")


def parse_roi_file(path):
    """解析 .rois_series 文件, 返回 ROI 列表: [(中心x, 中心y, 大小w, 大小h, 切片x, 切片y)]"""
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin1")
    pts = [(float(a), float(b)) for a, b in PT_RE.findall(text)]
    rois = []
    for i in range(0, len(pts) - 2, 3):
        cx, cy = pts[i]
        w, h = pts[i + 1]
        sx, sy = pts[i + 2]
        rois.append((round(cx, 2), round(cy, 2), round(w, 2), round(h, 2),
                     round(sx, 2), round(sy, 2)))
    return rois


def find_roi_files():
    """找到所有 ROI 文件, 返回 {xm编号: {source: path}}"""
    result = defaultdict(dict)
    for root, dirs, files in os.walk(GT_BASE):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for f in files:
            if f.startswith("._"):
                continue
            if "rois_series" not in f or f.endswith(".json"):
                continue
            m = XM_RE.search(f)
            if not m:
                continue
            xm = m.group(0)
            low = f.lower()
            if low.endswith("ai.rois_series"):
                result[xm]["AI"] = os.path.join(root, f)
            elif low.endswith("gy.rois_series"):
                result[xm]["GY"] = os.path.join(root, f)
    for p in os.listdir(DOCTOR_BASE):
        sub = os.path.join(DOCTOR_BASE, p)
        if not os.path.isdir(sub):
            continue
        roi = os.path.join(sub, "roi")
        if not os.path.isdir(roi):
            continue
        for root, dirs, files in os.walk(roi):
            for f in files:
                if f.startswith("._") or not f.endswith(".rois_series"):
                    continue
                m = DOCTOR_FILE_RE.match(f)
                if not m:
                    m = XM_RE.search(f)
                    if not m:
                        continue
                    xm = m.group(0)
                else:
                    xm = "XM" + m.group(1)
                result[xm]["医生"] = os.path.join(root, f)
    return result


def dist(roi1, roi2):
    """ROI 空间距离 (中心点欧氏距离 + 切片距离)"""
    cx1, cy1, w1, h1, sx1, sy1 = roi1
    cx2, cy2, w2, h2, sx2, sy2 = roi2
    dc = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
    ds = ((sx1 - sx2) ** 2 + (sy1 - sy2) ** 2) ** 0.5
    return dc, ds


def match_roi(roi, others, dc_thresh=5.0, ds_thresh=2.0):
    """判断 roi 是否与 others 中某个匹配"""
    for o in others:
        dc, ds = dist(roi, o)
        if dc <= dc_thresh and ds <= ds_thresh:
            return True, o
    return False, None


def main():
    files = find_roi_files()
    print(f"找到 {len(files)} 个 case (XM 编号)")
    rows = []
    stats = {"医生补漏(AI无)": 0, "医生删除(AI有)": 0, "医生漏诊(金标准有)": 0}

    for xm in sorted(files.keys()):
        src = files[xm]
        doctor = parse_roi_file(src["医生"]) if "医生" in src else []
        ai = parse_roi_file(src["AI"]) if "AI" in src else []
        gy = parse_roi_file(src["GY"]) if "GY" in src else []

        if not doctor and not ai and not gy:
            continue

        # 1. 医生标记 AI 未标记的: 医生 ROI 有 & AI ROI 无
        doc_only = [r for r in doctor if not match_roi(r, ai)[0]] if ai else doctor
        # 2. 医生删除 AI 标记的: AI ROI 有 & 医生 ROI 无
        ai_only = [r for r in ai if not match_roi(r, doctor)[0]] if doctor else ai
        # 3. 医生未标记但金标准标记的: 金标准 ROI 有 & 医生 ROI 无
        gy_only = [r for r in gy if not match_roi(r, doctor)[0]] if doctor else gy

        stats["医生补漏(AI无)"] += len(doc_only)
        stats["医生删除(AI有)"] += len(ai_only)
        stats["医生漏诊(金标准有)"] += len(gy_only)

        for r in doc_only:
            rows.append([xm, "1-医生标记AI未标记", len(doctor), len(ai), len(gy),
                         f"({r[0]},{r[1]})", f"{r[2]}x{r[3]}", "", ""])
        for r in ai_only:
            rows.append([xm, "2-医生删除AI标记", len(doctor), len(ai), len(gy),
                         "", "", f"({r[0]},{r[1]})", f"{r[2]}x{r[3]}"])
        for r in gy_only:
            rows.append([xm, "3-医生未标记金标准标记", len(doctor), len(ai), len(gy),
                         "", "", "", f"({r[0]},{r[1]}) {r[2]}x{r[3]}"])

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "verification", "three_events.csv")
    header = ["XM编号", "事件类别", "医生ROI数", "AI_ROI数", "金标准ROI数",
              "医生中心点", "医生大小", "AI中心点", "金标准中心点"]
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"\n三类事件统计:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\n清单已保存: {out}  (共 {len(rows)} 行)")


if __name__ == "__main__":
    main()
