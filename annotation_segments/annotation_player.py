import math
import os
import re
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import cv2

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QComboBox,
    QTableWidget, QTableWidgetItem, QHBoxLayout, QVBoxLayout, QSplitter,
    QFileDialog, QLineEdit, QSpinBox, QDoubleSpinBox, QMessageBox,
    QAbstractItemView, QHeaderView,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "annotation_events.csv")
RESULTS_PATH = os.path.join(BASE_DIR, "verification", "verification_results.csv")
OFFSET_CACHE = os.path.join(BASE_DIR, "verification", "offset_cache.json")
MANUAL_KEYFRAMES_PATH = os.path.join(BASE_DIR, "verification", "manual_keyframes.csv")

MANUAL_KF_COLUMNS = [
    "participant", "event_type", "event_ts", "event_ts_float",
    "window_start", "window_end", "duration_sec", "distance_px",
    "start_x", "start_y", "end_x", "end_y", "delete_key", "event_id", "notes",
]

ACTION_TYPE_MAP = {"标注动作": "annotation", "删除动作": "deletion"}
ACTION_LABEL_MAP = {"annotation": "标注动作", "deletion": "删除动作"}

ANNOTATION_COLOR = QColor(30, 144, 255)
DELETION_COLOR = QColor(255, 40, 40)
FOCUS_COLOR = QColor(255, 200, 0)
KEYFRAME_COLOR = QColor(0, 200, 100)
PENDING_COLOR = QColor(0, 255, 255)

DEFAULT_SCREEN_W = 1920
DEFAULT_SCREEN_H = 1080

SCREEN_REC_DIR_NAMES = ("眼动",)
TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ _](\d{2})-(\d{2})-(\d{2})")


def detect_video_base():
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    cand = os.path.join(desktop, "AI读片原数据")
    if os.path.isdir(cand):
        return cand
    return ""


def parse_filename_ts(name):
    m = TS_RE.match(name)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group(1)} {m.group(2)}:{m.group(3)}:{m.group(4)}",
                               "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    except ValueError:
        return None


def find_screen_recording(base, participant):
    sub = os.path.join(base, participant)
    if not os.path.isdir(sub):
        return None
    eye_dirs = []
    for root, dirs, files in os.walk(sub):
        for d in dirs:
            if d in SCREEN_REC_DIR_NAMES:
                eye_dirs.append(os.path.join(root, d))
    candidates = []
    for ed in eye_dirs:
        for root, dirs, files in os.walk(ed):
            for f in files:
                if f.lower() in ("raw.mp4", "overlay.mp4"):
                    candidates.append(os.path.join(root, f))
    if not candidates:
        return None

    def key(p):
        parent = os.path.basename(os.path.dirname(p))
        has_ts = 1 if TS_RE.match(parent) else 0
        is_raw = 1 if os.path.basename(p).lower() == "raw.mp4" else 0
        return (has_ts, is_raw)

    return max(candidates, key=key)


def to_float(v, default=0.0):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def format_ts(ts_float):
    try:
        return datetime.fromtimestamp(ts_float).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    except (OverflowError, OSError, ValueError):
        return ""


class ClickableLabel(QLabel):
    clicked = Signal(float, float)

    def mousePressEvent(self, event):
        pos = event.position()
        self.clicked.emit(pos.x(), pos.y())
        super().mousePressEvent(event)


class TimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(70)
        self.duration = 1.0
        self.current_time = 0.0
        self.events = []
        self.on_click = None

    def set_data(self, duration, current_time, events):
        self.duration = max(duration, 1e-6)
        self.current_time = current_time
        self.events = events
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        pad = 10
        track_h = 26
        track_y = h - track_h - 12
        track_w = w - 2 * pad

        p.fillRect(self.rect(), QColor(30, 30, 34))

        p.setPen(QPen(QColor(70, 70, 76)))
        p.setBrush(QColor(52, 52, 58))
        p.drawRect(pad, track_y, track_w, track_h)

        for t, etype, eid, is_manual in self.events:
            x = pad + int((t / self.duration) * track_w)
            if is_manual:
                color = KEYFRAME_COLOR
            elif etype == "deletion":
                color = DELETION_COLOR
            else:
                color = ANNOTATION_COLOR
            p.setPen(QPen(color, 2))
            p.drawLine(x, track_y + 3, x, track_y + track_h - 3)

        cursor_x = pad + int((self.current_time / self.duration) * track_w)
        p.setPen(QPen(QColor(255, 255, 255), 2))
        p.drawLine(cursor_x, track_y - 4, cursor_x, track_y + track_h + 4)

        p.setPen(QPen(QColor(200, 200, 200)))
        p.drawText(pad, track_y - 10, "0s")
        p.drawText(w - pad - 40, track_y - 10, f"{self.duration:.1f}s")

    def mousePressEvent(self, event):
        w = self.width()
        pad = 10
        track_w = w - 2 * pad
        x = event.position().x()
        t = ((x - pad) / track_w) * self.duration
        t = max(0.0, min(self.duration, t))
        if self.on_click:
            self.on_click(t)


class AnnotationPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Annotation Segment Player")
        self.resize(1400, 900)

        self.df = pd.DataFrame()
        self.events = []
        self.manual_events = []
        self.participant = None
        self.video_path = None
        self.cap = None
        self.fps = 30.0
        self.total_frames = 0
        self.duration = 1.0
        self.offset = 0.0
        self.current_frame = 0
        self.current_time = 0.0
        self.playing = False
        self.selected_event_id = None
        self.offsets = {}
        self.results = {}
        self.play_speed = 1.0
        self._frame_accum = 0.0
        self.last_frame_size = (0, 0)
        self._kf_start = None
        self._kf_end = None

        self.load_offsets()
        self.load_results()
        self.load_manual_keyframes()
        self.build_ui()
        self.load_csv()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_timer)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        control = QHBoxLayout()
        self.btn_load_video = QPushButton("加载视频")
        self.btn_load_video.clicked.connect(self.load_video)
        self.btn_auto_video = QPushButton("自动定位视频")
        self.btn_auto_video.clicked.connect(self.auto_locate_video)
        self.btn_load_csv = QPushButton("加载 CSV")
        self.btn_load_csv.clicked.connect(self.load_csv)
        self.cmb_participant = QComboBox()
        self.cmb_participant.currentIndexChanged.connect(self.on_participant_changed)
        self.btn_play = QPushButton("播放")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_prev_event = QPushButton("< 上一事件")
        self.btn_prev_event.clicked.connect(lambda: self.step_event(-1))
        self.btn_next_event = QPushButton("下一事件 >")
        self.btn_next_event.clicked.connect(lambda: self.step_event(1))

        self.cmb_speed = QComboBox()
        self.cmb_speed.addItems(["0.5x", "1x", "1.5x", "2x", "3x"])
        self.cmb_speed.setCurrentIndex(1)
        self.cmb_speed.currentIndexChanged.connect(self.on_speed_changed)

        control.addWidget(self.btn_load_video)
        control.addWidget(self.btn_auto_video)
        control.addWidget(self.btn_load_csv)
        control.addWidget(QLabel("参与者:"))
        control.addWidget(self.cmb_participant)
        control.addWidget(self.btn_play)
        control.addWidget(self.btn_prev_event)
        control.addWidget(self.btn_next_event)
        control.addStretch(1)

        control.addWidget(QLabel("倍速:"))
        control.addWidget(self.cmb_speed)

        control.addWidget(QLabel("显示半径(s):"))
        self.spin_radius = QDoubleSpinBox()
        self.spin_radius.setRange(0.0, 30.0)
        self.spin_radius.setValue(5.0)
        self.spin_radius.setSingleStep(0.5)
        self.spin_radius.valueChanged.connect(self.refresh_frame)
        control.addWidget(self.spin_radius)

        control.addWidget(QLabel("录屏分辨率:"))
        self.spin_scr_w = QSpinBox()
        self.spin_scr_w.setRange(0, 99999)
        self.spin_scr_w.setValue(DEFAULT_SCREEN_W)
        self.spin_scr_w.valueChanged.connect(self.refresh_frame)
        self.spin_scr_h = QSpinBox()
        self.spin_scr_h.setRange(0, 99999)
        self.spin_scr_h.setValue(DEFAULT_SCREEN_H)
        self.spin_scr_h.valueChanged.connect(self.refresh_frame)
        control.addWidget(self.spin_scr_w)
        control.addWidget(QLabel("x"))
        control.addWidget(self.spin_scr_h)

        root.addLayout(control)

        splitter = QSplitter(Qt.Horizontal)

        self.video_label = ClickableLabel("尚未加载视频。\n点击“加载视频”选择视频文件。")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("background-color:#101012; color:#999;")
        self.video_label.setScaledContents(False)
        self.video_label.clicked.connect(self.on_video_clicked)
        splitter.addWidget(self.video_label)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels(
            ["event_id", "类型", "时间", "时长s", "距离px", "判断", "备注"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.cellClicked.connect(self.on_cell_clicked)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        splitter.addWidget(self.table)
        splitter.setSizes([1000, 400])

        root.addWidget(splitter, 1)

        self.timeline = TimelineWidget()
        self.timeline.on_click = self.on_timeline_click
        root.addWidget(self.timeline)

        bottom = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("视频根目录:"))
        self.edit_video_base = QLineEdit()
        self.edit_video_base.setText(detect_video_base())
        row1.addWidget(self.edit_video_base, 1)

        self.btn_calibrate = QPushButton("校准(当前帧=选中事件)")
        self.btn_calibrate.clicked.connect(self.calibrate)
        self.btn_mark_correct = QPushButton("✓ 正确")
        self.btn_mark_correct.clicked.connect(self.mark_correct)
        self.btn_mark_wrong = QPushButton("✗ 错误")
        self.btn_mark_wrong.clicked.connect(self.mark_wrong)
        self.btn_clear_mark = QPushButton("清除判断")
        self.btn_clear_mark.clicked.connect(self.clear_mark)
        self.edit_note = QLineEdit()
        self.edit_note.setPlaceholderText("备注(回车保存)")
        self.edit_note.returnPressed.connect(self.save_note)
        row1.addWidget(self.btn_calibrate)
        row1.addWidget(self.btn_mark_correct)
        row1.addWidget(self.btn_mark_wrong)
        row1.addWidget(self.btn_clear_mark)
        row1.addWidget(self.edit_note)
        bottom.addLayout(row1)

        row2 = QHBoxLayout()
        self.cmb_action = QComboBox()
        self.cmb_action.addItems(list(ACTION_TYPE_MAP.keys()))
        self.cmb_action.currentIndexChanged.connect(self._on_action_changed)
        self.edit_keynote = QLineEdit()
        self.edit_keynote.setPlaceholderText("关键帧备注")
        self.edit_keynote.returnPressed.connect(self.add_keyframe)
        self.btn_add_keyframe = QPushButton("添加关键帧(K)")
        self.btn_add_keyframe.clicked.connect(self.add_keyframe)
        self.btn_delete_keyframe = QPushButton("删除关键帧(Del)")
        self.btn_delete_keyframe.clicked.connect(self.delete_keyframe)
        self.btn_export = QPushButton("导出最终表格")
        self.btn_export.clicked.connect(self.export_final_table)
        row2.addWidget(QLabel("医生动作:"))
        row2.addWidget(self.cmb_action)
        row2.addWidget(QLabel("备注:"))
        row2.addWidget(self.edit_keynote)
        row2.addWidget(self.btn_add_keyframe)
        row2.addWidget(self.btn_delete_keyframe)
        row2.addWidget(self.btn_export)
        row2.addStretch(1)
        bottom.addLayout(row2)

        self.lbl_status = QLabel("")
        bottom.addWidget(self.lbl_status)

        root.addLayout(bottom)

        self.set_controls_enabled(False)

    def set_controls_enabled(self, enabled):
        for w in [self.btn_play, self.btn_prev_event, self.btn_next_event,
                  self.btn_calibrate, self.btn_mark_correct, self.btn_mark_wrong,
                  self.btn_clear_mark, self.btn_add_keyframe, self.btn_delete_keyframe]:
            w.setEnabled(enabled)

    def load_offsets(self):
        if os.path.exists(OFFSET_CACHE):
            try:
                with open(OFFSET_CACHE, "r", encoding="utf-8") as f:
                    self.offsets = json.load(f)
            except Exception:
                self.offsets = {}

    def save_offsets(self):
        try:
            with open(OFFSET_CACHE, "w", encoding="utf-8") as f:
                json.dump(self.offsets, f, ensure_ascii=False, indent=2)
        except Exception as e:
            QMessageBox.warning(self, "提示", f"保存偏移失败: {e}")

    def load_results(self):
        if os.path.exists(RESULTS_PATH):
            try:
                rdf = pd.read_csv(RESULTS_PATH)
                for _, row in rdf.iterrows():
                    self.results[int(row["event_id"])] = {
                        "verified": str(row.get("verified", "")),
                        "notes": str(row.get("notes", "")) if not pd.isna(row.get("notes", "")) else "",
                    }
            except Exception:
                self.results = {}

    def save_results(self):
        rows = []
        for e in self.events:
            if e.get("is_manual"):
                continue
            r = self.results.get(int(e["event_id"]), {"verified": "", "notes": ""})
            rows.append({
                "event_id": int(e["event_id"]),
                "participant": e["participant"],
                "event_type": e["event_type"],
                "event_ts": e["event_ts"],
                "verified": r.get("verified", ""),
                "notes": r.get("notes", ""),
            })
        out = pd.DataFrame(rows, columns=["event_id", "participant", "event_type",
                                          "event_ts", "verified", "notes"])
        out.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
        self.lbl_status.setText(f"已保存 {len(rows)} 条判断结果 -> verification_results.csv")

    def load_manual_keyframes(self):
        self.manual_events = []
        if not os.path.exists(MANUAL_KEYFRAMES_PATH):
            return
        try:
            mdf = pd.read_csv(MANUAL_KEYFRAMES_PATH, encoding="utf-8-sig")
        except Exception:
            return
        for _, row in mdf.iterrows():
            m = {
                "participant": str(row["participant"]),
                "event_type": str(row["event_type"]),
                "event_ts": str(row["event_ts"]),
                "event_ts_float": to_float(row["event_ts_float"]),
                "window_start": str(row["window_start"]),
                "window_end": str(row["window_end"]),
                "duration_sec": to_float(row["duration_sec"]),
                "distance_px": to_float(row["distance_px"]),
                "start_x": to_float(row["start_x"]),
                "start_y": to_float(row["start_y"]),
                "end_x": to_float(row["end_x"]),
                "end_y": to_float(row["end_y"]),
                "delete_key": str(row["delete_key"]) if "delete_key" in mdf.columns else "",
                "event_id": int(to_float(row["event_id"])),
                "notes": str(row["notes"]) if "notes" in mdf.columns and not pd.isna(row["notes"]) else "",
                "is_manual": True,
            }
            self.manual_events.append(m)

    def save_manual_keyframes(self):
        rows = [{c: m.get(c, "") for c in MANUAL_KF_COLUMNS} for m in self.manual_events]
        out = pd.DataFrame(rows, columns=MANUAL_KF_COLUMNS)
        out.to_csv(MANUAL_KEYFRAMES_PATH, index=False, encoding="utf-8-sig")
        self.lbl_status.setText(f"已保存 {len(rows)} 条人工关键帧 -> manual_keyframes.csv")

    def export_final_table(self):
        if self.df.empty and not self.manual_events:
            QMessageBox.information(self, "提示", "暂无数据可导出")
            return
        rows = []
        for _, row in self.df.iterrows():
            eid = int(to_float(row["event_id"]))
            r = self.results.get(eid, {"verified": "", "notes": ""})
            verified = str(r.get("verified", ""))
            if verified == "ok":
                status = "ai标记人标记"
            elif verified == "bad":
                status = "ai标记人删除"
            else:
                status = "未判断"
            rows.append({
                "participant": str(row["participant"]),
                "event_id": eid,
                "status": status,
                "event_type": str(row["event_type"]),
                "动作": ACTION_LABEL_MAP.get(str(row["event_type"]), str(row["event_type"])),
                "event_ts": str(row["event_ts"]),
                "event_ts_float": to_float(row["event_ts_float"]),
                "start_x": to_float(row["start_x"]),
                "start_y": to_float(row["start_y"]),
                "end_x": to_float(row["end_x"]),
                "end_y": to_float(row["end_y"]),
                "distance_px": to_float(row["distance_px"]),
                "duration_sec": to_float(row["duration_sec"]),
                "notes": str(r.get("notes", "")),
            })
        for m in self.manual_events:
            rows.append({
                "participant": str(m["participant"]),
                "event_id": int(m["event_id"]),
                "status": "ai未标记人标记",
                "event_type": str(m["event_type"]),
                "动作": ACTION_LABEL_MAP.get(str(m["event_type"]), str(m["event_type"])),
                "event_ts": str(m["event_ts"]),
                "event_ts_float": to_float(m["event_ts_float"]),
                "start_x": to_float(m["start_x"]),
                "start_y": to_float(m["start_y"]),
                "end_x": to_float(m["end_x"]),
                "end_y": to_float(m["end_y"]),
                "distance_px": to_float(m["distance_px"]),
                "duration_sec": to_float(m["duration_sec"]),
                "notes": str(m.get("notes", "")),
            })
        rows.sort(key=lambda x: (x["participant"], x["event_ts_float"]))
        cols = ["participant", "event_id", "status", "event_type", "动作", "event_ts",
                "event_ts_float", "start_x", "start_y", "end_x", "end_y",
                "distance_px", "duration_sec", "notes"]
        out = pd.DataFrame(rows, columns=cols)
        default_name = f"final_annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出最终标记表格", os.path.join(BASE_DIR, default_name),
            "CSV (*.csv)")
        if not path:
            return
        out.to_csv(path, index=False, encoding="utf-8-sig")
        self.lbl_status.setText(f"已导出 {len(rows)} 条 -> {path}")
        QMessageBox.information(self, "导出成功", f"已导出 {len(rows)} 条记录到:\n{path}")

    def _build_events(self):
        events = []
        sub = self.df[self.df["participant"] == self.participant]
        for _, row in sub.iterrows():
            eid = int(to_float(row["event_id"]))
            r = self.results.get(eid, {"verified": "", "notes": ""})
            events.append({
                "participant": row["participant"],
                "event_type": row["event_type"],
                "event_ts": row["event_ts"],
                "event_ts_float": to_float(row["event_ts_float"]),
                "duration_sec": to_float(row["duration_sec"]),
                "distance_px": to_float(row["distance_px"]),
                "start_x": to_float(row["start_x"]),
                "start_y": to_float(row["start_y"]),
                "end_x": to_float(row["end_x"]),
                "end_y": to_float(row["end_y"]),
                "event_id": eid,
                "notes": r.get("notes", ""),
                "is_manual": False,
            })
        for m in self.manual_events:
            if m["participant"] == self.participant:
                events.append(dict(m))
        events.sort(key=lambda x: x["event_ts_float"])
        self.events = events

    def on_speed_changed(self):
        txt = self.cmb_speed.currentText().rstrip("x")
        try:
            self.play_speed = float(txt)
        except ValueError:
            self.play_speed = 1.0

    def load_csv(self):
        if not os.path.exists(CSV_PATH):
            QMessageBox.critical(self, "错误", f"找不到 CSV: {CSV_PATH}")
            return
        self.df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
        parts = sorted(self.df["participant"].dropna().unique().tolist())
        self.cmb_participant.blockSignals(True)
        self.cmb_participant.clear()
        self.cmb_participant.addItems(parts)
        self.cmb_participant.blockSignals(False)
        if parts:
            self.on_participant_changed()

    def on_participant_changed(self):
        self.participant = self.cmb_participant.currentText()
        self._kf_start = None
        self._kf_end = None
        self._build_events()

        if self.participant in self.offsets:
            self.offset = self.offsets[self.participant]
        else:
            self.offset = 0.0

        self.rebuild_table()
        self.update_timeline()
        self.set_controls_enabled(self.cap is not None and len(self.events) > 0)
        self.auto_locate_video()

    def rebuild_table(self):
        self.table.setRowCount(len(self.events))
        for i, e in enumerate(self.events):
            r = self.results.get(int(e["event_id"]), {"verified": "", "notes": ""})
            etype = e["event_type"] + ("(手动)" if e.get("is_manual") else "")
            self.table.setItem(i, 0, QTableWidgetItem(str(e["event_id"])))
            self.table.setItem(i, 1, QTableWidgetItem(etype))
            self.table.setItem(i, 2, QTableWidgetItem(str(e["event_ts"])))
            self.table.setItem(i, 3, QTableWidgetItem(f"{e['duration_sec']:.3f}"))
            self.table.setItem(i, 4, QTableWidgetItem(f"{e['distance_px']:.1f}"))
            self.table.setItem(i, 5, QTableWidgetItem(r.get("verified", "")))
            self.table.setItem(i, 6, QTableWidgetItem(e.get("notes", "")))
            for c in range(7):
                it = self.table.item(i, c)
                if it is None:
                    continue
                if e["event_type"] == "deletion":
                    it.setForeground(DELETION_COLOR)
                elif e.get("is_manual"):
                    it.setForeground(KEYFRAME_COLOR)

    def on_cell_clicked(self, row, col):
        self.select_event(self.events[row]["event_id"])

    def select_event(self, event_id):
        self.selected_event_id = event_id
        e = self._find_event(event_id)
        if e is None:
            return
        idx = self._index_of(event_id)
        if idx is not None:
            self.table.selectRow(idx)
        if self.cap is not None:
            self.seek_time(e["event_ts_float"] - self.offset)
        else:
            self.current_time = e["event_ts_float"] - self.offset
            self.update_timeline()
        self._refresh_status()

    def _find_event(self, event_id):
        for e in self.events:
            if e["event_id"] == event_id:
                return e
        return None

    def on_timeline_click(self, t):
        eid = self._nearest_event_at(t)
        if eid is not None:
            self.selected_event_id = eid
            idx = self._index_of(eid)
            if idx is not None:
                self.table.selectRow(idx)
            self._refresh_status()
        if self.cap is not None:
            self.seek_time(t)
        else:
            self.current_time = t
            self.update_timeline()

    def _nearest_event_at(self, t):
        best = None
        best_d = None
        for e in self.events:
            te = e["event_ts_float"] - self.offset
            d = abs(te - t)
            if d <= 0.5 and (best_d is None or d < best_d):
                best_d = d
                best = e["event_id"]
        return best

    def update_timeline(self):
        pts = [(e["event_ts_float"] - self.offset, e["event_type"], e["event_id"],
                bool(e.get("is_manual"))) for e in self.events]
        self.timeline.set_data(self.duration, self.current_time, pts)

    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频 (*.mp4 *.mkv *.avi *.mov *.flv *.ts *.m4v);;所有文件 (*.*)")
        if not path:
            return
        self.open_video(path)

    def auto_locate_video(self):
        if not self.participant:
            return
        base = self.edit_video_base.text().strip()
        if not base:
            return
        path = find_screen_recording(base, self.participant)
        if path:
            self.open_video(path)
        else:
            self.lbl_status.setText(f"未在 {base}\\{self.participant} 找到录屏视频")

    def open_video(self, path):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            alt = None
            if os.path.basename(path).lower() == "raw.mp4":
                cand = os.path.join(os.path.dirname(path), "overlay.mp4")
                if os.path.exists(cand):
                    alt = cand
            if alt:
                self.cap.release()
                self.cap = cv2.VideoCapture(alt)
                if self.cap.isOpened():
                    path = alt
                else:
                    self.cap = None
            if self.cap is None or not self.cap.isOpened():
                QMessageBox.critical(self, "错误", f"无法打开视频: {path}")
                self.cap = None
                return
        self.video_path = path
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps
        self.current_frame = 0
        self.current_time = 0.0

        auto_ts = None
        if self.participant and self.participant not in self.offsets:
            auto_ts = parse_filename_ts(os.path.basename(path))
            if auto_ts is None:
                auto_ts = parse_filename_ts(os.path.basename(os.path.dirname(path)))
            if auto_ts is not None:
                self.offset = auto_ts
                self.offsets[self.participant] = auto_ts
                self.save_offsets()
        elif self.participant in self.offsets:
            self.offset = self.offsets[self.participant]

        self.set_controls_enabled(True)
        self.update_timeline()
        self.refresh_frame()
        msg = f"已加载: {os.path.basename(path)}  "
        msg += f"{self.total_frames}帧  {self.fps:.2f}fps  {self.duration:.1f}s"
        if auto_ts is not None:
            msg += f"  [自动对齐 offset={auto_ts:.3f}]"
        self.lbl_status.setText(msg)

    def toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setText("暂停")
            self._frame_accum = 0.0
            interval = max(1, int(1000 / self.fps))
            self.timer.start(interval)
        else:
            self.btn_play.setText("播放")
            self.timer.stop()

    def on_timer(self):
        if not self.playing or self.cap is None:
            return
        self._frame_accum += self.play_speed
        steps = int(self._frame_accum)
        if steps >= 1:
            self._frame_accum -= steps
            self._advance_frames(steps)

    def _advance_frames(self, n):
        frame = None
        for _ in range(n):
            ok, frame = self.cap.read()
            if not ok:
                break
            self.current_frame += 1
        if frame is not None:
            self.current_time = self.current_frame / self.fps
            self.display_frame(frame)
            self.update_timeline()
        else:
            self.toggle_play()

    def refresh_frame(self):
        if self.cap is None:
            return
        self._seek_frame(self.current_frame)

    def _seek_frame(self, frame_idx):
        frame_idx = max(0, min(self.total_frames - 1, frame_idx))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if ok:
            self.current_frame = frame_idx
            self.current_time = frame_idx / self.fps
            self.display_frame(frame)
            self.update_timeline()

    def seek_time(self, t_sec):
        if self.cap is None:
            return
        t_sec = max(0.0, min(self.duration, t_sec))
        self._seek_frame(int(t_sec * self.fps))

    def seek_event_frame(self, event_id):
        e = self._find_event(event_id)
        if e is None or self.cap is None:
            return
        self.seek_time(e["event_ts_float"] - self.offset)

    def step_event(self, direction):
        if not self.events:
            return
        idx = self._index_of(self.selected_event_id)
        if idx is None:
            idx = 0
        else:
            idx += direction
        idx = max(0, min(len(self.events) - 1, idx))
        self.select_event(self.events[idx]["event_id"])
        self.table.selectRow(idx)

    def _index_of(self, event_id):
        for i, e in enumerate(self.events):
            if e["event_id"] == event_id:
                return i
        return None

    def _scale(self):
        if self.cap is None:
            return 1.0, 1.0
        vw = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        vh = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        sw = self.spin_scr_w.value()
        sh = self.spin_scr_h.value()
        sx = vw / sw if (sw > 0 and sw != vw) else 1.0
        sy = vh / sh if (sh > 0 and sh != vh) else 1.0
        return sx, sy

    def _visible_events(self, t_now):
        radius = self.spin_radius.value()
        out = []
        for e in self.events:
            te = e["event_ts_float"] - self.offset
            if abs(te - t_now) <= radius:
                out.append(e)
        return out

    def display_frame(self, frame):
        sx, sy = self._scale()
        t_now = self.current_frame / self.fps if self.cap is not None else self.current_time
        frame = frame.copy()

        for e in self._visible_events(t_now):
            is_focus = (e["event_id"] == self.selected_event_id)
            manual = bool(e.get("is_manual"))
            if e["event_type"] == "keyframe":
                color = FOCUS_COLOR if is_focus else KEYFRAME_COLOR
                self._draw_keyframe(frame, e, sx, sy, color, is_focus)
            elif e["event_type"] == "deletion":
                color = FOCUS_COLOR if is_focus else (KEYFRAME_COLOR if manual else DELETION_COLOR)
                self._draw_deletion(frame, e, sx, sy, color, is_focus)
            else:
                color = FOCUS_COLOR if is_focus else (KEYFRAME_COLOR if manual else ANNOTATION_COLOR)
                self._draw_annotation(frame, e, sx, sy, color, is_focus)

        self._draw_pending_keyframe(frame, sx, sy)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.last_frame_size = (w, h)
        img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(img)
        self.video_label.setPixmap(
            pix.scaled(self.video_label.size(), Qt.KeepAspectRatio,
                       Qt.SmoothTransformation))

    def _draw_annotation(self, frame, e, sx, sy, color, is_focus):
        x1, y1 = int(e["start_x"] * sx), int(e["start_y"] * sy)
        x2, y2 = int(e["end_x"] * sx), int(e["end_y"] * sy)
        r = 14 if is_focus else 11
        th = 3 if is_focus else 2
        bgr = (color.blue(), color.green(), color.red())
        if e["distance_px"] > 1:
            cv2.line(frame, (x1, y1), (x2, y2), bgr, th)
            cv2.circle(frame, (x2, y2), r - 4, bgr, -1)
        cv2.circle(frame, (x1, y1), r, bgr, th)
        cv2.circle(frame, (x1, y1), 2, bgr, -1)
        cv2.putText(frame, str(e["event_id"]), (x1 + r + 4, y1 - r - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    def _draw_deletion(self, frame, e, sx, sy, color, is_focus):
        x, y = int(e["start_x"] * sx), int(e["start_y"] * sy)
        s = 12 if is_focus else 9
        th = 3 if is_focus else 2
        bgr = (color.blue(), color.green(), color.red())
        cv2.line(frame, (x - s, y - s), (x + s, y + s), bgr, th)
        cv2.line(frame, (x - s, y + s), (x + s, y - s), bgr, th)
        cv2.circle(frame, (x, y), s + 2, bgr, th)
        cv2.putText(frame, f"DEL {e['event_id']}", (x + s + 4, y - s - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    def _draw_keyframe(self, frame, e, sx, sy, color, is_focus):
        x, y = int(e["start_x"] * sx), int(e["start_y"] * sy)
        s = 13 if is_focus else 10
        th = 3 if is_focus else 2
        bgr = (color.blue(), color.green(), color.red())
        cv2.line(frame, (x - s, y), (x + s, y), bgr, th)
        cv2.line(frame, (x, y - s), (x, y + s), bgr, th)
        cv2.circle(frame, (x, y), s, bgr, th)
        cv2.putText(frame, f"KF {e['event_id']}", (x + s + 4, y - s - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    def _draw_pending_keyframe(self, frame, sx, sy):
        if self._kf_start is None:
            return
        bgr = (PENDING_COLOR.blue(), PENDING_COLOR.green(), PENDING_COLOR.red())
        x0, y0 = int(self._kf_start[0] * sx), int(self._kf_start[1] * sy)
        ktype = ACTION_TYPE_MAP.get(self.cmb_action.currentText(), "annotation")
        if ktype == "deletion":
            s = 12
            cv2.line(frame, (x0 - s, y0 - s), (x0 + s, y0 + s), bgr, 2)
            cv2.line(frame, (x0 - s, y0 + s), (x0 + s, y0 - s), bgr, 2)
            cv2.circle(frame, (x0, y0), s + 2, bgr, 2)
            label = "删除点"
        else:
            cv2.circle(frame, (x0, y0), 12, bgr, 2)
            cv2.circle(frame, (x0, y0), 3, bgr, -1)
            label = "标注点"
        cv2.putText(frame, label, (x0 + 14, y0 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1, cv2.LINE_AA)

    def on_video_clicked(self, lx, ly):
        if self.cap is None:
            return
        pos = self._label_to_video(lx, ly)
        if pos is None:
            return
        vx, vy = pos
        hit = self._hit_test_manual_keyframe(vx, vy)
        if hit is not None:
            self.select_event(hit)
            return
        sx, sy = self._scale()
        px, py = vx / sx, vy / sy
        self._kf_start = (px, py)
        self._kf_end = (px, py)
        self._refresh_kf_marker_status()
        self.refresh_frame()

    def _hit_test_manual_keyframe(self, vx, vy):
        sx, sy = self._scale()
        radius = 20
        best = None
        best_d = None
        for e in self.events:
            if not e.get("is_manual"):
                continue
            ex = e["start_x"] * sx
            ey = e["start_y"] * sy
            d = math.hypot(vx - ex, vy - ey)
            if d <= radius and (best_d is None or d < best_d):
                best_d = d
                best = e["event_id"]
        return best

    def _label_to_video(self, lx, ly):
        fw, fh = self.last_frame_size
        if fw <= 0 or fh <= 0:
            return None
        lw = self.video_label.width()
        lh = self.video_label.height()
        if lw <= 0 or lh <= 0:
            return None
        scale = min(lw / fw, lh / fh)
        rw, rh = fw * scale, fh * scale
        ox = (lw - rw) / 2.0
        oy = (lh - rh) / 2.0
        vx = (lx - ox) / scale
        vy = (ly - oy) / scale
        if vx < 0 or vy < 0 or vx > fw or vy > fh:
            return None
        return vx, vy

    def _refresh_kf_marker_status(self):
        if self._kf_start is None:
            self.lbl_status.setText("点击视频可标记关键帧坐标")
            return
        ktype = ACTION_TYPE_MAP.get(self.cmb_action.currentText(), "annotation")
        label = ACTION_LABEL_MAP.get(ktype, ktype)
        self.lbl_status.setText(
            f"[{label}] 坐标=({self._kf_start[0]:.1f},{self._kf_start[1]:.1f})")

    def _on_action_changed(self):
        self._refresh_kf_marker_status()
        if self._kf_start is not None:
            self.refresh_frame()

    def _next_manual_id(self):
        ids = [int(e["event_id"]) for e in self.events]
        ids += [int(m["event_id"]) for m in self.manual_events]
        return (max(ids) if ids else 0) + 1

    def add_keyframe(self):
        if self.cap is None:
            QMessageBox.information(self, "提示", "请先加载视频")
            return
        if not self.participant:
            QMessageBox.information(self, "提示", "请先加载 CSV 并选择参与者")
            return
        ktype = ACTION_TYPE_MAP.get(self.cmb_action.currentText(), "annotation")
        note = self.edit_keynote.text().strip()
        ts_float = self.current_time + self.offset
        if self._kf_start is not None:
            x0, y0 = self._kf_start
            x1, y1 = self._kf_end if self._kf_end is not None else (x0, y0)
        else:
            x0, y0, x1, y1 = 0.0, 0.0, 0.0, 0.0
        distance = math.hypot(x1 - x0, y1 - y0)
        event_id = self._next_manual_id()
        ts_str = format_ts(ts_float)
        e = {
            "participant": self.participant,
            "event_type": ktype,
            "event_ts": ts_str,
            "event_ts_float": ts_float,
            "window_start": ts_str,
            "window_end": ts_str,
            "duration_sec": 0.0,
            "distance_px": round(distance, 1),
            "start_x": round(x0, 1),
            "start_y": round(y0, 1),
            "end_x": round(x1, 1),
            "end_y": round(y1, 1),
            "delete_key": "",
            "event_id": event_id,
            "notes": note,
            "is_manual": True,
        }
        self.manual_events.append(e)
        self.save_manual_keyframes()
        self._build_events()
        self.rebuild_table()
        self.update_timeline()
        self.selected_event_id = event_id
        idx = self._index_of(event_id)
        if idx is not None:
            self.table.selectRow(idx)
        self.edit_keynote.clear()
        self._kf_start = None
        self._kf_end = None
        self.refresh_frame()
        self.lbl_status.setText(
            f"已添加关键帧 #{event_id} [{ktype}] @ {self.current_time:.2f}s "
            f"坐标=({x0:.1f},{y0:.1f}) 备注={note}")

    def delete_keyframe(self):
        e = self._find_event(self.selected_event_id) if self.selected_event_id else None
        if e is None or not e.get("is_manual"):
            QMessageBox.information(self, "提示", "请先在事件表中选择一个人工添加的关键帧（类型带“手动”）")
            return
        ret = QMessageBox.question(
            self, "确认删除",
            f"确定删除人工关键帧 #{e['event_id']} ({e['event_type']}) @ {e['event_ts']} ?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ret != QMessageBox.Yes:
            return
        self.manual_events = [m for m in self.manual_events if m["event_id"] != e["event_id"]]
        self.save_manual_keyframes()
        if self.selected_event_id == e["event_id"]:
            self.selected_event_id = None
        self._build_events()
        self.rebuild_table()
        self.update_timeline()
        self.lbl_status.setText(f"已删除人工关键帧 #{e['event_id']}")

    def calibrate(self):
        if self.cap is None:
            QMessageBox.information(self, "提示", "请先加载视频")
            return
        if self.selected_event_id is None:
            QMessageBox.information(self, "提示", "请先在列表中选择一个事件")
            return
        e = self._find_event(self.selected_event_id)
        if e is None:
            return
        video_time = self.current_frame / self.fps
        self.offset = e["event_ts_float"] - video_time
        self.offsets[self.participant] = self.offset
        self.save_offsets()
        self.update_timeline()
        self.lbl_status.setText(
            f"已校准 {self.participant}: offset={self.offset:.3f}s "
            f"(第0帧 = {self.offset:.3f})")

    def mark_correct(self):
        self._set_mark("ok")

    def mark_wrong(self):
        self._set_mark("bad")

    def clear_mark(self):
        self._set_mark("")

    def _set_mark(self, value):
        if self.selected_event_id is None:
            QMessageBox.information(self, "提示", "请先选择一个事件")
            return
        e = self._find_event(self.selected_event_id)
        if e is not None and e.get("is_manual"):
            QMessageBox.information(self, "提示", "人工关键帧无需判断")
            return
        r = self.results.setdefault(int(self.selected_event_id),
                                    {"verified": "", "notes": ""})
        r["verified"] = value
        self.save_results()
        self.rebuild_table()

    def save_note(self):
        if self.selected_event_id is None:
            return
        e = self._find_event(self.selected_event_id)
        if e is None:
            return
        text = self.edit_note.text()
        if e.get("is_manual"):
            for m in self.manual_events:
                if m["event_id"] == e["event_id"]:
                    m["notes"] = text
                    break
            self.save_manual_keyframes()
            self._build_events()
            self.rebuild_table()
        else:
            r = self.results.setdefault(int(self.selected_event_id),
                                        {"verified": "", "notes": ""})
            r["notes"] = text
            self.save_results()
            self.rebuild_table()

    def _refresh_status(self):
        e = self._find_event(self.selected_event_id) if self.selected_event_id else None
        if e is None:
            self.lbl_status.setText("")
            return
        r = self.results.get(self.selected_event_id or 0, {})
        notes = e.get("notes", "")
        self.lbl_status.setText(
            f"事件 {e['event_id']}  {e['event_type']}{'(手动)' if e.get('is_manual') else ''}  "
            f"ts={e['event_ts']}  判断={r.get('verified','')}  备注={notes}")
        self.edit_note.setText(notes)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Left:
            self.step_event(-1)
        elif key == Qt.Key_Right:
            self.step_event(1)
        elif key == Qt.Key_BracketLeft:
            self._step_frame(-1)
        elif key == Qt.Key_BracketRight:
            self._step_frame(1)
        elif key == Qt.Key_C:
            self.calibrate()
        elif key == Qt.Key_K:
            self.add_keyframe()
        elif key == Qt.Key_Delete:
            self.delete_keyframe()
        elif key == Qt.Key_1:
            self.mark_correct()
        elif key == Qt.Key_2:
            self.mark_wrong()
        elif key == Qt.Key_0:
            self.clear_mark()
        else:
            super().keyPressEvent(event)

    def _step_frame(self, direction):
        if self.cap is None:
            return
        self._seek_frame(self.current_frame + direction)

    def closeEvent(self, event):
        self.save_results()
        if self.cap is not None:
            self.cap.release()
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = AnnotationPlayer()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
