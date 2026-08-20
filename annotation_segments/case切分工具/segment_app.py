"""
切分工具图形界面 (双击 start.bat 或本文件启动)
"""
import os
import sys
import json
import threading
import webbrowser
import tkinter as tk
from tkinter import ttk, messagebox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import segment_cases as sc
import visualize_cases as vc

BASE = r"C:\Users\wangzhi\Desktop\AI读片原数据"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "结果")
INDEX = os.path.join(OUT_DIR, "_index.html")


def load_events(path):
    ev = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            ev.append(json.loads(line))
        except Exception:
            pass
    return ev


def segment_participant(d):
    jl = sc.find_keymouse_jsonl(d)
    if not jl:
        return None, "无键鼠"
    dbl = sc.detect_double_clicks(load_events(jl), dt_max_ms=500, dist_max_px=50)
    dbl = sc.merge_reclicks(dbl)
    band, chain, _ = sc.best_sawtooth_band(dbl, gap=80, step_range=(10, 80))
    g = [len(x) for x in sc.chain_groups(chain)]
    n = len(chain)
    if n == 60 and g == [20, 20, 20]:
        status = "✓ 完美"
    elif abs(n - 60) <= 2:
        status = "接近"
    elif n > 60:
        status = f"偏高+{n - 60}"
    else:
        status = f"偏低{n - 60}"
    return (n, str(g)), status


class App:
    def __init__(self, root):
        self.root = root
        root.title("CT 病例切分工具")
        root.geometry("760x560")

        top = tk.Frame(root)
        top.pack(fill="x", padx=12, pady=10)
        tk.Label(top, text="CT 病例自动切分工具", font=("Segoe UI", 15, "bold")).pack(side="left")

        btn = tk.Frame(root)
        btn.pack(fill="x", padx=12)
        self.b_refresh = tk.Button(btn, text="① 运行切分", command=self.run_segment, width=16)
        self.b_refresh.pack(side="left", padx=4)
        self.b_report = tk.Button(btn, text="② 生成截图报告", command=self.run_report, width=16)
        self.b_report.pack(side="left", padx=4)
        self.b_open = tk.Button(btn, text="打开总报告", command=self.open_index, width=14)
        self.b_open.pack(side="left", padx=4)
        self.b_open_row = tk.Button(btn, text="打开选中参与者报告", command=self.open_row, width=20)
        self.b_open_row.pack(side="left", padx=4)

        cols = ("p", "n", "g", "status")
        self.tree = ttk.Treeview(root, columns=cols, show="headings")
        self.tree.heading("p", text="参与者")
        self.tree.heading("n", text="case 数")
        self.tree.heading("g", text="分组结构")
        self.tree.heading("status", text="状态")
        self.tree.column("p", width=200)
        self.tree.column("n", width=80)
        self.tree.column("g", width=180)
        self.tree.column("status", width=120)
        self.tree.pack(fill="both", expand=True, padx=12, pady=6)

        self.log = tk.Text(root, height=6, bg="#1e1e1e", fg="#ccc", state="disabled")
        self.log.pack(fill="x", padx=12, pady=(0, 12))

        self.participants = []
        self.results = {}

    def log_msg(self, msg):
        self.log.config(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def populate(self):
        self.participants = sorted(
            d for d in os.listdir(BASE) if os.path.isdir(os.path.join(BASE, d)))
        self.tree.delete(*self.tree.get_children())
        for p in self.participants:
            self.tree.insert("", "end", iid=p, values=(p, "-", "-", "未运行"))

    def run_segment(self):
        self.b_refresh.config(state="disabled")
        self.results = {}

        def work():
            for p in self.participants:
                d = os.path.join(BASE, p)
                info, status = segment_participant(d)
                self.results[p] = (info, status)
                if info is None:
                    self.tree.item(p, values=(p, "-", "-", status))
                else:
                    self.tree.item(p, values=(p, info[0], info[1], status))
                self.log_msg(f"  {p}: {info[1] if info else '-'}  {status}")
            self.log_msg("切分完成。")

        def done():
            self.b_refresh.config(state="normal")

        self.log_msg("开始切分...")
        threading.Thread(target=lambda: (work(), self.root.after(0, done))).start()

    def run_report(self):
        self.b_report.config(state="disabled")
        self.log_msg("开始生成截图报告 (约 1 分钟)...")

        def work():
            for p in self.participants:
                out = vc.generate_report(os.path.join(BASE, p))
                self.log_msg(f"  {p}: {'OK' if out else '跳过'}")
            self.log_msg("报告生成完毕。")

        def done():
            self.b_report.config(state="normal")
            self.open_index()

        threading.Thread(target=lambda: (work(), self.root.after(0, done))).start()

    def open_index(self):
        if os.path.exists(INDEX):
            webbrowser.open(INDEX)
        else:
            messagebox.showinfo("提示", "还没生成总报告, 先点 ②")

    def open_row(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("提示", "先在上表选中一个参与者")
            return
        p = sel[0]
        rep = os.path.join(OUT_DIR, f"{p}_report.html")
        if os.path.exists(rep):
            webbrowser.open(rep)
        else:
            messagebox.showinfo("提示", f"{p} 还没有报告, 先点 ② 生成")


def main():
    root = tk.Tk()
    app = App(root)
    app.populate()
    root.mainloop()


if __name__ == "__main__":
    main()
