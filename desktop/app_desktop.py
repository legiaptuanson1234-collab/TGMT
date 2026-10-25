# app_desktop.py (v8) - TRAFFIC COPILOT desktop (CustomTkinter)
# = Bản Streamlit v8 chạy TRỰC TIẾP trên máy: video + YOLO + ByteTrack
#   + CHATBOT giao thông real-time (hỏi trong lúc video chạy)
#   + CSV sự kiện ghi LIÊN TỤC (mở Excel ngay, không chờ kết thúc)
#   + KPI live (trạng thái/ tổng xe/ FPS/ km/h) + nút Tạm dừng/Dừng
import customtkinter as ctk
from tkinter import filedialog, scrolledtext
import cv2
from PIL import Image, ImageTk
import os
import sys
import time
import threading
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
# PyInstaller (bản đóng gói): data (best.pt, bytetrack.yaml) nằm trong _MEIPASS;
# bản chạy trực tiếp: nằm ở repo root.
_ROOT = sys._MEIPASS if getattr(sys, "frozen", False) else _REPO_ROOT
for _p in (_ROOT, _THIS_DIR, _REPO_ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from counting import LineCounter, PolygonCounter, MultiCounter
from traffic_tracker import TrafficTracker
from livestore import TrafficStore
import trafficbot
from ultralytics import YOLO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class TrafficCopilot(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🚦 TRAFFIC COPILOT — Hệ thống đếm xe & chatbot giao thông (UTT)")
        self.geometry("1360x800")

        print("[*] Đang load bộ não YOLOv8 (best.pt)...")
        pt = os.path.join(_ROOT, "best.pt") if os.path.exists(os.path.join(_ROOT, "best.pt")) else \
             os.path.join(_REPO_ROOT, "best.pt")
        self.model = YOLO(pt)
        self.tracker = TrafficTracker(self.model)

        self.store = TrafficStore()

        self.video_path = None
        self.cap = None
        self.is_running = False
        self.is_paused = False
        self.out_video = None

        self.draw_mode = ctk.StringVar(value="line")
        self.points = []
        self.counter_obj = None
        self._imgtk = None

        self.setup_ui()
        self._after(500, self._poll_kpi)

    # ---------------- UI ----------------
    def setup_ui(self):
        # ---------- SIDEBAR ----------
        side = ctk.CTkFrame(self, width=300, corner_radius=0)
        side.pack(side="left", fill="y", padx=8, pady=8)

        ctk.CTkLabel(side, text="BẢNG ĐIỀU KHIỂN", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=18, pady=(14, 4))

        self.btn_select = ctk.CTkButton(side, text="📂 1. CHỌN VIDEO", command=self.select_video, height=38)
        self.btn_select.pack(padx=16, pady=8, fill="x")
        self.lbl_vid = ctk.CTkLabel(side, text="Chưa chọn video...", text_color="gray", font=ctk.CTkFont(size=12))
        self.lbl_vid.pack(padx=18, anchor="w")

        ctk.CTkLabel(side, text="2. CHẾ ĐỘ ĐẾM", anchor="w", font=ctk.CTkFont(size=13)).pack(padx=18, pady=(10, 0))
        self.rb_line = ctk.CTkRadioButton(side, text="Đếm qua vạch (Line)", variable=self.draw_mode, value="line")
        self.rb_line.pack(padx=24, anchor="w")
        self.rb_roi = ctk.CTkRadioButton(side, text="Đếm trong vùng (ROI)", variable=self.draw_mode, value="roi")
        self.rb_roi.pack(padx=24, anchor="w")

        ctk.CTkLabel(side, text="3. HIỆU NĂNG", anchor="w", font=ctk.CTkFont(size=13)).pack(padx=18, pady=(8, 0))
        self.perf = ctk.StringVar(value="nhanh")
        self.rb_perf_light = ctk.CTkRadioButton(side, text="Nhanh (CPU · 640×360)", variable=self.perf, value="nhanh")
        self.rb_perf_light.pack(padx=24, anchor="w")
        self.rb_perf_full = ctk.CTkRadioButton(side, text="Chất lượng (GPU · 1280×720)", variable=self.perf, value="chat")
        self.rb_perf_full.pack(padx=24, anchor="w")

        self.btn_draw = ctk.CTkButton(side, text="🖊️ 4. VẼ VẠCH / VÙNG", fg_color="#E67E22", hover_color="#D35400",
                                      command=self.open_drawing_window, height=38)
        self.btn_draw.pack(padx=16, pady=10, fill="x")

        self.btn_start = ctk.CTkButton(side, text="🚀 BẮT ĐẦU CHẠY (chạy CHATBOT được ngay)", fg_color="green",
                                       command=self.start_processing, height=38)
        self.btn_start.pack(padx=16, pady=6, fill="x")
        self.btn_pause = ctk.CTkButton(side, text="⏯️ Tạm dừng / Tiếp tục", fg_color="#F1C40F",
                                       command=self.toggle_pause, height=32)
        self.btn_pause.pack(padx=16, pady=4, fill="x")
        self.btn_stop = ctk.CTkButton(side, text="⛔ Dừng & Lưu (xuất CSV)", fg_color="red",
                                      command=self.stop_processing, height=32)
        self.btn_stop.pack(padx=16, pady=6, fill="x")

        self.lbl_status = ctk.CTkLabel(side, text="Sẵn sàng", text_color="yellow", font=ctk.CTkFont(weight="bold", size=13))
        self.lbl_status.pack(pady=8)

        # KPI live
        ctk.CTkLabel(side, text="— CẢM BIẾN REAL-TIME —", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(4, 0))
        self.kpi_frame = ctk.CTkFrame(side, fg_color=("gray88", "gray17"))
        self.kpi_frame.pack(padx=12, pady=6, fill="x")
        self.lbl_kpi = ctk.CTkLabel(self.kpi_frame, text="Đang chờ chạy...\nTổng: 0 · Trong khung: 0 · FPS: -\nMật độ: - · Trạng thái: -",
                                    justify="left", font=ctk.CTkFont(size=12), text_color=("gray10", "gray90"))
        self.lbl_kpi.pack(padx=12, pady=8)

        # CSV live
        self.lbl_csv = ctk.CTkLabel(side, text="CSV live: chưa chạy", text_color="gray", font=ctk.CTkFont(size=11),
                                    anchor="w", wraplength=270)
        self.lbl_csv.pack(padx=16, pady=(2, 4), anchor="w")
        ctk.CTkButton(side, text="📂 Mở thư mục CSV (Bao_Cao)", height=28,
                      command=self._open_csv_dir).pack(padx=16, pady=(0, 4), fill="x")

        # ---------- PHẢI: video + CHATBOT ----------
        right = ctk.CTkFrame(self)
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self.video_frame = ctk.CTkFrame(right, fg_color=("gray20", "gray10"))
        self.video_frame.pack(fill="both", expand=True, padx=6, pady=(6, 4))
        self.video_label = ctk.CTkLabel(self.video_frame,
                                        text="Màn hình AI (live)\n\nLàm theo: 1-Chọn video → 2-Chế độ → 3-Hiệu năng → 4-Vẽ → 🚀 Chạy\n\n"
                                             "Khi chạy → hỏi CHATBOT bên dưới, không cần chờ video kết thúc.",
                                        font=ctk.CTkFont(size=16), text_color="gray")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- CHATBOT PANEL ----
        chat_box = ctk.CTkFrame(right, height=220)
        chat_box.pack(fill="x", padx=6, pady=6)
        ctk.CTkLabel(chat_box, text="💬 CHATBOT GIAO THÔNG (real-time) — ví dụ: 'nay có bao nhiêu xe?' · 'ùn tắc chưa?' · 'xuất csv'",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(6, 0), anchor="w")
        self.chat_log = scrolledtext.ScrolledText(chat_box, height=7, wrap="word", font=("Consolas", 11),
                                                  state="disabled", bg="#111", fg="#ddd")
        self.chat_log.pack(fill="both", expand=True, padx=8, pady=(4, 4))
        entry_row = ctk.CTkFrame(chat_box)
        entry_row.pack(fill="x", padx=8, pady=(0, 8))
        self.chat_entry = ctk.CTkEntry(entry_row, placeholder_text="Hỏi bot: bao nhiêu xe / loại nào / mật độ / tốc độ / xuất csv...")
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.chat_entry.bind("<Return>", lambda e: self.send_chat())
        ctk.CTkButton(entry_row, text="➤", width=44, command=self.send_chat).pack(side="right")
        for q in ("nay có bao nhiêu xe", "mật độ thế nào", "có bao nhiêu xe máy"):
            pass
        self._append_chat("🤖 Xin chào! Tôi là Traffic Bot. Chạy video (🚀) rồi hỏi tôi bất kỳ lúc nào:\n"
                          "  • 'nay có bao nhiêu xe?' → tổng + phân loại + đỉnh\n"
                          "  • 'từ 2 phút sang 5 phút có bao nhiêu xe?' → thống kê đoạn\n"
                          "  • 'ùn tắc chưa?' → trạng thái 🟢🟡🟠🔴\n"
                          "  • 'tốc độ trung bình?' / 'xuất csv'")

    # ---------------- helpers ----------------
    def _append_chat(self, who, text=""):
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", who + " " + text + "\n\n")
        self.chat_log.see("end")
        self.chat_log.configure(state="disabled")

    def _open_csv_dir(self):
        try:
            import subprocess
            d = os.path.join(os.getcwd(), "Bao_Cao")
            os.makedirs(d, exist_ok=True)
            subprocess.Popen(["explorer", d])
        except Exception as e:
            self.set_status("Không mở được thư mục CSV: %s" % e, "red")

    def _poll_kpi(self):
        """Lấy KPI từ store (thread AI ghi) -> cập nhật nhãn mỗi 0.8s (không chặn)."""
        try:
            s = self.store.stats()
            self.lbl_kpi.configure(text="Tổng: %d · Trong khung: %d · FPS: %s\nMật độ: %s xe/ph · Trạng thái: %s\nTốc độ TB: %.1f km/h"
                                   % (s["total"], s["active_now"], s["fps"], s["per_minute"], s["status"], s["avg_speed"]))
            if s["detail_csv"]:
                self.lbl_csv.configure(text="📡 CSV live đang ghi: %s" % os.path.basename(s["detail_csv"]), text_color="gray")
        except Exception:
            pass
        self._after(800, self._poll_kpi)

    def _after(self, ms, fn):
        try:
            self.after(ms, fn)
        except Exception:
            pass

    def set_status(self, text, color="yellow"):
        self.lbl_status.configure(text=text, text_color=color)

    # ---------------- video / drawing (giữ logic v7) ----------------
    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.lbl_vid.configure(text="Video: " + os.path.basename(path), text_color="white")
            self.points = []
            self.counter_obj = None
            self.store.reset()
            self.set_status("✅ Đã chọn. Vẽ vạch/vùng rồi bấm CHẠY.", "green")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = self.draw_mode.get()
            if mode == "line" and len(self.points) < 2:
                self.points.append([x, y])
            elif mode == "roi":
                self.points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.points:
                self.points.pop()

    def open_drawing_window(self):
        if not self.video_path:
            self.set_status("❌ Chọn video trước!", "red")
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        frame = cv2.resize(frame, (1280, 720))
        self.points = []

        win = "VE: CHUOT TRAI ve | CHUOT PHAI xoa diem | ENTER luu"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_callback)

        while True:
            tmp = frame.copy()
            mode = self.draw_mode.get()
            for p in self.points:
                cv2.circle(tmp, tuple(p), 5, (0, 0, 255), -1)
            if mode == "line" and len(self.points) == 2:
                cv2.line(tmp, tuple(self.points[0]), tuple(self.points[1]), (255, 0, 255), 3)
            elif mode == "roi" and len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(tmp, [pts], False, (255, 0, 255), 3)
                if len(self.points) > 2:
                    cv2.line(tmp, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 255), 1)
            cv2.imshow(win, tmp)
            if cv2.waitKey(1) & 0xFF in (13, ord("q")):
                break
        cv2.destroyAllWindows()

        mode = self.draw_mode.get()
        if mode == "line" and len(self.points) == 2:
            self.counter_obj = MultiCounter(lines=[tuple(self.points[0]), tuple(self.points[1])])
            self.set_status("✅ Đã lưu VẠCH. Bấm 🚀 Chạy!", "green")
        elif mode == "roi" and len(self.points) >= 3:
            self.counter_obj = MultiCounter(roi_polygons=[self.points])
            self.set_status("✅ Đã lưu VÙNG. Bấm 🚀 Chạy!", "green")
        else:
            self.set_status("❌ Vẽ chưa đủ (vạch cần 2 điểm, vùng cần ≥3).", "red")
            self.counter_obj = None

    # ---------------- chạy AI (thread nền) ----------------
    def start_processing(self):
        if not self.video_path or not self.counter_obj:
            self.set_status("❌ Chưa chọn video hoặc chưa vẽ!", "red")
            return
        if self.cap:
            self.cap.release()

        os.makedirs("Video_Xuat", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out_video = cv2.VideoWriter(os.path.join("Video_Xuat", "Video_AI_%s.mp4" % ts),
                                         cv2.VideoWriter_fourcc(*"mp4v"), 30, (1280, 720))

        # [v8] reset + nối chatbot/CSV
        self.store.reset()
        self.store.on_start(self.perf.get())
        self.tracker.reset_counters()
        self.tracker.store = self.store
        if self.perf.get() == "nhanh":
            self.tracker.proc_wh = (640, 360)
            self.tracker.conf = 0.35
        else:
            self.tracker.proc_wh = (1280, 720)
            self.tracker.conf = 0.5
        self.tracker.run_start = time.time()

        self.cap = cv2.VideoCapture(self.video_path)
        self.is_running = True
        self.is_paused = False
        self.set_status("🚀 AI đang chạy NỀN — HỎI CHATBOT BÊN DƯỚI NGAY, không cần chờ!", "green")
        threading.Thread(target=self._ai_loop, daemon=True).start()

    def _ai_loop(self):
        """Vòng xử lý trong thread nền (GUI không bị block)."""
        while self.is_running and self.cap is not None and self.cap.isOpened():
            # tạm dừng
            while self.is_paused and self.is_running:
                time.sleep(0.15)
            if not self.is_running:
                break
            ok, frame = self.cap.read()
            if not ok:
                break
            frame = cv2.resize(frame, (1280, 720))
            out = self.tracker.process_frame(frame, self.counter_obj)
            if self.out_video:
                self.out_video.write(out)
            # cập nhật khung live (GUI đọc)
            self.store.on_frame(out)
        self._finish_run()

    def _finish_run(self):
        if self.cap:
            self.cap.release()
        if self.out_video:
            self.out_video.release()
        self.is_running = False
        self.store.on_finish()
        self.store.close_csv()
        s = self.store.stats()
        self.set_status("✅ Kết thúc: %d xe · %s · CSV: %s" % (
            s["total"], s["status"], os.path.basename(s["detail_csv"]) if s["detail_csv"] else "-"), "yellow")

    def toggle_pause(self):
        if not self.is_running:
            self.set_status("Chưa có phiên chạy nào để tạm dừng.", "red")
            return
        self.is_paused = not self.is_paused
        self.set_status("⏸ Đã tạm dừng (bấm lại để tiếp)." if self.is_paused else "▶ Tiếp tục chạy.", "yellow")

    def stop_processing(self):
        self.is_running = False
        self.is_paused = False
        self._append_chat("👤 [người dùng]", "xuất csv")
        # chờ thread dừng hẳn rồi mới report
        self.after(600, self._finish_run)

    # ---------------- chatbot ----------------
    def send_chat(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        self.chat_entry.delete(0, "end")
        self._append_chat("👤", q)
        # tra dữ liệu real-time (nhanh, không cần thread)
        reply = trafficbot.ask(self.store, q, self.store.stats()["video_sec"])
        self._append_chat("🤖", reply)

    # ---------------- loop video hiển thị (GUI) ----------------
    def _poll_video(self):
        fr = self.store.get_frame()
        if fr is not None:
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            fw, fh = self.video_frame.winfo_width() - 20, self.video_frame.winfo_height() - 20
            if fw > 50 and fh > 50:
                img.thumbnail((fw, fh), Image.Resampling.LANCZOS)
            self._imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.configure(image=self._imgtk, text="")
        self._after(40, self._poll_video)


if __name__ == "__main__":
    # [smoke-test] python UTT_Traffic_Copilot.exe --smoke : kiem chung pipeline AI
    # (YOLO + ByteTrack + torchvision::nms) chay duoc trong moi truong dong goi,
    # bo qua UI de co the test bat-cuc dung ban build.
    if "--smoke" in sys.argv:
        try:
            import numpy as _np
            from ultralytics import YOLO as _Y
            _m = _Y(os.path.join(_ROOT, "best.pt") if os.path.exists(os.path.join(_ROOT, "best.pt")) else os.path.join(_REPO_ROOT, "best.pt"))
            _fr = (_np.random.rand(720, 1280, 3) * 255).astype("uint8")
            _res = _m.track(_fr, persist=True, tracker=os.path.join(os.getcwd(), "bytetrack.yaml") if os.path.exists("bytetrack.yaml") else "bytetrack.yaml", conf=0.35, verbose=False)
            print("SMOKE OK: YOLO + ByteTrack chay duoc (%d boxes)" % (len(_res[0].boxes) if _res and _res[0].boxes is not None else 0))
            sys.exit(0)
        except SystemExit:
            raise
        except Exception as _e:
            import traceback
            traceback.print_exc()
            print("SMOKE FAIL: %s" % _e)
            sys.exit(1)

    app = TrafficCopilot()
    app._after(100, app._poll_video)
    app.mainloop()
