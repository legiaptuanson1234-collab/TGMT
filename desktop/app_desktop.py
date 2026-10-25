# app_desktop.py - v9: BANG V7 CU (chay mat) + CHATBOT GIAO THONG (cap nhat lien tuc)
# === CHANT TRUYEN (giu y v7 - video chay nhu cu, Line & ROI deu chay duoc) ===
#   - LineCounter / PolygonCounter (KHONG dung MultiCounter)
#   - update_frame chui toi main-thread qua self.after (khong phai thread nen)
# === CHINH THEM (chatbot + CSV live) ===
#   - TrafficStore ghi su kien xe LIEN TUC; "bot" doc ngay, khong cho video het
#   - khung chat + nut hoi: 'nay co bao nhieu xe?', 'un tac chua?', 'xuat csv', ...
import customtkinter as ctk
from tkinter import filedialog, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import sys
import time
import threading
from datetime import datetime

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
# PyInstaller: best.pt/bytetrack.yaml/configs nam trong _MEIPASS;
# ban chay truc tiep: nam o repo root.
_ROOT = sys._MEIPASS if getattr(sys, "frozen", False) else _REPO_ROOT
for _p in (_ROOT, _THIS_DIR, _REPO_ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from counting import LineCounter, PolygonCounter
from traffic_tracker import TrafficTracker
from livestore import TrafficStore
import trafficbot
from ultralytics import YOLO

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class TrafficAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("🚦 HỆ THỐNG ĐẾM XE AI + CHATBOT (UTT) — chuẩn v7")
        self.geometry("1280x820")

        print("[*] Đang load bộ não YOLOv8...")
        pt = os.path.join(_ROOT, "best.pt") if os.path.exists(os.path.join(_ROOT, "best.pt")) else \
             os.path.join(_REPO_ROOT, "best.pt")
        self.model = YOLO(pt)
        self.tracker = TrafficTracker(self.model)
        # [them] store de chatbot doc du lieu LIEN TUC
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
        # tu cap nhat KPI + khung (moi 0.8s) - khong can user lam gi
        self._after(400, self._poll_live)

    # ================= UI (giu bang v7, them khung chat) =================
    def setup_ui(self):
        # ---------- SIDEBAR (y hieu v7) ----------
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)
        ctk.CTkLabel(self.sidebar, text="BẢNG ĐIỀU KHIỂN", font=ctk.CTkFont(size=20, weight="bold")).pack(padx=20, pady=20)

        self.btn_select = ctk.CTkButton(self.sidebar, text="📂 1. Chọn Video", command=self.select_video, height=40)
        self.btn_select.pack(padx=20, pady=10, fill="x")
        self.lbl_vid_name = ctk.CTkLabel(self.sidebar, text="Chưa chọn video...", text_color="gray")
        self.lbl_vid_name.pack(padx=20, pady=5)

        ctk.CTkLabel(self.sidebar, text="Chọn Chế Độ Đếm:").pack(padx=20, pady=(15, 0), anchor="w")
        self.radio_line = ctk.CTkRadioButton(self.sidebar, text="Đếm qua vạch (Line)", variable=self.draw_mode, value="line")
        self.radio_line.pack(padx=20, pady=5, anchor="w")
        self.radio_roi = ctk.CTkRadioButton(self.sidebar, text="Đếm trong vùng (ROI)", variable=self.draw_mode, value="roi")
        self.radio_roi.pack(padx=20, pady=5, anchor="w")

        self.btn_draw = ctk.CTkButton(self.sidebar, text="🖊️ 2. Vẽ Vạch / Vùng đếm", fg_color="#E67E22", hover_color="#D35400",
                                      command=self.open_drawing_window, height=40)
        self.btn_draw.pack(padx=20, pady=15, fill="x")

        self.btn_start = ctk.CTkButton(self.sidebar, text="▶️ 3. Bắt Đầu Đếm (+ mở CHATBOT)", fg_color="green",
                                       command=self.start_processing, height=40)
        self.btn_start.pack(padx=20, pady=10, fill="x")
        self.btn_pause = ctk.CTkButton(self.sidebar, text="⏯️ Tạm dừng / Tiếp", fg_color="#F1C40F",
                                       command=self.toggle_pause, height=32)
        self.btn_pause.pack(padx=20, pady=4, fill="x")
        self.btn_stop = ctk.CTkButton(self.sidebar, text="⏹️ Dừng & Lưu (xuất CSV)", fg_color="red",
                                      command=self.stop_processing, height=32)
        self.btn_stop.pack(padx=20, pady=10, fill="x")

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Sẵn sàng", text_color="yellow", font=ctk.CTkFont(weight="bold"))
        self.lbl_status.pack(pady=12)

        # [them] KPI LIVE
        ctk.CTkLabel(self.sidebar, text="— CẢM BIẾN REAL-TIME —", font=ctk.CTkFont(size=11, weight="bold")).pack(pady=(4, 0))
        kbox = ctk.CTkFrame(self.sidebar, fg_color=("gray88", "gray17"))
        kbox.pack(padx=14, pady=6, fill="x")
        self.lbl_kpi = ctk.CTkLabel(kbox, text="Đang chờ chạy...", justify="left",
                                    font=ctk.CTkFont(size=12), text_color=("gray10", "gray90"))
        self.lbl_kpi.pack(padx=12, pady=8)
        self.lbl_csv = ctk.CTkLabel(self.sidebar, text="CSV live: chưa chạy", text_color="gray",
                                    font=ctk.CTkFont(size=11), anchor="w", wraplength=270)
        self.lbl_csv.pack(padx=16, pady=(0, 4), anchor="w")
        ctk.CTkButton(self.sidebar, text="📂 Mở thư mục CSV (Bao_Cao)", height=28, command=self._open_csv_dir).pack(padx=16, pady=(0, 6), fill="x")

        # ---------- PHAI: video + CHATBOT ----------
        right = ctk.CTkFrame(self)
        right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.video_frame = ctk.CTkFrame(right, fg_color=("gray20", "gray10"))
        self.video_frame.pack(fill="x", expand=True)
        self.video_label = ctk.CTkLabel(self.video_frame,
                                       text="Màn hình hiển thị AI\n(1-Chọn video → 2-Vẽ → ▶️ Chạy\n  Khi chạy → HỎI CHATBOT bên dưới, không chờ video hết)",
                                       font=ctk.CTkFont(size=16), text_color="gray")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        # ---- CHATBOT PANEL [them] ----
        chat_box = ctk.CTkFrame(right, height=240)
        chat_box.pack(fill="x", pady=6)
        ctk.CTkLabel(chat_box,
                     text="💬 CHATBOT GIAO THÔNG (cập nhật liên tục) — ví dụ: 'nay có bao nhiêu xe?' · 'ùn tắc chưa?' · 'xuất csv'",
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
        self._append_chat("🤖 Xin chào! Tôi là Traffic Bot. Bấm ▶️ CHẠY video rồi HỎI TÔI bất kỳ lúc nào:\n"
                          "  • 'nay có bao nhiêu xe?' → tổng + phân loại + đỉnh cao điểm\n"
                          "  • 'từ 2 phút sang 5 phút có bao nhiêu xe máy?' → thống kê đoạn\n"
                          "  • 'ùn tắc chưa?' → trạng thái 🟢🟡🟠🔴\n"
                          "  • 'tốc độ trung bình?' / 'xuất csv' (file ghi liên tục, mở Excel ngay)")

    # ================= helpers [them] =================
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
            self._set_status("Không mở được thư mục CSV: %s" % e, "red")

    def _set_status(self, text, color="yellow"):
        self.lbl_status.configure(text=text, text_color=color)

    def _after(self, ms, fn):
        try:
            self.after(ms, fn)
        except Exception:
            pass

    def _poll_live(self):
        """Duy thep KPI + khung video moi 0.8s (doc tu store - thread AI ghi)."""
        try:
            s = self.store.stats()
            self.lbl_kpi.configure(text="Tổng: %d · Trong khung: %d · FPS: %s\nMật độ: %s xe/ph · Trạng thái: %s\nTốc độ TB: %.1f km/h"
                                   % (s["total"], s["active_now"], s["fps"], s["per_minute"], s["status"], s["avg_speed"]))
            if s["detail_csv"]:
                self.lbl_csv.configure(text="📡 CSV live đang ghi: %s" % os.path.basename(s["detail_csv"]))
            # hien khung moi nhat neu AI dang chay
            if self.is_running:
                fr = self.store.get_frame()
                if fr is not None:
                    self._show_frame(fr)
        except Exception:
            pass
        self._after(800, self._poll_live)

    def _fit_img(self, img):
        """[FIX vid be] co PHONG image cho duong khit khung video (giu ty le 16:9).
        Ban cu chi thumbnail (co nho) nen lan dau khung con nho -> vid bi be."""
        fw, fh = self.video_frame.winfo_width() - 20, self.video_frame.winfo_height() - 20
        if fw > 60 and fh > 60:
            w, h = img.size
            s = min(fw / w, fh / h)
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            if (nw, nh) != (w, h):
                img = img.resize((nw, nh), Image.LANCZOS)
        return img

    def _show_frame(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = self._fit_img(Image.fromarray(rgb))
        self._imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.configure(image=self._imgtk, text="")

    # ================= video / draw (y hieu v7) =================
    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.lbl_vid_name.configure(text="Video: " + os.path.basename(path), text_color="white")
            self.points = []
            self.counter_obj = None
            self.store.reset()

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
            self._set_status("❌ Chọn video trước!", "red")
            return
        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return
        frame = cv2.resize(frame, (1280, 720))
        self.points = []

        win = "VE VACH/VUNG: CLICK TRAI ve | CLICK PHAI xoa diem | ENTER luu"
        cv2.namedWindow(win)
        cv2.setMouseCallback(win, self.mouse_callback)

        while True:
            tmp = frame.copy()
            mode = self.draw_mode.get()
            for p in self.points:
                cv2.circle(tmp, tuple(p), 5, (0, 0, 255), -1)
            if mode == "line" and len(self.points) == 2:
                cv2.line(tmp, tuple(self.points[0]), tuple(self.points[1]), (255, 0, 255), 2)
            elif mode == "roi" and len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(tmp, [pts], False, (255, 0, 255), 2)
                if len(self.points) > 2:
                    cv2.line(tmp, tuple(self.points[-1]), tuple(self.points[0]), (0, 255, 255), 1)
            cv2.imshow(win, tmp)
            if cv2.waitKey(1) & 0xFF in (13, ord("q")):
                break
        cv2.destroyAllWindows()

        # [y hieu v7] LineCounter / PolygonCounter - deu chay duoc, khong lan
        mode = self.draw_mode.get()
        if mode == "line" and len(self.points) == 2:
            self.counter_obj = LineCounter([self.points[0], self.points[1]])
            self._set_status("✅ Đã lưu VẠCH (2 điểm). Bấm ▶️ Chạy!", "green")
        elif mode == "roi" and len(self.points) >= 3:
            self.counter_obj = PolygonCounter(self.points)
            self._set_status("✅ Đã lưu VÙNG (%d góc). Bấm ▶️ Chạy!" % len(self.points), "green")
        else:
            self._set_status("❌ Vẽ chưa đủ (vạch 2 điểm, vùng ≥3 góc).", "red")
            self.counter_obj = None

    # ================= chạy AI (dong v7 + ghi store) =================
    def start_processing(self):
        if not self.video_path or not self.counter_obj:
            self._set_status("❌ Chưa chọn video hoặc chưa vẽ!", "red")
            return
        if self.cap:
            self.cap.release()

        os.makedirs("Video_Xuat", exist_ok=True)
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join("Video_Xuat", "Video_AI_%s.mp4" % time_str)
        self.cap = cv2.VideoCapture(self.video_path)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30
        self.out_video = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (1280, 720))

        # [them] noi chatbot/CSV LIEN TUC
        self.store.reset()
        self.store.on_start()
        self.tracker.reset_counters()
        self.tracker.store = self.store
        self.tracker.run_start = time.time()

        self.is_running = True
        self.is_paused = False
        self._set_status("🚀 Đang chạy — HỎI CHATBOT dưới đây NGAY (không cần chờ video hết)!", "green")
        self._append_chat("🤖 Video bắt đầu lúc " + datetime.now().strftime("%H:%M:%S") +
                          " · CSV live: " + os.path.basename(self.store.detail_csv))
        self.update_frame()

    def update_frame(self):
        if self.is_running and self.cap is not None and self.cap.isOpened():
            if not self.is_paused:
                success, frame = self.cap.read()
                if success:
                    frame = cv2.resize(frame, (1280, 720))
                    processed_frame = self.tracker.process_frame(frame, self.counter_obj)
                    # ghi frame da ve vao file video
                    if self.out_video:
                        self.out_video.write(processed_frame)
                    # hien trang - [FIX vid be] co PHONG duong khit khung (khong chi co nho)
                    cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    img = self._fit_img(Image.fromarray(cv2image))
                    self._imgtk = ImageTk.PhotoImage(image=img)
                    self.video_label.configure(image=self._imgtk, text="")
                else:
                    self.stop_processing()
            self._after(10, self.update_frame)
        elif self.is_running:
            self.stop_processing()

    def toggle_pause(self):
        if not self.is_running:
            self._set_status("Chưa có phiên chạy để tạm dừng.", "red")
            return
        self.is_paused = not self.is_paused
        self._set_status("⏸ Đã tạm dừng (bấm lại để tiếp)." if self.is_paused else "▶ Tiếp tục.", "yellow")

    def stop_processing(self):
        self.is_running = False
        self.is_paused = False
        if self.cap:
            self.cap.release()
        if self.out_video:
            self.out_video.release()
        self.tracker.store = None
        self.store.on_finish()
        s = self.store.stats()
        # [them] xuac CSV thong ke + thong bao
        p = self.store.write_summary_csv()
        self.store.close_csv()
        self._set_status("✅ Dừng & Lưu: %d xe · CSV live: %s · Thống kê: %s"
                         % (s["total"], os.path.basename(s["detail_csv"] or ""), os.path.basename(p)), "yellow")
        self._append_chat("🤖 Kết thúc: tổng %d xe. Đã xuất CSV: %s" % (s["total"], os.path.basename(p)))

    # ================= chatbot [them] =================
    def send_chat(self):
        q = self.chat_entry.get().strip()
        if not q:
            return
        self.chat_entry.delete(0, "end")
        self._append_chat("👤", q)
        s = self.store.stats()
        reply = trafficbot.ask(self.store, q, s["video_sec"])
        self._append_chat("🤖", reply)

    # ================= smoke-test (chay: UTT_Traffic_Copilot.exe --smoke) =================
    if False:
        pass


if __name__ == "__main__":
    import sys as _sys
    if "--smoke" in _sys.argv:
        try:
            _fr = (np.random.rand(720, 1280, 3) * 255).astype("uint8")
            _m = YOLO(os.path.join(_ROOT, "best.pt") if os.path.exists(os.path.join(_ROOT, "best.pt"))
                      else os.path.join(_REPO_ROOT, "best.pt"))
            _res = _m.track(_fr, persist=True,
                            tracker=os.path.join(os.getcwd(), "bytetrack.yaml") if os.path.exists("bytetrack.yaml") else "bytetrack.yaml",
                            conf=0.35, verbose=False)
            print("SMOKE OK: YOLO + ByteTrack (%d boxes)" % (len(_res[0].boxes) if _res[0].boxes is not None else 0))
            _sys.exit(0)
        except Exception as _e:
            import traceback
            traceback.print_exc()
            print("SMOKE FAIL: %s" % _e)
            _sys.exit(1)

    app = TrafficAIApp()
    app.mainloop()
