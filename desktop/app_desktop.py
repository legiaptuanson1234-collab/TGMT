import customtkinter as ctk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import sys
from datetime import datetime
from ultralytics import YOLO

# Dùng CHUNG module đếm ở ROOT repo (counting.py), tránh bản trùng trong thư mục desktop.
# File tracking của desktop đã tách thành traffic_tracker.py (class TrafficTracker).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from counting import LineCounter, PolygonCounter
from traffic_tracker import TrafficTracker

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TrafficAIApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("HỆ THỐNG ĐẾM XE AI - UTT")
        self.geometry("1200x750")

        print("[*] Đang load bộ não YOLOv8...")
        self.model = YOLO(os.path.join(_REPO_ROOT, "best.pt"))
        self.tracker = TrafficTracker(self.model)

        self.video_path = None
        self.cap = None
        self.is_running = False
        self.out_video = None # Biến để lưu video xuất

        self.draw_mode = ctk.StringVar(value="line")
        self.points = []
        self.counter_obj = None

        self.setup_ui()

    def setup_ui(self):
        self.sidebar = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar.pack(side="left", fill="y", padx=10, pady=10)

        self.title_label = ctk.CTkLabel(self.sidebar, text="BẢNG ĐIỀU KHIỂN", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_select = ctk.CTkButton(self.sidebar, text="📂 1. Chọn Video", command=self.select_video, height=40)
        self.btn_select.pack(padx=20, pady=10, fill="x")

        self.lbl_vid_name = ctk.CTkLabel(self.sidebar, text="Chưa chọn video...", text_color="gray")
        self.lbl_vid_name.pack(padx=20, pady=5)

        self.lbl_mode = ctk.CTkLabel(self.sidebar, text="Chọn Chế Độ Đếm:")
        self.lbl_mode.pack(padx=20, pady=(15, 0), anchor="w")

        self.radio_line = ctk.CTkRadioButton(self.sidebar, text="Đếm qua vạch (Line)", variable=self.draw_mode, value="line")
        self.radio_line.pack(padx=20, pady=5, anchor="w")

        self.radio_roi = ctk.CTkRadioButton(self.sidebar, text="Đếm trong vùng (ROI)", variable=self.draw_mode, value="roi")
        self.radio_roi.pack(padx=20, pady=5, anchor="w")

        self.btn_draw = ctk.CTkButton(self.sidebar, text="🖊️ 2. Vẽ Vạch / Vùng đếm", fg_color="#E67E22", hover_color="#D35400", command=self.open_drawing_window, height=40)
        self.btn_draw.pack(padx=20, pady=15, fill="x")

        self.btn_start = ctk.CTkButton(self.sidebar, text="▶️ 3. Bắt Đầu Đếm", fg_color="green", command=self.start_processing, height=40)
        self.btn_start.pack(padx=20, pady=10, fill="x")

        self.btn_stop = ctk.CTkButton(self.sidebar, text="⏹️ Dừng & Lưu KQ", fg_color="red", command=self.stop_processing, height=40)
        self.btn_stop.pack(padx=20, pady=10, fill="x")

        self.lbl_status = ctk.CTkLabel(self.sidebar, text="Sẵn sàng", text_color="yellow", font=ctk.CTkFont(weight="bold"))
        self.lbl_status.pack(pady=20)

        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        self.video_label = ctk.CTkLabel(self.video_frame, text="Màn hình hiển thị AI\n(Vui lòng làm theo các bước 1-2-3)", font=ctk.CTkFont(size=20), text_color="gray")
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

    def select_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            filename = path.split("/")[-1]
            self.lbl_vid_name.configure(text=f"Video: {filename}", text_color="white")
            self.points = []
            self.counter_obj = None

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mode = self.draw_mode.get()
            if mode == "line" and len(self.points) < 2:
                self.points.append([x, y])
            elif mode == "roi":
                self.points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN: # ẤN CHUỘT PHẢI ĐỂ HOÀN TÁC (UNDO)
            if len(self.points) > 0:
                self.points.pop()

    def open_drawing_window(self):
        if not self.video_path:
            self.lbl_status.configure(text="❌ Chọn video trước!", text_color="red")
            return

        cap = cv2.VideoCapture(self.video_path)
        ret, frame = cap.read()
        cap.release()

        if not ret: return

        frame = cv2.resize(frame, (1280, 720))
        self.points = []

        window_name = "VE VACH: CLICK TRAI de ve | CLICK PHAI de xoa diem | ENTER de luu"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        while True:
            temp_frame = frame.copy()
            mode = self.draw_mode.get()

            for p in self.points:
                cv2.circle(temp_frame, tuple(p), 5, (0, 0, 255), -1)

            if mode == "line" and len(self.points) == 2:
                cv2.line(temp_frame, tuple(self.points[0]), tuple(self.points[1]), (255, 0, 255), 2)
            elif mode == "roi" and len(self.points) > 1:
                pts = np.array(self.points, np.int32)
                cv2.polylines(temp_frame, [pts], isClosed=False, color=(255, 0, 255), thickness=2)
                if len(self.points) > 2:
                    cv2.line(temp_frame, tuple(self.points[-1]), tuple(self.points[0]), (255, 0, 255), 1)

            cv2.imshow(window_name, temp_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 13 or key == ord('q'):
                break

        cv2.destroyAllWindows()

        mode = self.draw_mode.get()
        if mode == "line" and len(self.points) == 2:
            self.counter_obj = LineCounter(self.points)
            self.lbl_status.configure(text="✅ Đã lưu Line. Bấm Bắt đầu!", text_color="green")
        elif mode == "roi" and len(self.points) >= 3:
            self.counter_obj = PolygonCounter(self.points)
            self.lbl_status.configure(text="✅ Đã lưu ROI. Bấm Bắt đầu!", text_color="green")
        else:
            self.lbl_status.configure(text="❌ Vẽ chưa xong, hãy vẽ lại!", text_color="red")
            self.counter_obj = None

    def start_processing(self):
        if not self.video_path or not self.counter_obj:
            self.lbl_status.configure(text="❌ Chưa chọn video hoặc chưa vẽ!", text_color="red")
            return

        if self.cap: self.cap.release()

        # --- KHỞI TẠO CÔNG CỤ LƯU VIDEO ---
        os.makedirs("Video_Xuat", exist_ok=True)
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = f"Video_Xuat/Video_AI_{time_str}.mp4"

        self.cap = cv2.VideoCapture(self.video_path)
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        if fps == 0: fps = 30

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out_video = cv2.VideoWriter(out_path, fourcc, fps, (1280, 720))
        # ----------------------------------

        self.is_running = True
        self.lbl_status.configure(text="🚀 Hệ thống đang chạy...", text_color="green")
        self.update_frame()

    def stop_processing(self):
        self.is_running = False
        if self.out_video:
            self.out_video.release() # Đóng gói file video
        self.lbl_status.configure(text="✅ Đã dừng & Lưu KQ thành công!", text_color="yellow")

    def update_frame(self):
        if self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                frame = cv2.resize(frame, (1280, 720))

                processed_frame = self.tracker.process_frame(frame, self.counter_obj)

                # Ghi frame đã vẽ vào file video MP4
                self.out_video.write(processed_frame)

                cv2image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)

                f_w = self.video_frame.winfo_width() - 20
                f_h = self.video_frame.winfo_height() - 20
                if f_w > 0 and f_h > 0:
                    img.thumbnail((f_w, f_h), Image.Resampling.LANCZOS)

                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk, text="")

                self.after(10, self.update_frame)
            else:
                self.stop_processing()

if __name__ == "__main__":
    app = TrafficAIApp()
    app.mainloop()
