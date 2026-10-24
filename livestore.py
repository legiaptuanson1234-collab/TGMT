# livestore.py - Cơ sở dữ liệu giao thông thời gian thực (thread-safe)
# Ghi bởi thread AI, đọc bởi GUI + Chatbot MÀ KHÔNG KHÓA màn hình.
import os
import csv
import time
import threading
from collections import defaultdict
from datetime import datetime

from counting import _safe_dir

CLASS_NAMES = {0: 'Ô tô', 1: 'Xe máy', 2: 'Xe tải', 3: 'Xe bus', 4: 'Xe ba gác'}
ALL_CLASSES = [0, 1, 2, 3, 4]


class RunControl:
    """Điều khiển chạy/dừng/tạm dừng an toàn giữa 2 thread."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.stop = threading.Event()
        self.pause = threading.Event()

    def request_stop(self):
        self.stop.set()
        self.pause.clear()

    def toggle_pause(self):
        if self.pause.is_set():
            self.pause.clear()
        else:
            self.pause.set()

    @property
    def stopped(self):
        return self.stop.is_set()

    @property
    def paused(self):
        return self.pause.is_set()

    def wait_if_paused(self):
        while self.pause.is_set() and not self.stop.is_set():
            time.sleep(0.15)


def _status_of(active_now, per_minute):
    """Đánh giá mức mật độ (xe/phút trung bình 5 phút gần nhất + xe hiện có trong khung)."""
    if per_minute >= 25 or active_now >= 28:
        return "🔴 ÙN TẮC"
    if per_minute >= 15 or active_now >= 15:
        return "🟠 ĐÔNG"
    if per_minute >= 5 or active_now >= 5:
        return "🟡 VỪA PHẢI"
    return "🟢 VẮNG"


class TrafficStore:
    """Lưu mọi sự kiện xe (giữ lock khi ghi/đọc) + khung hình mới nhất.

    - events      : danh sách dict (mỗi xe đếm được 1 lần)
    - bins/phút   : thống kê theo phút video (mật độ, cao điểm)
    - detail_csv  : file "SuKien_*.csv" ghi LIÊN TỤC (real-time), không phải chờ
    - summary_csv : thống kê theo phút, xuất khi cần
    """

    def __init__(self):
        self.lock = threading.RLock()
        self.reset()

    def reset(self):
        with self.lock:
            self.events = []
            self.total = 0
            self.counts = {c: 0 for c in ALL_CLASSES}
            self.bins = defaultdict(lambda: {c: 0 for c in ALL_CLASSES})
            self.bin_total = defaultdict(int)
            self.speeds = []                    # các tốc độ (km/h) gần nhất
            self.active_now = 0                 # xe hiện có trong khung (mật độ tức thời)
            self.fps = 0.0
            self.video_sec = 0.0
            self.running = False
            self.finished = False
            self._frame = None                  # BGR 1280x720 mới nhất
            self._frame_ts = 0.0
            self.start_wall = None              # giờ thực tế bắt đầu
            self._csv_dir = None
            self.detail_csv = None
            self._csv_file = None
            self._csv_writer = None

    # ---------- ghi (từ thread AI) ----------
    def on_start(self, counter_name=""):
        with self.lock:
            self.running = True
            self.finished = False
            self.start_wall = datetime.now()
            self._csv_dir = _safe_dir("Bao_Cao")
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.detail_csv = os.path.join(self._csv_dir, 'SuKien_Live_%s.csv' % ts)
            self._csv_file = open(self.detail_csv, 'a', newline='', encoding='utf-8')
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["ThoiGianThucTe", "ThoiGianVideo(s)", "IDXe", "LoaiXe", "TocDo_kmh", "X", "Y"])

    def on_vehicle(self, track_id, cls_id, t_video_sec, speed_kmh, cx, cy):
        """Mỗi xe đếm được 1 lần -> ghi event + CSV live ngay lập tức."""
        with self.lock:
            ev = {
                "wall": datetime.now().strftime('%H:%M:%S'),
                "t": round(t_video_sec, 2),
                "track": int(track_id),
                "cls": int(cls_id),
                "speed": round(float(speed_kmh), 1),
                "x": int(cx), "y": int(cy),
            }
            self.events.append(ev)
            self.total += 1
            self.counts[ev["cls"]] = self.counts.get(ev["cls"], 0) + 1
            b = int(ev["t"] // 60)
            self.bins[b][ev["cls"]] = self.bins[b].get(ev["cls"], 0) + 1
            self.bin_total[b] += 1
            self.speeds.append(ev["speed"])
            if len(self.speeds) > 40:
                self.speeds.pop(0)
            if self._csv_writer is not None:
                self._csv_writer.writerow([ev["wall"], ev["t"], ev["track"],
                                           CLASS_NAMES.get(ev["cls"], "?"), ev["speed"], ev["x"], ev["y"]])
                self._csv_file.flush()  # LIEN TUC - doc file ngay duoc (khong phai cho ket thuc)

    def on_activity(self, t_video_sec, active_now, fps):
        with self.lock:
            self.active_now = int(active_now)
            self.fps = round(float(fps), 1)
            self.video_sec = float(t_video_sec)

    def on_frame(self, frame_bgr):
        with self.lock:
            self._frame = frame_bgr
            self._frame_ts = time.time()

    def on_finish(self, ok=True, error=""):
        with self.lock:
            self.running = False
            self.finished = True
            self.last_error = error

    # ---------- đọc (GUI / chatbot, không chặn thread AI) ----------
    def get_frame(self):
        with self.lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def stats(self):
        with self.lock:
            per_min = self._recent_per_minute()
            return {
                "total": self.total,
                "events": len(self.events),
                "counts": dict(self.counts),
                "active_now": self.active_now,
                "fps": self.fps,
                "video_sec": self.video_sec,
                "running": self.running,
                "finished": self.finished,
                "per_minute": per_min,
                "status": _status_of(self.active_now, per_min),
                "peak": self.peak(),
                "avg_speed": (round(sum(self.speeds) / len(self.speeds), 1) if self.speeds else 0.0),
                "detail_csv": self.detail_csv,
            }

    def _recent_per_minute(self):
        if not self.events:
            return 0.0
        last_min = int(self.video_sec // 60)
        window = [self.bin_total.get(m, 0) for m in range(max(0, last_min - 4), last_min + 1)]
        return round(sum(window) / len(window), 1)

    def peak(self):
        with self.lock:
            if not self.bin_total:
                return None
            m = max(self.bin_total, key=lambda k: self.bin_total[k])
            return {"minute": m, "total": self.bin_total[m],
                    "counts": dict(self.bins[m])}

    def query_range(self, start_sec=0, end_sec=None):
        """Trả về thống kê trong đoạn video [start_sec, end_sec). end_sec=None -> đến hiện tại."""
        with self.lock:
            if end_sec is None:
                end_sec = 10 ** 9
            sel = [e for e in self.events if start_sec <= e["t"] < end_sec]
            cnt = {c: 0 for c in ALL_CLASSES}
            for e in sel:
                cnt[e["cls"]] = cnt.get(e["cls"], 0) + 1
            mins_in = sorted({int(e["t"] // 60) for e in sel})
            per_min = round(len(sel) / len(mins_in), 1) if mins_in else 0.0
            return {
                "total": len(sel),
                "counts": cnt,
                "per_minute": per_min,
                "avg_speed": (round(sum(e["speed"] for e in sel) / len(sel), 1) if sel else 0.0),
                "status": _status_of(self.active_now, per_min),
                "start": start_sec, "end": end_sec,
            }

    def write_summary_csv(self):
        """Xuất CSV thống kê theo phút (gọi khi user yêu cầu / trước khi xem báo cáo)."""
        with self.lock:
            self._csv_dir = self._csv_dir or _safe_dir("Bao_Cao")
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(self._csv_dir, "ThongKe_Phut_%s.csv" % ts)
            with open(path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["PhutVideo", "TongXe", "OTo", "XeMay", "XeTai", "XeBus", "XeBaGac", "MatDo_xe_phut"])
                for m in sorted(self.bins.keys()):
                    c = self.bins[m]
                    w.writerow([m, self.bin_total[m],
                                c.get(0, 0), c.get(1, 0), c.get(2, 0), c.get(3, 0), c.get(4, 0),
                                self.bin_total[m]])
            return path

    def close_csv(self):
        with self.lock:
            if self._csv_writer is not None:
                try:
                    self._csv_writer.close()
                except Exception:
                    pass
                self._csv_writer = None
