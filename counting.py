import cv2
import numpy as np
import os
import csv
import tempfile
from datetime import datetime


def _safe_dir(name):
    """Thư mục lưu kết quả: dùng thư mục hiện tại nếu ghi được (local);
    nếu không (Streamlit Cloud, đĩa read-only) thì dùng thư mục tạm."""
    try:
        os.makedirs(name, exist_ok=True)
        return name
    except (PermissionError, OSError):
        alt = os.path.join(tempfile.gettempdir(), name)
        os.makedirs(alt, exist_ok=True)
        return alt

class BaseCounter:
    def __init__(self):
        self.counted_ids = set()
        self.total_vehicles = 0
        self.vehicle_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        self.class_names = {0: 'O To', 1: 'Xe May', 2: 'Xe Tai', 3: 'Xe Bus', 4: 'Xe Ba Gac'}
        # ==========================================
        # DÙNG ĐƯỜNG DẪN TƯƠNG ĐỐI (DÙNG ĐƯỢC TRÊN WEB)
        # ==========================================
        self.save_dir = _safe_dir('Bao_Cao')
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_file = os.path.join(self.save_dir, f'ThongKe_LuuLuong_{time_str}.csv')
        with open(self.csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Thoi Gian Ghi Nhan", "ID Xe", "Loai Phuong Tien"])

    def log_vehicle(self, track_id, cls_id):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        vehicle_name = self.class_names.get(cls_id, "Khong xac dinh")
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, track_id, vehicle_name])

    # ---- mặc định cho dạng "vạch": mọi xe đều được coi là active ----
    def is_active(self, cx, cy):
        return True
    @property
    def congestion_limit(self):
        return 28

# ===============================================
# 1. CHUYÊN GIA ĐẾM VÙNG (ROI POLYGON)
# ===============================================
class PolygonCounter(BaseCounter):
    def __init__(self, roi_points):
        super().__init__()
        self.roi_points = np.array(roi_points, dtype=np.int32)

    def check_and_count(self, cx, cy, track_id, cls_id):
        is_inside = cv2.pointPolygonTest(self.roi_points, (cx, cy), False)
        if is_inside >= 0 and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.total_vehicles += 1
            self.vehicle_counts[cls_id] += 1
            self.log_vehicle(track_id, cls_id)
            return True
        return False

    def is_active(self, cx, cy):
        return cv2.pointPolygonTest(self.roi_points, (cx, cy), False) >= 0

    @property
    def congestion_limit(self):
        return 10

    def draw_shape(self, frame, is_flash):
        color = (0, 255, 255) if is_flash else (0, 0, 255)
        cv2.polylines(frame, [self.roi_points], True, color, 5 if is_flash else 2)

# ===============================================
# 2. CHUYÊN GIA ĐẾM ĐƯỜNG KẺ (LINE CROSSING)
# ===============================================
class LineCounter(BaseCounter):
    def __init__(self, line_points):
        super().__init__()
        self.line_A = tuple(line_points[0])
        self.line_B = tuple(line_points[1])
        self.track_history = {}

    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def _intersect(self, A, B, C, D):
        return self._ccw(A, C, D) != self._ccw(B, C, D) and self._ccw(A, B, C) != self._ccw(A, B, D)

    def check_and_count(self, cx, cy, track_id, cls_id):
        current_pt = (cx, cy)
        is_counted_now = False
        if track_id in self.track_history and track_id not in self.counted_ids:
            prev_pt = self.track_history[track_id]
            if self._intersect(prev_pt, current_pt, self.line_A, self.line_B):
                self.counted_ids.add(track_id)
                self.total_vehicles += 1
                self.vehicle_counts[cls_id] += 1
                self.log_vehicle(track_id, cls_id)
                is_counted_now = True
        self.track_history[track_id] = current_pt
        return is_counted_now

    def draw_shape(self, frame, is_flash):
        color = (0, 255, 255) if is_flash else (255, 0, 255)
        cv2.line(frame, self.line_A, self.line_B, color, 5 if is_flash else 2)

# ===============================================
# 3. NHIỀU HÌNH CÙNG LÚC (nhiều VẠCH + nhiều VÙNG đa giác)
#    - lines:        danh sách (A, B) toạ độ 1280x720
#    - roi_polygons: danh sách vùng, mỗi vùng = list (x, y) toạ độ 1280x720
#    Đếm 1 xe MỘT LẦN khi: (vượt 1 vạch) HOẶC (nằm trong 1 vùng).
# ===============================================
class MultiCounter(BaseCounter):
    def __init__(self, lines=None, roi_polygons=None, roi_mask=None):
        super().__init__()
        self.lines = lines or []
        self.roi_polygons = [np.array(p, dtype=np.int32) for p in (roi_polygons or [])]
        self.roi_mask = roi_mask
        self._track_hist = {}

    def _crossed_any_line(self, prev, cur):
        for A, B in self.lines:
            if self._ccw(prev, A, B) != self._ccw(cur, A, B) and \
               self._ccw(prev, cur, A) != self._ccw(prev, cur, B):
                return True
        return False

    def _ccw(self, A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    def has_roi(self):
        return bool(self.roi_polygons) or (self.roi_mask is not None)

    def is_inside_roi(self, cx, cy):
        for poly in self.roi_polygons:
            if cv2.pointPolygonTest(poly, (cx, cy), False) >= 0:
                return True
        if self.roi_mask is not None:
            cx = int(np.clip(cx, 0, self.roi_mask.shape[1]-1))
            cy = int(np.clip(cy, 0, self.roi_mask.shape[0]-1))
            return bool(self.roi_mask[cy, cx])
        return False

    def check_and_count(self, cx, cy, track_id, cls_id):
        if track_id in self.counted_ids:
            return False
        do_count = False
        if self.lines:
            if track_id in self._track_hist:
                if self._crossed_any_line(self._track_hist[track_id], (cx, cy)):
                    do_count = True
            self._track_hist[track_id] = (cx, cy)
        if self.has_roi() and self.is_inside_roi(cx, cy):
            do_count = True
        if do_count:
            self.counted_ids.add(track_id)
            self.total_vehicles += 1
            self.vehicle_counts[cls_id] += 1
            self.log_vehicle(track_id, cls_id)
            return True
        return False

    def is_active(self, cx, cy):
        # cảnh báo ùn tắc chỉ tính xe nằm TRONG VÙNG (nếu có vẽ vùng);
        # chỉ vẽ vạch (không có vùng) thì mọi xe đều tính
        if self.has_roi():
            return self.is_inside_roi(cx, cy)
        return True

    @property
    def congestion_limit(self):
        return 10 if self.has_roi() else 28

    def draw_shape(self, frame, is_flash):
        # TẤT CẢ VẠCH
        for A, B in self.lines:
            color = (0, 255, 255) if is_flash else (255, 0, 255)
            cv2.line(frame, A, B, color, 4 if is_flash else 2)
        # TẤT CẢ VÙNG (FILL ĐỎ RÕ + VIỀN) - không "tàng hình" nữa
        for poly in self.roi_polygons:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [poly], (40, 80, 200))
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
            color = (0, 255, 255) if is_flash else (0, 0, 255)
            cv2.polylines(frame, [poly], True, color, 4 if is_flash else 2)
        if self.roi_mask is not None:
            m = (self.roi_mask.astype(np.uint8)) * 255
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            overlay = frame.copy()
            cv2.fillPoly(overlay, cnts, (40, 80, 200))
            cv2.addWeighted(overlay, 0.30, frame, 0.70, 0, frame)
            cv2.drawContours(frame, cnts, -1, (0, 255, 255), 2)

# ===============================================
# 4. TRẠM TRUNG CHUYỂN (DÀNH RIÊNG CHO WEB)
# ===============================================
def VehicleCounter(mode, points):
    """Hàm giúp app.py khởi tạo đúng Class tuỳ theo người dùng vẽ Line hay ROI."""
    if mode == "line":
        return LineCounter(points)
    else:
        return PolygonCounter(points)
