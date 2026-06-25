import cv2
import numpy as np
import os
import csv
from datetime import datetime

class BaseCounter:
    def __init__(self):
        self.counted_ids = set()
        self.total_vehicles = 0
        self.vehicle_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0} # 0-Oto, 1-Xemay, 2-XeTai, 3-XeBus, 4-XeBaGac
        self.class_names = {0: 'O To', 1: 'Xe May', 2: 'Xe Tai', 3: 'Xe Bus', 4: 'Xe Ba Gac'}
        
        # ==========================================
        # SỬA THÀNH ĐƯỜNG DẪN TƯƠNG ĐỐI (DÙNG ĐƯỢC TRÊN WEB)
        # ==========================================
        self.save_dir = 'Bao_Cao'
        os.makedirs(self.save_dir, exist_ok=True)
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
# 3. TRẠM TRUNG CHUYỂN (DÀNH RIÊNG CHO WEB)
# ===============================================
def VehicleCounter(mode, points):
    """
    Hàm này giúp app.py khởi tạo đúng Class tùy theo người dùng vẽ Line hay ROI trên Web.
    """
    if mode == "line":
        return LineCounter(points)
    else:
        return PolygonCounter(points)