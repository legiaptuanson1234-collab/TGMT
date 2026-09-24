import cv2
import time
import numpy as np
from collections import defaultdict

class TrafficTracker:
    def __init__(self, model):
        self.model = model
        self.class_names = ['O To', 'Xe May', 'Xe Tai', 'Xe Bus', 'Xe Ba Gac']
        self.track_history = defaultdict(lambda: [])
        self.track_time = defaultdict(lambda: [])
        self.speed_history = defaultdict(lambda: 0.0)
        self.PIXEL_TO_METER = 0.065
        self.prev_time = time.time()
        self.fps_smooth = 30.0

    def process_frame(self, frame, counter_obj):
        current_time = time.time()
        is_flash_frame = False
        current_active_vehicles = 0

        # Chạy YOLO Tracking
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.5, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Kiểm tra xe có qua vạch đếm chưa
                if counter_obj.check_and_count(cx, cy, track_id, cls_id):
                    is_flash_frame = True

                # Tính số lượng xe đang hoạt động (dùng cho cảnh báo ùn tắc)
                if hasattr(counter_obj, 'roi_points'):
                    if cv2.pointPolygonTest(counter_obj.roi_points, (cx, cy), False) >= 0:
                        current_active_vehicles += 1
                else:
                    current_active_vehicles += 1

                # Vẽ Box và Track ID
                color = (0, 255, 0) if cls_id == 4 else (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{self.class_names[cls_id]} ID:{track_id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Vẽ quỹ đạo (vệt đuôi xe)
                track = self.track_history[track_id]
                track.append((int(cx), int(cy)))
                if len(track) > 30: track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(255, 255, 0), thickness=2)

                # Tính Tốc Độ
                current_time_sec = current_time
                times = self.track_time[track_id]
                times.append(current_time_sec)
                if len(times) > 30: times.pop(0)
                speed_kmh = 0
                if len(track) >= 15:
                    dx, dy = track[-1][0] - track[0][0], track[-1][1] - track[0][1]
                    dist_px = np.sqrt(dx**2 + dy**2)
                    dist_m = dist_px * self.PIXEL_TO_METER
                    time_diff = times[-1] - times[0]
                    if time_diff > 0: speed_kmh = (dist_m / time_diff) * 3.6

                if speed_kmh > 0:
                    self.speed_history[track_id] = 0.8 * self.speed_history[track_id] + 0.2 * speed_kmh if self.speed_history[track_id] > 0 else speed_kmh
                    cv2.putText(frame, f"{int(self.speed_history[track_id])} km/h", (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        # Chuyên gia đếm xe vẽ vạch chớp
        counter_obj.draw_shape(frame, is_flash_frame)

        # Tính FPS mượt
        fps_current = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * fps_current

        # Hiển thị bảng Overlay tĩnh trên video
        status_text, status_color = "MAT DO: VANG", (0, 255, 0)
        limit = 10 if hasattr(counter_obj, 'roi_points') else 28
        if current_active_vehicles > limit:
            status_text, status_color = "CANH BAO: UN TAC!", (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"TONG SO XE: {counter_obj.total_vehicles}", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"- O To: {counter_obj.vehicle_counts[0]}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"- Xe May: {counter_obj.vehicle_counts[1]}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"- Xe Tai/Bus: {counter_obj.vehicle_counts[2] + counter_obj.vehicle_counts[3]}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"- XE BA GAC: {counter_obj.vehicle_counts[4]}", (20, 140), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, f"FPS: {self.fps_smooth:.1f}", (20, 170), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 128, 0), 1)
        cv2.rectangle(frame, (280, 10), (510, 45), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (290, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

        return frame