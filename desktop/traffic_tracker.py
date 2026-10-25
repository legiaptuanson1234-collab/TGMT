# traffic_tracker.py (v8) - Core YOLO+ByteTrack cho bản desktop
# [v8 mới] Tích hợp TrafficStore (livestore.py): ghi sự kiện xe + CSV LIVE
#          ngay khi đếm được -> chatbot desktop đọc real-time, không chờ video hết.
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
        # [v8] tích hợp chatbot + CSV live (phải là TrafficStore từ livestore.py)
        self.store = None
        self.run_start = None        # time.time() tại lúc BẮT ĐẦU chạy (đo video_sec)
        self.conf = 0.5
        self.proc_wh = (1280, 720)   # (640, 360) = chế độ nhẹ CPU

    def reset_counters(self):
        self.track_history.clear()
        self.track_time.clear()
        self.speed_history.clear()
        self.prev_time = time.time()
        self.fps_smooth = 30.0

    def process_frame(self, frame, counter_obj):
        current_time = time.time()
        is_flash_frame = False
        current_active_vehicles = 0
        t_video = (current_time - self.run_start) if (self.store is not None and self.run_start) else 0.0

        # [v8] resize theo hiệu năng (CPU nhẹ = 640x360, full = 1280x720) rồi qui toạ độ về 1280x720
        pw, ph = self.proc_wh
        proc = frame if (pw == 1280 and ph == 720) else cv2.resize(frame, (pw, ph))
        sx = 1280.0 / pw
        sy = 720.0 / ph

        # Chạy YOLO Tracking
        results = self.model.track(proc, persist=True, tracker="bytetrack.yaml", conf=self.conf, verbose=False)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                x1, y1 = int(box[0] * sx), int(box[1] * sy)
                x2, y2 = int(box[2] * sx), int(box[3] * sy)
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                # Kiểm tra xe đã qua vạch đếm chưa
                did_count = counter_obj.check_and_count(cx, cy, track_id, cls_id)
                if did_count:
                    is_flash_frame = True
                    # [v8] ghi sự kiện LIỀN (chatbot + CSV live đọc được ngay)
                    if self.store is not None:
                        self.store.on_vehicle(track_id, cls_id, t_video,
                                              self.speed_history.get(track_id, 0.0), cx, cy)

                # Đếm xe đang hoạt động (dùng cho cảnh báo ùn tắc)
                if hasattr(counter_obj, 'roi_points'):
                    if cv2.pointPolygonTest(counter_obj.roi_points, (cx, cy), False) >= 0:
                        current_active_vehicles += 1
                else:
                    current_active_vehicles += 1

                # Vẽ Box + Track ID (toạ độ đã qui về 1280x720 = đúng vị trí trên frame thật)
                color = (0, 255, 0) if cls_id == 4 else (255, 150, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "%s ID:%d" % (self.class_names[cls_id], track_id),
                            (x1, max(y1 - 10, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Vẽ quỹ đạo (vết dầu xe)
                track = self.track_history[track_id]
                track.append((cx, cy))
                if len(track) > 30:
                    track.pop(0)
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], False, (255, 255, 0), 2)

                # Tính tốc độ (EMA)
                times = self.track_time[track_id]
                times.append(current_time)
                if len(times) > 30:
                    times.pop(0)
                speed_kmh = 0
                if len(track) >= 15:
                    dx, dy = track[-1][0] - track[0][0], track[-1][1] - track[0][1]
                    dist_px = float(np.sqrt(dx * dx + dy * dy))
                    dist_m = dist_px * self.PIXEL_TO_METER
                    time_diff = times[-1] - times[0]
                    if time_diff > 0:
                        speed_kmh = (dist_m / time_diff) * 3.6

                if speed_kmh > 0:
                    if self.speed_history[track_id] > 0:
                        self.speed_history[track_id] = 0.8 * self.speed_history[track_id] + 0.2 * speed_kmh
                    else:
                        self.speed_history[track_id] = speed_kmh
                    cv2.putText(frame, "%d km/h" % int(self.speed_history[track_id]),
                                (x1, max(y1 - 25, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

        # Chuyên gia đếm xe vẽ vạch/vùng chớp
        counter_obj.draw_shape(frame, is_flash_frame)

        # Tính FPS mượt
        fps_current = 1 / (current_time - self.prev_time) if (current_time - self.prev_time) > 0 else 0
        self.prev_time = current_time
        self.fps_smooth = 0.9 * self.fps_smooth + 0.1 * fps_current

        # [v8] cập nhật mật độ tức thời cho chatbot (đo xe trong khung)
        if self.store is not None:
            self.store.on_activity(t_video, current_active_vehicles, self.fps_smooth)
            self.store.on_frame(frame)

        # Hiển thị bảng Overlay tĩnh trên video
        status_text, status_color = "MAT DO: VANG", (0, 255, 0)
        limit = 10 if hasattr(counter_obj, 'roi_points') else 28
        if current_active_vehicles > limit:
            status_text, status_color = "CANH BAO: UN TAC!", (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (260, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "TONG SO XE: %d" % counter_obj.total_vehicles, (20, 35),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, "- O To: %d" % counter_obj.vehicle_counts[0], (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Xe May: %d" % counter_obj.vehicle_counts[1], (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- Xe Tai/Bus: %d" % (counter_obj.vehicle_counts[2] + counter_obj.vehicle_counts[3]),
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "- XE BA GAC: %d" % counter_obj.vehicle_counts[4], (20, 140),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame, "FPS: %.1f" % self.fps_smooth, (20, 170),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 128, 0), 1)
        cv2.rectangle(frame, (280, 10), (510, 45), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (290, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

        return frame
