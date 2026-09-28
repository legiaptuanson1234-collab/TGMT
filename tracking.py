import os
import warnings
import logging
import cv2
import time
import numpy as np
from ultralytics import YOLO
from collections import defaultdict

warnings.filterwarnings("ignore")
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['YOLO_VERBOSE'] = 'False'
logging.getLogger("ultralytics").setLevel(logging.ERROR)
from counting import _safe_dir

# Kich thoai hinh HIEN THI / DOI (toa do can cu).
DISPLAY_W, DISPLAY_H = 1280, 720


def _run(model, cap, frame, proc_w, proc_h, conf):
    """Chay YOLO + ByteTrack tren frame co phien quyep, tra ra (kq, sx, sy)
    de qui ve toa do hinh that (1280x720). sx,sy = he so phoi tu phien quyep."""
    proc = frame if (proc_w == DISPLAY_W and proc_h == DISPLAY_H) else cv2.resize(frame, (proc_w, proc_h))
    res = model.track(proc, persist=True, tracker="bytetrack.yaml", conf=conf, verbose=False)
    sx = DISPLAY_W / float(proc_w)
    sy = DISPLAY_H / float(proc_h)
    return res, sx, sy


def run_ai_system(model_path, video_in, video_out, counter_obj, stframe,
                  proc_w=640, proc_h=360, skip=1, conf=0.35, disp_every=2):
    """Chay he dem xe.

    proc_w/proc_h: kich thoai dung cho YOLO (nho = nhanh; CPU dung 480x270,
                  GPU/chat luong cao dung 1280x720).
    skip: xu ly 1/khoang N frame (2 = moi 2 frame -> nhanh gap 2 lan).
    disp_every: gui 1 anh ve trinh duyet / disp_every frame (giam = nhe hon
                kenh Streamlit tren Cloud).
    Toa do dem ve luon qui ve hinh that 1280x720 (nong voi hinh nen va diem ve).
    """
    model = YOLO(model_path)
    class_names = ['O To', 'Xe May', 'Xe Tai', 'Xe Bus', 'Xe Ba Gac']

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        stframe.error("Không mở được video trên máy chủ: %s" % video_in)
        return
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    if fps_video == 0:
        fps_video = 30
    out = cv2.VideoWriter(video_out, cv2.VideoWriter_fourcc(*'mp4v'), fps_video, (DISPLAY_W, DISPLAY_H))
    save_folder = _safe_dir("Anh_Bang_Chung")

    prev_time = time.time()
    last_save_time = 0
    track_history = defaultdict(list)
    track_time = defaultdict(list)
    speed_history = defaultdict(float)
    PIXEL_TO_METER = 0.065
    fps_smooth = 30.0
    frame_count = 0

    print("[*] Chế độ hiệu năng: YOLO %dx%d, skip=%d, conf=%.2f" % (proc_w, proc_h, skip, conf))

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame is None:
            break
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        frame_count += 1
        if skip > 1 and frame_count % skip != 0:
            out.write(frame)  # van luon ghi, chi bat YOLO
            continue

        is_flash_frame = False
        current_active_vehicles = 0
        current_time = time.time()

        results, sx, sy = _run(model, cap, frame, proc_w, proc_h, conf)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, track_id, cls_id in zip(boxes, track_ids, classes):
                # qui ve toa do hinh that (1280x720)
                bx1, by1 = int(box[0] * sx), int(box[1] * sy)
                bx2, by2 = int(box[2] * sx), int(box[3] * sy)
                cx, cy = (bx1 + bx2) // 2, (by1 + by2) // 2

                if counter_obj.check_and_count(cx, cy, track_id, int(cls_id)):
                    is_flash_frame = True

                if counter_obj.is_active(cx, cy):
                    current_active_vehicles += 1

                color = (0, 255, 0) if int(cls_id) == 4 else (255, 150, 0)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, "%s ID:%d" % (class_names[int(cls_id)], track_id),
                            (bx1, max(by1 - 10, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                track = track_history[track_id]
                track.append((cx, cy))
                if len(track) > 30:
                    track.pop(0)
                pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, (255, 255, 0), 2)

                # toc do
                t_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                times = track_time[track_id]
                times.append(t_sec)
                if len(times) > 30:
                    times.pop(0)
                speed_kmh = 0.0
                if len(track) >= 15:
                    dx, dy = track[-1][0] - track[0][0], track[-1][1] - track[0][1]
                    dist_m = float(np.sqrt(dx * dx + dy * dy)) * PIXEL_TO_METER
                    dt = times[-1] - times[0]
                    if dt > 0:
                        speed_kmh = (dist_m / dt) * 3.6
                if speed_kmh > 0:
                    if speed_history[track_id] > 0:
                        speed_history[track_id] = 0.8 * speed_history[track_id] + 0.2 * speed_kmh
                    else:
                        speed_history[track_id] = speed_kmh
                    cv2.putText(frame, "%d km/h" % int(speed_history[track_id]),
                                (bx1, max(by1 - 25, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 0, 255), 2, cv2.LINE_AA)

        # Vạch / Vung (diem da o toa do hinh that)
        counter_obj.draw_shape(frame, is_flash_frame)

        # FPS + can bao
        fps_current = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time
        fps_smooth = 0.9 * fps_smooth + 0.1 * fps_current

        status_text, status_color = "MAT DO: VANG", (0, 255, 0)
        limit = getattr(counter_obj, 'congestion_limit', 28)
        if current_active_vehicles > limit:
            status_text, status_color = "CANH BAO: UN TAC!", (0, 0, 255)
            if current_time - last_save_time > 5:
                cv2.imwrite(os.path.join(save_folder, "UnTac_%d.jpg" % int(current_time)), frame)
                last_save_time = current_time

        # bang thong ke
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
        cv2.putText(frame, "FPS: %.1f" % fps_smooth, (20, 170),
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 128, 0), 1)
        cv2.rectangle(frame, (280, 10), (510, 45), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (290, 32), cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 1)

        out.write(frame)
        if frame_count % max(1, disp_every) == 0:
            disp = cv2.resize(frame, (848, 477))
            stframe.image(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    out.release()


def run_ai_quality(model_path, video_in, video_out, counter_obj, stframe):
    """Chat luong cao (GPU): YOLO 1280x720, moi frame."""
    return run_ai_system(model_path, video_in, video_out, counter_obj, stframe,
                        proc_w=1280, proc_h=720, skip=1, conf=0.45, disp_every=2)


def run_ai_fast(model_path, video_in, video_out, counter_obj, stframe):
    """Hieu nang (CPU): YOLO 640x360, moi frame, conf thap de bat du xe nho."""
    return run_ai_system(model_path, video_in, video_out, counter_obj, stframe,
                        proc_w=640, proc_h=360, skip=1, conf=0.35, disp_every=2)


def run_ai_turbo(model_path, video_in, video_out, counter_obj, stframe):
    """Sieu nhanh (CPU demo tren Cloud): YOLO 480x270, bat moi 2 frame,
    gui anh ve trinh duyet moi 4 frame -> FPS thuc dung nhat (nhanh ~3-4 lan
    so voi run_ai_fast). Dan: chi dung de DEMO tren may chu, don xe rat
    nhanh co the bi bot (track gap)."""
    return run_ai_system(model_path, video_in, video_out, counter_obj, stframe,
                        proc_w=480, proc_h=270, skip=2, conf=0.30, disp_every=4)
