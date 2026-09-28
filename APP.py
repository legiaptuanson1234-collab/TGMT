import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_fast, run_ai_quality, DISPLAY_W, DISPLAY_H
from counting import VehicleCounter

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False

BUILD_TAG = "TGMT-v5 · vẽ trực tiếp trên ảnh (canvas fix nền trắng Cloud) + 1 nút chạy"

# Kích thước chuẩn để toạ độ vẽ == toạ độ đếm (1280x720)
W, H = DISPLAY_W, DISPLAY_H  # 1280 x 720


def _workfile(name):
    try:
        probe = os.path.join(os.getcwd(), ".__probe")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return os.path.join(os.getcwd(), name)
    except OSError:
        return os.path.join(tempfile.gettempdir(), name)


def _frame_stats(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()), float(gray.std())


def _find_content_frame(video_path, target_idx=50, max_scan=400):
    """Lấy khung NỀN (khung có nội dung thật) từ video. Đọc tuần tự (seek thất
    bại trên Cloud), fallback ffmpeg nếu OpenCV không mở được.
    Trả về (frame|None, thông báo)."""
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target = min(target_idx, total // 4 if total > 0 else target_idx)
        best, best_score, best_idx = None, -1.0, -1
        scanned = 0
        while scanned < max_scan:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            scanned += 1
            mean, std = _frame_stats(frame)
            if std > 20 and 30 < mean < 225:
                score = std - abs(mean - 120.0) / 5.0
                if scanned - 1 == target:
                    cap.release()
                    return frame, "cv2 OK · khung nền #%d (std=%.0f)" % (scanned - 1, std)
                if score > best_score:
                    best, best_score, best_idx = frame, score, scanned - 1
        cap.release()
        if best is not None:
            return best, "cv2 OK · khung nền chi tiết nhất (#%d)" % best_idx
        return None, "cv2 mở được file nhưng %d frame đầu đều trắng/đen" % scanned

    try:
        import imageio
        reader = imageio.v2.get_reader(video_path, "ffmpeg")
        best, best_score, best_idx, n = None, -1.0, -1, 0
        while n < max_scan:
            fr = reader.read()
            if fr is None:
                break
            n += 1
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            mean, std = _frame_stats(fr)
            if std > 20 and 30 < mean < 225:
                score = std - abs(mean - 120.0) / 5.0
                if n - 1 == min(target_idx, 49):
                    return fr, "ffmpeg OK · khung nền #%d" % (n - 1)
                if score > best_score:
                    best, best_score, best_idx = fr, score, n - 1
        if best is not None:
            return best, "ffmpeg OK · khung nền chi tiết nhất (#%d)" % best_idx
        return None, "cả OpenCV lẫn ffmpeg không đọc được video"
    except Exception as _e:
        return None, "không đọc được video: %s" % _e


def _bake_grid(rgb_img_w1280x720):
    """Vẽ lưới 100px + mốc toạ độ 200px LÊN chính ảnh nền (để bạn căn khi vẽ).
    Ảnh đã resize về 1280x720 -> toạ độ vẽ == toạ độ đếm."""
    img = rgb_img_w1280x720.convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img)
    for x in range(0, w + 1, 100):
        d.line([(x, 0), (x, h)], fill=(255, 140, 0), width=1)
    for y in range(0, h + 1, 100):
        d.line([(0, y), (w, y)], fill=(255, 140, 0), width=1)
    for x in range(0, w + 1, 200):
        for y in range(0, h + 1, 200):
            d.text((x + 3, y + 3), "%d,%d" % (x, y), fill=(0, 0, 0))
            d.text((x + 3, y + 3), "%d,%d" % (x, y), fill=(255, 255, 0))
    return img


def _extract_points(objects, drawing_mode):
    """Lấy toạ độ AN TOÀN từ canvas (tránh KeyError khi canvas còn nhiều
    hình khác loại - vd vạch Line cũ khi đã chuyển sang ROI).
    Canvas được hiển thị đúng 1280x720 nên toạ độ trả về chính là toạ độ đếm."""
    if drawing_mode == "line":
        for obj in objects:
            if "x1" in obj and "x2" in obj:
                left, top = int(obj.get("left", 0)), int(obj.get("top", 0))
                return [(left + int(obj["x1"]), top + int(obj["y1"])),
                        (left + int(obj["x2"]), top + int(obj["y2"]))]
        return []
    for obj in objects:
        pts = []
        for p in obj.get("path", []):
            if isinstance(p, (list, tuple)):
                if len(p) >= 3:
                    pts.append((int(p[1]), int(p[2])))
                elif len(p) == 2:
                    pts.append((int(p[0]), int(p[1])))
        if len(pts) >= 3:
            return pts
    return []


# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")
st.title("🚦 HỆ THỐNG ĐẾM XE & CẢNH BÁO GIAO THÔNG AI")
st.markdown("**Đồ án Kỹ thuật - Sinh viên: Lê Giáp Tuấn Sơn - UTT**")
st.caption("🔖 " + BUILD_TAG)

# --- 2. BẢNG ĐIỀU KHIỂN ---
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    uploaded_file = st.file_uploader("1. Tải Video Lên", type=['mp4', 'avi', 'mov'])
    mode = st.radio("2. Chế Độ Phân Tích", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    perf_mode = st.radio(
        "3. Tốc Độ",
        ["Nhanh (CPU - demo)", "Chất lượng (GPU)"],
        help="Máy chủ chạy CPU: chọn Nhanh (YOLO 640x360). Có GPU: chọn Chất lượng (1280x720).")
    def _btn_full_width(label):
        """Nút chạy ngang (tương thích Streamlit cũ 1.24 lẫn mới ≥1.53)."""
        try:
            return st.button(label, type="primary", width="stretch")
        except TypeError:  # Streamlit <1.40 (local)
            return st.button(label, type="primary", use_container_width=True)

    btn_run = _btn_full_width("🚀 KHỞI ĐỘNG AI")

# --- 3. XỬ LÝ VIDEO: thêm video là TỰ HIỆN KHUNG ĐƯỜNG để vẽ ---
if uploaded_file is not None:
    video_path = _workfile("video_tam.mp4")

    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # Lấy khung nền (khung có nội dung thật), resize về 1280x720 cho khớp toạ độ
        frame, bg_info = _find_content_frame(video_path)
        if frame is not None:
            frame = cv2.resize(frame, (W, H))
            st.session_state.bg_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.caption("🎞️ Đã cắt khung để vẽ: " + bg_info)
        else:
            st.session_state.bg_image = None
            st.error("⚠️ " + bg_info)
            st.info("📹 Xem video gốc bằng trình duyệt:")
            st.video(uploaded_file)

    # --- 4. VẼ TRỰC TIẾP TRÊN KHUNG ĐƯỜNG (1 mặt vẽ duy nhất) ---
    if "bg_image" in st.session_state and st.session_state.bg_image is not None:
        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"
        # Ảnh nền CÓ LƯỚI + MỐC TOẠ ĐỘ, đúng 1280x720 -> vẽ chính bằng đếm
        bg_pil = _bake_grid(st.session_state.bg_image)

        st.subheader("🖊️ Vẽ vạch / vùng ngay trên ảnh (sau đó bấm KHỞI ĐỘNG AI)")

        canvas_pts = []
        if _HAS_CANVAS:
            try:
                canvas_result = st_canvas(
                    fill_color="rgba(255, 165, 0, 0.25)",
                    stroke_width=3,
                    stroke_color="#00FF00",
                    background_image=bg_pil,
                    update_streamlit=True,
                    height=H,
                    width=W,
                    drawing_mode=drawing_mode,
                    key="draw_canvas",
                )
                canvas_pts = _extract_points(
                    canvas_result.json_data["objects"]
                    if canvas_result.json_data is not None else [],
                    drawing_mode)
            except Exception as _ce:
                st.warning("Canvas không khả dụng trên máy chủ (%s). Dùng toạ độ thủ công bên dưới." % _ce)
                canvas_pts = []
                st.image(bg_pil)
        else:
            st.image(bg_pil)
            st.caption("(Không có canvas - nhập toạ độ thủ công bên dưới)")

        st.caption("Mẹo: Line = kéo 1 nét ngang dải đường · ROI = click từng góc. "
                   "Nhìn mốc toạ độ (màu vàng) để định vị.")

        # Ô toạ độ THỦ CÔNG (dự phòng + chỉnh lại số cho chính xác)
        default = " ".join("%d,%d" % (x, y) for x, y in canvas_pts) if canvas_pts else ""
        manual = st.text_input(
            "🎯 Toạ độ (sẽ ưu tiên dùng số bạn gõ): Line `x1,y1 x2,y2` · ROI `x1,y1 x2,y2 ...`",
            value=default, key="manual_pts",
            placeholder="vd Line: 100,450 1100,450")

        # --- 5. 1 NÚT CHẠY ĐẾM ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")

            points, used_manual = [], False
            if manual.strip():
                for pair in manual.split():
                    try:
                        xs, ys = pair.split(",")
                        points.append((int(xs), int(ys)))
                        used_manual = True
                    except ValueError:
                        pass
            if not used_manual:
                points = canvas_pts

            need = 2 if drawing_mode == "line" else 3
            if len(points) < need:
                st.warning("⚠️ Chưa đủ điểm (cần ≥%d). Vẽ trên ảnh hoặc nhập toạ độ ở ô trên rồi bấm lại." % need)
            else:
                st.write("Điểm đếm: " + " · ".join("%d,%d" % p for p in points)
                         + "   *(nguồn: %s)*" % ("nhập tay" if used_manual else "vẽ trên ảnh"))
                counter_obj = VehicleCounter(mode=drawing_mode, points=points)
                runner = run_ai_quality if perf_mode.startswith("Chất lượng") else run_ai_fast
                try:
                    runner("best.pt", video_path, _workfile("output.mp4"), counter_obj, st.empty())
                    st.balloons()
                    st.success("🎉 Hoàn tất! Báo cáo CSV: `Bao_Cao/` · Video kết quả: `Video_Xuat/`")
                except Exception as e:
                    st.error("❌ Lỗi khi tính toán: %s" % e)
