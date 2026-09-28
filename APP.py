import streamlit as st
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_system
from counting import VehicleCounter

import os
import tempfile

BUILD_TAG = "TGMT-v3 · cloud-fix (video-diagnostic + keyerror-fix)"


def _workfile(name):
    """Đường dẫn ghi file an toàn: dùng thư mục hiện tại nếu ghi được (local);
    nếu không (Streamlit Cloud, đĩa read-only) thì dùng thư mục tạm."""
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
    """Lấy khung làm NỀN cho canvas: đọc TUẦN TỰ từ frame 0 (lệnh seek
    cap.set(POS_FRAMES) thất bại im lặng trên Cloud), ưu tiên khung có
    NỘI DUNG THẬT (tránh khung trắng/đen mở đầu video).

    Nếu OpenCV không mở được file -> fallback bằng imageio-ffmpeg
    (bộ giải mã độc lập, hỗ trợ H.264).

    Trả về (frame|None, thông_báo_chẩn_đoán).
    """
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
            # "có nội dung thật": đủ chi tiết + không gần trắng/đen
            if std > 20 and 30 < mean < 225:
                score = std - abs(mean - 120.0) / 5.0
                if scanned - 1 == target:
                    cap.release()
                    return frame, ("cv2 OK · nền = frame #{}".format(scanned - 1)
                                   + " (std=%.0f) · video có %s frame" % (std, total or "?"))
                if score > best_score:
                    best, best_score, best_idx = frame, score, scanned - 1
        cap.release()
        if best is not None:
            return best, ("cv2 OK · nền = frame #%d (khung chi tiết nhất trong %d frame, std≈%.0f)"
                         % (best_idx, scanned, best_score))
        return None, ("cv2 mở được file nhưng %d frame đầu đều trắng/đen (video %s frame) "
                      "- video có thể toàn khung trắng" % (scanned, total or "?"))

    # ---- Fallback: imageio + ffmpeg (bộ giải mã độc lập với OpenCV) ----
    try:
        import imageio
        reader = imageio.v2.get_reader(video_path, "ffmpeg")
        best, best_score, best_idx, n = None, -1.0, -1, 0
        while n < max_scan:
            fr = reader.read()
            if fr is None or (hasattr(fr, "size") and fr.size == 0):
                break
            n += 1
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)  # trả về BGR, khớp OpenCV
            mean, std = _frame_stats(fr)
            if std > 20 and 30 < mean < 225:
                score = std - abs(mean - 120.0) / 5.0
                if n - 1 == min(target_idx, 49):
                    return fr, "cv2 KHÔNG mở được video -> đã dùng bộ giải mã ffmpeg (nền = frame #%d)" % (n - 1)
                if score > best_score:
                    best, best_score, best_idx = fr, score, n - 1
        if best is not None:
            return best, "cv2 KHÔNG mở được video -> đã dùng bộ giải mã ffmpeg (nền = frame #%d, chi tiết nhất)" % best_idx
        return None, ("cả OpenCV lẫn ffmpeg không đọc được video. File đã lưu: %s (%s KB)"
                      % (video_path, os.path.getsize(video_path) // 1024 if os.path.exists(video_path) else "?"))
    except Exception as _e:
        return None, ("cv2 KHÔNG MỞ được video (%s) và fallback ffmpeg lỗi: %s. Thử upload video MP4 (H.264) khác."
                      % (video_path, _e))


def _extract_points(objects, drawing_mode):
    """Lấy tọa độ từ st_canvas AN TOÀN (tránh KeyError khi canvas còn giữ
    nhiều hình khác loại - vd vạch Line cũ khi đã chuyển sang ROI)."""
    if drawing_mode == "line":
        for obj in objects:
            if "x1" in obj and "x2" in obj:
                left, top = int(obj.get("left", 0)), int(obj.get("top", 0))
                return [(left + int(obj["x1"]), top + int(obj["y1"])),
                        (left + int(obj["x2"]), top + int(obj["y2"]))]
        return []
    # polygon / free
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


# --- 1. CẤU HÌNH TRANG WEB ---
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
    btn_run = st.button("🚀 KHỞI ĐỘNG AI", type="primary", use_container_width=True)

# --- 3. XỬ LÝ VIDEO & BẢNG VẼ CANVAS (SIÊU MƯỢT) ---
if uploaded_file is not None:
    video_path = _workfile("video_tam.mp4")

    # CHỈ XỬ LÝ FILE KHI LÀ VIDEO MỚI (CHỐNG GIẬT LAG KHI CLICK CHUỘT)
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name

        # 1. Lưu video xuống đĩa 1 lần duy nhất
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())

        # 2. Lấy ảnh NỀN: đọc tuần tự, tự tìm khung có nội dung thật
        frame, bg_info = _find_content_frame(video_path)

        # 3. Chuyển hệ màu và lưu thẳng vào RAM
        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.bg_image = Image.fromarray(frame_rgb)
            st.caption("🎞️ " + bg_info)
        else:
            st.session_state.bg_image = None
            st.error("⚠️ " + bg_info)
            st.info("📹 Xem video gốc bằng trình duyệt (pro trình duyệt tự giải mã H.264):")
            st.video(uploaded_file)

    # --- 4. HIỂN THỊ CANVAS TỪ RAM ---
    if "bg_image" in st.session_state and st.session_state.bg_image is not None:
        image_pil = st.session_state.bg_image

        st.subheader("Bước 1: Vẽ Vạch/Vùng Cảnh Báo")
        st.markdown(f"**Chế độ hiện tại:** {mode}. Hãy dùng chuột click và vẽ trực tiếp lên ảnh dưới đây.")

        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"

        # Vẽ mượt mà, không bị load lại video
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=3,
            stroke_color="#00FF00",
            background_image=image_pil,
            update_streamlit=True,
            height=image_pil.height,
            width=image_pil.width,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        # --- 5. KÍCH HOẠT AI ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")

            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                objects = canvas_result.json_data["objects"]
                st.success("[✔] Đã nhận được tọa độ. Đang khởi động AI...")

                stframe = st.empty()

                # Lấy tọa độ AN TOÀN (không KeyError khi canvas còn giữ
                # hình khác loại - ví dụ vạch Line cũ khi đã chuyển sang ROI)
                points = _extract_points(objects, drawing_mode)
                need = 2 if drawing_mode == "line" else 3
                if len(points) < need:
                    st.warning("⚠️ Chưa đủ điểm để đếm (cần ≥%d). Xóa hình cũ (chuột phải) rồi vẽ lại." % need)
                else:
                    counter_obj = VehicleCounter(mode=drawing_mode, points=points)
                    try:
                        # Video đã có sẵn trên đĩa từ bước trên, gọi thẳng ra chạy
                        run_ai_system("best.pt", video_path, _workfile("output.mp4"), counter_obj, stframe)
                        st.balloons()
                        st.success("🎉 Luồng phân tích giao thông đã hoàn tất!")
                    except Exception as e:
                        st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán: {e}")
            else:
                st.error("❌ BẠN CHƯA VẼ VẠCH HAY VÙNG ROI! Vui lòng dùng chuột vẽ lên hình trước khi bấm Khởi Động AI.")
