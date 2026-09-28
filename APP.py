import streamlit as st
import cv2
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_fast, run_ai_quality
from counting import VehicleCounter

import os
import tempfile

BUILD_TAG = "TGMT-v4 · cloud-fix (nền tham khảo + toạ độ thủ công + hiệu năng)"


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
    """Lấy khung làm NỀN: đọc TUẦN TỰ từ frame 0 (seek cap.set thất bại trên
    Cloud), ưu tiên khung có NỘI DUNG THẬT (tránh khung trắng/đen mở đầu).
    Fallback bằng imageio/ffmpeg nếu OpenCV không mở được file.
    Trả về (frame|None, thông_báo_chẩn_đoán)."""
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
                    return frame, ("cv2 OK · nền = frame #{} (std=%.0f) · video có %s frame"
                                   % (scanned - 1, std, total or "?"))
                if score > best_score:
                    best, best_score, best_idx = frame, score, scanned - 1
        cap.release()
        if best is not None:
            return best, ("cv2 OK · nền = frame #%d (chi tiết nhất trong %d frame)"
                         % (best_idx, scanned))
        return None, ("cv2 mở được file nhưng %d frame đầu đều trắng/đen" % scanned)

    # ---- Fallback: imageio + ffmpeg (bộ giải mã độc lập với OpenCV) ----
    try:
        import imageio
        reader = imageio.v2.get_reader(video_path, "ffmpeg")
        best, best_score, best_idx, n = None, -1.0, -1, 0
        target = min(target_idx, 49)
        while n < max_scan:
            fr = reader.read()
            if fr is None:
                break
            n += 1
            fr = cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
            mean, std = _frame_stats(fr)
            if std > 20 and 30 < mean < 225:
                score = std - abs(mean - 120.0) / 5.0
                if n - 1 == target:
                    return fr, "cv2 KHÔNG mở được video -> đã dùng ffmpeg · nền = frame #%d" % (n - 1)
                if score > best_score:
                    best, best_score, best_idx = fr, score, n - 1
        if best is not None:
            return best, "cv2 KHÔNG mở được video -> đã dùng ffmpeg · nền = frame #%d" % best_idx
        return None, ("cả OpenCV lẫn ffmpeg không đọc được video (%s)" % video_path)
    except Exception as _e:
        return None, ("cv2 KHÔNG MỞ được video (%s) và fallback ffmpeg lỗi: %s"
                      % (video_path, _e))


def _extract_points(objects, drawing_mode):
    """Lấy toạ độ từ st_canvas AN TOÀN (tránh KeyError khi canvas còn nhiều
    hình khác loại - vd vạch Line cũ khi đã chuyển sang ROI)."""
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


def _with_grid(img_pil):
    """Vẽ lưới + mốc 100px để căn toạ độ (toạ độ đếm = toạ độ ảnh gốc
    1280x720, mốc lưới giúp ước lượng khi nhập thủ công)."""
    img = img_pil.convert("RGB")
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    for x in range(0, w + 1, 100):
        d.line([(x, 0), (x, h)], fill=(255, 80, 0, 120), width=1)
    for y in range(0, h + 1, 100):
        d.line([(0, y), (w, y)], fill=(255, 80, 0, 120), width=1)
    for x in range(0, w + 1, 200):
        for y in range(0, h + 1, 200):
            d.text((x + 3, y + 3), f"{x},{y}", fill=(255, 255, 0))
    return img


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
    perf_mode = st.radio(
        "3. Tốc Độ",
        ["Nhanh (CPU - khuyến nghị demo)", "Chất lượng (GPU)"],
        help="Máy chủ chạy CPU: chọn Nhanh (YOLO 640x360, ~gấp 3-4 lần). "
             "Máy có GPU: chọn Chất lượng (YOLO 1280x720).")
    btn_run = st.button("🚀 KHỞI ĐỘNG AI", type="primary", use_container_width=True)

# --- 3. XỬ LÝ VIDEO & BẢNG VẼ CANVAS ---
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
            st.info("📹 Xem video gốc bằng trình duyệt (tự giải mã H.264):")
            st.video(uploaded_file)

    # --- 4. HIỂN THỊ CANVAS TỪ RAM ---
    if "bg_image" in st.session_state and st.session_state.bg_image is not None:
        image_pil = st.session_state.bg_image

        st.subheader("Bước 1: Vẽ Vạch/Vùng Cảnh Báo")
        st.markdown(f"**Chế độ hiện tại:** {mode}. Vẽ trên canvas, hoặc **nhập toạ độ thủ công** bên dưới.")

        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"

        # (a) ẢNH NỀN THAM KHẢO (KHÔNG BỊ TRẮNG KHI CHUỘT CHẠY) - có lưới căn toạ độ
        st.markdown("**📷 Ảnh nền (tham khảo để căn vẽ):**")
        st.image(_with_grid(image_pil), use_container_width=True)
        st.caption("Toạ độ hệ 1280×720 (góc trên-trái = 0,0). Lưới mỗi 100px, mốc mỗi 200px.")

        # (b) Canvas để vẽ nhanh (chuột) - nếu nền trắng vẫn dùng được (c)
        st.markdown("**✏️ Vẽ nhanh bằng chuột (Line: kéo 1 nét · ROI: click từng góc):**")
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

        # (c) Ô NHẬP TOẠ ĐỘ THỦ CÔNG - LUÔN DÙNG ĐƯỢC, CHUẨN CHỈNH NHẤT
        canvas_pts = _extract_points(
            canvas_result.json_data["objects"]
            if canvas_result.json_data is not None else [],
            drawing_mode)
        default = (" ".join(f"{x},{y}" for x, y in canvas_pts)) if canvas_pts else ""
        manual = st.text_input(
            "🎯 Toạ độ chính xác (ưu tiên khi canvas bị trắng): Line: `x1,y1 x2,y2` · "
            "ROI: `x1,y1 x2,y2 x3,y3...` (toạ độ ảnh 1280×720, mốc lưới giúp căn)",
            value=default, key="manual_pts",
            placeholder="vd Line: 200,400 900,400 · vd ROI: 200,200 800,200 800,600 200,600")

        # --- 5. KÍCH HOẠT AI ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")

            # Ưu tiên toạ độ THỦ CÔNG (người dùng gõ = chuẩn nhất)
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
                st.warning(f"⚠️ Chưa đủ điểm (cần ≥{need}). Vẽ lại hoặc nhập toạ độ ở ô bên trên.")
            else:
                st.write(f"Điểm đang dùng: `{[f'{x},{y}' for x, y in points]}` "
                         f"({len(points)}/{'∞'} · nguồn: {'nhập tay' if used_manual else 'canvas'})")
                counter_obj = VehicleCounter(mode=drawing_mode, points=points)

                runner = run_ai_quality if perf_mode.startswith("Chất lượng") else run_ai_fast
                try:
                    runner("best.pt", video_path, _workfile("output.mp4"), counter_obj, st.empty())
                    st.balloons()
                    st.success("🎉 Luồng phân tích giao thông đã hoàn tất! "
                               "Báo cáo CSV lưu trong `Bao_Cao/`, video kết quả trong `Video_Xuat/`.")
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán: {e}")
