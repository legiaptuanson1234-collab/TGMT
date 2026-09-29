import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import tempfile

# --- NHẬP CÁC MODULE AI CỦA BẠN (kèm "tự chữa" build cũ) ---
import importlib, sys
for _m in ("counting", "tracking"):
    sys.modules.pop(_m, None)
import counting, tracking
from tracking import run_ai_fast, run_ai_quality, run_ai_turbo, DISPLAY_W, DISPLAY_H
from counting import MultiCounter

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False

BUILD_TAG = "TGMT-v7 · 1 hình + Line/ROI (giữ hình khi đổi) · đọc pixel (không lệch) · hết crash"

W, H = DISPLAY_W, DISPLAY_H       # 1280x720 (hệ toạ độ đếm)
DX, DY = 960, 540                # canvas hiển thị (fit màn, thấy đủ đường)


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


def _bake_grid(rgb_img):
    img = rgb_img.convert("RGB")
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


def _img(result):
    """Lấy ảnh canvas (RGBA) về DXxDY. None nếu CHƯA VẼ GÌ (image_data=None)
    hay lỗi — để app không crash khi load lại / chưa vẽ (sai IndexError cũ)."""
    if result is None:
        return None
    data = getattr(result, "image_data", None)
    if data is None:
        return None
    try:
        arr = np.asarray(data)
    except Exception:
        return None
    if arr is None or arr.size == 0 or arr.ndim < 2:   # None/0-dim -> an toàn
        return None
    if arr.ndim == 4:
        arr = arr[:, :, :, 0]
    if arr.ndim < 3:
        return None
    if arr.shape[0] != DY or arr.shape[1] != DX:
        arr = cv2.resize(arr, (DX, DY))
    return arr


def _diff_masks(result, ref_rgb):
    """So 'ảnh canvas đã vẽ' với 'ảnh nền' (ref) -> chỉ lấy PIXEL BẠN VẼ.
    Trả về (magenta_mask, fill_mask) DXxDY (uint8 0/255).
    magenta_mask = nét TÍM (VẠCH) · fill_mask = các pixel khác bạn vẽ (VÙNG)."""
    z = np.zeros((DY, DX), np.uint8)
    canvas = _img(result)
    if canvas is None:
        return z, z.copy()
    ref = np.array(ref_rgb.resize((DX, DY)).convert("RGB")).astype(np.int16)
    c = canvas[:, :, :3].astype(np.int16)
    changed = (np.abs(c - ref).max(axis=2) > 40)
    R = canvas[:, :, 0].astype(np.int16)
    G = canvas[:, :, 1].astype(np.int16)
    B = canvas[:, :, 2].astype(np.int16)
    is_mag = (R > 150) & (B > 150) & (G < 120)
    magenta_mask = (changed & is_mag).astype(np.uint8) * 255
    fill_mask = (changed & ~is_mag).astype(np.uint8) * 255
    return magenta_mask, fill_mask


def _lines_from_mask(mag_mask, W, H):
    """mask nét TÍM (DXxDY) -> [(A,B), ...] toạ độ 1280x720 (N vạch)."""
    out = []
    cnts, _ = cv2.findContours(mag_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sx, sy = W / float(DX), H / float(DY)
    for c in cnts:
        if c.size < 12:
            continue
        pts = c[:, 0, :].astype(np.float64)
        mean = pts.mean(axis=0)
        _, evecs = np.linalg.eigh(np.cov(pts.T))
        v = evecs[:, -1]
        t = (pts - mean) @ v
        i1, i2 = int(np.argmin(t)), int(np.argmax(t))
        p1 = (int(round(pts[i1][0] * sx)), int(round(pts[i1][1] * sy)))
        p2 = (int(round(pts[i2][0] * sx)), int(round(pts[i2][1] * sy)))
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) >= 20:
            out.append((p1, p2))
    return out


def _roi_mask_from(fill_mask, W, H, min_area=300):
    """mask pixel VẼ (vùng, DXxDY) -> mask bool HxW (1280x720) NHIỀU vùng
    (điền đầy bên trong, loại blob nhỏ do lưới/nhiễu)."""
    m = cv2.morphologyEx(fill_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(filled, [c], -1, 255, -1)
    filled = cv2.resize(filled, (W, H), interpolation=cv2.INTER_NEAREST)
    return filled > 0


def _bg_uri(bg_pil):
    import io, base64
    buf = io.BytesIO()
    bg_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")
st.title("🚦 HỆ THỐNG ĐẾM XE & CẢNH BÁO GIAO THÔNG AI")
st.markdown("**Đồ án Kỹ thuật - Sinh viên: Lê Giáp Tuấn Sơn - UTT**")
st.caption("🔖 " + BUILD_TAG)


def _btn_full_width(label):
    try:
        return st.button(label, type="primary", width="stretch")
    except TypeError:
        return st.button(label, type="primary", use_container_width=True)


# --- 2. BẢNG ĐIỀU KHIỂN (GIỮ PHONG CÁCH CŨ) ---
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    st.markdown("**1. Tải Video Lên**")
    uploaded_file = st.file_uploader("video", type=['mp4', 'avi', 'mov'])
    mode = st.radio("2. Chế Độ Phân Tích", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    perf_mode = st.radio(
        "3. Tốc Độ",
        ["Siêu nhanh (CPU demo)", "Nhanh (CPU)", "Chất lượng (GPU)"],
        help="Cloud chạy CPU. 'Siêu nhanh': YOLO 480x270 + bỏ frame (demo). "
             "'Nhanh': 640x360. 'Chất lượng': 1280x720 (có GPU).")
    btn_run = _btn_full_width("🚀 KHỞI ĐỘNG AI")

# --- 3. XỬ LÝ VIDEO: thêm video là TỰ HIỆN KHUNG ĐƯỜNG để vẽ ---
if uploaded_file is not None:
    video_path = _workfile("video_tam.mp4")
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())
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

    # --- 4. 1 HÌNH TO ĐỂ VẼ (giữ phong cách cũ) ---
    if "bg_image" in st.session_state and st.session_state.bg_image is not None:
        bg_pil_grid = _bake_grid(st.session_state.bg_image)

        _ck = st.session_state.get("_canvas_seq", 0)
        if st.button("🗑️ Xoá hết & Vẽ lại", key="reset_draw"):
            st.session_state["_canvas_seq"] = _ck + 1
            st.rerun()

        st.subheader("🖊️ Vẽ vạch / vùng trên ảnh (sau đó bấm KHỞI ĐỘNG AI)")

        # 1 CANVAS DUY NHẤT. Key CHỈ đổi khi "Xoá hết & Vẽ lại" (reset),
        # KHÔNG đổi khi chọn Line/ROI -> ĐỔI CHẾ ĐỘ MÀ HÌNH VẼ CŨ VẪN GIỮ.
        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"
        if _HAS_CANVAS:
            if drawing_mode == "line":
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=4, stroke_color="#FF00FF",
                    background_image=_bg_uri(bg_pil_grid),
                    update_streamlit=True, height=DY, width=DX,
                    drawing_mode="line",
                    return_image_data=True, key="draw_canvas_%d" % _ck)
            else:
                canvas_result = st_canvas(
                    fill_color="rgba(0,120,255,0.45)",
                    stroke_width=4, stroke_color="#00FF00",
                    background_image=_bg_uri(bg_pil_grid),
                    update_streamlit=True, height=DY, width=DX,
                    drawing_mode="polygon",
                    return_image_data=True, key="draw_canvas_%d" % _ck)
        else:
            st.image(bg_pil_grid)
            canvas_result = None
            st.caption("(Canvas không khả dụng trên máy chủ — chạy vẫn dùng được vạch/vùng bạn đã vẽ trước)")

        st.caption("💡 **Vạch** = kéo 1 nét · **Vùng** = click các góc (click **góc đầu** để khép, "
                   "**góc giữa** để bỏ góc). Đổi chế độ Line↔ROI **không mất hình đã vẽ**. "
                   "Sửa: Undo ↶ / thùng rác trên canvas, hoặc 'Xoá hết & Vẽ lại'.")

        # Đọc PIXEL bạn VẼ (không đọc toạ độ JSON -> KHÔNG LỆCH)
        ref_rgb = bg_pil_grid
        mag_m, fill_m = _diff_masks(canvas_result, ref_rgb)
        lines = _lines_from_mask(mag_m, W, H)
        roi_mask = _roi_mask_from(fill_m, W, H)
        _nreg, _ = cv2.connectedComponents(roi_mask.astype(np.uint8))
        n_regions = _nreg - 1

        st.write("📋 Nhận được: **%d vạch** · **%d vùng**  _(đọc từ pixel bạn vẽ — bám sát, không lệch)_"
                 % (len(lines), n_regions))

        # PREVIEW: hiện TẤT CẢ hình (vạch + mọi vùng) trước khi chạy
        if lines or roi_mask.any():
            prev = np.array(st.session_state.bg_image.convert("RGB"))
            for A, B in lines:
                cv2.line(prev, A, B, (255, 0, 255), 3)
            if roi_mask.any():
                cm = roi_mask.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(prev, cnts, -1, (0, 0, 255), 3)
                cv2.fillPoly(prev, cnts, (40, 40, 160))
            st.image(prev, channels="RGB",
                     caption="📌 PREVIEW — hình sẽ hiển thị khi chạy (vạch tím + vùng đỏ) = y hệt bạn vẽ")

        # --- 5. NÚT CHẠY ĐẾM (TẤT CẢ hình hoạt động) ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")
            if not lines and not roi_mask.any():
                st.warning("⚠️ Chưa vẽ gì. Vẽ ít nhất **1 vạch** HOẶC **1 vùng** rồi bấm lại.")
            else:
                st.write("Chạy với **%d vạch** + **%d vùng** — mọi hình trên đều được vẽ & đếm."
                         % (len(lines), n_regions))
                counter_obj = MultiCounter(lines=lines, roi_mask=(roi_mask if roi_mask.any() else None))
                if perf_mode.startswith("Chất lượng"):
                    runner = run_ai_quality
                elif perf_mode.startswith("Siêu nhanh"):
                    runner = run_ai_turbo
                else:
                    runner = run_ai_fast
                try:
                    runner("best.pt", video_path, _workfile("output.mp4"), counter_obj, st.empty())
                    st.balloons()
                    st.success("🎉 Hoàn tất! Báo cáo CSV: `Bao_Cao/` · Video kết quả: `Video_Xuat/`")
                except Exception as e:
                    st.error("❌ Lỗi khi tính toán: %s" % e)
