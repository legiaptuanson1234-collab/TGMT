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
from livestore import TrafficStore, RunControl
import trafficbot
import threading
import time

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


def _shapes_from_json(result, W, H):
    """DOC TRỰC TIẾP dữ liệu canvas (json_data.objects) -> (lines, polys)
    tại toạ độ 1280x720 (hệ đếm). Đây là nguồn ĐÚNG CHUẨN (canvas trả toạ
    độ chính xác từng nét/từng góc) - KHÔNG dùng "so màu pixel" (cách cũ bị
    lệch/tàng hình do rescale ảnh nền sai khác giữa thư viện canvas & OpenCV).

    - Vạch (line)   : obj có 'x1','y1','x2','y2' (tương đối so với left/top).
    - Vùng (polygon): obj có 'points' (TUYỆT ĐỐI, toạ độ canvas) - KHÔNG cộng
                      left/top (cộng thì bị LỆCH - đúng bug 'ROI tàng hình').
    Mỗi nét = 1 vạch, mỗi đa giác = 1 vùng (vẽ N hình được)."""
    sx, sy = W / float(DX), H / float(DY)
    lines, polys = [], []
    jd = getattr(result, "json_data", None)
    if not jd:
        return lines, polys
    objs = jd.get("objects", []) or []
    for o in objs:
        try:
            if "points" in o:                       # VÙNG (đa giác) - điểm TUYỆT ĐỐI
                pts = []
                raw = o.get("points", [])
                if raw and isinstance(raw, (list, tuple)) and len(raw) >= 2 and not isinstance(raw[0], (list, tuple, dict)):
                    # flat: [x0,y0,x1,y1,...]
                    for i in range(0, len(raw) - 1, 2):
                        pts.append((int(raw[i] * sx), int(raw[i + 1] * sy)))
                else:
                    for p in raw:
                        if isinstance(p, dict):
                            pts.append((int(p.get("x", 0) * sx), int(p.get("y", 0) * sy)))
                        elif isinstance(p, (list, tuple)) and len(p) >= 2:
                            pts.append((int(p[0] * sx), int(p[1] * sy)))
                if len(pts) >= 3:
                    polys.append(pts)
            elif "x1" in o and "x2" in o:           # VẠCH (line) - điểm TƯƠNG ĐỐI
                L, T = int(o.get("left", 0)), int(o.get("top", 0))
                a = (int((L + int(o["x1"])) * sx), int((T + int(o["y1"])) * sy))
                b = (int((L + int(o["x2"])) * sx), int((T + int(o["y2"])) * sy))
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) >= 20:
                    lines.append((a, b))
        except Exception:
            pass
    return lines, polys


def _bg_uri(bg_pil):
    import io, base64
    buf = io.BytesIO()
    bg_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _read_shapes_color(result, W, H):
    """DOC MÀU NÉT VẼ TRỰC TIẾP từ ảnh canvas (RGBA) - KHÔNG so với ảnh nền
    (cách cũ so nền bị LỆCH/PHÌNH khi render khác rescale -> 'tàng hình').

    - Nét TÍM  #FF00FF = VẠCH (Line)        -> đọc từng nét, lấy 2 đầu (PCA)
    - Nét XANH LÁ #00FF00 = VIỀN VÙNG (ROI)  -> contour ngoài -> điền đầy bên trong

    Hai màu này không có trong video thật / lưới cam / mốc vàng -> tách sạch,
    vẽ ở đâu đếm đúng đó. Vạch & vùng đọc CÙNG LÚC (đổi chế độ không mất hình).
    Trả về (lines, roi_mask) tại toạ độ 1280x720 (hệ đếm)."""
    sx, sy = W / float(DX), H / float(DY)
    canvas = _img(result)                      # (DY, DX, 4) RGBA hoặc None
    roi_mask = np.zeros((H, W), bool)
    lines = []
    if canvas is None:
        return lines, roi_mask
    R = canvas[:, :, 0].astype(np.int16)
    G = canvas[:, :, 1].astype(np.int16)
    B = canvas[:, :, 2].astype(np.int16)

    # --- VẠCH: nét TÍM (#FF00FF) ---
    mag = ((R > 150) & (B > 150) & (G < 100)).astype(np.uint8) * 255
    mag = cv2.dilate(mag, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mag, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for c in cnts:
        if c.size < 12:
            continue
        pts = c[:, 0, :].astype(np.float64)
        mean = pts.mean(axis=0)
        _, ev = np.linalg.eigh(np.cov(pts.T))
        v = ev[:, -1]
        t = (pts - mean) @ v
        i1, i2 = int(np.argmin(t)), int(np.argmax(t))
        p1 = (int(pts[i1][0] * sx), int(pts[i1][1] * sy))
        p2 = (int(pts[i2][0] * sx), int(pts[i2][1] * sy))
        if abs(p1[0] - p2[0]) + abs(p1[1] - p2[1]) >= 20:
            lines.append((p1, p2))

    # --- VÙNG: nét XANH LÁ (#00FF00 = viền đa giác) -> điền đầy bên trong ---
    grn = ((G > 170) & (R < 90) & (B < 90)).astype(np.uint8) * 255
    grn = cv2.morphologyEx(grn, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    gcnts, _ = cv2.findContours(grn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gmask = np.zeros_like(grn)
    for c in gcnts:
        if cv2.contourArea(c) >= 300:          # loại blob nhỏ (nhiễu)
            cv2.drawContours(gmask, [c], -1, 255, -1)   # điền bên trong
    roi_mask = cv2.resize(gmask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return lines, roi_mask


# --- 1. CẤU HÌNH ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")

# [CHATBOT] kho du lieu real-time (thread AI ghi, chatbot + CSV doc - khong kho man hinh)
if "store" not in st.session_state:
    st.session_state.store = TrafficStore()
if "ctrl" not in st.session_state:
    st.session_state.ctrl = RunControl()
if "ai_thread" not in st.session_state:
    st.session_state.ai_thread = None

# Frame tuom: thread AI khong viep UI (video live chi do fragment doc tu store) -> het lag
class _NoopFrame:
    def image(self, *a, **k):
        pass
    def error(self, *a, **k):
        pass
    def progress(self, *a, **k):
        pass
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
        st.success("✅ **Khung đường đã sẵn sàng** — hộp vẽ bên dưới hiển thị khung nền + lưới. "
                   "Nếu nền hộp vẽ (canvas) bị mất trên Cloud (bug nền của thư viện canvas), "
                   "**dùng khung tham khảo phía dưới làm chuẩn đối chiếu** để vẽ đúng vị trí.")

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

        # [FIX Cloud] Kan man hinh LUON DU (road + grid + toa do) - neu nen canvas
        # bi mat (bug 0.13 tren Cloud) thi van thay duong de doi chieu. O may (local
        # hoac nen canvas hien binh thuong) thi lam khung tham chieu phu.
        with st.expander("🖼️ Khung tham chiếu (đường + lưới toạ độ) — mở ra để vẽ không sai vị trí",
                         expanded=(not _HAS_CANVAS)):
            st.image(bg_pil_grid, width=DX,
                     caption="Ảnh nền có lưới 100px + mốc toạ độ. Vẽ trên canvas phía trên bám theo khung này.")

        # Đọc MÀU NÉT VẼ (tím=vạch, xanh-lá=vùng) - KHÔNG so nền -> không phình/lech
        lines, roi_mask = _read_shapes_color(canvas_result, W, H)
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

        # --- 5. NÚT CHẠY ĐẾM (v11: AI chay NEN (thread) -> song song voi chatbot, KHONG dong man hinh) ---
        _ctrl = st.session_state.ctrl
        _ai_thread = st.session_state.ai_thread

        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time (chạy NỀN — vẫn hỏi được CHATBOT bên dưới)")
            if not lines and not roi_mask.any():
                st.warning("⚠️ Chưa vẽ gì. Vẽ ít nhất **1 vạch** HOẶC **1 vùng** rồi bấm lại.")
            else:
                if _ai_thread is not None and _ai_thread.is_alive():
                    st.warning("AI đang chạy — bấm *Dừng & Lưu* trước khi chạy lại.")
                else:
                    st.write("🚀 Khởi động AI **nền**: video + KPI cập nhật bên dưới (không đơ màn hình), "
                             "**hỏi CHATBOT bất kỳ lúc nào** — không cần chờ video hết."
                             " Chạy với **%d vạch** + **%d vùng**." % (len(lines), n_regions))
                    counter_obj = MultiCounter(lines=lines, roi_mask=(roi_mask if roi_mask.any() else None))
                    if perf_mode.startswith("Chất lượng"):
                        _runner = run_ai_quality
                    elif perf_mode.startswith("Siêu nhanh"):
                        _runner = run_ai_turbo
                    else:
                        _runner = run_ai_fast
                    st.session_state.store.reset()
                    _ctrl.reset()
                    st.session_state._out_path = _workfile("output.mp4")

                    def _ai_worker(_runner_fn, _vid, _out, _ctr, _st, _store, _ctl):
                        # st.empty() NO-OP: video live chi do fragment _live_panel doc tu _store
                        # (khong st.image tung tu thread nen -> het lag/tranh don UI)
                        try:
                            _runner_fn("best.pt", _vid, _out, _ctr, _NoopFrame(), store=_store, control=_ctl)
                        except Exception as _e:
                            _store.on_finish(ok=False, error=str(_e))

                    st.session_state.ai_thread = threading.Thread(
                        target=_ai_worker,
                        args=(_runner, video_path, st.session_state._out_path,
                              counter_obj, None, st.session_state.store, _ctrl),
                        daemon=True)
                    st.session_state.ai_thread.start()
                    st.success("🚀 AI đang chạy NỀN (thread %s). Video + KPI cập nhật bên dưới · "
                               "hỏi CHATBOT ngay phía cuối trang!" % st.session_state.ai_thread.name)
        # Nut dieu khiene (an toan giua thread + gui)
        _running_now = _ai_thread is not None and _ai_thread.is_alive()
        _s0 = st.session_state.store.stats()
        if _running_now:
            _p1, _p2, _p3 = st.columns([1, 1, 2])
            if _p1.button("⏯️ Tạm dừng / Tiếp"):
                _ctrl.toggle_pause()
            if _p2.button("⛔ Dừng & Lưu CSV", type="secondary"):
                _ctrl.request_stop()
            _p3.markdown("📟 *AI chạy nền · %s · KPI cập nhật bên dưới*" % _s0["status"])
        elif _s0["finished"] and _s0["total"] > 0:
            st.balloons()
            st.success("🎉 Hoàn tất: **%d xe** đã đếm! Báo cáo CSV: `Bao_Cao/` · Video kết quả: `Video_Xuat/`"
                      % _s0["total"])
            _sp0 = st.session_state.store.write_summary_csv()
            with open(_sp0, "rb") as _fh0:
                st.download_button("📊 Tải CSV thống kê theo phút", _fh0,
                                   file_name=os.path.basename(_sp0), mime="text/csv")

# ================= [V11] PANEL VIDEO LIVE + KPI (top-level, ON DINH — chat KHÔNG làm mất) =================
# Fragment chi cap nhat PANEL NAY (video + KPI); canvas ben tren + chatbot cuoi trang
# deu la top-level -> moi lan hoi bot (rerun) van giu du, khong mat video.
# Cai element LUON DUY NHAT (1 image + 1 markdown KPI + 1 dataframe) -> chi DOI NOI DUNG
# tai choi -> het hien tuong nhip nhay an/hien (sai lech cu a v9.2/9.3).

def _live_panel_v11():
    """Panel video live (chi panel nay tu cap nhat). CAI ELEMENT ON DINH (luon co
    dung 1 image + 1 KPI + 1 bang) -> chi DOI NOI DUNG tai choi, khong tao lai
    widget => HET nhip nhay an/hien. Chua co khung AI: hien KHUNG DUONG DA VE
    (khong o den)."""
    _st = st.session_state.store
    _fr = _st.get_frame()
    _s = _st.stats()
    if _fr is not None:
        _rgb = cv2.cvtColor(cv2.resize(_fr, (960, 540)), cv2.COLOR_BGR2RGB)
        _cap = "📺 Khung mới nhất (AI chạy NỀN — cập nhật liên tục, không đơ)"
    elif "bg_image" in st.session_state and st.session_state.bg_image is not None:
        _rgb = np.array(st.session_state.bg_image)
        _cap = ("⏳ AI đang khởi động/chạy — hiển thị khung đường để vẽ"
                if _s["running"] else "Khung đường (vẽ xong, chưa chạy AI). Bấm ▶️ CHẠY AI bên trên.")
    else:
        _rgb = np.zeros((540, 960, 3), dtype=np.uint8)
        _cap = "Chưa có video — tải video + vẽ + ▶️ CHẠY AI (bên trên) rồi quay lại đây."
    st.image(_rgb, channels="RGB", width=960, caption=_cap)
    # KPI
    def m(ic, lb, val, delta=""):
        d = ('<div style="font-size:.72em;color:#7ee787">%s</div>' % delta) if delta else ""
        return ('<div style="flex:1;min-width:110px;padding:6px 10px;background:#0e1117;'
                'border:1px solid #26272b;border-radius:8px">'
                '<div style="font-size:.78em;color:#a3a8b3">%s %s</div>'
                '<div style="font-size:1.3em;font-weight:700;color:#f0f2f6">%s</div>%s</div>' % (ic, lb, val, d))
    _row = ('<div style="display:flex;gap:8px;flex-wrap:wrap">'
            + m("🚦", "Trạng thái", _s["status"],
                "đang chạy" if _s["running"] else ("kết thúc" if _s["finished"] else "chờ"))
            + m("∑", "Tổng xe", str(_s["total"]),
                ("%d/phút" % _s["per_minute"]) if _s["events"] else "")
            + m("▣", "Trong khung", str(_s["active_now"]))
            + m("⏱", "FPS", str(_s["fps"]))
            + m("⚡", "TB km/h", "%.1f" % _s["avg_speed"]) + "</div>")
    st.markdown(_row, unsafe_allow_html=True)
    # Bang su kien moi nhat (CAI ON DINH: luon la st.dataframe, bieng rong khi chua co su kien)
    _CLS_M = {0: "Ô tô", 1: "Xe máy", 2: "Xe tải", 3: "Xe bus", 4: "Xe ba gác"}
    _evs = _st.events[-12:][::-1]
    import pandas as pd
    if _evs:
        _df = pd.DataFrame(_evs)
        _df["Loai"] = [_CLS_M.get(e["cls"], "?") for e in _evs]
        _ev_cols = _df[["wall", "t", "Loai", "speed"]]
    else:
        _ev_cols = pd.DataFrame(columns=["wall", "t", "Loai", "speed"])
    st.dataframe(_ev_cols, use_container_width=True, height=220, hide_index=True)
    if not _ev_cols.empty:
        st.caption("📋 %d sự kiện mới nhất (mỗi xe đếm 1 lần) — CSV ghi liên tục khi có xe." % len(_ev_cols))


if hasattr(st, "fragment"):
    @st.fragment(run_every=0.7)
    def _live_auto_v11():
        _live_panel_v11()
    _live_auto_v11()
else:
    _live_panel_v11()

st.caption("📡 **CSV live đang ghi**: `"
           + (os.path.basename(st.session_state.store.stats()["detail_csv"]) if st.session_state.store.stats()["detail_csv"] else "(chưa chạy)")
           + "` — mở Excel xem được ngay trong lúc video chạy.")

# ================= [CHATBOT] v11 — top-level, dung song song voi video (khong mat) =================
store = st.session_state.store
st.markdown("---")
st.subheader("💬 Chatbot Giao Thông (hỏi **ngay trong lúc video đang chạy** — dữ liệu real-time)")
st.caption("Ví dụ: *'nay có bao nhiêu xe?'* · *'từ 2 phút sang 5 phút có bao nhiêu xe máy?'* · "
           "*'ùn tắc chưa?'* · *'tốc độ trung bình?'* · *'xuất csv'")

if "chat" not in st.session_state:
    st.session_state.chat = []
for _role, _txt in st.session_state.chat:
    with st.chat_message("user" if _role == "u" else "bot"):
        st.markdown(_txt)

_q = st.chat_input("Hỏi bot: bao nhiêu xe / loại nào / mật độ / tốc độ / xuất csv...")
if _q:
    st.session_state.chat.append(("u", _q))
    _reply = trafficbot.ask(store, _q, store.stats()["video_sec"])
    st.session_state.chat.append(("b", _reply))
    with st.chat_message("bot"):
        st.markdown(_reply)
if not st.session_state.chat:
    with st.chat_message("bot"):
        st.markdown(trafficbot.help_reply(store))
