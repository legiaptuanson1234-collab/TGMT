# APP.py - TGMT v9: "Traffic Copilot" (WEB BANA desktop v9.1 - y hieu desktop)
# Layout: BANG DIEU KHIEN (sidebar) + VIDEO LON + KPI LIVE + CHATBOT (ben duoi)
# Coi nhu app desktop: 1.Chon video -> 2.Che do -> 3.Hieu nang -> Ve -> CHAY -> Hoi bot.
# [KHONG DO] thread AI chi GHI vao TrafficStore; GUI poll khung/KPI bang st.fragment
# (chi panel live tu cap nhat, canvas + chatbot khong bi don).
import streamlit as st
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageDraw
import os
import tempfile

import importlib, sys
for _m in ("counting", "tracking"):
    sys.modules.pop(_m, None)
import counting, tracking
from tracking import run_ai_fast, run_ai_quality, run_ai_turbo, DISPLAY_W, DISPLAY_H
from counting import MultiCounter
from livestore import TrafficStore, RunControl
import trafficbot

try:
    from streamlit_drawable_canvas import st_canvas
    _HAS_CANVAS = True
except Exception:
    _HAS_CANVAS = False

BUILD_TAG = "TGMT-v9 · GUI giống app desktop · Video lớn · Chatbot real-time · CSV live"

W, H = DISPLAY_W, DISPLAY_H       # 1280x720 (he toa do dem)
DX, DY = 960, 540                 # canvas ve (fit man, thay du duong)

st.set_page_config(page_title="Traffic Copilot - UTT", layout="wide", page_icon="🚦")


# ================= helpers (giu nguyen - da test on) =================
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
                    return frame, "cv2 OK · khung nend #%d (std=%.0f)" % (scanned - 1, std)
                if score > best_score:
                    best, best_score, best_idx = frame, score, scanned - 1
        cap.release()
        if best is not None:
            return best, "cv2 OK · khung nend chi tiet nhat (#%d)" % best_idx
        return None, "cv2 mo duoc file nhung %d frame dau deu trang/đen" % scanned
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
                    return fr, "ffmpeg OK · khung nend #%d" % n - 1
                if score > best_score:
                    best, best_score, best_idx = fr, score
        if best is not None:
            return best, "ffmpeg OK · khung nend chi tiet nhat (#%d)" % best_idx
        return None, "ca OpenCV lan ffmpeg khong doc duoc video"
    except Exception as _e:
        return None, "khong doc duoc video: %s" % _e


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
    if result is None:
        return None
    data = getattr(result, "image_data", None)
    if data is None:
        return None
    try:
        arr = np.asarray(data)
    except Exception:
        return None
    if arr is None or arr.size == 0 or arr.ndim < 2:
        return None
    if arr.ndim == 4:
        arr = arr[:, :, :, 0]
    if arr.ndim < 3:
        return None
    if arr.shape[0] != DY or arr.shape[1] != DX:
        arr = cv2.resize(arr, (DX, DY))
    return arr


def _read_shapes_color(result, W, H):
    """Doc mau net ve: tím #FF00FF = vach, xanh-la #00FF00 = vien ROI -> dien day.
    Tra ve (lines=[(A,B),...], roi_mask) toa do 1280x720 - DUNG CẤU TRUC (moi
    vach = 1 phan tu (diemA, diemB); roi_mask = 1 matrix dien day)."""
    sx, sy = W / float(DX), H / float(DY)
    canvas = _img(result)
    roi_mask = np.zeros((H, W), bool)
    lines = []
    if canvas is None:
        return lines, roi_mask
    R = canvas[:, :, 0].astype(np.int16)
    G = canvas[:, :, 1].astype(np.int16)
    B = canvas[:, :, 2].astype(np.int16)
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
    grn = ((G > 170) & (R < 90) & (B < 90)).astype(np.uint8) * 255
    grn = cv2.morphologyEx(grn, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    gcnts, _ = cv2.findContours(grn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gmask = np.zeros_like(grn)
    for c in gcnts:
        if cv2.contourArea(c) >= 300:
            cv2.drawContours(gmask, [c], -1, 255, -1)
    roi_mask = cv2.resize(gmask, (W, H), interpolation=cv2.INTER_NEAREST) > 0
    return lines, roi_mask


def _btn_full_width(label):
    try:
        return st.button(label, type="primary", width="stretch")
    except TypeError:
        return st.button(label, type="primary", use_container_width=True)


# ================= session global =================
if "store" not in st.session_state:
    st.session_state.store = TrafficStore()
if "control" not in st.session_state:
    st.session_state.control = RunControl()
if "run_thread" not in st.session_state:
    st.session_state.run_thread = None
if "chat" not in st.session_state:
    st.session_state.chat = []
if "chat_q" not in st.session_state:
    st.session_state.chat_q = ""

class _StubFrame:
    """[v9.2] Thread AI chi duoc choStub -> NO-OP (UI chi do fragment cap nhat).
    Viep st.image/st.progress tu thread nen xac nghich voi fragment -> nhip nhay + 'an/hiện'."""
    def __getattr__(self, _name):
        return lambda *a, **k: None

store = st.session_state.store
control = st.session_state.control
_CLS = {0: "Ô tô", 1: "Xe máy", 2: "Xe tải", 3: "Xe bus", 4: "Xe ba gác"}


def _thread_done():
    th = st.session_state.run_thread
    return th is None or not th.is_alive()


def _kpi_row():
    s = store.stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🚦 Trạng thái", s["status"],
              delta="đang chạy" if s["running"] else ("kết thúc" if s["finished"] else "chờ"))
    c2.metric("Tổng xe", s["total"],
              delta=("%d/phút" % s["per_minute"]) if s["events"] else None)
    c3.metric("Trong khung", s["active_now"])
    c4.metric("FPS", s["fps"])
    c5.metric("⚡ TB km/h", s["avg_speed"])


def _mini_stats_box():
    s = store.stats()
    txt = "Ô tô %d · Xe máy %d · Xe tải %d · Xe bus %d · Xe ba gác %d" % (
        s["counts"].get(0, 0), s["counts"].get(1, 0), s["counts"].get(2, 0),
        s["counts"].get(3, 0), s["counts"].get(4, 0))
    st.caption("📊 " + txt + ("  |  Đỉnh: phút " + str(s["peak"]["minute"]) +
                              " (%d xe/phút)" % s["peak"]["total"] if s["peak"] else ""))


# ================= HEADER (giu nguyen nhu desktop) =================
st.title("🚦 TRAFFIC COPILOT — Hệ thống đếm xe & đối thoại giao thông AI")
st.caption("Đồ án Kỹ thuật — Lê Giáp Tuấn Sơn — UTT  ·  🔖 " + BUILD_TAG)

# ================= BẢNG ĐIỀU KHIỂN (sidebar - y hieu desktop) =================
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("**1. Video**")
    uploaded_file = st.file_uploader("video (mp4/avi/mov)", type=['mp4', 'avi', 'mov'])

    st.markdown("**2. Chế độ đếm**")
    mode = st.radio("Nguồn đếm", ["Đếm qua vạch (Line)", "Đếm trong vùng (ROI)"], label_visibility="collapsed")

    st.markdown("**3. Hiệu năng**")
    perf_mode = st.radio("Hiệu năng",
                         ["Nhanh (CPU 640x360)", "Siêu nhanh (CPU demo 480x270)", "Chất lượng (GPU 1280x720)"],
                         index=1, label_visibility="collapsed")
    st.caption("🌐 **Cloud không có GPU** (CPU only) — chọn *Chất lượng (GPU)* vẫn chạy bằng CPU nên chậm (FPS ~4-6). "
               "**Đã chọn sẵn 'Siêu nhanh (480x270)' cho demo mượt nhất.** "
               "Chạy app trên **máy có GPU** thì chọn *Chất lượng (GPU 1280x720)*.")

    if _thread_done():
        btn_run = _btn_full_width("▶️ BẮT ĐẦU CHẠY (+ mở CHATBOT)")
    else:
        st.info("⏳ AI đang chạy NỀN — dùng nút bên dưới để Tạm dừng/Dừng.")
        btn_run = False
        if st.button("⏯️ Tạm dừng / Tiếp tục"):
            control.toggle_pause()
        if st.button("⛔ Dừng hẳn & Lưu"):
            control.request_stop()

    if st.session_state.get("_video_loaded") and st.session_state.get("_has_shapes"):
        st.success("✅ Đã nạp %s · %d hình vẽ · sẵn sàng chạy" % (
            st.session_state.get("_vname", "?"), st.session_state.get("_nshapes", 0)))

    st.markdown("---")
    st.markdown("**Cảm biến Real-time**")
    _kpi_row()
    s = store.stats()
    if s["detail_csv"] and os.path.exists(s["detail_csv"]):
        st.markdown("📡 **CSV live đang ghi**: `%s`" % os.path.basename(s["detail_csv"]))
        with open(s["detail_csv"], "rb") as _fh:
            st.download_button("⬇️ Tải CSV sự kiện (live)", _fh,
                               file_name=os.path.basename(s["detail_csv"]), mime="text/csv")
    else:
        st.caption("CSV live: chưa chạy")
    if s["events"]:
        p = store.write_summary_csv()
        if os.path.exists(p):
            with open(p, "rb") as _fh2:
                st.download_button("📊 Tải CSV thống kê theo phút", _fh2,
                                   file_name=os.path.basename(p), mime="text/csv")

# ================= VIDEO + VẼ (main - khung rong to) =================
st.subheader("📥 Nạp video — 🖊️ Vẽ vạch (tím) / vùng (xanh lá) — rồi ▶️ CHẠY")
if uploaded_file is None:
    st.info("Tải video lên ở **Bảng điều khiển** (sidebar) để bắt đầu.")
else:
    video_path = _workfile("video_tam.mp4")
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.session_state["_video_loaded"] = False
        st.session_state["_has_shapes"] = False
        frame, bg_info = _find_content_frame(video_path)
        if frame is not None:
            frame = cv2.resize(frame, (W, H))
            st.session_state.bg_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            st.session_state._video_loaded = True
            st.session_state._vname = uploaded_file.name
            st.caption("🎞️ Đã cắt khung để vẽ: " + bg_info)
        else:
            st.error("⚠️ " + bg_info)

    if st.session_state.get("bg_image") is not None:
        bg_pil_grid = _bake_grid(st.session_state.bg_image)
        _ck = st.session_state.get("_canvas_seq", 0)
        if st.button("🗑️ Xoá hết & Vẽ lại"):
            st.session_state["_canvas_seq"] = _ck + 1
            st.session_state["_has_shapes"] = False
            st.rerun()

        drawing_mode = "line" if mode.startswith("Đếm qua vạch") else "polygon"
        canvas_result = None
        if _HAS_CANVAS:
            if drawing_mode == "line":
                canvas_result = st_canvas(
                    fill_color="rgba(0,0,0,0)",
                    stroke_width=4, stroke_color="#FF00FF",
                    background_image=_bake_grid(st.session_state.bg_image),
                    update_streamlit=True, height=DY, width=DX,
                    drawing_mode="line", return_image_data=True,
                    key="draw_canvas_%d" % _ck)
            else:
                canvas_result = st_canvas(
                    fill_color="rgba(0,120,255,0.45)",
                    stroke_width=4, stroke_color="#00FF00",
                    background_image=_bake_grid(st.session_state.bg_image),
                    update_streamlit=True, height=DY, width=DX,
                    drawing_mode="polygon", return_image_data=True,
                    key="draw_canvas_%d" % _ck)
        else:
            st.image(bg_pil_grid, width=DX)
            st.caption("(Canvas không khả dụng — dùng vạch/vùng đã vẽ trước)")

        lines, roi_mask = _read_shapes_color(canvas_result, W, H)
        _nreg, _ = cv2.connectedComponents(roi_mask.astype(np.uint8))
        n_shapes = len(lines) + (_nreg - 1)
        st.session_state["_has_shapes"] = n_shapes > 0
        st.session_state["_lines"] = lines
        st.session_state["_roi_mask"] = roi_mask
        st.write("📋 Nhận được: **%d vạch** · **%d vùng**" % (len(lines), _nreg - 1))

        if lines or roi_mask.any():
            prev = np.array(st.session_state.bg_image.convert("RGB"))
            for A, B in lines:
                cv2.line(prev, A, B, (255, 0, 255), 3)
            if roi_mask.any():
                cm = roi_mask.astype(np.uint8) * 255
                cnts, _ = cv2.findContours(cm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(prev, cnts, -1, (0, 0, 255), 3)
                cv2.fillPoly(prev, cnts, (40, 40, 160))
            st.image(prev, width=DX, caption="📌 PREVIEW hình khi chạy")

        st.caption("💡 Vạch = kéo 1 nét · Vùng = click các góc (góc đầu khép). "
                   "Sửa: Undo / thùng rác trên canvas.")

# ================= CHẠY (giu y hieu desktop: noi Line/ROI -> counter) =================
if btn_run:
    if not st.session_state.get("_has_shapes"):
        st.warning("⚠️ Vẽ ít nhất **1 vạch** HOẶC **1 vùng** phía trên rồi quay lại bấm.")
    else:
        # [Y HIEU DESKTOP] cau truc DUNG: moi vach = 1 phan tu (diemA, diemB);
        # roi_mask = matrix dien day (khong dung roi_polygons nua).
        _lines = st.session_state["_lines"] or None
        _mask = st.session_state["_roi_mask"]
        _use_mask = _mask is not None and _mask.any()
        counter_obj = MultiCounter(
            lines=_lines if not _use_mask else None,
            roi_mask=_mask if _use_mask else None)
        if perf_mode.startswith("Chất lượng"):
            runner = run_ai_quality
        elif perf_mode.startswith("Siêu nhanh"):
            runner = run_ai_turbo
        else:
            runner = run_ai_fast

        store.reset()
        control.reset()
        st.session_state.chat = []
        st.session_state.chat_q = ""
        store.on_start(perf_mode)
        st.session_state["out_path"] = _workfile("output.mp4")

        def _ai_worker(runner_fn, _m, _vin, _vout, _ctr, _st, _store, _ctl):
            try:
                runner_fn(_m, _vin, _vout, _ctr, _st, store=_store, control=_ctl)
            except Exception as e:
                _store.on_finish(ok=False, error=str(e))

        th = threading.Thread(
            target=_ai_worker,
            args=(runner, "best.pt", video_path, st.session_state["out_path"],
                  counter_obj, _StubFrame(), store, control),
            daemon=True)
        st.session_state.run_thread = th
        th.start()
        st.session_state.chat.append(("b", "🤖 Video bắt đầu lúc " + time.strftime("%H:%M:%S") +
                                      " · CSV live: " + os.path.basename(store.detail_csv)))
        st.success("🚀 AI đang chạy NỀN — video + KPI cập nhật bên dưới, HỎI CHATBOT ngay!")
        st.rerun()

# ================= VIDEO LIVE + KPI (fragment - chi panel nay tu cap nhat) =================
def _live_panel():
    """Chỉ panel này tự refresh (0.5s) — canvas, chatbot, sidebar KHÔNG bị rerun -> hết đơ."""
    fr = store.get_frame()
    if fr is not None:
        st.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), width=980,
                 caption="📺 Khung mới nhất (live từ thread AI)")
    elif store.stats()["running"]:
        st.info("Chờ khung đầu tiên (YOLO vừa khởi động)...")
    else:
        st.info("Chưa có khung — ▶️ CHẠY ở Bảng điều khiển rồi quay lại đây.")
    _kpi_row()
    _mini_stats_box()
    evs = store.events[-15:][::-1]
    if evs:
        st.markdown("**Bảng sự kiện mới nhất (live)**")
        import pandas as pd
        df = pd.DataFrame(evs)
        df["Loai"] = [_CLS.get(e["cls"], "?") for e in evs]
        st.dataframe(df[["wall", "t", "Loai", "speed"]], use_container_width=True, height=240)


def _live_panel_v92():
    """[v9.2] Key on dinh + anh nho (560px) -> fragment cap nhat TAI CHO (khong tao lai),
    payload nho nen web Cloud hong con nhip nhay / an-hien. run_every 0.8s (mat du) de AI kip ghi."""
    # [v9.2-fix] st.image/st.info/st.markdown KHÔNG có tham so key (Streamlit >=1.53
    # chi widget input moi nhan key) -> bo key de het TypeError; fragment van
    # cap nhat TAI CHO theo cau truc nen van khong nhip nhay.
    fr = store.get_frame()
    if fr is not None:
        rgb = cv2.cvtColor(cv2.resize(fr, (560, 315)), cv2.COLOR_BGR2RGB)
        st.image(rgb, channels="RGB", width=560,
                 caption="📺 Khung mới nhất (live từ thread AI) — cập nhật ~1.2s/lần, KHÔNG nhấp nháy")
    else:
        st.info("Chờ khung đầu tiên (YOLO vừa khởi động)..." if store.stats()["running"]
                else "Chưa có khung — ▶️ CHẠY ở Bảng điều khiển rồi quay lại đây.")
    _kpi_row()
    _mini_stats_box()
    evs = store.events[-15:][::-1]
    if evs:
        st.markdown("**Bảng sự kiện mới nhất (live)**")
        import pandas as pd
        df = pd.DataFrame(evs)
        df["Loai"] = [_CLS.get(e["cls"], "?") for e in evs]
        st.dataframe(df[["wall", "t", "Loai", "speed"]], use_container_width=True, height=240)


if hasattr(st, "fragment"):
    @st.fragment(run_every=0.8)
    def _live_auto():
        _live_panel_v92()
    _live_auto()
else:
    _live_panel_v92()

# ================= THỐNG KÊ & ĐỒ THỊ (an toan - sua KeyError v8) =================
st.subheader("📊 Thống kê & mật độ theo thời gian (cập nhật LIÊN TỤC khi video chạy)")
_s = store.stats()
if _s["events"]:
    import pandas as pd
    _c1, _c2 = st.columns(2)
    with _c1:
        st.markdown("**Xe/phút theo từng phút video**")
        _bdf = pd.DataFrame({"mat_do_xe_phut": [store.bin_total[m] for m in sorted(store.bin_total)]})
        _bdf = _bdf.reset_index().rename(columns={"index": "phut"})
        st.bar_chart(_bdf.set_index("phut")["mat_do_xe_phut"])
    with _c2:
        st.markdown("**Phân bố loại xe (tổng từ đầu)**")
        _total_now = _s["total"] or 1
        _cdf = pd.DataFrame({
            "Loai": [_CLS[c] for c in range(5)],
            "So xe": [_s["counts"].get(c, 0) for c in range(5)],
            "Tiep (%)": [round(100.0 * _s["counts"].get(c, 0) / _total_now, 1) for c in range(5)],
        })
        st.bar_chart(_cdf.set_index("Loai")["So xe"])
        st.dataframe(_cdf, use_container_width=True, hide_index=True)
    st.markdown("**Mật độ 5 phút gần nhất: %d xe/phút** · Trạng thái: %s · Tốc độ TB: %.1f km/h"
                % (_s["per_minute"], _s["status"], _s["avg_speed"]))
    if _s["peak"]:
        st.markdown("🔺 **Đỉnh cao điểm**: phút %d–%d (%d xe/phút)" % (
            _s["peak"]["minute"], _s["peak"]["minute"] + 1, _s["peak"]["total"]))
else:
    st.info("Chưa có dữ liệu — ▶️ CHẠY rồi quay lại. Đồ thị cập nhật LIÊN TỤC.")

# ================= CHATBOT (ben duoi - y hieu desktop) =================
st.subheader("💬 Chatbot Giao thông — hỏi TRONG LÚC video đang chạy")
st.caption("Mọi câu trả lời đọc **ngay từ bộ dữ liệu real-time** (không cần chờ video kết thúc). "
           "Ví dụ: *nay có bao nhiêu xe?* · *từ 2 phút sang 5 phút có bao nhiêu xe máy?* · *ùn tắc chưa?* · *tốc độ trung bình?* · *xuất csv*")

for _role, _txt in st.session_state.chat:
    with st.chat_message("user" if _role == "u" else "bot"):
        st.markdown(_txt)

if st.session_state.chat_q:
    st.session_state.chat.append(("u", st.session_state.chat_q))
    reply = trafficbot.ask(store, st.session_state.chat_q, store.stats()["video_sec"])
    st.session_state.chat.append(("b", reply))
    st.session_state.chat_q = ""
    st.rerun()

_user_q = st.chat_input("Hỏi bot: bao nhiêu xe / loại nào / mật độ / tốc độ / xuất csv...")
if _user_q:
    st.session_state.chat_q = _user_q
    st.rerun()

if not st.session_state.chat:
    st.markdown(trafficbot.help_reply(store))
