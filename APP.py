# APP.py - TGMT v8: "Traffic Copilot"
# Gop he dem xe (v7b) + CHATBOT GIAO THONG real-time (song song video chay)
# + CSV live (ghi lien tuc, khong phai cho ket thuc) + GUI 5 tab chuyen nghiep.
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

BUILD_TAG = "TGMT-v8 Traffic Copilot · GUI 5 tab · Chatbot real-time · CSV live · Nút Tạm dừng/Dừng"

W, H = DISPLAY_W, DISPLAY_H       # 1280x720 (hệ toạ độ đếm)
DX, DY = 960, 540                # canvas hiển thị (fit màn, thấy đủ đường)

st.set_page_config(page_title="Traffic Copilot - UTT", layout="wide", page_icon="🚦")


# ================= helpers cũ (giữ nguyên - đã test ổn) =================
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
                    best, best_score, best_idx = fr, score
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
    """Đọc màu nét vẽ từ ảnh canvas: tím #FF00FF = vạch, xanh-lá #00FF00 = viền ROI
    -> điền đầy bên trong. Trả về (lines, roi_mask) toạ độ 1280x720."""
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


def _bg_uri(bg_pil):
    import io, base64
    buf = io.BytesIO()
    bg_pil.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _btn_full_width(label):
    try:
        return st.button(label, type="primary", width="stretch")
    except TypeError:
        return st.button(label, type="primary", use_container_width=True)


# ================= session global (mắt quan sát thread AI) =================
if "store" not in st.session_state:
    st.session_state.store = TrafficStore()
if "control" not in st.session_state:
    st.session_state.control = RunControl()
if "run_thread" not in st.session_state:
    st.session_state.run_thread = None
if "chat" not in st.session_state:
    st.session_state.chat = []      # [(role, text), ...]
if "chat_q" not in st.session_state:
    st.session_state.chat_q = ""

store = st.session_state.store
control = st.session_state.control


def _thread_done():
    th = st.session_state.run_thread
    return th is None or not th.is_alive()


def _kpi_row():
    s = store.stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🚦 Trạng thái", s["status"],
              delta="chạy" if s["running"] else ("kết thúc" if s["finished"] else "chờ"))
    c2.metric("Tổng xe", s["total"],
              delta="%d/phút" % s["per_minute"] if s["events"] else None)
    c3.metric("Trong khung", s["active_now"])
    c4.metric("FPS", s["fps"])
    c5.metric("⚡ TB km/h", s["avg_speed"])


def _mini_stats_box():
    s = store.stats()
    txt = "Ô tô %d · Xe máy %d · Xe tải %d · Xe bus %d · Xe ba gác %d" % (
        s["counts"].get(0, 0), s["counts"].get(1, 0), s["counts"].get(2, 0),
        s["counts"].get(3, 0), s["counts"].get(4, 0))
    st.markdown("**_minisubtitle_** " + txt) if False else None
    st.caption("📊 " + txt + ("  |  Đỉnh: phút " + str(s["peak"]["minute"]) +
                              " (%d xe)" % s["peak"]["total"] if s["peak"] else ""))


# ================= HEADER =================
st.title("🚦 TRAFFIC COPILOT — Hệ thống đếm xe & đối thoại giao thông AI")
st.caption("Đồ án Kỹ thuật — Lê Giáp Tuấn Sơn — UTT  ·  🔖 " + BUILD_TAG)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1️⃣ Nhập video & Vẽ vạch/Vùng",
    "2️⃣ Giám sát Real-time",
    "3️⃣ Thống kê & Đồ thị",
    "4️⃣ 💬 Chatbot Giao thông",
    "5️⃣ Xuất báo cáo (CSV/PDF)",
])

with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    st.markdown("**1. Video**")
    uploaded_file = st.file_uploader("video (mp4/avi/mov)", type=['mp4', 'avi', 'mov'])
    st.markdown("**2. Nguồn đếm**")
    mode = st.radio("Chế độ", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    st.markdown("**3. Tốc độ AI**")
    perf_mode = st.radio("Hiệu năng",
                         ["Nhanh (CPU 640x360)", "Siêu nhanh (CPU demo 480x270)", "Chất lượng (GPU 1280x720)"])
    if _thread_done():
        btn_run = _btn_full_width("🚀 BẮT ĐẦU CHẠY AI (đồng thời mở chatbot)")
    else:
        st.info("⏳ AI đang chạy ở tab 2/3/4 — dùng nút Tạm dừng/Dừng bên đó.")
        btn_run = False
    if st.session_state.get("_video_loaded") and st.session_state.get("_has_shapes"):
        st.success("✅ Đã nạp %s · %d hình · sẵn sàng chạy" % (
            st.session_state.get("_vname", "?"), st.session_state.get("_nshapes", 0)))

with tab1:
    st.subheader("📥 Nạp video — 🖊️ Vẽ vạch (tím) / vùng (xanh lá) — rồi bấm CHẠY")
    if uploaded_file is None:
        st.info("Tải video lên ở bảng điều khiển (sidebar) để bắt đầu.")
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
                st.video(uploaded_file)

        if st.session_state.get("bg_image") is not None:
            bg_pil_grid = _bake_grid(st.session_state.bg_image)
            _ck = st.session_state.get("_canvas_seq", 0)
            if st.button("🗑️ Xoá hết & Vẽ lại"):
                st.session_state["_canvas_seq"] = _ck + 1
                st.session_state["_has_shapes"] = False
                st.rerun()

            drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"
            canvas_result = None
            if _HAS_CANVAS:
                if drawing_mode == "line":
                    canvas_result = st_canvas(
                        fill_color="rgba(0,0,0,0)",
                        stroke_width=4, stroke_color="#FF00FF",
                        background_image=_bg_uri(bg_pil_grid),
                        update_streamlit=True, height=DY, width=DX,
                        drawing_mode="line", return_image_data=True,
                        key="draw_canvas_%d" % _ck)
                else:
                    canvas_result = st_canvas(
                        fill_color="rgba(0,120,255,0.45)",
                        stroke_width=4, stroke_color="#00FF00",
                        background_image=_bg_uri(bg_pil_grid),
                        update_streamlit=True, height=DY, width=DX,
                        drawing_mode="polygon", return_image_data=True,
                        key="draw_canvas_%d" % _ck)
            else:
                st.image(bg_pil_grid)
                st.caption("(Canvas không khả dụng — chạy vẫn dùng vạch/vùng đã vẽ trước)")

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
                st.image(prev, channels="RGB", caption="📌 PREVIEW hình khi chạy")

            st.caption("💡 Vạch = kéo 1 nét · Vùng = click các góc (góc đầu để khép). "
                       "Đổi Line↔ROI không mất hình. Sửa: Undo / thùng rác trên canvas.")

if btn_run:
    if not st.session_state.get("_has_shapes"):
        st.warning("⚠️ Vẽ ít nhất **1 vạch** HOẶC **1 vùng** ở Tab 1 rồi quay lại bấm.")
    else:
        counter_obj = MultiCounter(lines=st.session_state["_lines"],
                                   roi_mask=(st.session_state["_roi_mask"] if st.session_state["_roi_mask"].any() else None))
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
        store.on_start(counter_name=perf_mode)
        st.session_state["out_path"] = _workfile("output.mp4")

        def _ai_worker(runner_fn, _m, _vin, _vout, _ctr, _st, _store, _ctl):
            try:
                runner_fn(_m, _vin, _vout, _ctr, _st, store=_store, control=_ctl)
            except Exception as e:
                _store.on_finish(ok=False, error=str(e))

        th = threading.Thread(
            target=_ai_worker,
            args=(runner, "best.pt", video_path, st.session_state["out_path"],
                  counter_obj, st.empty(), store, control),
            daemon=True)
        st.session_state.run_thread = th
        th.start()
        st.success("🚀 AI đang chạy NỀN — chuyển Tab 2 (Giám sát) hoặc Tab 4 (Chatbot) ngay, không phải chờ!")
        st.rerun()

# ---------------- TAB 2: GIÁM SÁT REAL-TIME ----------------
with tab2:
    st.subheader("📟 Giám sát trực tiếp (dữ liệu từ thread AI, không làm đơ màn hình)")
    _kpi_row()
    f1, f2 = st.columns([3, 1])
    with f1:
        fr = store.get_frame()
        if fr is not None:
            st.image(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), channels="RGB", caption="Khung mới nhất (live từ thread)")
        else:
            st.info("Chưa có khung — bấm 🚀 CHẠY ở Tab 1 (hoặc Sidebar) rồi quay lại đây.")
    with f2:
        st.markdown("**Điều khiển**")
        if not _thread_done():
            if st.button("⏯️ Tạm dừng / Tiếp tục"):
                control.toggle_pause()
            if st.button("⛔ Dừng hẳn"):
                control.request_stop()
            st.caption("Thread AI: " + ("⏸ tạm dừng" if control.paused else "▶ đang chạy"))
        else:
            s = store.stats()
            st.success("Kết thúc: %d xe · %s" % (s["total"], s["status"]))
        if st.button("🔄 Làm mới khung & KPI"):
            st.rerun()
    _mini_stats_box()
    # Bảng sự kiện mới nhất (live)
    evs = store.events[-15:][::-1]
    if evs:
        st.markdown("**Bảng sự kiện mới nhất (live)**")
        import pandas as pd
        df = pd.DataFrame(evs)[["wall", "t", "cls", "speed"]]
        df["Loai"] = [counting.BaseCounter().class_names.get(c, "?") for c in df["cls"]]
        st.dataframe(df.drop(columns=["cls"]), use_container_width=True, height=260)

# ---------------- TAB 3: THỐNG KÊ & ĐỒ THỊ ----------------
with tab3:
    st.subheader("📊 Thống kê & mật độ theo thời gian")
    _kpi_row()
    s = store.stats()
    if s["events"]:
        import pandas as pd
        bdf = pd.DataFrame({
            "phút": list(store.bins.keys()),
            **{("xe%d" % c): [store.bins[m].get(c, 0) for m in store.bins.keys()] for c in range(5)}
        })
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Mật độ xe/phút (cumulative)**")
            stack = bdf.set_index("phút")[["xe0", "xe1", "xe2", "xe3", "xe4"]]
            st.bar_chart(stack.cumsum().stack())
        with c2:
            st.markdown("**Phân bố loại xe (tổng)**")
            st.bar_chart({k: v for k, v in s["counts"].items() if v > 0})
        st.markdown("**Mật độ 5 phút gần nhất: %d xe/phút** · Trạng thái: %s · TB km/h: %.1f"
                    % (s["per_minute"], s["status"], s["avg_speed"]))
        if s["peak"]:
            st.markdown("🔺 **Đỉnh cao điểm**: phút %d–%d (%d xe — 50%% đỉnh = %d)" % (
                s["peak"]["minute"], s["peak"]["minute"] + 1, s["peak"]["total"],
                int(s["peak"]["total"] * 0.5)))
    else:
        st.info("Chưa có dữ liệu — chạy AI (Tab 1) rồi quay lại xem đồ thị. Đồ thị cập nhật LIÊN TỤC khi video chạy.")

# ---------------- TAB 4: CHATBOT ----------------
with tab4:
    st.subheader("💬 Chatbot Giao thông — hỏi trong lúc video đang chạy")
    st.caption("Mọi câu trả lời đọc **ngay từ bộ dữ liệu real-time** (không cần chờ video kết thúc). "
               "Ví dụ: *nay có bao nhiêu xe?* · *từ 2 phút sang 5 phút có bao nhiêu xe máy?* · *ùn tắc chưa?* · *xuất csv*")

    # Lịch sử chat
    for role, txt in st.session_state.chat:
        with st.chat_message("user" if role == "u" else "bot"):
            st.markdown(txt)

    if st.session_state.chat_q:
        st.session_state.chat.append(("u", st.session_state.chat_q))
        with st.spinner("Bot đang tra dữ liệu real-time..."):
            time.sleep(0.2)
        reply = trafficbot.ask(store, st.session_state.chat_q, store.stats()["video_sec"])
        st.session_state.chat.append(("b", reply))
        st.session_state.chat_q = ""
        st.rerun()

    user_q = st.chat_input("Hỏi bot về lưu lượng, mật độ, loại xe, tốc độ, xuất csv...")
    if user_q:
        st.session_state.chat_q = user_q
        st.rerun()

    if not st.session_state.chat:
        st.markdown(trafficbot.help_reply(store))

# ---------------- TAB 5: XUẤT BÁO CÁO ----------------
with tab5:
    st.subheader("📄 Xuất báo cáo & dữ liệu (CSV live)")
    s = store.stats()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**File sự kiện (ghi LIÊN TỤC khi video chạy)**: `%s`" % (s["detail_csv"] or "chưa chạy"))
        if s["detail_csv"] and os.path.exists(s["detail_csv"]):
            with open(s["detail_csv"], "rb") as fh:
                st.download_button("⬇️ Tải CSV sự kiện (live)", fh,
                                   file_name=os.path.basename(s["detail_csv"]), mime="text/csv")
        else:
            st.caption("(File sẽ xuất hiện ngay khi AI chạy — tải lại trang để thấy nội dung mới)")
    with c2:
        if st.button("📊 Tạo CSV thống kê theo phút"):
            p = store.write_summary_csv()
            st.success("Đã tạo: `%s`" % p)
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    st.download_button("⬇️ Tải CSV thống kê", fh,
                                       file_name=os.path.basename(p), mime="text/csv")
    if s["events"]:
        import pandas as pd
        st.markdown("**Xem nhanh (10 sự kiện gần nhất)**")
        evs = [dict(e, Loai=counting.BaseCounter().class_names.get(e["cls"], "?")) for e in store.events[-10:][::-1]]
        st.dataframe(pd.DataFrame(evs), use_container_width=True)
