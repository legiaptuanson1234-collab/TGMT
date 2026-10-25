# 🚦 UTT Traffic AI — Hệ thống Đếm Xe & Cảnh báo Giao thông

Nhận diện và đếm **5 loại phương tiện Việt Nam** (Ô tô, Xe máy, Xe tải, Xe bus, Xe ba gác)
từ video bằng **YOLOv8 + ByteTrack**. Ước lượng **tốc độ xe (km/h)**, **cảnh báo ùn tắc**
theo mật độ, xuất **báo cáo CSV + video kết quả**.

> **Mới (v9):** Giao diện **Web (Streamlit)** và **Desktop (CustomTkinter .exe)**
> chạy **y hệt nhau** — Bảng điều khiển + Video lớn + KPI real-time +
> **CHATBOT giao thông cập nhật liên tục** (hỏi trong lúc video đang chạy, không chờ kết thúc)
> + **CSV live** (ghi liên tục, mở Excel được ngay) + nút Tạm dừng / Dừng.

**Hai giao diện dùng chung một "bộ não" AI:**
- 🖥️ **Desktop (Windows .exe / CustomTkinter)** — chạy trực tiếp trên máy, không cần trình duyệt
- 🌐 **Web (Streamlit)** — chạy trên trình duyệt, hoặc bản demo deploy:
  `https://legiaptuanson1234-collab-tgmt.streamlit.app` *(tự cập nhật khi có commit mới)*

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🎯 Nhận diện 5 loại xe | YOLOv8n tự train, model chỉ ~6MB (`best.pt`) |
| 🖊️ 2 chế độ đếm | **Qua vạch (Line)** và **Trong vùng (ROI)** — vẽ trực tiếp trên khung hình |
| 🔁 Không đếm trùng | ByteTrack gắn **track ID bền**, mỗi xe chỉ đếm 1 lần |
| ⚡ Tốc độ xe | Ước lượng km/h (hệ số pixel→mét = 0.065, làm mượt EMA α=0.2, ≥15 khung) |
| 🚦 Cảnh báo ùn tắc | Vượt ngưỡng mật độ → cảnh báo real-time + tự chụp ảnh bằng chứng |
| 💬 **Chatbot giao thông (v9)** | Hỏi **"nay có bao nhiêu xe?", "từ 2 phút sang 5 phút có mấy xe máy?", "ùn tắc chưa?", "tốc độ trung bình?", "xuất csv"** — trả lời **cập nhật liên tục** từ dữ liệu real-time, KHÔNG cần chờ video kết thúc |
| 📡 **CSV live (v9)** | File sự kiện **ghi liên tục** (mỗi xe đếm được), mở Excel xem được ngay trong lúc video chạy + CSV thống kê theo phút |
| ⏯️ **Điều khiển (v9)** | Nút **Tạm dừng / Tiếp tục / Dừng & Lưu** — an toàn giữa thread AI và giao diện |
| 📊 Báo cáo | CSV (thời gian, ID xe, loại xe) + video ghi hình kết quả |

## 🧠 Chatbot giao thông (v9)

Bộ chatbot **rule-based, tiếng Việt** (`trafficbot.py`) đọc trực tiếp `TrafficStore`
(dữ liệu thread-safe, ghi bởi thread AI nên giao diện không bị đơ). Trả lời được:
- **Tổng lượng xe** + phân loại % (5 loại) + điểm cao nhất
- **Thống kê theo khoảng thời gian** ("từ X phút sang Y phút có bao nhiêu xe [loại]")
- **Mật độ / ùn tắc** (trạng thái 🟢 VẮNG → 🔴 ÙN TẮC + khuyến nghị)
- **Tốc độ trung bình** (thang pixel→mét của đề tài)
- **Xuất CSV** (liên tục / thống kê theo phút)

## 📦 Cấu trúc repo

```
TGMT/
├─ best.pt                     # Model YOLOv8 đã train (~6MB)
├─ bytetrack.yaml              # Cấu hình ByteTrack
├─ counting.py                 # LOGIC ĐẾM CHUNG: Line, ROI, Multi, xuất CSV
├─ tracking.py                 # Xử lý AI bản WEB (hàm run_ai_system, 3 mức hiệu năng)
├─ APP.py                      # Giao diện WEB (Streamlit) — v9 (giống desktop)
├─ livestore.py               # [v9] TrafficStore + RunControl: dữ liệu real-time (thread-safe), CSV live
├─ trafficbot.py              # [v9] Chatbot giao thông (rule-based, tiếng Việt)
├─ requirements.txt            # Dependencies (chạy từ máy nguồn)
├─ desktop/
│  ├─ app_desktop.py          # Giao diện DESKTOP (CustomTkinter) — v9 (giống web)
│  └─ traffic_tracker.py      # Bộ xử lý AI bản DESKTOP (class TrafficTracker)
├─ train/
│  ├─ doi_nhan.ipynb          # Gán nhãn xe ba gác (class 0→4)
│  └─ AI.ipynb               # Train + đánh giá YOLOv8n
├─ media/
│  └─ sample_input.mp4        # Video mẫu (~34MB) để chạy thử ngay
└─ outputs/
   └─ ThongKe_LuuLuong_sample.csv   # Báo cáo CSV mẫu
```

---

## 🚀 Chọn cách chạy

| Cách | Phù hợp khi | Cần có |
|---|---|---|
| **1. Bản .exe (Windows)** — KHUYẾN NGHỊ | Dùng nhanh, không muốn cài gì | Chỉ cần giải nén + chạy |
| **2. Chạy từ mã nguồn (Python)** | Dev, sửa code, train lại | Python 3.10–3.12 + requirements |
| **3. Streamlit Cloud** | Demo công khai, không cài máy | Repo GitHub + Streamlit Cloud |

### Cách 1 — Bản .exe sẵn (Windows)
Ứng dụng đóng gói bằng **PyInstaller** (bản v9, có chatbot): **chỉ cần tải, giải nén, chạy — không cần cài Python**.

> ⚠️ Bản mới có chatbot + CSV live (v9). Giải nén → chạy `UTT_Traffic_Copilot.exe`
> (lần đầu ~30s load YOLO). **Tải file .exe/build mới nhất từ người hướng dẫn (Sơn)**.

### Cách 2 — Chạy từ mã nguồn (desktop)
```bash
python -m venv venv
venv\Scripts\pip install -r requirements.txt
python desktop\app_desktop.py
```

### Cách 3 — Chạy bản web
```bash
streamlit run APP.py
```

## 🔧 Hướng dẫn dùng (v9 — chung cho web & desktop)
1. **Chọn video** (hoặc tải video lên, web: bảng điều khiển; desktop: nút "Chọn video")
2. **Chọn chế độ đếm**: qua vạch (Line) / trong vùng (ROI)
3. **Vẽ** vạch (1 nét tím) hoặc vùng (click các góc)
4. **Bấm ▶️ CHẠY** — video chạy, KPI + CSV cập nhật liên tục
5. **Hỏi CHATBOT** bất kỳ lúc nào (video đang chạy vẫn hỏi được, không chờ kết thúc)
6. **Tạm dừng / Dừng & Lưu** để xuất CSV bất kỳ thời điểm

## 📄 Ghi chú kỹ thuật
- Tốc độ: `speed = ( quãng_đường_m / Δt ) × 3.6 (km/h)`, `quãng_đường_m = số_pixel × 0.065`
- Làm mượt tốc độ bằng EMA: `v_new = 0.8·v_old + 0.2·v_tho` (đo khi quỹ đạo ≥15 khung)
- 3 mức hiệu năng: **Nhanh** (CPU 640×360), **Siêu nhanh** (CPU 480×270, demo Cloud),
  **Chất lượng** (GPU 1280×720)
