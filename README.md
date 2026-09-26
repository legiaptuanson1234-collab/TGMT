# 🚦 Hệ thống Đếm Xe & Cảnh Báo Giao Thông bằng AI

> Nhận diện và đếm **5 loại phương tiện Việt Nam** (Ô tô, Xe máy, Xe tải, Xe bus, Xe ba gác) từ video bằng **YOLOv8 + ByteTrack**, kèm ước lượng **tốc độ xe (km/h)**, **cảnh báo ùn tắc** theo mật độ và tự động **xuất báo cáo CSV**.

Kèm theo:
- 🖥️ **Ứng dụng Web (Streamlit)** — upload video, vẽ vạch/vùng đếm trên canvas, chạy realtime.
- 💻 **Ứng dụng Desktop (CustomTkinter)** — giao diện bản địa Windows, đóng gói được bằng PyInstaller.

---

## ✨ Tính năng chính

- 🎯 Model YOLOv8 **tự train** (`best.pt`, ~6MB) nhận diện 5 lớp phương tiện Việt Nam
- 📏 2 chế độ đếm: **qua vạch (Line Crossing)** và **trong vùng (ROI Polygon)** — vẽ trực tiếp trên giao diện
- 🚫 **Không đếm trùng** nhờ Object Tracking (ByteTrack, track ID bền theo thời gian)
- ⚡ **Ước lượng tốc độ** từng xe (đổi pixel→mét, làm mượt bằng EMA)
- 🚦 **Cảnh báo ùn tắc** realtime theo mật độ xe trong khu vực, tự chụp ảnh cảnh báo
- 📄 Xuất **báo cáo CSV** (thời gian, ID xe, loại phương tiện) + video kết quả

## 🧩 Công nghệ

| Hạng mục            | Công nghệ                                   |
| ------------------- | ------------------------------------------- |
| Object Detection    | YOLOv8 (Ultralytics)                        |
| Object Tracking     | ByteTrack (`persist=True`)                  |
| Xử lý video & đồ họa| OpenCV, NumPy, Pandas                       |
| Giao diện Web       | Streamlit + streamlit-drawable-canvas       |
| Giao diện Desktop   | CustomTkinter (Tkinter)                     |
| Nền tảng            | Python 3.10+                                 |

## 📦 Cài đặt

```bash
# 1. Clone
git clone https://github.com/legiaptuanson1234-collab/TGMT.git
cd TGMT

# 2. Tạo môi trường ảo
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 3. Cài dependencies
pip install -r requirements.txt
```

> Model `best.pt` (~6MB) đã có sẵn trong repo. Nếu chạy trên server không màn hình,
> dùng `opencv-python-headless` (đã khai báo trong requirements).

---

## 🌐 Cách chạy

### 1. Ứng dụng Web (Streamlit) — khuyến nghị demo nhanh
```bash
streamlit run APP.py
```
Mở trình duyệt tại `http://localhost:8501`.

**Cách dùng:**
1. **Tải video lên** (mp4 / avi / mov)
2. **Vẽ vạch hoặc vùng đếm** trực tiếp trên canvas (chuột trái thêm điểm, chuột phải xóa)
3. **Khởi động AI** — xem realtime: bounding box, loại xe + track ID, tốc độ, tổng số xe, FPS, cảnh báo ùn tắc
4. Kết quả tự lưu vào `Bao_Cao/` (CSV) và `output.mp4` (video)

### 2. Ứng dụng Desktop (CustomTkinter)
```bash
python desktop/app_desktop.py
```
Giao diện bản địa Windows (3 bước: chọn video → vẽ → đếm). Có thể đóng gói
thành `.exe` bằng PyInstaller:

```bash
pyinstaller --windowed --onefile \
    --add-data "best.pt;." \
    --add-data "counting.py;." \
    --add-data "desktop/traffic_tracker.py;desktop" \
    desktop/app_desktop.py
```
*(trên macOS/Linux dùng `--add-data "best.pt:."`)*

---

## 🎓 Huấn luyện model từ đầu

Thư mục `train/` chứa 2 notebook để **tái tạo toàn bộ quy trình train model YOLOv8n**:

| File | Mục đích |
|---|---|
| `train/doi_nhan.ipynb` | Đổi nhãn từ class `0` thành `4` (xe ba gác) trong dataset, chuẩn bị data trước khi train |
| `train/AI.ipynb` | Huấn luyện YOLOv8n (baseline + tối ưu), đánh giá: ma trận nhầm lẫn, loss, accuracy, test thử |

```bash
# Chạy notebook (cần Jupyter + dataset YOLO đã chuẩn bị)
jupyter notebook   # Mở train/AI.ipynb -> Run All
# Sau khi train xong, lấy best.pt từ output để thay file best.pt ở gốc repo
```

> 2 notebook đã được clear output (nhẹ, không lộ path cá nhân).

---

### 3. Ứng dụng sẵn build (.exe) — Windows

Không cần cài Python: tải file zip từ tab **Releases** của repo, giải nén
thành `UTT_Traffic_AI/` rồi chạy `UTT_Traffic_AI.exe` (app + model + thư viện đi kèm).

_(Link file exe sẽ được cập nhật tại tab Releases sau khi upload)_
## 🧠 Cách hoạt động

- Mỗi frame chạy YOLOv8 → lấy bounding box + lớp xe + **track ID** (ByteTrack `persist=True`).
- **Line Crossing**: kiểm tra đoạn vị trí cũ→mới của track có cắt vạch đếm
  (CCW intersection test) — chỉ đếm **1 lần** cho mỗi track ID.
- **ROI Polygon**: kiểm tra tâm box nằm trong đa giác bằng `cv2.pointPolygonTest`.
- **Tốc độ**: `(quãng đường px × 0.065 m/px) / thời gian`, làm mượt bằng EMA
  (khối lượng 0.8/0.2).
- **Cảnh báo ùn tắc**: đếm xe đang trong khu vực, vượt ngưỡng (Line=28, ROI=10)
  → đổi nhãn "CANH BAO: UN TAC!" + tự chụp ảnh cảnh báo mỗi 5 giây.

## 📊 Dữ liệu đầu ra (CSV)

| Thoi Gian Ghi Nhan      | ID Xe | Loai Phuong Tien |
| ----------------------- | ----- | ---------------- |
| 2026-07-01 13:10:09     | 8     | Xe May           |
| 2026-07-01 13:10:11     | 1     | O To             |
| 2026-07-01 13:10:11     | 2     | Xe Tai           |

*(Xem file mẫu: [`outputs/ThongKe_LuuLuong_sample.csv`](outputs/ThongKe_LuuLuong_sample.csv))*

## 🗂️ Cấu trúc dự án

```
TGMT/
├── APP.py                  # Giao diện web (Streamlit)
├── counting.py             # LOGIC ĐẾM CHUNG: Line & ROI + VehicleCounter + xuất CSV
├── tracking.py             # YOLOv8 + ByteTrack (bản WEB, hàm run_ai_system)
├── best.pt                 # Model đã train (~6MB)
├── requirements.txt
├── desktop/
│   ├── app_desktop.py      # Giao diện desktop (CustomTkinter) — bài đếm xe
│   └── traffic_tracker.py  # class TrafficTracker (bản DESKTOP: tốc độ, cảnh báo, vẽ)
└── outputs/
    └── ThongKe_LuuLuong_sample.csv   # Báo cáo mẫu
```

> ℹ️ **Chia sẻ module để tránh trùng code:**
> - `counting.py` (root) là **nguồn duy nhất** cho logic đếm (`LineCounter`, `PolygonCounter`, `VehicleCounter`, xuất CSV) — cả web lẫn desktop đều import từ đây.
> - `tracking.py` (web, hàm `run_ai_system`) và `desktop/traffic_tracker.py` (class `TrafficTracker`) là **2 bản riêng** vì web chạy 1 lần trong Streamlit còn desktop chạy vòng lặp realtime bằng `after(10ms)`; desktop import chung `counting.py` nên không còn bản đếm trùng.

## 📺 Video mẫu để thử

Thư mục `media/` có `sample_input.mp4` (video đưồng đưỗi có xe) để bạn thử app ngay, không cần tìm video riêng:

```
media/sample_input.mp4   # chọn vào video này khi chạy app (web hoặc desktop)
```

## 📄 License

MIT
