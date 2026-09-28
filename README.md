# 🚦 UTT Traffic AI — Hệ thống Đếm Xe & Cảnh Báo Giao Thông

> Nhận diện và đếm **5 loại phương tiện Việt Nam** (Ô tô, Xe máy, Xe tải, Xe bus, Xe ba gác)
> từ video bằng **YOLOv8 + ByteTrack**. Ước lượng **tốc độ xe (km/h)**, **cảnh báo ùn tắc**
> theo mật độ, xuất **báo cáo CSV** + video kết quả.

**Hai giao diện dùng chung một "bộ não" AI:**
- 💻 **Desktop (Windows .exe / CustomTkinter)** — chạy trực tiếp trên máy, không cần trình duyệt
- 🖥️ **Web (Streamlit)** — chạy trên trình duyệt, hoặc dùng bản demo đã deploy: `_________________` *(link Streamlit Cloud sẽ được cập nhật sau)*

---

## ✨ Tính năng

| Tính năng | Mô tả |
|---|---|
| 🎯 Nhận diện 5 loại xe | YOLOv8n tự train, model chỉ ~6MB (`best.pt`) |
| 📏 2 chế độ đếm | **Qua vạch (Line)** và **Trong vùng (ROI)** — vẽ trực tiếp trên khung hình |
| 🚫 Không đếm trùng | ByteTrack gán **track ID bền**, mỗi xe chỉ đếm 1 lần |
| ⚡ Tốc độ xe | Ước lượng km/h (pixel→mét + làm mượt EMA) |
| 🚦 Cảnh báo ùn tắc | Vượt ngưỡng mật độ → cảnh báo realtime + tự chụp ảnh bằng chứng |
| 📄 Báo cáo | CSV (thời gian, ID xe, loại xe) + video ghi hình kết quả |

## 🧩 Công nghệ

| Hạng mục | Công nghệ |
|---|---|
| Object Detection | YOLOv8n (Ultralytics) |
| Object Tracking | ByteTrack (`persist=True`) |
| Xử lý video & đồ họa | OpenCV, NumPy |
| Giao diện Desktop | CustomTkinter (Windows) |
| Giao diện Web | Streamlit + streamlit-drawable-canvas |
| Đóng gói Windows | PyInstaller (bản .exe sẵn, xem **Releases**) |

## 🗂️ Cấu trúc repo

```
TGMT/
├── best.pt                          # Model YOLOv8 đã train (~6MB)
├── counting.py                      # LOGIC ĐẾM CHUNG: Line, ROI, xuất CSV
├── tracking.py                      # Bộ xử lý AI bản WEB (hàm run_ai_system)
├── APP.py                           # Giao diện WEB (Streamlit)
├── requirements.txt                 # Dependencies (chạy từ mã nguồn)
├── desktop/
│   ├── app_desktop.py               # Giao diện DESKTOP (CustomTkinter)
│   └── traffic_tracker.py           # Bộ xử lý AI bản DESKTOP (class TrafficTracker)
├── train/
│   ├── doi_nhan.ipynb               # Đổi nhãn xe ba gác (class 0 → 4)
│   └── AI.ipynb                     # Train + đánh giá YOLOv8n từ đầu
├── media/
│   └── sample_input.mp4             # Video mẫu (34MB) để chạy thử ngay
└── outputs/
    └── ThongKe_LuuLuong_sample.csv  # Báo cáo CSV mẫu
```

---

# 🚀 CHỌN CÁCH CHẠY

| Cách | Phù hợp khi | Cần gì |
|---|---|---|
| **1. Bản .exe (Windows)** ⭐ khuyến nghị | Dùng nhanh, không muốn cài gì | Không cần Python |
| **2. Desktop từ mã nguồn** | Dev muốn chỉnh sửa code | Python 3.10+ |
| **3. Web (Streamlit)** | Demo, chạy trên server/máy khác | Python 3.10+ |

---

## Cách 1: Bản .exe sẵn build (Windows) ⭐

App đóng gói bằng PyInstaller: **chỉ cần tải về, giải nén, chạy — không cần cài Python**.

### Bước 1 — Tải file từ tab **Releases** của repo

Vì bộ app (~5GB sau nén thành 3 phần, do chứa thư viện PyTorch/TensorFlow),
file được chia 3 phần:

```
UTT_Traffic_AI_windows_part1.zip
UTT_Traffic_AI_windows_part2.zip
UTT_Traffic_AI_windows_part3.zip
```

> ⚠️ Tải **đủ cả 3** và để cùng một thư mục.

### Bước 2 — Ghép 3 phần lại thành 1 file zip

Mở **CMD** trong thư mục chứa 3 file (Shift + Chuột phải → Open in terminal), chạy:

```bat
copy /b UTT_Traffic_AI_windows_part1.zip + UTT_Traffic_AI_windows_part2.zip + UTT_Traffic_AI_windows_part3.zip UTT_Traffic_AI_windows.zip
```

> Hoặc trên **PowerShell**:
> ```powershell
> $out=[System.IO.File]::Create('UTT_Traffic_AI_windows.zip')
> foreach($f in 'part1','part2','part3'){ $b=[System.IO.File]::ReadAllBytes("UTT_Traffic_AI_windows_$f.zip"); $out.Write($b,0,$b.Length) }
> $out.Close()
> ```

Kết quả: 1 file `UTT_Traffic_AI_windows.zip` duy nhất.

### Bước 3 — Giải nén & chạy

1. Giải nén `UTT_Traffic_AI_windows.zip` (7-Zip / WinRAR / Win11 tích hợp)
   → ra thư mục `UTT_Traffic_AI/`
2. Double-click **`UTT_Traffic_AI.exe`**

```
UTT_Traffic_AI/
├── UTT_Traffic_AI.exe   ← chạy file này
├── _internal/           ← thư viện (không xóa, không mở)
├── best.pt              ← model (đi kèm sẵn)
└── Bao_Cao/ Video_Xuat/ ← nơi lưu kết quả (tự tạo khi chạy)
```

### Bước 4 — Chạy thử với video mẫu

Chọn video `media/sample_input.mp4` (có sẵn trong repo này) để thử ngay.

---

## Cách 2: Chạy Desktop từ mã nguồn

```bash
# 1. Clone repo
git clone https://github.com/legiaptuanson1234-collab/TGMT.git
cd TGMT

# 2. Tạo môi trường ảo & cài dependencies (Python 3.10+)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # macOS/Linux
pip install -r requirements.txt

# 3. Chạy app desktop
python desktop/app_desktop.py
```

> Trên máy có màn hình (Windows/macOS/Linux) hãy dùng `opencv-python` thay cho
> `opencv-python-headless` (bản headless không hiển thị được cửa sổ OpenCV):
> `pip install opencv-python`

## Cách 3: Chạy Web (Streamlit)

```bash
streamlit run APP.py
# Mở trình duyệt: http://localhost:8501
```

**Bản demo online:** `_________________` *(link Streamlit Cloud sẽ được cập nhật sau)*

### Deploy bản web lên Streamlit Cloud (miễn phí)

1. Vào https://share.streamlit.io → **Deploy an app** → chọn repo **`TGMT`**, branch **`main`**, main file **`APP.py`**
2. **Python version: `3.10`** · **requirements file: `requirements.txt`** (Cloud tự cài PyTorch CPU — không cần GPU)
3. Chờ ~2–3 phút build. App chạy ở link `https://...streamlit.app` → dán link đó vào 2 chỗ `_________________` trong README này

> Code đã được vá để chạy an toàn trên Cloud (ghi file ra thư mục tạm, không lỗi với đĩa read-only của Streamlit Cloud).

---

# 🖊️ CÁCH DÙNG APP (chung cho cả 3 cách)

Quy trình 3 bước trong giao diện:

1. **Chọn video** — (bước 1) chọn file video, hoặc dùng `media/sample_input.mp4`.
2. **Vẽ vạch / vùng đếm** — (bước 2)
   - Chọn chế độ **Line** (vạch đếm) hoặc **ROI** (vùng đếm)
   - Cửa sổ vẽ hiện ra: **chuột trái** = thêm điểm, **chuột phải** = xóa điểm vừa vẽ,
     **Enter** = hoàn tất
   - Line cần **2 điểm**; ROI cần **≥ 3 điểm** (vẽ bao quanh khu vực)
3. **Bắt đầu đếm** — (bước 3) chạy realtime:
   - Bounding box + loại xe + **track ID** của từng xe
   - Tốc độ ước lượng, tổng số xe theo loại, **FPS**
   - Nhãn cảnh báo **"CẢNH BÁO: ÙN TẮC"** khi vượt ngưỡng
4. **Kết quả tự lưu:**
   - CSV: thư mục `Bao_Cao/` (desktop) — cột: thời gian, ID xe, loại phương tiện
   - Video ghi hình: thư mục `Video_Xuat/` (desktop)

---

## 🧠 Cách hoạt động (tóm tắt kỹ thuật)

- Mỗi frame chạy YOLOv8 → bounding box + lớp xe + **track ID** (ByteTrack `persist=True`)
- **Line Crossing**: kiểm tra quỹ đạo (đoạn từ vị trí cũ → mới) của track có cắt vạch đếm
  (thử nghiệm giao đoạn thẳng CCW) → mỗi track chỉ đếm **1 lần** khi đi qua
- **ROI**: kiểm tra tâm box nằm trong đa giác bằng `cv2.pointPolygonTest`
- **Tốc độ**: `(độ dài quỹ đạo px × 0.065 m/px) ÷ thời gian`, làm mượt bằng EMA (0.8/0.2)
- **Ùn tắc**: đếm xe đang trong khu vực, vượt ngưỡng (Line ≥ 28, ROI ≥ 10) → cảnh báo
  + tự chụp ảnh bằng chứng mỗi 5 giây

## 📊 Dữ liệu đầu ra (CSV)

| Thoi Gian Ghi Nhan | ID Xe | Loai Phuong Tien |
|---|---|---|
| 2026-07-01 13:10:09 | 8 | Xe May |
| 2026-07-01 13:10:11 | 1 | O To |

File mẫu: [`outputs/ThongKe_LuuLuong_sample.csv`](outputs/ThongKe_LuuLuong_sample.csv)

---

# 🎓 Huấn luyện lại model (tuỳ chọn)

Thư mục `train/` chứa 2 notebook để tái tạo toàn bộ quy trình:

| File | Mục đích |
|---|---|
| `train/doi_nhan.ipynb` | Đổi nhãn class `0` → `4` (xe ba gác) trong dataset trước khi train |
| `train/AI.ipynb` | Train YOLOv8n (baseline + tối ưu) + đánh giá: ma trận nhầm lẫn, loss, accuracy, test thử |

```bash
pip install jupyter
jupyter notebook
# Mở train/doi_nhan.ipynb (chạy trước, sửa path dataset trong cell đầu),
# rồi train/AI.ipynb → Run All
# Sau khi train xong, thay best.pt ở gốc repo bằng file vừa train
```

---

# 🔨 Tự build .exe (tuỳ chọn, cho dev)

```bash
pip install pyinstaller

pyinstaller --windowed --name UTT_Traffic_AI ^
    --add-data "best.pt;." ^
    --add-data "counting.py;." ^
    --add-data "desktop\traffic_tracker.py;desktop" ^
    desktop/app_desktop.py
# Kết quả: dist/UTT_Traffic_AI/ (exe + _internal) -> copy nguyên thư mục này cho người dùng
```
*(macOS/Linux: thay `;` bằng `:` trong `--add-data`)*

---

## 📄 License

MIT
