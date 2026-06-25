import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import uuid

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_system
from counting import VehicleCounter 

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")
st.title("🚦 HỆ THỐNG ĐẾM XE & CẢNH BÁO GIAO THÔNG AI")
st.markdown("**Đồ án Kỹ thuật - Sinh viên: Lê Giáp Tuấn Sơn - UTT**")

# --- HÀM TRÍCH XUẤT FRAME ĐẦU TIÊN (CÓ CACHE CHỐNG XUNG ĐỘT VÀ LỖI VỠ FILE) ---
@st.cache_data
def get_first_frame(file_bytes):
    unique_id = uuid.uuid4().hex
    temp_filename = f"temp_frame_{unique_id}.mp4"
    
    # Ghi file tạm duy nhất cho phiên này
    with open(temp_filename, "wb") as f:
        f.write(file_bytes)
        
    cap = cv2.VideoCapture(temp_filename)
    ret, frame = cap.read()
    
    # CHỐNG LỖI: Nếu frame đầu lỗi, thử đọc thêm vài frame tiếp theo
    if not ret:
        for _ in range(5):
            ret, frame = cap.read()
            if ret: break
    cap.release()
    
    # Xóa file tạm ngay sau khi lấy được ảnh để tránh rác ổ cứng server
    try:
        os.remove(temp_filename)
    except:
        pass
        
    return frame if ret else None

# --- 2. BẢNG ĐIỀU KHIỂN (MENU TRÁI) ---
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("1. Tải Video Lên", type=['mp4', 'avi', 'mov'])
    mode = st.radio("2. Chế Độ Phân Tích", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    btn_run = st.button("🚀 KHỞI ĐỘNG AI", type="primary", use_container_width=True)

# --- 3. XỬ LÝ VIDEO & BẢNG VẼ CANVAS ---
if uploaded_file is not None:
    # Lấy bytes của file để truyền vào hàm cache
    file_bytes = uploaded_file.getvalue()
    
    # Gọi hàm bộ nhớ đệm lấy frame đầu tiên - HOÀN TOÀN KHÔNG BỊ GHI ĐÈ KHI VẼ
    frame = get_first_frame(file_bytes)

    if frame is not None:
        # Streamlit dùng hệ màu RGB, OpenCV dùng BGR nên phải chuyển đổi
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(frame_rgb)

        st.subheader("Bước 1: Vẽ Vạch/Vùng Cảnh Báo")
        st.markdown(f"**Chế độ hiện tại:** {mode}. Hãy dùng chuột click và vẽ trực tiếp lên ảnh dưới đây.")
        
        drawing_mode = "line" if mode == "Đếm Vạch (Line)" else "polygon"
        
        # CÔNG CỤ VẼ TRỰC TIẾP TRÊN WEB
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)", # Màu nền vùng ROI (trong suốt)
            stroke_width=3,
            stroke_color="#00FF00", # Màu vạch line xanh lá
            background_image=image_pil,
            update_streamlit=True,
            height=image_pil.height,
            width=image_pil.width,
            drawing_mode=drawing_mode,
            key="canvas",
        )

        # --- 4. KÍCH HOẠT YOLOv8 ---
        if btn_run:
            st.markdown("---")
            st.subheader("📟 Màn Hình Giám Sát Real-time")
            
            # Kiểm tra xem người dùng đã vẽ gì chưa
            if canvas_result.json_data is not None and len(canvas_result.json_data["objects"]) > 0:
                objects = canvas_result.json_data["objects"]
                st.success("[✔] Đã nhận diện được tọa độ. Đang khởi động AI...")
                
                # TẠO MỘT CỬA SỔ ẢO TRÊN WEB ĐỂ NHẬN VIDEO TỪ YOLO
                stframe = st.empty() 
                
                # --- TRÍCH XUẤT TỌA ĐỘ TỪ CANVAS ---
                points = []
                if drawing_mode == "line":
                    obj = objects[0]
                    left, top = obj["left"], obj["top"]
                    x1, y1 = int(left + obj["x1"]), int(top + obj["y1"])
                    x2, y2 = int(left + obj["x2"]), int(top + obj["y2"])
                    points = [(x1, y1), (x2, y2)]
                else:
                    obj = objects[0]
                    for pt in obj["path"]:
                        if len(pt) >= 3:
                            points.append((int(pt[1]), int(pt[2])))

                # --- GHI VIDEO THẬT RA Ổ CỨNG MỘT LẦN DUY NHẤT TRƯỚC KHI AI CHẠY ---
                video_path = "video_tam.mp4"
                with open(video_path, "wb") as f:
                    f.write(file_bytes)

                # Khởi tạo Object đếm, truyền chế độ và tọa độ vẽ trên Web vào cho nó
                counter_obj = VehicleCounter(mode=drawing_mode, points=points) 
                
                try:
                    # Truyền stframe vào để tracking.py bơm video ra
                    run_ai_system("best.pt", video_path, "output.mp4", counter_obj, stframe)
                    st.balloons()
                    st.success("🎉 Luồng phân tích giao thông đã hoàn tất!")
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán: {e}")
                
            else:
                st.error("❌ BẠN CHƯA VẼ VẠCH HAY VÙNG ROI! Vui lòng dùng chuột vẽ lên hình trước khi bấm Khởi Động AI.")