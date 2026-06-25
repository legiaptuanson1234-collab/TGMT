import streamlit as st
import cv2
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# --- NHẬP CÁC MODULE AI CỦA BẠN ---
from tracking import run_ai_system
from counting import VehicleCounter 

# --- 1. CẤU HÌNH TRANG WEB ---
st.set_page_config(page_title="Traffic AI - UTT", layout="wide", page_icon="🚦")
st.title("🚦 HỆ THỐNG ĐẾM XE & CẢNH BÁO GIAO THÔNG AI")
st.markdown("**Đồ án Kỹ thuật - Sinh viên: Lê Giáp Tuấn Sơn - UTT**")

# --- 2. BẢNG ĐIỀU KHIỂN ---
with st.sidebar:
    st.header("⚙️ BẢNG ĐIỀU KHIỂN")
    st.markdown("---")
    
    uploaded_file = st.file_uploader("1. Tải Video Lên", type=['mp4', 'avi', 'mov'])
    mode = st.radio("2. Chế Độ Phân Tích", ["Đếm Vạch (Line)", "Đếm Vùng (ROI)"])
    btn_run = st.button("🚀 KHỞI ĐỘNG AI", type="primary", use_container_width=True)

# --- 3. XỬ LÝ VIDEO & BẢNG VẼ CANVAS (SIÊU MƯỢT) ---
if uploaded_file is not None:
    video_path = "video_tam.mp4"
    
    # CHỈ XỬ LÝ FILE KHI LÀ VIDEO MỚI (CHỐNG GIẬT LAG KHI CLICK CHUỘT)
    if "last_filename" not in st.session_state or st.session_state.last_filename != uploaded_file.name:
        st.session_state.last_filename = uploaded_file.name
        
        # 1. Lưu video xuống ổ cứng 1 lần duy nhất
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getvalue())
            
        # 2. Trích xuất ảnh
        cap = cv2.VideoCapture(video_path)
        
        # Nhảy đến frame thứ 50 để né mọi màn hình trắng/đen đầu video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 50:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
            
        ret, frame = cap.read()
        cap.release()
        
        # 3. Chuyển hệ màu và lưu thẳng vào RAM ảo
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.session_state.bg_image = Image.fromarray(frame_rgb)
        else:
            st.session_state.bg_image = None

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
                st.success("[✔] Đã nhận diện được tọa độ. Đang khởi động AI...")
                
                stframe = st.empty() 
                
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

                counter_obj = VehicleCounter(mode=drawing_mode, points=points) 
                
                try:
                    # Video đã có sẵn trên đĩa từ bước trên, gọi thẳng ra chạy
                    run_ai_system("best.pt", video_path, "output.mp4", counter_obj, stframe)
                    st.balloons()
                    st.success("🎉 Luồng phân tích giao thông đã hoàn tất!")
                except Exception as e:
                    st.error(f"❌ Có lỗi xảy ra trong quá trình tính toán: {e}")
            else:
                st.error("❌ BẠN CHƯA VẼ VẠCH HAY VÙNG ROI! Vui lòng dùng chuột vẽ lên hình trước khi bấm Khởi Động AI.")