import streamlit as st
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np

# الموديل
@st.cache_resource
def load_model():
    return YOLO(r"best.pt")

model = load_model()

st.set_page_config(page_title="Ovarian Cyst Detection", page_icon="🩺", layout="centered")

st.title("Ovarian Cyst Detection Application")

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    conf = st.slider("Confidence", 0.01, 1.0, 0.10, 0.01)

    if st.button("🔍 Detect"):
        with st.spinner("Analyzing image... Please wait"):
            # Ultralytics يفضل صورة PIL مباشرة
            results = model(image, conf=conf)

            # إذا مافي نتائج
            if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
                st.warning("⚠ No cyst detected in the image.")
            else:
                # عرض صورة النتيجة
                result_image = results[0].plot()

                # غالبًا plot يرجع بصيغة BGR، لذلك نحول لـ RGB للعرض الصحيح
                # لو شغّلك السابق ممتاز غالبًا هذا لن يضر
                result_image = result_image[..., ::-1]

                st.image(result_image, caption="📌 Detection Result", use_container_width=True)

                # تفاصيل الاكتشافات
                boxes = results[0].boxes
                st.success(f"✅ Detection completed! {len(boxes)} detection(s).")

                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item())
                    confidence = float(boxes.conf[i].item())
                    class_name = model.names.get(class_id, str(class_id))

                    st.write(f"**Detection {i+1}:** {class_name} — {confidence:.2f}")

                # (اختياري) تحميل الصورة الناتجة
                img_pil = Image.fromarray(result_image)
                buf = io.BytesIO()
                img_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="⬇ Download Result Image",
                    data=byte_im,
                    file_name="detection_result.png",
                    mime="image/png"
                )
