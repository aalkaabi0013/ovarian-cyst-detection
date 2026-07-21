import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# ????? ???????
model = YOLO("C:/Users/abrar/Downloads/APP/best.pt")
st.title("Ovarian Cyst Detection Application")

uploaded_file = st.file_uploader("Upload an ultrasound image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Detect"):
        results = model(image)

        # ??? ?????? ??? ??????
        result_image = results[0].plot()
        st.image(result_image, caption="Detection Result", use_column_width=True)

        st.success("Detection completed successfully!")