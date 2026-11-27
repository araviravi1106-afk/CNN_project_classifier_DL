
import streamlit as st
from PIL import Image

from src.config import MODEL_PATH
from src.predict import predict_image

st.set_page_config(page_title="Simple CNN Classifier", page_icon="🖼️")
st.title("CIFAR-10 Image Classifier")
st.write("Upload an image of an airplane, car, bird, cat, deer, dog, frog, "
         "horse, ship or truck.")

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python -m src.train` first.")
    st.stop()

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded image", width=300)
    results = predict_image(image)
    st.subheader(f"Prediction: {results[0][0]} ({results[0][1] * 100:.1f}%)")
    for name, p in results:
        st.progress(p, text=f"{name}: {p * 100:.1f}%")
