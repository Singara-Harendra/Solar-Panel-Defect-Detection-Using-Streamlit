import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ====== Configuration ======
st.set_page_config(page_title="Solar Panel Defect Detection", layout="centered")
CLASS_NAMES = ['Clean', 'Electrical-damage', 'Physical-Damage', 'Bird-drop', 'Dusty', 'Snow-Covered']
IMG_SIZE = (224, 224)  # must match the size used during training
MODEL_PATH = "best_model.h5"

# ====== Load Model ======
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# ====== Streamlit UI ======

st.title("🔍 Solar Panel Defect Detection System")
st.write("Upload a solar panel image to detect surface defects like cracks, dust, snow, etc.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("🔍 Analyzing image..."):
        prediction = model.predict(img_batch)[0]

    sorted_indices = np.argsort(prediction)[::-1]
    sorted_classes = [CLASS_NAMES[i] for i in sorted_indices]
    sorted_probs = [prediction[i] for i in sorted_indices]

    # Display Results
    st.markdown("## 📊 Prediction Results")
    for i, (cls, prob) in enumerate(zip(sorted_classes, sorted_probs)):
        st.markdown(f"**{i+1}. {cls}**: {float(prob)*100:.2f}%")
        st.progress(int(float(prob) * 100))  # Convert to 0–100 int

        

    st.success(f"🚨 **Detected Defect:** {sorted_classes[0]}  \n🎯 **Confidence:** {sorted_probs[0]*100:.2f}%")

    # Probability Chart
    st.markdown("### 📈 Probability Distribution")
    fig, ax = plt.subplots()
    ax.barh(sorted_classes[::-1], [p * 100 for p in sorted_probs[::-1]], color='#4CAF50')
    ax.set_xlabel("Probability (%)")
    st.pyplot(fig)
