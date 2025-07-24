import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ====== App Config ======
st.set_page_config(page_title="Solar Panel Defect Detector", layout="centered")

@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("model.keras")  # or "model.h5"
    except:
        input_layer = tf.keras.layers.Input(shape=(244, 244, 3))
        return tf.keras.models.Model(
            inputs=input_layer,
            outputs=tf.keras.layers.TFSMLayer("model_save", call_endpoint='serving_default')(input_layer)
        )

# ====== Constants ======
IMG_SIZE = (244, 244)
CLASS_NAMES = ['Clean', 'Discoloration', 'Crack', 'Bird Drop', 'Dust']
model = load_model()

# ====== UI ======
st.title("🔍 Solar Panel Defect Detection System")
st.write("Upload a solar panel image to detect possible defects.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # ====== Image Processing ======
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)
    
    img_array = np.array(image.resize(IMG_SIZE)) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # ====== Prediction ======
    with st.spinner('Analyzing image...'):
        raw_prediction = model.predict(img_batch)
    
    if isinstance(raw_prediction, dict):
        probabilities = list(raw_prediction.values())[0][0]
    else:
        probabilities = raw_prediction[0]

    if len(probabilities) != len(CLASS_NAMES):
        probabilities = probabilities[:len(CLASS_NAMES)]

    # Sort probabilities from highest to lowest
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = [probabilities[i] for i in sorted_indices]
    sorted_classes = [CLASS_NAMES[i] for i in sorted_indices]
    predicted_index = sorted_indices[0]

    # ====== Results Display ======
    st.markdown("## 📊 Prediction Results")
    
    # 1. All Probabilities (sorted high to low)
    st.markdown("### 🔢 Probability Ranking")
    for i, (name, prob) in enumerate(zip(sorted_classes, sorted_probs)):
        # Highlight the highest probability
        if i == 0:
            st.markdown(
                f"""<div style='background-color:#4CAF50;padding:10px;
                    border-radius:5px;color:white;margin-bottom:10px'>
                    <b>1. {name}</b>: {prob*100:.1f}%
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**{i+1}. {name}**: {prob*100:.1f}%")
        st.progress(float(prob))

    # 2. Final Defect Announcement
    st.markdown("---")
    st.success(f"""
    🚨 **Final Detection Result**  
    **Defect Identified**: {sorted_classes[0]}  
    **Confidence Level**: {sorted_probs[0]*100:.1f}%
    """)

    # 3. Visual Chart
    st.markdown("### 📈 Probability Distribution")
    fig, ax = plt.subplots()
    bars = ax.barh(sorted_classes, np.array(sorted_probs) * 100,
                  color=['#4CAF50' if i == 0 else '#2196F3' 
                         for i in range(len(sorted_classes))])
    ax.set_xlim(0, 100)
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
    st.pyplot(fig)