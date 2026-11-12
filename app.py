
import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AgriSathi AI",
    page_icon="🌾",
    layout="wide"
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "agrisathi_mobilenetv2.keras")
CLASS_PATH = os.path.join(os.path.dirname(__file__), "artifacts", "class_names.json")

# LOAD MODEL
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_classes():
    with open(CLASS_PATH, "r") as file:
        return json.load(file)

model = load_model()
class_names = load_classes()

# CUSTOM STYLING
st.markdown("""
<style>
.main {
    background-color: #f7faf7;
}
.hero {
    padding: 28px;
    border-radius: 15px;
    background: linear-gradient(120deg, #14532d, #15803d);
    color: white;
    margin-bottom: 20px;
}
.hero h1 {
    color: white;
    margin-bottom: 5px;
}
.hero p {
    color: #e5f5e9;
    font-size: 17px;
}
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="hero">
    <h1>🌾 AgriSathi AI</h1>
    <p>AI-powered rice leaf disease screening</p>
    <p>Supporting smarter crop-health assessment through computer vision.</p>
</div>
""", unsafe_allow_html=True)

# NAVIGATION
tab1, tab2, tab3 = st.tabs([
    "🔍 Disease Screening",
    "📊 Model Performance",
    "ℹ️ About Project"
])

# TAB 1: DISEASE SCREENING
with tab1:

    st.subheader("Rice Leaf Analysis")

    st.write(
        "Upload a clear image of a rice leaf to screen "
        "for one of four disease categories."
    )

    uploaded_file = st.file_uploader(
        "Choose a rice leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Uploaded Rice Leaf",
                use_container_width=True
            )

        with col2:

            st.write("### Image Details")
            st.write(
                f"Dimensions: {image.size[0]} × {image.size[1]}"
            )
            st.write("Format: RGB")

            if st.button("Analyze Image", type="primary"):

                with st.spinner("Analyzing rice leaf..."):

                    resized = image.resize((224, 224))

                    image_array = np.asarray(
                        resized,
                        dtype=np.float32
                    ) / 255.0

                    image_array = np.expand_dims(
                        image_array,
                        axis=0
                    )

                    predictions = model.predict(
                        image_array,
                        verbose=0
                    )[0]

                    predicted_index = int(
                        np.argmax(predictions)
                    )

                    confidence = float(
                        predictions[predicted_index]
                    )

                st.divider()
                st.subheader("Screening Results")

                if confidence < 0.60:
                    st.warning(
                        "Prediction is uncertain. Please upload "
                        "a clearer image for another screening."
                    )
                else:
                    st.success(
                        "Predicted Category: "
                        + class_names[str(predicted_index)]
                    )

                st.metric(
                    "Model Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.write("#### Class Probabilities")

                for index, probability in enumerate(predictions):

                    st.write(
                        f"{class_names[str(index)]}: "
                        f"{probability * 100:.2f}%"
                    )

                    st.progress(float(probability))

    else:
        st.info("Upload a rice leaf image to begin screening.")

# TAB 2: MODEL PERFORMANCE
with tab2:

    st.subheader("Model Evaluation")

    st.write(
        "Performance measured on the independent test set "
        "after removing exact duplicate images."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Test Accuracy", "99.44%")
    c2.metric("Test Images", "720")
    c3.metric("Test Loss", "0.0360")

    st.divider()

    st.write("#### Dataset Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Unique Images", "4,794")
    c2.metric("Training Images", "3,355")
    c3.metric("Validation Images", "719")
    c4.metric("Test Images", "720")

    st.write("#### Classification Categories")

    for name in class_names.values():
        st.write(f"• {name}")

    st.caption(
        "These results represent the held-out test set. "
        "They do not establish performance across all real-world "
        "farms, cameras, regions, or growing conditions."
    )

# TAB 3: ABOUT PROJECT
with tab3:

    st.subheader("About AgriSathi AI")

    st.write(
        "AgriSathi AI is an experimental prototype that uses "
        "deep learning and image classification to screen rice "
        "leaf images for selected disease categories."
    )

    st.write("#### Technology Stack")

    st.markdown("""
    - **Deep Learning:** TensorFlow / Keras
    - **Model Architecture:** MobileNetV2
    - **Image Processing:** Pillow and NumPy
    - **Interface:** Streamlit
    - **Dataset:** Rice Leaf Disease Classification India
    """)

    st.write("#### Disease Categories")

    st.markdown("""
    1. Bacterial Blight
    2. Blast
    3. Brown Spot
    4. Tungro
    """)

    st.write("#### Responsible Use")

    st.warning(
        "AgriSathi AI is an experimental screening prototype, "
        "not a confirmed agricultural diagnosis. Model confidence "
        "is not a guarantee of correctness. Consult a qualified "
        "agricultural expert before making crop-treatment decisions."
    )

# FOOTER
st.divider()

st.caption(
    "AgriSathi AI | Rice Leaf Disease Screening Prototype | 2026"
)
