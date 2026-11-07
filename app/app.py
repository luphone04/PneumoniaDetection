"""
Streamlit Web UI for Pneumonia Detection.

Run with: streamlit run app/app.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
import time

from src.config import CHECKPOINT_PATH, CLASS_NAMES
from src.model import create_model, get_device
from src.predict import load_model, predict_image
from src.gradcam import generate_gradcam

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-normal {
        background-color: #E8F5E9;
        border: 2px solid #4CAF50;
    }
    .prediction-pneumonia {
        background-color: #FFEBEE;
        border: 2px solid #F44336;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Model Loading (Cached)
# =============================================================================

@st.cache_resource
def load_model_cached():
    """Load model once and cache it."""
    device = get_device()
    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    model = load_model(model, CHECKPOINT_PATH, device)
    return model, device


# =============================================================================
# Sample Selection (persisted across reruns)
# =============================================================================
# st.button() is only True on the single rerun triggered by its own click, so a
# local variable would be reset when "Analyze Image" triggers the next rerun.
# The selection is therefore stored in st.session_state via button callbacks.

SAMPLES_DIR = Path(__file__).parent / "samples"
SAMPLE_FILES = {
    "normal": SAMPLES_DIR / "normal_sample.jpeg",
    "pneumonia": SAMPLES_DIR / "pneumonia_sample.jpeg",
}


def select_sample(name: str):
    st.session_state["selected_sample"] = name


def clear_sample():
    # A new (or removed) upload replaces the sample selection.
    st.session_state.pop("selected_sample", None)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/lungs.png", width=80)
    st.title("About")

    st.markdown("""
    This application uses deep learning to detect **pneumonia** from chest X-ray images.

    **Model:** EfficientNet-B0
    **Accuracy:** 90.5%
    **Recall:** 98.2%
    """)

    st.divider()

    st.subheader("How to Use")
    st.markdown("""
    1. Upload a chest X-ray image
    2. Click **Analyze Image**
    3. View prediction and Grad-CAM
    """)

    st.divider()

    st.subheader("Model Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", "90.5%")
        st.metric("Precision", "88.0%")
    with col2:
        st.metric("Recall", "98.2%")
        st.metric("F1 Score", "92.8%")

    st.divider()

    st.markdown("""
    **Links:**
    [GitHub Repository](#) | [Live Demo](#)

    ---
    *Built with PyTorch & Streamlit*
    """)

# =============================================================================
# Main Content
# =============================================================================

# Header
st.markdown('<p class="main-header">🫁 Pneumonia Detection from Chest X-Rays</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image to detect pneumonia using AI</p>', unsafe_allow_html=True)

# Load model
try:
    model, device = load_model_cached()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

if model_loaded:
    # Create columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")

        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"],
            help="Supported formats: JPG, JPEG, PNG",
            on_change=clear_sample,
        )

        # Sample images section
        st.markdown("---")
        st.markdown("**Or try a sample image:**")

        sample_col1, sample_col2 = st.columns(2)

        with sample_col1:
            st.button("🟢 Normal Sample", width="stretch",
                      on_click=select_sample, args=("normal",))
        with sample_col2:
            st.button("🔴 Pneumonia Sample", width="stretch",
                      on_click=select_sample, args=("pneumonia",))

        # A selected sample takes precedence until a new file is uploaded
        selected_sample = st.session_state.get("selected_sample")
        image_caption = "Uploaded X-Ray"
        if selected_sample in SAMPLE_FILES and SAMPLE_FILES[selected_sample].exists():
            uploaded_file = SAMPLE_FILES[selected_sample]
            image_caption = f"Sample: {selected_sample.upper()} X-Ray"

    with col2:
        st.subheader("🔍 Analysis Results")
        results_placeholder = st.empty()

    # Load image if uploaded or sample selected
    image = None
    if uploaded_file is not None:
        try:
            # .convert("RGB") also handles grayscale, RGBA and palette images
            image = Image.open(uploaded_file).convert("RGB")
        except (UnidentifiedImageError, OSError):
            with col1:
                st.error("Could not read this file as an image. "
                         "Please upload a valid JPG or PNG chest X-ray.")

    if image is not None:
        if isinstance(uploaded_file, Path):
            st.session_state['image_source'] = str(uploaded_file)
        else:
            st.session_state['image_source'] = uploaded_file.name

        # Display uploaded image
        with col1:
            st.image(image, caption=image_caption, width="stretch")

        # Analyze button
        with col1:
            analyze_button = st.button("🔬 Analyze Image", type="primary", width="stretch")

        if analyze_button:
            with col2:
                with st.spinner("Analyzing image..."):
                    # Run prediction
                    start_time = time.time()
                    pred_class, confidence = predict_image(model, image, device)
                    inference_time = (time.time() - start_time) * 1000

                    # Generate Grad-CAM
                    cam_image, _, _, original = generate_gradcam(model, image, device)

                # Display results
                if pred_class == "PNEUMONIA":
                    st.markdown(f"""
                    <div class="prediction-box prediction-pneumonia">
                        <h2 style="color: #F44336; margin: 0;">⚠️ PNEUMONIA DETECTED</h2>
                        <p class="confidence-text">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-normal">
                        <h2 style="color: #4CAF50; margin: 0;">✅ NORMAL</h2>
                        <p class="confidence-text">Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics: prediction on its own row so the class name is never clipped
                st.metric("Prediction", pred_class)
                m2, m3 = st.columns(2)
                with m2:
                    st.metric("Confidence", f"{confidence:.1%}")
                with m3:
                    st.metric("Time", f"{inference_time:.0f}ms")

                # Grad-CAM visualization
                st.markdown("---")
                st.subheader("🔥 Grad-CAM Visualization")
                st.caption(
                    f"Highlighted regions most increased the model's score for the "
                    f"predicted class ({pred_class}). Grad-CAM is a coarse explanation "
                    f"of the model's attention, not a clinical localisation of disease."
                )

                gcol1, gcol2 = st.columns(2)
                with gcol1:
                    st.image(original, caption="Original", width="stretch")
                with gcol2:
                    st.image(cam_image, caption="Grad-CAM Heatmap", width="stretch")

                # Disclaimer
                st.warning("""
                **Disclaimer:** This tool is for educational purposes only and should not be used
                for medical diagnosis. Always consult a qualified healthcare professional.
                """)

else:
    st.error("Model could not be loaded. Please check the model file exists.")

# =============================================================================
# Footer
# =============================================================================

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888;'>Built with ❤️ using PyTorch, EfficientNet-B0, and Streamlit</p>",
    unsafe_allow_html=True
)
