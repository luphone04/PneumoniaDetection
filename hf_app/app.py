"""
Hugging Face Spaces - Pneumonia Detection App
This is a self-contained version for Hugging Face Spaces deployment.
Copy this file as 'app.py' to your HF Spaces repository.
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import time

# =============================================================================
# Configuration
# =============================================================================

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = "models/best_model.pt"

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🫁",
    layout="wide"
)

# =============================================================================
# Model Definition
# =============================================================================

class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


# =============================================================================
# Helper Functions
# =============================================================================

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    return img.permute(1, 2, 0).numpy().clip(0, 1)


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaClassifier()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model.to(device), device


def predict(model, image, device):
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    pred_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob
    return pred_class, confidence, img_tensor


def generate_gradcam(model, img_tensor, device):
    target_layer = model.backbone.features[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]

    rgb_img = denormalize(img_tensor[0])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return cam_image, rgb_img


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
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.title("🫁 About")
    st.markdown("""
    Deep learning model for detecting **pneumonia** from chest X-rays.

    **Model:** EfficientNet-B0
    **Accuracy:** 90.5%
    **Recall:** 98.2%
    """)

    st.divider()
    st.subheader("Model Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", "90.5%")
    col2.metric("Recall", "98.2%")

    st.divider()
    st.markdown("*Built with PyTorch & Streamlit*")

# =============================================================================
# Main Content
# =============================================================================

st.markdown('<p class="main-header">🫁 Pneumonia Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a chest X-ray image to detect pneumonia</p>', unsafe_allow_html=True)

# Load model
try:
    model, device = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Failed to load model: {e}")
    model_loaded = False

if model_loaded:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image",
            type=["jpg", "jpeg", "png"]
        )

        # Sample images
        st.subheader("🖼️ Or Try Sample Images")
        sample_col1, sample_col2 = st.columns(2)
        with sample_col1:
            if st.button("Normal Sample", use_container_width=True):
                st.session_state.sample = "samples/normal_sample.jpeg"
        with sample_col2:
            if st.button("Pneumonia Sample", use_container_width=True):
                st.session_state.sample = "samples/pneumonia_sample.jpeg"

    # Determine which image to use
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.session_state.sample = None  # Clear sample when uploading
    elif "sample" in st.session_state and st.session_state.sample:
        image = Image.open(st.session_state.sample).convert("RGB")

    if image is not None:

        with col1:
            st.image(image, caption="Uploaded X-Ray", use_container_width=True)
            analyze = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

        if analyze:
            with col2:
                with st.spinner("Analyzing..."):
                    start_time = time.time()
                    pred_class, confidence, img_tensor = predict(model, image, device)
                    cam_image, original = generate_gradcam(model, img_tensor, device)
                    inference_time = (time.time() - start_time) * 1000

                # Results
                if pred_class == "PNEUMONIA":
                    st.markdown(f"""
                    <div class="prediction-box prediction-pneumonia">
                        <h2 style="color: #F44336;">⚠️ PNEUMONIA DETECTED</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-box prediction-normal">
                        <h2 style="color: #4CAF50;">✅ NORMAL</h2>
                        <p>Confidence: {confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Metrics row
                m1, m2, m3 = st.columns(3)
                m1.metric("Prediction", pred_class)
                m2.metric("Confidence", f"{confidence:.1%}")
                m3.metric("Time", f"{inference_time:.0f}ms")

                # Grad-CAM
                st.subheader("🔥 Grad-CAM")
                gcol1, gcol2 = st.columns(2)
                gcol1.image(original, caption="Original", use_container_width=True)
                gcol2.image(cam_image, caption="Heatmap", use_container_width=True)

                st.warning("**Disclaimer:** For educational purposes only. Consult a healthcare professional.")

# Footers
st.markdown("---")
st.markdown("<p style='text-align:center;color:#888;'>Built with PyTorch & Streamlit</p>", unsafe_allow_html=True)
