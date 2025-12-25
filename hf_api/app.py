"""
FastAPI for Pneumonia Detection - Hugging Face Spaces Deployment
"""

import io
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
MODEL_PATH = Path("models/best_model.pt")

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
# Response Models
# =============================================================================

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probability: float
    processing_time_ms: float

# =============================================================================
# App Setup
# =============================================================================

app = FastAPI(
    title="Pneumonia Detection API",
    description="Deep learning API for detecting pneumonia from chest X-rays",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Loading
# =============================================================================

model = None
device = None

@app.on_event("startup")
async def load_model():
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        print(f"Warning: Model not found at {MODEL_PATH}")
        return

    model = PneumoniaClassifier()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully")

# =============================================================================
# Helper Functions
# =============================================================================

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

async def read_image(file: UploadFile) -> Image.Image:
    contents = await file.read()
    return Image.open(io.BytesIO(contents)).convert("RGB")

def predict(image: Image.Image):
    transform = get_transforms()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    pred_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob
    return pred_class, confidence, prob

# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    return {"message": "Pneumonia Detection API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="healthy" if model else "model_not_loaded",
        model_loaded=model is not None
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image = await read_image(file)

    start_time = time.time()
    pred_class, confidence, prob = predict(image)
    processing_time = (time.time() - start_time) * 1000

    return PredictionResponse(
        prediction=pred_class,
        confidence=confidence,
        probability=prob,
        processing_time_ms=round(processing_time, 2)
    )
