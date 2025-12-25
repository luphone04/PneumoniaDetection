"""
FastAPI application for Pneumonia Detection API.

Run with: uvicorn api.main:app --reload
"""

import io
import time
import base64
from pathlib import Path

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .schemas import (
    HealthResponse,
    PredictionResponse,
    GradCAMResponse,
    ErrorResponse
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CHECKPOINT_PATH, CLASS_NAMES, CONFIDENCE_THRESHOLD
from src.model import create_model, get_device
from src.predict import load_model, predict_image
from src.gradcam import generate_gradcam

# =============================================================================
# App Configuration
# =============================================================================

app = FastAPI(
    title="Pneumonia Detection API",
    description="Deep learning API for detecting pneumonia from chest X-ray images using EfficientNet-B0",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Model Loading (on startup)
# =============================================================================

model = None
device = None


@app.on_event("startup")
async def load_model_on_startup():
    """Load model when the API starts."""
    global model, device

    device = get_device()
    print(f"Using device: {device}")

    if not CHECKPOINT_PATH.exists():
        print(f"Warning: Model checkpoint not found at {CHECKPOINT_PATH}")
        return

    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    model = load_model(model, CHECKPOINT_PATH, device)
    print(f"Model loaded from {CHECKPOINT_PATH}")


# =============================================================================
# Helper Functions
# =============================================================================

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_image(file: UploadFile) -> None:
    """Validate uploaded image file."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid content type: {file.content_type}. Expected image/*"
        )

    ext = Path(file.filename).suffix.lower() if file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )


async def read_image(file: UploadFile) -> Image.Image:
    """Read uploaded file as PIL Image."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        return image
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to read image: {str(e)}"
        )


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to docs."""
    return {"message": "Pneumonia Detection API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the API status and model loading state.
    """
    return HealthResponse(
        status="healthy" if model is not None else "model_not_loaded",
        model_loaded=model is not None,
        model_path=str(CHECKPOINT_PATH)
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"]
)
async def predict(file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")):
    """
    Predict pneumonia from chest X-ray image.

    Upload a chest X-ray image and get the prediction (NORMAL or PNEUMONIA)
    with confidence score.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    validate_image(file)
    image = await read_image(file)

    # Run inference
    start_time = time.time()
    pred_class, confidence = predict_image(model, image, device)
    processing_time = (time.time() - start_time) * 1000  # Convert to ms

    # Calculate raw probability
    probability = confidence if pred_class == "PNEUMONIA" else 1 - confidence

    return PredictionResponse(
        prediction=pred_class,
        confidence=confidence,
        probability=probability,
        processing_time_ms=round(processing_time, 2)
    )


@app.post(
    "/predict/gradcam",
    response_model=GradCAMResponse,
    responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    tags=["Prediction"]
)
async def predict_with_gradcam(file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG)")):
    """
    Predict with Grad-CAM visualization.

    Returns prediction along with a Grad-CAM heatmap overlay showing
    which regions of the image influenced the prediction.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    validate_image(file)
    image = await read_image(file)

    # Run inference with Grad-CAM
    start_time = time.time()
    cam_image, pred_class, confidence, _ = generate_gradcam(model, image, device)
    processing_time = (time.time() - start_time) * 1000

    # Convert Grad-CAM image to base64
    cam_pil = Image.fromarray(cam_image)
    buffer = io.BytesIO()
    cam_pil.save(buffer, format="PNG")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    # Calculate raw probability
    probability = confidence if pred_class == "PNEUMONIA" else 1 - confidence

    return GradCAMResponse(
        prediction=pred_class,
        confidence=confidence,
        probability=probability,
        processing_time_ms=round(processing_time, 2),
        gradcam_image=f"data:image/png;base64,{img_base64}"
    )


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "detail": None}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)}
    )


# =============================================================================
# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
