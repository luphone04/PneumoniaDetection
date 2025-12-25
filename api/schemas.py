"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ClassLabel(str, Enum):
    """Prediction class labels."""
    NORMAL = "NORMAL"
    PNEUMONIA = "PNEUMONIA"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., example="healthy")
    model_loaded: bool = Field(..., example=True)
    model_path: str = Field(..., example="models/best_model.pt")


class PredictionResponse(BaseModel):
    """Prediction response."""
    prediction: ClassLabel = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probability: float = Field(..., ge=0, le=1, description="Raw probability for PNEUMONIA")
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "PNEUMONIA",
                "confidence": 0.92,
                "probability": 0.92,
                "processing_time_ms": 45.2
            }
        }


class GradCAMResponse(BaseModel):
    """Prediction with Grad-CAM visualization."""
    prediction: ClassLabel = Field(..., description="Predicted class")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    probability: float = Field(..., ge=0, le=1, description="Raw probability for PNEUMONIA")
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")
    gradcam_image: str = Field(..., description="Base64 encoded Grad-CAM overlay image")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "PNEUMONIA",
                "confidence": 0.92,
                "probability": 0.92,
                "processing_time_ms": 150.5,
                "gradcam_image": "data:image/png;base64,..."
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "detail": "Supported formats: JPEG, PNG"
            }
        }
