"""
Inference functions for Pneumonia classification.
"""

import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from typing import Union, Tuple

from .dataset import get_transforms
from .config import CLASS_NAMES, CHECKPOINT_PATH


def load_model(model: nn.Module, checkpoint_path: Path = CHECKPOINT_PATH, device: str = "cpu") -> nn.Module:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_image(
    model: nn.Module,
    image: Union[str, Path, Image.Image],
    device: torch.device
) -> Tuple[str, float]:
    """Predict class for a single image."""
    model.eval()

    # Load image if path
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')

    # Transform
    transform = get_transforms(is_training=False)
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    pred_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob

    return pred_class, confidence
