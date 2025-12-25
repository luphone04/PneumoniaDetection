"""
EfficientNet-B0 model for Pneumonia classification.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Tuple

from .config import DROPOUT_RATE, NUM_CLASSES


class PneumoniaClassifier(nn.Module):
    """EfficientNet-B0 based classifier for chest X-ray pneumonia detection."""

    def __init__(
        self,
        pretrained: bool = True,
        dropout_rate: float = DROPOUT_RATE,
        freeze_backbone: bool = True
    ):
        super().__init__()

        # Load pretrained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Get the number of features from the classifier
        in_features = self.backbone.classifier[1].in_features  # 1280

        # Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, NUM_CLASSES)
        )

        # Freeze backbone if specified
        if freeze_backbone:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all layers except the classifier."""
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.backbone.features.parameters():
            param.requires_grad = True

    def get_param_counts(self) -> Tuple[int, int]:
        """Return (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def create_model(
    pretrained: bool = True,
    dropout_rate: float = DROPOUT_RATE,
    freeze_backbone: bool = True,
    device: str = None
) -> PneumoniaClassifier:
    """Factory function to create the model."""
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"

    model = PneumoniaClassifier(
        pretrained=pretrained,
        dropout_rate=dropout_rate,
        freeze_backbone=freeze_backbone
    )

    return model.to(device)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
