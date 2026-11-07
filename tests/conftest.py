import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = {
    "NORMAL": ROOT / "app" / "samples" / "normal_sample.jpeg",
    "PNEUMONIA": ROOT / "app" / "samples" / "pneumonia_sample.jpeg",
}


@pytest.fixture(scope="session")
def model_and_device():
    from src.config import CHECKPOINT_PATH
    from src.model import create_model
    from src.predict import load_model

    device = torch.device("cpu")
    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    return load_model(model, CHECKPOINT_PATH, device), device
