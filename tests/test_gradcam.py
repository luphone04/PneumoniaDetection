"""Regression tests for checkpoint loading, inference and Grad-CAM targeting."""

import numpy as np
import pytest
import torch
from PIL import Image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

from tests.conftest import SAMPLES
from src.config import CHECKPOINT_PATH
from src.dataset import get_transforms
from src.gradcam import generate_gradcam, get_gradcam
from src.predict import predict_image


def test_checkpoint_loads_with_weights_only():
    # PyTorch >= 2.6 defaults torch.load to weights_only=True; the checkpoint
    # must stay loadable under that safer default.
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    assert "model_state_dict" in ckpt


def _cam_for(model, image, device, category):
    x = get_transforms(is_training=False)(image).unsqueeze(0).to(device)
    with get_gradcam(model) as cam:
        return cam(input_tensor=x, targets=[BinaryClassifierOutputTarget(category)])[0]


@pytest.mark.parametrize("label", ["NORMAL", "PNEUMONIA"])
def test_gradcam_explains_predicted_class(model_and_device, label):
    """The heatmap must explain the predicted class, not always PNEUMONIA.

    Sample images are functionality checks, not an accuracy measure.
    """
    model, device = model_and_device
    image = Image.open(SAMPLES[label]).convert("RGB")

    pred_class, _ = predict_image(model, image, device)
    assert pred_class == label  # pipeline sanity check on the bundled sample

    cam_image, gc_class, _, _ = generate_gradcam(model, image, device)
    assert gc_class == pred_class
    assert cam_image.shape == (224, 224, 3) and cam_image.dtype == np.uint8

    from pytorch_grad_cam.utils.image import show_cam_on_image
    from src.gradcam import denormalize_image
    x = get_transforms(is_training=False)(image)
    rgb = denormalize_image(x)

    predicted_idx = 1 if pred_class == "PNEUMONIA" else 0
    expected = show_cam_on_image(rgb, _cam_for(model, image, device, predicted_idx), use_rgb=True)
    np.testing.assert_array_equal(cam_image, expected)

    if pred_class == "NORMAL":
        # Before the fix, targets=None explained the PNEUMONIA logit here.
        wrong = show_cam_on_image(rgb, _cam_for(model, image, device, 1), use_rgb=True)
        assert not np.array_equal(cam_image, wrong)


@pytest.mark.parametrize("mode", ["L", "LA", "RGBA", "P", "I;16"])
def test_inference_on_varied_image_modes(model_and_device, mode):
    model, device = model_and_device
    base = Image.open(SAMPLES["NORMAL"])
    image = base.convert(mode).convert("RGB")
    pred_class, confidence = predict_image(model, image, device)
    assert pred_class in ("NORMAL", "PNEUMONIA") and 0.5 <= confidence <= 1.0
    cam_image, *_ = generate_gradcam(model, image, device)
    assert cam_image.shape == (224, 224, 3)
