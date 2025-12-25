"""
Grad-CAM visualization for model interpretability.
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .dataset import get_transforms
from .config import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES


def get_gradcam(model, target_layer=None):
    """Create GradCAM object for the model."""
    if target_layer is None:
        # Use the last conv layer of EfficientNet
        target_layer = model.backbone.features[-1]
    return GradCAM(model=model, target_layers=[target_layer])


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize tensor to numpy image [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = tensor.cpu() * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1)


def generate_gradcam(
    model,
    image: Union[str, Path, Image.Image],
    device: torch.device
) -> tuple:
    """Generate Grad-CAM heatmap for an image."""
    model.eval()

    # Load and transform image
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert('RGB')

    transform = get_transforms(is_training=False)
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()

    pred_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = prob if prob > 0.5 else 1 - prob

    # Generate Grad-CAM
    cam = get_gradcam(model)
    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]

    # Create visualization
    rgb_img = denormalize_image(img_tensor[0])
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return cam_image, pred_class, confidence, rgb_img


def plot_gradcam(
    model,
    image_path: Union[str, Path],
    true_label: str,
    device: torch.device,
    save_path: str = None
):
    """Plot original image with Grad-CAM overlay."""
    cam_image, pred_class, confidence, original = generate_gradcam(model, image_path, device)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Original
    axes[0].imshow(original)
    axes[0].set_title(f"Original\nTrue: {true_label}")
    axes[0].axis('off')

    # Grad-CAM
    color = 'green' if pred_class == true_label else 'red'
    axes[1].imshow(cam_image)
    axes[1].set_title(f"Grad-CAM\nPred: {pred_class} ({confidence:.1%})", color=color)
    axes[1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
    return pred_class, confidence


def plot_gradcam_grid(
    model,
    image_paths: list,
    true_labels: list,
    device: torch.device,
    save_path: str = None,
    title: str = "Grad-CAM Visualizations"
):
    """Plot grid of Grad-CAM visualizations."""
    n = len(image_paths)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3 * n))

    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (path, true_label) in enumerate(zip(image_paths, true_labels)):
        cam_image, pred_class, confidence, original = generate_gradcam(model, path, device)

        # Original
        axes[i, 0].imshow(original)
        axes[i, 0].set_title(f"True: {true_label}")
        axes[i, 0].axis('off')

        # Grad-CAM
        color = 'green' if pred_class == true_label else 'red'
        axes[i, 1].imshow(cam_image)
        axes[i, 1].set_title(f"Pred: {pred_class} ({confidence:.1%})", color=color)
        axes[i, 1].axis('off')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()
