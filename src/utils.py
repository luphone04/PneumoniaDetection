"""
Utility functions for visualization and helpers.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from .config import IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor from ImageNet normalization."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return tensor * std + mean


def show_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    predictions: Optional[torch.Tensor] = None,
    n_images: int = 8,
    save_path: Optional[str] = None
):
    """Display a batch of images with labels."""
    n_images = min(n_images, len(images))
    cols = 4
    rows = (n_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for idx in range(n_images):
        img = denormalize(images[idx]).permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        axes[idx].imshow(img)
        axes[idx].axis('off')

        label = CLASS_NAMES[labels[idx]]
        title = f"True: {label}"

        if predictions is not None:
            pred = CLASS_NAMES[predictions[idx]]
            color = 'green' if pred == label else 'red'
            title += f"\nPred: {pred}"
            axes[idx].set_title(title, color=color, fontsize=10)
        else:
            axes[idx].set_title(title, fontsize=10)

    # Hide empty subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
