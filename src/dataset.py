"""
PyTorch Dataset and DataLoader utilities for Chest X-Ray classification.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import random

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from .config import (
    DATA_DIR, IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS,
    IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES, SEED
)


class ChestXRayDataset(Dataset):
    """Dataset for Chest X-Ray images."""

    def __init__(
        self,
        image_paths: List[Path],
        labels: List[int],
        transform: Optional[transforms.Compose] = None
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(is_training: bool = True) -> transforms.Compose:
    """Get image transforms for training or validation/test."""
    if is_training:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])


def load_image_paths_and_labels(
    data_dir: Path,
    split: str
) -> Tuple[List[Path], List[int]]:
    """Load image paths and labels from a data split directory."""
    image_paths = []
    labels = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = data_dir / split / class_name
        if class_dir.exists():
            for img_path in class_dir.glob('*.jpeg'):
                image_paths.append(img_path)
                labels.append(class_idx)

    return image_paths, labels


def create_train_val_split(
    data_dir: Path = DATA_DIR,
    val_ratio: float = 0.15,
    seed: int = SEED
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    """Create stratified train/val split from training data."""
    # Load all training images
    train_paths, train_labels = load_image_paths_and_labels(data_dir, 'train')

    # Stratified split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels,
        test_size=val_ratio,
        stratify=train_labels,
        random_state=seed
    )

    return train_paths, train_labels, val_paths, val_labels


def get_class_weights(labels: List[int]) -> torch.Tensor:
    """Calculate class weights for imbalanced dataset."""
    class_counts = torch.bincount(torch.tensor(labels))
    total = len(labels)
    weights = total / (len(class_counts) * class_counts.float())
    return weights


def get_sampler(labels: List[int]) -> WeightedRandomSampler:
    """Create weighted sampler for balanced batches."""
    class_weights = get_class_weights(labels)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )
    return sampler


def get_dataloaders(
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_ratio: float = 0.15,
    use_weighted_sampling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders."""

    # Create train/val split
    train_paths, train_labels, val_paths, val_labels = create_train_val_split(
        data_dir, val_ratio
    )

    # Load test data
    test_paths, test_labels = load_image_paths_and_labels(data_dir, 'test')

    # Create datasets
    train_dataset = ChestXRayDataset(
        train_paths, train_labels, transform=get_transforms(is_training=True)
    )
    val_dataset = ChestXRayDataset(
        val_paths, val_labels, transform=get_transforms(is_training=False)
    )
    test_dataset = ChestXRayDataset(
        test_paths, test_labels, transform=get_transforms(is_training=False)
    )

    # Create sampler for training if using weighted sampling
    train_sampler = get_sampler(train_labels) if use_weighted_sampling else None

    # Only use pin_memory for CUDA (not supported on MPS)
    pin_memory = torch.cuda.is_available()

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Print dataset info
    print(f"Train: {len(train_dataset)} images")
    print(f"Val:   {len(val_dataset)} images")
    print(f"Test:  {len(test_dataset)} images")

    return train_loader, val_loader, test_loader


def get_pos_weight(labels: List[int]) -> torch.Tensor:
    """Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance."""
    labels_tensor = torch.tensor(labels)
    neg_count = (labels_tensor == 0).sum().float()  # NORMAL
    pos_count = (labels_tensor == 1).sum().float()  # PNEUMONIA
    pos_weight = neg_count / pos_count
    return pos_weight
