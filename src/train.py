"""
Training pipeline for Pneumonia classification.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .config import (
    STAGE1_EPOCHS, STAGE1_LR, STAGE2_EPOCHS, STAGE2_LR,
    WEIGHT_DECAY, SCHEDULER_PATIENCE, SCHEDULER_FACTOR,
    EARLY_STOP_PATIENCE, CHECKPOINT_PATH, MODEL_DIR
)
from .model import PneumoniaClassifier, get_device


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 7, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)

    return {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0)
    }


def train(
    model: PneumoniaClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
    stage: str = "stage1",
    use_wandb: bool = True,
    wandb_run = None
) -> Dict[str, list]:
    """Training loop with validation."""

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min',
        patience=SCHEDULER_PATIENCE,
        factor=SCHEDULER_FACTOR
    )
    early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr': []}
    best_val_loss = float('inf')

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        start = time.time()

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)

        # Get current LR
        current_lr = optimizer.param_groups[0]['lr']

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Log
        elapsed = time.time() - start
        print(f"[{stage}] Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.3f} | "
              f"Val F1: {val_metrics['f1']:.3f} | "
              f"LR: {current_lr:.2e}")

        # W&B logging
        if use_wandb and wandb_run:
            wandb_run.log({
                f"{stage}/train_loss": train_loss,
                f"{stage}/train_acc": train_acc,
                f"{stage}/val_loss": val_metrics['loss'],
                f"{stage}/val_acc": val_metrics['accuracy'],
                f"{stage}/val_precision": val_metrics['precision'],
                f"{stage}/val_recall": val_metrics['recall'],
                f"{stage}/val_f1": val_metrics['f1'],
                f"{stage}/lr": current_lr,
                "epoch": epoch + 1
            })

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        history['lr'].append(current_lr)

        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_metrics': val_metrics
            }, CHECKPOINT_PATH)
            print(f"  -> Saved best model (val_loss: {best_val_loss:.4f})")

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return history


def train_two_stage(
    model: PneumoniaClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor,
    device: torch.device,
    use_wandb: bool = True,
    wandb_run = None
) -> Dict[str, list]:
    """Two-stage training: frozen backbone then fine-tuning."""

    # Stage 1: Train classifier only
    print("\n" + "=" * 60)
    print("STAGE 1: Training classifier (backbone frozen)")
    print("=" * 60)
    model.freeze_backbone()
    trainable, total = model.get_param_counts()
    print(f"Trainable params: {trainable:,} / {total:,}")

    history1 = train(
        model, train_loader, val_loader, pos_weight,
        epochs=STAGE1_EPOCHS, lr=STAGE1_LR, device=device,
        stage="stage1", use_wandb=use_wandb, wandb_run=wandb_run
    )

    # Stage 2: Fine-tune entire network
    print("\n" + "=" * 60)
    print("STAGE 2: Fine-tuning entire network")
    print("=" * 60)
    model.unfreeze_backbone()
    trainable, total = model.get_param_counts()
    print(f"Trainable params: {trainable:,} / {total:,}")

    history2 = train(
        model, train_loader, val_loader, pos_weight,
        epochs=STAGE2_EPOCHS, lr=STAGE2_LR, device=device,
        stage="stage2", use_wandb=use_wandb, wandb_run=wandb_run
    )

    # Combine histories
    history = {k: history1[k] + history2[k] for k in history1}
    return history
