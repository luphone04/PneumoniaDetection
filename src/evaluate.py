"""
Evaluation functions for Pneumonia classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from .config import CLASS_NAMES


def predict_proba(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get predictions, probabilities, and true labels."""
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.numpy())

    return np.array(all_probs), np.array(all_preds), np.array(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict:
    """Compute all evaluation metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> Dict:
    """Full evaluation on a dataset."""
    probs, preds, labels = predict_proba(model, loader, device)
    metrics = compute_metrics(labels, preds, probs)

    print("=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\nConfusion Matrix:")
    print(f"  {CLASS_NAMES[0]:>10} {CLASS_NAMES[1]:>10}")
    for i, row in enumerate(metrics['confusion_matrix']):
        print(f"  {CLASS_NAMES[i]:>10} {row[0]:>10} {row[1]:>10}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=CLASS_NAMES))

    return metrics


def get_predictions_with_paths(
    model: nn.Module,
    dataset,
    device: torch.device
) -> list:
    """Get predictions with image paths for error analysis."""
    model.eval()
    results = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, label = dataset[idx]
            image = image.unsqueeze(0).to(device)

            output = model(image)
            prob = torch.sigmoid(output).item()
            pred = 1 if prob > 0.5 else 0

            results.append({
                'path': dataset.image_paths[idx],
                'true_label': label,
                'pred_label': pred,
                'probability': prob,
                'correct': pred == label
            })

    return results
