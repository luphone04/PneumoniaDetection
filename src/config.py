"""
Configuration constants for the Pneumonia Detection project.
All hyperparameters and paths are defined here for easy modification.
"""

from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"

# =============================================================================
# Data Configuration
# =============================================================================
IMAGE_SIZE = 224  # EfficientNet-B0 input size
BATCH_SIZE = 32
NUM_WORKERS = 4  # DataLoader workers

# ImageNet normalization (required for pretrained models)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Class labels
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 1  # Binary classification with sigmoid

# =============================================================================
# Model Configuration
# =============================================================================
MODEL_NAME = "efficientnet_b0"
DROPOUT_RATE = 0.3
PRETRAINED = True

# =============================================================================
# Training Configuration - Stage 1 (Frozen Backbone)
# =============================================================================
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-4
STAGE1_FREEZE_BACKBONE = True

# =============================================================================
# Training Configuration - Stage 2 (Fine-tuning)
# =============================================================================
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-5
STAGE2_FREEZE_BACKBONE = False

# =============================================================================
# Optimizer Configuration
# =============================================================================
WEIGHT_DECAY = 1e-4
BETAS = (0.9, 0.999)

# =============================================================================
# Scheduler Configuration
# =============================================================================
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5
SCHEDULER_MIN_LR = 1e-7

# =============================================================================
# Early Stopping Configuration
# =============================================================================
EARLY_STOP_PATIENCE = 7
EARLY_STOP_MIN_DELTA = 0.001

# =============================================================================
# Model Checkpointing
# =============================================================================
CHECKPOINT_PATH = MODEL_DIR / "best_model.pt"
SAVE_BEST_ONLY = True
MONITOR_METRIC = "val_loss"

# =============================================================================
# Weights & Biases Configuration
# =============================================================================
WANDB_PROJECT = "pneumonia-detection"
WANDB_ENTITY = None  # Set to your W&B username if needed

# =============================================================================
# Inference Configuration
# =============================================================================
CONFIDENCE_THRESHOLD = 0.5  # For binary classification
GRADCAM_TARGET_LAYER = "features"  # EfficientNet feature extractor

# =============================================================================
# Random Seed (for reproducibility)
# =============================================================================
SEED = 42


def create_directories():
    """Create all necessary directories if they don't exist."""
    for directory in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, FIGURES_DIR, LOGS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print configuration for verification
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Model Directory: {MODEL_DIR}")
    print(f"Image Size: {IMAGE_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Model: {MODEL_NAME}")
