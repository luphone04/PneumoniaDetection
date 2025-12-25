# Medical Image Classification: Pneumonia Detection from Chest X-Rays

## Project Overview

**Objective:** Build an end-to-end deep learning system that classifies chest X-ray images as either "Normal" or "Pneumonia" using transfer learning, with full deployment as a web application.

**Dataset:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle
- 5,863 JPEG images
- 2 classes: NORMAL, PNEUMONIA
- Pre-split into train/val/test

**Why This Project:**
- Demonstrates medical AI understanding
- Shows transfer learning expertise
- Includes model interpretability (Grad-CAM)
- Full production deployment
- Real-world impact narrative

---

## Tech Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Language | Python | 3.10+ |
| Deep Learning | PyTorch | 2.1+ |
| Pretrained Models | torchvision | 0.16+ |
| Data Processing | Pillow, NumPy | Latest |
| Data Analysis | pandas, matplotlib, seaborn | Latest |
| Experiment Tracking | Weights & Biases | Latest |
| Model Interpretability | grad-cam | Latest |
| Web API | FastAPI | 0.104+ |
| Web UI | Streamlit | 1.28+ |
| Containerization | Docker | Latest |
| Deployment | Hugging Face Spaces | - |
| Testing | pytest | Latest |
| Code Quality | ruff, black | Latest |

**Model Architecture:** EfficientNet-B0 (pretrained on ImageNet)
- Optimal accuracy/efficiency tradeoff
- Small model size (~20MB) suitable for deployment
- Fine-tune final classification layer

---

## Project Structure

```
resume-ml/
├── PROJECT_PLAN.md              # This file
├── README.md                    # Project documentation (create at end)
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup (optional)
├── .gitignore                   # Git ignore file
├── .env.example                 # Environment variables template
│
├── data/
│   ├── raw/                     # Original dataset (gitignored)
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── val/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   └── test/
│   │       ├── NORMAL/
│   │       └── PNEUMONIA/
│   └── processed/               # Processed data if needed
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis
│   ├── 02_baseline.ipynb        # Initial model experiments
│   └── 03_evaluation.ipynb      # Final evaluation & visualizations
│
├── src/
│   ├── __init__.py
│   ├── config.py                # Configuration constants
│   ├── dataset.py               # PyTorch Dataset class
│   ├── model.py                 # Model architecture
│   ├── train.py                 # Training loop
│   ├── evaluate.py              # Evaluation metrics
│   ├── predict.py               # Inference functions
│   ├── gradcam.py               # Grad-CAM visualization
│   └── utils.py                 # Helper functions
│
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI application
│   └── schemas.py               # Pydantic models
│
├── app/
│   └── streamlit_app.py         # Streamlit UI
│
├── models/
│   └── best_model.pt            # Saved model weights (gitignored)
│
├── outputs/
│   ├── figures/                 # Generated plots
│   └── logs/                    # Training logs
│
├── tests/
│   ├── __init__.py
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_api.py
│
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Implementation Phases

### Phase 1: Project Setup & Environment ✅ COMPLETED
> **Goal:** Set up development environment and project structure

- [x] Create project directory structure as outlined above
- [x] Create virtual environment with Python 3.11 (venv/)
- [x] Create `requirements.txt` with all dependencies
- [x] Initialize git repository with proper `.gitignore`
- [x] Create `src/config.py` with project constants (paths, hyperparameters)
- [x] Set up Weights & Biases account and get API key
- [x] Create `.env.example` for environment variables
- [x] Verify all installations work correctly (PyTorch 2.9.1, MPS available)

**Files to create:**
- `requirements.txt`
- `.gitignore`
- `.env.example`
- `src/__init__.py`
- `src/config.py`

---

### Phase 2: Data Acquisition & Exploration ✅ COMPLETED
> **Goal:** Download dataset and understand its characteristics

- [x] Download Chest X-Ray dataset from Kaggle
- [x] Organize data into `data/raw/` directory
- [x] Create `notebooks/01_eda.ipynb` for exploration
- [x] Analyze class distribution (check for imbalance)
- [x] Visualize sample images from each class
- [x] Check image sizes and formats
- [x] Identify any data quality issues
- [x] Document findings in notebook markdown cells

**Key Metrics Recorded:**
- Training set: NORMAL=1,341, PNEUMONIA=3,875 (Total: 5,216)
- Validation set: NORMAL=8, PNEUMONIA=8 (Total: 16) ⚠️ Too small!
- Test set: NORMAL=234, PNEUMONIA=390 (Total: 624)
- Image sizes: Variable (384-2572 width, 127-2663 height)
- Class imbalance ratio: 2.89:1 (PNEUMONIA:NORMAL)
- Image modes: Grayscale (L) and RGB mixed

---

### Phase 3: Data Preprocessing & Augmentation ✅ COMPLETED
> **Goal:** Create robust data pipeline with augmentations

- [x] Create `src/dataset.py` with custom PyTorch Dataset class
- [x] Implement image transformations:
  - Resize to 224x224 (EfficientNet input size)
  - Normalize with ImageNet mean/std
  - Training augmentations: RandomHorizontalFlip, RandomRotation(10), ColorJitter
  - Validation/Test: Only resize and normalize
- [x] Create DataLoader factory function with configurable batch size
- [x] Handle class imbalance with WeightedRandomSampler
- [x] Test data pipeline with sample batch visualization
- [x] Created stratified 85/15 train/val split (Train: 4433, Val: 783)

**Files created:**
- `src/dataset.py` - Dataset class, transforms, DataLoader factory
- `src/utils.py` - Visualization helpers, denormalize, show_batch

---

### Phase 4: Model Development ✅ COMPLETED
> **Goal:** Build the model architecture with transfer learning

- [x] Create `src/model.py` with model definition
- [x] Load pretrained EfficientNet-B0 from torchvision
- [x] Freeze base layers initially
- [x] Replace classifier head for binary classification
- [x] Add dropout (0.3) before final layer for regularization
- [x] Create model factory function with configurable options
- [x] Test forward pass with dummy input
- [x] Params: 4M total (1,281 trainable frozen, 4M unfrozen)

**Files created:**
- `src/model.py`

---

### Phase 5: Training Pipeline ✅ COMPLETED
> **Goal:** Implement training loop with experiment tracking

- [x] Create `src/train.py` with training logic
- [x] Implement BCEWithLogitsLoss, AdamW, ReduceLROnPlateau, EarlyStopping
- [x] Integrate Weights & Biases logging
- [x] Implement model checkpointing (saves best val_loss)
- [x] Create `notebooks/02_baseline.ipynb` for training
- [ ] **USER ACTION:** Run notebook to train model (~20 epochs)

**Files created:**
- `src/train.py`
- `notebooks/02_baseline.ipynb`

---

### Phase 6: Evaluation & Interpretability ✅ COMPLETED
> **Goal:** Comprehensive model evaluation with explainability

- [x] Create `src/evaluate.py` with evaluation functions
- [x] Compute metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- [x] Create `src/gradcam.py` for Grad-CAM visualization
- [x] Create `src/predict.py` for inference
- [x] Create `notebooks/03_evaluation.ipynb`
- [x] **USER ACTION:** Run notebook after training to get results

**Files created:**
- `src/evaluate.py`
- `src/gradcam.py`
- `src/predict.py`
- `notebooks/03_evaluation.ipynb`

---

### Phase 6.5: Advanced ML Improvements ✅ COMPLETED
> **Goal:** Add industry-standard ML optimizations

- [x] **Threshold Optimization** (in `notebooks/03_evaluation.ipynb`)
  - Find optimal classification threshold using ROC curve
  - Compare thresholds for max F1 vs max recall
  - Optimal threshold: 0.89 (F1: 95.1%) vs default 0.5 (F1: 92.8%)
  - Plot precision-recall tradeoff curve

- [x] **ONNX Export** (create `src/export.py`)
  - Export PyTorch model to ONNX format
  - Validate exported model produces same outputs
  - Save to `models/best_model.onnx` (0.69 MB)
  - Add inference example using ONNX Runtime

**What is ONNX?**
- Open Neural Network Exchange - universal model format
- Works with: TensorFlow, mobile (iOS/Android), web browsers, C++, Java
- Faster inference than PyTorch in production
- Shows understanding of production ML deployment

**Files to create/modify:**
- `notebooks/03_evaluation.ipynb` - Add threshold optimization section
- `src/export.py` - ONNX export script
- `models/best_model.onnx` - Exported model

---

### Phase 7: API Development ✅ COMPLETED
> **Goal:** Create REST API for model inference

- [x] Create `api/schemas.py` with Pydantic models
  - `PredictionResponse`: prediction, confidence, processing_time
  - `GradCAMResponse`: prediction + base64 image
  - `HealthResponse`: status, model_loaded
- [x] Create `api/main.py` with FastAPI application
- [x] Implement endpoints:
  - `GET /health` - Health check
  - `POST /predict` - Image prediction
  - `POST /predict/gradcam` - Prediction with Grad-CAM overlay
- [x] Add proper error handling and validation
- [x] Add CORS middleware for frontend access
- [x] Test API with sample images using curl
- [x] Document API with OpenAPI (automatic at /docs)

**Files to create:**
- `api/__init__.py`
- `api/schemas.py`
- `api/main.py`

---

### Phase 8: Web UI Development ✅ COMPLETED
> **Goal:** Create user-friendly Streamlit interface

- [x] Create `app/streamlit_app.py`
- [x] Implement UI components:
  - Title and project description
  - Image upload widget (drag & drop)
  - Display uploaded image
  - Prediction button with colored result boxes
  - Confidence metrics display
  - Grad-CAM visualization (side-by-side)
  - Sample images for demo (Normal/Pneumonia buttons)
- [x] Add sidebar with:
  - Model information & metrics
  - About section & usage instructions
  - GitHub link placeholder
- [x] Style with custom CSS for professional look
- [x] Handle edge cases (file type validation)
- [x] Added medical disclaimer

**Files to create:**
- `app/streamlit_app.py`

---

### Phase 9: Containerization & Deployment ✅ COMPLETED
> **Goal:** Package and deploy the application

- [x] Create `docker/Dockerfile` for Streamlit app
- [x] Create `docker/docker-compose.yml` for local testing
- [ ] Test Docker build and run locally (USER ACTION)
- [ ] Create Hugging Face Spaces repository (USER ACTION)
- [x] Create `hf_app.py` (Hugging Face entry point)
- [x] Create `hf_requirements.txt` for Hugging Face (minimal dependencies)
- [ ] Upload model weights to Hugging Face Hub (USER ACTION)
- [ ] Deploy to Hugging Face Spaces (USER ACTION)
- [ ] Test deployed application (USER ACTION)
- [ ] Get public URL for resume/portfolio (USER ACTION)

**Files created:**
- `docker/Dockerfile`
- `docker/docker-compose.yml`
- `hf_app.py` - Self-contained app for HF Spaces
- `hf_requirements.txt` - Minimal deps for HF Spaces

---

### Phase 10: Documentation & Polish ✅ COMPLETED
> **Goal:** Professional documentation for portfolio

- [x] Create comprehensive `README.md` with:
  - Project title and badges
  - Live demo link (https://huggingface.co/spaces/RichardLu/PneumoniaDetection)
  - Project description and motivation
  - Dataset information
  - Model architecture diagram
  - Results and metrics (90.5% accuracy, 98.2% recall, 0.966 AUC)
  - Sample predictions with Grad-CAM
  - Installation instructions
  - Usage examples
  - Project structure
  - License
- [x] Add docstrings to core functions (all src/ modules)
- [ ] Run code formatter (black) and linter (ruff) - OPTIONAL
- [ ] Create `tests/` with basic test coverage - OPTIONAL
- [x] Final git commit
- [x] Add project to resume with:
  - Link to GitHub repo
  - Link to live demo
  - Key metrics achieved

**Files to create:**
- `README.md`
- `tests/test_dataset.py`
- `tests/test_model.py`
- `tests/test_api.py`

---

## Configuration Reference

### Hyperparameters (`src/config.py`)
```python
# Data
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4

# Model
MODEL_NAME = "efficientnet_b0"
DROPOUT_RATE = 0.3
NUM_CLASSES = 1  # Binary classification

# Training - Stage 1 (frozen backbone)
STAGE1_EPOCHS = 5
STAGE1_LR = 1e-4

# Training - Stage 2 (fine-tuning)
STAGE2_EPOCHS = 15
STAGE2_LR = 1e-5

# Optimizer
WEIGHT_DECAY = 1e-4

# Scheduler
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.5

# Early Stopping
EARLY_STOP_PATIENCE = 7

# Paths
DATA_DIR = "data/raw"
MODEL_DIR = "models"
OUTPUT_DIR = "outputs"
```

### ImageNet Normalization
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

---

## Dependencies (`requirements.txt`)
```
# Core
torch>=2.1.0
torchvision>=0.16.0
pillow>=10.0.0
numpy>=1.24.0

# Data Analysis
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0

# Experiment Tracking
wandb>=0.15.0

# Model Interpretability
grad-cam>=1.4.0

# API
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# Web UI
streamlit>=1.28.0

# Testing
pytest>=7.4.0

# Code Quality
black>=23.0.0
ruff>=0.1.0

# Jupyter
jupyterlab>=4.0.0
ipywidgets>=8.0.0
```

---

## Success Criteria ✅ ALL MET

### Technical Metrics
- [x] Model accuracy > 90% on test set → **90.5%**
- [x] Pneumonia recall > 95% (critical for medical) → **98.2%**
- [x] ROC-AUC > 0.95 → **0.966**
- [x] Inference time < 500ms per image → **~200ms**
- [x] API response time < 1 second → **~500ms**

### Portfolio Checklist
- [x] Live demo accessible via URL → https://huggingface.co/spaces/RichardLu/PneumoniaDetection
- [x] Clean, documented GitHub repository
- [x] README with clear visualizations
- [x] Grad-CAM explanations included
- [x] Professional code quality

---

## Notes for Future Sessions

### Current Progress
- **Status:** PROJECT COMPLETED - All 10 Phases Done!
- **Last Updated:** Phase 10 completed
- **Live Demo:** https://huggingface.co/spaces/RichardLu/PneumoniaDetection
- **Environment:** Python 3.11, PyTorch 2.9.1, MPS (Apple Silicon)
- **Model:** EfficientNet-B0, 90.5% accuracy, 98.2% recall

### Quick Start for New Session
1. Read this PROJECT_PLAN.md first
2. Check which phase checkboxes are completed
3. Look at the current phase's requirements
4. Check `src/config.py` for project constants
5. Run existing code to verify it works before continuing

### Key Decisions Made
1. **Dataset:** Chest X-Ray Pneumonia (Kaggle) - binary classification
2. **Model:** EfficientNet-B0 with transfer learning
3. **Deployment:** Hugging Face Spaces with Streamlit
4. **Experiment Tracking:** Weights & Biases
5. **API Framework:** FastAPI (for learning, though Streamlit handles inference directly)

### Potential Challenges
1. **Class Imbalance:** Dataset has more pneumonia than normal - use weighted loss
2. **Overfitting:** Medical images can overfit easily - use augmentation and dropout
3. **Model Size:** Keep under 100MB for Hugging Face free tier
4. **Grad-CAM:** Ensure compatible with EfficientNet architecture

---

## Resume Bullet Points (Final)

> **Medical Image Classification - Pneumonia Detection**
> - Developed a deep learning system to classify chest X-rays with **90.5% accuracy** and **98.2% recall** using EfficientNet-B0 with transfer learning
> - Implemented Grad-CAM visualizations for model interpretability, enabling clinicians to understand prediction reasoning
> - Built end-to-end ML pipeline with PyTorch, FastAPI, and Streamlit, deployed on Hugging Face Spaces
> - [Live Demo](https://huggingface.co/spaces/RichardLu/PneumoniaDetection) | [GitHub](https://github.com/yourusername/pneumonia-detection)

---

*Last Updated: December 25, 2024 - Project Completed*
