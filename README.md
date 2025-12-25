# Pneumonia Detection from Chest X-Rays

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-yellow)](https://huggingface.co/spaces/RichardLu/PneumoniaDetection)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red)](https://pytorch.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)](https://streamlit.io)

A deep learning system for detecting pneumonia from chest X-ray images using EfficientNet-B0 with Grad-CAM explainability.

![Demo Screenshot](outputs/figures/gradcam_correct.png)

## Key Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 90.5% |
| **Recall** | 98.2% |
| **Precision** | 88.0% |
| **F1 Score** | 92.8% |
| **AUC-ROC** | 0.966 |

> High recall (98.2%) prioritized to minimize missed pneumonia cases - critical for medical screening applications.

## Features

- **EfficientNet-B0** backbone with transfer learning
- **Grad-CAM visualization** for model interpretability
- **Threshold optimization** for recall-precision tradeoff
- **REST API** with FastAPI for integration
- **Web UI** with Streamlit for easy interaction
- **Docker support** for containerized deployment
- **ONNX export** for production inference

## Live Demo

Try the model: [https://huggingface.co/spaces/RichardLu/PneumoniaDetection](https://huggingface.co/spaces/RichardLu/PneumoniaDetection)

## Project Structure

```
pneumonia-detection/
├── api/                    # FastAPI REST API
│   ├── main.py            # API endpoints
│   └── schemas.py         # Pydantic models
├── app/                    # Streamlit Web UI
│   └── streamlit_app.py   # Main UI application
├── data/
│   ├── raw/               # Original chest X-ray dataset
│   └── processed/         # Preprocessed data splits
├── docker/                 # Docker configuration
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/                 # Trained model checkpoints
│   └── best_model.pt
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb       # Exploratory data analysis
│   ├── 02_baseline.ipynb  # Model development
│   └── 03_evaluation.ipynb # Model evaluation
├── outputs/
│   └── figures/           # Generated visualizations
├── src/                    # Core source code
│   ├── config.py          # Configuration settings
│   ├── dataset.py         # Data loading & augmentation
│   ├── model.py           # Model architecture
│   ├── train.py           # Training pipeline
│   ├── evaluate.py        # Evaluation metrics
│   ├── predict.py         # Inference utilities
│   ├── gradcam.py         # Grad-CAM implementation
│   └── export.py          # ONNX export
├── hf_app.py              # Hugging Face Spaces app
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.11+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/pneumonia-detection.git
cd pneumonia-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle and extract to `data/raw/`.

## Usage

### Training

```bash
# Train the model
python -m src.train

# With custom parameters
python -m src.train --epochs 15 --batch-size 32 --lr 1e-4
```

### Evaluation

```bash
# Evaluate on test set
python -m src.evaluate
```

### Inference

```python
from src.predict import load_model, predict_image
from src.model import create_model, get_device
from PIL import Image

# Load model
device = get_device()
model = create_model(pretrained=False, device=device)
model = load_model(model, "models/best_model.pt", device)

# Predict
image = Image.open("xray.jpg")
prediction, confidence = predict_image(model, image, device)
print(f"Prediction: {prediction}, Confidence: {confidence:.2%}")
```

### Web UI (Streamlit)

```bash
streamlit run app/streamlit_app.py
```

### REST API (FastAPI)

```bash
uvicorn api.main:app --reload
```

API Endpoints:
- `GET /health` - Health check
- `POST /predict` - Get prediction
- `POST /predict/gradcam` - Get prediction with Grad-CAM

### Docker

```bash
# Build and run
cd docker
docker-compose up --build

# Access at http://localhost:8501
```

## Model Architecture

```
EfficientNet-B0 (pretrained on ImageNet)
    │
    ├── Features: Convolutional backbone (frozen initially)
    │
    └── Classifier:
        ├── Dropout (p=0.3)
        └── Linear (1280 → 1) + Sigmoid
```

**Training Strategy:**
1. Transfer learning from ImageNet weights
2. Freeze backbone, train classifier (5 epochs)
3. Unfreeze and fine-tune entire network (10 epochs)
4. Learning rate: 1e-4 with ReduceLROnPlateau
5. Mixed precision training (FP16) for efficiency

## Results

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### ROC Curve
![ROC Curve](outputs/figures/roc_curve.png)

### Training Curves
![Training Curves](outputs/figures/training_curves.png)

### Grad-CAM Visualizations
![Grad-CAM](outputs/figures/gradcam_correct.png)

### Threshold Analysis
![Threshold Optimization](outputs/figures/threshold_optimization.png)

## Technical Highlights

- **Data Augmentation**: RandomRotation, RandomAffine, ColorJitter, RandomErasing
- **Class Imbalance**: Weighted loss function (normal: 1.94, pneumonia: 0.67)
- **Mixed Precision**: FP16 training for 2x speedup
- **Early Stopping**: Patience=5 based on validation loss
- **Threshold Tuning**: Optimized for 98%+ recall (threshold=0.50)

## Technologies

- **Deep Learning**: PyTorch, torchvision
- **Model**: EfficientNet-B0 (transfer learning)
- **Explainability**: Grad-CAM (pytorch-grad-cam)
- **API**: FastAPI, Pydantic
- **Web UI**: Streamlit
- **Deployment**: Docker, Hugging Face Spaces
- **Data Analysis**: pandas, numpy, matplotlib, seaborn

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Dataset: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) by Paul Mooney
- Model: [EfficientNet](https://arxiv.org/abs/1905.11946) by Tan & Le

---

**Note**: This tool is for educational purposes only. Always consult a qualified healthcare professional for medical diagnosis.
