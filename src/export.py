"""
ONNX export utilities for model deployment.

ONNX (Open Neural Network Exchange) is a universal format that allows
models to run on different frameworks and platforms:
- TensorFlow, PyTorch, etc.
- Mobile devices (iOS, Android)
- Web browsers (ONNX.js)
- C++, Java, and other languages
- Optimized inference servers
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from .config import CHECKPOINT_PATH, MODEL_DIR, IMAGE_SIZE
from .model import create_model, get_device


def export_to_onnx(
    checkpoint_path: Path = CHECKPOINT_PATH,
    output_path: Optional[Path] = None,
    opset_version: int = 18
) -> Path:
    """
    Export PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to the PyTorch checkpoint
        output_path: Path for the ONNX model (default: models/best_model.onnx)
        opset_version: ONNX opset version (14 is widely compatible)

    Returns:
        Path to the exported ONNX model
    """
    if output_path is None:
        output_path = MODEL_DIR / "best_model.onnx"

    # Load model
    device = torch.device("cpu")  # Export on CPU for compatibility
    model = create_model(pretrained=False, freeze_backbone=False, device=device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create dummy input (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,  # Optimize constants
        input_names=['image'],
        output_names=['logits'],
        dynamic_axes={
            'image': {0: 'batch_size'},   # Variable batch size
            'logits': {0: 'batch_size'}
        }
    )

    print(f"Model exported to: {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    return output_path


def validate_onnx_model(
    onnx_path: Path,
    checkpoint_path: Path = CHECKPOINT_PATH,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> bool:
    """
    Validate that ONNX model produces same outputs as PyTorch model.

    Args:
        onnx_path: Path to ONNX model
        checkpoint_path: Path to PyTorch checkpoint
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        True if outputs match, False otherwise
    """
    import onnx
    import onnxruntime as ort

    # Check ONNX model is valid
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure is valid")

    # Load PyTorch model
    device = torch.device("cpu")
    model = create_model(pretrained=False, freeze_backbone=False, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Create test input
    test_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = model(test_input).numpy()

    # Get ONNX output
    ort_session = ort.InferenceSession(str(onnx_path))
    onnx_output = ort_session.run(
        None,
        {'image': test_input.numpy()}
    )[0]

    # Compare outputs
    is_close = np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)

    if is_close:
        print("Validation PASSED: ONNX outputs match PyTorch outputs")
        print(f"  PyTorch output: {pytorch_output.flatten()[:5]}...")
        print(f"  ONNX output:    {onnx_output.flatten()[:5]}...")
    else:
        print("Validation FAILED: Outputs do not match!")
        print(f"  Max difference: {np.max(np.abs(pytorch_output - onnx_output))}")

    return is_close


def predict_with_onnx(
    onnx_path: Path,
    image_tensor: np.ndarray
) -> Tuple[str, float]:
    """
    Run inference using ONNX Runtime.

    Args:
        onnx_path: Path to ONNX model
        image_tensor: Preprocessed image as numpy array (1, 3, 224, 224)

    Returns:
        Tuple of (predicted_class, confidence)
    """
    import onnxruntime as ort
    from .config import CLASS_NAMES

    # Create session
    ort_session = ort.InferenceSession(str(onnx_path))

    # Run inference
    logits = ort_session.run(
        None,
        {'image': image_tensor.astype(np.float32)}
    )[0]

    # Apply sigmoid and get prediction
    prob = 1 / (1 + np.exp(-logits[0, 0]))  # Sigmoid
    pred_class = CLASS_NAMES[1] if prob > 0.5 else CLASS_NAMES[0]
    confidence = float(prob if prob > 0.5 else 1 - prob)

    return pred_class, confidence


if __name__ == "__main__":
    # Export model
    print("=" * 50)
    print("EXPORTING MODEL TO ONNX")
    print("=" * 50)

    onnx_path = export_to_onnx()

    print("\n" + "=" * 50)
    print("VALIDATING ONNX MODEL")
    print("=" * 50)

    validate_onnx_model(onnx_path)

    print("\n" + "=" * 50)
    print("TESTING ONNX INFERENCE")
    print("=" * 50)

    # Test with random input
    test_input = np.random.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    pred_class, confidence = predict_with_onnx(onnx_path, test_input)
    print(f"Test prediction: {pred_class} ({confidence:.1%})")
