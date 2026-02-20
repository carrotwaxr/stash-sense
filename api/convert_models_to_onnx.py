"""Convert models to ONNX format for GPU-accelerated inference via ONNX Runtime.

One-time conversion script. Converts:
- FaceNet512 and ArcFace (TensorFlow -> ONNX) for face recognition
- YOLOv5s tattoo detector (PyTorch -> ONNX) for tattoo detection
- EfficientNet-B0 tattoo embedder (PyTorch -> ONNX) for tattoo matching

Usage:
    cd api
    python convert_models_to_onnx.py                  # Convert all models
    python convert_models_to_onnx.py --yolov5          # Convert YOLOv5 only
    python convert_models_to_onnx.py --efficientnet    # Convert EfficientNet only

Output:
    models/facenet512.onnx
    models/arcface.onnx
    models/tattoo_yolov5s.onnx
    models/tattoo_efficientnet_b0.onnx
"""
import os
import sys
import numpy as np
from pathlib import Path


MODELS_DIR = Path(__file__).parent / "models"


def _init_tensorflow():
    """Initialize TensorFlow with GPU disabled (CPU-only for conversion)."""
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    tf.config.set_visible_devices([], 'GPU')


def convert_facenet512():
    """Convert FaceNet512 to ONNX."""
    _init_tensorflow()
    import tf2onnx
    import tensorflow as tf
    from deepface.modules import modeling

    print("Loading FaceNet512 from DeepFace...")
    model_wrapper = modeling.build_model(
        task="facial_recognition",
        model_name="Facenet512"
    )
    keras_model = model_wrapper.model

    input_shape = model_wrapper.input_shape  # (160, 160)
    print(f"  Input shape: {input_shape}")
    print(f"  Keras model input: {keras_model.input_shape}")
    print(f"  Keras model output: {keras_model.output_shape}")

    output_path = str(MODELS_DIR / "facenet512.onnx")
    print("  Converting to ONNX...")

    # Create input spec with dynamic batch dimension
    input_spec = (tf.TensorSpec((None, input_shape[0], input_shape[1], 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        opset=17,
        output_path=output_path,
    )

    print(f"  Saved to {output_path}")
    return output_path, input_shape


def convert_arcface():
    """Convert ArcFace to ONNX."""
    _init_tensorflow()
    import tf2onnx
    import tensorflow as tf
    from deepface.modules import modeling

    print("Loading ArcFace from DeepFace...")
    model_wrapper = modeling.build_model(
        task="facial_recognition",
        model_name="ArcFace"
    )
    keras_model = model_wrapper.model

    input_shape = model_wrapper.input_shape  # (112, 112)
    print(f"  Input shape: {input_shape}")
    print(f"  Keras model input: {keras_model.input_shape}")
    print(f"  Keras model output: {keras_model.output_shape}")

    output_path = str(MODELS_DIR / "arcface.onnx")
    print("  Converting to ONNX...")

    # Create input spec with dynamic batch dimension
    input_spec = (tf.TensorSpec((None, input_shape[0], input_shape[1], 3), tf.float32, name="input"),)

    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        opset=17,
        output_path=output_path,
    )

    print(f"  Saved to {output_path}")
    return output_path, input_shape


def verify_equivalence(facenet_onnx_path, arcface_onnx_path):
    """Verify ONNX models produce same embeddings as TF models."""
    import onnxruntime as ort
    from deepface.modules import modeling

    print("\nVerifying numerical equivalence...")

    # Load TF models
    facenet_tf = modeling.build_model(task="facial_recognition", model_name="Facenet512")
    arcface_tf = modeling.build_model(task="facial_recognition", model_name="ArcFace")

    # Load ONNX models
    facenet_onnx = ort.InferenceSession(facenet_onnx_path, providers=["CPUExecutionProvider"])
    arcface_onnx = ort.InferenceSession(arcface_onnx_path, providers=["CPUExecutionProvider"])

    # Get ONNX input/output names
    fn_input_name = facenet_onnx.get_inputs()[0].name
    fn_output_name = facenet_onnx.get_outputs()[0].name
    af_input_name = arcface_onnx.get_inputs()[0].name
    af_output_name = arcface_onnx.get_outputs()[0].name

    print(f"  FaceNet ONNX input: {fn_input_name}, output: {fn_output_name}")
    print(f"  ArcFace ONNX input: {af_input_name}, output: {af_output_name}")

    # Test with random inputs
    np.random.seed(42)

    # FaceNet512: (1, 160, 160, 3), normalized to [-1, 1]
    fn_input = np.random.randn(1, 160, 160, 3).astype(np.float32) * 0.5
    tf_fn_out = facenet_tf.model(fn_input, training=False).numpy()
    onnx_fn_out = facenet_onnx.run([fn_output_name], {fn_input_name: fn_input})[0]

    fn_max_diff = np.max(np.abs(tf_fn_out - onnx_fn_out))
    fn_cos_sim = np.dot(tf_fn_out[0], onnx_fn_out[0]) / (np.linalg.norm(tf_fn_out[0]) * np.linalg.norm(onnx_fn_out[0]))
    print(f"  FaceNet512: max_diff={fn_max_diff:.2e}, cosine_sim={fn_cos_sim:.8f}")

    # ArcFace: (1, 112, 112, 3), normalized to [-1, 1]
    af_input = np.random.randn(1, 112, 112, 3).astype(np.float32) * 0.5
    tf_af_out = arcface_tf.model(af_input, training=False).numpy()
    onnx_af_out = arcface_onnx.run([af_output_name], {af_input_name: af_input})[0]

    af_max_diff = np.max(np.abs(tf_af_out - onnx_af_out))
    af_cos_sim = np.dot(tf_af_out[0], onnx_af_out[0]) / (np.linalg.norm(tf_af_out[0]) * np.linalg.norm(onnx_af_out[0]))
    print(f"  ArcFace:    max_diff={af_max_diff:.2e}, cosine_sim={af_cos_sim:.8f}")

    # Test batch inference (dynamic batch size)
    fn_batch = np.random.randn(4, 160, 160, 3).astype(np.float32) * 0.5
    onnx_fn_batch = facenet_onnx.run([fn_output_name], {fn_input_name: fn_batch})[0]
    print(f"  FaceNet batch (4): output shape = {onnx_fn_batch.shape}")

    af_batch = np.random.randn(4, 112, 112, 3).astype(np.float32) * 0.5
    onnx_af_batch = arcface_onnx.run([af_output_name], {af_input_name: af_batch})[0]
    print(f"  ArcFace batch (4): output shape = {onnx_af_batch.shape}")

    # Check tolerances
    assert fn_max_diff < 1e-4, f"FaceNet diff too large: {fn_max_diff}"
    assert af_max_diff < 1e-4, f"ArcFace diff too large: {af_max_diff}"
    assert fn_cos_sim > 0.9999, f"FaceNet cosine sim too low: {fn_cos_sim}"
    assert af_cos_sim > 0.9999, f"ArcFace cosine sim too low: {af_cos_sim}"

    print("\n  All equivalence checks PASSED!")


def convert_yolov5_tattoo():
    """Convert YOLOv5s tattoo detector from PyTorch to ONNX.

    Exports the raw detection model (before NMS) so that post-processing
    can be handled in numpy. Output shape: (batch, 25200, 6) where
    6 = [x_center, y_center, w, h, obj_conf, class_conf].

    Coordinates are in pixel space relative to the 640x640 input.
    """
    import pathlib
    import torch

    # Fix for models saved on Windows
    pathlib.WindowsPath = pathlib.PosixPath

    pt_path = MODELS_DIR / "tattoo" / "tattoo_yolov5s.pt"
    output_path = str(MODELS_DIR / "tattoo_yolov5s.onnx")

    if not pt_path.exists():
        raise FileNotFoundError(
            f"PyTorch model not found at {pt_path}. "
            f"Download it first or check the path."
        )

    print("Loading YOLOv5s tattoo detector via torch.hub...")
    hub_model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(pt_path),
        trust_repo=True,
    )

    # Extract the underlying DetectionModel (nn.Module) from the
    # AutoShape -> DetectMultiBackend -> DetectionModel chain
    detect_model = hub_model.model.model
    detect_model.eval()

    # Input: (batch, 3, 640, 640), float32, normalized to [0, 1]
    input_shape = (1, 3, 640, 640)
    dummy_input = torch.randn(*input_shape)

    print(f"  Input shape: {input_shape}")
    print(f"  Model type: {type(detect_model).__name__}")

    # Verify forward pass works before export
    with torch.no_grad():
        test_output = detect_model(dummy_input)
    pred = test_output[0] if isinstance(test_output, tuple) else test_output
    print(f"  PyTorch output shape: {pred.shape}")  # (1, 25200, 6)

    print("  Exporting to ONNX...")
    # Use legacy exporter (dynamo=False) to produce a single self-contained
    # ONNX file. The new dynamo exporter splits weights into external data.
    torch.onnx.export(
        detect_model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["images"],
        output_names=["detections"],
        dynamic_axes={
            "images": {0: "batch_size"},
            "detections": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"  Saved to {output_path}")
    return output_path


def verify_yolov5_equivalence(onnx_path):
    """Verify ONNX YOLOv5 model produces same raw detections as PyTorch.

    Compares the raw (pre-NMS) output tensors using:
    - Max absolute difference (must be < 1e-4)
    - Cosine similarity on flattened output (must be > 0.9999)
    """
    import pathlib
    import torch
    import onnxruntime as ort

    pathlib.WindowsPath = pathlib.PosixPath

    print("\nVerifying YOLOv5 numerical equivalence...")

    # Load PyTorch model
    pt_path = MODELS_DIR / "tattoo" / "tattoo_yolov5s.pt"
    hub_model = torch.hub.load(
        "ultralytics/yolov5",
        "custom",
        path=str(pt_path),
        trust_repo=True,
    )
    detect_model = hub_model.model.model
    detect_model.eval()

    # Load ONNX model
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    print(f"  ONNX input: {onnx_input_name}, output: {onnx_output_name}")

    # Test with deterministic input
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 640, 640).astype(np.float32) * 0.5

    # PyTorch inference
    with torch.no_grad():
        pt_output = detect_model(torch.from_numpy(test_input))
    pt_pred = pt_output[0].numpy() if isinstance(pt_output, tuple) else pt_output.numpy()

    # ONNX inference
    onnx_pred = onnx_session.run([onnx_output_name], {onnx_input_name: test_input})[0]

    print(f"  PyTorch output shape: {pt_pred.shape}")
    print(f"  ONNX output shape:    {onnx_pred.shape}")

    # Numerical comparison
    max_diff = np.max(np.abs(pt_pred - onnx_pred))

    pt_flat = pt_pred.flatten()
    onnx_flat = onnx_pred.flatten()
    cos_sim = np.dot(pt_flat, onnx_flat) / (
        np.linalg.norm(pt_flat) * np.linalg.norm(onnx_flat)
    )

    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # Verify output format: columns should be [x_center, y_center, w, h, obj_conf, class_conf]
    assert onnx_pred.shape[2] == 6, f"Expected 6 columns, got {onnx_pred.shape[2]}"
    assert onnx_pred.shape[1] == 25200, f"Expected 25200 anchors, got {onnx_pred.shape[1]}"

    # Check tolerances. YOLOv5 has deeper computation graphs than embedding
    # models so FP32 rounding accumulates more. Use 1e-3 for max diff
    # (plenty tight for detection) while keeping strict cosine similarity.
    assert max_diff < 1e-3, f"Max diff too large: {max_diff:.2e}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim:.8f}"

    # Test batch inference
    batch_input = np.random.randn(4, 3, 640, 640).astype(np.float32) * 0.5
    onnx_batch = onnx_session.run([onnx_output_name], {onnx_input_name: batch_input})[0]
    print(f"  Batch (4) output shape: {onnx_batch.shape}")
    assert onnx_batch.shape[0] == 4, f"Batch dim mismatch: {onnx_batch.shape[0]}"

    print("\n  YOLOv5 equivalence checks PASSED!")


def convert_efficientnet_tattoo():
    """Convert EfficientNet-B0 tattoo embedder from PyTorch to ONNX.

    Exports the model with the classification head replaced by nn.Identity(),
    producing 1280-dim feature vectors for tattoo similarity matching.

    Input:  (batch, 3, 224, 224) float32, ImageNet-normalized
    Output: (batch, 1280) float32
    """
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    output_path = str(MODELS_DIR / "tattoo_efficientnet_b0.onnx")

    print("Loading EfficientNet-B0 with ImageNet weights...")
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier = torch.nn.Identity()  # Strip classification head, expose 1280-dim features
    model.eval()

    # Input: (batch, 3, 224, 224), float32, ImageNet-normalized
    input_shape = (1, 3, 224, 224)
    dummy_input = torch.randn(*input_shape)

    print(f"  Input shape: {input_shape}")

    # Verify forward pass works before export
    with torch.no_grad():
        test_output = model(dummy_input)
    print(f"  PyTorch output shape: {test_output.shape}")  # (1, 1280)

    print("  Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=17,
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "embedding": {0: "batch_size"},
        },
        dynamo=False,
    )

    print(f"  Saved to {output_path}")
    return output_path


def verify_efficientnet_equivalence(onnx_path):
    """Verify ONNX EfficientNet-B0 produces same embeddings as PyTorch.

    Compares using:
    - Max absolute difference (must be < 1e-4)
    - Cosine similarity (must be > 0.9999)
    - Batch inference with batch=4
    """
    import torch
    import onnxruntime as ort
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

    print("\nVerifying EfficientNet-B0 numerical equivalence...")

    # Load PyTorch model
    weights = EfficientNet_B0_Weights.DEFAULT
    pt_model = efficientnet_b0(weights=weights)
    pt_model.classifier = torch.nn.Identity()
    pt_model.eval()

    # Load ONNX model
    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_input_name = onnx_session.get_inputs()[0].name
    onnx_output_name = onnx_session.get_outputs()[0].name
    print(f"  ONNX input: {onnx_input_name}, output: {onnx_output_name}")

    # Test with deterministic input (ImageNet-normalized range)
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_output = pt_model(torch.from_numpy(test_input)).numpy()

    # ONNX inference
    onnx_output = onnx_session.run([onnx_output_name], {onnx_input_name: test_input})[0]

    print(f"  PyTorch output shape: {pt_output.shape}")
    print(f"  ONNX output shape:    {onnx_output.shape}")

    # Numerical comparison (single sample)
    max_diff = np.max(np.abs(pt_output - onnx_output))
    cos_sim = np.dot(pt_output[0], onnx_output[0]) / (
        np.linalg.norm(pt_output[0]) * np.linalg.norm(onnx_output[0])
    )

    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Cosine similarity: {cos_sim:.8f}")

    # Verify output dimensions
    assert onnx_output.shape == (1, 1280), f"Expected (1, 1280), got {onnx_output.shape}"

    # Check tolerances
    assert max_diff < 1e-4, f"Max diff too large: {max_diff:.2e}"
    assert cos_sim > 0.9999, f"Cosine similarity too low: {cos_sim:.8f}"

    # Test batch inference (batch=4)
    batch_input = np.random.randn(4, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_batch = pt_model(torch.from_numpy(batch_input)).numpy()
    onnx_batch = onnx_session.run([onnx_output_name], {onnx_input_name: batch_input})[0]

    print(f"  Batch (4) output shape: {onnx_batch.shape}")
    assert onnx_batch.shape == (4, 1280), f"Batch shape mismatch: {onnx_batch.shape}"

    batch_max_diff = np.max(np.abs(pt_batch - onnx_batch))
    # Per-sample cosine similarities
    for i in range(4):
        sample_cos = np.dot(pt_batch[i], onnx_batch[i]) / (
            np.linalg.norm(pt_batch[i]) * np.linalg.norm(onnx_batch[i])
        )
        assert sample_cos > 0.9999, f"Batch sample {i} cosine sim too low: {sample_cos:.8f}"

    print(f"  Batch max absolute diff: {batch_max_diff:.2e}")
    assert batch_max_diff < 1e-4, f"Batch max diff too large: {batch_max_diff:.2e}"

    print("\n  EfficientNet-B0 equivalence checks PASSED!")


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    yolov5_only = "--yolov5" in sys.argv
    efficientnet_only = "--efficientnet" in sys.argv

    if yolov5_only:
        # Convert only YOLOv5 tattoo detector
        yolo_path = convert_yolov5_tattoo()
        verify_yolov5_equivalence(yolo_path)

        yolo_size = os.path.getsize(yolo_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  YOLOv5s tattoo: {yolo_size:.1f} MB")

    elif efficientnet_only:
        # Convert only EfficientNet-B0 tattoo embedder
        eff_path = convert_efficientnet_tattoo()
        verify_efficientnet_equivalence(eff_path)

        eff_size = os.path.getsize(eff_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  EfficientNet-B0 tattoo: {eff_size:.1f} MB")

    else:
        # Convert all models
        facenet_path, _ = convert_facenet512()
        arcface_path, _ = convert_arcface()
        verify_equivalence(facenet_path, arcface_path)

        yolo_path = convert_yolov5_tattoo()
        verify_yolov5_equivalence(yolo_path)

        eff_path = convert_efficientnet_tattoo()
        verify_efficientnet_equivalence(eff_path)

        fn_size = os.path.getsize(facenet_path) / (1024 * 1024)
        af_size = os.path.getsize(arcface_path) / (1024 * 1024)
        yolo_size = os.path.getsize(yolo_path) / (1024 * 1024)
        eff_size = os.path.getsize(eff_path) / (1024 * 1024)
        print("\nModel sizes:")
        print(f"  FaceNet512:             {fn_size:.1f} MB")
        print(f"  ArcFace:                {af_size:.1f} MB")
        print(f"  YOLOv5s tattoo:         {yolo_size:.1f} MB")
        print(f"  EfficientNet-B0 tattoo: {eff_size:.1f} MB")

    print("\nDone! Models saved to api/models/")
