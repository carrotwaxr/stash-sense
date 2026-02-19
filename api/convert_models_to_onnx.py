"""Convert DeepFace FaceNet512 and ArcFace models from TensorFlow to ONNX.

One-time conversion script. Run this to generate ONNX models that can be
used with ONNX Runtime (GPU-accelerated) instead of TensorFlow (CPU-only
on Blackwell GPUs).

Usage:
    cd api
    python convert_models_to_onnx.py

Output:
    models/facenet512.onnx
    models/arcface.onnx
"""
import os
import numpy as np
from pathlib import Path

# Suppress TF warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

import tf2onnx
from deepface.modules import modeling


MODELS_DIR = Path(__file__).parent / "models"


def convert_facenet512():
    """Convert FaceNet512 to ONNX."""
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


if __name__ == "__main__":
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    facenet_path, _ = convert_facenet512()
    arcface_path, _ = convert_arcface()

    verify_equivalence(facenet_path, arcface_path)

    # Print file sizes
    fn_size = os.path.getsize(facenet_path) / (1024 * 1024)
    af_size = os.path.getsize(arcface_path) / (1024 * 1024)
    print("\nModel sizes:")
    print(f"  FaceNet512: {fn_size:.1f} MB")
    print(f"  ArcFace:    {af_size:.1f} MB")
    print("\nDone! Models saved to api/models/")
