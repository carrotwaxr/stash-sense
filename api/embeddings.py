"""Face detection and embedding generation.

Uses:
- InsightFace RetinaFace for face detection (ONNX Runtime, GPU-compatible)
- DeepFace FaceNet512 + ArcFace for embeddings (TensorFlow, CPU-only on Blackwell GPUs)
"""
import io
import os
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# Reduce TensorFlow logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import TensorFlow and force it to CPU-only mode
# This is necessary because TensorFlow 2.20 doesn't support Blackwell GPUs (sm_120)
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs from TensorFlow

# InsightFace for RetinaFace detection (uses ONNX Runtime with GPU)
from insightface.app import FaceAnalysis

# Import DeepFace modules (will use TensorFlow on CPU)
from deepface.modules import modeling


@dataclass
class DetectedFace:
    """A detected face with its bounding box and confidence."""
    image: np.ndarray  # RGB image of the face (cropped and aligned)
    bbox: dict  # {x, y, w, h}
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5-point facial landmarks


@dataclass
class FaceEmbedding:
    """Face embeddings from multiple models."""
    facenet: np.ndarray  # 512-dim FaceNet512 embedding
    arcface: np.ndarray  # 512-dim ArcFace embedding


class FaceEmbeddingGenerator:
    """Generate face embeddings using RetinaFace detection + FaceNet512/ArcFace embeddings."""

    def __init__(self, device: str = None):
        """
        Initialize the embedding generator.

        Args:
            device: Device for detection ("cuda", "cpu", or None for auto)
        """
        # Auto-detect device
        if device is None:
            # Check if CUDA is available for ONNX Runtime
            import onnxruntime as ort
            providers = ort.get_available_providers()
            self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        else:
            self.device = device

        self._face_analyzer = None
        self._facenet_model = None
        self._arcface_model = None

    @property
    def face_analyzer(self):
        """Lazy-load InsightFace face analyzer with RetinaFace."""
        if self._face_analyzer is None:
            print(f"Loading RetinaFace detector on {self.device}...")

            # Set up providers based on device
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

            # Initialize with detection only (no recognition - we use DeepFace for that)
            self._face_analyzer = FaceAnalysis(
                name="buffalo_sc",  # Smaller model, detection-focused
                providers=providers,
            )
            # det_size controls input resolution for detection
            self._face_analyzer.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        return self._face_analyzer

    @property
    def facenet_model(self):
        """Lazy-load FaceNet512 model (runs on CPU)."""
        if self._facenet_model is None:
            print("Loading FaceNet512 model (CPU)...")
            self._facenet_model = modeling.build_model(
                task="facial_recognition",
                model_name="Facenet512"
            )
        return self._facenet_model

    @property
    def arcface_model(self):
        """Lazy-load ArcFace model (runs on CPU)."""
        if self._arcface_model is None:
            print("Loading ArcFace model (CPU)...")
            self._arcface_model = modeling.build_model(
                task="facial_recognition",
                model_name="ArcFace"
            )
        return self._arcface_model

    def detect_faces(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5,
    ) -> list[DetectedFace]:
        """
        Detect faces in an image using RetinaFace.

        Args:
            image: RGB image as numpy array
            min_confidence: Minimum detection confidence

        Returns:
            List of DetectedFace objects
        """
        # InsightFace expects BGR, but we'll handle RGB since our input is RGB
        # Actually InsightFace.get() handles RGB fine
        results = self.face_analyzer.get(image)

        faces = []
        for face in results:
            conf = float(face.det_score)
            if conf < min_confidence:
                continue

            # Get bounding box (x1, y1, x2, y2)
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1

            # Ensure bounds are within image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)

            # Crop face from image
            face_img = image[y1:y2, x1:x2]

            # Skip if crop is too small
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue

            faces.append(DetectedFace(
                image=face_img,
                bbox={"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                confidence=conf,
                landmarks=face.kps if hasattr(face, 'kps') else None,
            ))

        return faces

    def _preprocess_face(
        self,
        face: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Preprocess a face image for model inference."""
        # Resize to target size
        pil_img = Image.fromarray(face)
        pil_img = pil_img.resize(target_size, Image.Resampling.BILINEAR)
        resized = np.array(pil_img)

        # Expand dims for batch
        resized = np.expand_dims(resized, axis=0)

        # Normalize to [-1, 1] for FaceNet, [0, 1] for ArcFace handled by model
        resized = resized.astype(np.float32)

        return resized

    def get_embedding(self, face: np.ndarray) -> FaceEmbedding:
        """
        Generate embeddings for a single face image.

        Args:
            face: RGB face image as numpy array (cropped face from detect_faces)

        Returns:
            FaceEmbedding with FaceNet and ArcFace embeddings
        """
        # Get model input shapes
        facenet_size = self.facenet_model.input_shape
        arcface_size = self.arcface_model.input_shape

        # Preprocess for each model
        facenet_input = self._preprocess_face(face, facenet_size)
        arcface_input = self._preprocess_face(face, arcface_size)

        # Normalize for FaceNet (expects [-1, 1])
        facenet_input = (facenet_input - 127.5) / 127.5

        # Normalize for ArcFace (expects [0, 1])
        arcface_input = arcface_input / 255.0

        # Get embeddings
        facenet_emb = self.facenet_model.model(facenet_input, training=False).numpy()[0]
        arcface_emb = self.arcface_model.model(arcface_input, training=False).numpy()[0]

        return FaceEmbedding(facenet=facenet_emb, arcface=arcface_emb)

    def get_embeddings_batch(self, faces: list[np.ndarray]) -> list[FaceEmbedding]:
        """Generate embeddings for a batch of faces."""
        return [self.get_embedding(face) for face in faces]


def load_image(data: bytes) -> np.ndarray:
    """Load image data into numpy array."""
    image = Image.open(io.BytesIO(data))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def load_image_from_path(path: str) -> np.ndarray:
    """Load image from file path."""
    image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


if __name__ == "__main__":
    # Quick test
    import requests

    print("Testing face embedding generator...")

    # Check ONNX Runtime providers
    import onnxruntime as ort
    print(f"ONNX Runtime providers: {ort.get_available_providers()}")

    generator = FaceEmbeddingGenerator()
    print(f"Using device: {generator.device}")

    # Download a test image
    test_url = "https://stashdb.org/images/b0aef39d-a1d6-4e58-a136-293f02b84921"
    print(f"\nDownloading test image from {test_url}...")

    response = requests.get(test_url)
    image = load_image(response.content)
    print(f"Image shape: {image.shape}")

    # Detect faces
    print("\nDetecting faces...")
    faces = generator.detect_faces(image)
    print(f"Found {len(faces)} face(s)")

    if faces:
        face = faces[0]
        print(f"  Confidence: {face.confidence:.2f}")
        print(f"  Size: {face.bbox['w']}x{face.bbox['h']}")

        # Get embeddings
        print("\nGenerating embeddings...")
        embedding = generator.get_embedding(face.image)
        print(f"FaceNet embedding shape: {embedding.facenet.shape}")
        print(f"ArcFace embedding shape: {embedding.arcface.shape}")
        print(f"FaceNet first 5 values: {embedding.facenet[:5]}")
        print(f"ArcFace first 5 values: {embedding.arcface[:5]}")
