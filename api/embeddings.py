"""Face detection and embedding generation.

Uses:
- InsightFace RetinaFace for face detection (ONNX Runtime, GPU-compatible)
- FaceNet512 + ArcFace for embeddings (ONNX Runtime, GPU-accelerated)

All inference runs through ONNX Runtime with CUDAExecutionProvider when
available, falling back to CPU. No TensorFlow dependency at runtime.
"""
import io
import os
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
from typing import Optional
from dataclasses import dataclass
from pathlib import Path

# InsightFace for RetinaFace detection (uses ONNX Runtime with GPU)
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop


# Model search paths: DATA_DIR/models first, then ./models (relative to this file)
MODELS_DIR = Path(__file__).parent / "models"
DATA_MODELS_DIR = Path(os.environ.get("DATA_DIR", "./data")) / "models"


@dataclass
class DetectedFace:
    """A detected face with its bounding box and confidence."""
    image: np.ndarray  # RGB image of the face (cropped and aligned)
    bbox: dict  # {x, y, w, h}
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5-point facial landmarks
    yaw: Optional[float] = None    # Estimated yaw in degrees (-90 to +90)


@dataclass
class FaceEmbedding:
    """Face embeddings from multiple models."""
    facenet: np.ndarray  # 512-dim FaceNet512 embedding
    arcface: np.ndarray  # 512-dim ArcFace embedding


class FaceEmbeddingGenerator:
    """Generate face embeddings using RetinaFace detection + FaceNet512/ArcFace embeddings.

    All models run through ONNX Runtime with GPU acceleration when available.
    Supports batch inference for processing multiple faces in a single call.
    """

    def __init__(self, device: str = None, models_dir: Path = None):
        """
        Initialize the embedding generator.

        Args:
            device: Device for inference ("cuda", "cpu", or None for auto)
            models_dir: Directory containing facenet512.onnx and arcface.onnx.
                If None, checks DATA_DIR/models first (Docker/production),
                then falls back to ./models (local development).
        """
        # Auto-detect device
        if device is None:
            providers = ort.get_available_providers()
            self.device = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        else:
            self.device = device

        # Auto-detect models directory: DATA_DIR/models first, then local ./models
        if models_dir is None:
            if (DATA_MODELS_DIR / "facenet512.onnx").exists():
                models_dir = DATA_MODELS_DIR
            else:
                models_dir = MODELS_DIR
        self._models_dir = models_dir
        self._face_analyzer = None
        self._facenet_session = None
        self._arcface_session = None

        # ONNX model I/O names (set during lazy load)
        self._fn_input_name = None
        self._fn_output_name = None
        self._af_input_name = None
        self._af_output_name = None

    def _ort_providers(self) -> list[str]:
        """Get ONNX Runtime providers based on device."""
        if self.device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def face_analyzer(self):
        """Lazy-load InsightFace face analyzer with RetinaFace."""
        if self._face_analyzer is None:
            print(f"Loading RetinaFace detector on {self.device}...")
            self._face_analyzer = FaceAnalysis(
                name="buffalo_sc",
                providers=self._ort_providers(),
            )
            # Get detection size from settings, fall back to 640
            try:
                from settings import get_setting
                det_size = int(get_setting("detection_size"))
            except (RuntimeError, KeyError):
                det_size = 640
            self._face_analyzer.prepare(
                ctx_id=0 if self.device == "cuda" else -1,
                det_size=(det_size, det_size),
            )
        return self._face_analyzer

    @property
    def facenet_session(self) -> ort.InferenceSession:
        """Lazy-load FaceNet512 ONNX model."""
        if self._facenet_session is None:
            model_path = str(self._models_dir / "facenet512.onnx")
            print(f"Loading FaceNet512 ONNX model on {self.device}...")
            self._facenet_session = ort.InferenceSession(
                model_path, providers=self._ort_providers(),
            )
            self._fn_input_name = self._facenet_session.get_inputs()[0].name
            self._fn_output_name = self._facenet_session.get_outputs()[0].name
        return self._facenet_session

    @property
    def arcface_session(self) -> ort.InferenceSession:
        """Lazy-load ArcFace ONNX model."""
        if self._arcface_session is None:
            model_path = str(self._models_dir / "arcface.onnx")
            print(f"Loading ArcFace ONNX model on {self.device}...")
            self._arcface_session = ort.InferenceSession(
                model_path, providers=self._ort_providers(),
            )
            self._af_input_name = self._arcface_session.get_inputs()[0].name
            self._af_output_name = self._arcface_session.get_outputs()[0].name
        return self._arcface_session

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

            # Get 5-point landmarks
            kps = face.kps if hasattr(face, 'kps') else None

            # Align face using InsightFace's norm_crop (5-point similarity transform)
            # This produces a 112x112 aligned face matching ArcFace's training preprocessing
            if kps is not None and len(kps) >= 5:
                face_img = norm_crop(image, kps, image_size=112)
            else:
                # Fallback: raw bbox crop (shouldn't happen with RetinaFace)
                cx1 = max(0, x1)
                cy1 = max(0, y1)
                cx2 = min(image.shape[1], x2)
                cy2 = min(image.shape[0], y2)
                face_img = image[cy1:cy2, cx1:cx2]

            # Skip if crop is too small
            if face_img.shape[0] < 10 or face_img.shape[1] < 10:
                continue

            # Estimate yaw from 5-point landmarks
            yaw_estimate = None
            if kps is not None and len(kps) >= 3:
                left_eye, right_eye, nose = kps[0], kps[1], kps[2]
                eye_center_x = (left_eye[0] + right_eye[0]) / 2
                eye_width = np.linalg.norm(right_eye - left_eye)
                if eye_width > 0:
                    nose_offset = nose[0] - eye_center_x
                    yaw_estimate = float(np.degrees(np.arctan2(nose_offset, eye_width / 2)))

            faces.append(DetectedFace(
                image=face_img,
                bbox={"x": int(x1), "y": int(y1), "w": int(w), "h": int(h)},
                confidence=conf,
                landmarks=kps,
                yaw=yaw_estimate,
            ))

        return faces

    def _preprocess_face(
        self,
        face: np.ndarray,
        target_size: tuple[int, int],
    ) -> np.ndarray:
        """Preprocess a face image for model inference.

        Uses aspect-ratio-preserving resize with black padding,
        matching DeepFace's preprocessing pipeline.

        Returns array of shape (H, W, 3) as float32 (NOT batched).
        """
        target_h, target_w = target_size
        face_h, face_w = face.shape[:2]

        # Scale to fit within target while preserving aspect ratio
        factor = min(target_h / face_h, target_w / face_w)
        new_h = int(face_h * factor)
        new_w = int(face_w * factor)

        resized = cv2.resize(face, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create black canvas and paste centered
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        y_off = (target_h - new_h) // 2
        x_off = (target_w - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        return canvas.astype(np.float32)

    def get_embedding(self, face: np.ndarray) -> FaceEmbedding:
        """
        Generate embeddings for a single face image.

        Uses flip-averaging: generates embeddings for both the original and
        horizontally flipped face, then averages them for more stable results.

        Args:
            face: RGB face image as numpy array (cropped face from detect_faces)

        Returns:
            FaceEmbedding with FaceNet and ArcFace embeddings
        """
        return self.get_embeddings_batch([face])[0]

    def get_embeddings_batch(
        self,
        faces: list[np.ndarray],
        max_batch_size: Optional[int] = None,
    ) -> list[FaceEmbedding]:
        """
        Generate embeddings for a batch of faces using ONNX Runtime.

        Processes faces in sub-batches to avoid GPU memory exhaustion.
        Each sub-batch runs 2 model calls (1 FaceNet + 1 ArcFace).

        Args:
            faces: List of RGB face images as numpy arrays
            max_batch_size: Max faces per GPU inference call (limits VRAM usage).
                           If None, reads from settings (embedding_batch_size).

        Returns:
            List of FaceEmbedding objects, one per input face
        """
        if not faces:
            return []

        if max_batch_size is None:
            try:
                from settings import get_setting
                max_batch_size = int(get_setting("embedding_batch_size"))
            except (RuntimeError, KeyError):
                max_batch_size = 16

        # Ensure models are loaded (triggers lazy init + prints)
        _ = self.facenet_session
        _ = self.arcface_session

        # Process in sub-batches to limit GPU memory
        results = []
        for batch_start in range(0, len(faces), max_batch_size):
            batch = faces[batch_start:batch_start + max_batch_size]
            results.extend(self._embed_batch(batch))

        return results

    def _embed_batch(self, faces: list[np.ndarray]) -> list[FaceEmbedding]:
        """Run embedding inference on a single sub-batch of faces."""
        n = len(faces)

        # Preprocess all faces: original + flipped for each model
        fn_inputs = np.empty((n * 2, 160, 160, 3), dtype=np.float32)
        af_inputs = np.empty((n * 2, 112, 112, 3), dtype=np.float32)

        for i, face in enumerate(faces):
            flipped = face[:, ::-1, :]

            fn_orig = self._preprocess_face(face, (160, 160))
            fn_flip = self._preprocess_face(flipped, (160, 160))
            fn_inputs[i * 2] = (fn_orig - 127.5) / 127.5
            fn_inputs[i * 2 + 1] = (fn_flip - 127.5) / 127.5

            af_orig = self._preprocess_face(face, (112, 112))
            af_flip = self._preprocess_face(flipped, (112, 112))
            af_inputs[i * 2] = (af_orig - 127.5) / 128.0
            af_inputs[i * 2 + 1] = (af_flip - 127.5) / 128.0

        # Run batch inference
        fn_embeddings = self.facenet_session.run(
            [self._fn_output_name], {self._fn_input_name: fn_inputs}
        )[0]

        af_embeddings = self.arcface_session.run(
            [self._af_output_name], {self._af_input_name: af_inputs}
        )[0]

        # Average original + flipped pairs for flip-averaging
        results = []
        for i in range(n):
            fn_emb = (fn_embeddings[i * 2] + fn_embeddings[i * 2 + 1]) / 2.0
            af_emb = (af_embeddings[i * 2] + af_embeddings[i * 2 + 1]) / 2.0
            results.append(FaceEmbedding(facenet=fn_emb, arcface=af_emb))

        return results


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
    import time
    import requests

    print("Testing face embedding generator (ONNX Runtime)...")
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

        # Warmup
        print("\nWarming up models...")
        _ = generator.get_embedding(face.image)

        # Benchmark single face
        print("Benchmarking single face...")
        t0 = time.time()
        embedding = generator.get_embedding(face.image)
        t1 = time.time()
        print(f"  Single face: {(t1-t0)*1000:.1f}ms")
        print(f"  FaceNet shape: {embedding.facenet.shape}")
        print(f"  ArcFace shape: {embedding.arcface.shape}")

        # Benchmark batch (simulate 10 faces)
        print("\nBenchmarking batch of 10 faces...")
        batch_faces = [face.image] * 10
        t0 = time.time()
        batch_results = generator.get_embeddings_batch(batch_faces)
        t1 = time.time()
        print(f"  Batch of 10: {(t1-t0)*1000:.1f}ms ({(t1-t0)*100:.1f}ms/face)")
        print(f"  Results: {len(batch_results)} embeddings")
