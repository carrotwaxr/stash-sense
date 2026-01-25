"""Face detection and embedding generation using DeepFace."""
import io
import numpy as np
from PIL import Image
from typing import Optional
from dataclasses import dataclass

# Set DeepFace home before importing
import os
os.environ["DEEPFACE_HOME"] = os.path.dirname(os.path.abspath(__file__))

from deepface import DeepFace
from deepface.modules import modeling, preprocessing


@dataclass
class DetectedFace:
    """A detected face with its bounding box and confidence."""
    image: np.ndarray  # RGB image of the face
    bbox: dict  # {x, y, w, h}
    confidence: float


@dataclass
class FaceEmbedding:
    """Face embeddings from multiple models."""
    facenet: np.ndarray  # 512-dim FaceNet512 embedding
    arcface: np.ndarray  # 512-dim ArcFace embedding


class FaceEmbeddingGenerator:
    """Generate face embeddings using FaceNet512 and ArcFace models."""

    def __init__(self, detector_backend: str = "opencv"):
        """
        Initialize the embedding generator.

        Args:
            detector_backend: Face detector to use ("yolov8", "mediapipe", "retinaface")
        """
        self.detector_backend = detector_backend
        self._facenet_model = None
        self._arcface_model = None

    @property
    def facenet_model(self):
        """Lazy-load FaceNet512 model."""
        if self._facenet_model is None:
            print("Loading FaceNet512 model...")
            self._facenet_model = modeling.build_model(
                task="facial_recognition",
                model_name="Facenet512"
            )
        return self._facenet_model

    @property
    def arcface_model(self):
        """Lazy-load ArcFace model."""
        if self._arcface_model is None:
            print("Loading ArcFace model...")
            self._arcface_model = modeling.build_model(
                task="facial_recognition",
                model_name="ArcFace"
            )
        return self._arcface_model

    def detect_faces(
        self,
        image: np.ndarray,
        enforce_detection: bool = False,
    ) -> list[DetectedFace]:
        """
        Detect faces in an image.

        Args:
            image: RGB image as numpy array
            enforce_detection: Raise error if no faces found

        Returns:
            List of DetectedFace objects
        """
        try:
            faces = DeepFace.extract_faces(
                image,
                detector_backend=self.detector_backend,
                enforce_detection=enforce_detection,
            )
        except ValueError:
            return []

        return [
            DetectedFace(
                image=face["face"],
                bbox=face["facial_area"],
                confidence=face.get("confidence", 1.0),
            )
            for face in faces
        ]

    def _preprocess_face(
        self,
        face: np.ndarray,
        target_size: tuple[int, int],
        normalization: str,
    ) -> np.ndarray:
        """Preprocess a face image for model inference."""
        # Convert RGB to BGR (DeepFace expects BGR)
        face_bgr = face[:, :, ::-1]

        # Resize to model input size
        resized = preprocessing.resize_image(face_bgr, target_size)

        # Normalize
        normalized = preprocessing.normalize_input(resized, normalization)

        return normalized

    def get_embedding(self, face: np.ndarray) -> FaceEmbedding:
        """
        Generate embeddings for a single face image.

        Args:
            face: RGB face image as numpy array (from detect_faces)

        Returns:
            FaceEmbedding with FaceNet and ArcFace embeddings
        """
        # Create batch with original and horizontally flipped images
        # This improves accuracy by averaging embeddings
        face_flipped = face[:, ::-1, :]
        face_batch = np.stack([face, face_flipped], axis=0)

        # Get FaceNet embeddings
        facenet_batch = np.vstack([
            self._preprocess_face(f, self.facenet_model.input_shape, "Facenet2018")
            for f in face_batch
        ])
        facenet_embeddings = self.facenet_model.model(facenet_batch, training=False).numpy()
        facenet = np.mean(facenet_embeddings, axis=0)

        # Get ArcFace embeddings
        arcface_batch = np.vstack([
            self._preprocess_face(f, self.arcface_model.input_shape, "ArcFace")
            for f in face_batch
        ])
        arcface_embeddings = self.arcface_model.model(arcface_batch, training=False).numpy()
        arcface = np.mean(arcface_embeddings, axis=0)

        return FaceEmbedding(facenet=facenet, arcface=arcface)

    def get_embeddings_batch(self, faces: list[np.ndarray]) -> list[FaceEmbedding]:
        """
        Generate embeddings for a batch of faces.

        More efficient than calling get_embedding repeatedly.
        """
        if not faces:
            return []

        # Process all faces with flipped versions
        all_faces = []
        for face in faces:
            all_faces.append(face)
            all_faces.append(face[:, ::-1, :])  # flipped

        # FaceNet batch
        facenet_batch = np.vstack([
            self._preprocess_face(f, self.facenet_model.input_shape, "Facenet2018")
            for f in all_faces
        ])
        facenet_all = self.facenet_model.model(facenet_batch, training=False).numpy()

        # ArcFace batch
        arcface_batch = np.vstack([
            self._preprocess_face(f, self.arcface_model.input_shape, "ArcFace")
            for f in all_faces
        ])
        arcface_all = self.arcface_model.model(arcface_batch, training=False).numpy()

        # Average original and flipped for each face
        embeddings = []
        for i in range(len(faces)):
            facenet = np.mean(facenet_all[i*2:(i+1)*2], axis=0)
            arcface = np.mean(arcface_all[i*2:(i+1)*2], axis=0)
            embeddings.append(FaceEmbedding(facenet=facenet, arcface=arcface))

        return embeddings


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

    generator = FaceEmbeddingGenerator()

    # Download a test image
    test_url = "https://stashdb.org/images/b0aef39d-a1d6-4e58-a136-293f02b84921"
    print(f"Downloading test image from {test_url}...")

    response = requests.get(test_url)
    image = load_image(response.content)
    print(f"Image shape: {image.shape}")

    # Detect faces
    print("Detecting faces...")
    faces = generator.detect_faces(image)
    print(f"Found {len(faces)} face(s)")

    if faces:
        # Get embeddings
        print("Generating embeddings...")
        embedding = generator.get_embedding(faces[0].image)
        print(f"FaceNet embedding shape: {embedding.facenet.shape}")
        print(f"ArcFace embedding shape: {embedding.arcface.shape}")
        print(f"FaceNet first 5 values: {embedding.facenet[:5]}")
        print(f"ArcFace first 5 values: {embedding.arcface[:5]}")
