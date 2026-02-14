"""Shared face recognition configuration.

Single source of truth for parameters used by scene identification,
fingerprint generation, and face matching. Change values here to
tune all processes at once.
"""

# Frame extraction
NUM_FRAMES = 60           # Frames to sample per scene (tuned 2026-02-12 re-eval)
START_OFFSET_PCT = 0.05   # Skip first 5% (logos, intros)
END_OFFSET_PCT = 0.95     # Skip last 5% (credits, outros)

# Face detection
MIN_FACE_SIZE = 40        # Minimum face dimension in pixels
MIN_FACE_CONFIDENCE = 0.5 # Detection confidence threshold

# Matching
MAX_DISTANCE = 0.5        # Maximum match distance (tuned 2026-02-12 re-eval, plateau at 0.5)
TOP_K = 3                 # Top matches per person
CLUSTER_THRESHOLD = 0.6   # Cosine distance threshold for face clustering

# Fusion weights (when both models are healthy)
FACENET_WEIGHT = 0.5      # FaceNet contribution (tuned 2026-02-12 re-eval)
ARCFACE_WEIGHT = 0.5      # ArcFace contribution (tuned 2026-02-12 re-eval)
