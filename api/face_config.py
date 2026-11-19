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

# Gallery matching (matching_mode="gallery", #148). Tuned 2026-11-18 on the
# #139 benchmark (even scene ids, checked on odd): same recall as frequency
# mode with ~40% fewer false positives. Re-tune with
# python -m benchmark.failure replay after a face DB release.
GALLERY_CANDIDATE_K = 10              # per-face matches that nominate candidates
GALLERY_CANDIDATE_MAX_DISTANCE = 0.8  # per-face cutoff for nominating (scoring uses whole galleries)
GALLERY_KEEP_FRACTION = 0.75          # score = mean over the cluster's best 75% of faces
GALLERY_MAX_DISTANCE = 0.6            # cutoff for that mean; higher than MAX_DISTANCE, which is one best face
GALLERY_MIN_MARGIN = 0.01             # winner must beat the runner-up by this much

# Fusion weights (when both models are healthy)
FACENET_WEIGHT = 0.5      # FaceNet contribution (tuned 2026-02-12 re-eval)
ARCFACE_WEIGHT = 0.5      # ArcFace contribution (tuned 2026-02-12 re-eval)
