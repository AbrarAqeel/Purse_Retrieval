"""
Central configuration for the Purse Retrieval System.

Every tunable value -- paths, model names, thresholds -- lives here.
Nothing else in the project should hardcode a path or a magic number;
import this module and read from it instead.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = PROJECT_ROOT / "dataset" / "women"          # full model photos (indexed)
QUERY_DIR = PROJECT_ROOT / "dataset" / "purse_queries"     # standalone purse photos (used to search)
INDEX_DIR = PROJECT_ROOT / "embeddings"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
METADATA_PATH = INDEX_DIR / "metadata.pkl"

# ---------------------------------------------------------------------------
# Detection (YOLO) -- pretrained on COCO, no training required.
# COCO class 26 is "handbag", so this works out of the box.
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"          # smallest pretrained checkpoint, auto-downloads on first use
HANDBAG_CLASS_ID = 26
DETECTION_CONF_THRESHOLD = 0.15    # lower = more detections, more false positives
DETECTION_IOU_THRESHOLD = 0.45

# ---------------------------------------------------------------------------
# Embedding (DINOv2) -- strong at instance-level visual similarity,
# which is what "is this the same purse" actually needs (unlike CLIP,
# which is tuned for broader semantic/text alignment).
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "facebook/dinov2-small"   # ~90MB, CPU-friendly

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 8                  # candidate pool pulled from FAISS before filtering
SIMILARITY_THRESHOLD = 0.50  # floor a result must clear to be considered at all
MATCH_MARGIN = 0.05          # a result must ALSO be within this margin of the
                              # single BEST match to be shown. This is what stops
                              # loosely-similar results from flooding the output
                              # regardless of where the absolute similarity scale sits.
                              # Tune both of these using the raw numbers printed by
                              # build_index.py / your own testing -- don't guess blind.

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
DEVICE = "cpu"   # CPU-only by default -- plenty fast for a dataset this size.
                  # See README.md for how to switch to CUDA if you ever need it.


def ensure_dirs() -> None:
    """Create the dataset/index folders if they don't already exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    QUERY_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
