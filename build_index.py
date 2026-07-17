"""
Standalone indexing script -- the ONLY place YOLO detection runs.

This is intentionally separate from main.py so that launching the Streamlit
UI never has to pay the cost of loading YOLO or running detection; main.py
just reads whatever index this script last produced.

Usage
-----
    python build_index.py            # build fresh, or add any new photos
    python build_index.py --reset    # wipe the saved index first, then rebuild

What it does
------------
For every image in dataset/women/: detect the purse with YOLO, crop it
in-memory, embed the crop with DINOv2, and store {embedding, image_path}
in a FAISS index. The original photos are never modified or overwritten.

This is incremental: if an index already exists on disk, only images not
already indexed are processed and appended -- rerunning after adding new
photos to dataset/women/ is fast and doesn't redo existing work.
"""

import argparse

import numpy as np
import faiss

import config
from core.detection import load_yolo_model, detect_purse_bbox, crop_purse
from core.embedding import load_embedding_model, get_embedding
from core.indexer import load_database, save_database, index_exists, reset_index


def build_index() -> None:
    """Build or incrementally update the FAISS index from dataset/women/."""
    config.ensure_dirs()

    print(f"Loading YOLO ({config.YOLO_MODEL}) and DINOv2 ({config.EMBEDDING_MODEL_NAME}) on {config.DEVICE}...")
    yolo_model = load_yolo_model()
    embed_model, embed_processor = load_embedding_model()

    if index_exists():
        faiss_index, metadata = load_database()

        # Guard against silently reusing an index built with a different
        # embedding model -- mismatched dimensions would otherwise crash
        # deep inside FAISS with a cryptic error, or return nonsense.
        expected_dim = embed_model.config.hidden_size
        if faiss_index.d != expected_dim:
            raise ValueError(
                f"Existing index dimension ({faiss_index.d}) does not match the "
                f"current embedding model's dimension ({expected_dim}). This index "
                f"was almost certainly built with a different embedding model. "
                f"Run `python build_index.py --reset` to rebuild from scratch."
            )

        already_indexed = {m["path"] for m in metadata}
        print(f"Loaded existing index -- {len(metadata)} purse(s) already indexed.")
    else:
        faiss_index = None
        metadata = []
        already_indexed = set()
        print("No existing index found -- starting fresh.")

    image_paths = sorted(
        p for p in config.DATASET_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    new_paths = [p for p in image_paths if str(p) not in already_indexed]

    if not new_paths:
        print("No new images to index. Nothing to do.")
        return

    print(f"Indexing {len(new_paths)} new image(s)...")
    new_embeddings = []
    skipped = []

    for path in new_paths:
        bbox = detect_purse_bbox(path, yolo_model)
        if bbox is None:
            skipped.append(path)
            continue

        crop = crop_purse(path, bbox)
        embedding = get_embedding(crop, embed_model, embed_processor)

        new_embeddings.append(embedding)
        metadata.append({"path": str(path), "confidence": bbox[4]})
        print(f"  indexed {path.name}  (detection confidence={bbox[4]:.2f})")

    if skipped:
        print(f"\nNo purse detected in {len(skipped)} image(s):")
        for p in skipped:
            print(f"  - {p}")
        print("Use core.detection.debug_detections() on these to see what YOLO saw instead.\n")

    if not new_embeddings:
        print("No new purses were detected -- index unchanged.")
        return

    new_embeddings = np.vstack(new_embeddings)

    if faiss_index is None:
        # Inner product on L2-normalized vectors == cosine similarity
        faiss_index = faiss.IndexFlatIP(new_embeddings.shape[1])
    faiss_index.add(new_embeddings)

    save_database(faiss_index, metadata)
    print(f"Done. Index now has {faiss_index.ntotal} purse(s) total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build or update the purse FAISS index.")
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete the existing index/metadata before building (full rebuild).",
    )
    args = parser.parse_args()

    if args.reset:
        reset_index()

    build_index()
