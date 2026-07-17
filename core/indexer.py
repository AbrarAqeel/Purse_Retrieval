"""
FAISS index persistence: save, load, existence check, and reset.
"""

import os
import pickle
from typing import List, Tuple

import faiss

import config


def load_database() -> Tuple[faiss.Index, List[dict]]:
    """
    Load a previously saved FAISS index and metadata list from disk.

    Returns
    -------
    tuple
        (faiss_index, metadata)
    """
    faiss_index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return faiss_index, metadata


def save_database(faiss_index: faiss.Index, metadata: List[dict]) -> None:
    """
    Save the FAISS index and metadata list to disk.

    Parameters
    ----------
    faiss_index : faiss.Index
        The index to persist.
    metadata : list of dict
        Metadata aligned with the FAISS index rows.
    """
    config.ensure_dirs()
    faiss.write_index(faiss_index, str(config.FAISS_INDEX_PATH))
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved index ({faiss_index.ntotal} purse(s)) to {config.FAISS_INDEX_PATH}")


def index_exists() -> bool:
    """Whether a saved index + metadata pair currently exists on disk."""
    return config.FAISS_INDEX_PATH.exists() and config.METADATA_PATH.exists()


def reset_index() -> None:
    """Delete the saved FAISS index and metadata files, if present.

    Safe to call any time you change the embedding model or just want a
    clean rebuild. Does not touch dataset/ images.
    """
    for path in (config.FAISS_INDEX_PATH, config.METADATA_PATH):
        if path.exists():
            os.remove(path)
            print(f"Deleted {path}")
        else:
            print(f"Nothing to delete at {path}")
