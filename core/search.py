"""
Stage 2 -- search the FAISS index with a standalone purse image and
retrieve the original full model image(s) carrying the closest-matching
purse.
"""

from typing import List

from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config
from core.embedding import get_embedding


def search_purse(
    query_image: Image.Image,
    faiss_index,
    metadata: List[dict],
    embed_model: AutoModel,
    embed_processor: AutoImageProcessor,
) -> List[dict]:
    """
    Search the FAISS index for the closest matching purse(s).

    Two filters are applied, not just one:
      1. SIMILARITY_THRESHOLD -- a hard floor a result must clear at all.
      2. MATCH_MARGIN -- a result must ALSO be within this margin of the
         single best match. This is what prevents a flood of loosely
         similar results when the top match is a clear standout -- a
         fixed threshold alone can't tell that gap apart from a case
         where several genuinely-close candidates exist.

    Parameters
    ----------
    query_image : PIL.Image.Image
        Standalone purse image supplied by the user.
    faiss_index : faiss.Index
        FAISS index of purse embeddings.
    metadata : list of dict
        Metadata aligned with the FAISS index rows (each has a "path" key).
    embed_model, embed_processor
        Loaded DINOv2 model and image processor.

    Returns
    -------
    list of dict
        [{"path": ..., "similarity": float}, ...] sorted by similarity
        (highest first). Empty if nothing clears the threshold, or
        nothing is within MATCH_MARGIN of the best result.
    """
    embedding = get_embedding(query_image, embed_model, embed_processor)
    embedding = embedding.reshape(1, -1)

    similarities, indices = faiss_index.search(embedding, config.TOP_K)

    candidates = [
        (float(sim), int(idx))
        for sim, idx in zip(similarities[0], indices[0])
        if idx != -1 and sim >= config.SIMILARITY_THRESHOLD
    ]
    if not candidates:
        return []

    best_sim = candidates[0][0]  # FAISS already returns results sorted descending
    results = [
        {"path": metadata[idx]["path"], "similarity": sim}
        for sim, idx in candidates
        if best_sim - sim <= config.MATCH_MARGIN
    ]
    return results
