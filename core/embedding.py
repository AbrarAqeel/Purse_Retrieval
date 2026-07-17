"""
DINOv2-based image embedding helpers.

Produces L2-normalized feature vectors so FAISS inner-product search is
equivalent to cosine similarity. DINOv2 is used (rather than CLIP) because
it's specifically strong at instance-level visual similarity -- telling two
photos of the *same physical object* apart from two photos of similar-but-
different objects -- which is exactly what purse matching needs.
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import config


def load_embedding_model() -> Tuple[AutoModel, AutoImageProcessor]:
    """
    Load the pretrained DINOv2 model + matching image processor.

    Returns
    -------
    tuple
        (model, processor) -- DINOv2 model in eval mode on config.DEVICE,
        and its matching image preprocessing pipeline.
    """
    processor = AutoImageProcessor.from_pretrained(config.EMBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(config.EMBEDDING_MODEL_NAME)
    model = model.to(config.DEVICE).eval()
    return model, processor


@torch.no_grad()
def get_embedding(image: Image.Image, model: AutoModel, processor: AutoImageProcessor) -> np.ndarray:
    """
    Compute a normalized DINOv2 embedding for a single image.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image (purse crop or standalone purse photo).
    model : transformers model
        Loaded DINOv2 model.
    processor : transformers image processor
        DINOv2's matching preprocessing pipeline.

    Returns
    -------
    np.ndarray
        L2-normalized embedding, shape (hidden_dim,), dtype float32.
    """
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)

    # CLS token from the last hidden state is DINOv2's standard global
    # image descriptor for retrieval / similarity tasks.
    features = outputs.last_hidden_state[:, 0, :]
    features = features / features.norm(dim=-1, keepdim=True)
    return features.squeeze(0).cpu().numpy().astype("float32")
