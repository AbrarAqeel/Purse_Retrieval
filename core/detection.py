"""
Purse detection helpers built on a pretrained YOLO model (COCO weights).

Detection is read-only: original images are opened, inspected, and never
modified or saved over. The purse crop it produces is only ever used
in-memory to build an embedding -- it's never written to disk as an output.
"""

from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
from ultralytics import YOLO

import config

BBox = Tuple[float, float, float, float, float]  # x1, y1, x2, y2, confidence


def load_yolo_model() -> YOLO:
    """
    Load the pretrained YOLO model used for handbag detection.

    Returns
    -------
    YOLO
        Loaded ultralytics YOLO model (weights auto-download on first use).
    """
    return YOLO(config.YOLO_MODEL)


def detect_purse_bbox(image_path: Path, model: YOLO) -> Optional[BBox]:
    """
    Detect the purse/handbag in an image using pretrained YOLO (COCO weights).

    Parameters
    ----------
    image_path : Path
        Path to the full original image (e.g. a photo of a woman).
    model : YOLO
        Loaded YOLO model.

    Returns
    -------
    tuple or None
        (x1, y1, x2, y2, confidence) of the highest-confidence handbag
        detection, or None if nothing cleared the confidence threshold.
    """
    results = model.predict(
        source=str(image_path),
        conf=config.DETECTION_CONF_THRESHOLD,
        iou=config.DETECTION_IOU_THRESHOLD,
        classes=[config.HANDBAG_CLASS_ID],  # only look for handbags
        verbose=False,
    )

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    # If several handbags are detected, keep the most confident one
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
    confidence = float(boxes.conf[best_idx])
    return (x1, y1, x2, y2, confidence)


def crop_purse(image_path: Path, bbox: BBox) -> Image.Image:
    """
    Crop the purse region from the original image.

    Parameters
    ----------
    image_path : Path
        Path to the full original image.
    bbox : tuple
        (x1, y1, x2, y2, confidence) from detect_purse_bbox().

    Returns
    -------
    PIL.Image.Image
        Cropped purse region, RGB.
    """
    x1, y1, x2, y2, _ = bbox
    image = Image.open(image_path).convert("RGB")
    return image.crop((x1, y1, x2, y2))


def debug_detections(image_path: Path, model: YOLO, top_n: int = 8) -> None:
    """
    Print all objects YOLO detects in an image, regardless of class or
    confidence, to help diagnose why a purse wasn't found.

    Parameters
    ----------
    image_path : Path
        Path to the image to inspect.
    model : YOLO
        Loaded YOLO model.
    top_n : int
        Max number of detections to print, sorted by confidence.
    """
    results = model.predict(source=str(image_path), conf=0.01, verbose=False)
    boxes = results[0].boxes
    names = results[0].names

    if boxes is None or len(boxes) == 0:
        print(f"{image_path}: nothing detected at all, even at conf=0.01")
        return

    detections = sorted(zip(boxes.cls.tolist(), boxes.conf.tolist()), key=lambda x: -x[1])[:top_n]

    print(f"{image_path}:")
    for cls_id, conf in detections:
        marker = " <-- handbag" if int(cls_id) == config.HANDBAG_CLASS_ID else ""
        print(f"  {names[int(cls_id)]:<15} conf={conf:.3f}{marker}")
