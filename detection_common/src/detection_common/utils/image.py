"""
Utilities for working with images
"""
from enum import StrEnum
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from IPython.display import Image as IPyImage
from IPython.display import display as ipy_display
from matplotlib.figure import Figure
from PIL import Image as PILImage


class BBoxFormat(StrEnum):
    """ Commonly used bounding box formats """
    XYXY = "xyxy" # (topleft x, topleft y, bottomright x, bottomright y)
    XYWH = "xywh" # (topleft x, topleft y, width, height)  
    CXCYWH = "cxcywh" # (central x, central y, widht, heigh)

def display(
    image: Path | str | IPyImage | PILImage.Image | Figure | np.ndarray,
    width: int | None = None,
    format: str | None = "PNG",
    close: bool = False,
) -> None:
    """
    Display an image in a Jupyter notebook.
    """
    if isinstance(image, (Path, str)):
        image = IPyImage(filename=str(image), width=width)
    elif isinstance(image, Figure):
        figure = image
        buffer = BytesIO()
        figure.savefig(buffer, format=format, bbox_inches="tight")
        image = IPyImage(data=buffer.getvalue(), format=format, width=width)
        if close:
            import matplotlib.pyplot as plt

            plt.close(figure)
    elif isinstance(image, np.ndarray):
        image = PILImage.fromarray(image)
    elif isinstance(image, PILImage.Image):
        buffer = BytesIO()
        image.save(buffer, format=format)
        image = IPyImage(data=buffer.getvalue(), format=format, width=width)
    elif isinstance(image, IPyImage) and width is not None:
        image.width = width

    ipy_display(image)

def letterbox(
    image: np.ndarray,
    boxes: np.ndarray | None =None,
    size: tuple[int, int] = (640, 640),
    bbox_format: BBoxFormat = BBoxFormat.XYXY,
    normalized=False,
):
    """
    Resize with aspect ratio preserved and center-pad to the target size.

    Parameters
    ----------
    image: Array shaped (height, width) or (height, width, channels).
    boxes: Optional (N, 4) bounding boxes in `bbox_format`.
    size: Positive integer (width, height) pair; reverse of array axis order.
    bbox_format: Box representation: XYXY, XYWH, or CXCYWH.
    normalized: If True, input and output boxes are relative to their respective
        image dimensions; otherwise, coordinates are in pixels.

    Returns
    -------
    image: Resized and padded image.
    boxes: Transformed boxes in the input format, or None.
    scale: Scale factor before rounding resized dimensions to pixels.
    padding: Left and top padding in pixels: (left, top).

    Raises
    ------
    TypeError: If `size` or its dimensions have invalid types.
    ValueError: If image dimensionality, size values, or box format are invalid.
    """
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape of (H, W) or (H, W, C)")
    if not isinstance(size, (list, tuple, np.ndarray)):
        raise TypeError(f"Unsupported size type: {type(size)}")
    if len(size) != 2:
        raise ValueError("size must be a (width, height) pair of positive integers")
    for sz in size:
        if isinstance(sz, (bool, np.bool_)) or not isinstance(sz, (int, np.integer)):
            raise TypeError("size must be a (width, height) pair of positive integers")
        if sz <= 0:
            raise ValueError("size must be a (width, height) pair of positive integers")
    bbox_format = BBoxFormat(bbox_format)
    if bbox_format not in BBoxFormat:
        raise ValueError(f"Unsupported bbox_format {bbox_format!r}. ")

    target_w, target_h = size
    input_h, input_w = image.shape[:2]
    has_single_chnl = image.ndim == 3 and image.shape[2] == 1

    # Image transformation
    scale = min(target_w / input_w, target_h / input_h) # 640 / 2000, 640 / 1
    scaled_w, scaled_h = max(round(input_w * scale), 1), max(round(input_h * scale), 1)
    resized = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = target_w - scaled_w, target_h - scaled_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    image = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    # Restore the single channel axis if originally provided.
    if has_single_chnl:
        image = image[..., np.newaxis]

    if boxes is None:
        return image, None, scale, (left, top)

    boxes = np.asarray(boxes, dtype=np.float32).copy()

    # Convert normalized input -> absolute coordinates
    if normalized:
        boxes[:, [0, 2]] *= input_w
        boxes[:, [1, 3]] *= input_h

    # Convert requested format -> XYXY
    if bbox_format == BBoxFormat.XYWH:
        boxes[:, [2, 3]] += boxes[:, [0, 1]]    # x, y, w, h -> x1, y1, x2, y2
    elif bbox_format == BBoxFormat.CXCYWH:
        # cx, cy, w, h -> x1, y1, x2, y2
        cx = boxes[:, 0].copy()
        cy = boxes[:, 1].copy()
        bw = boxes[:, 2].copy()
        bh = boxes[:, 3].copy()
        boxes[:, 0] = cx - bw / 2
        boxes[:, 1] = cy - bh / 2
        boxes[:, 2] = cx + bw / 2
        boxes[:, 3] = cy + bh / 2

    # Apply letterbox transformation in XYXY coordinates
    boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + left
    boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + top

    # Convert XYXY -> requested output format
    if bbox_format == BBoxFormat.XYWH:
        boxes[:, [2, 3]] -= boxes[:, [0, 1]] # x1, y1, x2, y2 -> x, y, w, h
    elif bbox_format == BBoxFormat.CXCYWH:
        # x1, y1, x2, y2 -> cx, cy, w, h
        x1 = boxes[:, 0].copy()
        y1 = boxes[:, 1].copy()
        x2 = boxes[:, 2].copy()
        y2 = boxes[:, 3].copy()
        boxes[:, 0] = (x1 + x2) / 2
        boxes[:, 1] = (y1 + y2) / 2
        boxes[:, 2] = x2 - x1
        boxes[:, 3] = y2 - y1

    # Absolute output -> normalized output
    if normalized:
        boxes[:, [0, 2]] /= target_w
        boxes[:, [1, 3]] /= target_h

    return image, boxes, scale, (left, top)
