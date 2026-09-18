"""
Utilities for working with images
"""
import hashlib
import struct
from io import BytesIO
from pathlib import Path

import numpy as np
from IPython.display import Image as IPyImage
from IPython.display import display as ipy_display
from matplotlib.figure import Figure
from PIL import Image as PILImage
from PIL import ImageOps


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

def image_content_digest(
    image: Path | str | PILImage.Image,
    hash_alg: str = "sha256",
) -> str:
    """Hash and return a stable digest for an image's content.

    - Normalizes EXIF orientation and converts the image to RGBA.
    - Includes width and height before the pixel bytes.
    - Makes equivalent images hash the same even if metadata or source format differs.
    - Prevents collisions between images sharing pixels but with different dimensions.
    """

    if isinstance(image, (Path, str)):
        with PILImage.open(image) as opened:
            return image_content_digest(opened, hash_alg=hash_alg)

    image = ImageOps.exif_transpose(image).convert("RGBA")

    hasher = hashlib.new(hash_alg)
    hasher.update(struct.pack(">II", image.width, image.height))
    hasher.update(image.tobytes())

    return hasher.hexdigest()
