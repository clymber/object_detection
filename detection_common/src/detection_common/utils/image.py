"""
Utilities for working with images
"""
from io import BytesIO
from pathlib import Path

import numpy as np
from IPython.display import Image as IPyImage
from IPython.display import display as ipy_display
from matplotlib.figure import Figure
from PIL import Image as PILImage


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
