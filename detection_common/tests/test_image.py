"""Tests for dimension-aware decoded image hashing."""

from pathlib import Path

import pytest
from PIL import Image

from detection_common.utils.image import image_content_digest


@pytest.mark.parametrize("as_string", [False, True])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
def test_hash_image_accepts_file_paths(
    tmp_path: Path, as_string: bool, image_format: str
) -> None:
    """Paths and decoded images produce the same hash."""
    path = tmp_path / f"image.{image_format.lower()}"
    Image.new("RGB", (3, 2), (12, 34, 56)).save(path, format=image_format)
    with Image.open(path) as opened:
        opened.load()
        expected = image_content_digest(opened)

    assert image_content_digest(str(path) if as_string else path) == expected


@pytest.mark.parametrize("mode, color", [("L", 42), ("RGBA", (42, 42, 42, 255))])
def test_hash_image_normalizes_opaque_modes(mode: str, color: object) -> None:
    """Normalize opaque modes without changing caller images."""
    image = Image.new(mode, (3, 2), color)
    original_pixels = image.tobytes()

    assert image_content_digest(image) == image_content_digest(Image.new("RGB", (3, 2), (42, 42, 42)))
    assert image.mode == mode
    assert image.tobytes() == original_pixels


def test_hash_image_includes_dimensions() -> None:
    """Distinguish equal pixel byte sequences with different dimensions."""
    assert image_content_digest(Image.new("RGB", (1, 2))) != image_content_digest(
        Image.new("RGB", (2, 1))
    )



@pytest.mark.parametrize("alpha", [0, 128])
def test_hash_image_preserves_alpha(alpha: int) -> None:
    image = Image.new("RGBA", (3, 2), (42, 42, 42, alpha))
    assert image_content_digest(image) != image_content_digest(Image.new("RGB", (3, 2), (42, 42, 42)))


def test_hash_image_preserves_palette_transparency(tmp_path: Path) -> None:
    image = Image.new("P", (3, 2), 0)
    image.putpalette([42, 42, 42] + [0] * 765)
    image.info["transparency"] = 0
    path = tmp_path / "transparent.png"
    image.save(path)
    assert image_content_digest(path) == image_content_digest(Image.new("RGBA", (3, 2), (42, 42, 42, 0)))
    assert image_content_digest(path) != image_content_digest(Image.new("RGB", (3, 2), (42, 42, 42)))


@pytest.mark.parametrize("orientation, transform", [
    (2, Image.Transpose.FLIP_LEFT_RIGHT),
    (3, Image.Transpose.ROTATE_180),
    (4, Image.Transpose.FLIP_TOP_BOTTOM),
    (5, Image.Transpose.TRANSPOSE),
    (6, Image.Transpose.ROTATE_270),
    (7, Image.Transpose.TRANSVERSE),
    (8, Image.Transpose.ROTATE_90),
])
def test_hash_image_applies_exif_orientation(
    tmp_path: Path, orientation: int, transform: Image.Transpose
) -> None:
    image = Image.new("RGB", (3, 2))
    image.putdata([(i, i * 2, i * 3) for i in range(6)])
    expected = image.transpose(transform)
    image.getexif()[274] = orientation
    original_pixels = image.tobytes()
    path = tmp_path / "oriented.png"
    image.save(path, exif=image.getexif())

    assert image_content_digest(image) == image_content_digest(expected)
    assert image_content_digest(path) == image_content_digest(expected)
    assert image.size == (3, 2)
    assert image.tobytes() == original_pixels
    assert image.getexif()[274] == orientation
