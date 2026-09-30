"""Tests for dimension-aware decoded image hashing."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from detection_common.utils.image import image_content_digest, letterbox


class TestImageContentDigest:
    """
    Test `image_content_digest`: Hash a stable digest for an image's content.
    """

    @pytest.mark.parametrize("as_string", [False, True])
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    def test_accepts_file_paths(
            self, tmp_path: Path, as_string: bool, image_format: str
    ) -> None:
        """
        Hashing paths and decoded images produce the same digest
        """
        path = tmp_path / f"image.{image_format.lower()}"
        tmp_img = Image.new(mode="RGB", size=(3, 2), color=(12, 34, 56))
        tmp_img.save(path, format=image_format)
        tmp_img.close()

        with Image.open(path) as opened:
            opened.load()
            decoded_img_digest = image_content_digest(opened)

        filepath_digest = image_content_digest(str(path) if as_string else path)

        assert decoded_img_digest == filepath_digest

    @pytest.mark.parametrize("mode, color", [("L", 42), ("RGBA", (42, 42, 42, 255))])
    def test_normalizes_opaque_modes(
            self, mode: str, color: float | tuple[float, ...] | str | None
        ) -> None:
        """
        Normalize opaque modes without changing caller images. The alpha channel
        controls opacity: 255 means zero opacity.
        """
        reference = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))
        test_img = Image.new(mode=mode, size=(3, 2), color=color)
        orig_test_pixels = test_img.tobytes()

        assert image_content_digest(test_img) == image_content_digest(reference)
        assert test_img.mode == mode
        assert test_img.tobytes() == orig_test_pixels

    def test_includes_dimensions(self) -> None:
        """
        The digest should be different between images with different dimensions, even
        when they have equal pixel byte sequences.
        """
        image1 = Image.new(mode="RGB", size=(1, 2))
        image2 = Image.new(mode="RGB", size=(2, 1))
        assert image_content_digest(image1) != image_content_digest(image2)

    @pytest.mark.parametrize("alpha", [0, 128, 254]) # Not 255 !
    def test_preserves_alpha(self, alpha: int) -> None:
        """
        Hashing must take the image alpha into account.
        """
        reference = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))
        test_img  = Image.new(mode="RGBA", size=(3, 2), color=(42, 42, 42, alpha))
        assert image_content_digest(test_img) != image_content_digest(reference)

    def test_preserves_palette_transparency(self, tmp_path: Path) -> None:
        """
        Preserves transparency when loading a palette-based image. A palette
        image stores a color-table index for each pixel instead of storing RGB
        values directly.
        """
        path = tmp_path / "transparent.png"
        test_img = Image.new(mode="P", size=(3, 2), color=0)
        test_img.putpalette([42, 42, 42] + [0] * (255*3))
        test_img.info["transparency"] = 0
        test_img.save(path)
        test_img.close()
        reference_rgba = Image.new(mode="RGBA", size=(3, 2), color=(42, 42, 42, 0))
        reference_rgb = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))
    
        test_digest = image_content_digest(path)
        assert test_digest == image_content_digest(reference_rgba)
        assert test_digest != image_content_digest(reference_rgb)

    @pytest.mark.parametrize("orientation, transform", [
        (2, Image.Transpose.FLIP_LEFT_RIGHT),
        (3, Image.Transpose.ROTATE_180),
        (4, Image.Transpose.FLIP_TOP_BOTTOM),
        (5, Image.Transpose.TRANSPOSE),
        (6, Image.Transpose.ROTATE_270),
        (7, Image.Transpose.TRANSVERSE),
        (8, Image.Transpose.ROTATE_90),
    ])
    def test_applies_exif_orientation(
        self, tmp_path: Path, orientation: int, transform: Image.Transpose
    ) -> None:
        """
        Images hash identically when their displayed pixels are identical,
        whether their orientation comes from EXIF metadata or from physically
        rearranging the pixels.
        """
        path = tmp_path / "oriented.png"
        test_img = Image.new("RGB", (3, 2))
        test_img.putdata([(i, i * 2, i * 3) for i in range(6)])
        test_img.getexif()[274] = orientation
        test_img.save(path, exif=test_img.getexif())
        original_pixels = test_img.tobytes()

        expected = test_img.transpose(transform)

        assert image_content_digest(test_img) == image_content_digest(expected)
        assert image_content_digest(path) == image_content_digest(expected)
        assert test_img.size == (3, 2)
        assert test_img.tobytes() == original_pixels
        assert test_img.getexif()[274] == orientation


class TestLetterbox:
    """
    Test `letterbox`: Resize with aspect ratio preserved and center-pad to target size.
    """

    def test_letterbox_integer_target_size_succeeds(self) -> None:
        """
        Establish that integer target dimensions work for the float-case input.
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        output, boxes, scale, padding = letterbox(image, size=(640, 640))

        assert output.shape == (640, 640, 3)
        assert boxes is None
        assert scale == pytest.approx(3.2)
        assert padding == (0, 160)


    def test_letterbox_float_target_size_raise_type_error(self) -> None:
        """
        Raise type error for non-integer size.
        """
        image = np.zeros((100, 200, 3), dtype=np.uint8)

        err_msg = r"size must be a \(width, height\) pair of positive integers"
        with pytest.raises(TypeError, match=err_msg):
            letterbox(image, size=(640.0, 640.0))  # pyright: ignore


    @pytest.mark.parametrize("shape", [(1, 2000, 3), (2000, 1, 3)], ids=["wide", "tall"])
    def test_letterbox_preserves_thin_image_content(
            self, shape: tuple[int, int, int]
        ) -> None:
        """
        Require valid thin images to retain visible content in the target canvas.
        """
        image = np.full(shape, 255, dtype=np.uint8)

        output, boxes, scale, padding = letterbox(image)

        assert output.shape == (640, 640, 3)
        assert output.dtype == image.dtype
        assert boxes is None
        assert scale == pytest.approx(640 / 2000)
        left, top = padding
        assert np.all(output[top, left] == 255)
        assert np.any(np.all(output == 114, axis=-1))
