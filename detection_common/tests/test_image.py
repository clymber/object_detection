"""Tests for image transforms."""

import numpy as np
import pytest

from detection_common.utils.image import letterbox


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
