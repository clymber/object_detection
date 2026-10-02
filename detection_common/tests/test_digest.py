"""
Tests for stable byte, file, JSON, and decoded-image digests.
"""

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from detection_common.utils.digest import (
    bytes_digest,
    file_digest,
    image_content_digest,
    json_digest,
)


class TestBytesDigest:
    """
    Test raw in-memory byte digests.
    """

    @pytest.mark.parametrize(
        "data",
        [b"abc", bytearray(b"abc"), memoryview(b"abc")],
    )
    def test_matches_known_sha256_vector(
        self,
        data: bytes | bytearray | memoryview,
    ) -> None:
        """
        Hash every supported bytes-like input with the SHA-256 default.
        """
        assert bytes_digest(data) == (
            "ba7816bf8f01cfea414140de5dae2223"
            "b00361a396177a9cb410ff61f20015ad"
        )

    def test_supports_alternate_algorithms(self) -> None:
        """
        Pass algorithm names through to hashlib.
        """
        assert bytes_digest(b"abc", "sha512") == hashlib.sha512(b"abc").hexdigest()

    def test_rejects_unknown_algorithms(self) -> None:
        """
        Surface hashlib's error for an unknown algorithm name.
        """
        with pytest.raises(ValueError, match="unsupported hash type"):
            bytes_digest(b"abc", "not-a-hash")


class TestFileDigest:
    """
    Test streamed file digests.
    """

    @pytest.mark.parametrize("as_string", [False, True])
    def test_accepts_paths_without_loading_the_whole_file(
        self,
        tmp_path: Path,
        as_string: bool,
    ) -> None:
        """
        Hash a multi-chunk file supplied as either a Path or string.
        """
        data = b"digest-test-chunk" * 131_072
        path = tmp_path / "large.bin"
        path.write_bytes(data)

        source = str(path) if as_string else path
        assert file_digest(source) == bytes_digest(data)


class TestJsonDigest:
    """
    Test digests of the project's stable JSON representation.
    """

    def test_is_order_independent_and_normalizes_values(self) -> None:
        """
        Hash logically equal mappings identically after JSON normalization.
        """
        first = {"path": Path("models/detector.pt"), "enabled": True}
        second = {"enabled": True, "path": "models/detector.pt"}

        assert json_digest(first) == json_digest(second)
        assert json_digest(first) == (
            "c74929a85e2b1ed28fece123b4432317"
            "1f0b256195207d68c723e46db7649301"
        )

    def test_can_reject_nonfinite_numbers(self) -> None:
        """
        Preserve json_bytes strict-number behavior when requested.
        """
        with pytest.raises(ValueError, match="Out of range float values"):
            json_digest({"score": float("nan")}, allow_nan=False)


class TestImageContentDigest:
    """
    Test dimension-aware normalized image-content digests.
    """

    @pytest.mark.parametrize("as_string", [False, True])
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    def test_accepts_file_paths(
        self,
        tmp_path: Path,
        as_string: bool,
        image_format: str,
    ) -> None:
        """
        Hashing paths and decoded images produces the same digest.
        """
        path = tmp_path / f"image.{image_format.lower()}"
        temporary = Image.new(mode="RGB", size=(3, 2), color=(12, 34, 56))
        temporary.save(path, format=image_format)
        temporary.close()

        with Image.open(path) as opened:
            opened.load()
            decoded_digest = image_content_digest(opened)

        source = str(path) if as_string else path
        assert decoded_digest == image_content_digest(source)

    @pytest.mark.parametrize("mode, color", [("L", 42), ("RGBA", (42, 42, 42, 255))])
    def test_normalizes_opaque_modes(
        self,
        mode: str,
        color: float | tuple[float, ...] | str | None,
    ) -> None:
        """
        Normalize opaque modes without changing caller images.
        """
        reference = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))
        test_image = Image.new(mode=mode, size=(3, 2), color=color)
        original_pixels = test_image.tobytes()

        assert image_content_digest(test_image) == image_content_digest(reference)
        assert test_image.mode == mode
        assert test_image.tobytes() == original_pixels

    def test_includes_dimensions(self) -> None:
        """
        Distinguish dimensions even when the pixel byte sequences are equal.
        """
        first = Image.new(mode="RGB", size=(1, 2))
        second = Image.new(mode="RGB", size=(2, 1))

        assert image_content_digest(first) != image_content_digest(second)

    @pytest.mark.parametrize("alpha", [0, 128, 254])
    def test_preserves_alpha(self, alpha: int) -> None:
        """
        Include image alpha in the digest.
        """
        reference = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))
        test_image = Image.new(
            mode="RGBA",
            size=(3, 2),
            color=(42, 42, 42, alpha),
        )

        assert image_content_digest(test_image) != image_content_digest(reference)

    def test_preserves_palette_transparency(self, tmp_path: Path) -> None:
        """
        Preserve transparency when loading a palette-based image.
        """
        path = tmp_path / "transparent.png"
        test_image = Image.new(mode="P", size=(3, 2), color=0)
        test_image.putpalette([42, 42, 42] + [0] * (255 * 3))
        test_image.info["transparency"] = 0
        test_image.save(path)
        test_image.close()
        reference_rgba = Image.new(
            mode="RGBA",
            size=(3, 2),
            color=(42, 42, 42, 0),
        )
        reference_rgb = Image.new(mode="RGB", size=(3, 2), color=(42, 42, 42))

        digest = image_content_digest(path)
        assert digest == image_content_digest(reference_rgba)
        assert digest != image_content_digest(reference_rgb)

    @pytest.mark.parametrize(
        "orientation, transform",
        [
            (2, Image.Transpose.FLIP_LEFT_RIGHT),
            (3, Image.Transpose.ROTATE_180),
            (4, Image.Transpose.FLIP_TOP_BOTTOM),
            (5, Image.Transpose.TRANSPOSE),
            (6, Image.Transpose.ROTATE_270),
            (7, Image.Transpose.TRANSVERSE),
            (8, Image.Transpose.ROTATE_90),
        ],
    )
    def test_applies_exif_orientation(
        self,
        tmp_path: Path,
        orientation: int,
        transform: Image.Transpose,
    ) -> None:
        """
        Hash images by displayed pixels regardless of EXIF orientation.
        """
        path = tmp_path / "oriented.png"
        test_image = Image.new("RGB", (3, 2))
        test_image.putdata([(index, index * 2, index * 3) for index in range(6)])
        test_image.getexif()[274] = orientation
        test_image.save(path, exif=test_image.getexif())
        original_pixels = test_image.tobytes()
        expected = test_image.transpose(transform)

        assert image_content_digest(test_image) == image_content_digest(expected)
        assert image_content_digest(path) == image_content_digest(expected)
        assert test_image.size == (3, 2)
        assert test_image.tobytes() == original_pixels
        assert test_image.getexif()[274] == orientation
