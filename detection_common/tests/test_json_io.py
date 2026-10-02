import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pytest

from detection_common.utils.json_io import (
    json_bytes,
    json_normalize,
    json_ready,
    read_json,
    write_json,
)


class TestJsonReady:
    """
    Unit tests for `json_ready(...)`
    """

    def test_normalizes_mapping(self) -> None:
        """
        Mappings become JSON objects, so keys are normalized to strings.
        """
        test = {1: "value"}
        assert json_ready(test) == {"1": "value"}

    def test_normalizes_list_and_tuple(self) -> None:
        """
        Lists and tuples become recursively normalized JSON arrays.
        """
        test = [1, (2, Path("models") / "detector.pt")]
        assert json_ready(test) == [1, [2, "models/detector.pt"]]

    def test_normalizes_path(self) -> None:
        """
        Paths in mapping values become portable POSIX-style strings.
        """
        test1 = Path("models") / "detector.pt"
        assert json_ready(test1) == "models/detector.pt"

        test2 = {"model": Path("models") / "detector.pt"}
        assert json_ready(test2) == {"model": "models/detector.pt"}

    def test_normalizes_numpy_array(self) -> None:
        """
        NumPy arrays become nested Python lists.
        """
        test = np.array([[1, 2], [3, 4]])
        assert json_ready(test) == [[1, 2], [3, 4]]


class TestWriteJson:
    """
    Unit tests for `write_json(...)`
    """

    def test_write_to_file_paths(self, tmp_path: Path) -> None:
        """
        Write JSON to destinations supplied as Path and string values.
        """
        expected = '{\n  "value": 1\n}\n'
        destinations = (tmp_path / "path.json", str(tmp_path / "string.json"))

        for destination in destinations:
            written = write_json(destination, {"value": 1})

            assert written == len(expected)
            assert Path(destination).read_text(encoding="utf-8") == expected

    def test_write_to_text_streams(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """
        Write JSON to standard, in-memory, and opened text streams.
        """
        expected = '{\n  "value": 1\n}\n'

        assert write_json(sys.stdout, {"value": 1}) == len(expected)
        assert write_json(sys.stderr, {"value": 1}) == len(expected)
        captured = capsys.readouterr()
        assert captured.out == expected
        assert captured.err == expected

        memory_stream = StringIO()
        assert write_json(memory_stream, {"value": 1}) == len(expected)
        assert memory_stream.getvalue() == expected

        path = tmp_path / "opened.json"
        with path.open("w+", encoding="utf-8") as opened_stream:
            assert write_json(opened_stream, {"value": 1}) == len(expected)
            opened_stream.seek(0)
            assert opened_stream.read() == expected

    def test_write_can_reject_nonfinite_numbers(self, tmp_path: Path) -> None:
        """
        Reject nonstandard NaN values when writing strict JSON.
        """
        path = tmp_path / "nonfinite.json"

        with pytest.raises(ValueError, match="Out of range float values"):
            write_json(path, {"score": float("nan")}, allow_nan=False)

        assert not path.exists()


class TestJsonBytes:
    """
    Unit tests for `json_bytes(...)`.
    """

    def test_encodes_stable_utf8_json(self) -> None:
        """
        Normalize values and encode sorted, indented JSON with a trailing newline.
        """
        data = {"model": Path("models") / "detector.pt", "enabled": True}

        assert json_bytes(data) == (
            b'{\n  "enabled": true,\n  "model": "models/detector.pt"\n}\n'
        )

    def test_can_reject_nonfinite_numbers(self) -> None:
        """
        Reject nonstandard NaN values when strict JSON is required.
        """
        with pytest.raises(ValueError, match="Out of range float values"):
            json_bytes({"score": float("nan")}, allow_nan=False)


class TestJsonNormalize:
    """
    Unit tests for `json_normalize(...)`.
    """

    def test_returns_an_independent_json_native_value(self) -> None:
        """
        Normalize mapping keys, paths, tuples, and NumPy values recursively.
        """
        source = {
            1: (
                Path("models") / "detector.pt",
                {"values": np.array([1, 2])},
            )
        }

        normalized = json_normalize(source)

        assert normalized == {
            "1": ["models/detector.pt", {"values": [1, 2]}]
        }
        assert normalized is not source

    def test_can_reject_nonfinite_numbers(self) -> None:
        """
        Reject nonstandard NaN values when strict JSON is required.
        """
        with pytest.raises(ValueError, match="Out of range float values"):
            json_normalize({"score": float("nan")}, allow_nan=False)


class TestReadJson:
    """
    Unit tests of `read_json(...)`
    """

    def test_read_from_file_paths(self, tmp_path: Path) -> None:
        """
        Read JSON from sources supplied as Path and string values.
        """
        sources = (tmp_path / "path.json", tmp_path / "string.json")
        for source in sources:
            source.write_text('{"value": 1}', encoding="utf-8")

        assert read_json(sources[0]) == {"value": 1}
        assert read_json(str(sources[1])) == {"value": 1}

    def test_read_from_text_streams(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Read JSON from standard, in-memory, and opened text streams.
        """
        memory_stream = StringIO('{"value": 1}')
        assert read_json(memory_stream) == {"value": 1}

        monkeypatch.setattr(sys, "stdin", StringIO('{"value": 2}'))
        assert read_json(sys.stdin) == {"value": 2}

        path = tmp_path / "opened.json"
        path.write_text('{"value": 3}', encoding="utf-8")
        with path.open(encoding="utf-8") as opened_stream:
            assert read_json(opened_stream) == {"value": 3}
