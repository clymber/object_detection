"""
Check video staging provenance and timestamp safeguards without encoding videos.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
from subprocess import CompletedProcess
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIRECTORY))
TOOLS = runpy.run_path(str(SCRIPT_DIRECTORY / "stage_test_videos.py"))
sys.path.pop(0)


def metadata() -> dict:
    """
    Supply independently specified irregular timestamps in a millisecond time base.
    """
    return {
        "codec": "h264", "width": 960, "height": 540, "time_base": "1/1000",
        "average_frame_rate": "2997/100", "duration_seconds": 0.134,
        "rotation_degrees": [], "pts": [0, 33, 67, 100],
    }


class StageTestVideosTests(unittest.TestCase):
    """
    Protect original inputs and reject silently altered cached renditions.
    """

    def test_summary_retains_identity(self):
        """
        Keep frame extent and an ordered exact-timebase hash in the portable catalog.
        """
        summary = TOOLS["summary"](metadata())
        self.assertEqual(summary["frame_count"], 4)
        self.assertEqual((summary["first_pts"], summary["last_pts"]), (0, 100))
        expected = hashlib.sha256(
            b'{"time_base":"1/1000","pts":[0,33,67,100]}'
        ).hexdigest()
        self.assertEqual(summary["pts_sha256"], expected)

    def test_equivalent_media_times(self):
        """
        Permit a changed time base only when every presentation instant is identical.
        """
        source = metadata()
        converted = deepcopy(source)
        converted["time_base"] = "1/30000"
        converted["pts"] = [0, 990, 2010, 3000]
        TOOLS["verify_rendition"](source, converted)

    def test_changed_sequence_is_rejected(self):
        """
        Reject lost, duplicated, reordered or shifted frames despite similar cadence.
        """
        source = metadata()
        for pts in ([0, 33, 100], [0, 33, 33, 100], [0, 67, 33, 100],
                    [1, 34, 68, 101]):
            with self.subTest(pts=pts):
                converted = deepcopy(source)
                converted["pts"] = pts
                with self.assertRaises(ValueError):
                    TOOLS["verify_rendition"](source, converted)

    def test_changed_geometry_or_codec_is_rejected(self):
        """
        Reject resizing, rotation changes and an unsupported output codec.
        """
        for key, value in (("width", 640), ("height", 480),
                           ("rotation_degrees", [90]), ("codec", "vp8")):
            with self.subTest(field=key):
                converted = metadata()
                converted[key] = value
                with self.assertRaises(ValueError):
                    TOOLS["verify_rendition"](metadata(), converted)

    def test_changed_cached_rendition_is_rejected(self):
        """
        Refuse a source or cached-file hash mismatch before invoking any media tool.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.webm"
            target = root / "basketball_dribbling_h264.mp4"
            source.write_bytes(b"original bytes")
            target.write_bytes(b"rendition bytes")
            record = {
                "source_sha256": TOOLS["file_hash"](source),
                "sha256": TOOLS["file_hash"](target),
                "encoding_options": TOOLS["ENCODING"],
            }
            for field in ("source_sha256", "sha256", "encoding_options"):
                changed = deepcopy(record)
                changed[field] = "changed"
                target.with_suffix(".json").write_text(json.dumps(changed))
                with patch("subprocess.run") as command:
                    with self.assertRaises(ValueError):
                        TOOLS["compatible_video"](
                            source, root, metadata(), "ffmpeg", "ffprobe")
                    command.assert_not_called()
                self.assertEqual(source.read_bytes(), b"original bytes")
                self.assertEqual(target.read_bytes(), b"rendition bytes")

    def test_matching_cached_rendition_is_reused(self):
        """
        Probe the verified cache without re-encoding or replacing the input bytes.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.webm"
            target = root / "basketball_dribbling_h264.mp4"
            source.write_bytes(b"original bytes")
            target.write_bytes(b"rendition bytes")
            record = {
                "source_sha256": TOOLS["file_hash"](source),
                "sha256": TOOLS["file_hash"](target),
                "encoding_options": TOOLS["ENCODING"],
            }
            target.with_suffix(".json").write_text(json.dumps(record))
            probe_output = {
                "streams": [{
                    "codec_name": "h264", "width": 960, "height": 540,
                    "time_base": "1/1000", "avg_frame_rate": "2997/100",
                }],
                "frames": [{"best_effort_timestamp": value}
                           for value in [0, 33, 67, 100]],
                "format": {"duration": "0.134"},
            }
            with patch("subprocess.run", return_value=CompletedProcess(
                [], 0, stdout=json.dumps(probe_output), stderr=""
            )) as command:
                path, converted, provenance = TOOLS["compatible_video"](
                    source, root, metadata(), "ffmpeg", "ffprobe")
            command.assert_called_once()
            self.assertEqual(command.call_args.args[0][0], "ffprobe")
            self.assertEqual(path, target)
            self.assertEqual(converted["pts"], [0, 33, 67, 100])
            self.assertEqual(provenance, record)
            self.assertEqual(source.read_bytes(), b"original bytes")


if __name__ == "__main__":
    unittest.main()
