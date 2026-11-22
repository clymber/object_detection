"""
Copy real Swift-produced exports and independently verify their returned records.
"""

import json
from pathlib import Path
import shutil
import tempfile

from detection_evaluation.mobile_video import validate_transfer


def main():
    """
    Check original and transferred files, source accounting and recalculated metrics.
    """
    root = Path(__file__).resolve().parents[3]
    output = root / "outputs/ios_builds/video_pipeline"
    checked = 0
    for group in ("export_checks", "coverage_exports"):
        paths = json.loads((output / group / "fixture_index.json").read_text())
        with tempfile.TemporaryDirectory(prefix="roundtrip_", dir=output) as temporary:
            for name, path in paths.items():
                saved = Path(temporary) / name
                shutil.copytree(path, saved)
                report = validate_transfer(Path(path), saved)
                expected = ("complete" if group == "coverage_exports"
                            and name != "short" else "incomplete")
                assert report["coverage_status"] == expected, report
                assert report["metrics_recalculated"], report
                print(f"{group}/{name}: {report['records']} records; "
                      f"coverage {expected}; transfer and metrics passed")
                checked += 1
    print(f"Video export round trip: {checked} native exports passed")


if __name__ == "__main__":
    main()
