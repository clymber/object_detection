"""
Stage local development clips and a compatible WebM rendition for the iOS app.
"""

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from stage_benchmark_resources import file_hash, hardlink_file


CLIPS = (
    ("basketball-nba_clip.mp4", "basketball-nba_clip.mp4"),
    ("basketball-uhd_4096_2160_30fps.mp4", "basketball-uhd_4096_2160_30fps.mp4"),
    ("basketball_dribbling.webm", "basketball_dribbling_h264.mp4"),
)
ENCODING = [
    "-map", "0:v:0", "-an", "-c:v", "libx264", "-preset", "medium",
    "-crf", "18", "-pix_fmt", "yuv420p", "-bf", "0",
    "-fps_mode", "passthrough", "-enc_time_base", "demux",
    "-video_track_timescale", "1000", "-movflags", "+faststart",
]


def probe(path: Path, ffprobe: str) -> dict:
    """
    Decode every frame to retain actual presentation timestamps and stream metadata.
    """
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
        "-show_entries",
        "stream=codec_name,width,height,time_base,avg_frame_rate:"
        "stream_side_data=rotation:format=duration:frame=best_effort_timestamp",
        "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    pts = [int(frame["best_effort_timestamp"]) for frame in data["frames"]]
    if not pts or any(a >= b for a, b in zip(pts, pts[1:])):
        raise ValueError(f"Require ordered video presentation timestamps: {path}")
    return {
        "codec": stream["codec_name"], "width": stream["width"],
        "height": stream["height"], "time_base": stream["time_base"],
        "average_frame_rate": stream["avg_frame_rate"],
        "rotation_degrees": [
            item["rotation"] for item in stream.get("side_data_list", [])
            if "rotation" in item
        ],
        "duration_seconds": float(data["format"]["duration"]), "pts": pts,
    }


def summary(metadata: dict) -> dict:
    """
    Keep a compact ordered-PTS digest instead of committing thousands of rows.
    """
    return {
        key: value for key, value in metadata.items() if key != "pts"
    } | {
        "frame_count": len(metadata["pts"]), "first_pts": metadata["pts"][0],
        "last_pts": metadata["pts"][-1],
        "pts_sha256": hashlib.sha256(json.dumps({
            "time_base": metadata["time_base"], "pts": metadata["pts"],
        }, separators=(",", ":")).encode()).hexdigest(),
    }


def verify_rendition(source: dict, rendition: dict) -> None:
    """
    Require unchanged dimensions, orientation, frame count and exact media times.
    """
    for key in ("width", "height", "rotation_degrees"):
        if source[key] != rendition[key]:
            raise ValueError(f"Rendition changed {key}")
    before = [value * Fraction(source["time_base"]) for value in source["pts"]]
    after = [value * Fraction(rendition["time_base"]) for value in rendition["pts"]]
    if before != after:
        raise ValueError("Rendition changed frame count/order or presentation times")
    if rendition["codec"] != "h264":
        raise ValueError("Require an H.264 rendition")


def compatible_video(
    source: Path, directory: Path, metadata: dict, ffmpeg: str, ffprobe: str,
) -> tuple[Path, dict, dict]:
    """
    Cache a verified H.264 copy without modifying or silently replacing originals.
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / "basketball_dribbling_h264.mp4"
    provenance = destination.with_suffix(".json")
    source_hash = file_hash(source)
    if destination.exists() or provenance.exists():
        record = json.loads(provenance.read_text())
        if (record["source_sha256"] != source_hash
                or record["encoding_options"] != ENCODING
                or record["sha256"] != file_hash(destination)):
            raise ValueError(
                "Existing rendition/provenance changed; inspect before staging")
        converted = probe(destination, ffprobe)
    else:
        with tempfile.TemporaryDirectory(dir=directory) as temporary:
            target = Path(temporary) / destination.name
            subprocess.run([
                ffmpeg, "-v", "error", "-nostdin", "-n", "-copyts",
                "-i", str(source), *ENCODING, str(target),
            ], check=True)
            converted = probe(target, ffprobe)
            verify_rendition(metadata, converted)
            record = {
                "source_sha256": source_hash, "sha256": file_hash(target),
                "encoding_options": ENCODING,
                "ffmpeg_version": subprocess.run(
                    [ffmpeg, "-version"], check=True, capture_output=True, text=True
                ).stdout.splitlines()[0],
                "audio": "removed", "presentation_times": "unchanged",
            }
            os.rename(target, destination)
            provenance.write_text(json.dumps(record, indent=2) + "\n")
    verify_rendition(metadata, converted)
    return destination, converted, record


def stage(root: Path, ffmpeg: str, ffprobe: str) -> None:
    """
    Link all three developer clips and write portable source/rendition provenance.
    """
    source_directory = root / "data/sources/downloads/basketball_videos"
    target_directory = (
        root / "ios/ENABLECoreMLBench/ENABLECoreMLBench/Resources/TestVideos")
    samples = []
    for original_name, resource_name in CLIPS:
        source = source_directory / original_name
        source_metadata = probe(source, ffprobe)
        resource = source
        metadata = source_metadata
        conversion = None
        if source.suffix == ".webm":
            resource, metadata, conversion = compatible_video(
                source, root / "data/processed/ios_benchmark/videos",
                source_metadata, ffmpeg, ffprobe)
        elif metadata["codec"] != "h264":
            raise ValueError(f"Expected unchanged H.264 MP4: {source}")
        target = target_directory / resource_name
        hardlink_file(resource, target)
        digest = file_hash(resource)
        if file_hash(target) != digest:
            raise ValueError(f"Staged video bytes changed: {target}")
        samples.append({
            "id": f"bundled-video:{Path(resource_name).stem}",
            "file_name": resource_name, "sha256": digest,
            "source_path": source.relative_to(root).as_posix(),
            "source_sha256": file_hash(source),
            "resource_path": resource.relative_to(root).as_posix(),
            "source_video": summary(source_metadata), "video": summary(metadata),
            "frame_reference": {
                "time_base": metadata["time_base"], "pts": metadata["pts"],
            },
            "conversion": conversion,
        })
        print(f"Staged {resource_name}: {len(metadata['pts'])} frames")
    manifest = target_directory / "test_videos_manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1, "scope": "development_test_videos", "samples": samples,
    }, indent=2) + "\n")
    print(f"Saved development provenance: {manifest.relative_to(root)}")


def main() -> None:
    """
    Use existing FFmpeg tools in the evaluation environment to prepare app resources.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe:
        parser.error(
            "Run in the evaluation environment with ffmpeg and ffprobe available")
    stage(Path(__file__).resolve().parents[2], args.ffmpeg, args.ffprobe)


if __name__ == "__main__":
    main()
