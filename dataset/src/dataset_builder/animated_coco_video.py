"""
Turn annotated COCO images into an animated video and a bounding-box preview.
"""

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack, suppress
from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

from detection_common.utils.digest import file_digest


@dataclass(frozen=True)
class VideoSettings:
    """
    Set the canvas, cadence, duration and reproducible source selection.
    """

    width: int = 1280
    height: int = 720
    fps: int = 30
    duration: float = 180
    seconds_per_image: float = 0.5
    seed: int = 42

    def counts(self) -> tuple[int, int]:
        """
        Require whole frames and whole image segments rather than rounding duration.
        """
        if (self.width < 2 or self.height < 2 or self.width % 2
                or self.height % 2 or self.fps < 1):
            raise ValueError("Require positive FPS and even canvas dimensions >= 2")
        counts = [self.duration * self.fps, self.seconds_per_image * self.fps]
        if any(not math.isfinite(n) or n < 1
               or not math.isclose(n, round(n), abs_tol=1e-8) for n in counts):
            raise ValueError("Duration and seconds per image must give whole frames")
        total, per_image = map(round, counts)
        if per_image < 2 or total % per_image:
            raise ValueError("Require >= 2 frames per image and whole image segments")
        return total, per_image


def validate_coco(coco: dict) -> dict[int, list[dict]]:
    """
    Check image identities and bounding boxes before generating evaluation data.
    """
    images = {row["id"]: row for row in coco["images"]}
    categories = {row["id"] for row in coco["categories"]}
    if not images or len(images) != len(coco["images"]):
        raise ValueError("Require nonempty images with unique IDs")
    if not categories or len(categories) != len(coco["categories"]):
        raise ValueError("Require nonempty categories with unique IDs")
    for row in images.values():
        if row["width"] <= 0 or row["height"] <= 0:
            raise ValueError(f"Invalid source dimensions: image {row['id']}")
    by_image = defaultdict(list)
    annotation_ids = set()
    for row in coco["annotations"]:
        if row["id"] in annotation_ids:
            raise ValueError("Require unique annotation IDs")
        annotation_ids.add(row["id"])
        image = images.get(row["image_id"])
        if image is None or row["category_id"] not in categories:
            raise ValueError("Annotation references an unknown image or category")
        x, y, width, height = row["bbox"]
        if (not all(math.isfinite(v) for v in row["bbox"])
                or min(x, y) < 0 or min(width, height) <= 0
                or x + width > image["width"] + 1e-6
                or y + height > image["height"] + 1e-6):
            raise ValueError(f"Invalid or out-of-bounds box: annotation {row['id']}")
        by_image[row["image_id"]].append(row)
    return by_image


def select_images(coco: dict, count: int, seed: int) -> list[dict]:
    """
    Sample without replacement while retaining the positive/background image ratio.
    """
    images = sorted(coco["images"], key=lambda row: row["id"])
    if count < 1 or count > len(images):
        raise ValueError(f"Need {count} distinct images; the dataset has {len(images)}")
    positive_ids = {row["image_id"] for row in coco["annotations"]}
    positive = [row for row in images if row["id"] in positive_ids]
    negative = [row for row in images if row["id"] not in positive_ids]
    positive_count = round(count * len(positive) / len(images))
    rng = random.Random(seed)
    selected = (rng.sample(positive, positive_count)
                + rng.sample(negative, count - positive_count))
    rng.shuffle(selected)
    return selected


def render_frame(
    image: Image.Image, settings: VideoSettings, progress: float, segment: int,
) -> tuple[Image.Image, dict]:
    """
    Move between seeded opposite corners and zoom while keeping the image visible.
    """
    rng = random.Random(f"{settings.seed}:animation:{segment}")
    start_x, start_y = rng.choice(((0, 0), (1, 0), (0, 1), (1, 1)))
    zoom_in = rng.choice((False, True))
    eased = (1 - math.cos(math.pi * progress)) / 2
    zoom = eased if zoom_in else 1 - eased
    fit = min(settings.width / image.width, settings.height / image.height)
    scale = fit * (0.75 + 0.25 * zoom)
    width = max(1, round(image.width * scale))
    height = max(1, round(image.height * scale))
    pan_x = start_x + (1 - 2 * start_x) * eased
    pan_y = start_y + (1 - 2 * start_y) * eased
    left = round((settings.width - width) * pan_x)
    top = round((settings.height - height) * pan_y)
    frame = Image.new("RGB", (settings.width, settings.height), (32, 32, 32))
    frame.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, top))
    return frame, {
        "scale_x": width / image.width, "scale_y": height / image.height,
        "translate_x": left, "translate_y": top,
    }


def transform_box(box: list[float], transform: dict) -> list[float]:
    """
    Apply the same pixel geometry used to render the image to a COCO xywh box.
    """
    x, y, width, height = box
    return [
        x * transform["scale_x"] + transform["translate_x"],
        y * transform["scale_y"] + transform["translate_y"],
        width * transform["scale_x"], height * transform["scale_y"],
    ]


@contextmanager
def video_writer(path: Path, settings: VideoSettings, ffmpeg: str):
    """
    Stream RGB frames into H.264 without temporary image files or unbounded logs.
    """
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen([
            ffmpeg, "-v", "error", "-nostdin", "-f", "rawvideo", "-pix_fmt",
            "rgb24", "-video_size", f"{settings.width}x{settings.height}",
            "-framerate", str(settings.fps), "-i", "pipe:0", "-an",
            "-vf", "scale=in_range=full:out_range=tv:out_color_matrix=bt709",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
            "-threads", "2", "-pix_fmt", "yuv420p", "-color_range", "tv",
            "-colorspace", "bt709", "-color_primaries", "bt709",
            "-color_trc", "bt709", "-movflags", "+faststart", str(path),
        ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            yield process.stdin
            process.stdin.close()
            code = process.wait()
            if code:
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace"))
        except BaseException as error:
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if not process.stdin.closed:
                with suppress(BrokenPipeError):
                    process.stdin.close()
            if isinstance(error, BrokenPipeError):
                log.seek(0)
                raise RuntimeError(log.read().decode(errors="replace")) from error
            raise


def verify_video(path: Path, settings: VideoSettings, ffprobe: str) -> None:
    """
    Decode-count the completed MP4 and reject incorrect geometry, duration or cadence.
    """
    result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,"
        "nb_read_frames,duration,pix_fmt", "-of", "json", str(path),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    total, _ = settings.counts()
    if (stream["codec_name"] != "h264" or stream["pix_fmt"] != "yuv420p"
            or stream["width"] != settings.width or stream["height"] != settings.height
            or Fraction(stream["avg_frame_rate"]) != settings.fps
            or int(stream["nb_read_frames"]) != total
            or not math.isclose(float(stream["duration"]), total / settings.fps,
                                abs_tol=1e-5)):
        raise ValueError(f"Encoded video differs from requested settings: {path}")


def split_generation_metadata(data: dict, annotations_name: str) -> tuple[dict, dict]:
    """
    Separate evaluation labels from source provenance and rendering geometry.
    """
    generation_fields = {"seconds_per_image", "animation"}
    labels = {key: value for key, value in data.items()
              if key not in {"source", "preview"}}
    labels["schema_version"] = 2
    labels["info"] = {key: value for key, value in data["info"].items()
                      if key not in generation_fields}
    labels["images"] = [
        {key: value for key, value in row.items()
         if key not in {"source_image_id", "transform"}}
        for row in data["images"]
    ]
    labels["annotations"] = [
        {key: value for key, value in row.items() if key != "source_annotation_id"}
        for row in data["annotations"]
    ]
    generation = {
        "schema_version": 1, "type": "animated_coco_video_generation",
        "annotations_file": annotations_name,
        "info": {key: value for key, value in data["info"].items()
                 if key in generation_fields},
        "source": data["source"], "videos": data["videos"],
        "preview": data["preview"],
        "frames": [{"image_id": row["id"],
                    "source_image_id": row["source_image_id"],
                    "transform": row["transform"]} for row in data["images"]],
        "annotations": [{"annotation_id": row["id"],
                         "source_annotation_id": row["source_annotation_id"]}
                        for row in data["annotations"]],
    }
    return labels, generation


def create_video(
    annotations_path: Path, output: Path, *, images_directory: Path | None = None,
    settings: VideoSettings = VideoSettings(), ffmpeg: str = "ffmpeg",
    ffprobe: str = "ffprobe", progress=None,
) -> dict[str, Path]:
    """
    Write videos, evaluation labels, a generation manifest and handover instructions.
    """
    total, per_image = settings.counts()
    annotations_path, output = annotations_path.resolve(), output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension")
    outputs = {
        "video": output,
        "preview": output.with_name(f"{output.stem}_boxes.mp4"),
        "annotations": output.with_name(f"{output.stem}_annotations.json"),
        "generation_manifest": output.with_name(
            f"{output.stem}_generation_manifest.json"),
        "readme": output.with_name(f"{output.stem}_README.md"),
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite: {path}")
    coco = json.loads(annotations_path.read_text())
    by_image = validate_coco(coco)
    selected = select_images(coco, total // per_image, settings.seed)
    if images_directory is None:
        split = annotations_path.stem.removeprefix("instances_")
        images_directory = annotations_path.parent.parent / "images" / split
    images_directory = images_directory.resolve()
    source_files = {}
    source_images = []
    for row in selected:
        path = (images_directory / row["file_name"]).resolve()
        if not path.is_relative_to(images_directory):
            raise ValueError("Image file_name must stay within the images directory")
        with Image.open(path) as image:
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"COCO dimensions do not match image pixels: {path}")
        source_files[row["id"]] = path
        source_images.append({**row, "sha256": file_digest(path)})
    result = {
        "schema_version": 1, "type": "coco_video_boxes",
        "info": {
            "description": (
                "Animated test subset; correlated frames, not natural motion"),
            "bbox_format": "xywh", "coordinate_origin": "top_left",
            "coordinate_units": "pixels in the decoded video canvas",
            "frame_index_base": 0, "frame_id_rule": "images.id = frame_index + 1",
            "seconds_per_image": settings.seconds_per_image,
            "animation": (
                "seeded opposite-corner pan, 75-100% fit zoom, "
                "entire image visible, direct cuts"),
        },
        "source": {
            "annotations_file": annotations_path.name,
            "annotations_sha256": file_digest(annotations_path), "seed": settings.seed,
            "image_count": len(coco["images"]), "selected_image_count": len(selected),
            "selected_positive_count": sum(bool(by_image[row["id"]])
                                           for row in selected),
            "images": source_images,
        },
        "videos": [{
            "id": 1, "file_name": output.name, "width": settings.width,
            "height": settings.height, "fps": settings.fps, "frame_count": total,
            "duration_seconds": total / settings.fps,
            "frame_time_base": f"1/{settings.fps}",
        }],
        "categories": coco["categories"], "images": [], "annotations": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.stem}-", dir=output.parent,
    ) as tmp:
        temporary = {key: Path(tmp) / path.name for key, path in outputs.items()}
        with ExitStack() as stack:
            clean = stack.enter_context(video_writer(
                temporary["video"], settings, ffmpeg))
            boxed = stack.enter_context(video_writer(
                temporary["preview"], settings, ffmpeg))
            for segment, row in enumerate(selected):
                with Image.open(source_files[row["id"]]) as source:
                    image = source.convert("RGB")
                for local_index in range(per_image):
                    index = segment * per_image + local_index
                    frame, transform = render_frame(
                        image, settings, local_index / (per_image - 1), segment)
                    clean.write(frame.tobytes())
                    result["images"].append({
                        "id": index + 1, "video_id": 1, "frame_index": index,
                        "timestamp_seconds": index / settings.fps,
                        "source_image_id": row["id"], "width": settings.width,
                        "height": settings.height, "transform": transform,
                    })
                    draw = ImageDraw.Draw(frame)
                    for annotation in by_image[row["id"]]:
                        box = transform_box(annotation["bbox"], transform)
                        x, y, width, height = box
                        result["annotations"].append({
                            "id": len(result["annotations"]) + 1,
                            "image_id": index + 1,
                            "source_annotation_id": annotation["id"],
                            "category_id": annotation["category_id"], "bbox": box,
                            "area": width * height,
                            "iscrowd": annotation.get("iscrowd", 0),
                        })
                        draw.rectangle((x, y, x + width, y + height),
                                       outline=(0, 255, 0),
                                       width=max(2, settings.height // 240))
                    boxed.write(frame.tobytes())
                if progress is not None and ((segment + 1) % 30 == 0
                                             or segment + 1 == len(selected)):
                    progress(f"Rendered {segment + 1}/{len(selected)} images")
        for key in ("video", "preview"):
            verify_video(temporary[key], settings, ffprobe)
        result["videos"][0]["sha256"] = file_digest(temporary["video"])
        result["preview"] = {
            "file_name": outputs["preview"].name,
            "sha256": file_digest(temporary["preview"]),
        }
        labels, generation = split_generation_metadata(
            result, outputs["annotations"].name)
        for key, data in (("annotations", labels),
                          ("generation_manifest", generation)):
            temporary[key].write_text(
                json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary["readme"].write_text(
            f"# Animated COCO test subset\n\n"
            f"- `{output.name}`: clean inference input.\n"
            f"- `{outputs['preview'].name}`: green ground-truth boxes for inspection.\n"
            f"- `{outputs['annotations'].name}`: "
            "COCO-style per-frame box annotations for evaluation.\n"
            f"- `{outputs['generation_manifest'].name}`: "
            "optional source details and rendering transforms.\n\n"
            f"{settings.width}x{settings.height}, {settings.fps} FPS, "
            f"{total / settings.fps:g} seconds, {total} frames from "
            f"{len(selected)} distinct images; seed {settings.seed}.\n\n"
            "Each image moves between randomly selected opposite canvas corners.\n"
            "Zoom runs in or out between 75% and 100% of the largest fit, keeping\n"
            "the full image visible. The seed fixes motion as well as selection.\n\n"
            "Process every frame of the clean video in decode order, without skips.\n"
            "Warm up separately, then rewind before evaluation. Reset tracking at\n"
            "source-image changes, or disable it when evaluating detector F1.\n\n"
            "Frame indices start at 0; annotation `image_id` equals frame index + 1.\n"
            "The JSON `images` entries identify video frames, not image files.\n"
            "Boxes use `[x, y, width, height]` in video pixels from the top left.\n"
            "Map model predictions back to this canvas before matching boxes.\n"
            "Return frame index, category ID, box and confidence for each prediction;\n"
            "retain source category IDs or provide the model-to-source "
            "category map.\n\n"
            "Report animated test-subset F1 separately from full-test-image F1.\n"
            "Frames from the same image are correlated. This video does not measure\n"
            "natural-motion tracking quality. Only bounding boxes are transformed;\n"
            "segmentation masks and keypoints are not included.\n")
        for key, destination in outputs.items():
            # Publish complete files without overwriting another generation.
            destination.hardlink_to(temporary[key])
    return outputs


def main() -> None:
    """
    Accept a COCO annotation path and create the clean video and inspection package.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path, help="Path to instances_test.json")
    parser.add_argument("--output", type=Path, required=True, help="Clean output MP4")
    parser.add_argument("--images-dir", type=Path, help="Default: ../images/<split>")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=180, help="Video seconds")
    parser.add_argument("--seconds-per-image", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--ffprobe", default=shutil.which("ffprobe"))
    args = parser.parse_args()
    if not args.ffmpeg or not args.ffprobe:
        parser.error("Provide ffmpeg and ffprobe executables")
    settings = VideoSettings(args.width, args.height, args.fps, args.duration,
                             args.seconds_per_image, args.seed)
    try:
        outputs = create_video(
            args.annotations, args.output, images_directory=args.images_dir,
            settings=settings, ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    except (ValueError, OSError, KeyError, RuntimeError,
            subprocess.SubprocessError) as error:
        parser.exit(1, f"Error: {error}\n")
    for key, path in outputs.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
