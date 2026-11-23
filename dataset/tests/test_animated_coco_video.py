"""
Check source sampling and annotation alignment against actual encoded video pixels.
"""

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageDraw
import pytest

from dataset_builder.animated_coco_video import (
    VideoSettings, create_video, render_frame, select_images, transform_box,
)
from detection_common.utils.digest import file_digest


def test_default_selection_retains_background_ratio_without_reusing_sources():
    """
    Select 360 distinct sources with the full test split's positive/background ratio.
    """
    coco = {
        "images": [{"id": i} for i in range(867)],
        "annotations": [{"image_id": i} for i in range(725)],
    }
    chosen = select_images(coco, 360, 42)
    assert chosen == select_images(coco, 360, 42)
    assert len({row["id"] for row in chosen}) == 360
    assert sum(row["id"] < 725 for row in chosen) == 301
    assert chosen != select_images(coco, 360, 43)
    assert VideoSettings().counts() == (5400, 15)
    with pytest.raises(ValueError, match="distinct images"):
        select_images(coco, 868, 42)


@pytest.mark.parametrize("size", [(80, 80), (120, 40), (40, 120)])
def test_animation_keeps_images_visible_and_boxes_on_the_rendered_object(size):
    """
    Check box coordinates against rasterized object bounds for several aspect ratios.
    """
    image = Image.new("RGB", size, "black")
    box = [10, 10, 20, 20]
    ImageDraw.Draw(image).rectangle((10, 10, 29, 29), fill="red")
    settings = VideoSettings(width=256, height=144)
    for segment in range(4):
        for progress in (0, 0.5, 1):
            frame, transform = render_frame(image, settings, progress, segment)
            x, y, width, height = transform_box(box, transform)
            left, top = transform["translate_x"], transform["translate_y"]
            assert 0 <= left <= left + image.width * transform["scale_x"] <= 256
            assert 0 <= top <= top + image.height * transform["scale_y"] <= 144
            # Derive object bounds independently from the rendered red pixels.
            mask = frame.getchannel("R").point(lambda value: 255 if value > 128 else 0)
            actual = mask.getbbox()
            assert actual is not None
            assert all(abs(a - b) <= 1.5 for a, b in zip(
                actual, (x, y, x + width, y + height)))


def test_motion_reaches_corners_with_seeded_directions_and_larger_zoom():
    """
    Cover all diagonal directions and full-height zoom with reproducible motion.
    """
    image = Image.new("RGB", (80, 80), "red")
    settings = VideoSettings(width=256, height=144)
    directions = set()
    seeded_paths = []
    other_paths = []
    for segment in range(32):
        _, start = render_frame(image, settings, 0, segment)
        _, end = render_frame(image, settings, 1, segment)
        assert start == render_frame(image, settings, 0, segment)[1]
        heights = set()
        centers = []
        for transform in (start, end):
            width = round(image.width * transform["scale_x"])
            height = round(image.height * transform["scale_y"])
            left, top = transform["translate_x"], transform["translate_y"]
            assert left in (0, settings.width - width)
            assert top in (0, settings.height - height)
            heights.add(height)
            centers.append((left + width / 2, top + height / 2))
        assert heights == {108, 144}
        assert abs(centers[1][0] - centers[0][0]) > settings.width / 2
        directions.add((centers[1][0] > centers[0][0],
                        centers[1][1] > centers[0][1]))
        seeded_paths.append(start)
        other = VideoSettings(width=256, height=144, seed=43)
        other_paths.append(render_frame(image, other, 0, segment)[1])
    assert len(directions) == 4
    assert seeded_paths != other_paths


@pytest.fixture
def video_fixture(tmp_path):
    """
    Prepare four annotated images, including a background, and real codec tools.
    """
    ffmpeg = os.environ.get("IOS_BENCHMARK_FFMPEG") or shutil.which("ffmpeg")
    ffprobe = os.environ.get("IOS_BENCHMARK_FFPROBE") or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg and ffprobe required for real MP4 integration checks")
    images_dir = tmp_path / "coco/images/test"
    images_dir.mkdir(parents=True)
    coco = {"images": [], "annotations": [],
            "categories": [{"id": 1, "name": "basketball"}]}
    for index in range(4):
        image = Image.new("RGB", (40, 40), "black")
        if index < 3:
            ImageDraw.Draw(image).rectangle((10, 10, 29, 29), fill="red")
            coco["annotations"].append({
                "id": index + 1, "image_id": index + 1, "category_id": 1,
                "bbox": [10, 10, 20, 20], "iscrowd": 0,
            })
        image.save(images_dir / f"{index}.png")
        coco["images"].append({"id": index + 1, "file_name": f"{index}.png",
                               "width": 40, "height": 40})
    path = tmp_path / "coco/annotations/instances_test.json"
    path.parent.mkdir()
    path.write_text(json.dumps(coco))
    return {
        "annotations_path": path, "output": tmp_path / "results/test.mp4",
        "settings": VideoSettings(width=128, height=72, fps=10, duration=2),
        "ffmpeg": ffmpeg, "ffprobe": ffprobe,
    }


def decoded_frames(path: Path, ffmpeg: str) -> list[Image.Image]:
    """
    Decode real H.264 frame bytes for independent inspection of boxes and ordering.
    """
    raw = subprocess.run([
        ffmpeg, "-v", "error", "-i", str(path), "-f", "rawvideo",
        "-pix_fmt", "rgb24", "pipe:1",
    ], check=True, capture_output=True).stdout
    size = 128 * 72 * 3
    return [Image.frombytes("RGB", (128, 72), raw[start:start + size])
            for start in range(0, len(raw), size)]


def test_encoded_videos_match_frame_annotations_and_preview_boxes(video_fixture):
    """
    Validate real MP4 pixels, every timestamp, negative frames and the clean/boxed pair.
    """
    outputs = create_video(**video_fixture)
    labels = json.loads(outputs["annotations"].read_text())
    generation = json.loads(outputs["generation_manifest"].read_text())
    assert labels["schema_version"] == 2
    assert "source" not in labels and "preview" not in labels
    assert "animation" not in labels["info"]
    assert "seconds_per_image" not in labels["info"]
    assert generation["annotations_file"] == outputs["annotations"].name
    sources = {row["image_id"]: row for row in generation["frames"]}
    clean = decoded_frames(outputs["video"], video_fixture["ffmpeg"])
    preview = decoded_frames(outputs["preview"], video_fixture["ffmpeg"])
    assert len(clean) == len(preview) == len(labels["images"]) == 20
    assert Counter(row["source_image_id"] for row in generation["frames"]) == {
        1: 5, 2: 5, 3: 5, 4: 5,
    }
    assert len(labels["annotations"]) == 15
    assert labels["videos"][0]["sha256"] == file_digest(outputs["video"])
    assert generation["preview"]["sha256"] == file_digest(outputs["preview"])
    assert generation["source"]["selected_positive_count"] == 3
    assert {row["annotation_id"] for row in generation["annotations"]} == {
        row["id"] for row in labels["annotations"]}
    assert all("source_annotation_id" not in row for row in labels["annotations"])
    boxes = {row["image_id"]: row["bbox"] for row in labels["annotations"]}
    for index, row in enumerate(labels["images"]):
        assert row["id"] == index + 1 and row["frame_index"] == index
        assert row["timestamp_seconds"] == index / 10
        assert "source_image_id" not in row and "transform" not in row
        assert "transform" in sources[row["id"]]
        if sources[row["id"]]["source_image_id"] == 4:
            assert row["id"] not in boxes
            continue
        x, y, width, height = boxes[row["id"]]
        clean_pixels = clean[index].load()
        preview_pixels = preview[index].load()
        clean_rgb = [clean_pixels[x, y] for y in range(72) for x in range(128)]
        preview_rgb = [preview_pixels[x, y] for y in range(72) for x in range(128)]
        red = Image.new("L", clean[index].size)
        red.putdata([255 if r > 128 and r > g * 1.5 and r > b * 1.5 else 0
                     for r, g, b in clean_rgb])
        assert all(abs(a - b) <= 2 for a, b in zip(
            red.getbbox(), (x, y, x + width, y + height)))
        # A red source has no green border until the preview overlay is drawn.
        clean_green = sum(g > 120 and g > r * 1.5 for r, g, b in clean_rgb)
        preview_green = sum(g > 120 and g > r * 1.5 for r, g, b in preview_rgb)
        assert clean_green == 0 and preview_green > 30
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_video(**video_fixture)


def test_bad_source_geometry_does_not_publish_outputs(video_fixture):
    """
    Refuse stale COCO dimensions before publishing misleading annotations or video.
    """
    path = video_fixture["annotations_path"]
    coco = json.loads(path.read_text())
    coco["images"][0]["width"] = 41
    path.write_text(json.dumps(coco))
    with pytest.raises(ValueError, match="dimensions do not match"):
        create_video(**video_fixture)
    assert not video_fixture["output"].parent.exists()


@pytest.mark.parametrize("settings", [
    VideoSettings(width=127), VideoSettings(fps=0),
    VideoSettings(duration=0.01), VideoSettings(duration=1.1),
    VideoSettings(seconds_per_image=0.01),
])
def test_invalid_duration_or_canvas_is_rejected(settings):
    """
    Reject requests that require partial frames, partial segments or odd YUV dimensions.
    """
    with pytest.raises(ValueError):
        settings.counts()
