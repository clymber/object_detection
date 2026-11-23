# Dataset Builder

COCO/Datumaro preparation and Roboflow downloads. Input data defaults to
`WORKSPACE_ROOT/data`; generated datasets should be placed below
`DATA_ROOT/composed`. Do not store downloaded datasets in this source project.

From the workspace root, run `bash scripts/setup_conda_envs.sh dataset` and
`conda run -p .conda/envs/object-detection-dataset python -m pytest
dataset/tests`. Store the Roboflow API key in the workspace-root
`.roboflow_api_key` file before using Roboflow downloads. This ignored file is
preferred over the `ROBOFLOW_API_KEY` environment variable. The notebooks in
`notebooks/` use the `object-detection-dataset` kernel.

## Versioned datasets

`nb01.07.03-export_datasets.py` composes the reviewed master dataset into
`DATA_ROOT/composed/<target-name>/coco/`. COCO is the canonical detection format;
the matching Ultralytics YOLO export is generated when first requested:

```python
from dataset_builder import Dataset
from dataset_builder.config import DATA_ROOT

dataset = Dataset(DATA_ROOT / "composed" / "football-v0.1")
coco_path = dataset.coco
yolo_path = dataset.yolo
yolo_config = yolo_path / "data.yaml"
```

Both properties return `pathlib.Path` objects. Conversion requires the dataset
environment's Datumaro dependency and prefers hardlinked media, with a byte-copy
fallback. Existing YOLO exports are reused. Keep each version's COCO content
unchanged; compose a new version when its content changes.

The notebook also combines the basketball, football, and Gaelic football COCO
exports into `DATA_ROOT/composed/ball-v0.1/coco/`, with the single class `ball`.
This merge preserves source splits, digest filenames, and background images.

## Animated annotated test video

Create a three-minute 720p H.264 video at 30 FPS from a COCO test split:

```bash
conda run -p .conda/envs/object-detection-dataset \
  python -m dataset_builder.animated_coco_video \
  data/composed/basketball-v0.1/coco/annotations/instances_test.json \
  --output data/processed/ios_benchmark/annotated_test_video_v2/basketball_test.mp4 \
  --ffmpeg .conda/envs/object-detection-evaluation/bin/ffmpeg \
  --ffprobe .conda/envs/object-detection-evaluation/bin/ffprobe
```

Images default to the COCO root's `images/test/`; override with `--images-dir`.
The fixed seed selects 360 distinct images, preserving the positive/background
ratio. Each image moves between randomly selected opposite canvas corners for
0.5 seconds, zooming in or out between 75% and 100% of the largest size that fits.
The full image stays visible. The seed fixes both image selection and motion.
The program streams frames directly to FFmpeg and verifies both completed videos.

Outputs beside the clean MP4 are `_boxes.mp4` for inspecting ground-truth boxes,
`_annotations.json` with final boxes for evaluation, `_generation_manifest.json`
with source details and rendering transforms, and `_README.md` with instructions.
The generation manifest is optional for evaluation. Annotation schema version 2
keeps source IDs, transforms and animation settings out of the evaluation JSON.
The JSON uses COCO-style boxes; its `images` entries identify video frames rather
than image files. Frame indices start at zero; annotation `image_id` equals frame
index + 1. Do not run inference on the boxed preview. Existing output files are
never overwritten.

Use `--duration`, `--seconds-per-image`, `--fps`, `--width`, `--height` and `--seed`
to change the defaults. Durations must produce whole frames and image segments;
the split must contain enough distinct source images. Animated subset F1 is a
separate result from full-test-image F1 and natural-video tracking quality.

## Platform environments

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-dataset` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
