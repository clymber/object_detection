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

## Platform environments

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-dataset` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
