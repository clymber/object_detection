# Ultralytics Pipeline

Ultralytics training helpers and dataset setup. Model-specific prediction
exporters belong here; cross-model evaluation belongs in `evaluation/`.

From the workspace root, run `bash scripts/setup_conda_envs.sh ultralytics` and
`conda run -n ultralytics-dev python -m pytest ultralytics/tests`. Notebooks
under `notebooks/` use the `object-detection-ultralytics` kernel. Source data
defaults to `WORKSPACE_ROOT/data`; generated runs belong below
`OUTPUT_ROOT/runs/basketball`.

The macOS/MPS Conda environment enables PyTorch's CPU fallback for unsupported MPS
operations, including Torchvision NMS. The notebooks also set it in their first
cell because VS Code may launch the interpreter without activating Conda.
Restart the kernel and run from the first cell before training. The setting has
no effect on CUDA.

Export a completed basketball checkpoint without retraining:

```bash
conda run -n ultralytics-dev export-ultralytics-baseline \
  --dataset-dir /absolute/path/to/coco_basketball_large_dataset \
  --output-dir /absolute/path/to/new/artifact-directory \
  --device cpu
```

Use `--run-dir` to select a specific run; otherwise the newest complete
`yolo11n_basketball_large_dataset*` run below `OUTPUT_ROOT/runs/basketball`
is selected. This full export needs a real checkpoint and dataset. The local
smoke check only exercises imports and the CLI boundary.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles keep the same
named Conda environment; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

The Renku profile uses CUDA 13.0 PyTorch wheels and the headless
Ultralytics distribution. Confirm CUDA, Torchvision NMS, and
headless OpenCV before running training notebooks.
