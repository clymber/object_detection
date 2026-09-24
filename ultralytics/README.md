# Ultralytics Pipeline

Ultralytics training helpers and dataset setup. Model-specific prediction
exporters belong here; cross-model evaluation belongs in `evaluation/`.

From the workspace root, run `bash scripts/setup_conda_envs.sh ultralytics` and
`conda run -p .conda/envs/object-detection-ultralytics python -m
pytest ultralytics/tests`. Notebooks
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
conda run -p .conda/envs/object-detection-ultralytics export-ultralytics-baseline \
  --dataset-dir /absolute/path/to/coco_basketball \
  --device cpu
```

Use `--run-dir` to select a specific run; otherwise the newest complete
`yolo11n_<UTC timestamp>` run below `OUTPUT_ROOT/runs/basketball` is selected.
Artifacts are written beneath `<run-dir>/evaluation/val/` and
`<run-dir>/evaluation/test/`. This full export needs a real checkpoint and
dataset. The local smoke check only exercises imports and the CLI boundary.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-ultralytics` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

The Renku profile uses CUDA 13.0 PyTorch wheels and the headless
Ultralytics distribution. Confirm CUDA, Torchvision NMS, and
headless OpenCV before running training notebooks.

For new-protocol basketball comparison bundles, use the owning
`producer.recover_and_publish` helper with `producer.settings_from_run(run_dir)`
and `producer.resolve_dataset_paths(...)`. The saved settings restore smoke
identity for recovery/resume without repeating shell flags. Smoke layouts are
shared below `DATA_ROOT/processed/smoke/<source-name>/<source-fingerprint>/`.
The full-only baseline CLI above does not publish comparison bundles.
