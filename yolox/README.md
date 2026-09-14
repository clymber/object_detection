# YOLOX Pipeline

YOLOX training and inference helpers. The upstream YOLOX source checkout is
not vendored; set `YOLOX_SOURCE` to its absolute directory, or place it beside
this workspace as `../YOLOX`. The macOS setup script installs that checkout
with the environment's PyTorch and without build isolation. The currently
tested sibling checkout is commit `6ddff48` of upstream YOLOX.

From the workspace root, run `bash scripts/setup_conda_envs.sh yolox` and
`conda run -n yolox-dev python -m pytest yolox/tests`. Notebooks under
`notebooks/` use the `object-detection-yolox` kernel. Source data defaults to
`WORKSPACE_ROOT/data`; generated runs belong below
`OUTPUT_ROOT/runs/basketball`.

Export a completed basketball checkpoint without retraining:

```bash
conda run -n yolox-dev export-yolox-baseline \
  --model tiny \
  --dataset-dir /absolute/path/to/coco_basketball_large_dataset \
  --output-dir /absolute/path/to/new/artifact-directory \
  --device cpu
```

Select `--model nano` for the Nano run, or `--run-dir` for an exact run.
Without `--run-dir`, the newest complete matching run below
`OUTPUT_ROOT/runs/basketball` is selected. Full export needs a real
checkpoint and dataset; the local smoke check only covers the CLI boundary.

Upstream YOLOX metadata requires `onnx-simplifier==0.4.10`, while the
macOS environment uses a newer `onnxsim`. `pip check` reports that metadata
mismatch. Training and unit tests do not depend on the pinned simplifier;
validate ONNX export separately before relying on it.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles keep the same
named Conda environment; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

On Renku, setup fetches commit 6ddff48 and builds YOLOX with the
C++17 and headless OpenCV adjustments. The Linux environment
includes CUDA 13.0 PyTorch wheels and onnx-simplifier 0.5.0;
the local macOS ONNX metadata caveat remains unchanged.
