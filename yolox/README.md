# YOLOX Pipeline

YOLOX training and inference helpers. Official YOLOX is pinned at commit
`6ddff48` as the `thirdparty/YOLOX` Git submodule. Clone the workspace with
`--recurse-submodules`, or initialize an existing clone with
`git submodule update --init -- thirdparty/YOLOX`. Setup builds a temporary
copy, leaving the submodule unchanged.

From the workspace root, run `bash scripts/setup_conda_envs.sh yolox` and
`conda run -p .conda/envs/object-detection-yolox python -m pytest
yolox/tests`. Notebooks under
`notebooks/` use the `object-detection-yolox` kernel. Source data defaults to
`WORKSPACE_ROOT/data`; generated runs belong below
`OUTPUT_ROOT/runs/basketball`.

Export a completed basketball checkpoint without retraining:

```bash
conda run -p .conda/envs/object-detection-yolox export-yolox-baseline \
  --model tiny \
  --dataset-dir /absolute/path/to/coco_basketball \
  --device cpu
```

Select `--model nano` for the Nano run, or `--run-dir` for an exact run.
Without `--run-dir`, the newest complete matching run below
`OUTPUT_ROOT/runs/basketball` is selected. Artifacts are written beneath
`<run-dir>/evaluation/val/` and `<run-dir>/evaluation/test/`. Full export
needs a real checkpoint and dataset; the local smoke check only covers the CLI
boundary.

The Conda manifests provide platform-compatible OpenCV and ONNX simplifier
packages. Setup removes YOLOX's obsolete package pins from the staged build
metadata, so the environment remains the source of those dependency choices.
Validate ONNX export separately before relying on it.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-yolox` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

Both platforms build the pinned submodule with C++17. The Linux environment
uses CUDA 13.0 PyTorch wheels, headless OpenCV, and onnx-simplifier 0.5.0;
macOS uses its MPS-compatible PyTorch and regular OpenCV packages.

For new-protocol basketball comparison bundles, use the owning
`producer.recover_and_publish` helper with `producer.settings_from_run(run_dir)`
and `producer.resolve_dataset_paths(...)`. The saved settings restore smoke
identity for recovery/resume without repeating shell flags. Smoke layouts are
shared below `DATA_ROOT/processed/smoke/<source-name>/<source-fingerprint>/`.
The full-only baseline CLI above does not publish comparison bundles.
