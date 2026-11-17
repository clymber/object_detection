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

## Basketball Nano training and runtime exports

[nb03.03.01](notebooks/nb03.03.01-yolox_nano-basketball.py) trains Nano, selects
its checkpoint using validation metrics, and retains checkpoint reports and
published shared prediction bundles. It does not export ONNX.

Run [nb03.03.02](notebooks/nb03.03.02-yolox_nano-basketball-export.py) in a fresh
YOLOX kernel after training. It uses
`yolox_pipeline.export_validation.export_and_validate`, restores saved resolution,
category mapping and NMS IoU, and prefers selected EMA weights. Loading does not
initialize training, download pretrained replacement weights or update provenance.

`RUNS_DIR` defaults to `OUTPUT_ROOT/runs/basketball`. `RUN_DIR=None` selects the
latest completed matching run by modification time; an absolute override selects
that exact run or raises. Supply `DATASET_DIR` for local copies of archived Linux
runs. Default checkpoint/ONNX inference is CPU/FP32. macOS also exports a fixed
batch-one traced Core ML graph with FP16 and `all` compute units. Both exported
graphs retain raw grids and use external decoding and the saved NMS policy.

The single export set lives at
`models/exports/basketball/yolox_nano/<run>/{onnx,coreml}/`, with no attempt-ID
subdirectory. `weights/best_ckpt.{onnx,mlpackage}` in the run are relative links to
those artifacts. Reruns replace that set and remove obsolete attempt folders.
The current report set is updated at
`outputs/exports/basketball/yolox_nano/<run>/`, also available through the run's
`export_validation` link. It includes development parity, complete
validation/test prediction artifacts and metrics, qualitative overlays, and host
timing with 30 warmups, 300 fixed test IDs and three repetitions. Failed formats
and parity failures remain diagnostic and unaccepted. After saving all available
results, the final cell prints the validation error so the notebook task runner
can continue. Training checkpoints, provenance and existing
evaluation bundles remain unchanged. Compare saved results with
[nb05.05](../evaluation/notebooks/nb05.05-basketball_runtime_comparison.py).

`CPU_THREADS` defaults to four for PyTorch and ONNX Runtime; the actual policy is
recorded with each configuration. Change it explicitly for a separate host test.
