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

## Independent export and validation

Open the [Nano export notebook](notebooks/nb03.03.02-yolox_nano-basketball-export.py) in a fresh
`object-detection-yolox` kernel. Each notebook has eight numbered steps:
configure, prepare/inspect, export, parity, accuracy, host timing, saved comparison
images, and completion review.

The `yolox_pipeline.export_validation` API exposes the same explicit stages:

```python
plan = prepare_export(RUN_DIR, DATASET_DIR, **settings)
export_set = export_models(plan)
parity = check_parity(export_set)
accuracy = evaluate_accuracy(export_set)
timing = benchmark_host(export_set)
result = read_results(export_set)
```

Preparation verifies YOLOX Nano run metadata and the local dataset
without loading weights or writing artifacts. `RUN_DIR=None` selects the latest completed matching
run by modification time; an absolute override pins an exact run or raises.
`DATASET_DIR` resolves archived Linux paths against the local dataset without
changing saved provenance. Export uses fresh checkpoint state for each format.

Accuracy uses the full test split; validation keeps its 25-image parity check.
Host timing uses five untimed warmups and 100 measured images once. The shared
`runtime_workflow.HOST_TIMING` constant defines this budget. Existing exports keep
their saved protocol. `CHECKPOINT_DEVICE="mps"` selects native PyTorch Apple GPU
inference; ONNX stays on CPU and native Core ML uses its `all` compute policy.

Each validation call executes only its named stage. Parity and accuracy can be
repeated independently; either clears old timing while preserving the other
report. Accuracy remains diagnostic after failed parity. Timing requires saved
successful parity and accuracy and reports missing or failed prerequisites.
Independent format failures remain visible without discarding successful results.

After restarting the kernel, import the API and resume without conversion:

```python
export_set = load_export_set(Path("<saved model_manifest.json path>"))
result = read_results(export_set)
```

Skip preparation/export and run the desired validation cells. The handle pins the
export ID; another export makes an older handle stale. Source, dataset and graph
identities are checked before each validation stage. Existing schema-1 export
manifests resume from verified saved evidence.

One physical export set lives under
`models/exports/basketball/<model>/<run>/{onnx,coreml}/`, without attempt-ID
subdirectories. The run's `weights/` paths link to those artifacts. An explicit
export replaces that set; validation never replaces it. Reports live under
`outputs/exports/basketball/<model>/<run>/`, exposed by the run's
`export_validation` link. Stage writes are locked and manifests update atomically.
Training checkpoints and provenance remain unchanged.

CPU/FP32 is the checkpoint and ONNX Runtime default on Mac and Renku. macOS also
defaults to Core ML FP32 with `all` compute units. `CPU_THREADS` defaults to four.
Parity retains the fixed development selection and tolerances. These host
measurements do not establish iPhone or CUDA equivalence. Compare saved results using
[nb05.05](../evaluation/notebooks/nb05.05-basketball_runtime_comparison.py).

The final notebook cell prints unfinished or failed validation after displaying
saved diagnostic evidence, so batch notebook runners can continue.

YOLOX preserves saved resolution, classes, seed, EMA/regular weights and NMS IoU.
Both exported graphs retain raw grids with external decoding and class-aware CPU
NMS. Core ML retains its fixed batch-one trace and iOS 16 target. The standalone
baseline export CLI remains unchanged.
