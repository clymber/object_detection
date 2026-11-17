# RF-DETR Pipeline

RF-DETR training and inference helpers, ONNX export, and model-owned prediction
artifact production. Dataset preprocessing lives in `dataset-builder`;
framework-neutral evaluation lives in `detection-evaluation`.

From the workspace root, run `bash scripts/setup_conda_envs.sh rfdetr` and
`conda run -p .conda/envs/object-detection-rfdetr python -m pytest
rfdetr/tests`. Notebooks under
`notebooks/` use the `object-detection-rfdetr` kernel. Inputs default to
`WORKSPACE_ROOT/data`; runs default to `OUTPUT_ROOT/runs/basketball`.
Override shared roots with absolute `OBJECT_DETECTION_DATA_ROOT` and
`OBJECT_DETECTION_OUTPUT_ROOT` values when needed.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-rfdetr` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

On Renku, the Linux profile uses CUDA 13.0 PyTorch wheels and
headless OpenCV. Verify CUDA NMS and a synthetic RF-DETR Small
prediction before training with real data.

Training automatically enables `torch.compile` when CUDA is available. The first
training steps include compilation overhead.

## Training notebooks and detached runs

The training notebooks fine-tune COCO-pretrained RF-DETR on `basketball-v0.1`:

- `notebooks/nb04.02.01-rfdetr_small-basketball.py` trains RF-DETR Small.
- `notebooks/nb04.04.01-rfdetr_nano-basketball.py` trains RF-DETR Nano.

Both default to 640 x 640 inputs and up to 100 epochs, display training history,
evaluate the selected checkpoint, review prediction grids, and summarize
validation/test performance. Training requires a Renku CUDA session. Set
`DATASET_NAME` in the experiment settings to select a dataset
under `DATA_ROOT/composed/<DATASET_NAME>/coco/`. Image counts, category IDs, and
class names come from the COCO annotations. Outputs stay in the run directory;
the notebook does not publish a comparison bundle.

Synchronize the project notebooks, then run a bounded Nano GPU smoke check through
the RF-DETR Conda kernel:

```bash
make sync-notebooks
RFDETR_MODE=fresh RFDETR_SMOKE=1 bash scripts/run_notebook_tmux.sh run \
  --file rfdetr/notebooks/nb04.04.01-rfdetr_nano-basketball.ipynb \
  --session rfdetr-nano-smoke
bash scripts/run_notebook_tmux.sh check --session rfdetr-nano-smoke
```

For a full run, set `RFDETR_SMOKE=0` and use a new session name. To continue
an interrupted run, set `RFDETR_MODE=resume` and `RFDETR_RUN_DIR` to the exact
recorded run directory in the matching model notebook. The Nano notebook pins
`model_size="nano"`; shared helpers also accept `RFDETR_MODEL_SIZE=small|nano`.
Run directories and prediction artifacts identify the selected variant, and
resume rejects changes to the model identity. After a completed new-protocol run,
regenerate outputs and safely republish its bundle without training:

```bash
conda run -p .conda/envs/object-detection-rfdetr python -m \
  rfdetr_pipeline.postprocess_cli --run-dir <run>
```

Recovery requires a run with `run_protocol.json`; start a fresh training run
when working with older outputs.

RF-DETR's ONNX worker is also available as
`conda run -p .conda/envs/object-detection-rfdetr python -m
rfdetr_pipeline.onnx_cli --run-dir <run>`.

## Independent export and validation

Open the [Small](notebooks/nb04.02.02-rfdetr_small-basketball-export.py) and
[Nano](notebooks/nb04.04.02-rfdetr_nano-basketball-export.py) export notebooks in a fresh
`object-detection-rfdetr` kernel. Each notebook has eight numbered steps:
configure, prepare/inspect, export, parity, accuracy, host timing, saved comparison
images, and completion review.

The `rfdetr_pipeline.export_validation` API exposes the same explicit stages:

```python
plan = prepare_export(RUN_DIR, DATASET_DIR, model="rfdetr_nano", **settings)
export_set = export_models(plan)
parity = check_parity(export_set)
accuracy = evaluate_accuracy(export_set)
timing = benchmark_host(export_set)
result = read_results(export_set)
```

Preparation verifies RF-DETR Small/Nano run metadata and the local dataset
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
defaults to Core ML FP16 with `all` compute units. `CPU_THREADS` defaults to four.
Parity retains the fixed development selection and tolerances. These host
measurements do not establish iPhone or CUDA equivalence. Compare saved results using
[nb05.05](../evaluation/notebooks/nb05.05-basketball_runtime_comparison.py).

The final notebook cell prints unfinished or failed validation after displaying
saved diagnostic evidence, so batch notebook runners can continue.

RF-DETR preserves the saved input resolution, class mapping, `num_select`, selected
EMA/regular checkpoint provenance, background ranking and unclipped source boxes.
`evaluate_checkpoint_run(..., publish=False)` and standalone ONNX export retain
their existing behavior.
