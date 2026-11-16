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

After training, open the corresponding export notebook in a fresh RF-DETR kernel:

- [Small export](notebooks/nb04.02.02-rfdetr_small-basketball-export.py).
- [Nano export](notebooks/nb04.04.02-rfdetr_nano-basketball-export.py).

`RUN_DIR=None` selects the latest completed matching run through the shared
modification-time discovery utility. Set an absolute `RUN_DIR` to select an exact
historical run; invalid overrides raise. `DATASET_DIR` resolves archived Linux
paths against the supplied local COCO dataset without modifying saved provenance.
The notebooks use `rfdetr_pipeline.export_validation.export_and_validate`.

ONNX export and CPU/FP32 validation run on Mac or Renku. macOS also defaults to
native Core ML FP16 with `all` compute units. Set `EXPORT_FORMATS` to request
formats explicitly. Each format uses a fresh checkpoint instance. The single export set lives at
`models/exports/basketball/<model>/<run>/{onnx,coreml}/`, without an attempt-ID
subdirectory. Reruns replace the existing set. The run's
`weights/checkpoint_best_total.{onnx,mlpackage}` paths are relative links to those
artifacts, so only one physical copy is kept.

The single report set lives at `outputs/exports/basketball/<model>/<run>/`, also
available through the run's `export_validation` link. It contains the current
manifest, full-split predictions, metrics, qualitative overlays, development
parity and host timing. Obsolete timestamp/UUID attempt folders are removed.

Training checkpoints and immutable provenance stay unchanged.
Failed conversion and failed parity remain explicit, independent outcomes. The
final notebook cell raises after preserving successful and diagnostic results.
Read these with [nb05.05](../evaluation/notebooks/nb05.05-basketball_runtime_comparison.py).
Host measurements do not establish iPhone or CUDA performance equivalence.
`evaluate_checkpoint_run(run_dir, device=..., publish=False)` creates training
checkpoint reports without exporting; the legacy `postprocess_run` recovery
wrapper retains its ONNX behavior and original immutable notebook provenance.

`CPU_THREADS` defaults to four for PyTorch and ONNX Runtime; the actual policy is
recorded with each configuration. Change it explicitly for a separate host test.
