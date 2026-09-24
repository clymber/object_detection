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

## Detached Runs and YOLO Comparison

Synchronize the project notebook, then run a bounded GPU smoke check through
the RF-DETR Conda kernel:

```bash
make sync-notebooks
RFDETR_MODE=fresh RFDETR_SMOKE=1 bash scripts/run_notebook_tmux.sh run \
  --file rfdetr/notebooks/nb04.02-rfdetr_small_large_basketball.ipynb \
  --session rfdetr-small-smoke
bash scripts/run_notebook_tmux.sh check --session rfdetr-small-smoke
```

For a full run, set `RFDETR_SMOKE=0` and use a new session name. To continue
an interrupted run, set `RFDETR_MODE=resume` and `RFDETR_RUN_DIR` to the exact
recorded run directory. After a completed new-protocol run, regenerate outputs
and safely republish its bundle without training:

```bash
conda run -p .conda/envs/object-detection-rfdetr python -m \
  rfdetr_pipeline.postprocess_cli --run-dir <run>
```

Runs created before `run_protocol.json` cannot be recovered or converted; start
a fresh experiment for shared comparison.

For YOLO comparison, run `export-ultralytics-baseline` through
`.conda/envs/object-detection-ultralytics` and
`export-yolox-baseline --model tiny` / `--model nano` through
`.conda/envs/object-detection-yolox`. Pass each command the same full `--dataset-dir` and
`--output-dir`; the project READMEs show their arguments. Set
`RFDETR_BASELINE_EXPORT_DIR` to that output directory when running this
notebook. The exporters create the `<model>_<split>_predictions.json` files
that its comparison cell reads. RF-DETR's ONNX worker is available as
`conda run -p .conda/envs/object-detection-rfdetr python -m
rfdetr_pipeline.onnx_cli --run-dir <run>`.
