# Selected basketball models for iOS benchmarking

Inspected on **16 November 2026**. Paths are relative to the workspace root.
These selections identify the source checkpoints for Core ML conversion and ONNX
handoff. Use each run's best checkpoint, selected by validation AP50:95. Epoch
numbers below record the provenance of those saved weights; they are not inputs
for selecting or re-creating a checkpoint.

## Selected runs

All three models are fine-tuned on `basketball-v0.1`, use **640×640** inputs and
have one foreground class, **basketball**. Foreground prediction index **0** maps
to source COCO category **1**. All runs record `smoke_run: false`.

| Model | Run directory below `outputs/runs/basketball/` | Best checkpoint |
| --- | --- | --- |
| RF-DETR Small | `rfdetr_small_20261106T124419` | `checkpoint_best_total.pth` |
| RF-DETR Nano | `rfdetr_nano_20261116T115819` | `checkpoint_best_total.pth` |
| YOLOX Nano | `yolox_nano_20261105T200341` | `weights/best_ckpt.pth` |

## RF-DETR Small

Run: `outputs/runs/basketball/rfdetr_small_20261106T124419/`.

- Selected checkpoint: `checkpoint_best_total.pth`, containing selected **EMA**
  weights from epoch **29**, chosen by validation AP50:95.
- Companion source: `checkpoint_best_ema.pth`. Retain this file and the saved
  configuration for the repository's checkpoint-loading helper.
- Configuration: `run_config.json`, `resolved_config.json`,
  `training_config.json`, `dataset_manifest.json` and `best_checkpoint.json`.
- Training: RF-DETR **1.11.2**, 100-epoch budget, **38 completed epochs**, batch
  **4**, accumulation **1**, seed **42**, AMP enabled, learning rate **0.0001**
  and encoder learning rate **0.00015**. Early stopping uses EMA with patience
  **10** and minimum delta **0.001**. Saved model configuration identifies
  `RFDETRSmall`, one class and resolution **640**.
- Existing export: `checkpoint_best_total.onnx`, **115,135,654 bytes**, opset **17**.
- Existing reference artifacts: `val_predictions.json`, `test_predictions.json`,
  `val_metrics.json`, `test_metrics.json`, native metric files and training CSVs.

SHA256 of the selected checkpoint:

```text
85fc2abfa9fd2aa0964c7b355900711aae8f0c6185776f63bfefc597bd287c64
```

SHA256 of the existing ONNX export:

```text
3254af0ccc47c588cb1838f365eea7d4f30084c02d8b2be50cfe2730374a2890
```

## RF-DETR Nano

Run: `outputs/runs/basketball/rfdetr_nano_20261116T115819/`.

- Selected checkpoint: `checkpoint_best_total.pth`, containing selected **EMA**
  weights from epoch **26**, chosen by validation AP50:95.
- Companion source: `checkpoint_best_ema.pth`. Retain this file and the saved
  configuration for the repository's checkpoint-loading helper.
- Configuration: `run_config.json`, `resolved_config.json`,
  `training_config.json`, `dataset_manifest.json` and `best_checkpoint.json`.
- Training: RF-DETR **1.11.2**, 100-epoch budget, **36 completed epochs**, batch
  **4**, accumulation **1**, seed **42**, AMP enabled, learning rate **0.0001**
  and encoder learning rate **0.00015**. Early stopping uses EMA with patience
  **10** and minimum delta **0.001**. The companion checkpoint configuration
  identifies `RFDETRNano`, one class and resolution **640**.
- Existing export: `checkpoint_best_total.onnx`, **109,222,362 bytes**, opset **17**.
- Existing reference artifacts: `val_predictions.json`, `test_predictions.json`,
  `val_metrics.json`, `test_metrics.json`, native metric files and training CSVs.

SHA256 of the selected checkpoint:

```text
ab9e434ea8d92d261be2fd3f028cf2de7004cdcb7237d4892a5c875ae10f4563
```

SHA256 of the existing ONNX export:

```text
8039da7c2911f574c4e3ab5abc02bff53dd2fd4387c20bd3a078214671d83ad6
```

For both RF-DETR exports, graph interfaces are:

- Input `input`: float32 `[1, 3, 640, 640]`.
- Output `dets`: float32 `[1, 300, 4]`.
- Output `labels`: float32 `[1, 300, 2]`.

The two logit channels do not mean two foreground classes. The existing decoder
excludes the last channel and maps foreground index zero to COCO category one.
Retain the trained 640 resolution rather than substituting variant defaults.

## YOLOX Nano

Run: `outputs/runs/basketball/yolox_nano_20261105T200341/`.

- Selected checkpoint: `weights/best_ckpt.pth`. Load its **`ema_model`** weights;
  the selected checkpoint records completed epoch **99** and validation
  AP50:95 **0.6920105210556013**. The run completed **100 epochs**.
- Configuration: `args.yaml`, `run_protocol.json`, checkpoint `training_config`,
  `training.json` and `dataset_identity.json`.
- Training: batch **8**, image size **640**, seed **42**, 100-epoch budget,
  initialized from `models/pretrained/yolox/yolox_nano.pth`. The source notebook
  is `nb03.03-yolox_nano_large_basketball.ipynb`.
- Reconstruct with `yolox_pipeline.yolox.YOLOXNanoExp`: depth **0.33**, width
  **0.25**, depthwise convolutions and **one class**. A CPU reload with this
  experiment succeeded; all three classification heads have one output channel.
- Saved NMS IoU threshold: **0.65**. `args.yaml` records training/evaluation
  `test_conf: 0.01`; the shared prediction records use score floor **0.001**.
  Apply the frozen mobile protocol's retention and F1 thresholds separately.
- Existing export: `weights/best_ckpt.onnx`, **3,652,340 bytes**, opset **11**.
  Input `images` is float32 `[1, 3, 640, 640]`; output `output` is float32
  `[1, 8400, 6]`. The repository exporter disables graph-side YOLOX decoding;
  preserve the reference decoding, letterboxing and external NMS contract.
- Existing reference predictions: `evaluation/val_predictions.json` and
  `evaluation/test_predictions.json`. `weights/last_ckpt.pth` is also present;
  use the validation-selected best checkpoint for conversion.

SHA256 of the selected checkpoint:

```text
d9b9d32ee716c90480bd1bb1a40304c6ee0fed064d6d996d2876ea5a470e99d3
```

SHA256 of the existing ONNX export:

```text
1af2fd17c337ab43069856007273471e72da745a4586fc63dc66a03961ffe5af
```

## Checks completed and remaining validation

- Recomputed all three selected checkpoint hashes; they match the saved
  validation/test prediction metadata. Both RF-DETR hashes also match
  `best_checkpoint.json`.
- Verified RF-DETR selected weights equal their companion EMA source weights,
  normalizing compiled parameter-name prefixes, and verified selected epochs.
- Loaded YOLOX Nano with the custom one-class experiment on CPU. This check did
  not execute a forward pass.
- Loaded all three existing ONNX files and passed `onnx.checker.check_model` in
  each model's owning Conda environment. Recorded graph shapes, types and hashes.
- Verified all selected runs share canonical dataset source fingerprint
  `7c83e6147e754f9ef6c8104ac71863bbc4356703692baaaf20e5ea23f3a4d3e2`.
  Current source annotation hashes match the RF-DETR manifests and all three
  models' saved validation/test prediction artifacts. Prediction image IDs cover
  the complete validation/test splits: **1,257 / 867 images**, including
  **180 / 142 negatives**. The training split contains **5,730 images**.
- Detailed inspection records are saved below
  `outputs/ios_benchmarking/checkpoint_inventory/`, one JSON file per run.

All three ONNX graphs have empty custom metadata. Their filenames and structural
checks do not prove numerical equivalence to the selected checkpoints. Validate
postprocessed prediction parity before accepting the exports for Unity handoff.
Core ML export, fixed-set numerical parity and on-device validation remain
pending. No `.mlpackage` was found under `outputs/runs/basketball/`.

Some saved metadata contains original Linux absolute paths. Resolve artifacts
using the workspace-relative run directories above, preserving original files.

## Stage 1 storage update

The inventory above records the original artifacts and source notebook names.
The stage-one workflow keeps one export set at
`models/exports/basketball/<model>/<run>/{onnx,coreml}/`. Uniform relative links
in each run's `weights/` expose these artifacts: RF-DETR uses `checkpoint_best_total.{onnx,mlpackage}`, and YOLOX uses
`best_ckpt.{onnx,mlpackage}`. RF-DETR training checkpoints remain at their original
run-root paths. Each run has an `export_validation` link to
`outputs/exports/basketball/<model>/<run>/`, with one `model_manifest.json` and the
current export/accuracy/parity/timing status. Reruns replace this export set rather
than creating timestamped package and report trees. Consult the current manifest
for newly exported artifact hashes and conversion outcomes.
