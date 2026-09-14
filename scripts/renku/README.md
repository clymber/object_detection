# Object Control Notebooks on Renku

This guide documents the existing legacy root virtual environments. The new
project-owned Renku workflow uses Miniforge and platform-specific Conda
manifests; see [the workspace README](../../README.md). Keep this path
available until the new environments pass the Stage B checks.

The DCU Renku platform is available at
[soc-gpu.computing.dcu.ie](https://soc-gpu.computing.dcu.ie/).

## Notebook location

Renku and macOS use the same Jupytext notebook sources in `notebooks/`.
On the local MacBook, this directory is:

```text
~/studio/object_ctrl/notebooks
```

In a typical Renku session, it is:

```text
~/work/object_ctrl/notebooks
```

Do not create or maintain separate Renku-specific notebook copies. The setup
script generates paired, ignored `.ipynb` files beside the existing
`notebooks/*.py` sources.

## Set up a Renku session

1. Start a Renku Linux session with an NVIDIA GPU allocation.

2. Open a terminal and confirm that the Renku host environment and GPU are
   available:

   ```bash
   which python
   nvidia-smi
   ```

3. From the repository root, run the setup script:

   ```bash
   cd ~/work/object_ctrl
   bash scripts/setup_renku.sh 2>&1 | tee setup_renku.log
   ```

The script:

- creates or reuses the isolated `.venv-renku` project environment;
- installs the CUDA-enabled PyTorch stack and notebook dependencies;
- installs `object_ctrl` and the compatible YOLOX revision;
- verifies CUDA, OpenCV, ONNX, and the installed dependencies;
- registers the `Python (object_ctrl Renku)` Jupyter kernel; and
- synchronizes every `notebooks/*.py` source to a paired `.ipynb` file.

The setup is safe to run again after pulling dependency or notebook changes.

## Activate the Renku terminal environment

After setup, source the activation script in each new terminal shell:

```bash
source scripts/activate_renku_env.sh
```

This activates `.venv-renku`, selects the registered Renku notebook kernel,
prevents user-installed Python packages from leaking into the environment, and
adds the project's `scripts/` directory to `PATH`.

`scripts/tmux_notebook.sh run` sources this activation script automatically,
so detached notebook execution does not require prior shell activation.

## Run a notebook

1. In Renku's file browser, open a generated notebook under `notebooks/`, for
   example `notebooks/nb02.02-ultra_yolo11n_large_basketball.ipynb`.
2. Select the `Python (object_ctrl Renku)` kernel.
3. Run the notebook cells normally.

Confirm the selected kernel from a notebook cell when needed:

```python
import sys

import torch

print(sys.executable)
print(torch.cuda.get_device_name(0))
print(torch.cuda.is_available())
```

The Python path should end in `.venv-renku/bin/python`, and CUDA availability
should be `True`.

## RF-DETR Small on the large basketball dataset

Use `nb04.02-rfdetr_small_large_basketball` in a Linux Renku GPU session.
Keep `.venv-renku` for Ultralytics and YOLOX; RF-DETR has its own environment
and kernel. The local small-dataset experiment remains deferred.

### Set up and select the RF-DETR runtime

From the repository root:

```bash
bash scripts/setup_renku_rfdetr.sh

export OBJCTRL_RENKU_VENV="$PWD/.venv-renku-rfdetr"
export OBJCTRL_RENKU_KERNEL_NAME=object-ctrl-renku-rfdetr
export NOTEBOOK_KERNEL=object-ctrl-renku-rfdetr
source scripts/activate_renku_env.sh
```

Set all three overrides: an existing `NOTEBOOK_KERNEL` takes precedence over
the activation script's default. In Renku's Jupyter interface, select kernel
`object-ctrl-renku-rfdetr`, displayed as `Python (object_ctrl Renku RF-DETR)`.
The interpreter should end in `.venv-renku-rfdetr/bin/python`.

Setup installs the pinned RF-DETR training and ONNX export stack, headless
OpenCV, the shared project package, and notebook tools. It checks CUDA
tensor/NMS operations and a pretrained Small forward pass at 640 pixels,
registers the kernel, and syncs only this notebook pair. It records GPU,
package, kernel, and host/YOLO package inventories under
`outputs/environment/rfdetr/`. Run setup again to check repeatability;
notebook training, resume, and evaluation still require their own Renku smoke
checks.

The default PyTorch wheel index is CUDA 13.0. For a driver requiring another
supported build of the pinned torch/torchvision pair, rerun setup with, for
example, `OBJCTRL_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`.
The installed stack must pass the actual GPU checks. Setup-specific overrides
are `OBJCTRL_RFDETR_PYTHON`, `OBJCTRL_RFDETR_VENV`, and
`OBJCTRL_RFDETR_KERNEL_NAME`; when customizing the latter two, also update
the activation overrides above.

### Smoke, train, resume, and evaluate

The notebook reuses the frozen
`datasets/composed/coco_basketball_11501_1156_1395` splits. It prepares linked
RF-DETR inputs under `data/processed/rfdetr/`; weights and caches stay under
`models/`, and experiment artifacts go under `outputs/runs/basketball/`.

First run two epochs on bounded subsets of 16 training, 8 validation, and
8 test images:

```bash
unset RFDETR_RUN_DIR
export RFDETR_MODE=fresh RFDETR_SMOKE=1 RFDETR_EPOCHS=2
bash scripts/tmux_notebook.sh run \
  --file notebooks/nb04.02-rfdetr_small_large_basketball.ipynb \
  --session rfdetr-small-smoke
bash scripts/tmux_notebook.sh check --session rfdetr-small-smoke
```

Logs are under `outputs/notebook_logs/`. Use a distinct tmux session name for
each execution; completed panes remain available for inspection. Fresh runs
automatically create a new experiment directory. Smoke artifacts are marked
and excluded from the full comparison.

The TQDM progress bar is live when running interactively in Jupyter. Detached
`nbconvert` captures cell output in the executed notebook instead of streaming
it into the tmux log. During a detached run, check the process and GPU activity:

```bash
bash scripts/tmux_notebook.sh check --session rfdetr-small-large
watch -n 5 nvidia-smi
tail -F outputs/runs/basketball/SELECTED_RFDETR_RUN/metrics.csv
```

Replace `SELECTED_RFDETR_RUN` with the allocated run directory. `metrics.csv`
is updated after logged training and validation steps; the GPU display is the
useful heartbeat while the first epoch is still running.

After the smoke checks pass, start the full 100-epoch experiment:

```bash
unset RFDETR_RUN_DIR
export RFDETR_MODE=fresh RFDETR_SMOKE=0 RFDETR_EPOCHS=100
bash scripts/tmux_notebook.sh run \
  --file notebooks/nb04.02-rfdetr_small_large_basketball.ipynb \
  --session rfdetr-small-large
```

The input is fixed at 640 pixels, overriding Small's 512-pixel default.
Training defaults to microbatch 4 with accumulation 1; increase
`RFDETR_BATCH_SIZE` if GPU memory permits. RF-DETR 1.10.1 and the pinned
Lightning release both normalize accumulated losses, so
`RFDETR_GRAD_ACCUM_STEPS>1` changes gradient scaling. It is an explicit
advanced override, not an equivalent replacement for a larger physical batch.
The run records its actual batch and training settings.

One hundred epochs is the maximum. Early stopping monitors EMA validation
AP50:95 every epoch and stops after 10 consecutive epochs without an
improvement of at least 0.001. Override the patience for a new run with
`RFDETR_EARLY_STOPPING_PATIENCE`; set `RFDETR_EARLY_STOPPING=0` to require the
full budget. The held-out test split never controls stopping.

To continue an interrupted run, use the exact directory printed in its log:

```bash
unset RFDETR_SMOKE RFDETR_EPOCHS
export RFDETR_MODE=resume
export RFDETR_RUN_DIR="$PWD/outputs/runs/basketball/SELECTED_RFDETR_RUN"
bash scripts/tmux_notebook.sh run \
  --file notebooks/nb04.02-rfdetr_small_large_basketball.ipynb \
  --session rfdetr-small-resume
```

Replace `SELECTED_RFDETR_RUN` with the selected run, including any numeric or
smoke suffix. Resume requires `last.ckpt` from a completed epoch, including
optimizer/scheduler state; it retains the original epoch budget and verifies
the saved dataset identity and settings. Clear other experimental overrides
if they differ from that run. Check resume on an interrupted smoke run before
relying on it for a full experiment.

For a completed run, reload the selected checkpoint and rerun the plots,
evaluation, and comparison without fitting:

```bash
unset RFDETR_SMOKE RFDETR_EPOCHS
export RFDETR_MODE=evaluate
export RFDETR_RUN_DIR="$PWD/outputs/runs/basketball/SELECTED_RFDETR_RUN"
bash scripts/tmux_notebook.sh run \
  --file notebooks/nb04.02-rfdetr_small_large_basketball.ipynb \
  --session rfdetr-small-evaluate
```

Keep `RFDETR_RUN_DIR` set to that completed run. Best weights are selected
using validation AP50:95; fitting does not evaluate the held-out test split.

The notebook launches ONNX export through a fresh Python worker instead of
inside IPython. The worker disables RF-DETR's verbose graph dump, validates a
temporary ONNX graph, and atomically replaces the stable artifact only after
validation succeeds. Existing exports are validated before reuse. To run the
same worker directly from the repository root:

```bash
.venv-renku-rfdetr/bin/python scripts/export_rfdetr_onnx.py \
  --run-dir outputs/runs/basketball/SELECTED_RFDETR_RUN
```

### Export the existing YOLO baselines

Export nb02.02 (YOLO11n), nb03.02 (YOLOX-Tiny), and nb03.03 (YOLOX-Nano)
with one command:

```bash
bash scripts/export_basketball_baselines.sh

export RFDETR_BASELINE_EXPORT_DIR=\
"$PWD/outputs/comparisons/basketball_large_dataset/baselines"
```

For each model, the exporter finds directories matching its notebook run name
under `outputs/runs/basketball`, excludes incomplete and evaluation-only
directories, and selects the one with the newest directory modification time.
Pass `--run-dir` to `export_basketball_predictions.py` to select an exact run.
The wrapper defaults can also be changed with `BASKETBALL_RUNS_DIR`,
`BASKETBALL_DATASET_DIR`, `BASKETBALL_EXPORT_DIR`,
`BASKETBALL_EXPORT_DEVICE`, and `BASKETBALL_EXPORT_RESOLUTION`.

The exporter loads `weights/best.pt` or `weights/best_ckpt.pth`, exports both
validation and test predictions by default, and never retrains. Existing export
files are not overwritten; choose another output directory for a new comparison.
Set `RFDETR_BASELINE_EXPORT_DIR` before notebook execution, or rerun in evaluation
mode after exporting. Missing baseline exports appear as unavailable rather than
fabricated scores.

The notebook recomputes all three models' metrics with one COCO evaluator:
AP at score >= 0.001 and `maxDets=[1,10,100]`, plus precision/recall/F1 at
score >= 0.25 and IoU >= 0.50. Negative-image false detections are reported
separately. Native framework metrics remain distinct, and original YOLO
training data identity is limited by the provenance saved in those old runs.

For timings, add `--benchmark` to both exports. RF-DETR benchmarking is enabled
by default in the notebook; set `RFDETR_BENCHMARK=0` only to skip it. Remeasure
all models on the same Renku GPU. These batch-one FP32 measurements include
preprocessing and native postprocessing, exclude file loading, and use
warm-up/CUDA synchronization; they are not model-only latency and must not be
ranked against historical MPS timings. Comparison tables are saved under
`outputs/comparisons/basketball_large_dataset/<RF-DETR run>/`.

## Notes

- Keep edits in the Jupytext `.py` sources. Generated `.ipynb` files are
  ignored by Git.
- Datasets remain under `datasets/`, models under `models/`, and training
  results under `outputs/`.
- The large Ultralytics notebook defaults to zero dataloader subprocesses to
  avoid exhausting Renku's limited `/dev/shm` allocation.
- If imports come from Renku's host `.venv`, reselect the
  `Python (object_ctrl Renku)` kernel and restart it.
- If CUDA is unavailable, stop the session and start one with a GPU allocation
  before rerunning the setup script.
