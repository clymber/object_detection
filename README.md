# Object Detection Workspace

This Git repository contains separate Python projects for dataset building,
Ultralytics, YOLOX, RF-DETR, evaluation, and shared utilities. Open
[`object-detection.code-workspace`](object-detection.code-workspace) in VS Code
to give each project its own interpreter, tests, and notebook kernel.

## Projects

| Project | Purpose | Local Conda prefix |
| --- | --- | --- |
| [Detection Common](detection_common/README.md) | Shared helpers and paths | `object-detection-common` |
| [Evaluation](evaluation/README.md) | Neutral artifacts and metrics | `object-detection-evaluation` |
| [Dataset](dataset/README.md) | COCO/Datumaro preparation | `object-detection-dataset` |
| [RF-DETR](rfdetr/README.md) | RF-DETR experiments | `object-detection-rfdetr` |
| [Ultralytics](ultralytics/README.md) | Ultralytics experiments | `object-detection-ultralytics` |
| [YOLOX](yolox/README.md) | YOLOX experiments | `object-detection-yolox` |
| [Notebook Tools](notebook-tools/README.md) | Jupytext/Jupyter tooling | `object-detection-notebooks` |

Supporting reports and archives are in [`documents/`](documents/).

Clone the workspace with its pinned official YOLOX submodule:

```bash
git clone --recurse-submodules <repository-url>
```

For an existing clone, initialize it with
`git submodule update --init -- thirdparty/YOLOX`.

## Conda Setup and Tests

On macOS, use your existing Conda installation. On Renku Linux x86_64,
install Miniforge once on the persistent work volume:

```bash
bash scripts/install_miniforge_renku.sh
source /home/renku/work/miniforge3/etc/profile.d/conda.sh
```

From the repository root, set up only the projects you need:

```bash
bash scripts/setup_conda_envs.sh detection_common evaluation dataset
bash scripts/setup_conda_envs.sh rfdetr
bash scripts/setup_conda_envs.sh ultralytics yolox
bash scripts/setup_conda_envs.sh notebook-tools
```

With no arguments, the script sets up all projects. It creates or updates each
environment below `WORKSPACE_ROOT/.conda/envs/<object-detection-name>` from
`environment-macos-mps.yml` on macOS or `environment-linux-cuda.yml` on Renku,
installs internal packages editable in dependency order, and registers kernels
inside the project environments. Descriptive prefix names stay legible in Conda
and VS Code while remaining isolated from another workspace copy.
Setup also registers the workspace-local `envs` directory in the user's Conda
configuration so `conda env list` can show these prefix names from any shell.
On both platforms, setup builds the pinned `thirdparty/YOLOX` submodule from a
temporary copy with C++17. The YOLOX environment manifests select regular
OpenCV on macOS and headless OpenCV on Renku.

Run tests independently with the owning interpreter, for example:

```bash
conda run -p .conda/envs/object-detection-common python -m pytest detection_common/tests
conda run -p .conda/envs/object-detection-evaluation python -m pytest evaluation/tests
conda run -p .conda/envs/object-detection-dataset python -m pytest dataset/tests
conda run -p .conda/envs/object-detection-rfdetr python -m pytest rfdetr/tests
conda run -p .conda/envs/object-detection-ultralytics python -m pytest ultralytics/tests
conda run -p .conda/envs/object-detection-yolox python -m pytest yolox/tests
```

The Linux model environments use official CUDA 13.0 PyTorch wheels inside
Conda. Verify CUDA, Torchvision NMS, headless OpenCV, and project smoke tests
on an allocated GPU. Full training and real-checkpoint exports require their
own datasets and checkpoints; local macOS tests do not establish GPU support.

## Paths and Artifacts

Every installed source project detects `SUBPROJECT_ROOT` from its own module
when its notebook imports `<package>.config`. `WORKSPACE_ROOT` defaults to the
parent repository; `DATA_ROOT` defaults to `WORKSPACE_ROOT/data`, and
`OUTPUT_ROOT` to `WORKSPACE_ROOT/outputs`. Set absolute
`OBJECT_DETECTION_WORKSPACE_ROOT`, `OBJECT_DETECTION_DATA_ROOT`, or
`OBJECT_DETECTION_OUTPUT_ROOT` to override these paths. Notebook working
directory does not determine project ownership.

Keep downloaded data and checkpoints out of source-project roots.
Model producers write versioned prediction artifacts; `evaluation` reads
them without importing model frameworks.

## Notebooks

Project-owned notebooks are stored as Jupytext `py:percent` sources under each
project's `notebooks/` directory. Generated `.ipynb` files are ignored by Git.
To synchronize existing files manually or run an explicit notebook through its
project kernel:

```bash
make sync-notebooks
make run-notebook NOTEBOOK=dataset/notebooks/smoke.ipynb
```

The runner intentionally does not execute every training notebook by default.
Select the corresponding `object-detection-<project>` kernel in VS Code.

## Notebook Task Lists

Copy the template and set the notebooks you want to run to `pending`:

```bash
cp scripts/notebook_tasks_template.json notebook_tasks.json
make sync-notebooks
python3 scripts/run_notebooks.py -f notebook_tasks.json --dry-run
python3 scripts/run_notebooks.py -f notebook_tasks.json
```

Tasks run in listed order and stop on the first failure.

- **Paths:** `-f|--file` is required and resolves from your current directory.
  Notebook and log paths resolve from the workspace root. A `.py` entry runs its paired
  `.ipynb`, generated by `make sync-notebooks`.
- **Status:** Only `pending` entries or entries without a status run; other statuses are
  skipped. Each run records `running`, then `finished` or `error`. Set an entry back to
  `pending` to retry it.
- **Environment:** Each entry's `env` overrides shell settings for that notebook only.
  See [the plan](documents/plans/run_notebooks.md) for available settings.
- **Logs:** Process output appears in the terminal and is appended to `log`. If omitted,
  a unique log path under `outputs/notebook_logs/` is saved in the task file. Executed
  notebooks are saved in place.
- **Preview:** `--dry-run` shows tasks, skip reasons, environment overrides, and log
  paths without changing files.

Edit statuses while the batch is stopped. Ctrl+C stops execution and records `error`. If
a forced exit leaves an entry `running`, confirm the old process has stopped before
resetting it to `pending`.

Omit notebooks requiring interactive FiftyOne App operations or mark them `inactivated`.

For a detached run, start tmux session manually then run the same foreground command:

```bash
tmux new -s notebooks
python3 scripts/run_notebooks.py -f notebook_tasks.json
```

Detach with Ctrl+B, then D, and reattach with `tmux attach -t notebooks`.
Single-notebook execution through `make run-notebook` or
`bash scripts/run_notebook.sh nb03.05` remains available.
