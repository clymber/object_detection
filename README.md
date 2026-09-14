# Object Detection Workspace

This Git repository contains separate Python projects for dataset building,
Ultralytics, YOLOX, RF-DETR, evaluation, and shared utilities. Open
[`object-detection.code-workspace`](object-detection.code-workspace) in VS Code
to give each project its own interpreter, tests, and notebook kernel.

Stage B now adds Renku Conda environments alongside the locally tested macOS
ones. The existing root `object_ctrl` package, notebooks, and virtual
environments remain a compatibility path until the new project environments
pass their Renku checks. Outstanding local VS Code and clean-creation review
items remain open in the migration status.

## Projects

| Project | Purpose | Conda environment |
| --- | --- | --- |
| [Detection Common](detection_common/README.md) | Shared helpers and paths | `detection-common-dev` |
| [Evaluation](evaluation/README.md) | Neutral artifacts and metrics | `evaluation-dev` |
| [Dataset](dataset/README.md) | COCO/Datumaro preparation | `dataset-dev` |
| [RF-DETR](rfdetr/README.md) | RF-DETR experiments | `rfdetr-dev` |
| [Ultralytics](ultralytics/README.md) | Ultralytics experiments | `ultralytics-dev` |
| [YOLOX](yolox/README.md) | YOLOX experiments | `yolox-dev` |
| [Notebook Tools](notebook-tools/README.md) | Jupytext/Jupyter tooling | `notebook-tools` |

Workspace-wide material remains in [`documents/`](documents/). The former
Colab tree is archived as `documents/notebooks-colabs.tgz` and is unsupported
in this migration. The [monorepo plan](plans/monorepo_plan.md) records the
local completion gate and deferred Renku work. The
[local migration status](documents/local_migration_status.md) records checks
and remaining review items.

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
named Conda environment from `environment-macos-mps.yml` on macOS or
`environment-linux-cuda.yml` on Renku, installs internal
packages editable in dependency order, and registers notebook kernels.
On macOS, YOLOX needs an upstream checkout at `../YOLOX` or an absolute
`YOLOX_SOURCE`. On Renku, setup fetches and builds the pinned revision with
headless OpenCV and C++17 adjustments.

Run tests independently with the owning interpreter, for example:

```bash
conda run -n detection-common-dev python -m pytest detection_common/tests
conda run -n evaluation-dev python -m pytest evaluation/tests
conda run -n dataset-dev python -m pytest dataset/tests
conda run -n rfdetr-dev python -m pytest rfdetr/tests
conda run -n ultralytics-dev python -m pytest ultralytics/tests
conda run -n yolox-dev python -m pytest yolox/tests
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

The old ignored `datasets/` directory is not moved automatically. If your
local data is still there, set `OBJECT_DETECTION_DATA_ROOT` to its absolute
path before running the new notebooks. Keep downloaded data and checkpoints
out of source-project roots. Model producers write versioned prediction
artifacts; `evaluation` reads them without importing model frameworks.

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

## Renku Conda and Legacy Compatibility

Miniforge and the named environments live at
`/home/renku/work/miniforge3`, outside the repository. Setup is selective;
the temporary root `objctrl` profile is installed only with
`bash scripts/setup_conda_envs.sh legacy-root`. Its new kernel is
`object-ctrl-renku-conda` so it cannot replace the existing legacy kernel
during validation. The separate root `environment-rfdetr.yml` remains
unchanged while full RF-DETR parity checks stay open.

The existing `scripts/setup_renku.sh`, `scripts/setup_renku_rfdetr.sh`,
`scripts/tmux_notebook.sh`, and their [operator guide](scripts/renku/README.md)
remain available for root notebooks until the Conda replacements pass. Use
`scripts/tmux_conda_notebook.sh` for detached project-owned notebooks, for
example:

```bash
bash scripts/tmux_conda_notebook.sh run --file evaluation/notebooks/smoke.ipynb --session evaluation-smoke
bash scripts/tmux_conda_notebook.sh check --session evaluation-smoke
```

The [monorepo plan](plans/monorepo_plan.md) defines the cutover checks; the
[migration status](documents/local_migration_status.md) records what has
actually passed.
