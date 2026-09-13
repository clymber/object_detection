# Object Detection Workspace

This Git repository contains separate Python projects for dataset building,
Ultralytics, YOLOX, RF-DETR, evaluation, and shared utilities. Open
[`object-detection.code-workspace`](object-detection.code-workspace) in VS Code
to give each project its own interpreter, tests, and notebook kernel.

The current migration stage is **local Conda development**. The existing root
`object_ctrl` package, root notebooks, and Renku setup scripts remain as a
temporary compatibility path. Do not use their single-environment instructions
for new local subproject work. Renku migration begins only after the local
workflow is reviewed.

## Projects

| Project | Purpose | Local environment |
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

## Local Setup and Tests

Install Conda first. From this repository root, set
up only the projects you need:

```bash
bash scripts/setup_conda_envs.sh detection_common evaluation dataset
bash scripts/setup_conda_envs.sh rfdetr
bash scripts/setup_conda_envs.sh ultralytics yolox
bash scripts/setup_conda_envs.sh notebook-tools
```

With no arguments, the script sets up all projects. It creates or updates each
named Conda environment from its own `environment.yml`, installs internal
packages editable in dependency order, and registers notebook kernels. YOLOX
also needs an upstream source checkout beside this repository at `../YOLOX`,
or an absolute `YOLOX_SOURCE` pointing to one. Its version is recorded in
[`yolox/README.md`](yolox/README.md).

Run tests independently with the owning interpreter, for example:

```bash
conda run -n detection-common-dev python -m pytest detection_common/tests
conda run -n evaluation-dev python -m pytest evaluation/tests
conda run -n dataset-dev python -m pytest dataset/tests
conda run -n rfdetr-dev python -m pytest rfdetr/tests
conda run -n ultralytics-dev python -m pytest ultralytics/tests
conda run -n yolox-dev python -m pytest yolox/tests
```

GPU-only and full-training checks are deferred to Renku. Package-local unit
tests and lightweight notebook smoke tests are the local acceptance checks.

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
Synchronize the pairs and run an explicit notebook through its project kernel:

```bash
make sync-notebooks
make run-notebook NOTEBOOK=dataset/notebooks/smoke.ipynb
```

The runner intentionally does not execute every training notebook by default.
Select the corresponding `object-detection-<project>` kernel in VS Code.

## Renku Compatibility During Local Migration

The root `scripts/setup_renku.sh`, `scripts/setup_renku_rfdetr.sh`, and
`scripts/tmux_notebook.sh` still support the legacy root notebooks. See the
[existing Renku operator guide](scripts/renku/README.md). Do not infer that
new project-local `.venv` environments or CUDA paths are validated yet; those
are Stage B of the plan.
