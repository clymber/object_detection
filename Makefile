# Project-owned Jupytext notebooks on macOS and Renku.
SHELL := /bin/bash

CONDA_BIN ?= conda

CONDA_RUN := env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \
    -u PIP_PREFIX -u PIP_TARGET -u PIP_USER $(CONDA_BIN) run -p

NOTEBOOK_SOURCES := $(wildcard dataset/notebooks/*.py) \
    $(wildcard ultralytics/notebooks/*.py) \
    $(wildcard yolox/notebooks/*.py) \
    $(wildcard rfdetr/notebooks/*.py) \
    $(wildcard evaluation/notebooks/*.py)

.PHONY: sync-notebooks check-notebooks run-notebook test \
    test-common test-evaluation test-dataset test-rfdetr test-ultralytics test-yolox

sync-notebooks:
	$(CONDA_RUN) .conda/envs/object-detection-notebooks \
	    jupytext --sync $(NOTEBOOK_SOURCES)

check-notebooks:
	$(CONDA_RUN) .conda/envs/object-detection-notebooks \
	    jupytext --test-strict --to ipynb \
	    $(NOTEBOOK_SOURCES)

run-notebook:
	@test -n "$(NOTEBOOK)" || \
	    (echo 'Set NOTEBOOK=<project>/notebooks/<name>.ipynb' >&2; exit 2)
	bash scripts/run_notebook.sh "$(NOTEBOOK)"

test: test-common test-dataset test-rfdetr test-ultralytics test-yolox test-evaluation

test-common:
	@$(CONDA_RUN) .conda/envs/object-detection-common \
	    python -m pytest detection_common/tests

test-evaluation:
	@$(CONDA_RUN) .conda/envs/object-detection-evaluation \
	    python -m pytest evaluation/tests

test-dataset:
	@$(CONDA_RUN) .conda/envs/object-detection-dataset \
	    python -m pytest dataset/tests

test-rfdetr:
	@$(CONDA_RUN) .conda/envs/object-detection-rfdetr \
	    python -m pytest rfdetr/tests

test-ultralytics:
	@$(CONDA_RUN) .conda/envs/object-detection-ultralytics \
	    python -m pytest ultralytics/tests

test-yolox:
	@$(CONDA_RUN) .conda/envs/object-detection-yolox \
	    python -m pytest yolox/tests
