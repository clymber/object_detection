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

.PHONY: sync-notebooks check-notebooks run-notebook

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
