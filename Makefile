# Local project-owned Jupytext notebooks. Legacy Renku scripts remain separate.
SHELL := /bin/bash

NOTEBOOK_SOURCES := $(wildcard dataset/notebooks/*.py) \
    $(wildcard ultralytics/notebooks/*.py) \
    $(wildcard yolox/notebooks/*.py) \
    $(wildcard rfdetr/notebooks/*.py) \
    $(wildcard evaluation/notebooks/*.py)

.PHONY: sync-notebooks check-notebooks run-notebook

sync-notebooks:
	conda run -n notebook-tools jupytext --sync $(NOTEBOOK_SOURCES)

check-notebooks:
	conda run -n notebook-tools jupytext --test-strict --to ipynb \
	    $(NOTEBOOK_SOURCES)

run-notebook:
	@test -n "$(NOTEBOOK)" || \
	    (echo 'Set NOTEBOOK=<project>/notebooks/<name>.ipynb' >&2; exit 2)
	bash scripts/run_local_notebook.sh "$(NOTEBOOK)"
