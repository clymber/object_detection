# Todo List

## New features

- [ ] feat: Train a yolox nano on football
  - [x] Model: yolox nano; dataset: football
  - [ ] Save staged checkpoint every 10 epochs
  - [ ] Export staged checkpoints to ONNX format
- [ ] feat: a script to export latest trained models
  - [ ] export the latest models
  - [ ] automatically discover latest trained models
- [x] feat: support reading notebook list from a file
  - [x] `scripts/run_notebooks.py` requires `-f|--file`
  - [x] read ordered notebook tasks and statuses from a JSON file
  - [x] support per-notebook environment settings and log paths
- [ ] feat: support real-time saving notebooks on CLI execution
  - [ ] `scripts/run_notebook.sh` (used by `scripts/run_notebooks.py`)
  - [ ] Makefile
- [ ] feat: add `class Dataset` under `dataset/src/dataset_builder`
  - [ ] directory tree: `<dataset_name>/{coco,yolo}`
  - [ ] method: `Dataset.get_dataset(fmt: str) -> str`

## Fix

- [ ] fix: notebooks on command line execution are too verbose
- [x] fix: container image - `vim` command is absent
