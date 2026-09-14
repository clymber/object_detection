# RF-DETR Pipeline

RF-DETR training and inference helpers, ONNX export, and model-owned prediction
artifact production. Dataset preprocessing lives in `dataset-builder`;
framework-neutral evaluation lives in `detection-evaluation`.

From the workspace root, run `bash scripts/setup_conda_envs.sh rfdetr` and
`conda run -n rfdetr-dev python -m pytest rfdetr/tests`. Notebooks under
`notebooks/` use the `object-detection-rfdetr` kernel. Inputs default to
`WORKSPACE_ROOT/data`; runs default to `OUTPUT_ROOT/runs/basketball`.
Override shared roots with absolute `OBJECT_DETECTION_DATA_ROOT` and
`OBJECT_DETECTION_OUTPUT_ROOT` values when needed.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles keep the same
named Conda environment; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.

On Renku, the Linux profile uses CUDA 13.0 PyTorch wheels and
headless OpenCV. Verify CUDA NMS and a synthetic RF-DETR Small
prediction before training with real data.
