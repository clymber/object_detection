# Dataset Builder

COCO/Datumaro preparation and Roboflow downloads. Input data defaults to
`WORKSPACE_ROOT/data`; generated datasets should be placed below
`DATA_ROOT/composed`. Do not store downloaded datasets in this source project.

From the workspace root, run `bash scripts/setup_conda_envs.sh dataset` and
`conda run -n dataset-dev python -m pytest dataset/tests`. Set
`ROBOFLOW_API_KEY` in the environment before using Roboflow downloads. The
notebooks in `notebooks/` use the `object-detection-dataset` kernel.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles keep the same
named Conda environment; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
