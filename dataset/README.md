# Dataset Builder

COCO/Datumaro preparation and Roboflow downloads. Input data defaults to
`WORKSPACE_ROOT/data`; generated datasets should be placed below
`DATA_ROOT/composed`. Do not store downloaded datasets in this source project.

From the workspace root, run `bash scripts/setup_conda_envs.sh dataset` and
`conda run -n dataset-dev python -m pytest dataset/tests`. Set
`ROBOFLOW_API_KEY` in the environment before using Roboflow downloads. The
notebooks in `notebooks/` use the `object-detection-dataset` kernel.
