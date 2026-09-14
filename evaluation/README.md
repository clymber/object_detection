# Evaluation

Framework-neutral COCO evaluation, versioned prediction artifacts, and
cross-model comparison. This environment must not install Ultralytics, YOLOX,
or RF-DETR; producers in those projects write the shared artifact schema.

From the workspace root, run `bash scripts/setup_conda_envs.sh evaluation`.
Then use `conda run -n evaluation-dev python -m pytest evaluation/tests`.
Artifacts should be written below `OUTPUT_ROOT/evaluation`, which defaults to
`WORKSPACE_ROOT/outputs/evaluation`.

`detection_evaluation.baseline` validates frozen, full-dataset YOLO runs and
writes versioned split artifacts plus metrics. The model packages own their
checkpoint loading and native prediction adapters. The evaluation environment
does not import or install those frameworks.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles keep the same
named Conda environment; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
