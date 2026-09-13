# Detection Common

Framework-neutral path, file, and display helpers used by the Python
subprojects. Keep model-framework imports out of this package.

From the workspace root, run `bash scripts/setup_conda_envs.sh detection_common`.
Then use `conda run -n detection-common-dev python -m pytest
detection_common/tests`. Data defaults to `WORKSPACE_ROOT/data` and generated
artifacts to `WORKSPACE_ROOT/outputs`; set absolute
`OBJECT_DETECTION_DATA_ROOT` or `OBJECT_DETECTION_OUTPUT_ROOT` to override them.
