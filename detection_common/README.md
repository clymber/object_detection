# Detection Common

Framework-neutral path, file, and display helpers used by the Python
subprojects. Keep model-framework imports out of this package.

From the workspace root, run `bash scripts/setup_conda_envs.sh
detection_common`. Then use `conda run -p
.conda/envs/object-detection-common python -m pytest
detection_common/tests`. Data defaults to `WORKSPACE_ROOT/data` and
generated artifacts to `WORKSPACE_ROOT/outputs`; set absolute
`OBJECT_DETECTION_DATA_ROOT` or `OBJECT_DETECTION_OUTPUT_ROOT` to override them.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-common` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
