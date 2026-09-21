# Evaluation

Framework-neutral COCO evaluation, versioned prediction artifacts, and
cross-model comparison. This environment must not install Ultralytics, YOLOX,
or RF-DETR; producers in those projects write the shared artifact schema.

From the workspace root, run `bash scripts/setup_conda_envs.sh evaluation`.
Then use `conda run -p .conda/envs/object-detection-evaluation
python -m pytest evaluation/tests`.
Artifacts should be written below `OUTPUT_ROOT/evaluation`, which defaults to
`WORKSPACE_ROOT/outputs/evaluation`.

`detection_evaluation.baseline` validates frozen, full-dataset YOLO runs and
writes versioned split artifacts plus metrics. The model packages own their
checkpoint loading and native prediction adapters. The evaluation environment
does not import or install those frameworks.

## Run Protocol And Bundles

`create_run_protocol` captures immutable dataset/model provenance in a run
before timed training. It records the logical dataset and model, source
notebook, original UTC, settings, persisted smoke flag, and compatible
canonical/loader dataset identities. `start_training_attempt` and
`finalize_training_attempt` persist synchronized monotonic attempts; callers
provide their framework-specific synchronization callback and may inject a
clock for tests. `capture_training_hardware` records training hardware without
depending on inference benchmarks.

The training attempt owns an exclusive file lock until finalization. A concurrent
trainer is rejected; explicit resume after a hard termination preserves the old
attempt with unknown duration. Once resumed training completes, its bundle is
valid but total/average training time remains unavailable. Recovery never adds
an attempt. Model callers provide completed epoch counts and available GPU/runtime
details without reconstructing elapsed time.

`publish_bundle` stages schema-v1 validation/test prediction artifacts and the
training record below a supplied bundle directory, fsyncs them into an immutable
generation, then atomically switches `manifest.json`. `read_bundle` snapshots
one manifest and validates all referenced SHA256 digests, provenance, schemas,
and smoke policy. Pass `allow_smoke=True` only for explicit smoke inspection;
full comparison readers retain the default rejection.

The setup command selects environment-macos-mps.yml on Apple Silicon and
environment-linux-cuda.yml on Renku Linux. Both profiles use the checkout-local
`.conda/envs/object-detection-evaluation` prefix; on Renku, bootstrap Miniforge first from the
workspace root with bash scripts/install_miniforge_renku.sh.
