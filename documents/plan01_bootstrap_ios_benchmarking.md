# ENABLE — Native iOS Core ML benchmarking plan

- **Device:** iPhone 12 Pro
- **Models:** RF-DETR Small, RF-DETR Nano, YOLOX Nano
- **Platforms:** Native Swift/Core ML and MoveAhead Unity + iOS

Compare detection throughput, latency, freshness and frame coverage under the
[shared protocol][protocol]. Then test Apple Vision tracking with all three models.
MoveAhead owns the Unity implementation and measurements.

## Current baseline

- Protocol **0.4-draft**: Release, native compute units `.all`, detector only,
  preview off, 640×640 model inputs.
- Primary clip: `basketball_dribbling_h264.mp4`, 960×540 at approximately 29.97 FPS.
  Use its actual timestamps for paced playback. Camera capture target: 30 FPS,
  with the actual rate logged. NBA and UHD clips are additional inputs.
- The [video catalog][catalog] records file identities and timestamps. Benchmark
  frame selections and portable input/model manifests still need freezing.
- RF-DETR Core ML uses FP16; YOLOX Nano uses FP32. Selected runs, checkpoint
  provenance and export details are in the [model inventory][inventory].

## Completed implementation

- [x] Configure Xcode, signing and the physical iPhone; build and launch Release.
- [x] Export all three selected checkpoints to ONNX and Core ML, preserving EMA
  selection and recording tensor contracts, transforms, hashes and dependencies.
- [x] Implement worker-based model loading, preprocessing, decoding, model switching
  and cancellation; pass Mac development parity and iOS typechecks.
- [x] Add model/input selectors, still-image inference, photo/video import and
  bundled test inputs.
- [x] Implement offline and paced video detection with timestamps, bounded queues,
  freshness rules, per-frame outcomes and separate model/frame/display metrics.
- [x] Implement live camera capture, timestamp mapping, drop accounting, lifecycle
  handling and optional processed-frame previews with aligned boxes.
- [x] Reconcile coverage and export `frames.jsonl`, `timings.csv`, `mobile_run.json`;
  verify saving/sharing and saved-folder readback on the iPhone.
- [x] Run RF-DETR Small compute-unit diagnostics and inspect Xcode reports.

Physical camera, orientation, Start/Stop, background-return and export checks passed
on 23 November 2026. Implementation details and checks are in the
[integration guide][integration]. These diagnostic runs are separate from formal
benchmark collection.

## Validation status

| Model | Frozen still-image parity on iPhone | Video parity |
| --- | --- | --- |
| RF-DETR Small | 25/25 pass | Pending |
| RF-DETR Nano | 24/25 pass; confidence mismatch on image 6998 | Pending |
| YOLOX Nano | 25/25 pass | Pending |

Resolve the [Nano mismatch][nano-mismatch] using matching decoded pixels and the
existing tolerances. Parity failures block timing acceptance for that configuration;
other validated configurations can proceed.

RF-DETR Small's fusion warnings recur alongside successful detection. Xcode reports
succeed for `all`, CPU-only and CPU/Neural Engine. CPU/GPU report creation fails
with error `-5`, while app inference and numerical checks under `.cpuAndGPU` pass.
Keep `.all` as the baseline; the [diagnostic report][coreml-diagnostics] preserves
the evidence and limitations. Warning timing and performance impact remain open.

## Next work

### 1. Freeze inputs and complete video parity

Start with RF-DETR Small on the dribbling clip, then use the same workflow for Nano
and YOLOX Nano.

- [ ] Create `input_manifest.json` from the existing catalog. Fix the ordered parity,
  30 warm-up and 300 timed frame selections, decoded-pixel hashes, loop policy and
  preload memory budget.
- [ ] Fix the camera resolution, lens, pixel format, orientation, mirroring and scene
  conditions, with a 30 FPS capture target.
- [ ] Complete export provenance, including the exact Python version in each export
  manifest. Commit a portable `model_manifest.json` with selected artifact identities,
  category mapping, transforms and runtime/export versions.
- [ ] Confirm the shared inputs and settings with MoveAhead before collection.
- [ ] Generate checkpoint and ONNX references and compare identical decoded pixels
  with iPhone Core ML, without pacing or skipping. Include empty detections,
  confidence/NMS boundary cases and source-coordinate box checks.
- [ ] Save per-model parity reports, resolve failures and verify hashes and coverage
  against the frozen manifests. Recheck the benchmark export round trip.
- [ ] During an identified device parity run, preserve the full Xcode console with
  app/model identity and correlate fusion messages with load/prediction phases.

### 2. Add formal benchmark controls

Reuse the existing video/camera runners and exports; the remaining work is the
protocol's measurement controller.

- [ ] Preload frozen decoded frames and the model, then automate 30 warm-up frames,
  300 timed frames and three repetitions. Keep decode, loading, rendering and file
  I/O outside timed requests.
- [ ] Add separate model-only and end-to-end measurement modes and benchmark UI
  controls for run configuration, progress and result export.
- [ ] Add sustained video/camera runs: 30 completed warm-up frames followed by a
  fixed 10-minute window. Preserve the shared queue/freshness policy and account for
  late, skipped, failed and unfinished work at the cutoff.
- [ ] Produce per-repetition, per-minute and full-run metrics with reproducible
  metadata and explicit invalid/interrupted-run reasons.

### 3. Collect native iPhone results

Use only configurations that pass parity. Follow the protocol's device conditions,
model order, cooldowns, timing boundaries and viability criteria.

- [ ] Run three short offline repetitions for each model.
- [ ] Run sustained paced video and live camera for each model.
- [ ] Record model/end-to-end FPS, latency mean/p50/p95, fresh-result rate, result
  age, skips/drops/failures, coverage, thermal state and charge change.
- [ ] Save raw predictions, timings, settings and summary metrics under
  `outputs/evaluation/ios_benchmark/<run_id>/`; retain failed or missing results
  with reasons.
- [ ] Classify each sustained configuration using both fresh-result rate and age.

### 4. Deliver and compare Unity results

- [ ] Package validated ONNX artifacts with portable manifests, hashes, class/tensor
  contracts, preprocessing/postprocessing instructions and video reference frames.
- [ ] Deliver the package to MoveAhead and record the supplied versions.
- [ ] Add MoveAhead's measurements under the same protocol, including Unity/backend/
  rendering versions and settings. Keep offline, paced-video and camera rates separate.
- [ ] Report conversion issues, bottlenecks and limitations; recommend a model and
  detection cadence for live use or offline extraction.

### 5. Complete required Apple Vision tracking

Run this after detector-only benchmarks. Tracking remains part of completion.

- [ ] Define `ObjectTracker` and implement a prerecorded-video baseline with
  initialization, detection association, updates and recovery after track loss.
- [ ] Test RF-DETR Small, RF-DETR Nano and YOLOX Nano.
- [ ] Save track IDs, source timestamps, continuity, losses/reinitializations and
  incremental tracking cost for each detector. Keep detector-only and
  detection-plus-tracking measurements separate.
- [ ] Document the tracking-runtime choice using the [licensing assessment][licensing].

## Completion criteria

- [ ] All three models have validated video outputs and reproducible native
  offline/video/camera benchmark results.
- [ ] Validated ONNX artifacts and integration instructions are delivered; native
  and Unity results are compared with unavailable results explicitly identified.
- [ ] Apple Vision tracking results are saved for all three detectors.
- [ ] The app supports automated benchmark runs, metrics and verified exports.

## Follow-up experiments

- [ ] Annotate representative held-out video/camera footage and define the skipped-
  frame evaluation policy before reporting accuracy or F1.
- [ ] Audit the source datasets' stretch-to-square preprocessing. Both basketball
  exports were resized to 640×640 before training; current RF-DETR stretches wide
  inputs and YOLOX letterboxes them. Recover original aspect ratios where possible
  and evaluate deployment-like data before changing training or input shape.
- [ ] Compare supported rectangular inputs, resolutions, compute policies and
  precision/quantization as separate validated configurations. Keep benchmark clips
  held out from training and record new artifact hashes.
- [ ] Compare ByteTrack with Apple Vision and test detection on keyframes with
  tracking between detections. Add identity switches/IDF1/HOTA when ground-truth
  track IDs exist.
- [ ] Profile compute-unit use, power, memory and sustained thermal behaviour where
  measured bottlenecks justify it. Background processing remains outside this scope.

[protocol]: ../ios/documents/protocol_ios_detection_benchmarking.md
[catalog]: ../ios/ENABLECoreMLBench/ENABLECoreMLBench/Resources/TestVideos/test_videos_manifest.json
[integration]: ../ios/documents/native_detector_integration.md
[inventory]: ios_benchmark_model_inventory.md
[coreml-diagnostics]: ../ios/documents/coreml_diagnostics_2026-11-23.md
[nano-mismatch]: ../ios/documents/native_detector_integration.md#nano-confidence-mismatch
[licensing]: 02.supervision_licensing.md
