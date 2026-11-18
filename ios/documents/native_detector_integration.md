# Native detector integration

The shared model layer is implemented under
`ENABLECoreMLBench/ENABLECoreMLBench/Detection/`. SwiftUI session wiring and
still-image overlays are the next plan tasks. Model loading, preprocessing and
decoding stay on worker actors.

```swift
let engine = DetectorEngine(modelURLs: artifactURLs)
try await engine.selectModel(.rfdetrSmall)
let detections = try await engine.detect(in: uprightImage)
```

`artifactURLs` maps `DetectorModel` values to existing `.mlpackage` or `.mlmodelc`
URLs. Keep reusable packages below `models/exports/`; the loader compiles explicit
packages before loading. For app-bundled models, omit the URL mapping and provide
distinct resources named `rfdetr_small`, `rfdetr_nano` and `yolox_nano` (compiled
models or packages), at the bundle root or in `Models/`. Missing resources and
incompatible tensor contracts throw errors. This layer does not download models.

The first-device resources are staged for **RF-DETR Small**:

```text
ENABLECoreMLBench/ENABLECoreMLBench/
  Models/rfdetr_small.mlpackage/
  Resources/TestImages/positive_5731.jpg
  Resources/TestImages/negative_5812.jpg
  Resources/TestImages/sample_manifest.json
```

The package's four files and both JPEGs are hardlinks to their original export
and dataset files. Treat staged resources as read-only: editing a hardlinked file
also changes its source. Directories are created separately; originals remain
under `models/exports/` and `data/`. The package and JPEGs are ignored by Git.
The sample manifest records hashes, source IDs, dimensions, ground truth and
expected checkpoint/Mac Core ML detections at confidence 0.25. The positive image
has one expected basketball detection; the negative image has none.

Recreate or verify these links from the workspace root with:

```bash
conda run -p .conda/envs/object-detection-evaluation \
  python ios/ENABLECoreMLBench/stage_rfdetr_small_resources.py
```

The helper refuses to overwrite conflicting destination files and verifies the
source checkpoint, package and reference-record hashes. The existing Xcode
folder membership includes these resources in the `ENABLECoreMLBench` target;
Xcode compiles `rfdetr_small.mlpackage` and copies the two JPEGs and sample
manifest into the app bundle. Build and run the app from Xcode to install those
resources on the iPhone together with the app.

Verified on 18 November 2026 with an unsigned generic-iOS Debug build: the app
bundle contains `rfdetr_small.mlmodelc`, both original-byte JPEGs and the sample
manifest. Source and staged files still share their verified inodes and hashes.
This packaging check does not establish on-device inference or rendered boxes.

All selected artifacts take FP32 `[1,3,640,640]` tensors. RF-DETR compute/outputs
are FP16; the selected YOLOX artifact is FP32. Output names and shapes are checked
against the selected export contracts. Default compute units are `.all`; CPU-only
host checks use a separate configuration.

Callers supply decoded, upright RGB images; resolve photo/video orientation
before calling the detector. Returned detections use unclipped top-left source
pixel boxes and source category IDs (`basketball` is 1). Model adapters retain
candidates down to score 0.001; the parity/display selection is confidence
at least 0.25 with the stable top-100 cap. Successful empty images return `[]`.

RF-DETR matches the Python half-pixel resize and ImageNet normalization. YOLOX
uses portable OpenCV-style uint8 bilinear resize, BGR top-left letterboxing and
padding 114. The installed OpenCV 5 ARM downscale kernel differs from the portable
OpenCV 4.11 path by up to two byte levels in the downscale fixture. Preserve this
as a preprocessing limitation when testing new source sizes; it does not relax
the box/score parity tolerances.

Model switches commit only after successful loading; failed switches preserve
the previous working model. Newer selection requests supersede older pending
loads and in-flight detection results. Cancellation is checked before/after
Core ML inference; it does not interrupt a synchronous prediction already running.

Run fast semantic checks and iOS arm64 type-checking from the workspace root:

```bash
bash ios/ENABLECoreMLBench/tests/run_detection_checks.sh
```

To regenerate ignored fixtures and run the actual local exports against saved
checkpoint predictions for all 25 development parity images per model:

```bash
bash ios/ENABLECoreMLBench/tests/run_detection_checks.sh --with-models
```

These are Mac checks. They do not establish iPhone inference, frozen mobile-set
parity, dataset F1 or device performance. Fixtures and build products live below
`outputs/ios_builds/detection_pipeline/`.

Implementation references: [Core ML model descriptions][model-description],
[multiarray storage][multiarray], and [OpenCV resize arithmetic][opencv-resize].

[model-description]: https://developer.apple.com/documentation/coreml/mlmodeldescription
[multiarray]: https://developer.apple.com/documentation/coreml/mlmultiarray
[opencv-resize]: https://github.com/opencv/opencv/blob/4.11.0/modules/imgproc/src/resize.cpp
