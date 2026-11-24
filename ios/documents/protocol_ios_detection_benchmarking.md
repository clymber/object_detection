# ENABLE — iOS detection benchmarking protocol

**Version:** 0.5-draft · **Date:** 24 November 2026

Compare Native Core ML and MoveAhead Unity + iOS on the **iPhone 12 Pro**. This stage
measures accuracy, offline throughput and paced delivery using one annotated video.
Live camera and tracking follow separately.

## 1. Inputs and models

Use these two files from
`data/processed/ios_benchmark/annotated_test_video_v2/`:

- `basketball_test.mp4`: clean input, 1280×720, 30 FPS, 180 seconds, 5,400 frames.
- `basketball_test_annotations.json`: ground-truth boxes for those frames.

Use identical files on both platforms. Check the video's SHA-256 against the JSON
and record both file hashes. The boxed video is for visual inspection only.

- **Required models:** RF-DETR Small and YOLOX Nano.
- **Optional model:** RF-DETR Nano.

Use the existing exports and preprocessing/postprocessing in the
[model inventory][inventory] and [reference adapters][adapters]. Native RF-DETR
uses FP16; native YOLOX uses FP32. Record each platform's model hash and precision.

## 2. Shared settings

| Setting | Value |
| --- | --- |
| Build | Release, without debugger/profiler |
| Native compute units | `.all` |
| Model input | 640×640, batch 1, one inference in flight |
| Detection | Confidence ≥0.25, score floor 0.001, top 100 per frame |
| Preview and tracking | Off |
| Warm-up | 30 untimed predictions on the first frame; then restart at frame 0 |
| Paced target | ≥10 fresh results/s, result age ≤200 ms |

Run in the foreground with the screen unlocked and Low Power Mode off. Start with
the phone cool, allow it to cool between runs, and record thermal state and battery
level at the start and end. Keep device conditions comparable between platforms.

## 3. Run both modes

Import the video and annotations, select the model, and run **one complete pass per
mode**. Save the export before switching model or mode. Retain failed/interrupted
attempts with a reason; rerun them under a new run ID.

| Mode | Processing rule | Main measurements |
| --- | --- | --- |
| Offline | All 5,400 frames in order, without pacing or skipping | Full-video F1, detection FPS, processing time |
| Paced | One 180-second pass at source cadence; prefer recent frames when busy | Processed-frame F1, completed/fresh FPS, result age, coverage |

For paced processing, allow one active and at most one pending frame. Replace an
older pending frame with the newest; skip pending work older than 200 ms. Let active
inference finish, but publish its result only at age ≤200 ms. Account for every
source frame, including skips, failures and unfinished work.

## 4. Metrics

- **F1:** confidence ≥0.25, IoU ≥0.5, score-ordered one-to-one matching within each
  category. Aggregate TP/FP/FN, then calculate `F1 = 2TP / (2TP + FP + FN)`.
  Negative frames count. Category 1 is basketball.
- **Alignment:** use zero-based source frame indices and timestamps. Return boxes
  as `[x, y, width, height]` in the 1280×720 video canvas, origin top-left.
- **Offline accuracy:** score every annotated frame; failed inference contributes
  empty predictions. Interrupted or incomplete source passes have no final F1.
- **Paced accuracy:** score successful detections completed within the playback
  window. Exclude skipped, failed and post-window frames from F1; report their
  counts and coverage. Use offline F1 for the main accuracy comparison.
- **Offline FPS:** successful detections divided by seconds from first frame ready
  to last detection complete. Include preprocessing, inference, postprocessing and
  inter-frame decode/dispatch gaps. Also report measured processing time.
- **Paced FPS:** completions within the playback window divided by its duration.
  Fresh FPS counts results published within that window at age ≤200 ms. Coverage
  is successful completions within the window divided by 5,400.
- **Result age:** completion/publication time minus the source timestamp mapped to
  one monotonic host clock. Report completed age p95 and published age p50/p95/max.
  Exclude post-window completions/publications from paced rates.

Loading, warm-up, F1 scoring and export are outside detection timing. Successful
empty detections count toward FPS. Playback/display FPS is a separate rate.

## 5. Export and decide

Save one folder per platform/model/mode/run ID with these three files:

| File | Contents |
| --- | --- |
| `frames.jsonl` | Frame index/PTS, detections, processing/publication status and failure/skip reasons |
| `mobile_runs.json` | Settings, device/app/backend versions, model/input hashes, counts, F1, FPS, coverage and device conditions |
| `times.csv` | Per-frame timing endpoints for recalculating FPS, latency and result age |

The native app currently names the latter two files `mobile_run.json` and
`timings.csv`; accept those existing names as equivalents. Represent unavailable
metrics with a reason.

Check hashes, frame accounting and annotation alignment before comparing runs.
Summarize offline F1/FPS/processing time and paced fresh FPS/age/coverage for each
model/platform. Report full-run and per-minute paced rates to expose slowdowns.

The confirmed paced target is **≥10 fresh results/s in every full 60-second window,
completed-result age p95 ≤200 ms, and published result age ≤200 ms**. A valid run
that misses it is a candidate for offline extraction or a lower detection cadence.
Meeting it on this animated video supports a later live-camera test; natural-motion
accuracy and sustained camera performance still need separate evaluation.

[inventory]: ../../documents/ios_benchmark_model_inventory.md
[adapters]: ../../evaluation/src/detection_evaluation/onnx_inference.py
