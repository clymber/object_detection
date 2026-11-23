# ENABLE — iOS detection benchmarking protocol

- **Version:** 0.4-draft · **Date:** 23 November 2026
- **Device:** iPhone 12 Pro · **Platforms:** Native Core ML and MoveAhead Unity + iOS

Compare RF-DETR Small, RF-DETR Nano and YOLOX Nano using the same inputs and rules.
ENABLE freezes the settings and validates results; each platform owner runs the
tests and returns raw logs. Changes to inputs, artifacts or rules require rerunning
affected comparisons. Keep video, camera, tracking and preview-on results separate.

## 1. Shared settings

| Setting | Value |
| --- | --- |
| Model input | 640 × 640, batch 1, one inference in flight |
| Detection | Confidence ≥0.25, score floor 0.001, top 100 per frame |
| Baseline | Release, native compute units `.all`, detector only, preview off |
| Short timing | 30 warm-up + 300 timed frames, three repetitions |
| Sustained timing | 10 minutes per source/model/platform after 30 warm-up frames |
| Freshness target | ≥10 fresh results/s, published age ≤200 ms |

## 2. Inputs

### Recorded videos

| Role | File | Size / FPS |
| --- | --- | --- |
| Primary | `basketball_dribbling_h264.mp4` | 960 × 540 / ≈29.97 |
| Additional | `basketball-nba_clip.mp4` | 1280 × 720 / 60 |
| Additional | `basketball-uhd_4096_2160_30fps.mp4` | 4096 × 2160 / 30 |

The primary clip is in `data/processed/ios_benchmark/videos/`; originals are in
`data/sources/downloads/basketball_videos/`. The [video catalog][catalog] records
file hashes, conversion details and all presentation timestamps (PTS).

Both platforms must use identical video bytes, ordered parity/warm-up/timed frame
selections and preload memory budgets. Freeze these in `input_manifest.json`, with
clip metadata, frame indices, PTS/time bases, loop policy and category mapping.
Preserve originals and identify new renditions separately. Pace using actual PTS.

### Live camera

Target **30 FPS** and log the actual rate. Use the same phone/lens, resolution,
pixel format, orientation and mirroring for each model/platform. Record these in
`input_manifest.json`, with exposure/focus/stabilization settings, automatic
adjustments and scene conditions: camera position, distance, lighting and activity.
Map capture timestamps to the host clock. Camera runs have different source frames
between attempts and are reported separately from video.

## 3. Models and outputs

Use the selected checkpoints and exports from these [runs][runs]:

| Model | Run directory | Core ML precision |
| --- | --- | --- |
| RF-DETR Small | `rfdetr_small_20261106T124419` | FP16 |
| RF-DETR Nano | `rfdetr_nano_20261116T115819` | FP16 |
| YOLOX Nano | `yolox_nano_20261105T200341` | FP32 |

Keep artifact/configuration records and use the owning loaders to preserve EMA
selection. The [model inventory][inventory] gives checkpoint paths and tensor
contracts; ONNX inputs/outputs/weights are FP32, with input shape `[1,3,640,640]`.

Match the [reference adapters][adapters], including interpolation and rounding:

- **RF-DETR:** RGB square resize, ImageNet normalization, normalized centre boxes;
  sigmoid logits, top 300 class slots, background removal, then the shared top 100.
  No NMS.
- **YOLOX:** BGR top-left letterbox, padding 114, unnormalized 0–255 floats;
  strides 8/16/32, score = objectness × class probability, class-aware NMS IoU 0.65.
- Map foreground index 0 to category 1 (`basketball`). Export unrounded scores and
  unclipped floating `[x,y,width,height]` boxes in top-left source-frame pixels.
  Preserve candidate-index tie order; apply embedded operations once. Clip only
  for preview.

Freeze `model_manifest.json` with checkpoint/export hashes, export command/revision,
runtime, precision, tensor contracts, transforms and reference predictions. Hash
directory packages using a sorted per-file checksum manifest. Keep reusable models
under `models/`; changed precision or artifact identity is a separate configuration.

## 4. Validation

### Parity

Before timing, run the frozen frames through the checkpoint, Python ONNX and each
mobile runner without deadlines or skipping. Use identical decoded pixels; record
pixel hashes, dimensions, orientation and colour conversion to identify decoder
differences. Compare final detections at confidence 0.25, capped at 100.

| Check | Pass rule |
| --- | --- |
| Count/categories | Same count per category, including empty results |
| Matching | Reference score order; highest-IoU unmatched box of the same category |
| Box corners | Each axis difference ≤ `max(2 px, 0.002 × source axis size)` |
| Confidence | Difference ≤0.02; finite scores in `[0,1]` |

Use stable tie order. Include threshold/NMS boundary cases; crossings fail parity.
Save per-frame differences and resolve failures before timing that model/platform.
Tolerance changes require a new protocol version and parity reruns; other validated
configurations can proceed. Verify hashes and a coverage/export round trip as well.

F1 requires ground truth and a frozen evaluation/coverage policy. The current clips
are unannotated; parity and tracking observations do not establish accuracy.

## 5. Timing

Use one monotonic host clock in milliseconds and record the source-clock mapping.
Synchronize backend completion/readback. Run model-only and end-to-end modes
separately.

| Time | Meaning |
| --- | --- |
| `t_ready` | Frame available, before queueing |
| `t_submit` | Input prepared; backend submission/upload begins |
| `t_model_done` | Backend execution and required output readback complete |
| `t_done` | Final CPU detections ready in source coordinates |
| `t_due` | Source PTS or capture timestamp mapped to the host clock |
| `t_publish` | Result delivered, including when preview is off |

- **Model latency:** `t_model_done - t_submit`, including scheduling/transfers.
  Model FPS = `1000 × N / sum(model_latency_ms)`.
- **End-to-end latency:** `t_done - t_ready`, including queueing, preprocessing,
  inference, readback, decoding/NMS and coordinate restoration.
  End-to-end FPS = `1000 × N / (last t_done - first t_ready)`, including dispatch
  gaps. `N` counts successful completions, including empty detections.
- **Result age:** completed = `t_done - t_due`; published = `t_publish - t_due`.
  This includes video decode delays or camera capture delivery delays.

Record embedded preprocessing/NMS where model-only scopes differ. Display FPS is
separate from processing and fresh-result rates.

## 6. Run the tests

### Device conditions

- Use Release without a debugger/deep profiler; keep inference independent of UI.
- Run foreground/unlocked on battery, initially 50–80% charged, Low Power Mode off,
  brightness 50%, case removed and stable ambient temperature. Record radio settings.
- Start each repetition at nominal thermal state. Cool at least five minutes between
  configurations/repetitions and until nominal; record temperature and thermal/charge
  changes throughout testing.
- Disable preview, overlays and recording; buffer logs until timing ends.
- Record app/toolchain/device/OS versions, backend, precision, compute policy and
  placement/fallback where available. For Unity, include render/update scheduling,
  target FPS, vSync and GC settings; include engine scheduling in timing.

### Short offline timing

Preload the frozen decoded frames and load the model before warm-up. Run the shared
30 warm-up + 300 timed frames three times, without pacing or cached predictions.
Exclude reads/decode, loading/compilation, warm-up, drawing and export; perform no
file I/O between timed requests.

Rotate model order: Small/Nano/YOLOX, Nano/YOLOX/Small, YOLOX/Small/Nano; alternate
platform order. Report every repetition, median FPS with min–max and latency
mean/p50/p95. Use nearest-rank percentiles, `ceil(p × N)`; do not average instantaneous
FPS. Keep failed/aborted logs; rerun interruptions with new IDs.

### Sustained video and camera

After 30 completed warm-up frames, run each source once per model/platform for
600 seconds. Loop the primary video at actual source cadence with a continuous
playback clock and loop index. Keep camera capture running at the agreed settings.

- Allow **one active + at most one pending frame**. Replace older pending frames
  with the newest (`skipped_superseded`); skip waiting frames older than 200 ms
  (`skipped_stale`). Let active inference finish; publish only at age ≤200 ms.
- Record every scheduled video frame/delivered camera frame and its terminal status.
  Keep completion and publication outcomes separate. Count successful empty results
  normally and each fresh source frame once; distinguish skips, capture/decode drops
  and inference failures. Record capture-drop callbacks separately.
- Stop at 600 seconds, classify unfinished work and exclude post-window completions.
  Mark coverage incomplete when losses cannot be reconstructed. Retain late results
  and thermally degraded periods.
- For each 60-second window and the full run, report input/completed/fresh rates,
  latency mean/p50/p95, age p50/p95/max, coverage, skip/drop/late/error counts and
  fractions, thermal state and charge change. Count by completion/publication time
  using full 60/600-second denominators; report display FPS only when enabled.

| Rating | Rule |
| --- | --- |
| Viable | Every 60 s: ≥10 fresh FPS, completed p95 age ≤200 ms, published age ≤200 ms |
| Borderline | Valid run averages ≥5 fresh FPS but misses the viable criteria |
| Too slow | Valid run averages <5 fresh FPS |
| Not established | Missing/invalid run, unbounded queue or crash |

## 7. Results and handoff

Use unique IDs containing UTC start, platform, model, mode and repetition. Save each
configuration under `outputs/evaluation/ios_benchmark/<run_id>/`:

| File | Required contents |
| --- | --- |
| `frames.jsonl` | Frame identities, timestamps, detections, statuses and reasons |
| `timings.csv` | Raw timing endpoints and frame identities |
| `mobile_run.json` | Schema-v1 metadata, settings, counts, metrics and validity |
| `parity_report.json` | Reference frames, tolerances, differences and outcomes |
| Input/model manifests | Frozen settings, provenance and artifact/file hashes |

Frame records need source/loop/index, PTS/capture time, timing endpoints, inference
and publication status, and detections with `category_id`, `bbox`, `score`. Empty
successes use `[]`; failures/skips need explicit reasons. Reconcile coverage.

Run metadata must include protocol/version, app/toolchain/backend/device/OS, UTC,
input/checkpoint/export hashes, model size/shape/precision, compute and detection
settings, mode/preview, run order, conditions and failure reasons. Use `null` with
a reason for unavailable values; include hashes of returned files. Validate exports
in the evaluation environment and retain raw records for recalculation.

Summarize each model/platform/source/configuration with parity, model/end-to-end
FPS, latency p95, fresh-result rate, age, skip/late fractions, rating and run IDs.
Flag device/backend/precision differences; mark missing results as blocked, not
tested, invalid or pending, with reasons.

After detection benchmarks, test Apple Vision tracking with all three detectors.
Record cadence, association/recovery rules, track IDs, continuity, losses and added
cost separately. ByteTrack is follow-up work; identity metrics require ground-truth
IDs.

## 8. Frozen file hashes

Use these SHA-256 values in the handoff manifests.

```text
NBA video            718a363e60769b0baf0ec48f7c57c8c258076cdbaec0ef7031c34f871b3ee40e
UHD video            4f8c3554a5dd01ab12c4b98cb6f18205e13b9a47203e80a57086fdad0f633bcf
Dribbling video      1a592d92d8359729b679067977b295526fa376a9b15f92d3d7be3671941aa023
Dribbling H264 MP4   11ca01f291ca346a65a5e4552ee88525f5ca6a07f43b7502fbd8a73fd3045184
RF Small checkpoint  85fc2abfa9fd2aa0964c7b355900711aae8f0c6185776f63bfefc597bd287c64
RF Small ONNX        3254af0ccc47c588cb1838f365eea7d4f30084c02d8b2be50cfe2730374a2890
RF Nano checkpoint   ab9e434ea8d92d261be2fd3f028cf2de7004cdcb7237d4892a5c875ae10f4563
RF Nano ONNX         8039da7c2911f574c4e3ab5abc02bff53dd2fd4387c20bd3a078214671d83ad6
YOLOX checkpoint     d9b9d32ee716c90480bd1bb1a40304c6ee0fed064d6d996d2876ea5a470e99d3
YOLOX ONNX           b8ef9e3a7b491608e4482b81a8bcac65cbd22e48da96e1654b7cffeac3e8152f
```

[catalog]: ../ENABLECoreMLBench/ENABLECoreMLBench/Resources/TestVideos/test_videos_manifest.json
[runs]: ../../outputs/runs/basketball/
[inventory]: ../../documents/ios_benchmark_model_inventory.md
[adapters]: ../../evaluation/src/detection_evaluation/onnx_inference.py
