# ENABLE — Shared iOS detection benchmarking protocol

- **Version:** 0.1-draft · **Date:** 16 November 2026
- **Platforms:** Native iOS + Core ML; MoveAhead Unity + iOS
- **Device:** iPhone 12 Pro · **Models:** RF-DETR Small/Nano, YOLOX Nano

Compare detector accuracy and speed using the same inputs, checkpoints and rules.
These settings are proposed defaults. Freeze the manifests and protocol version
before measurement; changes to inputs, artifacts or rules require affected reruns.

| Owner | Responsibility |
| --- | --- |
| ENABLE benchmark owner | Freeze inputs/settings and evaluate accuracy centrally. |
| MoveAhead | Implement the Unity runner and return the agreed artifacts. |
| Native iOS owner | Implement the Core ML runner and return the same artifacts. |

## 1. Shared settings

| Setting | Decision |
| --- | --- |
| Accuracy | Micro F1; IoU **≥ 0.50**, confidence **≥ 0.25**. |
| Candidates | Score floor **0.001**; final top **100** per image. |
| Model input | **640 × 640**, batch **1**, one inference in flight. |
| Short timing | **30** warm-up frames, **300** timed frames, **3** repetitions. |
| Sustained timing | **10 minutes**, primary video looped at source cadence. |
| Freshness target | **10 fresh results/s**, maximum published age **200 ms**. |
| Baseline | **Release**, preview/overlays off, detector only. |

Accuracy uses every test image; timing uses a separate fixed sequence. Tracking,
live-camera and preview-on measurements have separate configurations and run IDs.

## 2. Inputs

Use the existing [COCO dataset][images] and [basketball videos][videos]. Paths are
relative to the checkout. Image root: `data/composed/basketball-v0.1/coco/`.
Resolve images as `images/<split>/<file_name>` using the COCO image ID and name;
annotations are `annotations/instances_<split>.json` below that root.

| Split | Images | Use |
| --- | ---: | --- |
| `train` | 5,730 | Training provenance only. |
| `val` | 1,257 | Adapter development and parity; includes 180 negative images. |
| `test` | 867 | Complete F1 evaluation; 725 positive and 142 negative images. |

- Keep split membership, source dimensions and COCO coordinates fixed. Check
  source-recording leakage before freezing; use validation for threshold exceptions.
- Select **25 validation parity images**: 20 positive and 5 negative, covering
  small objects, overlaps and empty outputs; record unavailable cases.
- Freeze **300 ordered test image IDs**, including positives and negatives.
  Repeat IDs if needed for a shared preload memory budget; warm up on the first 30.
- All referenced images were present and annotations had no crowd/ignore boxes
  when checked on 16 November 2026. Recheck changed inputs; the F1 evaluator
  rejects crowd/ignore annotations. Keep zero-annotation images in coverage.

Video root: `data/sources/downloads/basketball_videos/`.

| Role | Filename | Source size / nominal FPS |
| --- | --- | --- |
| Primary | `basketball-nba_clip.mp4` | 1280 × 720 / 60 |
| Additional | `basketball-uhd_4096_2160_30fps.mp4` | 4096 × 2160 / 30 |
| Additional | `basketball_dribbling.webm` | 960 × 540 / 29.97 |

- Validate decoding on both devices. If a compatible rendition is needed, share
  one frozen file, preserving source downloads; save derivatives below
  `data/processed/ios_benchmark/videos/`. Record source/rendition hashes and any
  dimension, cadence or timestamp mapping changes. Use identical bytes on both apps.
- Use actual presentation timestamps (PTS) and time bases, not nominal FPS alone.
  These clips have no accompanying frame annotations: use them for performance,
  coverage and tracking observations. Video F1 remains pending annotations and
  the skipped-frame policy; use COCO images for the main accuracy result.

Create `input_manifest.json` with dataset identity/fingerprint, annotation hashes,
ordered image IDs, per-image paths/hashes/dimensions/splits, category maps, parity
and timing selections. For video, include source/rendition paths and hashes, codec,
orientation, dimensions, duration, cadence, PTS/time base, frame indices and
loop policy.

## 3. Models and artifacts

Use these fixed runs under [`outputs/runs/basketball/`][runs]; keep their artifacts
unchanged. All three are completed fine-tuned runs with `smoke_run: false`.

| Model | Run directory |
| --- | --- |
| RF-DETR Small | `rfdetr_small_20261106T124419` |
| RF-DETR Nano | `rfdetr_nano_20261116T115819` |
| YOLOX Nano | `yolox_nano_20261105T200341` |

Paths below are relative to each run:

- **RF-DETR:** `checkpoint_best_total.pth`, `weights/checkpoint_best_total.onnx`;
  references `val_predictions.json` / `test_predictions.json`. Retain
  `best_checkpoint.json`, its selected `checkpoint_best_ema.pth`, `run_config.json`,
  `training_config.json`, `resolved_config.json`, `dataset_manifest.json` and metrics.
  Use the owning RF-DETR loader to preserve EMA selection and model configuration.
- **YOLOX:** `weights/best_ckpt.pth`, `weights/best_ckpt.onnx`;
  references `evaluation/val_predictions.json` / `evaluation/test_predictions.json`;
  configuration `args.yaml` with NMS IoU **0.65**.
- **All runs:** retain `run_protocol.json`, `dataset_identity.json`, `training.json`.
  Resolve historical host paths using these run-relative paths. References share
  canonical dataset identity and annotation hashes for 1,257 val / 867 test images.

Map foreground index **0** to COCO category **1**, `basketball`. Existing ONNX
inputs/outputs/weights are **FP32**, input shape `[1, 3, 640, 640]`; all three
passed structural checks. Apply the [Python ONNX adapters][adapters] consistently:

| Family | ONNX input | Outputs | Opset |
| --- | --- | --- | ---: |
| RF-DETR | `input` | `dets [1,300,4]`, `labels [1,300,2]` | 17 |
| YOLOX | `images` | `output [1,8400,6]` | 11 |

- **RF-DETR:** RGB square resize, ImageNet normalization; normalized centre boxes.
  `labels` holds logits: sigmoid foreground, exclude background; `num_select: 300`,
  no NMS. Then apply the shared top-100 policy.
- **YOLOX:** BGR top-left letterbox, padding **114**, unnormalized 0–255 floats;
  decode strides **8/16/32**, score = objectness × class probability; NMS **0.65**.
- Match interpolation, rounding and inverse transforms; apply embedded operations
  once. Export clipped, positive-area floating COCO `[x,y,width,height]` boxes in
  top-left-origin source pixels. Preserve scores and stable candidate-index tie order.

Build `model_manifest.json` from existing records: artifact/checkpoint hashes,
export command/revision, runtime, precision, tensor contracts, transforms and parity
samples. Native Core ML conversion remains pending: use the same checkpoints,
initially target FP16, and record package hashes/actual precision. Hash a sorted
per-file checksum manifest for directory packages. Keep new models below `models/`;
changed precision/artifacts are separate configurations. Collect fresh mobile results.

## 4. Conversion parity and accuracy

Run the 25 parity images through the checkpoint, Python ONNX and each mobile
runner. Validate exports against the checkpoint, then mobile outputs against the
reference, using final source-coordinate boxes at confidence 0.25 and the shared cap.

| Check | Acceptance |
| --- | --- |
| Count/classes | Same retained count and category multiset, including empty results. |
| Correspondence | Reference score order; unmatched same-class box with highest IoU. |
| Box corners | Each x/y difference ≤ `max(2 px, 0.002 × source width/height)`. |
| Confidence | Absolute difference ≤ **0.02**; finite scores in `[0,1]`. |

Use stable output order for ties. Threshold/NMS crossings fail parity; save per-image
predictions, differences and status. Resolve failures before benchmarking that
model/platform. Tolerance changes require a new protocol version and parity reruns;
one platform's conversion blocker does not block another validated platform.

For F1, process **all 867 test images once**, with no deadlines, frame dropping,
tracking or reused predictions. An inference/read error makes the run incomplete.
Evaluate centrally using [`evaluate_predictions`][metrics] with
`score_floor=0.001`, `confidence_threshold=0.25`, `iou_threshold=0.50`,
`max_dets=(1,10,100)`:

1. Keep top 100 per image, then scores ≥ 0.25; visit predictions in score order.
2. Match each to the unmatched same-category ground truth with maximum IoU.
   IoU ≥ 0.50 gives TP; unmatched/duplicate predictions give FP; unmatched GT gives FN.
3. Aggregate TP/FP/FN across the split, including negatives, before calculating:
   `precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2*TP/(2*TP+FP+FN)`.

Zero denominators return zero. Report TP/FP/FN, precision, recall, F1 (0–1, four
decimals), image/negative counts and FP per negative image. Threshold exceptions
require validation-only evidence and a separately labelled frozen configuration.

## 5. Timing definitions

Use a monotonic host clock in milliseconds; synchronize backend completion and
required output readback. Run model-only and end-to-end modes separately.

| Time | Boundary |
| --- | --- |
| `t_ready` | Decoded source frame offered to runner, before queuing. |
| `t_submit` | Input prepared; backend submission/upload begins. |
| `t_model_done` | Backend execution and required output readback complete. |
| `t_done` | Final CPU detection list in source coordinates ready. |
| `t_due` | Scheduled video PTS mapped to the host playback clock. |
| `t_publish` | Result delivered to consumer, including with preview off. |

- **Model latency:** `t_model_done - t_submit`; include scheduling/transfers.
  Model FPS = `1000 × N / sum(model_latency_ms)` for serial execution.
- **End-to-end latency:** `t_done - t_ready`; include queueing, preprocessing,
  inference, readback, decoding/NMS and coordinate restoration.
  End-to-end FPS = `1000 × N / (last t_done - first t_ready)`; count completed
  frames, including empty detections, and include request-dispatch gaps.
- **Video age:** completed = `t_done - t_due`; published = `t_publish - t_due`.
  Use the same playback clock for all endpoints; age includes decode delays.

Preload decoded inputs before warm-up. Offline timing excludes reads/decode, model
load/compilation, warm-up, drawing and disk export; no file I/O between timed requests
or playback pacing. Record graph-embedded preprocessing/NMS if model-only scopes
vary; use end-to-end FPS for the main comparison. Display FPS is a separate metric.

## 6. Device conditions and short timing runs

- Prefer the same physical iPhone 12 Pro; record device/OS/build and app revision.
  Use optimized Release builds, no debugger/deep profiler; record toolchain versions.
- Native compute units: `.all`. Record Unity/backend versions, execution policy,
  precision, CPU fallback and observed operator placement where available.
- Keep inference independent of UI/display. Record Unity render pipeline, scene/update
  scheduling, target FPS, vSync and GC settings; include engine scheduling in timing.
- Battery power, initial charge **50–80%**, Low Power Mode off, brightness **50%**,
  case removed, stable ambient temperature; record temperature and radio/OS settings.
- Foreground/unlocked; preview, overlays and recording off. Buffer logs until timing
  ends. Start each repetition at nominal thermal state; cool at least five minutes
  between configurations/repetitions and until nominal. Record thermal changes.
- Per mode/model/platform: **30 warm-up + 300 timed frames**, **3 repetitions**,
  one request in flight, no batching or cached predictions. Rotate order:
  Small/Nano/YOLOX, Nano/YOLOX/Small, YOLOX/Small/Nano; alternate platform order.
- Keep failed/aborted logs; rerun interruptions with new IDs. Report every repetition,
  median FPS with min–max, and latency mean/p50/p95. Use nearest-rank percentile
  `ceil(p × N)`; do not average instantaneous FPS or select only the fastest run.

## 7. Sustained video and skipped frames

Loop the **NBA primary clip** at source cadence for **10 minutes after warm-up**,
keeping a continuous playback clock and loop index. Run once per model/platform;
additional clips use separate configurations. Keep late/thermally degraded periods.

- Use **one active + at most one pending frame**. Replace older pending work with
  the newest frame (`skipped_superseded`); skip pending frames aged over 200 ms
  (`skipped_stale`). Let active inference finish; publish only if age at delivery
  is **≤ 200 ms**. Fresh empty detection lists count as results.
- Log every scheduled frame: clip/loop/index, PTS, timing endpoints, detections,
  inference status, publication status and skip/drop/error reason. Reconcile
  scheduled frames to terminal statuses; published/late are outcomes of completion.
- Stop at 600 seconds; classify unfinished work and exclude post-window completions.
  Report incomplete coverage if decode losses cannot be reconstructed.
- For the full run and each 60-second window, report input/completed/fresh FPS,
  latency mean/p50/p95, age p50/p95/max, coverage, skip/drop/late/error counts and
  fractions, thermal state and charge change. Count by completion/publication time;
  use full 60/600-second denominators. Record display FPS only when enabled.

| Speed label | Rule |
| --- | --- |
| Viable | Each 60 s: ≥10 fresh FPS; completed p95 age ≤200 ms; published age ≤200 ms. |
| Borderline | Valid run averages ≥5 fresh FPS but fails the viable rule. |
| Too slow | Valid run averages <5 fresh FPS. |
| Not established | Missing/invalid run, unbounded queue or crash. |

**Skipped-frame video F1 is deferred.** Preserve logs; do not assign empty boxes,
carry detections forward or choose its denominator yet. The main F1 uses complete
COCO test inputs. Report speed labels alongside accuracy.

## 8. Results and handoff

Use unique run IDs (UTC start, platform/model/mode/repetition). Save runs below
`outputs/evaluation/ios_benchmark/<run_id>/`. Return one package per configuration;
shared manifests/parity evidence may be referenced by path and hash.

| File | Contents / applicable run |
| --- | --- |
| `predictions.json` | Complete offline accuracy; existing schema-v1 artifact. |
| `frames.jsonl` | Coverage, statuses, detections, timestamps and skips. |
| `timings.csv` | Raw per-frame times, units and input identity for timing runs. |
| `mobile_run.json` | Versioned metadata, settings, metrics and validity, each run. |
| `parity_report.json` | References, tolerances and per-image outcomes for parity. |
| Manifests | Frozen input/model records and hashes of returned files. |

Follow the [prediction artifact contract][metrics]: `schema_version: 1`, exact
`annotation_sha256`, sorted unique `image_ids`, and COCO prediction rows with
`image_id`, `category_id`, `bbox`, `score`. Empty images have no prediction rows.
Metadata requires `model`, `split: test`, `run_dir`, `checkpoint`, `checkpoint_sha256`,
`resolution: 640`, boolean `smoke_run` and `postprocessing.score_floor: 0.001`.
Verify actual coverage from frame logs; the artifact writer cannot establish it.
Keep mobile timings outside `metadata.benchmark`, reserved for the Python protocol.

`mobile_run.json` is a new schema-v1 handoff record. Include protocol/run IDs, mode,
UTC, app revision, toolchain/backend/device/OS versions, input/checkpoint/export hashes,
shape/precision/compute policy, thresholds/caps/NMS, timing/preview settings,
counts/order/conditions, thermal observations, metrics, validity and failure reasons.
Use `null` with a reason for unavailable values. Validate centrally in the evaluation
Conda environment; retain raw records for recalculation.

- Report one row per model/platform/configuration: size/precision, parity, F1,
  median end-to-end FPS (range), model FPS, latency p95, sustained fresh FPS,
  skip/late fractions, speed label, run IDs and protocol version.
- In the plan table, F1 = complete-test micro F1; FPS = preview-off offline
  end-to-end throughput. Show sustained video rates separately. Flag device/backend/
  precision differences; investigate absolute mobile/reference F1 differences >0.01.
- Mark missing results blocked/not tested/invalid/pending with reasons. Complete
  tracking after detection: test Apple Vision with all three detectors, separately
  recording cadence, association/loss rules, IDs, continuity and incremental overhead.
  ByteTrack and identity metrics remain follow-up work; identity metrics need GT IDs.

Before collection, freeze manifests/version/contacts, verify hashes and parity,
and validate a coverage/export round trip. Then collect complete mobile accuracy,
three short timing repetitions per mode and the sustained baseline; return raw logs.

## 9. Frozen file hashes

SHA256 values verified on 16 November 2026. RF Small/Nano refer to the RF-DETR runs
above; YOLOX refers to YOLOX Nano. Copy these into the handoff manifests.

```text
Validation COCO      c7e78acea3f87a068d10ba4ff6d11e4a20cbf369435cd1e4433b9af4c919042f
Test COCO            c3dab2d03660d191bee4ef2ebc8da22e00a9f146c4137bae847b6cf7d44a3215
NBA video            718a363e60769b0baf0ec48f7c57c8c258076cdbaec0ef7031c34f871b3ee40e
UHD video            4f8c3554a5dd01ab12c4b98cb6f18205e13b9a47203e80a57086fdad0f633bcf
Dribbling video      1a592d92d8359729b679067977b295526fa376a9b15f92d3d7be3671941aa023
RF Small checkpoint  85fc2abfa9fd2aa0964c7b355900711aae8f0c6185776f63bfefc597bd287c64
RF Small ONNX        3254af0ccc47c588cb1838f365eea7d4f30084c02d8b2be50cfe2730374a2890
RF Nano checkpoint   ab9e434ea8d92d261be2fd3f028cf2de7004cdcb7237d4892a5c875ae10f4563
RF Nano ONNX         8039da7c2911f574c4e3ab5abc02bff53dd2fd4387c20bd3a078214671d83ad6
YOLOX checkpoint     d9b9d32ee716c90480bd1bb1a40304c6ee0fed064d6d996d2876ea5a470e99d3
YOLOX ONNX           1af2fd17c337ab43069856007273471e72da745a4586fc63dc66a03961ffe5af
```

[images]: ../../data/composed/basketball-v0.1/coco/
[videos]: ../../data/sources/downloads/basketball_videos/
[runs]: ../../outputs/runs/basketball/
[adapters]: ../../evaluation/src/detection_evaluation/onnx_inference.py
[metrics]: ../../evaluation/src/detection_evaluation/metrics.py
