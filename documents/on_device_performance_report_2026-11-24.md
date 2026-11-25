# ENABLE — On-device basketball detection performance

**Report date:** 24 November 2026  
**Development period:** 18–24 November 2026, inclusive  
**Platforms:** Native iOS/Core ML and Unity on iOS/ONNX  
**Models:** RF-DETR Small, RF-DETR Nano and YOLOX Nano

## 1. Findings and recommendation

- **YOLOX Nano leads the native results:** F1 **0.8525**, offline throughput
  **14.79 FPS**, and **18.58 fresh results/s** in the two-minute camera test.
  RF-DETR Small and Nano each deliver approximately **11 fresh results/s**.
- **Live use:** start with YOLOX, process the newest available frame, and evaluate
  tracking between detections. No native model processes every frame at 30 FPS.
- **Complete analysis:** use offline processing when every frame matters. YOLOX
  takes **6m 05s** to process the three-minute test video.
- **Results remain preliminary:** camera F1 and long-duration performance are
  unmeasured. Unity lacks Nano results and comparable test conditions.

## 2. Test scope and definitions

Evidence comprises six native video runs, three camera runs and MoveAhead's
Unity summary. Uploaded results:

- [Live-camera inference tests](https://drive.google.com/drive/folders/1GBKLMeQPGDJlbep53E984wpieNk4v5I7?usp=drive_link)
- [Recorded-video inference tests](https://drive.google.com/drive/folders/1AuC-oWXkxun3bybumxVfZHd8LEHCPBDw?usp=drive_link)

### Metrics

- **F1:** balances precision (correctness of detections) and recall (labelled
  objects detected); higher is better. `F1 = 2TP / (2TP + FP + FN)`.
  TP/FP/FN mean correct detections, false detections and missed objects.
- **Offline FPS:** all-frame throughput from first frame ready to last result
  complete, including preprocessing, inference, postprocessing and decode/dispatch
  gaps. Loading, warm-up, scoring and export are excluded.
- **Completed FPS:** successfully processed frames per second within the playback
  or capture window, including frames with no detections.
- **Fresh FPS:** results delivered within that window at source age **≤200 ms**.
- **Coverage:** successful in-window completions divided by input frames.
- **Latency:** processing time. **Result age:** time since the source timestamp.
  **p95:** the value at or below which 95% of measurements fall.

### Native video conditions

- **Device:** iPhone 12 Pro; recorded iOS **27.2.1 (24B156)**; Release app 1.0/build 1.
- **Configuration:** Core ML allows all compute units; hardware placement was not
  profiled. Input 640×640, one active inference; RF-DETR FP16 and YOLOX FP32.
- **Input:** 1280×720, 30 FPS, 180 seconds, **5,400 frames**; **5,460 labelled boxes**
  and **885 negative frames**. The video animates 360 test images. Correlated frames
  do not establish natural-motion accuracy or tracking quality.
- **Scoring:** confidence ≥0.25, box-overlap IoU ≥0.5, maximum 100 predictions/frame;
  confidence-ordered, same-category, one-to-one matching. Counts are pooled across
  frames, including false detections on negative frames.
- **Preparation:** 30 untimed warm-up predictions recorded in accuracy metadata;
  preview off, frames not preloaded, one run per model/mode.
- **Offline mode:** processes all frames without skipping or playback deadlines.
- **Paced mode:** follows playback timing and prefers recent frames when busy.
  Rates exclude work completed after playback. F1 covers successful in-window
  completions; use offline F1 for the common-frame accuracy comparison.

### Camera and Unity conditions

- **Camera:** approximately 120 seconds/model; rear wide lens, 1280×720 rotated
  into 720×1280 upright frames, no mirroring. Target 30 FPS; delivered rate
  29.92–29.93 FPS. Preview off, annotated MP4 recording on, no warm-up/repetitions.
- **Scene limits:** recordings show displayed basketball footage, with different
  content across models. They do not establish same-scene or in-person basketball
  performance. No ground truth exists; prediction overlays cannot substitute for it.
- **Unity:** MoveAhead reports F1 and latency for Small and YOLOX only. Device,
  inputs, model identity/precision, scoring, backend, timing boundaries and device
  conditions are unspecified. Raw predictions and timings are unavailable.

## 3. Platform comparison

| Model | Unity F1 | Native F1 | Unity FPS equivalent¹ | Native offline FPS |
| --- | ---: | ---: | ---: | ---: |
| RF-DETR Small | 0.77 | 0.8448 | 1.39 | 8.10 |
| YOLOX Nano | 0.81 | **0.8525** | 38.46 | **14.79** |
| RF-DETR Nano | Unavailable | 0.8416 | Unavailable | 7.49 |

¹ **1,000 / latency in ms:** YOLOX 26 ms; Small 721.7 ms. These are serial latency
equivalents, not measured application throughput or evidence of 30 FPS delivery.

- Native F1/FPS use the same complete annotated video. All models process
  5,400/5,400 frames without inference failures or unfinished work.
- YOLOX has the higher reported Unity F1 and lower latency. Unknown test conditions
  prevent attributing native/Unity differences to platform or conversion, or
  calculating a valid platform speedup.

### Native offline accuracy and processing time

| Model | Precision | Recall | TP / FP / FN | Processing time | Time / clip |
| --- | ---: | ---: | --- | ---: | ---: |
| RF-DETR Small | 91.35% | 78.57% | 4,290 / 406 / 1,170 | 11m 06s | 3.70× |
| RF-DETR Nano | **93.79%** | 76.32% | 4,167 / 276 / 1,293 | 12m 01s | 4.01× |
| YOLOX Nano | 90.57% | **80.53%** | 4,397 / 458 / 1,063 | **6m 05s** | **2.03×** |

- YOLOX is **1.83× faster than Small** and **1.98× faster than Nano**. Its F1 lead
  over Small is **0.0077**, or **0.77 percentage points**; repeatability is untested.
- Nano has the fewest false detections, the most misses, and no offline speed
  advantage over Small in these runs.

### Native offline latency

| Model | Model latency: mean / p95 (ms) | Frame latency: mean / p95 (ms) |
| --- | ---: | ---: |
| RF-DETR Small | 96.80 / 100.31 | 122.39 / 129.32 |
| RF-DETR Nano | 100.51 / 118.43 | 132.25 / 162.47 |
| YOLOX Nano | 42.62 / 47.74 | 66.87 / 78.41 |

Frame latency covers preprocessing, inference and postprocessing, excluding decode
and rendering. YOLOX spends **24.25 ms/frame** outside the model call within this
interval; profiling is needed to identify optimization opportunities.

## 4. Paced video: freshness and coverage

| Model | F1 on processed frames | Completed FPS | Fresh FPS | Frames scored | Coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| RF-DETR Small | 0.8484 | 9.99 | 9.99 | 1,799 | 33.31% |
| RF-DETR Nano | 0.8396 | 7.32 | 7.19 | 1,318 | 24.41% |
| YOLOX Nano | **0.8527** | **18.27** | **18.27** | **3,288** | **60.89%** |

- The queue permits one active and one waiting frame; newer input replaces waiting
  input. Skips: **3,599 Small**, **4,080 Nano**, **2,110 YOLOX**. Each run completes
  two extra frames after playback; there are no inference failures.
- Nano has 24 late results overall: **1,295 fresh deliveries** versus **1,318
  completions** within playback.
- Models score different frame subsets. Paced F1 does not measure skipped-frame
  accuracy or overall event recall.

### Performance over time

**Target:** ≥10 fresh results/s in every full 60-second window, completed-result
age p95 ≤200 ms, and all delivered-result ages ≤200 ms.

| Model | Minute 1 | Minute 2 | Minute 3 | Lowest rolling 60s rate | Rate target |
| --- | ---: | ---: | ---: | ---: | --- |
| RF-DETR Small | 11.32 | 10.23 | 8.43 | 8.43 | Missed |
| RF-DETR Nano | 9.40 | 5.72 | 6.47 | 5.65 | Missed |
| YOLOX Nano | 18.48 | 18.50 | 17.82 | **17.80** | **Met** |

Rates are fresh results/s; rolling checks include all contained 60-second windows.

| Model | Completed-result age p95 (ms) | Delivered-result age maximum (ms) |
| --- | ---: | ---: |
| RF-DETR Small | 147.12 | 163.53 |
| RF-DETR Nano | 193.03 | 199.91 |
| YOLOX Nano | **88.41** | **97.52** |

**Only YOLOX meets all observed paced targets.** Fresh rates fall from minute 1 to
3 by 25.5% for Small, 31.2% for Nano and 3.6% for YOLOX. Thermal conditions were
not recorded, so thermal throttling is unconfirmed.

## 5. Live Camera: two-minute results

| Model | Frames received | Completed in window | Fresh FPS | Coverage |
| --- | ---: | ---: | ---: | --- |
| RF-DETR Small | 3,592 | 1,316 | 10.97 | 36.64% |
| RF-DETR Nano | 3,591 | 1,322 | 11.02 | 36.81% |
| YOLOX Nano | 3,591 | **2,230** | **18.58** | **62.10%** |

Completed FPS equals fresh FPS. Coverage uses received frames; all are accounted
for, including skips. Frames never delivered by the sensor remain unverified.

### Performance over time

| Model | Minute 1 fresh FPS | Minute 2 fresh FPS | Lowest rolling 60s fresh FPS |
| --- | ---: | ---: | ---: |
| RF-DETR Small | 11.48 | 10.45 | 10.43 |
| RF-DETR Nano | 11.77 | 10.27 | 10.27 |
| YOLOX Nano | 18.47 | 18.70 | **18.47** |

| Model | Frame latency: mean / p95 (ms) | Completed age p95 (ms) | Delivered age: p95 / max (ms) |
| --- | ---: | ---: | ---: |
| RF-DETR Small | 111.24 / 134.20 | 169.08 | 169.18 / 177.53 |
| RF-DETR Nano | 110.26 / 135.00 | 169.38 | 169.47 / 182.56 |
| YOLOX Nano | **74.75 / 91.95** | **123.09** | **123.20 / 190.11** |

- **All models meet the observed two-minute rate/age targets.** Small and Nano have
  only 4.3% and 2.7% minimum rate headroom. Both slow in minute 2; YOLOX stays stable.
- Camera/video differences cannot isolate a cause: content, conditions, warm-up
  and recording load differ.

### Skips and recordings

No capture-drop events, inference failures or unfinished frames are reported.
Queue skips are separate from capture drops:

| Model | Queue skips | Total completed | Completed after cutoff |
| --- | ---: | ---: | ---: |
| RF-DETR Small | 2,275 | 1,317 | 1 |
| RF-DETR Nano | 2,267 | 1,324 | 2 |
| YOLOX Nano | 1,359 | 2,232 | 2 |

- Nano has one late delivery overall. Completion and delivery use separate event
  timestamps, so their in-window counts can still be equal.
- MP4s contain successful processed frames, including late/post-cutoff results,
  with original capture gaps. Counts match total completions. Playback FPS is not
  fresh FPS, and omitted frames prevent full-input camera accuracy evaluation.
