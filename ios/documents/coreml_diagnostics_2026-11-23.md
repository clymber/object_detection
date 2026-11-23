# RF-DETR Small Core ML diagnostics — 23 November 2026

The physical-iPhone diagnostic completed for `all`, `cpuAndGPU` and `cpuOnly`.
All three passed the 25-image checkpoint/ONNX/Mac Core ML comparisons and every
repeated prediction. Keep `all` as the agreed benchmark baseline. These results
do not close frozen-video parity or threshold-boundary validation. The subsequent
Xcode reports and CPU/GPU retest are recorded below.

**Xcode follow-up:** The user subsequently confirmed that the fusion messages
still recur in the Xcode console while interactive detection completes in a
reported **110–150 ms**. The supplied screenshots show **133.4 ms**, image **6988**,
RF-DETR Small, one basketball at **96.5%**, and the **ENABLECoreMLBench Release**
scheme with an iPhone target. The warning recurrence is confirmed; its performance
impact remains unconfirmed.

**Performance-report follow-up:** Xcode reports succeeded for `all`, CPU-only and
CPU/Neural Engine. CPU/GPU report creation failed while building an execution plan
with error `-5`. A fresh app diagnostic with `.cpuAndGPU` subsequently succeeded
again, including numerical parity. This establishes a discrepancy between the
report and app paths; it does not establish general GPU incompatibility.

## Configuration and evidence

- Device: iPhone 12 Pro (`iPhone13,3`), physical iOS 27.2.1 (`24B156`).
- Build: Release, Xcode 27.2 (`27B52`), Swift 6.4.2, iPhoneOS 27.2 SDK.
  Launched using `devicectl`, without an attached debugger.
- Model: selected RF-DETR Small FP16 package, unchanged from the staged manifest.
  Package SHA-256:
  `7a705b0cfb132a52ecdfaeaea9e35e1fdc9195ebd8dd27fe4a48bd71605de1d3`.
- Checkpoint SHA-256:
  `85fc2abfa9fd2aa0964c7b355900711aae8f0c6185776f63bfefc597bd287c64`.
- Git HEAD: `a2511b09e9c360dcc0468ca17a2dd1123c33c803`, with local changes.
  The exact app sources/manifests and signed executable are identified in
  `outputs/ios_builds/coreml_diagnostics/build_provenance.json`; source digest:
  `d3c151d74421f9b2d23785eb5c7383317647287eb5aad659d86b334636f12084`.
- Input: the original bundled JPEG `test_positive_6988.jpg`, decoded once per
  launch. Image SHA-256:
  `e2ce0701572f458a8c634324efb17ca78ab7afbb760f86a828c912e6fb9480f0`.
  The exact input used for the 20 November observation was not confirmed.
- Each configuration used a fresh process: load, first prediction, 30 warm-ups,
  30 measured predictions on the same decoded image, 25-image parity batch,
  then compute-plan inspection. The first prediction is additional to warm-up.
- USB connected; preview off; nominal thermal state at both timing endpoints;
  Low Power Mode off. Order: `all`, `cpuAndGPU`, `cpuOnly`, without prescribed
  cooldown. Charge, brightness, case, ambient temperature and radios were not
  recorded. A diagnostic progress indicator remained visible.

Raw reports, precise detections, phase-marked console logs, comparison reports,
build log and a recalculated `summary.json` are retained below:

```text
outputs/ios_builds/coreml_diagnostics/
  all_console.log
  cpuAndGPU_console.log
  cpuOnly_console.log
  build_provenance.json
  summary.json
  user_xcode_observation_20261123.json
  reports/<device-run-UUID>/
    diagnostic.json
    rfdetr_small_test_<UUID>_predictions.json
    parity_report.json
    repeated_parity_report.json
```

The raw report records UTC creation time, provenance, conditions, first/warm-up/
measured predictions and monotonic stage endpoints. No file or console writes
occur between measured requests. Model timing includes runtime scheduling,
input-tensor preparation at the runtime boundary and required output readback;
image preprocessing and final decoding/filtering are measured separately.
Loading, JPEG decode, preview, parity export and compute-plan inspection are
outside the timed requests.

## Timing results

Times are milliseconds, recalculated from the 30 measured requests using the
protocol's nearest-rank percentiles. Each configuration has one diagnostic run.

| Compute units | Preprocess mean | Model mean / p50 / p95 | Postprocess mean |
| --- | ---: | ---: | ---: |
| `all` | 11.23 | 71.19 / 71.35 / 72.69 | 0.05 |
| `cpuAndGPU` | 45.62 | 197.63 / 197.33 / 199.79 | 0.36 |
| `cpuOnly` | 12.82 | 102.09 / 102.21 / 103.70 | 0.06 |

| Compute units | End-to-end mean / p50 / p95 | Load | First prediction |
| --- | ---: | ---: | ---: |
| `all` | 82.47 / 82.58 / 83.92 | 16,933.29 | 115.63 |
| `cpuAndGPU` | 243.61 / 242.73 / 248.63 | 353.72 | 6,988.85 |
| `cpuOnly` | 114.97 / 115.16 / 117.12 | 320.18 | 125.23 |

The new warmed `all` result does not reproduce 1,575.1 ms per prediction on this
input. Different original build/target/input/warm-up conditions prevent a direct
comparison. Load and first-prediction costs can occur at different boundaries,
as the CPU/GPU first prediction illustrates. These samples do not establish
10-minute sustained rates or formal 300-frame, three-repetition video results.

## Numerical checks and placement

- Each configuration passed the existing 25-image catalog, including its five
  negative examples, against checkpoint, ONNX and Mac Core ML references.
  Each of the 61 repeated image-6988 predictions also passed all three references.
- Alternate compute-unit comparisons use the comparator's diagnostic override,
  after separately checking physical iPhone and Release metadata. The comparator's
  default requirement for physical iPhone plus `all` remains unchanged.
- For all three named `grid_sampler*_cast_fp16` operations (`ios16.resample`),
  the `all` compute plan prefers CPU and lists CPU/GPU support. The CPU/GPU
  plan prefers GPU and lists CPU/GPU support; CPU-only lists and prefers CPU.
- Among nonconstant operations, the `all` plan prefers Neural Engine for 618
  operations and CPU for 38. This is evidence of planned mixed placement.
- A compute plan describes supported/preferred devices and estimated costs;
  it is not an execution trace. See Apple's
  [compute-plan and Performance-report documentation][apple-performance].

## Xcode Performance reports and CPU/GPU retest

The user exported three `.mlperf` packages to:

```text
ios/ENABLECoreMLBench/results/xcode_model_performance_report/rfdetr_small/
```

The originals were preserved. Snapshots, source paths/checksums and extracted data
are saved below `outputs/ios_builds/coreml_diagnostics/xcode_followup/`, in
`exports/` and `xcode_report_summary.json`. The exported reports identify an
iPhone 12 Pro on iOS 27.2.1 and the staged `rfdetr_small.mlpackage` source URL.
The current package at that URL was rehashed and matches the diagnostic model
hash above. The reports themselves do not contain a package checksum.

| Configuration | Xcode outcome | Median model prediction |
| --- | --- | ---: |
| `all` | Report exported | 74.19 ms |
| CPU-only | Report exported | 95.51 ms |
| CPU/Neural Engine | Report exported | 75.03 ms |
| CPU/GPU | Execution-plan/report creation failed, error `-5` | Unavailable |

These medians were recalculated from all 120 prediction samples per successful
export: three experiment iterations of 40 predictions. First predictions remain
included; no warm-up exclusion was applied. These are Xcode model measurements,
separate from the 30-warm-up/30-measured app runs and formal video timing.

In both the `all` and CPU/Neural Engine exports, the preferred-device metadata for
656 nonconstant operations lists Neural Engine for 606 and CPU for 41; nine have
no preferred-device entry. No operation has GPU as its preferred device in these
two reports. All three named grid-sampling operations list CPU as preferred and
supported. These exports provide no per-operation cost weights for those two
configurations, so they cannot quantify the resampling cost. CPU-only lists CPU
for all 656 nonconstant operations. The reported mappings differ from the earlier
app compute plans; a precise cause for that difference has not been established.

The failed CPU/GPU screenshot reports an error building the model execution plan
from `rfdetr_small.mlmodelc/model.mil`, with error code `-5`. No failed `.mlperf`
export is available. This does not establish whether prediction was attempted or
which operation caused the report failure. The exact meaning of this internal
error code has not been established.

To check actual app behavior after that failure, the preserved Release executable
and its recorded sources were reverified, installed and run on the connected
iPhone with explicit `.cpuAndGPU`. The retest completed model loading, the first
prediction, 30 warm-ups, 30 measured predictions, the 25-image parity batch and
compute-plan inspection. All repeated predictions and all 25 images passed the
checkpoint/ONNX/Mac Core ML comparisons. Mean model time was **197.71 ms** and mean
end-to-end time **243.19 ms**, with nominal thermal state at both timing endpoints.
Its compute plan prefers GPU for 653 nonconstant operations and CPU for three.
See `cpuAndGPU_retest_summary.json`, `cpuAndGPU_console.log` and `retest/` in the
follow-up output directory.

The CPU/GPU configuration has now completed two app runs with valid outputs.
Apple's [`cpuAndGPU` definition][apple-cpu-gpu] permits both CPU and GPU execution
and excludes Neural Engine; success under this policy does not prove that every
operation ran on GPU. Planned placement is also distinct from an execution trace.
The evidence supports keeping the report failure open as a profiling discrepancy,
while continuing with the successfully profiled `all` baseline. A runtime trace
would be needed to confirm actual GPU utilization or isolate a GPU-specific fault.

## Warning status and remaining acceptance checks

No fusion or PointerUI warning text appeared in the earlier `devicectl`
stdout/stderr capture. The user's subsequent Xcode screenshots show that the fusion
warnings still recur for all three named grid-sampling operations, each followed
by `Fusion Error: Could not complete graph group mapping step`. The capture methods
therefore expose different diagnostic output; the command-line capture cannot
establish the absence of the warnings.

The same follow-up shows successful interactive detection at 133.4 ms; the user
reports a 110–150 ms range. The displayed timer covers preprocessing, inference
and decoding/filtering, with explicit image/model loading and rendering excluded.
It is comparable in scope to the diagnostic end-to-end timer, with different
execution and warm-up conditions. It must not be compared directly to the
71.19 ms model-only mean or used as a sustained FPS measurement.

The screenshots establish recurrence alongside a completed detection; they do
not include a numerical parity export from that interactive attempt. The preceding
automated 25-image/repeated-image parity results remain separate evidence. The
original 20 November warning survives as the excerpt in the plan. Neither capture
establishes the warnings' load-versus-prediction phase, exact cause or incremental
cost. The improved latency does not establish why the earlier 1,575.1 ms occurred
or that the warning has no performance impact.

The screenshot transcription and user-reported range are preserved in
`user_xcode_observation_20261123.json`; it is explicitly labelled as an observation,
with unverified attempt metadata left unknown. The original screenshots remain in
the conversation; a full raw Xcode console export has not been saved locally.

1. Xcode reports for `all`, CPU-only and CPU/Neural Engine have been collected and
   inspected. The CPU/GPU report failure is recorded alongside a successful app
   retest. Its cause remains open; further profiling is relevant if CPU/GPU becomes
   a required execution policy or a measured bottleneck needs investigation.
2. Preserve a full Xcode console capture during model load and warmed predictions.
   Recurrence is already confirmed by the user's screenshots; correlate subsequent
   messages with the diagnostic phase markers to establish when they occur. Record
   the exact app/model identity and compare predictions from that attempt.
3. Complete frozen-video parity using identical decoded pixels against checkpoint
   and ONNX references, with successful empty results and numerical threshold-boundary
   cases. Existing decoder boundary tests do not establish conversion parity there.

Formal benchmark acceptance remains blocked on those checks for this configuration.
No model, export precision, compute-unit baseline or parity tolerance was changed.

## Repeat the diagnostic

Run the focused checks from the workspace root:

```bash
bash ios/ENABLECoreMLBench/tests/run_coreml_diagnostic_checks.sh
```

Build/install the Release app using the existing integration workflow, then launch
it with the following options. Supply the actual build revision/source identity,
toolchain and observed conditions for that attempt; change compute units for each
separate run. Launching normally opens the ordinary detector UI.

```bash
xcrun devicectl device process launch \
  --device '<device identifier>' --terminate-existing --console \
  com.clymberloongpersonalteam.ENABLECoreMLBench -- \
  --coreml-diagnostics --compute-units all \
  --revision '<actual build revision and source digest>' \
  --toolchain '<Xcode, build, Swift and SDK versions>' \
  --conditions '<observed device conditions>'
```

The diagnostic saves `Documents/CoreMLDiagnostics/<UUID>/diagnostic.json` and its
parity export in the app container, then terminates with exit code 0 on success
or 1 on failure. Copy that directory with `devicectl device copy from` using
`--domain-type appDataContainer` and the app bundle ID as `--domain-identifier`.
The console reports the exact saved path. Model defaults to RF-DETR Small;
`--model`, `--warmup` and `--samples` accept explicit alternatives.

[apple-performance]: https://developer.apple.com/documentation/coreml/analyzing-a-core-ml-model-s-performance-in-xcode
[apple-cpu-gpu]: https://developer.apple.com/documentation/coreml/mlcomputeunits/cpuandgpu
