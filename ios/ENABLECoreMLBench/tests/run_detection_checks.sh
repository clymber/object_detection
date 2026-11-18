#!/usr/bin/env bash
# Run focused Swift checks; --with-models also checks the actual local exports.
set -euo pipefail

detector_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$detector_workspace"
detector_build=outputs/ios_builds/detection_pipeline
detector_tmp="$detector_workspace/$detector_build/tmp"
mkdir -p "$detector_build" "$detector_tmp" outputs/cache/swift_module_cache
detector_sources=(ios/ENABLECoreMLBench/ENABLECoreMLBench/Detection/*.swift)
detector_flags=(
  -parse-as-library -swift-version 5 -default-isolation MainActor
  -strict-concurrency=complete -warnings-as-errors
  -enable-upcoming-feature NonisolatedNonsendingByDefault
  -enable-upcoming-feature InferIsolatedConformances
  -enable-upcoming-feature InferSendableFromCaptures
  -enable-upcoming-feature DisableOutwardActorInference
  -enable-upcoming-feature GlobalActorIsolatedTypesUsability
  -module-cache-path outputs/cache/swift_module_cache
)

for detector_test in DetectionContract RFDETRDecoder YOLOXDecoder DetectorEngine; do
  xcrun swiftc "${detector_flags[@]}" "${detector_sources[@]}" \
    "ios/ENABLECoreMLBench/tests/${detector_test}SmokeTests.swift" \
    -o "$detector_build/$detector_test"
  "$detector_build/$detector_test"
done

detector_sdk=$(xcrun --sdk iphoneos --show-sdk-path)
xcrun swiftc "${detector_flags[@]}" -typecheck -target arm64-apple-ios26.6 \
  -sdk "$detector_sdk" "${detector_sources[@]}"

if [[ "${1:-}" == "--with-models" ]]; then
  .conda/envs/object-detection-yolox/bin/python \
    ios/ENABLECoreMLBench/tests/make_detector_fixtures.py "$detector_workspace"
  for detector_test in Preprocessing DetectorParity; do
    xcrun swiftc -O "${detector_flags[@]}" "${detector_sources[@]}" \
      "ios/ENABLECoreMLBench/tests/${detector_test}SmokeTests.swift" \
      -o "$detector_build/$detector_test"
    TMPDIR="$detector_tmp" "$detector_build/$detector_test" \
      "$detector_build/fixtures"
  done
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash ios/ENABLECoreMLBench/tests/run_detection_checks.sh [--with-models]" >&2
  exit 2
fi
