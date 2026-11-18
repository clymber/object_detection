#!/usr/bin/env bash
# Check session lifecycle/overlay geometry and optionally the actual Small model.
set -euo pipefail

still_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$still_workspace"
still_build=outputs/ios_builds/still_image
mkdir -p "$still_build/tmp" outputs/cache/swift_module_cache
still_sources=(
  ios/ENABLECoreMLBench/ENABLECoreMLBench/Detection/*.swift
  ios/ENABLECoreMLBench/ENABLECoreMLBench/Benchmark/StillImage*.swift
)
still_flags=(
  -parse-as-library -swift-version 5 -default-isolation MainActor
  -strict-concurrency=complete -warnings-as-errors
  -enable-upcoming-feature NonisolatedNonsendingByDefault
  -module-cache-path outputs/cache/swift_module_cache
)
for still_test in StillImageSession PreviewGeometry; do
  xcrun swiftc "${still_flags[@]}" "${still_sources[@]}" \
    "ios/ENABLECoreMLBench/tests/${still_test}SmokeTests.swift" \
    -o "$still_build/$still_test"
  "$still_build/$still_test"
done

if [[ "${1:-}" == "--with-model" ]]; then
  xcrun swiftc -O "${still_flags[@]}" "${still_sources[@]}" \
    ios/ENABLECoreMLBench/ENABLECoreMLBench/App/FramePreview.swift \
    ios/ENABLECoreMLBench/tests/StillImageIntegrationSmokeTests.swift \
    -o "$still_build/StillImageIntegration"
  TMPDIR="$still_workspace/$still_build/tmp" "$still_build/StillImageIntegration" \
    "$still_workspace" \
    "$still_workspace/outputs/ios_builds/rfdetr_small_resources/Build/Products/Debug-iphoneos/ENABLECoreMLBench.app"
elif [[ $# -ne 0 ]]; then
  echo "Usage: bash ios/ENABLECoreMLBench/tests/run_still_image_checks.sh [--with-model]" >&2
  exit 2
fi
