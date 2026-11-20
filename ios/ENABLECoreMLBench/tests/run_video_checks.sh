#!/usr/bin/env bash
# Check frame contracts and real video decoding without loading models.
set -euo pipefail

video_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$video_workspace"
video_build=outputs/ios_builds/video_pipeline
mkdir -p "$video_build" outputs/cache/swift_module_cache
video_sources=(ios/ENABLECoreMLBench/ENABLECoreMLBench/Video/*.swift)
video_flags=(
  -parse-as-library -swift-version 5 -default-isolation MainActor
  -strict-concurrency=complete -warnings-as-errors
  -enable-upcoming-feature NonisolatedNonsendingByDefault
  -enable-upcoming-feature MemberImportVisibility
  -module-cache-path outputs/cache/swift_module_cache
)

for video_test in VideoFrameContract VideoFrameReader; do
  xcrun swiftc "${video_flags[@]}" "${video_sources[@]}" \
    "ios/ENABLECoreMLBench/tests/${video_test}SmokeTests.swift" \
    -o "$video_build/$video_test"
  "$video_build/$video_test"
done

video_sdk=$(xcrun --sdk iphoneos --show-sdk-path)
xcrun swiftc "${video_flags[@]}" -typecheck -target arm64-apple-ios26.6 \
  -sdk "$video_sdk" "${video_sources[@]}"
