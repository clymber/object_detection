#!/usr/bin/env bash
# Validate diagnostic launch options and typecheck device-only APIs.
set -euo pipefail

diagnostic_workspace=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$diagnostic_workspace"
diagnostic_build=outputs/ios_builds/coreml_diagnostics
mkdir -p "$diagnostic_build" outputs/cache/swift_module_cache
diagnostic_sources=(
  ios/ENABLECoreMLBench/ENABLECoreMLBench/Detection/*.swift
  ios/ENABLECoreMLBench/ENABLECoreMLBench/Benchmark/StillImage*.swift
  ios/ENABLECoreMLBench/ENABLECoreMLBench/Benchmark/CoreMLDiagnosticRunner.swift
)
diagnostic_flags=(
  -parse-as-library -swift-version 5 -default-isolation MainActor
  -strict-concurrency=complete -warnings-as-errors
  -enable-upcoming-feature NonisolatedNonsendingByDefault
  -enable-upcoming-feature MemberImportVisibility
  -module-cache-path outputs/cache/swift_module_cache
)
xcrun swiftc "${diagnostic_flags[@]}" "${diagnostic_sources[@]}" \
  ios/ENABLECoreMLBench/tests/CoreMLDiagnosticSmokeTests.swift \
  -o "$diagnostic_build/CoreMLDiagnostic"
"$diagnostic_build/CoreMLDiagnostic"
diagnostic_sdk=$(xcrun --sdk iphoneos --show-sdk-path)
xcrun swiftc "${diagnostic_flags[@]}" -typecheck -target arm64-apple-ios26.6 \
  -sdk "$diagnostic_sdk" "${diagnostic_sources[@]}" \
  ios/ENABLECoreMLBench/ENABLECoreMLBench/App/CoreMLDiagnosticView.swift
