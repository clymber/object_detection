"""
Discover and validate framework-neutral basketball comparison bundles.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from detection_common.utils.json_io import write_json

from .metrics import write_comparison
from .run_protocol import BUNDLE_MANIFEST_FILE_NAME, read_bundle

MODEL_NAMES = ("yolo11n", "yolox_tiny", "yolox_nano", "rfdetr_small")
LOGICAL_DATASET = "basketball"


@dataclass(frozen=True)
class BundleSelection:
    """
    Hold one selected comparison bundle or an unavailable recovery result.
    """

    model: str
    bundle_dir: Path | None
    bundle: dict[str, Any] | None
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        """
        Return whether this selection contains a validated full bundle.
        """
        return self.bundle is not None

    def artifact_path(self, split: str) -> Path | None:
        """
        Return the selected immutable artifact path for one comparison split.
        """
        if not self.bundle_dir or not self.bundle or split not in {"val", "test"}:
            return None
        name = f"{split}_predictions"
        return self.bundle_dir / self.bundle["manifest"]["artifacts"][name]["path"]

    def training_summary(self) -> dict[str, Any] | None:
        """
        Return the independently validated training summary for report generation.
        """
        if self.bundle is None:
            return None
        return dict(self.bundle["training"]["summary"])

    def report_details(self) -> dict[str, Any]:
        """
        Return selected provenance suitable for JSON and tabular reports.
        """
        if self.bundle is None:
            return {
                "status": "unavailable",
                "recovery": self.unavailable_reason,
            }
        manifest = self.bundle["manifest"]
        return {
            "status": "available",
            "bundle_dir": str(self.bundle_dir),
            "generation": manifest["generation"],
            "selected_checkpoint": dict(manifest["selected_checkpoint"]),
            "resolution": manifest["resolution"],
            "parameter_count": manifest["parameter_count"],
            "protocol": dict(manifest["provenance"]),
            "training_summary": self.training_summary(),
            "training": dict(self.bundle["training"]),
        }


def _is_full_dataset(protocol: Mapping[str, Any]) -> bool:
    """
    Return whether provenance describes a non-smoke, non-truncated dataset run.
    """
    if protocol["smoke_run"]:
        return False
    settings = protocol["training_settings"]
    if settings.get("train_batch_limit") is not None:
        return False
    fraction = settings.get("fraction", 1.0)
    return isinstance(fraction, (int, float)) and not isinstance(fraction, bool) and fraction == 1


def _validate_bundle(model: str, bundle: Mapping[str, Any]) -> None:
    """
    Validate consumer-specific model, split, checkpoint, and protocol invariants.
    """
    manifest = bundle["manifest"]
    protocol = manifest["provenance"]
    if protocol["model"] != model:
        raise ValueError(f"Bundle model differs from directory model: {model}")
    if protocol["logical_dataset"] != LOGICAL_DATASET:
        raise ValueError(f"Bundle for {model} has the wrong logical dataset")
    if not _is_full_dataset(protocol):
        raise ValueError(f"Bundle for {model} is smoke or uses a truncated dataset")
    for split in ("val", "test"):
        artifact = bundle[f"{split}_predictions"]
        metadata = artifact["metadata"]
        if metadata["split"] != split or metadata["model"] != model:
            raise ValueError(f"Bundle for {model} has an invalid {split} split mapping")
        if metadata["checkpoint_sha256"] != manifest["selected_checkpoint"]["sha256"]:
            raise ValueError(f"Bundle for {model} has a different selected checkpoint")
        if metadata["resolution"] != manifest["resolution"]:
            raise ValueError(f"Bundle for {model} has a different selected resolution")
    if manifest["parameter_count"] <= 0:
        raise ValueError(f"Bundle for {model} has an invalid parameter count")


def _sort_key(selection: BundleSelection) -> tuple[datetime, str]:
    """
    Return the stable newest-first candidate sort key from immutable provenance.
    """
    assert selection.bundle is not None and selection.bundle_dir is not None
    timestamp = selection.bundle["manifest"]["provenance"]["original_utc"]
    return datetime.fromisoformat(timestamp.replace("Z", "+00:00")), selection.bundle_dir.name


def _read_candidate(
    model: str,
    bundle_dir: Path,
    *,
    allow_smoke: bool = False,
) -> BundleSelection:
    """
    Read one published bundle exactly once and apply consumer-specific validation.
    """
    bundle = read_bundle(bundle_dir, allow_smoke=allow_smoke)
    _validate_bundle(model, bundle)
    return BundleSelection(model=model, bundle_dir=bundle_dir, bundle=bundle)


def _selection_path(root: Path, model: str, requested: str | Path) -> Path:
    """
    Resolve a requested run name or explicit bundle directory under one model root.
    """
    path = Path(requested).expanduser()
    if not path.is_absolute():
        path = root / model / path
    return path.resolve()


def discover_bundles(
    output_root: Path | str,
    *,
    models: Sequence[str] = MODEL_NAMES,
    selections: Mapping[str, str | Path] | None = None,
) -> dict[str, BundleSelection]:
    """
    Select compatible complete full bundles, retaining missing models as unavailable.

    Automatic selection reads every published candidate once, then chooses the
    newest by immutable protocol UTC timestamp and run directory name. Explicit
    selections are strict and therefore surface missing, invalid, or smoke paths.
    """
    root = Path(output_root).expanduser().resolve() / "evaluation" / "basketball_large_dataset"
    requested = selections or {}
    unknown = set(requested) - set(models)
    if unknown:
        raise ValueError(f"Unknown comparison model selections: {sorted(unknown)}")
    results: dict[str, BundleSelection] = {}
    for model in models:
        model_root = root / model
        if model in requested:
            path = _selection_path(root, model, requested[model])
            if not (path / BUNDLE_MANIFEST_FILE_NAME).is_file():
                raise FileNotFoundError(
                    f"No published bundle for explicitly selected {model}: {path}"
                )
            results[model] = _read_candidate(model, path)
            continue
        if not model_root.is_dir():
            results[model] = BundleSelection(
                model=model,
                bundle_dir=None,
                bundle=None,
                unavailable_reason=(
                    "No published bundle. Finish or recover a new-protocol full run "
                    "in the owning model project, then publish its evaluation bundle."
                ),
            )
            continue
        candidates = []
        for path in model_root.iterdir():
            if not path.is_dir() or not (path / BUNDLE_MANIFEST_FILE_NAME).is_file():
                continue
            try:
                candidates.append(_read_candidate(model, path, allow_smoke=True))
            except ValueError as error:
                if "smoke or uses a truncated dataset" not in str(error):
                    raise
        if not candidates:
            results[model] = BundleSelection(
                model=model,
                bundle_dir=None,
                bundle=None,
                unavailable_reason=(
                    "No complete published bundle. Finish or recover a new-protocol "
                    "full run in the owning model project."
                ),
            )
            continue
        results[model] = max(candidates, key=_sort_key)
    _validate_canonical_identity(results)
    return results


def _validate_canonical_identity(selections: Mapping[str, BundleSelection]) -> None:
    """
    Require all available bundles to share the same canonical source fingerprint.
    """
    identities = {
        model: selection.bundle["manifest"]["provenance"]["dataset_identity"]
        ["canonical"]["source_fingerprint"]
        for model, selection in selections.items()
        if selection.bundle is not None
    }
    if len(set(identities.values())) > 1:
        rendered = ", ".join(f"{model}={value}" for model, value in identities.items())
        raise ValueError(f"Selected bundles use different canonical sources: {rendered}")


def _write_selection_report(
    selections: Mapping[str, BundleSelection],
    output_dir: Path,
) -> None:
    """
    Persist selected paths and immutable provenance without loading model code.
    """
    details = {model: selection.report_details() for model, selection in selections.items()}
    rows = []
    for model, detail in details.items():
        checkpoint = detail.get("selected_checkpoint", {})
        protocol = detail.get("protocol", {})
        summary = detail.get("training_summary", {})
        rows.append(
            {
                "model": model,
                "status": detail["status"],
                "bundle_dir": detail.get("bundle_dir"),
                "generation": detail.get("generation"),
                "checkpoint_path": checkpoint.get("path"),
                "checkpoint_sha256": checkpoint.get("sha256"),
                "original_utc": protocol.get("original_utc"),
                "resolution": detail.get("resolution"),
                "parameter_count": detail.get("parameter_count"),
                "training_status": summary.get("status"),
                "training_timing_available": summary.get("timing_available"),
                "training_total_hours": summary.get("total_hours"),
                "training_completed_epochs": summary.get("completed_epochs"),
                "recovery": detail.get("recovery"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "selection.csv", index=False)
    write_json(output_dir / "selection.json", {"models": details})
    lines = ["# Basketball bundle selection", ""]
    lines.extend(
        [
            "| Model | Status | Bundle | Generation | Checkpoint SHA256 | Recovery |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {model} | {status} | {bundle_dir} | {generation} | "
            "{checkpoint_sha256} | {recovery} |".format(
                **{key: value if value is not None else "-" for key, value in row.items()}
            )
        )
    (output_dir / "selection.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_bundle_comparisons(
    selections: Mapping[str, BundleSelection],
    annotations: Mapping[str, Path | str],
    output_dir: Path | str,
) -> dict[str, pd.DataFrame]:
    """
    Re-evaluate selected bundle snapshots and write comparison plus provenance reports.
    """
    if set(annotations) != {"val", "test"}:
        raise ValueError("Comparison annotations must contain exactly val and test")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _validate_canonical_identity(selections)
    summaries = {model: selection.training_summary() for model, selection in selections.items()}
    reports = {}
    for split, annotation_path in annotations.items():
        reports[split] = write_comparison(
            {
                model: selection.artifact_path(split)
                for model, selection in selections.items()
            },
            annotation_path,
            destination,
            split=split,
            training_summaries=summaries,
        )
    _write_selection_report(selections, destination)
    return reports