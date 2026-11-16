"""
Read and compare saved export attempts without importing inference frameworks.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from detection_common.utils.digest import file_digest
from detection_common.utils.json_io import read_json, write_json

from .baseline import find_latest_path
from .metrics import evaluate_predictions, read_prediction_artifact
from .runtime_workflow import (
    FORMATS,
    MODELS,
    artifact_identity,
    checkpoint_path,
    export_in_progress,
    select_run,
    validate_run,
    workspace_path,
)


def select_export(model: str, runs_dir: Path, *,
                  run_selection: Path | None = None,
                  export_selection: Path | None = None) -> dict:
    """
    Resolve an exact selection or the single export manifest of the selected run.
    """
    if export_selection is not None:
        manifest_path = workspace_path(export_selection)
        manifest = read_json(manifest_path)
        if manifest.get("schema_version") != 1 or manifest.get("model") != model:
            raise ValueError(f"Invalid explicit export manifest for {model}")
        run = Path(manifest["run_dir"]).resolve()
        validate_run(run, model)
        if run_selection is not None and workspace_path(run_selection) != run:
            raise ValueError("Explicit run/export selections identify different runs")
    else:
        try:
            run = select_run(runs_dir, model, run_selection)
        except FileNotFoundError as error:
            if run_selection is not None:
                raise
            return {"model": model, "run_dir": None, "manifest_path": None,
                    "reason": str(error)}
        manifest_path = find_latest_path(
            run / "export_validation", "model_manifest.json")
    return {"model": model, "run_dir": str(run),
            "manifest_path": str(manifest_path) if manifest_path else None,
            "reason": None if manifest_path else "Latest selected run has no export results"}


def checked_record(record: dict) -> Path:
    """
    Verify file/package identity before consuming saved runtime evidence.
    """
    path = Path(record["path"])
    actual = artifact_identity(path)
    if actual["sha256"] != record["sha256"] or actual.get("files") != record.get("files"):
        raise ValueError(f"Artifact hash mismatch: {path}")
    return path


def timing_signature(timing: dict) -> dict:
    """
    Compare timings only across matching host, pixels, sequence and boundaries.
    """
    return {key: timing[key] for key in ("kind", "host", "sequence", "sequence_sha256",
            "warmups", "samples", "repetitions", "batch_size", "boundaries")}


def validate_manifest(manifest: dict, selection: dict, dataset_dir: Path) -> dict:
    """
    Validate selected checkpoint, immutable provenance and local annotation identity.
    """
    if manifest.get("schema_version") != 1 or manifest.get("model") != selection["model"]:
        raise ValueError("Unsupported export manifest/model")
    run = Path(selection["run_dir"]).resolve()
    if Path(manifest["run_dir"]).resolve() != run:
        raise ValueError("Export manifest source run differs from selection")
    protocol = validate_run(run, selection["model"])
    if manifest["run_protocol_sha256"] != file_digest(run / "run_protocol.json"):
        raise ValueError("Export run provenance hash differs")
    if manifest.get("source_run_unchanged") is False:
        raise ValueError("Export attempt did not verify source run immutability")
    checkpoint = checkpoint_path(run, selection["model"])
    if manifest["checkpoint"]["sha256"] != file_digest(checkpoint):
        raise ValueError("Export selected checkpoint differs from source run")
    if manifest["dataset"]["fingerprint"] != protocol["dataset_identity"]["canonical"]["source_fingerprint"]:
        raise ValueError("Export dataset fingerprint differs from run")
    annotations = {}
    for split in ("val", "test"):
        path = dataset_dir / "annotations" / f"instances_{split}.json"
        saved = manifest["dataset"]["annotations"][split]
        if file_digest(path) != saved["sha256"]:
            raise ValueError("Comparison annotations differ from export dataset")
        if sorted(i["id"] for i in read_json(path)["images"]) != saved["image_ids"]:
            raise ValueError("Comparison image coverage differs from export dataset")
        annotations[split] = path
    checked_record(manifest["parity_manifest"])
    if "source_snapshot" in manifest:
        checked_record(manifest["source_snapshot"])
    return annotations


def compare_runtime_results(*, runs_dir: Path, dataset_dir: Path,
                            output_dir: Path, run_selections: dict | None = None,
                            export_selections: dict | None = None,
                            models: tuple = MODELS) -> dict:
    """
    Recompute accuracy, retain diagnostic failures and gate host latency comparisons.
    """
    run_selections = run_selections or {}
    export_selections = export_selections or {}
    if (set(run_selections) | set(export_selections))-set(models):
        raise ValueError("Selection mapping contains an unknown model")
    selections = [select_export(model, runs_dir,
                  run_selection=run_selections.get(model),
                  export_selection=export_selections.get(model)) for model in models]
    rows, timing_records, annotation_identity = [], {}, None
    for selection in selections:
        print(f"{selection['model']}: run={selection['run_dir']}; "
              f"export={selection['manifest_path']}")
        if selection["manifest_path"] is None:
            for name in FORMATS:
                rows.append({"model": selection["model"], "runtime": name,
                             "status": "unavailable", "reason": selection["reason"]})
            continue
        if export_in_progress(Path(selection["run_dir"])):
            for name in FORMATS:
                rows.append({"model": selection["model"], "runtime": name,
                             "status": "incomplete", "accepted": False,
                             "reason": "The run's canonical export is being updated"})
            continue
        manifest = read_json(selection["manifest_path"])
        annotations = validate_manifest(manifest, selection, workspace_path(dataset_dir))
        if manifest.get("source_run_unchanged") is not True:
            for name in FORMATS:
                rows.append({"model": selection["model"], "runtime": name,
                             "status": "incomplete", "accepted": False,
                             "run_dir": selection["run_dir"],
                             "manifest_path": selection["manifest_path"],
                             "reason": "Attempt has not finished source immutability verification"})
            continue
        identity = {"fingerprint": manifest["dataset"]["fingerprint"],
                    "annotations": {s: file_digest(p) for s, p in annotations.items()}}
        if annotation_identity is not None and annotation_identity != identity:
            raise ValueError("Cannot compare incompatible datasets or annotations")
        annotation_identity = identity
        baseline_metrics = {}
        for name in FORMATS:
            outcome = manifest["formats"][name]
            status = outcome["status"]
            if status not in {"completed", "failed", "not_requested"}:
                raise ValueError(f"Invalid per-format outcome: {status}")
            contract = outcome.get("contract", {})
            if "artifact" in contract:
                checked_record(contract["artifact"])
            if name == "pytorch" and contract.get("artifact", {}).get("sha256") != manifest["checkpoint"]["sha256"]:
                if status == "completed":
                    raise ValueError("Checkpoint reference runtime has a different checkpoint")
            if "source_checkpoint" in contract:
                checked_record(contract["source_checkpoint"])
            parity = outcome.get("parity_status", "unavailable")
            if "parity" in outcome:
                saved_parity = read_json(checked_record(outcome["parity"]))
                if saved_parity["status"] != parity:
                    raise ValueError("Parity status differs from evidence")
                if parity == "passed" and not all(i["passed"] for i in saved_parity["images"]):
                    raise ValueError("Passed parity contains failed images")
            accepted = status == "completed" and parity in {"passed", "reference"}
            common = {"model": selection["model"], "runtime": name,
                      "status": status, "accepted": accepted, "parity_status": parity,
                      "runtime_precision": contract.get("precision"),
                      "resolution": manifest["configuration"]["resolution"],
                      "backend": contract.get("backend"),
                      "device": contract.get("device"), "compute_units": contract.get("compute_units"),
                      "cpu_threads": contract.get("cpu_threads"),
                      "run_dir": selection["run_dir"], "manifest_path": selection["manifest_path"],
                      "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                      "artifact_sha256": contract.get("artifact", {}).get("sha256"),
                      "reason": outcome.get("error") or outcome.get("timing_unavailable_reason")}
            if "timing" in outcome:
                timing_records[(selection["model"], name)] = read_json(checked_record(outcome["timing"]))
            complete = set(outcome.get("predictions", {})) == {"val", "test"}
            if status == "completed" and not complete:
                raise ValueError("Completed evaluation has incomplete split coverage")
            if not outcome.get("predictions"):
                rows.append({**common, "timing_status": "unavailable"})
                continue
            for missing_split in sorted({"val", "test"} - set(outcome["predictions"])):
                rows.append({**common, "split": missing_split,
                             "evaluation_complete": False,
                             "reason": outcome.get("error") or "Full split predictions unavailable"})
            for split, record in outcome["predictions"].items():
                if split not in annotations:
                    raise ValueError("Unknown evaluation split")
                artifact = read_prediction_artifact(checked_record(record), annotations[split])
                metadata = artifact["metadata"]
                if (metadata["checkpoint_sha256"] != manifest["checkpoint"]["sha256"]
                        or metadata["model"] != selection["model"]
                        or metadata["split"] != split or metadata.get("runtime") != name
                        or metadata.get("runtime_contract") != contract
                        or metadata.get("dataset_fingerprint") != identity["fingerprint"]
                        or metadata["resolution"] != manifest["configuration"]["resolution"]
                        or metadata["postprocessing"] != manifest["configuration"]["postprocessing"]):
                    raise ValueError("Prediction artifact provenance differs from manifest")
                metrics = evaluate_predictions(annotations[split], artifact["predictions"])
                if name == "pytorch":
                    baseline_metrics[split] = metrics
                row = {**common, "split": split, "evaluation_complete": complete}
                for metric in ("ap50", "ap50_95", "ar100", "precision", "recall", "f1",
                               "false_positives_per_negative_image", "negative_image_false_positive_fraction"):
                    row[metric] = metrics[metric]
                    reference = baseline_metrics.get(split, {}).get(metric)
                    row[f"delta_{metric}"] = (metrics[metric]-reference
                        if metrics[metric] is not None and reference is not None else None)
                rows.append(row)
    # A comparison group needs the entire shared timing protocol and host identity.
    signatures = []
    for key, timing in timing_records.items():
        signature = timing_signature(timing)
        if signature not in signatures:
            signatures.append(signature)
    for row in rows:
        timing = timing_records.get((row["model"], row["runtime"]))
        if timing is None:
            row["timing_status"] = "unavailable"
        else:
            row["timing_group"] = signatures.index(timing_signature(timing)) + 1
            row["timing_status"] = ("compatible host measurements" if len(signatures) == 1
                                    else "incompatible across groups; compare within group only")
            row["timing_boundaries"] = timing["boundaries"]
            row["host"] = timing["host"]["name"]
            for key in ("mean_ms", "p50_ms", "p95_ms", "throughput_images_second"):
                row[key] = timing[key]
    frame = pd.DataFrame(rows)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv = destination / "runtime_comparison.csv"
    json_path = destination / "runtime_comparison.json"
    markdown = destination / "runtime_comparison.md"
    frame.to_csv(csv, index=False)
    write_json(json_path, {"schema_version": 1, "selections": selections, "rows": rows,
                          "dataset": annotation_identity,
                          "timing_groups": signatures})
    columns = [c for c in ("model", "runtime", "split", "status", "accepted", "parity_status",
                           "runtime_precision", "resolution", "backend", "compute_units",
                           "ap50_95", "delta_ap50_95", "mean_ms", "timing_group")
               if c in frame]
    lines = ["# Basketball runtime comparison", "",
             "Host timings include preprocessing, prediction and postprocessing on decoded pixels.",
             "These are not iPhone measurements. Compare latency only within a timing group.",
             "Rows with accepted=False are diagnostic results.", "",
             "| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    lines += ["", "Selections:"]
    lines += [f"- {s['model']}: run `{s['run_dir']}`; manifest `{s['manifest_path']}`. "
              f"{s['reason'] or ''}" for s in selections]
    markdown.write_text("\n".join(lines) + "\n")
    return {"table": frame, "selections": selections, "csv_path": csv,
            "json_path": json_path, "markdown_path": markdown}
