#!/usr/bin/env python3
"""Run pending tasks from a JSON notebook manifest in the foreground."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
PROJECTS = {"dataset", "ultralytics", "yolox", "rfdetr", "evaluation"}
STATUSES = {"pending", "inactivated", "finished", "error", "running"}


def workspace_path(value, label):
    """Resolve a relative path and reject paths outside the workspace."""
    if not isinstance(value, str) or not value or "\0" in value:
        raise ValueError(f"{label} must be a nonempty path string")
    if Path(value).is_absolute():
        raise ValueError(f"{label} must be relative to the workspace root: {value}")
    path = (WORKSPACE_ROOT / value).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT) or path == WORKSPACE_ROOT:
        raise ValueError(f"{label} must stay inside the workspace: {value}")
    return path


def load_tasks(manifest):
    """Validate the manifest and return entries with resolved notebook and log paths."""
    with manifest.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise TypeError("The manifest must be a JSON array")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + f"-{os.getpid()}"
    paths = []
    protected = {manifest, manifest.with_name(manifest.name + ".lock")}
    for index, entry in enumerate(entries, 1):
        label = f"Task {index}"
        if not isinstance(entry, dict):
            raise TypeError(f"{label} must be an object")
        status = entry.get("status", "pending")
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError(f"{label} has an invalid status: {status!r}")
        environment = entry.get("env", {})
        if not isinstance(environment, dict):
            raise TypeError(f"{label} env must be an object of string values")
        for key, value in environment.items():
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"{label} has an invalid environment name: {key!r}")
            if not isinstance(value, str) or "\0" in value:
                raise ValueError(
                    f"{label} env[{key!r}] must be a string without NUL bytes"
                )

        source = workspace_path(entry.get("path"), f"{label} path")
        relative = source.relative_to(WORKSPACE_ROOT)
        if (
            len(relative.parts) < 3
            or relative.parts[0] not in PROJECTS
            or relative.parts[1] != "notebooks"
            or source.suffix not in {".py", ".ipynb"}
        ):
            raise ValueError(
                f"{label} path must be a .py or .ipynb in a project notebooks directory"
            )
        notebook = workspace_path(
            str(relative.with_suffix(".ipynb")), f"{label} notebook"
        )
        if notebook.relative_to(WORKSPACE_ROOT).parts[:2] != relative.parts[:2]:
            raise ValueError(
                f"{label} paired notebook must stay in its project notebooks directory"
            )
        protected.update({source, notebook})
        if status == "pending":
            if source.suffix == ".py" and not source.is_file():
                raise ValueError(f"{label} notebook source not found: {relative}")
            if not notebook.is_file():
                raise ValueError(
                    f"{label} notebook not found: {notebook.relative_to(WORKSPACE_ROOT)}. "
                    "Generate notebooks first with make sync-notebooks."
                )

        default_log = f"outputs/notebook_logs/{run_id}/{index:03d}-{relative.parts[0]}-{source.stem}.log"
        log = workspace_path(entry.get("log", default_log), f"{label} log")
        if log.is_dir():
            raise ValueError(
                f"{label} log is a directory: {log.relative_to(WORKSPACE_ROOT)}"
            )
        paths.append((notebook, log))
    for _, log in paths:
        if log in protected:
            raise ValueError(f"Log path would overwrite a manifest or notebook: {log}")
    return entries, paths


@contextmanager
def manifest_lock(manifest):
    """Hold an exclusive lock to prevent concurrent runs of the same manifest."""
    # Lock a stable sidecar: saving statuses replaces the manifest's inode.
    with manifest.with_name(manifest.name + ".lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                f"Another runner is using this manifest: {manifest}"
            ) from None
        yield


def save_manifest(manifest, entries):
    """Atomically save task updates while preserving the manifest's permissions."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=manifest.parent,
            prefix=f".{manifest.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(entries, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        temporary.chmod(manifest.stat().st_mode & 0o777)
        os.replace(temporary, manifest)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def stop_process(process):
    """Terminate the notebook process group and kill any remaining members."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    # The leader can exit before its descendants; stop any remaining group members.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_notebook(notebook, log, overrides):
    """Run one notebook with environment overrides and stream output to its log."""
    environment = os.environ.copy()
    environment.update(overrides)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as stream:
        started = datetime.now(UTC).isoformat()
        stream.write(
            f"\n=== Attempt {started}: {notebook.relative_to(WORKSPACE_ROOT)} ===\n".encode()
        )
        stream.flush()
        process = subprocess.Popen(
            [
                "bash",
                str(WORKSPACE_ROOT / "scripts/run_notebook.sh"),
                str(notebook.relative_to(WORKSPACE_ROOT)),
            ],
            cwd=WORKSPACE_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            with process.stdout:
                while chunk := process.stdout.read1(65536):
                    stream.write(chunk)
                    stream.flush()
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            status = process.wait()
        except BaseException:
            stop_process(process)
            stream.write(b"\n=== Attempt stopped ===\n")
            raise
        stream.write(f"\n=== Exit status: {status} ===\n".encode())
        return status


def describe_task(index, total, entry, notebook, log, dry_run=False):
    """Print task progress, skip reasons, and execution or preview details."""
    status = entry.get("status", "pending")
    action = "Would run" if dry_run else "Running"
    if status != "pending":
        action = f"Skipping ({status})"
    print(f"[{index}/{total}] {action}: {entry['path']}", flush=True)
    if status == "running":
        print(
            "  Confirm the previous process has stopped before resetting to pending.",
            flush=True,
        )
    if status == "pending":
        print(f"  Notebook: {notebook.relative_to(WORKSPACE_ROOT)}", flush=True)
        print(f"  Log: {log.relative_to(WORKSPACE_ROOT)}", flush=True)
        if dry_run:
            print(
                f"  Env overrides: {json.dumps(entry.get('env', {}), ensure_ascii=False)}",
                flush=True,
            )


def run_batch(manifest, entries, paths):
    """Run pending tasks in order, save statuses, and stop on failure or interruption."""
    finished = failed = 0
    skipped = sum(entry.get("status", "pending") != "pending" for entry in entries)
    result = 0
    for index, (entry, (notebook, log)) in enumerate(zip(entries, paths), 1):
        describe_task(index, len(entries), entry, notebook, log)
        if entry.get("status", "pending") != "pending":
            continue
        try:
            entry["log"] = str(log.relative_to(WORKSPACE_ROOT))
            entry["status"] = "running"
            save_manifest(manifest, entries)
            status = run_notebook(notebook, log, entry.get("env", {}))
        except KeyboardInterrupt:
            print("\nInterrupted; stopping the batch.", flush=True)
            status = result = 130
        except OSError as error:
            print(
                f"Error running {entry['path']}: {error}", file=sys.stderr, flush=True
            )
            status = 1
        entry["status"] = "error" if status else "finished"
        save_manifest(manifest, entries)
        if status:
            failed += 1
            result = result or 1
            print(f"Failed: {entry['path']} (exit status {status}).", flush=True)
            break
        finished += 1
    pending = sum(entry.get("status", "pending") == "pending" for entry in entries)
    print(
        f"Summary: {finished} finished, {failed} failed, {skipped} skipped, {pending} pending.",
        flush=True,
    )
    return result


def main(argv=None):
    """Parse CLI arguments and preview or execute the requested task manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="JSON task manifest (relative to the invoking directory)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show pending notebooks and configured overrides without changing files",
    )
    args = parser.parse_args(argv)

    try:
        manifest = args.file.resolve()
        if args.dry_run:
            entries, paths = load_tasks(manifest)
            for index, (entry, (notebook, log)) in enumerate(zip(entries, paths), 1):
                describe_task(index, len(entries), entry, notebook, log, dry_run=True)
            return 0

        with manifest_lock(manifest):
            entries, paths = load_tasks(manifest)
            return run_batch(manifest, entries, paths)
    except (OSError, TypeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
