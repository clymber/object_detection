"""Exercise the manifest CLI and real shell runner with a fake Conda process."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
FAKE_CONDA = r"""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

assert sys.argv[1:3] == ["run", "--no-capture-output"]
notebook = Path(sys.argv[sys.argv.index("nbconvert") + 1])
keys = ("BATCH_SCOPE", "NOTEBOOK_TIMEOUT", "PYTHONNOUSERSITE", "VIRTUAL_ENV",
        "PYTHONPATH", "PYTHONHOME", "PIP_PREFIX", "PIP_TARGET", "PIP_USER")
record = {
    "notebook": str(notebook.relative_to(Path.cwd())),
    "env": {key: os.environ.get(key) for key in keys},
    "args": sys.argv[1:],
    "statuses": [entry.get("status", "pending") for entry in
                 json.loads(Path(os.environ["NOTEBOOK_TEST_MANIFEST"]).read_text())],
}
with Path(os.environ["NOTEBOOK_TEST_RECORD"]).open("a") as stream:
    stream.write(json.dumps(record) + "\n")
print("fake stdout", flush=True)
print("fake stderr", file=sys.stderr, flush=True)
if os.environ.get("NOTEBOOK_TEST_WAIT"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    def stop(signum, frame):
        child.terminate()
        child.wait()
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    sys.stdout.write("partial output")
    sys.stdout.flush()
    Path(os.environ["NOTEBOOK_TEST_WAIT"]).write_text(json.dumps([os.getpid(), child.pid]))
    time.sleep(60)
sys.exit(int(os.environ.get("NOTEBOOK_TEST_EXIT", "0")))
"""


class NotebookRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="notebook runner ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        for name in ("run_notebooks.py", "run_notebook.sh"):
            shutil.copy2(SOURCE_ROOT / "scripts" / name, self.root / "scripts" / name)
        shutil.copy2(SOURCE_ROOT / "Makefile", self.root / "Makefile")
        self.conda = self.root / "fake conda"
        self.conda.write_text(FAKE_CONDA)
        self.conda.chmod(0o755)
        self.manifest = self.root / "task list.json"
        self.record = self.root / "calls.jsonl"
        self.environment = dict(
            os.environ,
            CONDA_EXE=str(self.conda),
            NOTEBOOK_TEST_RECORD=str(self.record),
            NOTEBOOK_TEST_MANIFEST=str(self.manifest),
            BATCH_SCOPE="inherited",
        )
        self.command = [sys.executable, str(self.root / "scripts/run_notebooks.py")]

    def task(self, path="evaluation/notebooks/notebook with spaces.py", **fields):
        source = self.root / path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("# %%\n" if source.suffix == ".py" else "{}")
        source.with_suffix(".ipynb").write_text("{}")
        return {"path": path, **fields}

    def write_manifest(self, entries):
        self.manifest.write_text(json.dumps(entries, indent=2) + "\n")

    def read_manifest(self):
        return json.loads(self.manifest.read_text())

    def calls(self):
        return (
            [json.loads(line) for line in self.record.read_text().splitlines()]
            if self.record.exists()
            else []
        )

    def run_cli(self, *arguments):
        return subprocess.run(
            self.command + list(arguments),
            cwd=self.root.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    def wait_until(self, predicate, process):
        deadline = time.monotonic() + 10
        while not predicate():
            if process.poll() is not None:
                self.fail(f"Runner exited early: {process.communicate()}")
            if time.monotonic() > deadline:
                self.fail("Runner did not reach the expected state")
            time.sleep(0.02)

    def test_required_manifest_and_removed_options(self):
        for arguments in ([], ["-f"], ["--file"], ["--all"], ["nb03.05"]):
            with self.subTest(arguments=arguments):
                result = self.run_cli(*arguments)
                self.assertEqual(result.returncode, 2)
        self.write_manifest([])
        result = self.run_cli("-f", str(self.manifest), "--keep-going")
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.record.exists())

    def test_order_environment_isolation_paths_and_logs(self):
        first = self.task(
            env={
                "BATCH_SCOPE": "first",
                "NOTEBOOK_TIMEOUT": "42",
                **{
                    key: "unwanted"
                    for key in (
                        "VIRTUAL_ENV",
                        "PYTHONPATH",
                        "PYTHONHOME",
                        "PIP_PREFIX",
                        "PIP_TARGET",
                        "PIP_USER",
                    )
                },
            },
            note={"preserve": True},
        )
        second = self.task("dataset/notebooks/second.ipynb", status="pending", env={})
        self.write_manifest([first, second])
        relative_manifest = str(self.manifest.relative_to(self.root.parent))
        result = self.run_cli("--file", relative_manifest)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertEqual(
            [call["notebook"] for call in calls],
            [first["path"].replace(".py", ".ipynb"), second["path"]],
        )
        self.assertEqual(calls[0]["env"]["BATCH_SCOPE"], "first")
        self.assertEqual(calls[1]["env"]["BATCH_SCOPE"], "inherited")
        self.assertEqual(calls[0]["statuses"], ["running", "pending"])
        self.assertEqual(calls[1]["statuses"], ["finished", "running"])
        self.assertIn("--ExecutePreprocessor.timeout=42", calls[0]["args"])
        self.assertIn(
            "--ExecutePreprocessor.kernel_name=object-detection-dataset",
            calls[1]["args"],
        )
        for key in (
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "PIP_PREFIX",
            "PIP_TARGET",
            "PIP_USER",
        ):
            self.assertIsNone(calls[0]["env"][key])
        self.assertEqual(calls[0]["env"]["PYTHONNOUSERSITE"], "1")
        entries = self.read_manifest()
        self.assertEqual(
            [entry["status"] for entry in entries], ["finished", "finished"]
        )
        self.assertEqual(entries[0]["note"], {"preserve": True})
        self.assertEqual(entries[0]["path"], first["path"])
        self.assertNotEqual(entries[0]["log"], entries[1]["log"])
        for entry in entries:
            self.assertFalse(Path(entry["log"]).is_absolute())
            log = (self.root / entry["log"]).read_text()
            self.assertIn("fake stdout", log)
            self.assertIn("fake stderr", log)
            self.assertIn("Exit status: 0", log)
        self.assertIn("2 finished, 0 failed", result.stdout)

    def test_skip_all_nonpending_statuses_even_without_generated_notebooks(self):
        entries = [
            {"path": f"evaluation/notebooks/{status}.py", "status": status}
            for status in ("inactivated", "finished", "error", "running")
        ]
        self.write_manifest(entries)
        before = self.manifest.read_bytes()
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertEqual(self.calls(), [])
        self.assertIn("Confirm the previous process has stopped", result.stdout)
        self.assertIn("4 skipped", result.stdout)

    def test_dry_run_does_not_change_any_files(self):
        self.write_manifest(
            [
                self.task(env={"BATCH_SCOPE": "preview"}),
                {"path": "dataset/notebooks/missing.py", "status": "error"},
            ]
        )
        before = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        result = self.run_cli("-f", str(self.manifest), "--dry-run")
        self.assertEqual(result.returncode, 0, result.stderr)
        after = {
            path.relative_to(self.root): path.read_bytes()
            for path in self.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertIn("Would run", result.stdout)
        self.assertIn('"BATCH_SCOPE": "preview"', result.stdout)
        self.assertIn("outputs/notebook_logs/", result.stdout)
        self.assertIn("Skipping (error)", result.stdout)

    def test_stop_on_failure_and_retry_requires_pending(self):
        first = self.task(
            env={"NOTEBOOK_TEST_EXIT": "7"}, log="outputs/explicit log.txt"
        )
        second = self.task("dataset/notebooks/second.ipynb")
        self.write_manifest([first, second])
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 1)
        entries = self.read_manifest()
        self.assertEqual(entries[0]["status"], "error")
        self.assertNotIn("status", entries[1])
        self.assertEqual(len(self.calls()), 1)

        self.write_manifest([entries[0]])
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 1)
        entry = self.read_manifest()[0]
        entry["status"] = "pending"
        entry["env"]["NOTEBOOK_TEST_EXIT"] = "0"
        self.write_manifest([entry])
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 2)
        log = (self.root / entry["log"]).read_text()
        self.assertEqual(log.count("=== Attempt "), 2)
        self.assertIn("Exit status: 7", log)
        self.assertIn("Exit status: 0", log)

    def test_validate_complete_manifest_before_any_execution(self):
        valid = self.task()
        invalid_entries = [
            {"path": "evaluation/notebooks/missing.ipynb"},
            {"path": "evaluation/notebooks/first.py", "status": "unknown"},
            {"path": "evaluation/notebooks/first.py", "status": []},
            {"path": "evaluation/notebooks/first.py", "env": []},
            {"path": valid["path"], "env": {"INTEGER": 42}},
            {"path": valid["path"], "env": {"INVALID=NAME": "1"}},
            {"path": valid["path"], "env": {"NUL": "\0"}},
            {"path": valid["path"], "log": "../outside.log"},
            {"path": valid["path"], "log": str(self.root / "absolute.log")},
            {"path": valid["path"], "log": "task list.json"},
            {"path": valid["path"], "log": valid["path"]},
            {"path": "../outside.ipynb"},
            {"path": "other/notebooks/first.ipynb"},
            {"path": str(self.root / valid["path"])},
            {"path": "evaluation/notebooks/notebook.txt"},
            {},
            42,
        ]
        for invalid in invalid_entries:
            with self.subTest(invalid=invalid):
                self.write_manifest([valid, invalid])
                before = self.manifest.read_bytes()
                result = self.run_cli("-f", str(self.manifest))
                self.assertEqual(result.returncode, 1)
                self.assertEqual(self.manifest.read_bytes(), before)
                self.assertFalse(self.record.exists())
        self.write_manifest({"notebooks": []})
        self.assertEqual(self.run_cli("-f", str(self.manifest)).returncode, 1)
        self.manifest.write_text("[invalid JSON")
        self.assertEqual(self.run_cli("-f", str(self.manifest)).returncode, 1)

    def test_missing_pair_and_symlink_escape(self):
        entry = self.task()
        (self.root / entry["path"]).with_suffix(".ipynb").unlink()
        self.write_manifest([entry])
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 1)
        self.assertIn("make sync-notebooks", result.stderr)
        (self.root / "outside.ipynb").write_text("{}")
        (self.root / entry["path"]).with_suffix(".ipynb").symlink_to(
            self.root / "outside.ipynb"
        )
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.record.exists())

    def test_interruption_locking_and_live_partial_output(self):
        ready = self.root / "ready.json"
        first = self.task(
            env={"NOTEBOOK_TEST_WAIT": str(ready)}, log="outputs/live.log"
        )
        second = self.task("dataset/notebooks/second.ipynb")
        self.write_manifest([first, second])
        process = subprocess.Popen(
            self.command + ["-f", str(self.manifest)],
            cwd=self.root.parent,
            env=self.environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            self.wait_until(ready.exists, process)
            self.wait_until(
                lambda: "partial output" in (self.root / first["log"]).read_text(),
                process,
            )
            contender = self.run_cli("-f", str(self.manifest))
            self.assertEqual(contender.returncode, 1)
            self.assertIn("Another runner", contender.stderr)
            process.send_signal(signal.SIGINT)
            stdout, stderr = process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130, stderr)
            self.assertIn("1 failed", stdout)
            self.assertEqual(self.read_manifest()[0]["status"], "error")
            self.assertNotIn("status", self.read_manifest()[1])
            self.assertEqual(len(self.calls()), 1)
            for pid in json.loads(ready.read_text()):
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            self.write_manifest([self.read_manifest()[0]])
            self.assertEqual(self.run_cli("-f", str(self.manifest)).returncode, 0)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                process.communicate(timeout=10)

    def test_log_creation_failure_records_error(self):
        (self.root / "blocked").write_text("not a directory")
        self.write_manifest([self.task(log="blocked/notebook.log")])
        result = self.run_cli("-f", str(self.manifest))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self.read_manifest()[0]["status"], "error")
        self.assertEqual(self.calls(), [])

    def test_make_run_notebook_and_shorthand_still_work(self):
        entry = self.task("evaluation/notebooks/nb99.99-example.ipynb")
        self.write_manifest([entry])
        result = subprocess.run(
            ["make", "run-notebook", f"NOTEBOOK={entry['path']}"],
            cwd=self.root,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run(
            ["bash", str(self.root / "scripts/run_notebook.sh"), "nb99.99"],
            cwd=self.root.parent,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(self.calls()), 2)


if __name__ == "__main__":
    unittest.main()
