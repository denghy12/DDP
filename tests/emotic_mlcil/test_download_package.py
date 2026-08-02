import hashlib
import importlib.util
import json
import tarfile
import tempfile
import unittest
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "package_benchmark_download.py"
)
SPEC = importlib.util.spec_from_file_location("package_benchmark_download", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_bundle(root, run_id, include_checkpoint=False):
    bundle = (
        root
        / "benchmarks"
        / "test_protocol"
        / "A"
        / "CSC"
        / "seed0"
        / "results_to_sync"
        / run_id
    )
    files = {
        "config_resolved.json": json.dumps(
            {"protocol": {"tasks": [["a"], ["b"]]}}
        ),
        "run_manifest.json": json.dumps(
            {
                "git_commit": "abc123",
                "git_dirty": False,
                "reporting_split": "val",
                "configuration_locked": False,
            }
        ),
        "report.html": "<html></html>",
        "train.log": "training complete\n",
        "metrics/task_metrics.json": json.dumps({"tasks": []}),
        "metrics/summary.json": json.dumps({"main_table": {}}),
        "scores/task0_scores.pt": "scores-0",
        "scores/task1_scores.pt": "scores-1",
    }
    records = []
    for relative, content in files.items():
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    (bundle / "sync_manifest.json").write_text(
        json.dumps({"contains_pth": False, "files": records}),
        encoding="utf-8",
    )
    if include_checkpoint:
        (bundle / "nested").mkdir()
        (bundle / "nested" / "task1.pth").write_text(
            "forbidden", encoding="utf-8"
        )
    return bundle


class DownloadPackageTest(unittest.TestCase):
    def test_builds_single_archive_with_results_logs_manifest_and_no_pth(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_id = "csc_seed0_val_test"
            run_root = base / run_id
            make_bundle(run_root, run_id)
            log_dir = base / "_launcher_logs"
            log_dir.mkdir()
            launcher_log = log_dir / f"{run_id}.log"
            launcher_log.write_text("launcher complete\n", encoding="utf-8")

            result = MODULE.build_download_package(
                run_root=run_root,
                run_id=run_id,
                expected_bundles=1,
            )
            archive = Path(result["archive"])
            checksum = Path(result["checksum_file"])
            self.assertTrue(archive.is_file())
            self.assertTrue(checksum.is_file())
            self.assertEqual(result["launcher_log_count"], 1)
            self.assertFalse(result["contains_pth"])
            self.assertEqual(result["archive_sha256"], file_sha256(archive))
            self.assertIn(result["archive_sha256"], checksum.read_text())
            with tarfile.open(archive, "r:gz") as stream:
                members = [member.name for member in stream.getmembers()]
            self.assertTrue(any(name.endswith("metrics/summary.json") for name in members))
            self.assertTrue(any(name.endswith("scores/task1_scores.pt") for name in members))
            self.assertTrue(any(name.endswith(f"logs/{run_id}.log") for name in members))
            self.assertTrue(any(name.endswith("download_manifest.json") for name in members))
            self.assertFalse(any(name.lower().endswith(".pth") for name in members))

    def test_rejects_nested_checkpoint_even_if_sync_manifest_omits_it(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            run_id = "checkpoint_leak"
            run_root = base / run_id
            make_bundle(run_root, run_id, include_checkpoint=True)
            with self.assertRaisesRegex(RuntimeError, "forbidden .pth"):
                MODULE.build_download_package(
                    run_root=run_root,
                    run_id=run_id,
                    expected_bundles=1,
                )

    def test_rejects_incomplete_bundle_count(self):
        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory) / "run"
            run_root.mkdir()
            with self.assertRaisesRegex(RuntimeError, "Expected 1"):
                MODULE.build_download_package(
                    run_root=run_root,
                    run_id="missing_bundle",
                    expected_bundles=1,
                )


if __name__ == "__main__":
    unittest.main()
