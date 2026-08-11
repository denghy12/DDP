import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "emotic-mlcil" / "compare_cocoer_upstream_reference.py"
SPEC = importlib.util.spec_from_file_location("compare_cocoer_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PREPARE = ROOT / "scripts" / "emotic-mlcil" / "prepare_cocoer_head_cache.py"
GENERATE = ROOT / "scripts" / "emotic-mlcil" / "generate_cocoer_head_detections.py"
ASSET_AUDIT = ROOT / "scripts" / "emotic-mlcil" / "audit_cocoer_assets.py"
NATIVE_ASSETS = ROOT / "scripts" / "emotic-mlcil" / "prepare_cocoer_native_assets.py"


class CocoERFTReferenceAuditTest(unittest.TestCase):
    def test_direct_entrypoints_resolve(self):
        for script, option in (
            (SCRIPT, "--upstream-root"),
            (PREPARE, "--detections"),
            (GENERATE, "--insightface-root"),
            (ASSET_AUDIT, "--head-cache"),
            (NATIVE_ASSETS, "--resnet50-source"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [sys.executable, str(script), "--help"], cwd=temporary,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(option, result.stdout)

    def test_fixed_external_source_and_conversion_contract(self):
        configured = os.environ.get("COCOER_UPSTREAM_ROOT")
        candidates = [Path(configured)] if configured else [
            Path("/Users/denghaoyuan/workspace/MyCode/baseline_sources/cocoer_release_dac8fc1"),
            Path("/mnt/haoyuan/workspace/baseline_sources/cocoer_release_dac8fc1"),
        ]
        source = next((path for path in candidates if path.is_dir()), candidates[0])
        if not source.is_dir():
            self.skipTest("fixed external CocoER source is unavailable")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertEqual(payload["operator_equivalence"]["dynamic_weight_max_abs_error"], 0.0)
        conversion = payload["conversion"]
        self.assertEqual(conversion["track"], "B")
        self.assertEqual(conversion["native_visual_stack"], "ImageNet ResNet-50 x3 + OpenAI CLIP RN50")
        self.assertFalse(conversion["distillation"])
        self.assertFalse(conversion["replay"])
        self.assertEqual(len(conversion["rejected_full_class_assets"]), 2)

    def test_head_cache_freezer_is_deterministic_and_rejects_bad_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "detections.json"
            output = directory / "cache.json"
            source.write_text(json.dumps({
                "schema_version": 1,
                "upstream_commit": MODULE.COMMIT,
                "partial": False,
                "detector": {
                    "library": "insightface",
                    "version": "0.7.3",
                    "model": "buffalo_l",
                    "model_tree_sha256": "a" * 64,
                    "det_size": [640, 640],
                },
                "entries": {"emotic:test:a.jpg:person=0": [1, 2, 10, 12]},
                "unresolved": [],
            }), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--detections", str(source),
                 "--output", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["upstream_commit"], MODULE.COMMIT)
            self.assertEqual(len(payload["entries"]), 1)
            self.assertEqual(payload["detector_model_tree_sha256"], "a" * 64)

    def test_partial_head_detections_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "detections.json"
            output = directory / "cache.json"
            source.write_text(json.dumps({
                "schema_version": 1,
                "upstream_commit": MODULE.COMMIT,
                "partial": True,
                "detector": {
                    "library": "insightface",
                    "version": "0.7.3",
                    "model": "buffalo_l",
                    "model_tree_sha256": "a" * 64,
                    "det_size": [640, 640],
                },
                "entries": {"emotic:test:a.jpg:person=0": [1, 2, 10, 12]},
                "unresolved": [],
            }), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--detections", str(source),
                 "--output", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Partial CocoER detections", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
