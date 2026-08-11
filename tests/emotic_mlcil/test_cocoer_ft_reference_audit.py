import importlib.util
import json
import os
import subprocess
import sys
import tarfile
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
INSIGHTFACE_ASSETS = ROOT / "scripts" / "emotic-mlcil" / "prepare_cocoer_insightface_assets.py"
SMOKE = ROOT / "scripts" / "emotic-mlcil" / "smoke_cocoer_ft_training.py"
PACKAGE = ROOT / "scripts" / "emotic-mlcil" / "package_cocoer_head_preprocess.py"
FORMAL_VALIDATOR = ROOT / "scripts" / "emotic-mlcil" / "validate_cocoer_ft_formal_results.py"
FORMAL_LAUNCHER = ROOT / "scripts" / "emotic-mlcil" / "launch_cocoer_ft_formal_seed012_tmux.sh"
PACKAGE_SPEC = importlib.util.spec_from_file_location(
    "package_cocoer_head_preprocess", PACKAGE
)
PACKAGE_MODULE = importlib.util.module_from_spec(PACKAGE_SPEC)
PACKAGE_SPEC.loader.exec_module(PACKAGE_MODULE)
FORMAL_SPEC = importlib.util.spec_from_file_location(
    "validate_cocoer_ft_formal_results", FORMAL_VALIDATOR
)
FORMAL_MODULE = importlib.util.module_from_spec(FORMAL_SPEC)
FORMAL_SPEC.loader.exec_module(FORMAL_MODULE)
LAUNCHER = ROOT / "scripts" / "emotic-mlcil" / "launch_cocoer_ft_seed0_tmux.sh"
HEAD_RESULT = ROOT / "docs" / "benchmarks" / "results" / "cocoer_head_preprocess_v0.1.json"
BUFFALO_L_TREE_SHA256 = "50fa1383e97d137f2902b53de7b7305ffbd35eb4ae32135d95d1e25d5a9d9d3d"


def _detection_payload(*, partial=False):
    return {
        "schema_version": 2,
        "upstream_commit": MODULE.COMMIT,
        "partial": partial,
        "processed_samples": 2,
        "native_resolved_samples": 1,
        "native_unresolved_samples": 1,
        "fallback_samples": 1,
        "unresolved_samples": 0,
        "processed_by_split": {"train": 2},
        "native_resolved_by_split": {"train": 1},
        "native_unresolved_by_split": {"train": 1},
        "fallback_by_split": {"train": 1},
        "detector": {
            "library": "insightface",
            "version": "0.7.3",
            "model": "buffalo_l",
            "detector_file": "det_10g.onnx",
            "model_tree_sha256": BUFFALO_L_TREE_SHA256,
            "implementation": "insightface_scrfd_only",
            "requested_device": "cuda",
            "requested_providers": [
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ],
            "actual_providers": [
                "CUDAExecutionProvider", "CPUExecutionProvider"
            ],
            "det_size": [640, 640],
            "faceanalysis_equivalence": {
                "samples": 2,
                "integer_boxes_exact_match": True,
                "max_abs_error_after_integer_clipping": 0.0,
                "faceanalysis_detector_actual_providers": [
                    "CUDAExecutionProvider", "CPUExecutionProvider"
                ],
            },
        },
        "conversion": {
            "name": "buffalo_l_strict_then_train_median_v0.1",
            "sample_preserving": True,
            "native_source": "buffalo_l_strict",
            "fallback_source": "train_median_relative_geometry",
            "fallback_calibration_split": "train",
            "fallback_uses_labels": False,
            "fallback_uses_val_or_test_statistics": False,
            "fallback_statistic": "componentwise_median",
            "fallback_relative_to": "annotated_body_xyxy",
            "fallback_rounding": "clip_to_image_then_round_half_up",
            "train_native_calibration_samples": 1,
            "median_relative_head_box": [0.2, 0.0, 0.8, 0.3],
        },
        "entries": {
            "emotic:train:a.jpg:person=0": [2, 0, 8, 3],
            "emotic:train:b.jpg:person=0": [2, 0, 8, 3],
        },
        "fallback_sample_ids": ["emotic:train:b.jpg:person=0"],
        "unresolved": [],
    }


class CocoERFTReferenceAuditTest(unittest.TestCase):
    def test_registered_head_result_and_memory_gate(self):
        payload = json.loads(HEAD_RESULT.read_text(encoding="utf-8"))
        self.assertEqual(payload["coverage"]["processed_samples"], 23766)
        self.assertEqual(payload["coverage"]["native_resolved_samples"], 20611)
        self.assertEqual(payload["coverage"]["fallback_samples"], 3155)
        self.assertEqual(payload["coverage"]["unresolved_samples"], 0)
        self.assertTrue(payload["assets"]["assets_valid"])
        self.assertFalse(payload["gates"]["cuda_memory_smoke_passed"])
        launcher = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("smoke_cocoer_ft_training.py", launcher)
        self.assertIn("--batch-size \"${TRAIN_BATCH_SIZE}\"", launcher)
        self.assertIn("MEMORY_SMOKE_JSON", launcher)

    def test_runtime_entrypoints_restore_repository_root(self):
        for script in (GENERATE, ASSET_AUDIT):
            with self.subTest(script=script.name):
                spec = importlib.util.spec_from_file_location(
                    f"runtime_path_{script.stem}", script
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                original_path = list(sys.path)
                try:
                    sys.path[:] = [item for item in sys.path if item != str(ROOT)]
                    module._ensure_repository_root()
                    self.assertEqual(sys.path[0], str(ROOT))
                finally:
                    sys.path[:] = original_path

    def test_direct_entrypoints_resolve(self):
        for script, option in (
            (SCRIPT, "--upstream-root"),
            (PREPARE, "--detections"),
            (GENERATE, "--insightface-root"),
            (ASSET_AUDIT, "--head-cache"),
            (NATIVE_ASSETS, "--resnet50-source"),
            (INSIGHTFACE_ASSETS, "--output-root"),
            (SMOKE, "--resnet50-init"),
            (PACKAGE, "--package-name"),
            (FORMAL_VALIDATOR, "--expected-git-commit"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [sys.executable, str(script), "--help"], cwd=temporary,
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(option, result.stdout)

    def test_preprocess_packager_uses_relative_non_self_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sources = {}
            for name in (
                "detections.json",
                "head_cache.json",
                "asset_audit.json",
                "native_manifest.json",
            ):
                path = root / "sources" / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"name": name}), encoding="utf-8")
                sources[name] = path
            log = root / "preprocess.log"
            log.write_text("complete\n", encoding="utf-8")
            result = PACKAGE_MODULE.build_package(
                output_base=root / "output",
                package_name="cocoer_head_test",
                detections=sources["detections.json"],
                head_cache=sources["head_cache.json"],
                asset_audit=sources["asset_audit.json"],
                native_manifest=sources["native_manifest.json"],
                logs=[log],
            )
            destination = Path(result["download_directory"])
            manifest = json.loads(
                (destination / "download_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            paths = [record["path"] for record in manifest["files"]]
            self.assertTrue(manifest["paths_are_relative"])
            self.assertFalse(manifest["manifest_self_included"])
            self.assertNotIn("download_manifest.json", paths)
            self.assertTrue(all(not Path(path).is_absolute() for path in paths))
            self.assertFalse(result["contains_pth"])
            self.assertFalse(result["contains_onnx"])
            checksum = Path(result["checksum_file"]).read_text(encoding="utf-8")
            self.assertIn(Path(result["archive"]).name, checksum)
            self.assertNotIn(str(Path(result["archive"]).parent), checksum)
            with tarfile.open(result["archive"], "r:gz") as stream:
                members = [member.name.lower() for member in stream.getmembers()]
            self.assertFalse(any(name.endswith((".pth", ".onnx")) for name in members))

    def test_formal_validator_enforces_three_locked_track_b_seeds(self):
        commit = "a" * 40
        provenance = {
            "source_tree_hash": "tree",
            "class_order_hash": "classes",
            "data_split_hash": {"task0": "split"},
            "core_base_commit": "core",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for seed in range(3):
                seed_root = root / "benchmarks" / "protocol" / "B" / "CocoER-FT" / f"seed{seed}"
                (seed_root / "metrics").mkdir(parents=True)
                method = dict(FORMAL_MODULE.LOCKED_METHOD_CONFIGURATION)
                method.update(FORMAL_MODULE.EXPECTED_ASSET_HASHES)
                manifest = {
                    "method": "CocoER-FT",
                    "protocol_id": "emotic_b5c3_track_b_v0.1",
                    "track": "B",
                    "seed": seed,
                    "git_commit": commit,
                    "git_dirty": False,
                    "core_runtime_version": "0.10.1",
                    "test_labels_used_for_selection": False,
                    "checkpoint_selection_split": "val",
                    "reporting_split": "test",
                    "configuration_locked": True,
                    "prediction_reused_task_ids": [],
                    "eligible_for_main_table": True,
                    "replay_memory_samples": 0,
                    "replay_memory_bytes": 0,
                    "total_parameters": 307509666,
                    "trainable_parameters": 269192770,
                    "incremental_parameters": 26985,
                    "protocol_hash": f"protocol-{seed}",
                    "method_configuration": method,
                    **provenance,
                }
                config = {
                    "runner": dict(FORMAL_MODULE.LOCKED_RUNNER_CONFIGURATION),
                    "protocol": {
                        "track": "B",
                        "seed": seed,
                        "threshold_policy": {"value": 0.5},
                    },
                }
                metrics = {
                    name: float(seed + index + 1)
                    for index, name in enumerate(FORMAL_MODULE.AGGREGATE_METRICS)
                }
                summary = {"main_table": {**metrics, "replay_memory": {"samples": 0, "bytes": 0}}}
                tasks = {"tasks": [{"task_id": task, "mAP": float(task + seed)} for task in range(8)]}
                for name, payload in (
                    ("run_manifest.json", manifest),
                    ("config_resolved.json", config),
                    ("metrics/summary.json", summary),
                    ("metrics/task_metrics.json", tasks),
                ):
                    path = seed_root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(payload), encoding="utf-8")
                (seed_root / "train.log").write_text("complete\n", encoding="utf-8")
            result = FORMAL_MODULE.validate(root, "formal", (0, 1, 2), (1, 2, 3), commit)
            self.assertEqual(result["status"], "eligible_for_main_table")
            self.assertEqual(result["aggregate"]["final_mAP"]["mean"], 2.0)
            self.assertEqual(result["aggregate"]["final_mAP"]["std"], 1.0)
            with self.assertRaisesRegex(ValueError, "distinct GPUs"):
                FORMAL_MODULE.validate(root, "formal", (0, 1, 2), (1, 1, 2), commit)

    def test_formal_launcher_requires_locked_test_and_memory_capacity(self):
        text = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        worker = (ROOT / "scripts" / "emotic-mlcil" / "run_cocoer_ft_formal_seed012.sh").read_text(encoding="utf-8")
        self.assertIn("COCOER_FT_TRACK_B_V0_1", text)
        self.assertIn("COCOER_MIN_FREE_MIB:-18000", text)
        self.assertIn("smoke_cocoer_ft_training.py", text)
        self.assertIn("REPORTING_SPLIT=test", worker)
        self.assertIn("TRAIN_BATCH_SIZE=64", worker)

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
            source.write_text(
                json.dumps(_detection_payload()), encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--detections", str(source),
                 "--output", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["upstream_commit"], MODULE.COMMIT)
            self.assertEqual(payload["schema_version"], 2)
            self.assertEqual(len(payload["entries"]), 2)
            self.assertEqual(payload["fallback_samples"], 1)
            self.assertEqual(
                payload["conversion"]["fallback_calibration_split"], "train"
            )
            self.assertEqual(
                payload["detector_model_tree_sha256"], BUFFALO_L_TREE_SHA256
            )

    def test_partial_head_detections_cannot_be_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "detections.json"
            output = directory / "cache.json"
            source.write_text(
                json.dumps(_detection_payload(partial=True)), encoding="utf-8"
            )
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--detections", str(source),
                 "--output", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Partial CocoER detections", result.stdout)
            self.assertFalse(output.exists())

    def test_freezer_rejects_provider_fallback_and_test_calibration(self):
        for mutate, message in (
            (
                lambda payload: payload["detector"].update(
                    {"actual_providers": ["CPUExecutionProvider"]}
                ),
                "actual ONNX provider",
            ),
            (
                lambda payload: payload["detector"][
                    "faceanalysis_equivalence"
                ].update(
                    {"faceanalysis_detector_actual_providers": [
                        "CPUExecutionProvider"
                    ]}
                ),
                "SCRFD-only equivalence",
            ),
            (
                lambda payload: payload["conversion"].update(
                    {"fallback_calibration_split": "test"}
                ),
                "conversion contract",
            ),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                source = directory / "detections.json"
                output = directory / "cache.json"
                payload = _detection_payload()
                mutate(payload)
                source.write_text(json.dumps(payload), encoding="utf-8")
                result = subprocess.run(
                    [sys.executable, str(PREPARE), "--detections", str(source),
                     "--output", str(output)],
                    text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stdout)
                self.assertFalse(output.exists())

    def test_freezer_rejects_inconsistent_sample_accounting(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / "detections.json"
            output = directory / "cache.json"
            payload = _detection_payload()
            payload["fallback_by_split"] = {"train": 0}
            source.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(PREPARE), "--detections", str(source),
                 "--output", str(output)],
                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("split totals", result.stdout)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
