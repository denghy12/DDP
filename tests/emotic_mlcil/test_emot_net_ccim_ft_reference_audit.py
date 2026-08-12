import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "emotic-mlcil" / "compare_ccim_upstream_reference.py"
PREPARE = ROOT / "scripts" / "emotic-mlcil" / "prepare_ccim_task0_dictionary.py"
PREPARE_SOURCE = ROOT / "scripts" / "emotic-mlcil" / "prepare_ccim_source.sh"
CONVERTER = ROOT / "scripts" / "emotic-mlcil" / "convert_places365_resnet152_caffe.py"
FORMAL_RUNNER = ROOT / "scripts" / "emotic-mlcil" / "run_emot_net_ccim_ft_formal_seed0.sh"
FORMAL_LAUNCHER = ROOT / "scripts" / "emotic-mlcil" / "launch_emot_net_ccim_ft_formal_seed0_tmux.sh"
FORMAL_VALIDATOR = ROOT / "scripts" / "emotic-mlcil" / "validate_emot_net_ccim_ft_formal_result.py"
SPEC = importlib.util.spec_from_file_location("compare_ccim_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EMOTNetCCIMFTReferenceAuditTest(unittest.TestCase):
    def test_direct_entrypoints_resolve_repository(self):
        for script, expected in (
            (SCRIPT, "--upstream-root"),
            (PREPARE, "--places365-checkpoint"),
            (CONVERTER, "--caffemodel"),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                result = subprocess.run(
                    [sys.executable, str(script), "--help"],
                    cwd=temporary,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn(expected, result.stdout)

    def test_source_preparation_script_pins_registered_identity(self):
        source = PREPARE_SOURCE.read_text(encoding="utf-8")
        self.assertIn(MODULE.CCIM_UPSTREAM_COMMIT, source)
        self.assertIn(MODULE.CCIM_SOURCE_SHA256, source)
        self.assertIn(MODULE.CCIM_UPSTREAM_REPOSITORY, source)

    def test_places365_converter_pins_official_assets_and_audits_pool5(self):
        source = CONVERTER.read_text(encoding="utf-8")
        self.assertIn("PLACES365_CAFFE_MODEL_SHA256", source)
        self.assertIn("PLACES365_PROTOTXT_SHA256", source)
        self.assertIn('net.forward("pool5")', source)
        self.assertIn("operator_equivalence", source)

    def test_formal_runner_is_locked_and_checkpoint_free(self):
        runner = FORMAL_RUNNER.read_text(encoding="utf-8")
        launcher = FORMAL_LAUNCHER.read_text(encoding="utf-8")
        validator = FORMAL_VALIDATOR.read_text(encoding="utf-8")
        self.assertIn("REPORTING_SPLIT=test", runner)
        self.assertIn("EMOT_NET_CCIM_TOWER_MODEL_PARALLEL=1", runner)
        self.assertIn("WORKERS=8", runner)
        self.assertIn('GPUS="${GPUS:-2,3,4}"', launcher)
        self.assertIn("--tower-model-parallel", launcher)
        self.assertIn("package_benchmark_download.py", runner)
        self.assertIn('"configuration_locked": True', validator)

    def test_fixed_external_source_and_operator_equivalence(self):
        configured = os.environ.get("CCIM_SOURCE_ROOT")
        candidates = (
            [Path(configured)]
            if configured
            else [
                Path("/Users/denghaoyuan/workspace/MyCode/baseline_sources/ccim_official"),
                Path("/mnt/haoyuan/workspace/baseline_sources/ccim_official"),
            ]
        )
        source = next((path for path in candidates if path.is_dir()), candidates[0])
        if not source.is_dir():
            self.skipTest("fixed external CCIM source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.CCIM_UPSTREAM_COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertFalse(payload["incremental_mapping"]["future_task_images_used"])
        self.assertFalse(payload["incremental_mapping"]["adapter"])
        self.assertLess(max(payload["operator_equivalence"].values()), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
