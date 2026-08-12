import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class BENetFTReferenceAuditTest(unittest.TestCase):
    def test_direct_script_entrypoints_resolve_repository(self):
        for name, option in (
            ("compare_benet_upstream_reference.py", "--source-root"),
            ("smoke_benet_ft_training.py", "--pretrained-weights"),
        ):
            script = ROOT / "scripts" / "emotic-mlcil" / name
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
            self.assertIn(option, result.stdout)

    def test_fixed_source_and_download_standard_are_registered(self):
        comparison = (ROOT / "scripts/emotic-mlcil/compare_benet_upstream_reference.py").read_text()
        launcher = (ROOT / "scripts/emotic-mlcil/launch_benet_ft_seed0_tmux.sh").read_text()
        document = (ROOT / "docs/benchmarks/BENET_FT_TRACK_B.md").read_text()
        self.assertIn("b86747e0e259b1ec70fc84ca76efd7ea3bb3728e", comparison)
        self.assertIn("package_benchmark_download.py", launcher)
        self.assertIn("exclude", document.lower())
        self.assertIn(".pth", document)

    def test_smoke_expands_before_cuda_transfer(self):
        smoke = (ROOT / "scripts/emotic-mlcil/smoke_benet_ft_training.py").read_text()
        self.assertLess(smoke.index("model.add_head(5)"), smoke.index("model = model.cuda().train()"))
        self.assertIn('devices != {"cuda"}', smoke)


if __name__ == "__main__":
    unittest.main()
