import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "emotic-mlcil" / "compare_emot_net_upstream_reference.py"
SPEC = importlib.util.spec_from_file_location("compare_emot_net_upstream", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

PREPARE_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "emotic-mlcil"
    / "prepare_emot_net_native_initialization.py"
)
PREPARE_SPEC = importlib.util.spec_from_file_location(
    "prepare_emot_net_native", PREPARE_SCRIPT
)
PREPARE = importlib.util.module_from_spec(PREPARE_SPEC)
PREPARE_SPEC.loader.exec_module(PREPARE)


class FakeTorchObject:
    def __init__(self, typename, modules=(), weight=None, bias=None):
        self._typename = typename.encode("utf-8")
        self.modules = list(modules)
        if weight is not None:
            self.weight = np.asarray(weight, dtype=np.float32)
        if bias is not None:
            self.bias = np.asarray(bias, dtype=np.float32)


class EMOTNetFTReferenceAuditTest(unittest.TestCase):
    def test_fixed_external_source_and_operator_contract(self):
        configured = os.environ.get("EMOT_NET_UPSTREAM_ROOT")
        candidates = (
            [Path(configured)]
            if configured
            else [
                Path(
                    "/Users/denghaoyuan/workspace/MyCode/baseline_sources/"
                    "emot_net_release_69c3a51"
                ),
                Path(
                    "/mnt/haoyuan/workspace/baseline_sources/"
                    "emot_net_release_69c3a51"
                ),
            ]
        )
        source = next(
            (candidate for candidate in candidates if candidate.is_dir()),
            candidates[0],
        )
        if not source.is_dir():
            self.skipTest("fixed external EMOT-Net source is not available")
        payload = MODULE.compare(source)
        self.assertEqual(payload["upstream"]["commit"], MODULE.COMMIT)
        self.assertEqual(payload["upstream"]["license"], "MIT")
        self.assertFalse(payload["upstream"]["source_copied_into_repository"])
        self.assertFalse(payload["conversion"]["clip_used"])
        release = payload["official_release"]
        self.assertEqual(release["body_variant"], "AlexNet, 256-D")
        self.assertEqual(release["fusion_input_dim"], 896)
        self.assertEqual(release["epochs"], 21)
        self.assertAlmostEqual(release["discrete_loss_weight"], 1.0 / 6.0)
        self.assertEqual(
            payload["upstream"]["registered_release_native_body_asset"],
            "alexnet_features.t7",
        )
        self.assertLess(max(payload["operator_equivalence"].values()), 1.0e-12)

    def test_torch7_mapping_and_identical_replica_selection(self):
        def tower(offset=0.0):
            convolution = FakeTorchObject(
                "cudnn.SpatialConvolution",
                weight=np.full((2, 1, 3, 3), offset, dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            )
            return FakeTorchObject("nn.Sequential", modules=(convolution,))

        first = tower()
        second = tower()
        root = {b"model": FakeTorchObject("nn.DataParallel", modules=(first, second))}
        convolutions, batch_norms, digests = PREPARE._find_tower(
            root, [(2, 1, 3, 3)], 0, "test"
        )
        self.assertEqual(len(convolutions), 1)
        self.assertEqual(batch_norms, [])
        self.assertEqual(len(digests), 2)
        self.assertEqual(digests[0], digests[1])
        self.assertEqual(PREPARE._typename(convolutions[0]), "cudnn.SpatialConvolution")

        divergent = {
            b"model": FakeTorchObject(
                "nn.DataParallel", modules=(tower(), tower(offset=1.0))
            )
        }
        selected, _, divergent_digests = PREPARE._find_tower(
            divergent, [(2, 1, 3, 3)], 0, "test"
        )
        self.assertEqual(float(selected[0].weight[0, 0, 0, 0]), 0.0)
        self.assertNotEqual(divergent_digests[0], divergent_digests[1])


if __name__ == "__main__":
    unittest.main()
