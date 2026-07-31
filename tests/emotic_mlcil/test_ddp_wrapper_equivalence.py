import inspect
import json
import os
import re
import unittest
from pathlib import Path

import torch
from torch import nn

from benchmarks.emotic_mlcil.data_module import EMOTICMLCILDataModule
from benchmarks.emotic_mlcil.evaluator import (
    BenchmarkEvaluator,
    assert_prediction_equivalence,
)
from benchmarks.emotic_mlcil.methods.ddp import (
    DDPBenchmarkMethod,
    legacy_ddp_predict_scores,
)
from benchmarks.emotic_mlcil.protocol import load_protocol
from benchmarks.emotic_mlcil.types import EvaluationBatch
from tests.emotic_mlcil import make_protocol, task_context


class FakeLegacyDDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, images, cls_id=None, inference=False):
        seen = cls_id[1] - cls_id[0]
        positive = images.reshape(images.shape[0], -1)[:, :seen] + self.anchor
        return torch.stack([-positive, positive], dim=1)


def fake_loader(protocol):
    targets = torch.tensor(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=torch.float32,
    )
    images = torch.tensor(
        [
            [0.9, 0.1, 0.2, 0.3, 0.4],
            [0.2, 0.8, 0.1, 0.3, 0.4],
            [0.1, 0.2, 0.7, 0.3, 0.4],
        ],
        dtype=torch.float32,
    )
    return [
        EvaluationBatch(
            images=images[:2],
            sample_ids=["sample-0", "sample-1"],
            targets_seen=targets[:2],
            class_order_hash=protocol.class_order_hash,
            split_hash="same-loader",
        ),
        EvaluationBatch(
            images=images[2:],
            sample_ids=["sample-2"],
            targets_seen=targets[2:],
            class_order_hash=protocol.class_order_hash,
            split_hash="same-loader",
        ),
    ]


class DDPWrapperEquivalenceTest(unittest.TestCase):
    def test_wrapper_and_legacy_formula_are_strictly_equivalent(self):
        protocol = make_protocol()
        model = FakeLegacyDDP()
        method = DDPBenchmarkMethod(protocol, model=model, device="cpu")
        context = task_context(protocol, 0)
        method.begin_task(context)
        benchmark = method.predict_scores(fake_loader(protocol))
        legacy = legacy_ddp_predict_scores(
            model,
            fake_loader(protocol),
            context,
            device="cpu",
            temperature=method.current_temperature,
        )
        max_error = assert_prediction_equivalence(
            legacy,
            benchmark,
            score_atol=1e-7,
        )
        self.assertEqual(max_error, 0.0)

        evaluator = BenchmarkEvaluator(protocol)
        benchmark_metrics = evaluator.evaluate_task(
            0,
            benchmark,
            benchmark.sample_ids,
            benchmark.split_hash,
        )
        legacy_metrics = evaluator.evaluate_task(
            0,
            legacy,
            legacy.sample_ids,
            legacy.split_hash,
        )
        self.assertEqual(
            legacy_metrics.per_class_ap,
            benchmark_metrics.per_class_ap,
        )
        self.assertEqual(legacy_metrics.mAP, benchmark_metrics.mAP)

    def test_wrapper_does_not_define_or_create_a_second_ddp_model(self):
        source = inspect.getsource(
            __import__(
                "benchmarks.emotic_mlcil.methods.ddp.method",
                fromlist=["DDPBenchmarkMethod"],
            )
        )
        self.assertIsNone(re.search(r"class\s+DDP\s*\(", source))
        model = FakeLegacyDDP()
        method = DDPBenchmarkMethod(make_protocol(), model=model, device="cpu")
        self.assertIs(method.model, model)
        self.assertEqual(
            sum(value is model for value in vars(method).values()),
            1,
        )


_REAL_ENV = (
    "EMOTIC_DDP_CHECKPOINT",
    "EMOTIC_DATA_ROOT",
    "EMOTIC_CLIP_MODEL_PATH",
)


@unittest.skipUnless(
    all(os.environ.get(name) for name in _REAL_ENV),
    "Set EMOTIC_DDP_CHECKPOINT, EMOTIC_DATA_ROOT, and "
    "EMOTIC_CLIP_MODEL_PATH for strict server equivalence",
)
class RealDDPWrapperIntegrationTest(unittest.TestCase):
    def test_real_checkpoint_same_loader_equivalence(self):
        from train_emotic_ddp_internal_adapter import eval_transform

        project_root = Path(__file__).resolve().parents[2]
        protocol = load_protocol(
            project_root
            / "configs"
            / "emotic_mlcil"
            / "protocol_b5c3.yaml"
        )
        task_id = int(os.environ.get("EMOTIC_DDP_TASK_ID", "7"))
        split = os.environ.get("EMOTIC_DDP_SPLIT", protocol.test_split)
        workers = int(os.environ.get("EMOTIC_DDP_WORKERS", "0"))
        batch_size = int(os.environ.get("EMOTIC_DDP_BATCH_SIZE", "1"))
        device = os.environ.get(
            "EMOTIC_DDP_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        if device == "cuda" and torch.cuda.device_count() > 1:
            free_memory = [
                torch.cuda.mem_get_info(device_id)[0]
                for device_id in range(torch.cuda.device_count())
            ]
            device = f"cuda:{max(range(len(free_memory)), key=free_memory.__getitem__)}"
        if torch.device(device).type == "cuda":
            torch.cuda.set_device(torch.device(device))
        transform = eval_transform()
        data_module = EMOTICMLCILDataModule(
            protocol,
            data_root=os.environ["EMOTIC_DATA_ROOT"],
            train_transform=transform,
            eval_transform=transform,
            input_mode="full",
        )
        evaluator = BenchmarkEvaluator(protocol)
        dataset = data_module.evaluator_dataset(
            task_id,
            split,
            evaluator.access_token,
        )
        method = DDPBenchmarkMethod(
            protocol,
            checkpoint_paths={
                task_id: os.environ["EMOTIC_DDP_CHECKPOINT"]
            },
            clip_model_path=os.environ["EMOTIC_CLIP_MODEL_PATH"],
            device=device,
        )
        context = task_context(protocol, task_id)
        method.begin_task(context)
        benchmark_loader = data_module.evaluator_loader(
            task_id,
            split,
            evaluator.access_token,
            batch_size=batch_size,
            num_workers=workers,
        )
        benchmark = method.predict_scores(benchmark_loader)
        del benchmark_loader
        legacy_loader = data_module.evaluator_loader(
            task_id,
            split,
            evaluator.access_token,
            batch_size=batch_size,
            num_workers=workers,
        )
        legacy = legacy_ddp_predict_scores(
            method.model,
            legacy_loader,
            context,
            device=device,
            temperature=method.current_temperature,
        )
        max_error = assert_prediction_equivalence(
            legacy,
            benchmark,
            score_atol=1e-7,
        )
        benchmark_metrics = evaluator.evaluate_task(
            task_id,
            benchmark,
            dataset.sample_ids,
            dataset.split_hash,
        )
        legacy_metrics = evaluator.evaluate_task(
            task_id,
            legacy,
            dataset.sample_ids,
            dataset.split_hash,
        )
        self.assertEqual(
            legacy_metrics.per_class_ap,
            benchmark_metrics.per_class_ap,
        )
        self.assertEqual(legacy_metrics.mAP, benchmark_metrics.mAP)
        print(
            json.dumps(
                {
                    "task": task_id,
                    "split": split,
                    "device": device,
                    "batch_size": batch_size,
                    "samples": len(dataset),
                    "max_abs_error": max_error,
                    "legacy_mAP": legacy_metrics.mAP,
                    "benchmark_mAP": benchmark_metrics.mAP,
                    "sample_ids_equal": legacy.sample_ids
                    == benchmark.sample_ids,
                    "targets_equal": torch.equal(
                        legacy.targets, benchmark.targets
                    ),
                    "per_class_ap_equal": legacy_metrics.per_class_ap
                    == benchmark_metrics.per_class_ap,
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    unittest.main()
