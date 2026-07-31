import inspect
import unittest

import torch
from torch import nn

from benchmarks.emotic_mlcil.method_base import BenchmarkMethod
from benchmarks.emotic_mlcil.methods.ddp import DDPBenchmarkMethod
from benchmarks.emotic_mlcil.registry import method_class, method_names
from benchmarks.emotic_mlcil.runner import BenchmarkRunner
from tests.emotic_mlcil import make_protocol, task_context


class FakeDDP(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, images, cls_id=None, inference=False):
        seen = cls_id[1] - cls_id[0]
        return self.weight.view(1, 2, 1).expand(images.shape[0], 2, seen)


class FakePromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx_pos = nn.Parameter(torch.ones(2))
        self.ctx_neg = nn.Parameter(torch.ones(3))


class FakeDDPWithLegacyOptimizerScope(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = FakePromptLearner()
        self.visual_prompts = nn.Parameter(torch.ones(5))
        # This deliberately remains requires_grad=True but is not registered by
        # DDP.build_optimizer_scheduler and must not be reported as trainable.
        self.unoptimized_weight = nn.Parameter(torch.ones(7))


class MethodInterfaceTest(unittest.TestCase):
    def test_interface_contains_all_required_methods(self):
        required = {
            "begin_task",
            "train_task",
            "predict_scores",
            "end_task",
            "save_checkpoint",
            "load_checkpoint",
            "parameter_statistics",
            "memory_statistics",
        }
        self.assertTrue(required.issubset(dict(inspect.getmembers(BenchmarkMethod))))
        self.assertTrue(issubclass(DDPBenchmarkMethod, BenchmarkMethod))
        self.assertFalse(inspect.isabstract(DDPBenchmarkMethod))

    def test_ddp_is_registered(self):
        self.assertIn("ddp", method_names())
        self.assertIs(method_class("DDP"), DDPBenchmarkMethod)

    def test_ddp_wrapper_keeps_exactly_the_injected_existing_model(self):
        protocol = make_protocol()
        fake = FakeDDP()
        method = DDPBenchmarkMethod(protocol, model=fake, device="cpu")
        self.assertIs(method.model, fake)
        self.assertEqual(
            sum(value is fake for value in vars(method).values()),
            1,
        )
        method.begin_task(task_context(protocol, 0))
        method.train_task([], [])
        self.assertEqual(method.memory_statistics().replay_memory_samples, 0)

    def test_ddp_parameter_statistics_match_legacy_optimizer_scope(self):
        protocol = make_protocol()
        method = DDPBenchmarkMethod(
            protocol,
            model=FakeDDPWithLegacyOptimizerScope(),
            device="cpu",
        )
        stats = method.parameter_statistics()
        self.assertEqual(stats.total_parameters, 17)
        self.assertEqual(stats.trainable_parameters, 10)
        self.assertEqual(stats.incremental_parameters, 0)
        self.assertEqual(
            stats.per_task_incremental_parameters,
            {task_id: 0 for task_id in range(protocol.num_tasks)},
        )

    def test_runner_refuses_test_before_configuration_lock(self):
        protocol = make_protocol()
        method = DDPBenchmarkMethod(protocol, model=FakeDDP(), device="cpu")
        runner = BenchmarkRunner(
            protocol,
            data_module=None,
            method=method,
            artifact_store=None,
            num_workers=0,
        )
        with self.assertRaisesRegex(RuntimeError, "lock_configuration"):
            runner.run(reporting_split=protocol.test_split)


if __name__ == "__main__":
    unittest.main()
