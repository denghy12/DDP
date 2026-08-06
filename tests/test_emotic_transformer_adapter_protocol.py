import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TransformerAdapterProtocolTest(unittest.TestCase):
    def test_trainer_uses_asl_and_never_constructs_test_split(self):
        source = (ROOT / "train_emotic_ddp_transformer_adapter.py").read_text()
        self.assertIn('"adapter_loss": "asl"', source)
        self.assertIn('"ddp_main_loss": "two_way_bce_frozen_checkpoint"', source)
        self.assertNotIn('eval_splits=("test",)', source)
        self.assertIn('"test_dataset_constructed": False', source)
        self.assertIn('"validation_used_for_checkpoint_selection": False', source)

    def test_evaluator_locks_threshold_and_has_no_validation_sweep(self):
        source = (ROOT / "eval_emotic_ddp_transformer_adapter_bank.py").read_text()
        self.assertIn('default=0.5', source)
        self.assertIn('"test_used_for_selection": False', source)
        self.assertNotIn("select_threshold", source)
        self.assertNotIn('eval_splits=("val",)', source)

    def test_runner_locks_paper_reported_training_hyperparameters(self):
        source = (
            ROOT
            / "scripts/emotic-ddp-transformer-adapter-bank"
            / "run_train.sh"
        ).read_text()
        self.assertIn("--epochs 20", source)
        self.assertIn("--lr 4e-4", source)
        self.assertIn("--effective_batch_size 64", source)
        self.assertIn("--bottleneck_dim 128", source)
        self.assertIn("--asl_gamma_neg 9.8", source)


if __name__ == "__main__":
    unittest.main()
