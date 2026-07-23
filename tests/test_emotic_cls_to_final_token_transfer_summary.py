import unittest

from summarize_emotic_ddp_cls_to_final_token_transfer import (
    DIAGNOSTIC_METHODS,
    collect_structural_diagnostics,
)


def _distribution(value):
    return {
        "mean": float(value),
        "rms": float(value) + 0.5,
        "std": 0.1,
        "min": float(value) - 0.25,
        "max": float(value) + 0.75,
        "count": 10,
    }


class CLSToFinalTokenTransferSummaryTest(unittest.TestCase):
    def test_structural_diagnostics_are_aggregated_across_seeds(self):
        method_task_sets = {}
        for method_index, method in enumerate(DIAGNOSTIC_METHODS):
            seed_task_sets = []
            for seed in range(3):
                task_rows = []
                for task_id in range(8):
                    value = 10 * method_index + seed + task_id
                    task_rows.append(
                        {
                            "structural_diagnostics": {
                                "test": {
                                    "attention_kl": _distribution(value),
                                    "patch_token_delta_l2": _distribution(
                                        value + 1
                                    ),
                                }
                            }
                        }
                    )
                seed_task_sets.append(task_rows)
            method_task_sets[method] = seed_task_sets

        rows = collect_structural_diagnostics(method_task_sets)
        self.assertEqual(len(rows), 2 * 8 * 2)
        row = next(
            item
            for item in rows
            if item["method"] == "all_tokens"
            and item["task"] == 0
            and item["metric"] == "attention_kl"
        )
        self.assertEqual(row["mean_across_seeds"], 1.0)
        self.assertAlmostEqual(
            row["std_across_seeds"],
            (2.0 / 3.0) ** 0.5,
        )
        self.assertEqual(row["minimum_across_seeds"], -0.25)
        self.assertEqual(row["maximum_across_seeds"], 2.75)

    def test_missing_diagnostics_are_rejected(self):
        method_task_sets = {
            method: [[{} for _ in range(8)] for _ in range(3)]
            for method in DIAGNOSTIC_METHODS
        }
        with self.assertRaisesRegex(ValueError, "missing structural"):
            collect_structural_diagnostics(method_task_sets)


if __name__ == "__main__":
    unittest.main()
