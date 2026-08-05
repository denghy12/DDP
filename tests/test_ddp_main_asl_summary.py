import unittest

from summarize_emotic_ddp_main_losses import forgetting_from_detail, mean_std


class DDPMainASLSummaryTest(unittest.TestCase):
    def test_forgetting_is_zero_when_ap_only_improves(self):
        rows = [
            {
                "class_task": 0,
                **{f"task{task}_ap": 0.1 + 0.01 * task for task in range(8)},
            }
        ]
        self.assertEqual(forgetting_from_detail(rows), 0.0)

    def test_forgetting_uses_best_historical_ap(self):
        rows = [
            {
                "class_task": 0,
                "task0_ap": 0.2,
                "task1_ap": 0.5,
                "task2_ap": 0.4,
                "task3_ap": 0.3,
                "task4_ap": 0.3,
                "task5_ap": 0.3,
                "task6_ap": 0.3,
                "task7_ap": 0.25,
            },
            {"class_task": 7, "task7_ap": 0.8},
        ]
        self.assertAlmostEqual(forgetting_from_detail(rows), 25.0)

    def test_reported_std_is_population_std(self):
        stats = mean_std([1.0, 2.0, 3.0])
        self.assertEqual(stats["mean"], 2.0)
        self.assertAlmostEqual(stats["std"], (2.0 / 3.0) ** 0.5)


if __name__ == "__main__":
    unittest.main()
