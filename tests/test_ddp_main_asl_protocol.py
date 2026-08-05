import unittest

from opts import arg_parser


class DDPMainASLProtocolTest(unittest.TestCase):
    def test_original_two_way_bce_remains_default(self):
        args = arg_parser().parse_args([])
        self.assertEqual(args.ddp_classification_loss, "two_way_bce")
        self.assertEqual(args.loss_w, 0.03)

    def test_locked_asl_parameters_match_primary_protocol(self):
        args = arg_parser().parse_args(
            ["--ddp_classification_loss", "asl"]
        )
        self.assertEqual(args.ddp_classification_loss, "asl")
        self.assertEqual(args.ddp_asl_gamma_neg, 9.8)
        self.assertEqual(args.ddp_asl_gamma_pos, 0.0)
        self.assertEqual(args.ddp_asl_clip, 0.05)
        self.assertEqual(args.ddp_asl_eps, 1e-8)

    def test_loss_choice_rejects_unregistered_objective(self):
        with self.assertRaises(SystemExit):
            arg_parser().parse_args(
                ["--ddp_classification_loss", "weighted_bce"]
            )


if __name__ == "__main__":
    unittest.main()
