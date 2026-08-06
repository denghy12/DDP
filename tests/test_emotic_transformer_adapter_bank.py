import unittest

import torch

from emotic_transformer_adapter_bank import (
    P2L_CA_LAYER_INDICES,
    P2L_CA_LAYER_NUMBERS,
    TaskRoutedTransformerAdapterBank,
    TransformerBlockAdapter,
    TransformerTaskAdapter,
    path_task_ids_for_classes,
)


class TransformerAdapterBankTest(unittest.TestCase):
    def test_p2l_ca_best_depth_is_layers_4_through_12(self):
        self.assertEqual(P2L_CA_LAYER_NUMBERS, tuple(range(4, 13)))
        self.assertEqual(P2L_CA_LAYER_INDICES, tuple(range(3, 12)))

    def test_zero_up_projection_is_exact_identity_branch(self):
        adapter = TransformerBlockAdapter(768, 128)
        tokens = torch.randn(5, 3, 768)
        self.assertTrue(torch.equal(adapter(tokens), torch.zeros_like(tokens)))

    def test_parameter_count_matches_nine_768_128_768_adapters(self):
        adapter = TransformerTaskAdapter()
        self.assertEqual(adapter.parameter_count, 1_777_536)

    def test_class_paths_are_routed_per_sample_and_per_sign(self):
        routed = path_task_ids_for_classes(
            [0, 5, 8], batch_size=2, device=torch.device("cpu")
        )
        expected = torch.tensor(
            [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
        )
        self.assertTrue(torch.equal(routed, expected))

    def test_bank_dispatches_each_path_to_its_task_adapter(self):
        task0 = TransformerTaskAdapter()
        task1 = TransformerTaskAdapter()
        with torch.no_grad():
            task0.blocks["3"].up.bias.fill_(1.0)
            task1.blocks["3"].up.bias.fill_(2.0)
        bank = TaskRoutedTransformerAdapterBank({0: task0, 1: task1})
        tokens = torch.randn(2, 4, 768)
        result = bank.delta_for_layer(3, tokens, torch.tensor([0, 1, 0, 1]))
        self.assertTrue(torch.equal(result[:, 0], torch.ones_like(result[:, 0])))
        self.assertTrue(torch.equal(result[:, 2], torch.ones_like(result[:, 2])))
        self.assertTrue(
            torch.equal(result[:, 1], torch.full_like(result[:, 1], 2.0))
        )
        self.assertTrue(
            torch.equal(result[:, 3], torch.full_like(result[:, 3], 2.0))
        )

    def test_unselected_layer_has_no_effect(self):
        bank = TaskRoutedTransformerAdapterBank({0: TransformerTaskAdapter()})
        tokens = torch.randn(2, 1, 768)
        result = bank.delta_for_layer(2, tokens, torch.tensor([0]))
        self.assertTrue(torch.equal(result, torch.zeros_like(tokens)))


if __name__ == "__main__":
    unittest.main()
