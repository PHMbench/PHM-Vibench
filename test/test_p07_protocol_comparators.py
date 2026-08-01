from __future__ import annotations

import unittest

import torch

from src.model_factory.X_model.UXFD.operator_attention import (
    ExecutableOperatorPath1D,
    ExecutableOperatorPathConfig,
)
from src.utils.p07_protocol.comparators import (
    COMPARATOR_SPECS,
    RAW_PATH_COUNT,
    DenseOperatorMixture1D,
    RandomDictionaryOperatorPath1D,
    assert_parameter_matched,
    enumerate_raw_paths,
    select_discrete_path,
)


class ComparatorProtocolTests(unittest.TestCase):
    @staticmethod
    def _cfg() -> ExecutableOperatorPathConfig:
        return ExecutableOperatorPathConfig(hidden_dim=8)

    def test_dense_mixture_changes_only_projection_and_is_dense(self) -> None:
        torch.manual_seed(3)
        sparse = ExecutableOperatorPath1D(2, self._cfg())
        dense = DenseOperatorMixture1D(2, self._cfg())
        dense.gates.load_state_dict(sparse.gates.state_dict())
        x = torch.randn(4, 64, 2)
        _, trace = dense.relaxed_forward(x)
        self.assertNotEqual(dense.dictionary_sha256, sparse.dictionary_sha256)
        self.assertEqual(dense.dictionary_manifest()["relaxation"], "softmax")
        for weights in trace.stage_weights:
            self.assertTrue(torch.all(weights[:, :6] > 0))
            self.assertTrue(torch.equal(weights[:, 6], torch.zeros(4)))
            torch.testing.assert_close(weights.sum(dim=1), torch.ones(4))

    def test_random_dictionary_is_seeded_distinct_and_rng_isolated(self) -> None:
        x = torch.linspace(-1.0, 1.0, 256).reshape(1, 128, 2)
        model_a = RandomDictionaryOperatorPath1D(
            2, random_dictionary_seed=701, cfg=self._cfg()
        )
        model_b = RandomDictionaryOperatorPath1D(
            2, random_dictionary_seed=701, cfg=self._cfg()
        )
        model_b.load_state_dict(model_a.state_dict())
        state = torch.random.get_rng_state().clone()
        out_a, _ = model_a.relaxed_forward(x)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        out_b, _ = model_b.relaxed_forward(x)
        torch.testing.assert_close(out_a, out_b)
        self.assertEqual(model_a.dictionary_sha256, model_b.dictionary_sha256)
        other = RandomDictionaryOperatorPath1D(
            2, random_dictionary_seed=709, cfg=self._cfg()
        )
        self.assertNotEqual(model_a.dictionary_sha256, other.dictionary_sha256)

    def test_discrete_search_registry_prefix_and_tie_rule(self) -> None:
        paths = enumerate_raw_paths()
        self.assertEqual(len(paths), RAW_PATH_COUNT)
        losses = {path: 1.0 for path in paths[:12]}
        result = select_discrete_path(losses, evaluation_budget=12)
        self.assertEqual(result.selected_path, paths[0])
        losses[paths[7]] = 0.25
        result = select_discrete_path(losses, evaluation_budget=12)
        self.assertEqual(result.selected_path, paths[7])

    def test_comparator_registry_has_exact_contract_arms(self) -> None:
        self.assertEqual(
            {spec.comparator_id for spec in COMPARATOR_SPECS},
            {
                "dense_operator_mixture",
                "discrete_search",
                "feature_attention",
                "parameter_matched_black_box",
                "random_dictionary",
            },
        )
        path_arms = {spec.comparator_id for spec in COMPARATOR_SPECS if spec.path_producing}
        self.assertEqual(path_arms, {"dense_operator_mixture", "discrete_search"})

    def test_parameter_match_is_fail_closed(self) -> None:
        assert_parameter_matched(2938, 2913, maximum_relative_gap=0.05)
        with self.assertRaises(ValueError):
            assert_parameter_matched(4000, 2913, maximum_relative_gap=0.05)
        with self.assertRaises(TypeError):
            assert_parameter_matched(True, 2913)


if __name__ == "__main__":
    unittest.main()
