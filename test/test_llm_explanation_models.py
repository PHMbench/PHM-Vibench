"""Integration tests use the retained model implementations, not fabricated traces."""
import json

import pytest
import torch

from phmfactory.explanation import (
    build_llm_packet, state_from_tspn_uxfd_fuzzy_trace,
)
from test.test_tspn_uxfd_assembly import TSPNUXFD, _make_args




@pytest.mark.parametrize("max_rules", [None, 1])
def test_fuzzy_state_matches_real_same_forward_output(max_rules):
    torch.manual_seed(9)
    model = TSPNUXFD(_make_args(enable_fuzzy=True, fuzzy_logit_scale=0.7)).eval()
    with torch.no_grad():
        output = model.forward_with_fuzzy_trace(torch.randn(2, 128, 2))
        state = state_from_tspn_uxfd_fuzzy_trace(output, sample_id="fuzzy-check", sample_index=1, max_rules=max_rules)
    assert state.prediction.logits == pytest.approx(output.logits[1].tolist(), abs=1e-6)
    if max_rules is None:
        for i in range(output.logits.shape[1]):
            value = sum(c.value for c in state.contributions if c.target_label == f"class_{i}")
            assert value == pytest.approx(output.logits[1, i].item(), abs=1e-6)
        assert "decision_reconstruction" in state.capabilities
    else:
        assert "partial_contribution" in state.capabilities
        assert "decision_reconstruction" not in state.capabilities


@pytest.mark.parametrize("class_names", [("fault", "fault"), ("fault", " fault ")])
def test_duplicate_class_labels_are_rejected(class_names):
    model = TSPNUXFD(_make_args(enable_fuzzy=True, num_classes=2)).eval()
    with torch.no_grad():
        output = model.forward_with_fuzzy_trace(torch.randn(2, 128, 2))
    with pytest.raises(ValueError, match="unique labels"):
        state_from_tspn_uxfd_fuzzy_trace(output, sample_id="duplicate", class_names=class_names)


@pytest.mark.parametrize("per_sample", [False, True])
def test_fuzzy_intervention_exports_the_selected_effective_values(per_sample):
    torch.manual_seed(19)
    model = TSPNUXFD(_make_args(enable_fuzzy=True)).eval()
    x = torch.randn(2, 128, 2)
    with torch.no_grad():
        original = model.forward_with_fuzzy_trace(x)
        rules = original.fuzzy_trace.rule_mask.shape[1]
        mask = torch.ones(2, rules, dtype=torch.bool)
        mask[1, 0] = False
        permutation = torch.arange(rules - 1, -1, -1)
        if per_sample:
            permutation = torch.stack([torch.arange(rules), permutation])
        output = model.forward_with_fuzzy_trace(x, rule_mask=mask, consequent_permutation=permutation)
        state = state_from_tspn_uxfd_fuzzy_trace(output, sample_id="intervention", sample_index=1)
    metadata = state.to_dict()["metadata"]
    assert metadata["rule_mask"] == output.fuzzy_trace.rule_mask[1].tolist()
    expected = output.fuzzy_trace.consequent_permutation
    if per_sample:
        expected = expected[1]
    assert metadata["consequent_permutation"] == expected.tolist()
    assert state.prediction.logits == pytest.approx(output.logits[1].tolist(), abs=1e-6)
    original_state = state_from_tspn_uxfd_fuzzy_trace(original, sample_id="original", sample_index=1)
    original_metadata = original_state.to_dict()["metadata"]
    assert original_metadata["rule_mask"] == [True] * rules
    assert original_metadata["consequent_permutation"] == list(range(rules))
