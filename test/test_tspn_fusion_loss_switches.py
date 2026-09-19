"""CPU float64 checks for the fixed O/UO/RO/RC objective interventions."""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.task_factory.Components.tspn_fusion_loss import TSPNFusionLoss, domain_means


ARMS = {
    "O": ("mean_source", "relative", "correction", 0.),
    "UO": ("worst_source", "absolute", "candidate", .1),
    "RO": ("worst_source", "relative", "candidate", .1),
    "RC": ("worst_source", "relative", "correction", .1),
}


def _output(candidate, reference):
    return {
        "candidate_logits": candidate,
        "raw_logits": reference,
        "candidate_probs": candidate.softmax(-1),
        "raw_probs": (reference/1.3).softmax(-1),
        "reference_temperature": 1.3,
    }


def _fixture():
    generator = torch.Generator().manual_seed(20260919)
    logits = [torch.randn(7, 3, generator=generator, dtype=torch.float64,
                          requires_grad=True) for _ in range(4)]
    left, right = _output(logits[0], logits[1]), _output(logits[2], logits[3])
    labels = torch.tensor([0, 1, 2, 0, 1, 0, 2])
    groups = torch.tensor([0, 0, 1, 2, 3, 3, 4])
    domains = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    return left, right, labels, groups, domains


def _evaluate(objective, left, right, labels, groups, domains):
    ids = torch.arange(len(labels))
    return objective(left, labels, groups, domains, paired=right,
                     paired_target=labels, sample_ids=ids, paired_sample_ids=ids)


def _manual_domain_means(values, groups, domains):
    # Deliberately use indexing/stacking, independently of scatter-based reduction.
    return torch.stack([
        torch.stack([values[(domains == domain) & (groups == group)].mean()
                     for group in groups[domains == domain].unique()]).mean()
        for domain in domains.unique()
    ])


def _manual_terms(left, right, labels, groups, domains):
    onehot = F.one_hot(labels, 3)

    def score(probs):
        return -probs[torch.arange(len(labels)), labels].log() + .25*(probs-onehot).square().sum(-1)

    q, qp = left["candidate_probs"], right["candidate_probs"]
    p0, p0p = left["raw_probs"].detach(), right["raw_probs"].detach()
    risk = _manual_domain_means((score(q)+score(qp))/2, groups, domains)
    reference = _manual_domain_means((score(p0)+score(p0p))/2, groups, domains)
    output_penalty = _manual_domain_means((qp-q).square().sum(-1), groups, domains).mean()
    correction_penalty = _manual_domain_means(((qp-p0p)-(q-p0)).square().sum(-1), groups, domains).mean()
    return risk, reference, output_penalty, correction_penalty


def _legacy_formula(left, right, labels, groups, domains, tau, reduction):
    """Numerical formula from the pre-switch objective, with its operation order."""
    def excess(out):
        log0 = F.log_softmax(out["raw_logits"].detach()/float(out["reference_temperature"]), -1)
        logq = F.log_softmax(out["candidate_logits"], -1)
        logp = logq if tau == 1 else torch.logaddexp(log0+math.log1p(-tau), logq+math.log(tau))
        p0, p = log0.exp(), logp.exp()
        onehot = F.one_hot(labels, 3).to(logp.dtype)
        b0, bp = (p0-onehot).square().sum(-1), (p-onehot).square().sum(-1)
        ce0 = F.nll_loss(log0, labels, reduction="none")
        cep = F.nll_loss(logp, labels, reduction="none")
        return cep-ce0+.25*(bp-b0)

    _, risks = domain_means(.5*(excess(left)+excess(right)), groups, domains)
    v = left["candidate_probs"]-left["raw_probs"].detach()
    vp = right["candidate_probs"]-right["raw_probs"].detach()
    _, delta = domain_means((v-vp).square().sum(-1), groups, domains)
    if reduction == "worst_source":
        robust = .25*(torch.logsumexp(risks/.25, dim=0)-math.log(len(risks)))
        weights = F.softmax(risks.detach()/.25, dim=0)
    else:
        robust = risks.mean()
        weights = torch.full_like(risks, 1/len(risks))
    return robust+.1*delta.mean(), weights


@pytest.mark.parametrize("tau", [1., .4])
@pytest.mark.parametrize("reduction", ["worst_source", "mean_source"])
def test_legacy_defaults_exact_loss_gradients_weights_and_outputs(tau, reduction):
    left, right, labels, groups, domains = _fixture()
    before = [{key: value.detach().clone() for key, value in out.items()
               if isinstance(value, torch.Tensor)} for out in (left, right)]
    # Legacy positional arguments remain valid; switches are deliberately omitted.
    objective = TSPNFusionLoss(tau, .25, .25, .1, reduction)
    actual = _evaluate(objective, left, right, labels, groups, domains)
    expected, weights = _legacy_formula(left, right, labels, groups, domains, tau, reduction)
    assert torch.equal(actual["loss"], expected)
    assert torch.equal(actual["domain_weights"], weights)
    candidates = (left["candidate_logits"], right["candidate_logits"])
    actual_grad = torch.autograd.grad(actual["loss"], candidates, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, candidates)
    assert all(torch.equal(a, b) for a, b in zip(actual_grad, expected_grad))
    for out, original in zip((left, right), before):
        assert all(torch.equal(out[key], value) for key, value in original.items())


@pytest.mark.parametrize("arm", ARMS)
def test_four_arms_match_independent_float64_formula(arm):
    left, right, labels, groups, domains = _fixture()
    reduction, reference, consistency, coefficient = ARMS[arm]
    objective = TSPNFusionLoss(reduction=reduction, risk_reference=reference,
                               consistency_target=consistency, lambda_delta=coefficient)
    actual = _evaluate(objective, left, right, labels, groups, domains)
    risk, raw, output_penalty, correction_penalty = _manual_terms(left, right, labels, groups, domains)
    selected = risk if reference == "absolute" else risk-raw
    robust = selected.mean() if reduction == "mean_source" else .25*torch.log(torch.exp(selected/.25).mean())
    penalty = correction_penalty if consistency == "correction" else output_penalty
    expected = robust+coefficient*penalty
    for value, wanted in (
        (actual["loss"], expected), (actual["risk_objective"], robust),
        (actual["domain_candidate_risk"], risk), (actual["domain_reference_risk"], raw),
        (actual["domain_excess"], risk-raw), (actual["pair_penalty"], penalty),
        (actual["candidate_output_consistency"], output_penalty),
        (actual["correction_consistency"], correction_penalty),
    ):
        torch.testing.assert_close(value, wanted, rtol=1e-12, atol=1e-12)
    candidates = (left["candidate_logits"], right["candidate_logits"])
    actual_grad = torch.autograd.grad(actual["loss"], candidates, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, candidates, retain_graph=True)
    for value, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(value, wanted, rtol=1e-12, atol=1e-12)
        assert value.abs().sum() > 0
    # Frozen references cannot receive gradients through either risk or penalty.
    assert torch.autograd.grad(actual["loss"], (left["raw_logits"], right["raw_logits"]),
                               allow_unused=True) == (None, None)


def test_equal_domain_reference_risk_preserves_tilted_candidate_gradient():
    left, right, labels, groups, domains = _fixture()
    left = _output(left["candidate_logits"], torch.zeros_like(left["raw_logits"]))
    right = _output(right["candidate_logits"], torch.zeros_like(right["raw_logits"]))
    results = [_evaluate(TSPNFusionLoss(risk_reference=reference, consistency_target="candidate"),
                         left, right, labels, groups, domains)
               for reference in ("relative", "absolute")]
    relative, absolute = results
    raw = relative["domain_reference_risk"]
    torch.testing.assert_close(raw, raw[0].expand_as(raw), rtol=0, atol=0)
    torch.testing.assert_close(relative["domain_weights"], absolute["domain_weights"], rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(absolute["loss"]-relative["loss"], raw[0], rtol=1e-12, atol=1e-12)
    candidates = (left["candidate_logits"], right["candidate_logits"])
    gradients = [torch.autograd.grad(result["loss"], candidates, retain_graph=True) for result in results]
    for relative_grad, absolute_grad in zip(*gradients):
        torch.testing.assert_close(relative_grad, absolute_grad, rtol=1e-12, atol=1e-12)


def test_unchanged_reference_response_preserves_pair_penalty_and_gradient():
    left, right, labels, groups, domains = _fixture()
    right = _output(right["candidate_logits"], left["raw_logits"])
    results = [_evaluate(TSPNFusionLoss(consistency_target=target), left, right, labels, groups, domains)
               for target in ("correction", "candidate")]
    assert results[0]["reference_output_consistency"] == 0
    torch.testing.assert_close(results[0]["pair_penalty"], results[1]["pair_penalty"], rtol=1e-12, atol=1e-12)
    candidates = (left["candidate_logits"], right["candidate_logits"])
    gradients = [torch.autograd.grad(result["pair_penalty"], candidates, retain_graph=True) for result in results]
    for correction_grad, candidate_grad in zip(*gradients):
        torch.testing.assert_close(correction_grad, candidate_grad, rtol=1e-12, atol=1e-12)


def test_nondegenerate_switches_change_weights_and_penalty_gradient():
    left, right, labels, groups, domains = _fixture()
    relative = _evaluate(TSPNFusionLoss(), left, right, labels, groups, domains)
    absolute = _evaluate(TSPNFusionLoss(risk_reference="absolute", consistency_target="candidate"),
                         left, right, labels, groups, domains)
    assert not torch.allclose(relative["domain_reference_risk"][0], relative["domain_reference_risk"][1])
    assert not torch.allclose(relative["domain_weights"], absolute["domain_weights"])
    assert relative["reference_output_consistency"] > 0
    assert not torch.allclose(relative["pair_penalty"], absolute["pair_penalty"])
    # The correction diagnostic must retain its meaning in a candidate-penalty arm.
    assert torch.equal(relative["correction_consistency"], absolute["correction_consistency"])
    candidates = (left["candidate_logits"], right["candidate_logits"])
    correction_grad = torch.autograd.grad(relative["pair_penalty"], candidates, retain_graph=True)
    candidate_grad = torch.autograd.grad(absolute["pair_penalty"], candidates)
    assert all(not torch.allclose(a, b) for a, b in zip(correction_grad, candidate_grad))


@pytest.mark.parametrize("risk_reference", ["relative", "absolute"])
def test_lambda_zero_retains_supervision_at_both_endpoints(risk_reference):
    left, right, labels, groups, domains = _fixture()
    objective = TSPNFusionLoss(lambda_delta=0, reduction="mean_source", risk_reference=risk_reference)
    actual = _evaluate(objective, left, right, labels, groups, domains)
    expected = .5*(objective(left, labels, groups, domains)["loss"] +
                   objective(right, labels, groups, domains)["loss"])
    torch.testing.assert_close(actual["loss"], expected, rtol=1e-12, atol=1e-12)
    gradients = torch.autograd.grad(actual["loss"], (left["candidate_logits"], right["candidate_logits"]))
    assert all(gradient.abs().sum() > 0 for gradient in gradients)


@pytest.mark.parametrize("options,field", [({"risk_reference": "unknown"}, "risk_reference"),
                                          ({"consistency_target": "unknown"}, "consistency_target")])
def test_invalid_intervention_values_fail(options, field):
    with pytest.raises(ValueError, match=field):
        TSPNFusionLoss(**options)


def test_new_switches_are_keyword_only():
    with pytest.raises(TypeError):
        TSPNFusionLoss(1., .25, .25, .1, "worst_source", "absolute", "candidate")
