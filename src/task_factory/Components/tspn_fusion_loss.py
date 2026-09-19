"""Diagnostic risk training over observed source domains.

The trainable candidate is optimized with CE + beta*Brier on a nonzero-tau
mixture. Source risk may be absolute or relative to the frozen reference;
same-label pairs regularize either candidate outputs or reference corrections.
This training loss does not certify target accuracy, F1 or final mixture alpha.
"""
from __future__ import annotations
import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def per_unit_mean(value: Tensor, unit_ids: Tensor) -> Tensor:
    """Mean of windows within each unit, sorted by identity."""
    if value.ndim != 1 or unit_ids.shape != value.shape or unit_ids.dtype != torch.long or not value.numel():
        raise ValueError("Supply one value and a long unit ID per window.")
    _, inverse, counts = torch.unique(unit_ids, sorted=True, return_inverse=True, return_counts=True)
    return value.new_zeros(len(counts)).scatter_add_(0, inverse, value)/counts.to(value.dtype)


def domain_means(value: Tensor, unit_ids: Tensor, domain_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Within each domain, average windows within unit then units equally.

    A specimen observed in two source domains is one repeated physical specimen,
    even though two domain-specific means are required by this objective. This
    reduction does not make such observations independent for later assessment.
    """
    if domain_ids.shape != value.shape or domain_ids.dtype != torch.long or domain_ids.device != value.device:
        raise ValueError("Supply one long source-domain ID per window on the same device.")
    domains = torch.unique(domain_ids, sorted=True)
    return domains, torch.stack([per_unit_mean(value[domain_ids==d], unit_ids[domain_ids==d]).mean() for d in domains])


class TSPNFusionLoss(nn.Module):
    def __init__(self, tau: float = 1., brier_weight: float = .25,
                 domain_temperature: float = .25, lambda_delta: float = .1,
                 reduction: str = "worst_source", *,
                 risk_reference: str = "relative",
                 consistency_target: str = "correction"):
        super().__init__()
        values = [tau, brier_weight, domain_temperature, lambda_delta]
        if not all(map(math.isfinite, values)) or not 0 < tau <= 1 or min(brier_weight, lambda_delta) < 0 or domain_temperature <= 0:
            raise ValueError("Invalid nonzero tau, loss weights or source-domain temperature.")
        if reduction not in {"worst_source", "mean_source"}:
            raise ValueError("reduction must be worst_source or mean_source.")
        if risk_reference not in {"relative", "absolute"}:
            raise ValueError("risk_reference must be relative or absolute.")
        if consistency_target not in {"correction", "candidate"}:
            raise ValueError("consistency_target must be correction or candidate.")
        self.tau, self.brier_weight = float(tau), float(brier_weight)
        self.domain_temperature, self.lambda_delta = float(domain_temperature), float(lambda_delta)
        self.reduction = reduction
        self.risk_reference = risk_reference
        self.consistency_target = consistency_target

    def _terms(self, out: dict, target: Tensor) -> dict[str, Tensor]:
        raw_logits = out["raw_logits"].detach()
        logits = out["candidate_logits"]
        if target.dtype != torch.long or target.shape != (logits.shape[0],) or logits.shape != raw_logits.shape:
            raise ValueError("Mismatched logits and integer labels.")
        temperature = float(out["reference_temperature"])
        log0 = F.log_softmax(raw_logits/temperature, -1)
        logq = F.log_softmax(logits, -1)
        if self.tau == 1:
            logp = logq
        else:
            logp = torch.logaddexp(log0+math.log1p(-self.tau), logq+math.log(self.tau))
        p0, p = log0.exp(), logp.exp()
        onehot = F.one_hot(target, logits.shape[-1]).to(logits.dtype)
        b0, bp = (p0-onehot).square().sum(-1), (p-onehot).square().sum(-1)
        ce0, cep = F.nll_loss(log0, target, reduction="none"), F.nll_loss(logp, target, reduction="none")
        return {"ce": cep, "raw_ce": ce0, "brier": bp, "raw_brier": b0,
                "ce_excess": cep-ce0, "brier_excess": bp-b0,
                "excess": cep-ce0+self.brier_weight*(bp-b0)}

    def forward(self, out: dict, target: Tensor, unit_ids: Tensor, domain_ids: Tensor,
                paired: dict | None = None, paired_target: Tensor | None = None,
                *, sample_ids: Tensor | None = None,
                paired_sample_ids: Tensor | None = None) -> dict[str, Tensor]:
        terms = self._terms(out, target)
        consistency = terms["excess"].new_zeros(target.shape)
        candidate_consistency = torch.zeros_like(consistency)
        reference_consistency = torch.zeros_like(consistency)
        if paired is None:
            if self.lambda_delta > 0 or paired_target is not None or paired_sample_ids is not None:
                raise ValueError("Pair consistency requires explicit same-label paired observations.")
        else:
            if paired_target is None or not torch.equal(paired_target, target):
                raise ValueError("Paired inputs must retain labels.")
            # Labels alone cannot detect a permutation among equal-class windows.
            for identity in (sample_ids, paired_sample_ids):
                if identity is None or identity.dtype != torch.long or identity.shape != target.shape or identity.device != target.device:
                    raise ValueError("Paired observations require explicit sample identities on both sides.")
            if sample_ids.unique().numel() != target.numel() or not torch.equal(sample_ids, paired_sample_ids):
                raise ValueError("Paired sample identities must be unique and retain row correspondence.")
            right = self._terms(paired, paired_target)
            terms = {key: .5*(value+right[key]) for key,value in terms.items()}
            v = out["candidate_probs"]-out["raw_probs"].detach()
            vp = paired["candidate_probs"]-paired["raw_probs"].detach()
            consistency = (v-vp).square().sum(-1)
            candidate_consistency = (out["candidate_probs"]-paired["candidate_probs"]).square().sum(-1)
            reference_consistency = (out["raw_probs"].detach()-paired["raw_probs"].detach()).square().sum(-1)
        domains, excess = domain_means(terms["excess"], unit_ids, domain_ids)
        _, candidate_risk = domain_means(terms["ce"]+self.brier_weight*terms["brier"], unit_ids, domain_ids)
        _, reference_risk = domain_means(terms["raw_ce"]+self.brier_weight*terms["raw_brier"], unit_ids, domain_ids)
        risk = excess if self.risk_reference == "relative" else candidate_risk
        if self.reduction == "worst_source" and len(domains) < 2:
            raise ValueError("worst_source requires at least two source domains in each optimization batch.")
        if self.reduction == "worst_source":
            t = self.domain_temperature
            robust = t*(torch.logsumexp(risk/t, dim=0)-math.log(len(risk)))
            diagnostic_excess = t*(torch.logsumexp(excess/t, dim=0)-math.log(len(excess)))
            weights = F.softmax(risk.detach()/t, dim=0)
        else:
            robust = risk.mean()
            diagnostic_excess = excess.mean()
            weights = torch.full_like(excess, 1/len(excess))
        _, delta = domain_means(consistency, unit_ids, domain_ids)
        _, candidate_delta = domain_means(candidate_consistency, unit_ids, domain_ids)
        _, reference_delta = domain_means(reference_consistency, unit_ids, domain_ids)
        pair_penalty = delta.mean() if self.consistency_target == "correction" else candidate_delta.mean()
        result = {"loss": robust+self.lambda_delta*pair_penalty,
                  "risk_objective": robust, "pair_penalty": pair_penalty,
                  "diagnostic_excess": diagnostic_excess, "max_source_excess": excess.max(),
                  "source_envelope": self.domain_temperature*torch.logsumexp(excess/self.domain_temperature, 0),
                  "correction_consistency": delta.mean(),
                  "candidate_output_consistency": candidate_delta.mean(),
                  "reference_output_consistency": reference_delta.mean(),
                  "domain_candidate_risk": candidate_risk.detach(),
                  "domain_reference_risk": reference_risk.detach(),
                  "domain_excess": excess.detach(),
                  "domain_weights": weights, "domains": domains}
        for key in ("ce", "raw_ce", "brier", "raw_brier", "ce_excess", "brier_excess"):
            _, value = domain_means(terms[key], unit_ids, domain_ids)
            result[key] = value.mean()
        return result
