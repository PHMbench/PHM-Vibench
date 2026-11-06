"""Explicit adapters from PHMFactory model-native traces to PHM-EIR.

Adapters are intentionally model-specific. A tensor or key is never assigned a
physical meaning by name guessing. Unsupported trace types fail explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from numbers import Integral
from typing import Any

from .schema import (
    ClassContribution,
    EvidenceAtom,
    EvidencePath,
    MechanismRelation,
    PHMExplanationState,
    PredictionState,
    UncertaintyState,
    freeze_mapping,
)


def _python(value: Any) -> Any:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return value


def _field(source: Any, name: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _sample(value: Any, sample_index: int, *, name: str) -> Any:
    value = _python(value)
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not 0 <= sample_index < len(value):
            raise IndexError(f"{name} sample_index={sample_index} is out of range")
        return value[sample_index]
    if sample_index != 0:
        raise IndexError(f"{name} is scalar; sample_index must be 0")
    return value


def _float(value: Any, *, name: str) -> float:
    value = _python(value)
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: Any, sample_index: int, *, name: str) -> tuple[float, ...]:
    row = _sample(value, sample_index, name=name)
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
        raise TypeError(f"{name} must contain one vector per sample")
    return tuple(_float(item, name=f"{name}[]") for item in row)


def _matrix(value: Any, sample_index: int, *, name: str) -> tuple[tuple[float, ...], ...]:
    rows = _sample(value, sample_index, name=name)
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
        raise TypeError(f"{name} must contain one matrix per sample")
    matrix: list[tuple[float, ...]] = []
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise TypeError(f"{name} rows must be sequences")
        matrix.append(tuple(_float(item, name=f"{name}[][]") for item in row))
    return tuple(matrix)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    if not logits:
        raise ValueError("logits must not be empty")
    maximum = max(logits)
    exponentials = [math.exp(value - maximum) for value in logits]
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _labels(class_names: Sequence[str] | None, size: int) -> tuple[str, ...]:
    if class_names is None:
        return tuple(f"class_{index}" for index in range(size))
    labels = tuple(str(item).strip() for item in class_names)
    if len(labels) != size or any(not item for item in labels):
        raise ValueError(f"class_names must contain exactly {size} non-empty labels")
    if len(set(labels)) != len(labels):
        raise ValueError("class_names must contain unique labels")
    return labels


def _prediction(logits: tuple[float, ...], class_names: Sequence[str] | None) -> PredictionState:
    labels = _labels(class_names, len(logits))
    probabilities = _softmax(logits)
    index = max(range(len(logits)), key=logits.__getitem__)
    return PredictionState(
        label=labels[index],
        class_index=index,
        confidence=probabilities[index],
        logits=logits,
    )






def export_tspn_uxfd_fuzzy_state(
    model: Any,
    x: Any,
    *,
    sample_id: str,
    task: str = "fault_diagnosis",
    sample_index: int = 0,
    class_names: Sequence[str] | None = None,
    rule_names: Sequence[str] | None = None,
    rule_relations: Mapping[int, tuple[str, str]] | None = None,
    operating_conditions: Mapping[str, Any] | None = None,
    rule_mask: Any = None,
    consequent_permutation: Any = None,
    max_rules: int | None = None,
) -> PHMExplanationState:
    """Run ``TSPN_UXFD.forward_with_fuzzy_trace`` and adapt one batch element."""

    method = getattr(model, "forward_with_fuzzy_trace", None)
    if not callable(method):
        raise TypeError("fuzzy adapter requires model.forward_with_fuzzy_trace")
    output = method(
        x,
        rule_mask=rule_mask,
        consequent_permutation=consequent_permutation,
    )
    return state_from_tspn_uxfd_fuzzy_trace(
        output,
        sample_id=sample_id,
        task=task,
        sample_index=sample_index,
        class_names=class_names,
        rule_names=rule_names,
        rule_relations=rule_relations,
        operating_conditions=operating_conditions,
        max_rules=max_rules,
    )


def state_from_tspn_uxfd_fuzzy_trace(
    output: Any,
    *,
    sample_id: str,
    task: str = "fault_diagnosis",
    sample_index: int = 0,
    class_names: Sequence[str] | None = None,
    rule_names: Sequence[str] | None = None,
    rule_relations: Mapping[int, tuple[str, str]] | None = None,
    operating_conditions: Mapping[str, Any] | None = None,
    max_rules: int | None = None,
) -> PHMExplanationState:
    """Adapt ``FuzzyTraceOutput`` without importing its concrete Torch class."""

    trace = _field(output, "fuzzy_trace")
    if trace is None:
        raise ValueError("fuzzy trace output requires fuzzy_trace")
    logits = _vector(_field(output, "logits"), sample_index, name="logits")
    non_fuzzy_logits = _vector(
        _field(output, "non_fuzzy_logits"), sample_index, name="non_fuzzy_logits"
    )
    labels = _labels(class_names, len(logits))
    prediction = _prediction(logits, labels)
    scale = _float(_field(output, "fuzzy_scale"), name="fuzzy_scale")

    firing = _vector(
        _field(trace, "normalized_rule_firing"),
        sample_index,
        name="normalized_rule_firing",
    )
    contributions = _matrix(
        _field(trace, "rule_contributions"),
        sample_index,
        name="rule_contributions",
    )
    if len(contributions) != len(firing):
        raise ValueError("rule_contributions and normalized_rule_firing disagree")
    if any(len(row) != len(labels) for row in contributions):
        raise ValueError("rule_contributions class dimension disagrees with logits")

    if rule_names is None:
        names = tuple(f"rule_{index}" for index in range(len(firing)))
    else:
        names = tuple(str(item).strip() for item in rule_names)
        if len(names) != len(firing) or any(not item for item in names):
            raise ValueError("rule_names must match the number of fuzzy rules")

    ranked = sorted(range(len(firing)), key=lambda index: (-firing[index], index))
    if max_rules is not None:
        if isinstance(max_rules, bool) or not isinstance(max_rules, Integral) or max_rules <= 0:
            raise ValueError("max_rules must be a positive integer or None")
        ranked = ranked[:max_rules]

    atoms: list[EvidenceAtom] = [
        EvidenceAtom(
            id="component:non_fuzzy",
            kind="model_component",
            name="non-fuzzy decision branch",
            value={"logits": list(non_fuzzy_logits)},
            source="TSPN_UXFD.forward_with_fuzzy_trace",
        )
    ]
    paths: list[EvidencePath] = []
    class_contributions: list[ClassContribution] = []
    relations: list[MechanismRelation] = []

    for class_index, label in enumerate(labels):
        class_contributions.append(
            ClassContribution(
                source_id="component:non_fuzzy",
                target_label=label,
                value=non_fuzzy_logits[class_index],
            )
        )

    for index in ranked:
        atom_id = f"fuzzy_rule:{index}"
        path_id = f"rule_path:{index}"
        atoms.append(
            EvidenceAtom(
                id=atom_id,
                kind="fuzzy_rule",
                name=names[index],
                value={"normalized_firing": firing[index]},
                source="TSPN_UXFD.FuzzyTrace",
            )
        )
        paths.append(
            EvidencePath(
                id=path_id,
                atom_ids=(atom_id,),
                relation="fuzzy antecedent activation contributes to class logits",
                score=firing[index],
            )
        )
        for class_index, label in enumerate(labels):
            class_contributions.append(
                ClassContribution(
                    source_id=atom_id,
                    target_label=label,
                    value=scale * contributions[index][class_index],
                )
            )
        if rule_relations is not None and index in rule_relations:
            predicate, target_claim = rule_relations[index]
            relations.append(
                MechanismRelation(
                    id=f"relation:rule:{index}",
                    source_ids=(path_id,),
                    predicate=predicate,
                    target_claim=target_claim,
                    status="mechanism-constrained",
                )
            )

    exported_fuzzy_reconstructed = [0.0 for _ in labels]
    for rule_index in ranked:
        row = contributions[rule_index]
        for class_index, value in enumerate(row):
            exported_fuzzy_reconstructed[class_index] += scale * value
    reconstructed = tuple(
        non_fuzzy_logits[index] + exported_fuzzy_reconstructed[index]
        for index in range(len(labels))
    )
    residual = max(abs(logits[index] - reconstructed[index]) for index in range(len(labels)))

    tiny = 1.0e-12
    entropy = -sum(weight * math.log(max(weight, tiny)) for weight in firing)
    if len(firing) > 1:
        entropy /= math.log(len(firing))
    else:
        entropy = 0.0
    top_share = max(firing) if firing else 0.0
    confidence = prediction.confidence if prediction.confidence is not None else 0.0
    metrics = (
        ("confidence_risk", 1.0 - confidence),
        ("normalized_rule_firing_entropy", entropy),
        ("rule_fragmentation", 1.0 - top_share),
        ("max_logit_reconstruction_residual", residual),
    )
    limitations = [
        "Fuzzy rules are model-native constructs; physical mechanism meaning requires an explicit external rule mapping.",
    ]
    if max_rules is not None and max_rules < len(firing):
        limitations.append(
            "The exported state contains only the highest-firing rules and cannot claim complete rule coverage."
        )

    effective_mask = _sample(
        _field(trace, "rule_mask"), sample_index, name="rule_mask"
    )
    effective_permutation = _python(_field(trace, "consequent_permutation"))
    # Native permutations are shared [rules] or per-sample [batch, rules].
    if effective_permutation is not None and isinstance(effective_permutation[0], Sequence):
        effective_permutation = _sample(
            effective_permutation, sample_index, name="consequent_permutation"
        )
    if effective_mask is None or effective_permutation is None:
        limitations.append("Effective intervention values were not supplied in this trace.")

    return PHMExplanationState(
        sample_id=sample_id,
        task=task,
        model_family="TSPN_UXFD",
        trace_kind="reconstructable_additive_fuzzy_trace",
        prediction=prediction,
        evidence_atoms=tuple(atoms),
        evidence_paths=tuple(paths),
        contributions=tuple(class_contributions),
        mechanism_relations=tuple(relations),
        uncertainty=UncertaintyState(
            metrics=metrics,
            calibration_state="external_validation_required",
        ),
        operating_conditions=freeze_mapping(
            operating_conditions, name="operating_conditions"
        ),
        capabilities=tuple(
            [
                "prediction",
                "typed_evidence",
                "fuzzy_rule_firing",
                "additive_contribution",
                "rule_intervention",
                "uncertainty",
            ]
            + (["decision_reconstruction"] if len(ranked) == len(firing) else ["partial_contribution"])
        ),
        limitations=tuple(limitations),
        metadata=freeze_mapping(
            {
                "fuzzy_scale": scale,
                "exported_rule_count": len(ranked),
                "total_rule_count": len(firing),
                "rule_mask": effective_mask,
                "consequent_permutation": effective_permutation,
            },
            name="metadata",
        ),
    )


__all__ = [
    "export_tspn_uxfd_fuzzy_state",
    "state_from_tspn_uxfd_fuzzy_trace",
]
