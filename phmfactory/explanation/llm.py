"""Provider-neutral LLM verbalization contract for PHM-EIR."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .schema import JsonValue, PHMExplanationState


@dataclass(frozen=True)
class ExplanationClaim:
    text: str
    evidence_ids: tuple[str, ...]
    path_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    uncertainty: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("claim.text must be a non-empty string")
        object.__setattr__(self, "text", self.text.strip())
        for name in ("evidence_ids", "path_ids", "relation_ids"):
            raw = getattr(self, name)
            normalized = _string_sequence(raw, f"claim.{name}")
            if any(not item for item in normalized):
                raise ValueError(f"claim.{name} contains an empty identifier")
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"claim.{name} contains duplicate identifiers")
            object.__setattr__(self, name, normalized)
        if not isinstance(self.uncertainty, str):
            raise TypeError("claim.uncertainty must be a string")
        object.__setattr__(self, "uncertainty", self.uncertainty.strip())

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "text": self.text,
            "evidence_ids": list(self.evidence_ids),
            "path_ids": list(self.path_ids),
            "relation_ids": list(self.relation_ids),
            "uncertainty": self.uncertainty,
        }


@dataclass(frozen=True)
class LLMExplanation:
    summary: str
    claims: tuple[ExplanationClaim, ...]
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ValueError("summary must be a non-empty string")
        object.__setattr__(self, "summary", self.summary.strip())
        if not self.claims:
            raise ValueError("claims must not be empty")
        normalized_limitations = _string_sequence(self.limitations, "limitations")
        if any(not item for item in normalized_limitations):
            raise ValueError("limitations contains an empty item")
        object.__setattr__(self, "limitations", normalized_limitations)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "summary": self.summary,
            "claims": [claim.to_dict() for claim in self.claims],
            "limitations": list(self.limitations),
        }


def build_llm_packet(
    state: PHMExplanationState,
    *,
    audience: str = "PHM engineer",
    detail: str = "concise",
) -> dict[str, JsonValue]:
    """Build a complete request packet without binding to one LLM provider."""

    if not isinstance(state, PHMExplanationState):
        raise TypeError("state must be PHMExplanationState")
    if not isinstance(audience, str) or not audience.strip():
        raise ValueError("audience must be a non-empty string")
    if detail not in {"concise", "standard", "detailed"}:
        raise ValueError("detail must be concise, standard, or detailed")

    return {
        "instruction": (
            "Translate the supplied PHM-EIR state into a public explanation. "
            "Use only supplied evidence, path, and relation identifiers. "
            "Do not infer a physical mechanism from a model-native operator or rule name. "
            "State uncertainty and every relevant limitation."
        ),
        "audience": audience.strip(),
        "detail": detail,
        "constraints": [
            "Do not invent signal observations, evidence identifiers, paths, relations, units, or operating conditions.",
            "Separate model-native structure from mechanism-constrained relations.",
            "Do not claim causal or physical validity when the state capability does not support it.",
            "Every substantive claim must cite at least one supplied evidence or path identifier.",
            "Return exactly the declared JSON object; do not include hidden reasoning or chain of thought.",
        ],
        "state": state.to_dict(),
        "output_contract": {
            "summary": "string",
            "claims": [
                {
                    "text": "string",
                    "evidence_ids": ["known evidence ID"],
                    "path_ids": ["known path ID"],
                    "relation_ids": ["known relation ID"],
                    "uncertainty": "string",
                }
            ],
            "limitations": ["string"],
        },
    }


def _string_sequence(value: Any, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of strings")
    if any(not isinstance(item, str) for item in value):
        raise TypeError(f"{name} must contain only strings")
    result = tuple(item.strip() for item in value)
    if any(not item for item in result):
        raise ValueError(f"{name} contains an empty string")
    return result


def parse_llm_explanation(
    payload: Mapping[str, Any],
    state: PHMExplanationState,
) -> LLMExplanation:
    """Parse one LLM response and reject unsupported references."""

    if not isinstance(payload, Mapping):
        raise TypeError("LLM response must be a mapping")
    expected = {"summary", "claims", "limitations"}
    if set(payload) != expected:
        raise ValueError(f"LLM response keys must be exactly {sorted(expected)}")
    raw_claims = payload["claims"]
    if isinstance(raw_claims, (str, bytes)) or not isinstance(raw_claims, Sequence):
        raise TypeError("claims must be a sequence")

    claims: list[ExplanationClaim] = []
    for index, raw_claim in enumerate(raw_claims):
        if not isinstance(raw_claim, Mapping):
            raise TypeError(f"claims[{index}] must be a mapping")
        claim_keys = {"text", "evidence_ids", "path_ids", "relation_ids", "uncertainty"}
        if set(raw_claim) != claim_keys:
            raise ValueError(
                f"claims[{index}] keys must be exactly {sorted(claim_keys)}"
            )
        claim = ExplanationClaim(
            text=raw_claim["text"],
            evidence_ids=_string_sequence(raw_claim["evidence_ids"], f"claims[{index}].evidence_ids"),
            path_ids=_string_sequence(raw_claim["path_ids"], f"claims[{index}].path_ids"),
            relation_ids=_string_sequence(raw_claim["relation_ids"], f"claims[{index}].relation_ids"),
            uncertainty=raw_claim["uncertainty"],
        )
        if not claim.evidence_ids and not claim.path_ids:
            raise ValueError(
                f"claims[{index}] must cite at least one evidence or path identifier"
            )
        unknown_evidence = set(claim.evidence_ids) - state.evidence_ids
        unknown_paths = set(claim.path_ids) - state.path_ids
        unknown_relations = set(claim.relation_ids) - state.relation_ids
        if unknown_evidence:
            raise ValueError(
                f"claims[{index}] references unknown evidence IDs: {sorted(unknown_evidence)}"
            )
        if unknown_paths:
            raise ValueError(
                f"claims[{index}] references unknown path IDs: {sorted(unknown_paths)}"
            )
        if unknown_relations:
            raise ValueError(
                f"claims[{index}] references unknown relation IDs: {sorted(unknown_relations)}"
            )
        claims.append(claim)

    limitations = _string_sequence(payload["limitations"], "limitations")
    return LLMExplanation(
        summary=payload["summary"],
        claims=tuple(claims),
        limitations=limitations,
    )


def explain_with_llm(
    state: PHMExplanationState,
    generate: Callable[[Mapping[str, JsonValue]], Mapping[str, Any]],
    *,
    audience: str = "PHM engineer",
    detail: str = "concise",
) -> LLMExplanation:
    """Call one user-supplied LLM client and validate its public explanation.

    ``generate`` is the only provider-specific boundary. It receives a JSON-safe
    packet and must return a mapping matching the output contract. There is no
    retry, fallback, or silent repair in this function.
    """

    if not callable(generate):
        raise TypeError("generate must be callable")
    packet = build_llm_packet(state, audience=audience, detail=detail)
    response = generate(packet)
    return parse_llm_explanation(response, state)


__all__ = [
    "ExplanationClaim",
    "LLMExplanation",
    "build_llm_packet",
    "explain_with_llm",
    "parse_llm_explanation",
]
