"""LLM verbalization of model-native PHM explanation traces."""

from .adapters import (
    export_tspn_uxfd_fuzzy_state,
    export_xoan_state,
    state_from_tspn_uxfd_fuzzy_trace,
    state_from_xoan_report,
)
from .llm import (
    ExplanationClaim,
    LLMExplanation,
    build_llm_packet,
    explain_with_llm,
    parse_llm_explanation,
)
from .schema import (
    SCHEMA_VERSION,
    ClassContribution,
    EvidenceAtom,
    EvidencePath,
    MechanismRelation,
    PHMExplanationState,
    PredictionState,
    UncertaintyState,
    freeze_mapping,
)

__all__ = [
    "SCHEMA_VERSION",
    "ClassContribution",
    "EvidenceAtom",
    "EvidencePath",
    "ExplanationClaim",
    "LLMExplanation",
    "MechanismRelation",
    "PHMExplanationState",
    "PredictionState",
    "UncertaintyState",
    "build_llm_packet",
    "explain_with_llm",
    "export_tspn_uxfd_fuzzy_state",
    "export_xoan_state",
    "freeze_mapping",
    "parse_llm_explanation",
    "state_from_tspn_uxfd_fuzzy_trace",
    "state_from_xoan_report",
]
