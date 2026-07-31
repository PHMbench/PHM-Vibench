"""Canonical Pipeline identifiers and compatibility aliases for PHMFactory."""

from __future__ import annotations

import warnings


class PipelineNameDeprecationWarning(FutureWarning):
    """Visible warning for legacy Pipeline identifiers."""


CANONICAL_PIPELINES: tuple[str, ...] = (
    "Pipeline_01_Fault_Diagnosis",
    "Pipeline_02_Pretraining_Few_Shot",
    "Pipeline_03_Multitask_Pretraining_Finetuning",
    "Pipeline_04_Unified_Evaluation",
    "Pipeline_05_Explainable_Fault_Diagnosis",
    "Pipeline_06_Generative_Modeling",
    "Pipeline_ID",
)

PIPELINE_ALIASES: dict[str, str] = {
    "Pipeline_01_default": "Pipeline_01_Fault_Diagnosis",
    "Pipeline_02_pretrain_fewshot": "Pipeline_02_Pretraining_Few_Shot",
    "Pipeline_03_multitask_pretrain_finetune": (
        "Pipeline_03_Multitask_Pretraining_Finetuning"
    ),
    "Pipeline_04_unified_metric": "Pipeline_04_Unified_Evaluation",
    "Pipeline_05_default_w_explain": "Pipeline_05_Explainable_Fault_Diagnosis",
    "Pipeline_06_generative": "Pipeline_06_Generative_Modeling",
}


def canonical_pipeline_name(name: str, *, warn: bool = True) -> str:
    """Return the canonical Pipeline identifier or raise a bounded error."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("pipeline must be a non-empty string")

    requested = name.strip()
    canonical = PIPELINE_ALIASES.get(requested, requested)
    if requested != canonical and warn:
        warnings.warn(
            f"Pipeline identifier {requested!r} is deprecated; use {canonical!r}.",
            PipelineNameDeprecationWarning,
            stacklevel=2,
        )

    if canonical not in CANONICAL_PIPELINES:
        accepted = ", ".join(CANONICAL_PIPELINES)
        raise ValueError(
            f"Unknown pipeline {requested!r}. Canonical identifiers: {accepted}"
        )
    return canonical


def pipeline_module_name(name: str, *, warn: bool = True) -> str:
    """Resolve a public or legacy identifier to its protected runtime module."""
    return f"src.{canonical_pipeline_name(name, warn=warn)}"
