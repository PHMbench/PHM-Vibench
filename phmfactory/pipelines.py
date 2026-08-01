"""Canonical Pipeline identifiers, maturity, and compatibility aliases."""

from __future__ import annotations

from dataclasses import dataclass
import warnings


class PipelineNameDeprecationWarning(FutureWarning):
    """Visible warning for legacy Pipeline identifiers."""


class PipelineMaturityError(RuntimeError):
    """Raised when an opt-in Pipeline is selected without explicit authorization."""


@dataclass(frozen=True)
class PipelineDescriptor:
    """Machine-readable public maturity contract for one protected Pipeline."""

    name: str
    maturity: str
    opt_in_required: bool = False
    reason: str = ""

    @property
    def module_name(self) -> str:
        return f"src.{self.name}"


PIPELINE_DESCRIPTORS: dict[str, PipelineDescriptor] = {
    "Pipeline_01_Fault_Diagnosis": PipelineDescriptor(
        name="Pipeline_01_Fault_Diagnosis",
        maturity="supported",
    ),
    "Pipeline_02_Pretraining_Few_Shot": PipelineDescriptor(
        name="Pipeline_02_Pretraining_Few_Shot",
        maturity="supported_limited",
        reason="release support is limited to the maintained single-stage demo",
    ),
    "Pipeline_03_Multitask_Pretraining_Finetuning": PipelineDescriptor(
        name="Pipeline_03_Multitask_Pretraining_Finetuning",
        maturity="experimental",
        opt_in_required=True,
        reason=(
            "no maintained smoke combination; legacy implementation catches stage "
            "errors and contains unverified checkpoint compatibility paths"
        ),
    ),
    "Pipeline_04_Unified_Evaluation": PipelineDescriptor(
        name="Pipeline_04_Unified_Evaluation",
        maturity="experimental_blocked",
        opt_in_required=True,
        reason=(
            "legacy implementation contains environment-specific paths, sys.path "
            "mutation, broad fallback, and unverified partial checkpoint loading"
        ),
    ),
    "Pipeline_05_Explainable_Fault_Diagnosis": PipelineDescriptor(
        name="Pipeline_05_Explainable_Fault_Diagnosis",
        maturity="compatibility",
        reason="UXFD focused contract exists; no release-supported demo combination",
    ),
    "Pipeline_06_Generative_Modeling": PipelineDescriptor(
        name="Pipeline_06_Generative_Modeling",
        maturity="experimental_contract",
        reason="guarded CFM contract evidence; no release-supported benchmark claim",
    ),
    "Pipeline_ID": PipelineDescriptor(
        name="Pipeline_ID",
        maturity="compatibility",
        reason="legacy research entrypoint outside the maintained demo matrix",
    ),
}

CANONICAL_PIPELINES: tuple[str, ...] = tuple(PIPELINE_DESCRIPTORS)

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

    if canonical not in PIPELINE_DESCRIPTORS:
        accepted = ", ".join(CANONICAL_PIPELINES)
        raise ValueError(
            f"Unknown pipeline {requested!r}. Canonical identifiers: {accepted}"
        )
    return canonical


def pipeline_descriptor(name: str, *, warn: bool = True) -> PipelineDescriptor:
    """Return the descriptor for a canonical or legacy Pipeline identifier."""
    return PIPELINE_DESCRIPTORS[canonical_pipeline_name(name, warn=warn)]


def require_pipeline_access(
    name: str,
    *,
    allow_experimental: bool = False,
    warn: bool = True,
) -> PipelineDescriptor:
    """Require explicit opt-in for Pipelines outside the safe default surface."""
    descriptor = pipeline_descriptor(name, warn=warn)
    if descriptor.opt_in_required and not allow_experimental:
        detail = f" {descriptor.reason}" if descriptor.reason else ""
        raise PipelineMaturityError(
            f"Pipeline {descriptor.name!r} has maturity {descriptor.maturity!r} and "
            f"requires --allow-experimental.{detail}"
        )
    return descriptor


def pipeline_module_name(name: str, *, warn: bool = True) -> str:
    """Resolve a public or legacy identifier to its protected runtime module."""
    return pipeline_descriptor(name, warn=warn).module_name
