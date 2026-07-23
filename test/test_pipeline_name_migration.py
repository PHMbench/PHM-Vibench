from __future__ import annotations

from pathlib import Path

import pytest

from phmfactory.pipelines import (
    CANONICAL_PIPELINES,
    PIPELINE_ALIASES,
    PipelineNameDeprecationWarning,
    canonical_pipeline_name,
    pipeline_module_name,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CANONICAL = {
    "Pipeline_01_Fault_Diagnosis",
    "Pipeline_02_Pretraining_Few_Shot",
    "Pipeline_03_Multitask_Pretraining_Finetuning",
    "Pipeline_04_Unified_Evaluation",
    "Pipeline_05_Explainable_Fault_Diagnosis",
    "Pipeline_06_Generative_Modeling",
    "Pipeline_ID",
}


def test_canonical_pipeline_inventory_is_explicit() -> None:
    assert set(CANONICAL_PIPELINES) == EXPECTED_CANONICAL


@pytest.mark.parametrize("legacy,canonical", tuple(PIPELINE_ALIASES.items()))
def test_legacy_pipeline_identifiers_resolve_with_warning(
    legacy: str,
    canonical: str,
) -> None:
    with pytest.warns(PipelineNameDeprecationWarning):
        assert canonical_pipeline_name(legacy) == canonical
    assert pipeline_module_name(legacy, warn=False) == f"src.{canonical}"


@pytest.mark.parametrize("canonical", tuple(sorted(EXPECTED_CANONICAL)))
def test_canonical_pipeline_modules_exist(canonical: str) -> None:
    assert canonical_pipeline_name(canonical) == canonical
    assert (REPOSITORY_ROOT / "src" / f"{canonical}.py").is_file()


def test_unknown_pipeline_is_rejected_before_dynamic_import() -> None:
    with pytest.raises(ValueError, match="Unknown pipeline"):
        canonical_pipeline_name("Pipeline_99_Unknown")
