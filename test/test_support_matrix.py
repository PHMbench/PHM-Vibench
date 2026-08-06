from __future__ import annotations

from pathlib import Path

from scripts.gen_config_atlas import read_registry
from scripts.gen_support_matrix import (
    render_combinations,
    render_components,
    verified_demos,
)


ROOT = Path(__file__).resolve().parents[1]


def _demos():
    return verified_demos(read_registry(ROOT / "configs/config_registry.csv"))


def test_all_sanity_ok_demos_have_independent_protocol_status() -> None:
    demos = _demos()
    assert len(demos) == 7
    assert {demo.execution_status for demo in demos} == {"sanity_ok"}
    assert {demo.protocol_status for demo in demos} == {"smoke_only"}
    assert all((ROOT / demo.path).is_file() for demo in demos)


def test_gfs_demo_resolves_without_implying_protocol_validity() -> None:
    demo = next(
        item for item in _demos() if item.config_id == "demo_04_cross_system_fewshot"
    )
    assert demo.path.endswith("/gfs_dlinear.yaml")
    assert demo.model == "ISFM/M_01_ISFM"
    assert demo.embedding == "E_01_HSE"
    assert demo.backbone == "B_04_Dlinear"
    assert demo.task == "GFS/classification"
    assert demo.execution_status == "sanity_ok"
    assert demo.protocol_status == "smoke_only"


def test_committed_support_documents_are_generated() -> None:
    demos = _demos()
    assert (ROOT / "SUPPORTED_COMPONENTS.md").read_text(encoding="utf-8") == (
        render_components(demos)
    )
    assert (ROOT / "SUPPORTED_COMBINATIONS.md").read_text(encoding="utf-8") == (
        render_combinations(demos)
    )


def test_support_table_does_not_promote_smoke_to_scientific_support() -> None:
    combinations = render_combinations(_demos())
    assert "Execution evidence" in combinations
    assert "Protocol status" in combinations
    assert "`sanity_ok`" in combinations
    assert "`smoke_only`" in combinations
    assert "release-supported only when" not in combinations
    assert "benchmark validity" in combinations
