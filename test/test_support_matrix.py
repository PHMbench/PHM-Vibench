from __future__ import annotations

from pathlib import Path

from scripts.gen_config_atlas import read_registry
from scripts.gen_support_matrix import (
    render_combinations,
    render_components,
    supported_demos,
)


ROOT = Path(__file__).resolve().parents[1]


def _demos():
    return supported_demos(read_registry(ROOT / "configs/config_registry.csv"))


def test_all_sanity_ok_demos_resolve_into_support_records() -> None:
    demos = _demos()
    assert len(demos) == 7
    assert {demo.status for demo in demos} == {"sanity_ok"}
    assert all((ROOT / demo.path).is_file() for demo in demos)


def test_gfs_demo_name_matches_resolved_dlinear_contract() -> None:
    demo = next(
        item for item in _demos() if item.config_id == "demo_04_cross_system_fewshot"
    )
    assert demo.path.endswith("/gfs_dlinear.yaml")
    assert demo.model == "ISFM/M_01_ISFM"
    assert demo.embedding == "E_01_HSE"
    assert demo.backbone == "B_04_Dlinear"
    assert demo.task == "GFS/classification"
    assert "tspn" not in demo.path.casefold()
    assert "tspn" not in demo.description.casefold()


def test_committed_support_documents_are_generated() -> None:
    demos = _demos()
    assert (ROOT / "SUPPORTED_COMPONENTS.md").read_text(encoding="utf-8") == (
        render_components(demos)
    )
    assert (ROOT / "SUPPORTED_COMBINATIONS.md").read_text(encoding="utf-8") == (
        render_combinations(demos)
    )


def test_support_table_uses_resolved_not_filename_inference() -> None:
    combinations = render_combinations(_demos())
    assert "`GFS/classification`" in combinations
    assert "`ISFM/M_01_ISFM`" in combinations
    assert "cross_system_tspn" not in combinations
