from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from tools.repo import check_release_readiness as readiness


VALID_CWRU_MANIFEST = {
    "schema_version": 1,
    "bundle_id": "cwru-demo-v1",
    "dataset_name": "CWRU",
    "files": {
        "metadata": {"filename": "metadata.xlsx", "required": True},
        "signals": {"filename": "RM_001_CWRU.h5", "required": True},
        "corpus": {"filename": "corpus.xlsx", "required": False},
    },
    "metadata": {
        "id_column": "Id",
        "required_columns": ["Dataset_id", "Label", "Domain_id"],
        "selector": {"column": "Name", "values": ["RM_001_CWRU"]},
        "column_aliases": {
            "sample_length": ["Sample_lenth", "Sample_length"],
            "channel_count": ["Channel"],
        },
    },
    "providers": {
        "huggingface": {
            "repo_id": "PHMbench/PHM-Vibench",
            "revision": "main",
            "files": {
                "metadata": "metadata.xlsx",
                "signals": "RM_001_CWRU.h5",
            },
        },
        "modelscope": {
            "repo_id": "PHMbench/PHM-Vibench",
            "revision": "master",
            "files": {
                "metadata": "metadata.xlsx",
                "signals": "RM_001_CWRU.h5",
            },
        },
    },
}


def test_cwru_release_contract_is_scientific_not_hash_based() -> None:
    payload = deepcopy(VALID_CWRU_MANIFEST)
    payload["expected_sha256"] = {}

    assert readiness._cwru_contract_errors(payload) == ()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda payload: payload["metadata"].update({"id_column": ""}), "id_column"),
        (
            lambda payload: payload["metadata"].update(
                {"required_columns": ["Dataset_id", "Domain_id"]}
            ),
            "Label",
        ),
        (
            lambda payload: payload["providers"]["huggingface"].update(
                {"revision": ""}
            ),
            "revision",
        ),
        (
            lambda payload: payload["providers"]["modelscope"]["files"].update(
                {"signals": "wrong.h5"}
            ),
            "wrong.h5",
        ),
    ],
)
def test_cwru_release_contract_rejects_semantic_gaps(mutator, message: str) -> None:
    payload = deepcopy(VALID_CWRU_MANIFEST)
    mutator(payload)

    assert message in "; ".join(readiness._cwru_contract_errors(payload))


def _baseline_row(*, protocol_status: str = "baseline_valid") -> dict[str, str]:
    return {
        "id": readiness.BASELINE_REGISTRY_ID,
        "category": "baseline",
        "path": readiness.BASELINE_CONFIG_PATH,
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "status": "sanity_ok",
        "protocol_status": protocol_status,
    }


def test_baseline_valid_reference_accepts_the_exact_reviewed_row() -> None:
    assert readiness._baseline_reference_findings([_baseline_row()]) == ()


def test_smoke_only_reference_reports_revalidation_not_corruption() -> None:
    findings = readiness._baseline_reference_findings(
        [_baseline_row(protocol_status="smoke_only")]
    )

    assert [finding.code for finding in findings] == [
        "BASELINE_REVALIDATION_REQUIRED"
    ]
    assert "smoke_only" in findings[0].detail
    assert "current-source" in findings[0].detail


def test_malformed_baseline_reference_remains_invalid() -> None:
    row = _baseline_row(protocol_status="smoke_only")
    row["path"] = "configs/baselines/wrong.yaml"

    findings = readiness._baseline_reference_findings([row])

    assert [finding.code for finding in findings] == [
        "BASELINE_VALID_REFERENCE_INVALID"
    ]
    assert "wrong.yaml" in findings[0].detail


@pytest.mark.parametrize(("mode", "exit_code"), [("audit", 0), ("release", 1)])
def test_revalidation_blocker_is_visible_and_blocks_release(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    mode: str,
    exit_code: int,
) -> None:
    blocker = readiness.Finding(
        "BASELINE_REVALIDATION_REQUIRED",
        "MFPT remains protocol_status='smoke_only'",
    )
    monkeypatch.setattr(readiness, "collect_findings", lambda: (blocker,))
    monkeypatch.setattr(
        readiness.sys,
        "argv",
        ["check_release_readiness.py", "--mode", mode],
    )

    assert readiness.main() == exit_code
    output = capsys.readouterr().out
    assert "BASELINE_REVALIDATION_REQUIRED" in output
    assert "readiness BLOCKED: 1 blocker(s)" in output


def test_registry_reader_requires_release_authority_columns(tmp_path) -> None:
    registry = tmp_path / "registry.csv"
    registry.write_text("id,path\nbaseline,x.yaml\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        readiness._read_registry_rows(registry)


def test_gitlinks_enumerates_raw_unconfigured_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = (
        "100644 blob " + "1" * 40 + "\tREADME.md\n"
        "160000 commit " + "2" * 40 + "\tunconfigured/raw\n"
    )
    monkeypatch.setattr(
        readiness.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=output),
    )
    assert readiness._gitlinks() == {"unconfigured/raw": "2" * 40}
