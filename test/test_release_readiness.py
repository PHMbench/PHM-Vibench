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


def test_baseline_valid_reference_requires_the_exact_reviewed_row() -> None:
    row = {
        "id": readiness.BASELINE_REGISTRY_ID,
        "category": "baseline",
        "path": readiness.BASELINE_CONFIG_PATH,
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "status": "sanity_ok",
        "protocol_status": "baseline_valid",
    }

    assert readiness._baseline_valid_error([row]) == ""

    row["protocol_status"] = "smoke_only"
    assert "baseline_valid" in readiness._baseline_valid_error([row])


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
