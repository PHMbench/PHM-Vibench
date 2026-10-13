"""Prepared source populations and public stages fail before target access."""
from copy import deepcopy
import json
import pandas as pd
import pytest

from phmfactory.config import analyze_config
from src.task_factory.task.classification.multiview_signal.runtime import _prepare, execute


def _metadata_fixture(tmp_path):
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    config = analyze_config("configs/experiments/p19/paderborn_dg_0.yaml").runtime_config()
    assignments, rows = {}, []
    for role, count in config["study"]["role_counts"].items():
        for number in range(count):
            unit = f"synthetic_{role}_{number}"
            assignments[unit] = role
            for domain in range(4):
                rows.append(dict(Id=len(rows) + 1, Dataset_id=20, Name="RM_027_PU",
                    File=f"{unit}/synthetic_{domain}.mat", Label=number % 3, Domain_id=domain,
                    Sample_rate=64000, Sample_lenth=8192, Channel=3))
    metadata = raw_root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    roles = tmp_path / "physical_roles.json"
    roles.write_text(json.dumps({"assignments": assignments}))
    config["data"].update(data_dir=str(raw_root), metadata_file=str(metadata))
    config["study"]["role_map"] = str(roles)
    return config


def test_condition_and_entity_isolation_before_raw_reads(tmp_path):
    config = _metadata_fixture(tmp_path)
    output = tmp_path / "prepared"
    output.mkdir()
    prepared = _prepare(config, output)
    source = pd.read_csv(prepared["source_metadata"])
    full = pd.read_csv(prepared["evaluation_metadata"])
    target = full.loc[full.Role.eq("test")]
    assert set(source.Domain_id) == {1, 2, 3}
    assert set(target.Domain_id) == {0}
    assert not set(source.Physical_unit) & set(target.Physical_unit)
    assert set(source.Role) == {"fit", "tune", "grow", "select"}
    assert len(source) + len(target) + len(pd.read_csv(output / "excluded_cross_combinations.csv")) == 128
    assert prepared["source_config"]["data"]["metadata_file"] == prepared["source_metadata"]
    assert prepared["source_config"]["trainer"]["test_after_fit"] is False
    assert not list(tmp_path.glob("*.h5"))  # Metadata qualification did not read signals.


def test_infeasible_profile_and_roles_stop_with_failure_record(tmp_path):
    config = _metadata_fixture(tmp_path)
    broken = deepcopy(config)
    broken["study"]["program"]["bands_hz"][0][0] = 33000
    with pytest.raises(ValueError, match="Nyquist"):
        execute(broken, "sanity", tmp_path / "invalid_profile")
    config["study"]["role_counts"]["fit"] = 6
    with pytest.raises(ValueError, match="role counts"):
        execute(config, "sanity", tmp_path / "invalid_roles")
    failure = json.loads((tmp_path / "invalid_roles" / "failure.json").read_text())
    assert failure["phase"] == "sanity" and failure["status"] == "failed"
