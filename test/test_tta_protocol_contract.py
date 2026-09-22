from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from src.config_schema import AdaptationProtocolConfig, TaskConfig


ROOT = Path(__file__).resolve().parents[1]


def protocol(**changes):
    values = dict(
        regime="online_tta",
        source_access="checkpoint_only",
        target_label_access="none",
        timing="predict_then_update",
        state_persistence="persistent",
        domain_boundary="hidden",
        label_space="closed_set",
        passes=1,
        source_artifacts=[],
    )
    values.update(changes)
    return values


def test_base_tta_protocol_fragment_is_strictly_validated():
    payload = yaml.safe_load(
        (ROOT / "configs/base/task/tta_protocol.yaml").read_text(encoding="utf-8")
    )
    task = TaskConfig.model_validate(payload["task"])
    assert task.type == "TTA"
    assert task.name == "protocol_only"
    assert task.protocol is not None
    assert task.protocol.regime == "online_tta"
    # B00 intentionally provides protocol semantics, not a runnable method module.
    assert not (ROOT / "src/task_factory/task/TTA/protocol_only.py").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("regime", "tta"),
        ("source_access", "source_free"),
        ("target_label_access", "maybe"),
        ("timing", "simultaneous"),
        ("state_persistence", "sometimes"),
        ("domain_boundary", "guessed"),
        ("label_space", "unknown"),
    ],
)
def test_protocol_enums_fail_closed(field, value):
    with pytest.raises(ValidationError):
        AdaptationProtocolConfig.model_validate(protocol(**{field: value}))


@pytest.mark.parametrize(
    "changes,detail",
    [
        ({"passes": 2}, "passes=1"),
        (
            {"regime": "episodic_tta", "state_persistence": "persistent"},
            "episodic_reset",
        ),
        (
            {"regime": "online_tta", "state_persistence": "episodic_reset"},
            "episodic_tta",
        ),
        (
            {"regime": "continual_tta", "state_persistence": "domain_reset",
             "domain_boundary": "known"},
            "persistent",
        ),
        (
            {"state_persistence": "domain_reset", "domain_boundary": "hidden"},
            "domain_boundary=known",
        ),
        (
            {"source_access": "checkpoint_plus_artifact", "source_artifacts": []},
            "source_artifacts",
        ),
        (
            {"source_access": "checkpoint_only", "source_artifacts": ["fisher"]},
            "checkpoint_plus_artifact",
        ),
    ],
)
def test_online_protocol_contradictions_are_rejected(changes, detail):
    with pytest.raises(ValidationError, match=detail):
        AdaptationProtocolConfig.model_validate(protocol(**changes))


@pytest.mark.parametrize(
    "regime,access",
    [
        ("source_only", "none"),
        ("episodic_tta", "none"),
        ("online_tta", "none"),
        ("continual_tta", "none"),
        ("offline_sfda", "none"),
        ("continual_sfda", "none"),
        ("delayed_label_adaptation", "delayed"),
        ("online_supervised_continual", "online_supervised"),
    ],
)
def test_regime_fixes_target_label_access(regime, access):
    changes = dict(regime=regime, target_label_access=access)
    if regime == "episodic_tta":
        changes["state_persistence"] = "episodic_reset"
    if regime in {"continual_tta", "continual_sfda", "online_supervised_continual"}:
        changes["state_persistence"] = "persistent"
    if regime in {"offline_sfda", "continual_sfda"}:
        changes.update(adapt_population="target_adapt", evaluation_population="target_eval")
    if regime == "delayed_label_adaptation":
        changes["state_persistence"] = "persistent"
    AdaptationProtocolConfig.model_validate(protocol(**changes))

    wrong = "none" if access != "none" else "delayed"
    changes["target_label_access"] = wrong
    with pytest.raises(ValidationError, match="target_label_access"):
        AdaptationProtocolConfig.model_validate(protocol(**changes))


def test_sfda_requires_distinct_adaptation_and_evaluation_populations():
    base = protocol(
        regime="offline_sfda",
        state_persistence="episodic_reset",
        adapt_population="target_adapt",
        evaluation_population="target_eval",
        passes=5,
    )
    accepted = AdaptationProtocolConfig.model_validate(base)
    assert accepted.passes == 5

    for mutation in (
        {"adapt_population": None},
        {"evaluation_population": None},
        {"evaluation_population": "target_adapt"},
        {"source_access": "source_data_available"},
    ):
        with pytest.raises(ValidationError):
            AdaptationProtocolConfig.model_validate({**base, **mutation})


def test_protocol_is_required_only_for_tta_tasks():
    with pytest.raises(ValidationError, match="requires task.protocol"):
        TaskConfig.model_validate({"type": "TTA", "name": "protocol_only"})
    with pytest.raises(ValidationError, match="reserved for task.type=TTA"):
        TaskConfig.model_validate(
            {"type": "DG", "name": "classification", "protocol": protocol()}
        )
