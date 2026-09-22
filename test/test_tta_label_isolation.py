from copy import deepcopy

import pytest

from src.config_schema import AdaptationProtocolConfig
from phmfactory.adaptation_protocol import (
    build_adaptation_view,
    execute_protocol_step,
    label_event_for_update,
)


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
    return AdaptationProtocolConfig.model_validate(values)


class ToyAdapter:
    def __init__(self):
        self.parameter = 0.0
        self.optimizer_state = {"steps": 0}
        self.buffer = []
        self.decisions = []
        self.unsupervised_losses = []

    def predict(self, view):
        return self.parameter + sum(view["x"])

    def adapt(self, view):
        loss = float(sum(abs(value) for value in view["x"]))
        self.parameter += 0.25 * sum(view["x"])
        self.optimizer_state["steps"] += 1
        self.buffer.append(tuple(view["x"]))
        self.decisions.append(True)
        self.unsupervised_losses.append(loss)
        return {"update_applied": True, "loss": loss}

    def export_state(self):
        return deepcopy(
            {
                "parameter": self.parameter,
                "optimizer_state": self.optimizer_state,
                "buffer": self.buffer,
                "decisions": self.decisions,
                "unsupervised_losses": self.unsupervised_losses,
            }
        )


def test_target_labels_never_enter_adaptation_view_or_change_state():
    states = []
    predictions = []
    evaluations = []
    for y in ([0, 1], [1, 0], [0, 0]):
        adapter = ToyAdapter()
        step = execute_protocol_step(
            adapter,
            {
                "x": [2.0, -1.0],
                "y": y,
                "file_id": 7,
                "sample_id": 11,
                "timestamp": 42,
                "sequence_id": "asset-A",
                "domain_id": 3,
            },
            protocol(),
        )
        states.append(adapter.export_state())
        predictions.append(step.prediction)
        evaluations.append(step.evaluation["y"])

    assert states[0] == states[1] == states[2]
    assert predictions[0] == predictions[1] == predictions[2]
    assert evaluations == [[0, 1], [1, 0], [0, 0]]


def test_hidden_domain_boundary_removes_domain_id_but_known_boundary_exposes_it():
    batch = {"x": [1.0], "y": [0], "domain_id": 9, "speed_rpm": 1200}
    hidden = build_adaptation_view(
        batch, protocol(), allowed_physical_metadata=("speed_rpm",)
    )
    assert hidden == {"x": [1.0], "speed_rpm": 1200}

    known = build_adaptation_view(
        batch,
        protocol(domain_boundary="known"),
        allowed_physical_metadata=("speed_rpm",),
    )
    assert known["domain_id"] == 9
    assert "y" not in known


@pytest.mark.parametrize("key", ["y", "label", "labels", "target", "future_y"])
def test_target_like_metadata_cannot_be_whitelisted(key):
    with pytest.raises(ValueError, match="target label"):
        build_adaptation_view(
            {"x": [1.0], "y": [0], key: [0]},
            protocol(),
            allowed_physical_metadata=(key,),
        )


def test_hidden_boundary_cannot_be_whitelisted_back_in():
    with pytest.raises(ValueError, match="hidden domain"):
        build_adaptation_view(
            {"x": [1.0], "y": [0], "domain_id": 2},
            protocol(),
            allowed_physical_metadata=("domain_id",),
        )


def test_predict_then_update_and_update_then_predict_are_observably_different():
    batch = {"x": [4.0], "y": [1]}

    prequential = ToyAdapter()
    first = execute_protocol_step(
        prequential, batch, protocol(timing="predict_then_update")
    )
    assert first.prediction == 4.0
    assert prequential.parameter == 1.0

    transductive = ToyAdapter()
    second = execute_protocol_step(
        transductive, batch, protocol(timing="update_then_predict")
    )
    assert second.prediction == 5.0
    assert transductive.parameter == 1.0


def test_source_only_never_calls_update():
    adapter = ToyAdapter()
    step = execute_protocol_step(
        adapter,
        {"x": [2.0], "y": [1]},
        protocol(regime="source_only", state_persistence="episodic_reset"),
    )
    assert adapter.export_state()["optimizer_state"]["steps"] == 0
    assert step.update_result == {"update_applied": False, "reason": "source_only"}


def test_delayed_label_is_unavailable_before_declared_step():
    delayed = protocol(
        regime="delayed_label_adaptation",
        target_label_access="delayed",
        state_persistence="persistent",
    )
    batch = {"x": [1.0], "y": 3, "label_available_step": 5}
    assert "y" not in build_adaptation_view(batch, delayed)

    with pytest.raises(RuntimeError, match="unavailable"):
        label_event_for_update(batch, delayed, current_step=4)
    assert label_event_for_update(batch, delayed, current_step=5) == 3


def test_unlabeled_protocol_cannot_request_label_event():
    with pytest.raises(PermissionError, match="forbids"):
        label_event_for_update(
            {"x": [1.0], "y": 1}, protocol(), current_step=0
        )
