"""Real-torch Tent fidelity, label separation, lifecycle and installed-factory tests."""
from copy import deepcopy
from importlib.resources import files
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from phmfactory.tent_stream import run_tent_stream
from phmfactory.source_stream import run_source_only_stream
from src.config_schema import AdaptationProtocolConfig
from src.model_factory import build_model
from src.model_factory.model_factory import load_ckpt
from src.task_factory.Components.tent import Tent
from src.task_factory.Components.metrics import get_metrics, prepare_metric_inputs
from src.utils.run_summary import write_run_summary

_spec = importlib.util.spec_from_file_location(
    "_phm_tent_reference", Path(__file__).parent / "fixtures/tent_upstream/tent.py")
upstream = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = upstream
_spec.loader.exec_module(upstream)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def protocol(**changes):
    fields = dict(regime="online_tta", source_access="checkpoint_only",
                  target_label_access="none", timing="predict_then_update",
                  state_persistence="persistent", domain_boundary="hidden",
                  label_space="closed_set")
    fields.update(changes)
    return AdaptationProtocolConfig(**fields)


class TinyBN(nn.Module):
    def __init__(self, dimensions=1, dropout=0.25):
        super().__init__()
        self.dimensions = dimensions
        self.embed = nn.Linear(2, 4)
        self.bn = (nn.BatchNorm1d if dimensions == 1 else nn.BatchNorm2d)(4)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4, 3)
        self.register_buffer("aux", torch.tensor(7.), persistent=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        z = self.embed(x).transpose(1, 2).contiguous()
        z = self.bn(z) if self.dimensions == 1 else self.bn(z.unsqueeze(-1)).squeeze(-1)
        # Both oracle paths use the same contiguous normalization/dropout layout.
        return self.head(self.drop(z.contiguous().relu()).mean(-1))


class Rows(Dataset):
    def __init__(self, n=7, labels=None):
        generator = torch.Generator().manual_seed(23)
        self.x = torch.randn(n, 32, 2, generator=generator)
        self.y = torch.arange(n) % 3 if labels is None else labels
        self.reads = []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        self.reads.append(i)
        # Deliberately class-encoding fields must never become model arguments.
        return dict(x=self.x[i], y=self.y[i], sample_id=i, file_id=100+self.y[i],
                    domain_id=self.y[i], fault_type=self.y[i], condition_id=self.y[i])


def checkpoint(tmp_path, model, prefixed=False):
    path = tmp_path / "source.pt"
    state = model.state_dict()
    torch.save({"state_dict": {f"network.{k}": v for k, v in state.items()}}
               if prefixed else state, path)
    return path


def assert_tree(a, b, *, tolerance=0):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_tree(a[key], b[key], tolerance=tolerance)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_tree(x, y, tolerance=tolerance)
    else:
        assert a == b


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("shape", [(3, 32, 2), (1, 17, 2), (4, 1, 2)])
@pytest.mark.parametrize("dropout", [0.0, 0.25])
def test_three_updates_match_unmodified_upstream_logits_gradients_adam_rng(dimensions, shape, dropout):
    torch.manual_seed(123)
    oracle = TinyBN(2, dropout).eval()
    model = TinyBN(dimensions, dropout).eval()
    model.load_state_dict(oracle.state_dict(), strict=True)
    actual = Tent(model, learning_rate=1e-3)
    upstream.configure_model(oracle)
    oracle_parameters, _ = upstream.collect_params(oracle)
    optimizer = torch.optim.Adam(oracle_parameters, lr=1e-3, betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=0)
    actual_grads, expected_grads = {}, {}
    def capture(optimizer, network, destination):
        original_step = optimizer.step
        def step():
            destination.update({n: p.grad.detach().clone() for n, p in network.named_parameters()
                                if p.requires_grad})
            original_step()
        optimizer.step = step
    capture(actual.optimizer, model, actual_grads)
    capture(optimizer, oracle, expected_grads)
    for _ in range(3):
        x = torch.randn(*shape)
        rng = torch.get_rng_state()
        prediction = actual.predict(x)
        loss = actual.adapt()
        after = torch.get_rng_state()
        torch.set_rng_state(rng)
        expected = upstream.forward_and_adapt(x, oracle, optimizer)
        torch.testing.assert_close(prediction, expected.detach(), rtol=1e-6, atol=1e-7)
        assert loss == pytest.approx(float(upstream.softmax_entropy(expected.detach()).mean()), rel=1e-6)
        assert_tree(actual_grads, expected_grads, tolerance=1e-7)
        assert_tree(model.state_dict(), oracle.state_dict(), tolerance=1e-7)
        assert_tree(actual.optimizer.state_dict(), optimizer.state_dict(), tolerance=1e-7)
        assert torch.equal(after, torch.get_rng_state())
    assert actual.num_updates == 3 and model.calls == oracle.calls == 3
    assert set(actual.parameter_names) == {"bn.weight", "bn.bias"}
    assert model.aux.item() == 7


@pytest.mark.parametrize("prefixed", [False, True])
@pytest.mark.parametrize("regime", ["online_tta", "continual_tta"])
def test_labels_cannot_change_predictions_losses_update_trajectory_or_optimizer(tmp_path, prefixed, regime):
    torch.manual_seed(1)
    template = TinyBN().eval()
    path = checkpoint(tmp_path, template, prefixed)
    results = []
    for labels in (torch.arange(7) % 3, torch.tensor([2, 1, 0, 0, 2, 1, 0]), torch.zeros(7).long()):
        model = deepcopy(template)
        data = Rows(labels=labels)
        predictions, states, losses, metrics, identities = [], [], [], [], []
        def evaluate(pred, view):
            assert not pred.requires_grad and "x" not in view and "fault_type" not in view
            predictions.append(pred.clone())
            losses.append(float(upstream.softmax_entropy(pred).mean()))
            states.append(deepcopy(model.state_dict()))
            metrics.extend((pred.argmax(1) == view["y"]).tolist())
            identities.extend(view["sample_id"].tolist())
            pred.fill_(999)  # the evaluator cannot change the private update logits
        torch.manual_seed(71)
        adapter = run_tent_stream(model, DataLoader(data, batch_size=3), protocol(regime=regime),
                                 checkpoint_path=path, learning_rate=1e-3, evaluate=evaluate)
        assert data.reads == identities == list(range(7))
        assert model.calls == adapter.num_updates == 3 and adapter.num_samples == 7
        assert model.bn.running_mean is None and model.bn.running_var is None
        results.append((predictions, states, losses, adapter.state_dict()))
    assert_tree(results[0], results[1])
    assert_tree(results[0], results[2])


def test_first_prediction_is_batch_normalized_preupdate_not_frozen_or_recomputed():
    torch.manual_seed(22)
    source = TinyBN(dropout=0).eval()
    x = 15 + 3 * torch.randn(3, 32, 2)
    with torch.no_grad():
        frozen = source(x)
    adapter = Tent(deepcopy(source), learning_rate=0.1)
    weights = adapter.model.bn.weight.detach().clone()
    prediction = adapter.predict(x)
    assert torch.equal(weights, adapter.model.bn.weight)
    assert not torch.allclose(frozen, prediction)
    adapter.adapt()
    assert not torch.equal(weights, adapter.model.bn.weight)
    assert adapter.model.calls == source.calls + 1  # exactly one new forward
    after = adapter.predict(x)
    assert not torch.allclose(after, prediction)


@pytest.mark.parametrize("rate", [True, "0.001", 0, -1, float("inf"), float("nan")])
def test_invalid_learning_rate_fails_without_repair(rate):
    model = TinyBN().eval()
    with pytest.raises(ValueError, match="learning_rate"):
        Tent(model, learning_rate=rate)
    assert not model.training


@pytest.mark.parametrize("kind", ["no_bn", "non_affine", "bn3d", "train", "gradient", "nonfinite"])
def test_invalid_model_fails_before_algorithm_setup(kind):
    model = TinyBN().eval()
    if kind == "no_bn":
        model.bn = nn.Identity()
    elif kind == "non_affine":
        model.bn = nn.BatchNorm1d(4, affine=False).eval()
    elif kind == "bn3d":
        model.bn = nn.BatchNorm3d(4).eval()
    elif kind == "train":
        model.train()
    elif kind == "gradient":
        model.head.weight.grad = torch.ones_like(model.head.weight)
    else:
        with torch.no_grad():
            model.head.weight.fill_(float("nan"))
    with pytest.raises(ValueError):
        Tent(model, learning_rate=1e-3)
    assert model.calls == 0


@pytest.mark.parametrize("change", [dict(regime="source_only"), dict(timing="update_then_predict"),
    dict(regime="episodic_tta", state_persistence="episodic_reset"),
    dict(state_persistence="domain_reset", domain_boundary="known"),
    dict(regime="offline_sfda", adapt_population="adapt", evaluation_population="eval"),
    dict(regime="continual_sfda", adapt_population="adapt", evaluation_population="eval"),
    dict(regime="delayed_label_adaptation", target_label_access="delayed"),
    dict(regime="online_supervised_continual", target_label_access="online_supervised"),
    dict(source_access="source_data_available"), dict(label_space="open_set"),
    dict(source_access="checkpoint_plus_artifact", source_artifacts=["fisher"])])
def test_unsupported_protocol_rejected_before_checkpoint_or_data(tmp_path, change):
    data = Rows()
    with pytest.raises(ValueError, match="B02"):
        run_tent_stream(TinyBN().eval(), DataLoader(data, batch_size=3), protocol(**change),
                        checkpoint_path=tmp_path / "absent", learning_rate=1e-3, evaluate=lambda *a: None)
    assert data.reads == []


@pytest.mark.parametrize("kwargs", [dict(shuffle=True), dict(drop_last=True), dict(num_workers=1)])
def test_loader_contract_reused_without_order_or_tail_repair(tmp_path, kwargs):
    data = Rows()
    with pytest.raises(ValueError, match="B01"):
        run_tent_stream(TinyBN().eval(), DataLoader(data, batch_size=3, **kwargs), protocol(),
                        checkpoint_path=tmp_path / "absent", learning_rate=1e-3, evaluate=lambda *a: None)
    assert data.reads == []


@pytest.mark.parametrize("kind", ["affine", "frozen", "buffer", "mode", "bn_mode", "gradient"])
def test_evaluator_state_mutation_fails_before_adaptation(tmp_path, kind):
    model = TinyBN().eval()
    path = checkpoint(tmp_path, model)
    def evaluate(*_):
        with torch.no_grad():
            if kind == "affine": model.bn.weight.add_(1)
            elif kind == "frozen": model.head.weight.add_(1)
            elif kind == "buffer": model.aux.add_(1)
            elif kind == "mode": model.drop.eval()
            elif kind == "bn_mode": model.bn.track_running_stats = True
            else: model.bn.weight.grad = torch.ones_like(model.bn.weight)
    with pytest.raises(RuntimeError, match="Tent"):
        run_tent_stream(model, DataLoader(Rows(), batch_size=3), protocol(),
                        checkpoint_path=path, learning_rate=1e-3, evaluate=evaluate)
    assert model.calls == 1


def test_predict_update_lifecycle_and_no_grad_context():
    adapter = Tent(TinyBN().eval(), learning_rate=1e-3)
    with pytest.raises(RuntimeError, match="pending prediction"):
        adapter.adapt()
    with torch.no_grad():
        pred = adapter.predict(Rows().x[:3])
        assert not pred.requires_grad
        with pytest.raises(RuntimeError, match="pending"):
            adapter.predict(Rows().x[:3])
        with pytest.raises(RuntimeError, match="pending"):
            adapter.state_dict()
        adapter.adapt()
    assert adapter.num_updates == 1
    with torch.inference_mode(), pytest.raises(RuntimeError, match="inference_mode"):
        adapter.predict(Rows().x[:3])


@pytest.mark.parametrize("kind", ["nan", "rank", "integer", "bn_single_value", "nondifferentiable"])
def test_bad_inputs_and_logits_fail_without_easier_update(kind):
    model = TinyBN().eval()
    adapter = Tent(model, learning_rate=1e-3)
    x = Rows().x[:3]
    if kind == "nan": x.fill_(float("nan"))
    elif kind == "rank": x = x[0]
    elif kind == "integer": x = x.long()
    elif kind == "bn_single_value": x = x[:1, :1]
    else:
        original = model.forward
        model.forward = lambda x: original(x).detach()
    with pytest.raises(ValueError):
        adapter.predict(x)
    assert adapter.num_updates == 0


def test_save_fresh_process_restore_next_prediction_optimizer_and_rng(tmp_path):
    torch.manual_seed(33)
    adapter = Tent(TinyBN().eval(), learning_rate=1e-3)
    x = Rows().x[:3]
    for _ in range(2):
        adapter.predict(x)
        adapter.adapt()
    torch.save(adapter.state_dict(), tmp_path / "paused.pt")
    torch.save(x, tmp_path / "x.pt")
    expected = adapter.predict(x)
    adapter.adapt()
    expected_state = adapter.state_dict()
    script = """
import runpy, sys, torch
from pathlib import Path
from src.task_factory.Components.tent import Tent
torch.set_num_threads(1)
ns = runpy.run_path(sys.argv[1])
root = Path(sys.argv[2])
torch.manual_seed(999)
t = Tent(ns['TinyBN']().eval(), learning_rate=1e-3)
t.load_state_dict(torch.load(root / 'paused.pt', weights_only=False))
pred = t.predict(torch.load(root / 'x.pt', weights_only=True))
t.adapt()
torch.save((pred, t.state_dict()), root / 'resumed.pt')
"""
    process = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(tmp_path)],
                             text=True, capture_output=True, timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    observed, state = torch.load(tmp_path / "resumed.pt", weights_only=False)
    assert_tree(observed, expected)
    assert_tree(state, expected_state)


def test_population_metrics_and_existing_results_are_not_batch_mean(tmp_path):
    model = TinyBN(dropout=0).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([3., 0., 0.]))
    path = checkpoint(tmp_path, model)
    data = Rows(labels=torch.tensor([0, 0, 0, 1, 1, 1, 2]))
    metadata = {i: {"Name": "fixture", "Dataset_id": 0, "Label": i} for i in range(3)}
    metrics = get_metrics(["acc", "f1"], metadata, loss_name="CE")["fixture"]
    predictions = []
    def evaluate(logits, view):
        predictions.extend(logits.argmax(1).tolist())
        for name in ("acc", "f1"):
            pred, target = prepare_metric_inputs(name, logits, view["y"], loss_name="CE")
            metrics[f"test_{name}"].update(pred, target)
    adapter = run_tent_stream(model, DataLoader(data, batch_size=3), protocol(),
                             checkpoint_path=path, learning_rate=1e-3, evaluate=evaluate)
    result = {k: float(metrics[k].compute()) for k in ("test_acc", "test_f1")}
    expected = f1_score(data.y, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
    batch_mean = np.mean([f1_score(data.y[i:i+3], predictions[i:i+3], labels=[0, 1, 2],
                                  average="macro", zero_division=0) for i in (0, 3, 6)])
    assert result["test_f1"] == pytest.approx(expected)
    assert abs(expected - batch_mean) > 0.05 and adapter.num_samples == 7
    pd.DataFrame([result]).to_csv(tmp_path / "all_results.csv", index=False)
    write_run_summary(tmp_path / "run_summary.json", [result], [33])
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["metrics"]["test_f1"]["mean"] == pytest.approx(expected)


def test_actual_factories_bundled_dummy_resnet1d_and_native_target_loader(tmp_path):
    from phmfactory.config import analyze_config
    from src.configs.config_utils import dict_to_namespace, transfer_namespace
    from src.data_factory import build_data
    config = str(files("configs") / "experiments/model_integration/timesnet_dummy.yaml")
    overrides = [f"data.data_dir={files('data')}", f"data.cache_dir={tmp_path / 'cache'}",
                 "data.num_workers=0", "data.batch_size=3"]
    cfg = dict_to_namespace(analyze_config(config, override_values=overrides).effective_config)
    factory = build_data(transfer_namespace(cfg.data), transfer_namespace(cfg.task))
    try:
        loader = factory.get_dataloader("test")
        args = SimpleNamespace(type="CNN", name="ResNet1D", input_dim=2, num_classes=2,
                               layers=[1, 1, 1, 1], initial_channels=4)
        torch.manual_seed(29)
        model = build_model(args, metadata=None).eval()
        path = checkpoint(tmp_path, model, prefixed=True)
        frozen = deepcopy(model.state_dict())
        control = deepcopy(model).eval()
        with torch.no_grad():
            ordinary = torch.cat([control(batch["x"]) for batch in loader])
        source_outputs = []
        run_source_only_stream(control, loader, protocol(regime="source_only"),
                               checkpoint_path=path,
                               evaluate=lambda p, v: source_outputs.append(p))
        torch.testing.assert_close(torch.cat(source_outputs), ordinary, rtol=1e-6, atol=1e-7)
        outputs, targets = [], []
        def evaluate(prediction, view):
            outputs.append(prediction)
            targets.append(view["y"])
        adapter = run_tent_stream(model, loader, protocol(), checkpoint_path=path,
                                 learning_rate=1e-3, evaluate=evaluate)
        assert adapter.num_samples == len(loader.dataset)
        assert adapter.num_updates == len(loader) and torch.cat(outputs).isfinite().all()
        assert len(torch.cat(targets)) == adapter.num_samples
        assert any(not torch.equal(frozen[n], p) for n, p in model.named_parameters()
                   if n in adapter.parameter_names)
        for name, parameter in model.named_parameters():
            if name not in adapter.parameter_names:
                assert torch.equal(parameter, frozen[name]), name
    finally:
        factory.data.close()


@pytest.mark.parametrize("kind", ["missing", "wrong_keys"])
def test_checkpoint_failure_does_not_read_target(tmp_path, kind):
    path = tmp_path / "source.pt"
    if kind == "wrong_keys":
        torch.save({"bogus": torch.tensor(0.)}, path)
    rows = Rows()
    with pytest.raises((FileNotFoundError, RuntimeError)):
        run_tent_stream(TinyBN().eval(), DataLoader(rows, batch_size=3), protocol(),
                        checkpoint_path=path, learning_rate=1e-3, evaluate=lambda *a: None)
    assert rows.reads == []


@pytest.mark.parametrize("kind", ["no_y", "mask", "forward_buffer", "gradient_nan", "evaluator_error"])
def test_failure_cannot_produce_completed_population(tmp_path, kind):
    model = TinyBN().eval()
    path = checkpoint(tmp_path, model)
    rows = Rows()
    from torch.utils.data import default_collate
    def collate(batch):
        batch = default_collate(batch)
        if kind == "no_y": del batch["y"]
        elif kind == "mask": batch["mask"] = torch.ones(3, 32)
        return batch
    original = model.forward
    if kind == "forward_buffer":
        def forward(x):
            pred = original(x)
            model.aux.add_(1)
            return pred
        model.forward = forward
    if kind == "gradient_nan":
        model.bn.weight.register_hook(lambda g: g * float("nan"))
    def evaluate(*_):
        if kind == "evaluator_error":
            raise LookupError("original evaluator failure")
    before = model.bn.weight.detach().clone()
    with pytest.raises((ValueError, RuntimeError, KeyError, LookupError)):
        run_tent_stream(model, DataLoader(rows, batch_size=3, collate_fn=collate), protocol(),
                        checkpoint_path=path, learning_rate=1e-3, evaluate=evaluate)
    assert torch.equal(before, model.bn.weight)  # no optimizer step before these failures


@pytest.mark.parametrize("kind", ["lr", "amsgrad", "nan_optimizer", "counter", "buffers", "last_loss",
                                  "missing_moments", "missing_second_moment", "wrong_step", "moment_shape"])
def test_invalid_resume_fails_closed(kind):
    adapter = Tent(TinyBN().eval(), learning_rate=1e-3)
    adapter.predict(Rows().x[:3]); adapter.adapt()
    state = adapter.state_dict()
    if kind == "lr": state["optimizer"]["param_groups"][0]["lr"] = 1.
    elif kind == "amsgrad": state["optimizer"]["param_groups"][0]["amsgrad"] = True
    elif kind == "nan_optimizer":
        next(iter(state["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    elif kind == "counter": state["num_updates"] = True
    elif kind == "buffers": state["extra_buffers"] = {}
    elif kind == "missing_moments": state["optimizer"]["state"] = {}
    elif kind == "missing_second_moment":
        del next(iter(state["optimizer"]["state"].values()))["exp_avg_sq"]
    elif kind == "wrong_step":
        next(iter(state["optimizer"]["state"].values()))["step"].zero_()
    elif kind == "moment_shape":
        next(iter(state["optimizer"]["state"].values()))["exp_avg"] = torch.zeros(1)
    else: state["last_loss"] = float("inf")
    with pytest.raises(ValueError):
        Tent(TinyBN().eval(), learning_rate=1e-3).load_state_dict(state)
