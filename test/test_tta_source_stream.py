"""B01 real torch/factory tests; also run unchanged outside the installed wheel."""
from copy import deepcopy
from importlib.resources import files
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from sklearn.metrics import f1_score

from phmfactory.source_stream import run_source_only_stream
from src.config_schema import AdaptationProtocolConfig
from src.model_factory import build_model
from src.model_factory.model_factory import load_ckpt
from src.task_factory.Components.metrics import get_metrics, prepare_metric_inputs
from src.utils.run_summary import write_run_summary


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def protocol(**changes):
    fields = dict(regime="source_only", source_access="checkpoint_only",
                  target_label_access="none", timing="predict_then_update",
                  state_persistence="persistent", domain_boundary="hidden",
                  label_space="closed_set")
    fields.update(changes)
    return AdaptationProtocolConfig(**fields)


class Rows(Dataset):
    def __init__(self, n=7, labels=None, mask=False):
        self.x = torch.arange(n * 32 * 2, dtype=torch.float32).reshape(n, 32, 2) / 64
        self.y = torch.arange(n) % 3 if labels is None else labels
        self.mask = mask
        self.reads = []

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        self.reads.append(i)
        result = dict(x=self.x[i], y=self.y[i], sample_id=i, file_id=100+self.y[i],
                      domain_id=self.y[i], fault_type=self.y[i], condition_id=self.y[i])
        if self.mask:
            result["mask"] = torch.ones(32)
        return result


class FrozenNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(2)
        self.dropout = nn.Dropout(0.7)
        self.head = nn.Linear(2, 3)
        self.register_buffer("aux", torch.tensor(0.), persistent=False)

    def forward(self, x):
        return self.head(self.dropout(self.bn(x.transpose(1, 2)).mean(-1)))


def checkpoint(tmp_path, model, prefixed=False):
    path = tmp_path / "source.pt"
    state = model.state_dict()
    torch.save({"state_dict": {f"network.{k}": v for k, v in state.items()}}
               if prefixed else state, path)
    return path


def run_collect(model, loader, path, proto=None):
    outputs, evaluations = [], []
    def evaluate(pred, view):
        assert not pred.requires_grad
        assert "x" not in view and "fault_type" not in view
        outputs.append(pred)
        evaluations.append(view)
    count = run_source_only_stream(model, loader, proto or protocol(),
                                   checkpoint_path=path, evaluate=evaluate)
    return count, torch.cat(outputs), evaluations


@pytest.mark.parametrize("prefixed", [False, True])
def test_bn_dropout_checkpoint_parity_order_tail_and_all_state(tmp_path, prefixed):
    torch.manual_seed(23)
    model = FrozenNet().eval()
    path = checkpoint(tmp_path, model, prefixed)
    ordinary = deepcopy(model).eval()
    load_ckpt(ordinary, path, strict=True)
    data = Rows()
    with torch.no_grad():
        expected = torch.cat([ordinary(b["x"]) for b in DataLoader(data, batch_size=3)])
    data.reads.clear()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    n, actual, views = run_collect(model, DataLoader(data, batch_size=3), path)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert n == 7 and data.reads == list(range(7))
    assert [len(v["y"]) for v in views] == [3, 3, 1]
    assert torch.cat([v["sample_id"] for v in views]).tolist() == list(range(7))
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert model.aux.item() == 0 and all(p.grad is None for p in model.parameters())
    assert all(not m.training for m in model.modules())


def test_true_permuted_zero_labels_cannot_change_predictions_or_state(tmp_path):
    torch.manual_seed(19)
    template = FrozenNet().eval()
    path = checkpoint(tmp_path, template)
    outputs, states, rngs = [], [], []
    labels = torch.arange(7) % 3
    for y in (labels, labels.flip(0), torch.zeros_like(labels)):
        model = deepcopy(template)
        torch.manual_seed(44)
        _, output, views = run_collect(model, DataLoader(Rows(labels=y), batch_size=3), path)
        assert torch.equal(torch.cat([v["y"] for v in views]), y)
        outputs.append(output)
        states.append({k: v.clone() for k, v in model.state_dict().items()})
        rngs.append(torch.get_rng_state())
    for i in (1, 2):
        assert torch.equal(outputs[0], outputs[i])
        assert all(torch.equal(states[0][k], v) for k, v in states[i].items())
        assert torch.equal(rngs[0], rngs[i])


@pytest.mark.parametrize("change", [dict(regime="online_tta"),
    dict(regime="continual_tta"), dict(regime="episodic_tta", state_persistence="episodic_reset"),
    dict(regime="offline_sfda", adapt_population="adapt", evaluation_population="eval"),
    dict(regime="continual_sfda", adapt_population="adapt", evaluation_population="eval"),
    dict(regime="delayed_label_adaptation", target_label_access="delayed"),
    dict(regime="online_supervised_continual", target_label_access="online_supervised"),
    dict(state_persistence="domain_reset", domain_boundary="known"),
    dict(timing="update_then_predict"), dict(label_space="open_set"),
    dict(label_space="partial_set"), dict(source_access="source_data_available"),
    dict(source_access="checkpoint_plus_artifact", source_artifacts=["fisher"])])
def test_unsupported_protocol_before_checkpoint_or_data(tmp_path, change):
    data = Rows()
    with pytest.raises(ValueError, match="B01 requires"):
        run_collect(FrozenNet().eval(), DataLoader(data, batch_size=3),
                    tmp_path / "absent.pt", protocol(**change))
    assert data.reads == []


def test_mutated_protocol_revalidated(tmp_path):
    proto = protocol()
    proto.passes = 2
    with pytest.raises(ValueError, match="passes=1"):
        run_collect(FrozenNet().eval(), DataLoader(Rows(), batch_size=3),
                    tmp_path / "absent", proto)


@pytest.mark.parametrize("failure", ["train", "child_train", "gradient", "missing", "wrong_keys"])
def test_invalid_initial_state_or_checkpoint_before_data(tmp_path, failure):
    model = FrozenNet().eval()
    path = checkpoint(tmp_path, model)
    if failure == "train":
        model.train()
    elif failure == "child_train":
        model.bn.train()
    elif failure == "gradient":
        model.head.weight.grad = torch.ones_like(model.head.weight)
    elif failure == "missing":
        path = tmp_path / "absent.pt"
    else:
        torch.save({"bogus": torch.ones(1)}, path)
    data = Rows()
    with pytest.raises((ValueError, FileNotFoundError, RuntimeError)):
        run_collect(model, DataLoader(data, batch_size=3), path)
    assert data.reads == []


@pytest.mark.parametrize("kind", ["buffer", "nonpersistent", "parameter", "mode", "new_buffer"])
def test_even_eval_forward_mutation_fails_before_evaluation(tmp_path, kind):
    class Mutator(FrozenNet):
        def forward(self, x):
            prediction = super().forward(x)
            if kind == "buffer":
                self.bn.running_mean.add_(1)
            elif kind == "nonpersistent":
                self.aux.add_(1)
            elif kind == "parameter":
                self.head.bias.add_(1)
            elif kind == "mode":
                self.train()
            else:
                self.register_buffer("unexpected", torch.zeros(1))
            return prediction
    model = Mutator().eval()
    path = checkpoint(tmp_path, model)
    events = []
    with pytest.raises(RuntimeError, match="source-only inference"):
        run_source_only_stream(model, DataLoader(Rows(), batch_size=3), protocol(),
                               checkpoint_path=path, evaluate=lambda *args: events.append(args))
    assert events == []


def test_evaluator_cannot_change_model_for_next_prediction(tmp_path):
    model = FrozenNet().eval()
    path = checkpoint(tmp_path, model)
    def evaluator(pred, view):
        with torch.no_grad():
            model.head.bias.add_(view["y"].float().sum())
    with pytest.raises(RuntimeError, match="mutated parameter"):
        run_source_only_stream(model, DataLoader(Rows(), batch_size=3), protocol(),
                               checkpoint_path=path, evaluate=evaluator)


@pytest.mark.parametrize("kind", ["shuffle", "drop_last", "workers", "custom", "empty"])
def test_unsupported_loader_before_data(tmp_path, kind):
    data = Rows(0 if kind == "empty" else 7)
    kwargs = dict(batch_size=3)
    if kind == "shuffle":
        kwargs["shuffle"] = True
    elif kind == "drop_last":
        kwargs["drop_last"] = True
    elif kind == "workers":
        kwargs["num_workers"] = 1
    elif kind == "custom":
        class Reverse(Sampler):
            def __iter__(self):
                return iter(range(6, -1, -1))
            def __len__(self):
                return 7
        kwargs["sampler"] = Reverse()
    with pytest.raises(ValueError, match="B01"):
        run_collect(FrozenNet().eval(), DataLoader(data, **kwargs), tmp_path / "absent")
    assert data.reads == []


@pytest.mark.parametrize("kind", ["mask", "nan_x", "nan_output", "wrong_shape", "dropped_rows"])
def test_invalid_input_output_or_incomplete_population_fails(tmp_path, kind):
    data = Rows(mask=kind == "mask")
    model = FrozenNet().eval()
    if kind == "nan_x":
        data.x[0, 0, 0] = float("nan")
    if kind == "nan_output":
        model.forward = lambda x: torch.full((len(x), 3), float("nan"))
    if kind == "wrong_shape":
        model.forward = lambda x: torch.zeros(len(x))
    kwargs = {}
    if kind == "dropped_rows":
        from torch.utils.data import default_collate
        kwargs["collate_fn"] = lambda batch: default_collate(batch[:1])
    with pytest.raises((ValueError, RuntimeError)):
        run_collect(model, DataLoader(data, batch_size=3, **kwargs), checkpoint(tmp_path, model))


def test_existing_task_metrics_are_population_not_batch_f1(tmp_path):
    model = FrozenNet().eval()
    # Explicit analytic predictions: labels are invisible to forward.
    data = Rows(7, labels=torch.tensor([0, 0, 0, 1, 1, 2, 2]))
    predictions = torch.tensor([0, 0, 1, 1, 2, 2, 0])
    data.x = torch.nn.functional.one_hot(predictions, 3).float()[:, None, :].repeat(1, 32, 1)
    model.forward = lambda x: x[:, 0, :]
    metadata = {i: {"Name": "fixture", "Dataset_id": 0, "Label": i} for i in range(3)}
    metrics = get_metrics(["acc", "f1"], metadata, loss_name="CE")["fixture"]
    def evaluator(logits, view):
        for name in ("acc", "f1"):
            pred, target = prepare_metric_inputs(name, logits, view["y"], loss_name="CE")
            metrics[f"test_{name}"].update(pred, target)
    run_source_only_stream(model, DataLoader(data, batch_size=3), protocol(),
                           checkpoint_path=checkpoint(tmp_path, model), evaluate=evaluator)
    result = {k: float(metrics[k].compute()) for k in ("test_acc", "test_f1")}
    expected = f1_score(data.y, predictions, labels=[0, 1, 2], average="macro", zero_division=0)
    batch_mean = np.mean([f1_score(data.y[i:i+3], predictions[i:i+3], labels=[0, 1, 2],
                                  average="macro", zero_division=0) for i in (0, 3, 6)])
    assert result["test_f1"] == pytest.approx(expected)
    assert abs(expected - batch_mean) > 0.1
    output = tmp_path / "existing-results"
    output.mkdir()
    pd.DataFrame([result]).to_csv(output / "all_results.csv", index=False)
    write_run_summary(output / "run_summary.json", [result], [23])
    summary = json.loads((output / "run_summary.json").read_text())
    assert summary["metrics"]["test_f1"]["mean"] == pytest.approx(expected)
    assert summary["metrics"]["test_f1"]["sample_std"] is None


@pytest.mark.parametrize("model_name", ["GlobalAverageLinear", "TimesNet"])
def test_actual_data_model_factory_native_loader_checkpoint_parity(tmp_path, model_name):
    from phmfactory.config import analyze_config
    from src.configs.config_utils import dict_to_namespace, transfer_namespace
    from src.data_factory import build_data

    config = str(files("configs") / "experiments/model_integration/timesnet_dummy.yaml")
    overrides = [f"data.data_dir={files('data')}", f"data.cache_dir={tmp_path / 'cache'}",
                 "data.num_workers=0", "data.batch_size=3"]
    cfg = dict_to_namespace(analyze_config(config, override_values=overrides).effective_config)
    args_data, args_task = (transfer_namespace(getattr(cfg, key)) for key in ("data", "task"))
    factory = build_data(args_data, args_task)
    try:
        loader = factory.get_dataloader("test")
        args = SimpleNamespace(type="Baseline" if model_name == "GlobalAverageLinear" else "CNN",
                               name=model_name, input_dim=2, num_classes=2, seq_len=128,
                               d_model=8, d_ff=16, e_layers=1, top_k=2, num_kernels=2, dropout=0.1)
        torch.manual_seed(29)
        source = build_model(args, metadata=None).eval()
        path = checkpoint(tmp_path, source, prefixed=True)
        reference = build_model(args, metadata=None).eval()
        load_ckpt(reference, path, strict=True)
        with torch.no_grad():
            expected = torch.cat([reference(b["x"], b["file_id"], task_id="classification") for b in loader])
        n, observed, views = run_collect(source, loader, path)
        assert n == len(loader.dataset)
        assert sum(len(v["file_id"]) for v in views) == n
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
    finally:
        factory.data.close()


@pytest.mark.parametrize("case", ["valid", "shuffle", "missing", "duplicate", "label_group"])
def test_native_sampler_order_is_preserved_and_incomplete_population_rejected(tmp_path, case):
    from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
    from src.data_factory.samplers.Sampler import Same_system_Sampler
    rows = Rows(6)
    # Native evaluation deliberately groups systems; this is not timestamp sorting.
    dataset = IdIncludedDataset({10: [rows[0], rows[1]], 20: [rows[2], rows[3]],
                                 30: [rows[4], rows[5]]},
        metadata={10: {"Dataset_id": 0}, 20: {"Dataset_id": 1}, 30: {"Dataset_id": 0}})
    sampler = Same_system_Sampler(dataset, batch_size=3, shuffle=False, drop_last=False)
    if case == "shuffle":
        sampler.shuffle = True
    elif case == "missing":
        sampler.indices_per_system[0].pop()
    elif case == "duplicate":
        sampler.indices_per_system[0][-1] = 0
    elif case == "label_group":
        sampler.system_metadata_key = "Label"
    loader = DataLoader(dataset, batch_sampler=sampler)
    model = FrozenNet().eval()
    path = checkpoint(tmp_path, model)
    if case == "valid":
        _, _, views = run_collect(model, loader, path)
        assert torch.cat([view["sample_id"] for view in views]).tolist() == [0, 1, 4, 5, 2, 3]
    else:
        with pytest.raises(ValueError, match="B01"):
            run_collect(model, loader, path)


def test_forward_failure_is_not_reported_as_success(tmp_path):
    class Broken(FrozenNet):
        def forward(self, x):
            raise LookupError("original forward failure")
    model = Broken().eval()
    events = []
    with pytest.raises(LookupError, match="original forward failure"):
        run_source_only_stream(model, DataLoader(Rows(), batch_size=3), protocol(),
                               checkpoint_path=checkpoint(tmp_path, model),
                               evaluate=lambda *args: events.append(args))
    assert events == []
