"""SAR fidelity, label isolation, reliable filtering, recovery and resume tests."""
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

from phmfactory.sar_stream import run_sar_stream
from phmfactory.source_stream import run_source_only_stream
from src.config_schema import AdaptationProtocolConfig
from src.model_factory import build_model
from src.task_factory.Components.metrics import get_metrics, prepare_metric_inputs
from src.task_factory.Components.sar import SAR
from src.utils.run_summary import write_run_summary

_FIXTURE = Path(__file__).parent / "fixtures/sar_upstream"
_sam_spec = importlib.util.spec_from_file_location("sam", _FIXTURE / "sam.py")
upstream_sam = importlib.util.module_from_spec(_sam_spec)
sys.modules["sam"] = upstream_sam
_sam_spec.loader.exec_module(upstream_sam)
_sar_spec = importlib.util.spec_from_file_location("_phm_sar_reference", _FIXTURE / "sar.py")
upstream_sar = importlib.util.module_from_spec(_sar_spec)
_sar_spec.loader.exec_module(upstream_sar)


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


class TinyNorm(nn.Module):
    def __init__(self, norm="bn2d", dropout=0.0):
        super().__init__()
        self.norm_kind = norm
        self.embed = nn.Linear(2, 4)
        if norm == "bn1d": self.adapt_norm = nn.BatchNorm1d(4)
        elif norm == "bn2d": self.adapt_norm = nn.BatchNorm2d(4)
        elif norm == "gn": self.adapt_norm = nn.GroupNorm(2, 4)
        elif norm == "ln": self.adapt_norm = nn.LayerNorm(4)
        else: raise ValueError(norm)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(4, 3)
        self.register_buffer("aux", torch.tensor(7.), persistent=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        z = self.embed(x)
        if self.norm_kind == "ln":
            z = self.adapt_norm(z)
            z = z.transpose(1, 2).contiguous()
        else:
            z = z.transpose(1, 2).contiguous()
            if self.norm_kind == "bn2d": z = self.adapt_norm(z.unsqueeze(-1)).squeeze(-1)
            else: z = self.adapt_norm(z)
        return self.head(self.drop(z.contiguous().relu()).mean(-1))


class SkipLayer4(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(2, 4)
        self.bn = nn.BatchNorm1d(4)
        self.layer4 = nn.Sequential(nn.BatchNorm1d(4))
        self.head = nn.Linear(4, 3)
    def forward(self, x):
        z = self.embed(x).transpose(1, 2).contiguous()
        return self.head(self.layer4(self.bn(z)).relu().mean(-1))


class Rows(Dataset):
    def __init__(self, n=7, labels=None):
        g = torch.Generator().manual_seed(23)
        self.x = torch.randn(n, 32, 2, generator=g)
        self.y = torch.arange(n) % 3 if labels is None else labels
        self.reads = []
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        self.reads.append(i)
        return dict(x=self.x[i], y=self.y[i], sample_id=i, file_id=100+self.y[i],
                    domain_id=self.y[i], fault_type=self.y[i], condition_id=self.y[i])


def checkpoint(tmp_path, model, prefixed=False):
    path = tmp_path / "source.pt"
    state = model.state_dict()
    torch.save({"state_dict": {f"network.{k}": v for k, v in state.items()}}
               if prefixed else state, path)
    return path


def assert_tree(a, b, *, rtol=0, atol=0):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: assert_tree(a[key], b[key], rtol=rtol, atol=atol)
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_tree(x, y, rtol=rtol, atol=atol)
    else:
        assert a == b


def upstream_adapter(model, lr=1e-3, margin=10.0):
    model = upstream_sar.configure_model(model)
    params, _ = upstream_sar.collect_params(model)
    opt = upstream_sam.SAM(params, torch.optim.SGD, lr=lr, momentum=0.9)
    return upstream_sar.SAR(model, opt, margin_e0=margin)


@pytest.mark.parametrize("norm", ["bn2d", "gn", "ln"])
def test_three_batches_match_unmodified_upstream(norm):
    torch.manual_seed(7)
    source = TinyNorm(norm=norm, dropout=0)
    reference_model = deepcopy(source)
    production_model = deepcopy(source).eval()
    reference = upstream_adapter(reference_model, margin=10.0)
    production = SAR(production_model, learning_rate=1e-3, margin_e0=10.0)
    batches = [Rows().x[:3], Rows().x[3:6], Rows().x[6:]]
    for x in batches:
        expected = reference(x)
        observed = production.predict(x)
        production.adapt()
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
        for (_, a), (_, b) in zip(reference.model.named_parameters(), production.model.named_parameters()):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
        assert production.ema == pytest.approx(reference.ema, rel=2e-6, abs=2e-7)
    assert production.num_updates == 3 and production.num_skipped_batches == 0


def test_bn1d_operator_extension_matches_upstream_bn2d():
    torch.manual_seed(11)
    bn2d = TinyNorm("bn2d", dropout=0)
    bn1d = TinyNorm("bn1d", dropout=0)
    bn1d.load_state_dict(bn2d.state_dict(), strict=True)
    reference = upstream_adapter(bn2d, margin=10.0)
    production = SAR(bn1d.eval(), learning_rate=1e-3, margin_e0=10.0)
    for x in (Rows().x[:3], Rows().x[3:6]):
        expected = reference(x)
        observed = production.predict(x)
        production.adapt()
        torch.testing.assert_close(observed, expected, rtol=1e-6, atol=1e-7)
        for (_, a), (_, b) in zip(reference.model.named_parameters(), production.model.named_parameters()):
            torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)


def test_official_top_layer_selection_is_preserved():
    model = SkipLayer4().eval()
    source = deepcopy(model.state_dict())
    adapter = SAR(model, learning_rate=1e-3, margin_e0=10.0)
    assert "bn.weight" in adapter.parameter_names and "bn.bias" in adapter.parameter_names
    assert not any(name.startswith("layer4.") for name in adapter.parameter_names)
    adapter.predict(Rows().x[:3]); adapter.adapt()
    for name, parameter in model.named_parameters():
        if name.startswith("layer4."):
            assert torch.equal(parameter, source[name])


@pytest.mark.parametrize("regime", ["online_tta", "continual_tta"])
@pytest.mark.parametrize("prefixed", [False, True])
def test_labels_cannot_change_sar_trajectory(tmp_path, regime, prefixed):
    torch.manual_seed(17)
    source = TinyNorm("bn1d", dropout=0)
    path = checkpoint(tmp_path, source, prefixed=prefixed)
    states, outputs = [], []
    label_sets = [torch.arange(7) % 3, torch.tensor([2,1,0,0,2,1,0]), torch.zeros(7).long()]
    for labels in label_sets:
        torch.manual_seed(1701)
        model = deepcopy(source).eval()
        preds = []
        adapter = run_sar_stream(
            model, DataLoader(Rows(labels=labels), batch_size=3), protocol(regime=regime),
            checkpoint_path=path, learning_rate=1e-3, margin_e0=10.0,
            evaluate=lambda p, v: preds.append(p.clone()),
        )
        outputs.append(torch.cat(preds))
        states.append(adapter.state_dict())
    torch.testing.assert_close(outputs[0], outputs[1]); torch.testing.assert_close(outputs[0], outputs[2])
    assert_tree(states[0], states[1]); assert_tree(states[0], states[2])


@pytest.mark.parametrize("change", [
    dict(regime="source_only"), dict(regime="episodic_tta", state_persistence="episodic_reset"),
    dict(regime="offline_sfda", adapt_population="a", evaluation_population="e"),
    dict(regime="delayed_label_adaptation", target_label_access="delayed"),
    dict(source_access="source_data_available"), dict(regime="online_supervised_continual", target_label_access="online_supervised"),
    dict(timing="update_then_predict"), dict(state_persistence="domain_reset", domain_boundary="known"),
    dict(label_space="open_set"),
])
def test_unsupported_protocols_fail_before_target_read(tmp_path, change):
    rows = Rows()
    p = protocol(**change)
    with pytest.raises(ValueError, match="B03"):
        run_sar_stream(TinyNorm("bn1d").eval(), DataLoader(rows, batch_size=3), p,
                       checkpoint_path=tmp_path / "unused", learning_rate=1e-3,
                       margin_e0=10.0, evaluate=lambda *_: None)
    assert rows.reads == []


def test_empty_reliable_set_is_explicit_no_update():
    model = TinyNorm("bn1d", dropout=0).eval()
    adapter = SAR(model, learning_rate=1e-3, margin_e0=1e-12)
    before = deepcopy(model.state_dict())
    prediction = adapter.predict(Rows().x[:3])
    assert prediction.isfinite().all()
    assert adapter.adapt() is None
    assert adapter.num_updates == 0 and adapter.num_skipped_batches == 1
    assert adapter.last_reliable_first == 0 and adapter.ema is None
    for name, value in model.state_dict().items(): assert torch.equal(value, before[name])


def test_low_entropy_triggers_source_recovery():
    model = TinyNorm("bn1d", dropout=0).eval()
    with torch.no_grad():
        model.head.weight.zero_(); model.head.bias.copy_(torch.tensor([12., -12., -12.]))
    adapter = SAR(model, learning_rate=1e-3, margin_e0=10.0)
    source, extra = adapter._capture_model()
    adapter.predict(Rows().x[:3]); adapter.adapt()
    assert adapter.num_updates == 1 and adapter.num_recoveries == 1
    assert adapter.ema is not None and adapter.ema < 0.2
    current, current_extra = adapter._capture_model()
    assert_tree(current, source); assert_tree(current_extra, extra)
    assert adapter.optimizer.base_optimizer.state == {}


@pytest.mark.parametrize("mutation", ["rng", "frozen", "buffer", "mode", "bn_mode"])
def test_evaluator_cannot_change_sar_update_inputs_or_state(tmp_path, mutation):
    model = TinyNorm("bn1d", dropout=.25).eval()
    path = checkpoint(tmp_path, model)
    def evaluate(*_):
        with torch.no_grad():
            if mutation == "rng": torch.rand(1)
            elif mutation == "frozen": model.head.weight.add_(1)
            elif mutation == "buffer": model.aux.add_(1)
            elif mutation == "mode": model.drop.eval()
            else: model.adapt_norm.track_running_stats = True
    with pytest.raises(RuntimeError, match="SAR"):
        run_sar_stream(model, DataLoader(Rows(), batch_size=3), protocol(),
                       checkpoint_path=path, learning_rate=1e-3, margin_e0=10.0,
                       evaluate=evaluate)
    assert model.calls == 1


@pytest.mark.parametrize("norm", ["bn1d", "bn2d", "gn", "ln"])
def test_predict_then_two_forward_update_contract(norm):
    model = TinyNorm(norm, dropout=0).eval()
    adapter = SAR(model, learning_rate=1e-3, margin_e0=10.0)
    prediction = adapter.predict(Rows().x[:3])
    assert model.calls == 1 and not prediction.requires_grad
    adapter.adapt()
    assert model.calls == 2 and adapter.num_updates == 1
    with pytest.raises(RuntimeError, match="pending"):
        adapter.adapt()


def test_save_new_process_restore_next_prediction_optimizer_ema_rng(tmp_path):
    torch.manual_seed(33)
    source = TinyNorm("bn1d", dropout=.25).eval()
    source_state = deepcopy(source.state_dict())
    adapter = SAR(source, learning_rate=1e-3, margin_e0=10.0)
    x = Rows().x[:3]
    for _ in range(2): adapter.predict(x); adapter.adapt()
    torch.save(adapter.state_dict(), tmp_path / "paused.pt")
    torch.save((source_state, x), tmp_path / "fixture.pt")
    expected = adapter.predict(x); adapter.adapt(); expected_state = adapter.state_dict()
    script = r'''
import runpy, sys, torch
from pathlib import Path
from src.task_factory.Components.sar import SAR
ns = runpy.run_path(sys.argv[1]); root=Path(sys.argv[2]); torch.set_num_threads(1)
source_state, x = torch.load(root/'fixture.pt', weights_only=False)
torch.manual_seed(999)
model=ns['TinyNorm']('bn1d', dropout=.25).eval(); model.load_state_dict(source_state)
adapter=SAR(model, learning_rate=1e-3, margin_e0=10.0)
adapter.load_state_dict(torch.load(root/'paused.pt', weights_only=False))
pred=adapter.predict(x); adapter.adapt(); torch.save((pred,adapter.state_dict()), root/'resumed.pt')
'''
    process = subprocess.run([sys.executable,"-c",script,str(Path(__file__).resolve()),str(tmp_path)],
                             text=True,capture_output=True,timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    observed, state = torch.load(tmp_path / "resumed.pt", weights_only=False)
    assert_tree(observed, expected)
    assert_tree(state, expected_state)


@pytest.mark.parametrize("kind", ["lr","margin","counter","nan_momentum","missing_momentum",
                                  "moment_shape","source_optimizer","source_lr","last_loss",
                                  "source_model","source_model_value"])
def test_invalid_resume_fails_closed(kind):
    model = TinyNorm("bn1d", dropout=0).eval()
    adapter = SAR(model, learning_rate=1e-3, margin_e0=10.0)
    adapter.predict(Rows().x[:3]); adapter.adapt()
    state = adapter.state_dict()
    if kind == "lr": state["optimizer"]["param_groups"][0]["lr"] = 1.
    elif kind == "margin": state["margin_e0"] = 9.
    elif kind == "counter": state["num_updates"] = True
    elif kind == "nan_momentum": next(iter(state["optimizer"]["state"].values()))["momentum_buffer"].fill_(float("nan"))
    elif kind == "missing_momentum": state["optimizer"]["state"] = {}
    elif kind == "moment_shape": next(iter(state["optimizer"]["state"].values()))["momentum_buffer"] = torch.zeros(1)
    elif kind == "source_optimizer": state["source_optimizer"]["state"] = {0:{"momentum_buffer":torch.ones(1)}}
    elif kind == "source_lr": state["source_optimizer"]["param_groups"][0]["lr"] = 1.0
    elif kind == "last_loss": state["last_loss"] = float("inf")
    elif kind == "source_model": state["source_model"].pop(next(iter(state["source_model"])))
    else:
        key = next(iter(state["source_model"])); state["source_model"][key].add_(1)
    with pytest.raises((ValueError, RuntimeError)):
        SAR(TinyNorm("bn1d", dropout=0).eval(), learning_rate=1e-3, margin_e0=10.0).load_state_dict(state)


def test_population_f1_uses_complete_population_and_existing_writer(tmp_path):
    model = TinyNorm("bn1d", dropout=0).eval()
    with torch.no_grad(): model.head.weight.zero_(); model.head.bias.copy_(torch.tensor([3.,0.,0.]))
    path = checkpoint(tmp_path, model)
    data = Rows(labels=torch.tensor([0,0,0,1,1,1,2]))
    metadata = {i:{"Name":"fixture","Dataset_id":0,"Label":i} for i in range(3)}
    metrics = get_metrics(["acc","f1"], metadata, loss_name="CE")["fixture"]
    predictions=[]
    def evaluate(logits, view):
        predictions.extend(logits.argmax(1).tolist())
        for name in ("acc","f1"):
            pred,target=prepare_metric_inputs(name,logits,view["y"],loss_name="CE")
            metrics[f"test_{name}"].update(pred,target)
    adapter=run_sar_stream(model,DataLoader(data,batch_size=3),protocol(),checkpoint_path=path,
                           learning_rate=1e-3,margin_e0=10.0,evaluate=evaluate)
    result={k:float(metrics[k].compute()) for k in ("test_acc","test_f1")}
    expected=f1_score(data.y,predictions,labels=[0,1,2],average="macro",zero_division=0)
    batch_mean=np.mean([f1_score(data.y[i:i+3],predictions[i:i+3],labels=[0,1,2],average="macro",zero_division=0) for i in (0,3,6)])
    assert result["test_f1"] == pytest.approx(expected) and abs(expected-batch_mean)>0.05
    assert adapter.num_samples == 7
    pd.DataFrame([result]).to_csv(tmp_path/"all_results.csv",index=False)
    write_run_summary(tmp_path/"run_summary.json",[result],[33])
    summary=json.loads((tmp_path/"run_summary.json").read_text())
    assert summary["metrics"]["test_f1"]["mean"] == pytest.approx(expected)


def test_actual_resnet1d_factory_and_native_target_loader(tmp_path):
    from phmfactory.config import analyze_config
    from src.configs.config_utils import dict_to_namespace, transfer_namespace
    from src.data_factory import build_data
    config=str(files("configs")/"experiments/model_integration/timesnet_dummy.yaml")
    overrides=[f"data.data_dir={files('data')}",f"data.cache_dir={tmp_path/'cache'}","data.num_workers=0","data.batch_size=3"]
    cfg=dict_to_namespace(analyze_config(config,override_values=overrides).effective_config)
    factory=build_data(transfer_namespace(cfg.data),transfer_namespace(cfg.task))
    try:
        loader=factory.get_dataloader("test")
        args=SimpleNamespace(type="CNN",name="ResNet1D",input_dim=2,num_classes=2,layers=[1,1,1,1],initial_channels=4)
        torch.manual_seed(29); model=build_model(args,metadata=None).eval(); path=checkpoint(tmp_path,model,prefixed=True)
        frozen=deepcopy(model.state_dict()); outputs=[]
        adapter=run_sar_stream(model,loader,protocol(),checkpoint_path=path,learning_rate=1e-3,
                               margin_e0=10.0,evaluate=lambda p,v: outputs.append(p))
        assert adapter.num_samples==len(loader.dataset) and torch.cat(outputs).isfinite().all()
        assert adapter.num_updates==len(loader)
        assert any(not torch.equal(frozen[n],p) for n,p in model.named_parameters() if n in adapter.parameter_names)
        for name,p in model.named_parameters():
            if name not in adapter.parameter_names: assert torch.equal(p,frozen[name]), name
        assert not any(name.startswith("layer4.") for name in adapter.parameter_names)
    finally:
        factory.data.close()


@pytest.mark.parametrize("kind", ["missing","wrong_keys"])
def test_checkpoint_failure_does_not_read_target(tmp_path,kind):
    path=tmp_path/"source.pt"
    if kind=="wrong_keys": torch.save({"bogus":torch.tensor(0.)},path)
    rows=Rows()
    with pytest.raises((FileNotFoundError,RuntimeError)):
        run_sar_stream(TinyNorm("bn1d").eval(),DataLoader(rows,batch_size=3),protocol(),checkpoint_path=path,
                       learning_rate=1e-3,margin_e0=10.0,evaluate=lambda *_:None)
    assert rows.reads==[]


@pytest.mark.parametrize("kind", ["no_y","mask","forward_buffer","gradient_nan","evaluator_error"])
def test_failure_cannot_produce_completed_population(tmp_path,kind):
    model=TinyNorm("bn1d",dropout=0).eval(); path=checkpoint(tmp_path,model); rows=Rows()
    from torch.utils.data import default_collate
    def collate(batch):
        batch=default_collate(batch)
        if kind=="no_y": del batch["y"]
        elif kind=="mask": batch["mask"]=torch.ones(3,32)
        return batch
    original=model.forward
    if kind=="forward_buffer":
        def forward(x):
            pred=original(x); model.aux.add_(1); return pred
        model.forward=forward
    if kind=="gradient_nan": model.adapt_norm.weight.register_hook(lambda g:g*float("nan"))
    def evaluate(*_):
        if kind=="evaluator_error": raise LookupError("original evaluator failure")
    with pytest.raises((ValueError,RuntimeError,KeyError,LookupError)):
        run_sar_stream(model,DataLoader(rows,batch_size=3,collate_fn=collate),protocol(),checkpoint_path=path,
                       learning_rate=1e-3,margin_e0=10.0,evaluate=evaluate)
