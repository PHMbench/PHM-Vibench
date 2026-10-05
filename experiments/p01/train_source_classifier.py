"""Source-only training of the existing TSPN reference or declared comparator.

Both roles reuse physical-group sampling, acquisition windows and the fusion
evaluator. Condition-DG reference development uses paired ordinary CE; legacy
reference runs remain unpaired. Comparators retain their declared native paired
objective and share CE + .25 Brier selection relative to the frozen reference.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.p01.fusion_data import read_records, group_pools, sample_units
from experiments.p01.baseline_qualification import qualify_source_prediction_arrays, record_training_failure
from experiments.p01.condition_contract import is_condition_dg
from experiments.p01.train_tspn_fusion_v2 import evaluate, write_csv
from experiments.p01.window_io import materialize, pack_batch
from src.model_factory.model_factory import model_factory, load_ckpt
from src.task_factory.Components.tspn_fusion_loss import domain_means


class _EvaluationView(nn.Module):
    """Expose existing classifiers to the shared evaluator without changing them."""

    def __init__(self, candidate: nn.Module, reference: nn.Module | None,
                 classes: int, temperature: float):
        super().__init__()
        self.candidate, self.reference = candidate, reference
        self.num_classes = classes
        self.reference_temperature = temperature

    def forward_details(self, x: Tensor) -> dict[str, Tensor]:
        logits = self.candidate(x)
        raw = logits.detach() if self.reference is None else self.reference(x)
        return {"raw_logits": raw, "candidate_logits": logits}


def supervised_loss(logits: Tensor, labels: Tensor, groups: Tensor, domains: Tensor,
                    *, brier_weight: float, paired_logits: Tensor | None = None) -> Tensor:
    """Equal-domain, equal-group supervision with both endpoints retained."""
    def score(value: Tensor) -> Tensor:
        loss = F.cross_entropy(value, labels, reduction="none")
        if brier_weight:
            onehot = F.one_hot(labels, value.shape[-1])
            loss = loss+brier_weight*(value.softmax(-1)-onehot).square().sum(-1)
        return loss

    losses = score(logits)
    if paired_logits is not None:
        losses = .5*(losses+score(paired_logits))
    return domain_means(losses, groups, domains)[1].mean()


def _clock(device: str) -> float:
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def _resources(units: list[dict]) -> dict:
    return {"groups": len({u["unit_id"] for u in units}),
            "acquisitions": len(units), "windows": sum(len(u["x"]) for u in units),
            "labelled_acquisitions": len(units)}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("reference", "baseline"), required=True)
    for name in ("model-config", "data-config", "dataset", "output"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=50)
    p.add_argument("--units-per-domain", type=int, default=2)
    p.add_argument("--lr", type=float, default=.001)
    p.add_argument("--weight-decay", type=float, default=0.)
    p.add_argument("--scheduler", choices=("none","cosine"), default="none")
    p.add_argument("--dg", action="store_true", help="Require source-only metadata; allow explicitly supported classification baselines.")
    p.add_argument("--pair-shift", type=int, default=0)
    p.add_argument("--reference-checkpoint")
    p.add_argument("--reference-config")
    p.add_argument("--reference-temperature", type=float, default=1.)
    return p


def _run(args: argparse.Namespace) -> Path:
    if min(args.epochs, args.steps_per_epoch, args.units_per_domain) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("A positive training budget and learning rate are required.")
    if not math.isfinite(args.reference_temperature) or args.reference_temperature <= 0:
        raise ValueError("The frozen reference temperature must be positive.")
    if args.pair_shift < 0:
        raise ValueError("The declared circular pair shift must be nonnegative.")
    if args.role == "reference" and (args.reference_checkpoint or args.reference_config):
        raise ValueError("Reference development uses ordinary CE without a prior checkpoint.")
    if args.role == "baseline" and (args.pair_shift < 1 or not args.reference_checkpoint or not args.reference_config):
        raise ValueError("Baseline training requires the frozen reference and an explicit paired shift.")
    if args.device.startswith("cuda"):
        if args.device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or not torch.cuda.is_available():
            raise ValueError("D1 training requires physical GPU0, CUDA_VISIBLE_DEVICES=0 and device cuda:0.")
    elif args.device != "cpu":
        raise ValueError("Choose cpu or the explicitly bound cuda:0 device.")

    cfg = yaml.safe_load(Path(args.model_config).read_text())
    data = yaml.safe_load(Path(args.data_config).read_text())
    settings = copy.deepcopy(cfg["model"])
    identity = (settings["type"], settings["name"])
    expected = ("X_model", "TSPN") if args.role == "reference" else ("CNN", "ResNet1D")
    supported={("CNN","ResNet1D"),("CNN","TCN"),("Transformer","PatchTST"),("X_model","BASE_ExplainableCNN"),("X_model","MWA_CNN"),("X_model","TSPN")}
    allowed=(identity in supported) if args.role=="baseline" and args.dg else identity==expected
    if not allowed or settings.get("weights_path"):
        raise ValueError(f"Role {args.role} requires the existing {expected} model initialized without weights.")
    if identity == ("X_model", "MWA_CNN") and settings.get("depth") != 6:
        raise ValueError("The declared MWA-CNN-6 baseline requires explicit depth=6; a four-level model is not equivalent.")
    tspn_comparator = args.role == "baseline" and identity == ("X_model", "TSPN")
    if tspn_comparator and (cfg.get("arm") != "TSPN_TON" or cfg.get("training") != {"objective": "ce"}
                            or cfg.get("provenance", {}).get("status") not in {"implementation_unverified", "documented_adaptation"}):
        raise ValueError("The independent TSPN_TON baseline requires explicit CE and declared adaptation provenance.")
    if tspn_comparator and cfg['provenance']['status'] == 'documented_adaptation' and not cfg['provenance'].get('differences'):
        raise ValueError('A documented TSPN_TON adaptation must disclose its implemented differences.')
    objective = "ce" if args.role == "reference" or tspn_comparator else "ce_plus_0.25_brier"
    if cfg.get("training", {"objective": objective}) != {"objective": objective}:
        raise ValueError("The classifier training objective differs from the declared method recipe.")
    if args.role == "baseline" and not args.dg and (settings.get("layers") != [2, 2, 2, 2] or
                                     settings.get("initial_channels") != 64 or settings.get("block_type") != "basic"):
        raise ValueError("The declared ResNet1D baseline uses basic blocks [2,2,2,2] and initial_channels=64.")
    classes = int(settings["num_classes"])
    class_names = data["model"]["class_names"]
    if classes != int(data["model"]["num_classes"]) or len(class_names) != classes or len(set(class_names)) != classes:
        raise ValueError("Classifier and data must share the explicit ordered class space.")
    dataset = next(d for d in data["datasets"] if d["name"] == args.dataset)
    formal_dg = is_condition_dg(dataset, data)
    if args.dg and not formal_dg:
        raise ValueError("DG training requires a bound, verified physical-condition contract.")
    if (args.dg or formal_dg) and dataset.get("access_scope") != "source":
        raise ValueError("DG training requires a separately bound source-only metadata file.")
    if args.role == "reference":
        if formal_dg and args.pair_shift < 1:
            raise ValueError("Condition-DG reference training requires the same declared paired shift as the comparator families.")
        if not formal_dg and args.pair_shift:
            raise ValueError("Legacy reference development uses unpaired ordinary CE.")
    length = int(data["data"]["window_size"])
    if args.pair_shift >= length:
        raise ValueError("The circular pair shift must be shorter than the observed window.")
    if identity == ("X_model", "TSPN") and (int(settings["in_dim"]) != length or int(settings["out_dim"]) != length):
        raise ValueError("Every TSPN input/output interval must match the declared window.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    rng, pair_rng = random.Random(args.seed), random.Random(args.seed+10000)
    settings["device"] = args.device
    cfg["model"] = settings
    model = model_factory(SimpleNamespace(**settings), metadata=None).to(args.device)
    reference, reference_settings, frozen = None, None, None
    if args.role == "baseline":
        reference_settings = copy.deepcopy(yaml.safe_load(Path(args.reference_config).read_text())["model"])
        if (reference_settings["type"], reference_settings["name"]) != ("X_model", "TSPN") or int(reference_settings["num_classes"]) != classes:
            raise ValueError("The fixed reference must be the original TSPN in the same class space.")
        if int(reference_settings["in_dim"]) != length or int(reference_settings["in_channels"]) != int(settings.get("input_dim",settings.get("in_channels"))):
            raise ValueError("Baseline and reference must consume the same window and channels.")
        reference_settings["device"] = args.device
        reference = model_factory(SimpleNamespace(**reference_settings), metadata=None).to(args.device)
        load_ckpt(reference, args.reference_checkpoint, strict=True)
        reference.requires_grad_(False).eval()
        frozen = {key: value.detach().clone() for key, value in reference.state_dict().items()}

    if args.dg and identity==("CNN","TCN") and int(settings.get("kernel_size",2))<2:
        raise ValueError("This pinned TCN requires kernel_size>=2 (zero-chomp kernels are excluded).")
    records = read_records(dataset, data)
    sources = list(map(str, dataset["source_domains"]))
    if len(sources) < 2 or len({r["sample_rate_hz"] for r in records}) != 1:
        raise ValueError("D1 requires at least two source conditions with one sampling convention.")
    train = materialize([r for r in records if r["split"] == "update" and r["domain"] in sources], dataset, data)
    validation = materialize([r for r in records if r["split"] == "validation" and r["domain"] in sources], dataset, data)
    expected_channels = int(settings["in_channels"] if args.role == "reference" else settings.get("input_dim",settings.get("in_channels")))
    if any(u["x"].shape[-1] != expected_channels for u in train+validation):
        raise ValueError("Observed channels differ from the declared classifier input.")
    pools = group_pools(train, sources)
    if any(len(pool) < args.units_per_domain for pool in pools.values()):
        raise ValueError("Insufficient physical groups for the fixed source sampling budget.")
    output = Path(args.output)
    command = dict(vars(args), python=sys.executable, torch_version=str(torch.__version__),
                   numpy_version=str(np.__version__), code_commit=subprocess.check_output(
                       ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip())
    (output/"command.json").write_text(json.dumps(command, indent=2))
    (output/"model_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (output/"data_config.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    groups = sorted({(u["unit_id"], u["split"]) for u in train+validation})
    write_csv(output/"development_groups.csv", [dict(group_id=group, split=split) for group, split in groups])
    torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()}, output/"initial_candidate_state.pt")
    view = _EvaluationView(model, reference, classes, args.reference_temperature)
    if not math.isfinite(args.weight_decay) or args.weight_decay<0:
        raise ValueError("weight_decay must be finite and nonnegative.")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs) if args.scheduler=="cosine" else None
    best, best_epoch = float("inf"), None
    history, sampling, batches = [], [], []
    fit_seconds = selection_seconds = export_seconds = 0.
    beta = 0. if objective == "ce" else .25
    selector_beta = 0. if args.role == "reference" else .25
    scope = dict(status="running", role=args.role, seed=args.seed, permanent_test_predicted=False,
                 independent_assessment_performed=False, fit=_resources(train), selection=_resources(validation),
                 objective=("paired_group_balanced_CE" if args.pair_shift else "ordinary_group_balanced_CE")
                     if objective == "ce" else "paired_CE_plus_0.25_Brier",
                 selector="worst_domain_CE" if args.role == "reference" else "worst_domain_CE_plus_0.25_Brier_excess",
                 arm="p0" if args.role == "reference" else cfg.get("arm", settings["name"]),
                 provenance=copy.deepcopy(cfg.get("provenance", {})),
                 checkpoint_tie_break="earliest_epoch", trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                 total_parameters=sum(p.numel() for p in model.parameters()),
                 reference_parameters=sum(p.numel() for p in reference.parameters()) if reference is not None else 0,
                 supervised_endpoints=2 if args.pair_shift else 1)
    (output/"result_scope.json").write_text(json.dumps(scope, indent=2))
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    try:
        for epoch in range(args.epochs):
            model.train()
            losses = []
            started = _clock(args.device)
            for step in range(args.steps_per_epoch):
                units = sample_units(pools, args.units_per_domain, rng)
                x, y, unit_ids, _ = pack_batch(units, args.device)
                domain_ids = torch.cat([torch.full((len(u["x"]),), sources.index(u["domain"]), dtype=torch.long)
                                        for u in units]).to(args.device)
                shift = pair_rng.randint(1, args.pair_shift) if args.pair_shift else 0
                logits = model(x)
                paired = model(x.roll(shift, dims=1)) if shift else None
                loss = supervised_loss(logits, y, unit_ids, domain_ids, brier_weight=beta, paired_logits=paired)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite classifier training objective; the recipe is unchanged.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Nonfinite classifier gradient; the recipe is unchanged.")
                optimizer.step()
                losses.append(float(loss.detach()))
                batches.append(dict(epoch=epoch, step=step, loss=losses[-1], pair_shift=shift, windows=len(y)))
                sampling.extend(dict(epoch=epoch, step=step, domain=u["domain"], group_id=u["unit_id"],
                                     acquisition_id=u["acquisition_id"], windows=len(u["x"]),
                                     window_ids=json.dumps(list(range(len(u["x"])))), pair_shift=shift) for u in units)
            fit_seconds += _clock(args.device)-started
            started = _clock(args.device)
            arrays = {}
            score, rows, summary = evaluate(view, validation, args.device, 1., selector_beta, pack_batch,
                                            prediction_arrays=arrays)
            if args.role == "reference":
                score = max(row["ce"] for row in summary if row["predictor"] == "candidate")
            selection_seconds += _clock(args.device)-started
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite source validation score.")
            history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), worst_validation_score=score))
            write_csv(output/"training.csv", history)
            write_csv(output/"training_batches.csv", batches)
            write_csv(output/"sampling.csv", sampling)
            if score < best:
                best, best_epoch = score, epoch
                started = _clock(args.device)
                saved = dict(kind="classifier", state_dict=model.state_dict(), model=settings, epoch=epoch,
                             arm=scope["arm"], objective=objective, provenance=scope["provenance"])
                if reference is not None:
                    saved.update(reference_model=reference_settings, reference_state_dict=reference.state_dict(),
                                 reference_temperature=args.reference_temperature)
                torch.save(saved, output/"selected_candidate.pt")
                write_csv(output/"selected_source_validation.csv", summary)
                write_csv(output/"selected_source_validation_units.csv", rows)
                predictions = {key: np.concatenate(value) for key, value in arrays.items()}
                predictions.update(raw_class_names=np.asarray(class_names), candidate_class_names=np.asarray(class_names),
                                   arm=np.asarray(scope["arm"]), seed=np.asarray(args.seed))
                np.savez_compressed(output/"selected_source_validation_windows.npz", **predictions)
                # Save the selected predictor's development qualification without
                # discarding a weak HPO trial or selecting a different checkpoint.
                qualification = qualify_source_prediction_arrays(predictions, sources, classes)
                (output/"source_qualification.json").write_text(json.dumps(qualification, indent=2))
                export_seconds += _clock(args.device)-started
            if scheduler is not None: scheduler.step()
            print(history[-1], flush=True)
        if reference is not None:
            if reference.training or any(p.requires_grad or p.grad is not None for p in reference.parameters()) or not all(
                torch.equal(frozen[key], value) for key, value in reference.state_dict().items()
            ):
                raise AssertionError("Frozen reference parameters or buffers changed.")
        # This trainer writes bare model keys. TCN itself owns a `network`
        # submodule, which is not the Lightning wrapper prefix.
        selected=torch.load(output/"selected_candidate.pt",map_location=args.device,weights_only=True)
        model.load_state_dict(selected["state_dict"],strict=True)
        model.eval()
        scope.update(status="classifier_fitted", best_epoch=best_epoch, selected_score=best,
                     reference_state_unchanged=True if reference is not None else None,
                     training_seconds=fit_seconds, selection_seconds=selection_seconds, export_seconds=export_seconds,
                     optimizer_steps=args.epochs*args.steps_per_epoch,
                     peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated() if args.device.startswith("cuda") else None,
                     aggregation="windows/acquisition; equal acquisitions/physical group; equal groups/condition")
    except Exception as error:
        scope.update(status="failed", error_type=type(error).__name__, error=str(error), completed_epochs=len(history))
        raise
    finally:
        sampled_windows = sum(batch["windows"] for batch in batches)
        scope.update(sampled_windows=sampled_windows, sampled_acquisitions=len(sampling),
                     supervised_window_endpoints=sampled_windows*scope["supervised_endpoints"])
        (output/"result_scope.json").write_text(json.dumps(scope, indent=2))
    print(output, flush=True)
    return output


def run(args: argparse.Namespace) -> Path:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output/"command.json").write_text(json.dumps(dict(vars(args)), indent=2))
    (output/"result_scope.json").write_text(json.dumps(dict(
        status="running", role=args.role, seed=args.seed,
        permanent_test_predicted=False, independent_assessment_performed=False), indent=2))
    try:
        return _run(args)
    except (Exception, KeyboardInterrupt) as error:
        record_training_failure(output, error)
        raise


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
