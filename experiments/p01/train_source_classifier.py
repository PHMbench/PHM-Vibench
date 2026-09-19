"""Source-only training of the existing TSPN reference or ResNet1D comparator.

Both roles reuse physical-group sampling, acquisition windows and the fusion
evaluator. Reference development uses ordinary CE. The comparator uses paired
CE + .25 Brier and selection relative to the complete frozen reference.
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
    p.add_argument("--pair-shift", type=int, default=0)
    p.add_argument("--reference-checkpoint")
    p.add_argument("--reference-config")
    p.add_argument("--reference-temperature", type=float, default=1.)
    return p


def run(args: argparse.Namespace) -> Path:
    if min(args.epochs, args.steps_per_epoch, args.units_per_domain) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("A positive training budget and learning rate are required.")
    if not math.isfinite(args.reference_temperature) or args.reference_temperature <= 0:
        raise ValueError("The frozen reference temperature must be positive.")
    if args.role == "reference" and (args.pair_shift or args.reference_checkpoint or args.reference_config):
        raise ValueError("Reference development uses unpaired ordinary CE without a prior checkpoint.")
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
    if identity != expected or settings.get("weights_path"):
        raise ValueError(f"Role {args.role} requires the existing {expected} model initialized without weights.")
    if args.role == "baseline" and (settings.get("layers") != [2, 2, 2, 2] or
                                     settings.get("initial_channels") != 64 or settings.get("block_type") != "basic"):
        raise ValueError("The declared ResNet1D baseline uses basic blocks [2,2,2,2] and initial_channels=64.")
    classes = int(settings["num_classes"])
    class_names = data["model"]["class_names"]
    if classes != int(data["model"]["num_classes"]) or len(class_names) != classes or len(set(class_names)) != classes:
        raise ValueError("Classifier and data must share the explicit ordered class space.")
    dataset = next(d for d in data["datasets"] if d["name"] == args.dataset)
    length = int(data["data"]["window_size"])
    if args.pair_shift >= length:
        raise ValueError("The circular pair shift must be shorter than the observed window.")
    if args.role == "reference" and int(settings["in_dim"]) != length:
        raise ValueError("The original TSPN input interval must match the declared window.")

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
        if int(reference_settings["in_dim"]) != length or int(reference_settings["in_channels"]) != int(settings["input_dim"]):
            raise ValueError("Baseline and reference must consume the same window and channels.")
        reference_settings["device"] = args.device
        reference = model_factory(SimpleNamespace(**reference_settings), metadata=None).to(args.device)
        load_ckpt(reference, args.reference_checkpoint, strict=True)
        reference.requires_grad_(False).eval()
        frozen = {key: value.detach().clone() for key, value in reference.state_dict().items()}

    records = read_records(dataset, data)
    sources = list(map(str, dataset["source_domains"]))
    if len(sources) < 2 or len({r["sample_rate_hz"] for r in records}) != 1:
        raise ValueError("D1 requires at least two source conditions with one sampling convention.")
    train = materialize([r for r in records if r["split"] == "update" and r["domain"] in sources], dataset, data)
    validation = materialize([r for r in records if r["split"] == "validation" and r["domain"] in sources], dataset, data)
    expected_channels = int(settings["in_channels"] if args.role == "reference" else settings["input_dim"])
    if any(u["x"].shape[-1] != expected_channels for u in train+validation):
        raise ValueError("Observed channels differ from the declared classifier input.")
    pools = group_pools(train, sources)
    if any(len(pool) < args.units_per_domain for pool in pools.values()):
        raise ValueError("Insufficient physical groups for the fixed source sampling budget.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best, best_epoch = float("inf"), None
    history, sampling, batches = [], [], []
    fit_seconds = selection_seconds = export_seconds = 0.
    beta = 0. if args.role == "reference" else .25
    scope = dict(status="running", role=args.role, seed=args.seed, permanent_test_predicted=False,
                 independent_assessment_performed=False, fit=_resources(train), selection=_resources(validation),
                 objective="ordinary_group_balanced_CE" if args.role == "reference" else "paired_CE_plus_0.25_Brier",
                 selector="worst_domain_CE" if args.role == "reference" else "worst_domain_CE_plus_0.25_Brier_excess",
                 checkpoint_tie_break="earliest_epoch", trainable_parameters=sum(p.numel() for p in model.parameters()))
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
                shift = pair_rng.randint(1, args.pair_shift) if args.role == "baseline" else 0
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
            score, rows, summary = evaluate(view, validation, args.device, 1., beta, pack_batch,
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
                saved = dict(kind="classifier", state_dict=model.state_dict(), model=settings, epoch=epoch)
                if reference is not None:
                    saved.update(reference_model=reference_settings, reference_state_dict=reference.state_dict(),
                                 reference_temperature=args.reference_temperature)
                torch.save(saved, output/"selected_candidate.pt")
                write_csv(output/"selected_source_validation.csv", summary)
                write_csv(output/"selected_source_validation_units.csv", rows)
                predictions = {key: np.concatenate(value) for key, value in arrays.items()}
                predictions.update(raw_class_names=np.asarray(class_names), candidate_class_names=np.asarray(class_names),
                                   arm=np.asarray("p0" if args.role == "reference" else "ResNet1D"), seed=np.asarray(args.seed))
                np.savez_compressed(output/"selected_source_validation_windows.npz", **predictions)
                export_seconds += _clock(args.device)-started
            print(history[-1], flush=True)
        if reference is not None:
            if reference.training or any(p.requires_grad or p.grad is not None for p in reference.parameters()) or not all(
                torch.equal(frozen[key], value) for key, value in reference.state_dict().items()
            ):
                raise AssertionError("Frozen reference parameters or buffers changed.")
        load_ckpt(model, str(output/"selected_candidate.pt"), strict=True)
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
        (output/"result_scope.json").write_text(json.dumps(scope, indent=2))
    print(output, flush=True)
    return output


def main() -> None:
    run(parser().parse_args())


if __name__ == "__main__":
    main()
