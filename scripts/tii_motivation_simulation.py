"""Bounded constructed S/U learning and readout-coordinate experiment.

The CLI executes the frozen 2026-10-17 protocol once on CPU. Tensor fixtures
have analytically defined units/support; they are not qualified industrial data.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace as NS
from typing import Any

import numpy as np
import pandas as pd
import pytorch_lightning
import scipy
from scipy.special import logsumexp
import torch

from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.tii_physical import project_window
from src.data_factory.tii_sampling import SourceRounds, fit_source_rms
from src.model_factory.ISFM.M_01_ISFM import Model
from src.task_factory.Components.tii_target_head import fit_head, make_episode, mean_pool
from src.task_factory.task.DG.tii_joint import task as JointTask


TASKS = ("common_informative", "increment_informative")
REPEATS = ((101, 0), (202, 1), (303, 2))
ARMS = ("support", "ordinary")


@dataclass(frozen=True)
class RunSpec:
    """Internal fixture sizes; the public CLI always uses these defaults."""

    steps: int = 300
    validation_interval: int = 50
    batch_size: int = 32
    train_records: int = 64
    validation_records: int = 32
    target_records: int = 128
    repeats: tuple[tuple[int, int], ...] = REPEATS
    tasks: tuple[str, ...] = TASKS

    def validate(self) -> None:
        if (self.steps < 1 or self.validation_interval < 1
                or self.steps % self.validation_interval or self.batch_size < 1):
            raise ValueError("positive steps/batch and complete validation intervals required")
        if any(n < 4 or n % 4 for n in (self.train_records, self.validation_records)):
            raise ValueError("source role counts must be positive multiples of four")
        if self.target_records < 16 or self.target_records % 4:
            raise ValueError("target count must be a multiple of four with fixed-episode coverage")
        if not self.tasks or any(t not in TASKS for t in self.tasks) or len(set(self.tasks)) != len(self.tasks):
            raise ValueError("unique declared simulation tasks required")
        if not self.repeats or len(set(self.repeats)) != len(self.repeats):
            raise ValueError("unique data/optimizer seed pairs required")
        if any(a < 0 or b < 0 for a, b in self.repeats):
            raise ValueError("nonnegative data and optimizer seeds required")


def write_json(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty scientific table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def generate_fixture(task_name: str, data_seed: int, spec: RunSpec) -> dict[str, Any]:
    """Independent records and roles; balanced independent latent frequencies."""
    if task_name not in TASKS:
        raise ValueError(f"undeclared task {task_name!r}")
    inventories: dict[str, dict[int, dict[str, Any]]] = {"source_train": {}, "source_val": {}, "target": {}}
    rows: list[dict[str, Any]] = []
    raw_blocks: dict[str, torch.Tensor] = {}
    for source in (1, 2, 3):
        rate = 16 if source == 2 else 64
        roles = (("target", spec.target_records),) if source == 3 else (
            ("source_train", spec.train_records), ("source_val", spec.validation_records))
        offset = 0
        for role_index, (role, count) in enumerate(roles):
            rng = np.random.default_rng(np.random.SeedSequence([data_seed, source, role_index]))
            latent = np.tile(np.array(((0, 0), (0, 1), (1, 0), (1, 1))), (count // 4, 1))
            rng.shuffle(latent)
            phases = rng.uniform(0, 2 * np.pi, size=(count, 2))
            amplitudes = rng.uniform(.8, 1.2, size=(count, 2))
            noise = rng.normal(0, .1, size=(count, rate))
            times = np.arange(rate, dtype=np.float64) / rate
            common_hz = np.where(latent[:, 0] == 0, 3., 5.)
            increment_hz = np.where(latent[:, 1] == 0, 12., 16.)
            wave = amplitudes[:, 0, None] * np.sin(2 * np.pi * common_hz[:, None] * times + phases[:, 0, None])
            if source != 2:
                wave += amplitudes[:, 1, None] * np.sin(2 * np.pi * increment_hz[:, None] * times + phases[:, 1, None])
            raw = torch.from_numpy((wave + noise).astype(np.float32))[:, :, None]
            labels = 1 - latent[:, 0] if source == 2 else latent[:, 0 if task_name == "common_informative" else 1]
            ids = [100_000_000 + data_seed * 10_000 + source * 1_000 + offset + i for i in range(count)]
            projected = [project_window(
                value, channel=0, native_rate_hz=rate, effective_rate_hz=rate,
                grid_rate_hz=64, duration_s=1, num_patches=4, patch_size=16,
                common_bands_hz=[(1, 8)], increment_bands_hz=[(8, 24)],
                common_support="fully_usable",
                increment_support="unavailable" if source == 2 else "fully_usable",
                input_unit="m/s²", output_unit="m/s²", unit_scale=1,
                support_basis="tensor_fixture", support_evidence="Analytically generated independent periodic records; protocol.md 2026-10-17",
            ) for value in raw]
            item = dict(x=torch.stack([p["common"] for p in projected]),
                        incremental=torch.stack([p["incremental"] for p in projected]),
                        availability=torch.tensor([int(p["increment_available"]) for p in projected]),
                        y=torch.from_numpy(labels.astype(np.int64)), file_id=ids,
                        recording_id=ids.copy(), group=ids.copy(), role=[role] * count)
            inventories[role][source] = item
            raw_blocks[f"{source}_{role}"] = raw
            for i, identity in enumerate(ids):
                rows.append(dict(recording_id=identity, group=identity, file_id=identity,
                                 dataset=source, role=role, label=int(labels[i]),
                                 latent_common=int(latent[i, 0]), latent_increment=int(latent[i, 1]),
                                 common_hz=float(common_hz[i]), increment_hz=float(increment_hz[i]) if source != 2 else "absent",
                                 phase_common=float(phases[i, 0]), phase_increment=float(phases[i, 1]) if source != 2 else "absent",
                                 amplitude_common=float(amplitudes[i, 0]), amplitude_increment=float(amplitudes[i, 1]) if source != 2 else "absent",
                                 native_rate_hz=rate, effective_rate_hz=rate, duration_s=1,
                                 unit="m/s²", support_basis="tensor_fixture", data_seed=data_seed, task=task_name))
            offset += count
    target = inventories["target"][3]
    episode = make_episode([dict(recording_id=i, group=g, label=int(y))
                            for i, g, y in zip(target["recording_id"], target["group"], target["y"])], shots=5, seed=1729)
    role_by_id = {r["recording_id"]: r["role"] for r in episode}
    target["role"] = [role_by_id[i] for i in target["recording_id"]]
    for row in rows:
        if row["dataset"] == 3:
            row["role"] = role_by_id[row["recording_id"]]
    all_ids = [r["recording_id"] for r in rows]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("constructed source roles and target must have disjoint records/groups")
    return dict(inventories=inventories, rows=rows, raw_blocks=raw_blocks, episode=episode)


def make_task(arm: str, metadata: MetadataAccessor, rms: float, spec: RunSpec,
              optimizer_seed: int = 0) -> JointTask:
    model_args = NS(type="ISFM", name="M_01_ISFM", embedding="SupportConditionedTokenizer",
                    token_organization=arm, backbone="B_04_Dlinear", task_head="H_01_Linear_cla",
                    output_dim=8, d_model=16, patch_size_L=16, patch_size_C=1,
                    num_patches=4, source_rms=rms)
    model = Model(model_args, metadata).cpu()
    return JointTask(model, NS(source_batch_size=spec.batch_size, rounds=spec.steps), model_args,
                     NS(loss="CE", optimizer="adamw", lambda_common=0, lambda_private=0,
                        scheduler=None, metrics=["acc"], source_system_ids=[1, 2], lr=.001, weight_decay=.0001),
                     NS(num_epochs=1, early_stopping=False, devices=1, monitor="val_group_nll",
                        monitor_mode="min", checkpoint_min_delta=1e-8, num_sanity_val_steps=0,
                        val_check_interval=spec.validation_interval, device="cpu"),
                     NS(seed=optimizer_seed), metadata).cpu()


def first_affine_diagnostics(model: Model) -> dict[str, Any]:
    """Unprojected branch-weight stack, equal to U's first linear input map.

    For S this omits the distinct physical projections and is not its effective
    input operator. A full-rank U stack excludes only that layer's collisions.
    """
    stacked = torch.cat([model.embedding.common_branch[0].weight,
                         model.embedding.increment_branch[0].weight], dim=0).detach().double()
    singular = torch.linalg.svdvals(stacked)
    return dict(rows=stacked.shape[0], columns=stacked.shape[1],
                operator="unprojected vertical branch-weight stack; actual first affine input map only for U",
                rank=int(torch.linalg.matrix_rank(stacked)), sigma_min=float(singular[-1]),
                rank_dtype="float64", rank_tolerance=float(max(stacked.shape) * torch.finfo(stacked.dtype).eps * singular[0]),
                singular_values=singular.tolist())


def source_diagnostics(joint: JointTask, inventories: dict[str, dict[int, dict[str, Any]]],
                       step: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Complete source inventories; validation selection uses the native estimator."""
    summaries: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    joint.eval()
    with torch.inference_mode():
        for role in ("source_train", "source_val"):
            if set(inventories[role]) != {1, 2}:
                raise ValueError(f"{role} must contain both sources")
            if role == "source_val":
                joint.on_validation_epoch_start()
            for source, original in inventories[role].items():
                if any(r != role for r in original["role"]):
                    raise ValueError(f"source diagnostics require exclusively {role}")
                batch = dict(original, source=source)
                joint._validate_source_batch(source, batch)
                logits = joint(batch).detach().double()
                if not torch.isfinite(logits).all():
                    raise ValueError("nonfinite source logits")
                nll = torch.nn.functional.cross_entropy(logits, batch["y"], reduction="none")
                correct = logits.argmax(-1) == batch["y"]
                summaries.append(dict(step=step, role=role, source=source,
                                      nll=float(nll.mean()), accuracy=float(correct.double().mean()), records=len(nll)))
                for i, identity in enumerate(batch["recording_id"]):
                    predictions.append(dict(step=step, role=role, source=source,
                                            recording_id=identity, group=batch["group"][i], label=int(batch["y"][i]),
                                            logit_0=float(logits[i, 0]), logit_1=float(logits[i, 1]),
                                            nll=float(nll[i]), correct=bool(correct[i])))
                if role == "source_val":
                    joint.validation_step(batch, 0)
    val_nll, val_accuracy = joint.validation_risk()
    if not np.isfinite(val_nll) or not np.isfinite(val_accuracy):
        raise ValueError("nonfinite native source-validation risk")
    for role in ("source_train", "source_val"):
        selected = [r for r in summaries if r["role"] == role]
        summaries.append(dict(step=step, role=role, source="equal_source_mean",
                              nll=float(np.mean([r["nll"] for r in selected])),
                              accuracy=float(np.mean([r["accuracy"] for r in selected])),
                              records=sum(r["records"] for r in selected)))
    expected = next(r["nll"] for r in summaries if r["source"] == "equal_source_mean" and r["role"] == "source_val")
    if not np.isclose(expected, val_nll, rtol=0, atol=1e-12):
        raise ValueError("source diagnostics disagree with native group-balanced estimator")
    return summaries, predictions, val_nll


def query_metrics(logits: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    if logits.shape != (len(labels), 2) or not np.isfinite(logits).all():
        raise ValueError("finite binary logits required for every declared query record")
    nll = logsumexp(logits, axis=1) - logits[np.arange(len(labels)), labels]
    return dict(nll=float(nll.mean()), accuracy=float((logits.argmax(-1) == labels).mean()))


def readout_diagnostic(features: np.ndarray, target: dict[str, Any], output: Path) -> dict[str, Any]:
    roles = np.asarray(target["role"])
    support, query = roles == "support", roles == "query"
    labels = target["y"].numpy()
    if (not support.any() or not query.any() or set(roles) != {"support", "query"}
            or set(np.asarray(target["group"])[support]) & set(np.asarray(target["group"])[query])):
        raise ValueError("fixed support and query must have complete, disjoint independent groups")
    if not np.array_equal(np.bincount(labels[support], minlength=2), [5, 5]):
        raise ValueError("the fixed target episode must retain exactly five records per class")
    groups = np.asarray(target["group"])[support].tolist()
    head = fit_head(features[support], labels[support], groups, .01)
    original = head.logits(features[query])
    transformed = (2 * features[query]) @ (head.weight / 2).T + head.bias
    difference = float(np.max(np.abs(original - transformed)))
    if not np.allclose(original, transformed, rtol=0, atol=1e-12):
        raise ValueError("inverse-weight scaling failed to preserve frozen logits")
    scaled = fit_head(2 * features[support], labels[support], groups, .01)
    refit = scaled.logits(2 * features[query])
    metrics = {"original": query_metrics(original, labels[query]),
               "inverse_weight": query_metrics(transformed, labels[query]),
               "scaled_refit": query_metrics(refit, labels[query])}
    summary = dict(alpha=.01, scale=2, logits_max_absolute_difference=difference,
                   inverse_weight_decisions_equal=bool(np.array_equal(original.argmax(-1), transformed.argmax(-1))),
                   scaled_refit_decision_changes=int(np.sum(original.argmax(-1) != refit.argmax(-1))),
                   original={k: v for k, v in asdict(head).items() if k not in ("weight", "bias")},
                   scaled_refit={k: v for k, v in asdict(scaled).items() if k not in ("weight", "bias")},
                   metrics=metrics)
    np.savez(output / "heads_and_query_logits.npz", weight=head.weight, bias=head.bias,
             scaled_weight=scaled.weight, scaled_bias=scaled.bias, query_labels=labels[query],
             query_ids=np.asarray(target["recording_id"])[query], original_logits=original,
             inverse_weight_logits=transformed, scaled_refit_logits=refit)
    write_json(output / "readout.json", summary)
    prediction_rows = []
    for i, identity in enumerate(np.asarray(target["recording_id"])[query]):
        row = dict(recording_id=int(identity), group=int(identity), label=int(labels[query][i]))
        for name, logits in (("original", original), ("inverse_weight", transformed), ("scaled_refit", refit)):
            row.update({f"{name}_logit_0": float(logits[i, 0]), f"{name}_logit_1": float(logits[i, 1]),
                        f"{name}_nll": float(logsumexp(logits[i]) - logits[i, labels[query][i]]),
                        f"{name}_decision": int(logits[i].argmax())})
        prediction_rows.append(row)
    write_csv(output / "query_predictions.csv", prediction_rows)
    return summary


def resources(start: float, before: resource.struct_rusage) -> dict[str, float | int]:
    after = resource.getrusage(resource.RUSAGE_SELF)
    return dict(wall_seconds=time.monotonic() - start, cpu_user_seconds=after.ru_utime - before.ru_utime,
                cpu_system_seconds=after.ru_stime - before.ru_stime,
                process_peak_rss_kib=int(after.ru_maxrss))


def run_fit(joint: JointTask, fixture: dict[str, Any], optimizer_seed: int,
            spec: RunSpec, output: Path) -> dict[str, Any]:
    output.mkdir()
    started, before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    write_json(output / "status.json", dict(status="running"))
    try:
        optimizer = joint.configure_optimizers()
        rounds = SourceRounds(fixture["inventories"]["source_train"], rounds=spec.steps,
                              batch_size=spec.batch_size, seed=optimizer_seed)
        initial_rank = first_affine_diagnostics(joint.network)
        curve, _, best = source_diagnostics(joint, fixture["inventories"], 0)
        write_csv(output / "source_curve.csv", curve)
        selected_step = 0
        selected = copy.deepcopy(joint.network.state_dict())
        selected_optimizer = copy.deepcopy(optimizer.state_dict())
        for step in range(1, spec.steps + 1):
            joint.train()
            optimizer.zero_grad(set_to_none=True)
            loss = joint.joint_loss(rounds[step - 1])
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite source loss at step {step}")
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in joint.parameters()):
                raise ValueError(f"nonfinite source gradient at step {step}")
            optimizer.step()
            if step % spec.validation_interval == 0:
                summaries, _, val_nll = source_diagnostics(joint, fixture["inventories"], step)
                curve.extend(summaries)
                write_csv(output / "source_curve.csv", curve)
                if val_nll < best - 1e-8:
                    best, selected_step = val_nll, step
                    selected = copy.deepcopy(joint.network.state_dict())
                    selected_optimizer = copy.deepcopy(optimizer.state_dict())
        terminal_rank = first_affine_diagnostics(joint.network)
        joint.network.load_state_dict(selected, strict=True)
        joint.eval()
        selected_rank = first_affine_diagnostics(joint.network)
        # Validate in the selection execution mode before freezing parameters;
        # disabling gradients can select a different float32 linear kernel.
        selected_summaries, selected_predictions, restored_val = source_diagnostics(joint, fixture["inventories"], selected_step)
        if restored_val != best:
            raise ValueError(f"selected state failed to restore source-only selection score: {restored_val!r} versus {best!r}")
        for parameter in joint.network.parameters():
            parameter.requires_grad_(False)
        torch.save(dict(model_state=selected, optimizer_state=selected_optimizer, selected_step=selected_step,
                        source_val_nll=best, source_rms=float(joint.network.embedding.source_rms),
                        model_config=vars(joint.network.args_m)), output / "selected.pt")
        write_csv(output / "source_selected_predictions.csv", selected_predictions)
        write_csv(output / "source_selected_metrics.csv", selected_summaries)
        target = fixture["inventories"]["target"][3]
        with torch.inference_mode():
            tokens = joint.network.encode(target["x"], incremental=target["incremental"], availability=target["availability"])
        features = mean_pool(tokens.cpu().numpy())
        np.savez(output / "frozen_features.npz", tokens=tokens.cpu().numpy(), features=features,
                 recording_ids=np.asarray(target["recording_id"]), groups=np.asarray(target["group"]),
                 roles=np.asarray(target["role"]), labels=target["y"].numpy())
        readout = readout_diagnostic(features, target, output)
        summary = dict(status="completed", selected_step=selected_step, source_val_nll=best,
                       source_train_nll=next(r["nll"] for r in selected_summaries
                                             if r["role"] == "source_train" and r["source"] == "equal_source_mean"),
                       initial_first_affine=initial_rank, selected_first_affine=selected_rank,
                       terminal_first_affine=terminal_rank, readout=readout, resources=resources(started, before))
        write_json(output / "status.json", summary)
        return summary
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", error_type=type(error).__name__,
                   error=str(error), traceback=traceback.format_exc(), resources=resources(started, before)))
        raise


def aggregate_results(rows: list[dict[str, Any]], spec: RunSpec) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    expected = {(t, ds, opt, arm) for t in spec.tasks for ds, opt in spec.repeats for arm in ARMS}
    table = {(r["task"], r["data_seed"], r["optimizer_seed"], r["arm"]): r for r in rows}
    if len(table) != len(rows) or set(table) != expected:
        raise ValueError("cannot aggregate incomplete or duplicate declared fits")
    differences = []
    for task_name in spec.tasks:
        for ds, opt in spec.repeats:
            s, u = (table[(task_name, ds, opt, arm)] for arm in ARMS)
            differences.append(dict(task=task_name, data_seed=ds, optimizer_seed=opt,
                                    nll_S_minus_U=s["query_nll"] - u["query_nll"],
                                    accuracy_S_minus_U=s["query_accuracy"] - u["query_accuracy"],
                                    S_nll=s["query_nll"], U_nll=u["query_nll"],
                                    S_scaled_refit_nll_change=s["scaled_refit_nll_change"],
                                    U_scaled_refit_nll_change=u["scaled_refit_nll_change"]))
    aggregates = []
    for task_name in spec.tasks:
        selected = [r for r in differences if r["task"] == task_name]
        for metric in ("nll_S_minus_U", "accuracy_S_minus_U", "S_nll", "U_nll",
                       "S_scaled_refit_nll_change", "U_scaled_refit_nll_change"):
            values = np.array([r[metric] for r in selected])
            aggregates.append(dict(task=task_name, metric=metric, repetitions=len(values), mean=float(values.mean()),
                                   sample_sd=float(values.std(ddof=1)) if len(values) > 1 else None,
                                   minimum=float(values.min()), maximum=float(values.max())))
    return differences, aggregates


def run(output: Path, spec: RunSpec, command: list[str]) -> None:
    """Execute one declared pass; preserve the failing fit and original exception."""
    spec.validate()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU protocol requires explicitly setting CUDA_VISIBLE_DEVICES=''")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started, before = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    write_json(output / "config.json", dict(
        spec=asdict(spec), command=command, code_commit=commit, evidence_kind="tensor_fixture", device="cpu",
        cuda_visible_devices=os.environ["CUDA_VISIBLE_DEVICES"], threads=torch.get_num_threads(),
        model=dict(name="M_01_ISFM", embedding="SupportConditionedTokenizer", backbone="B_04_Dlinear",
                   head="H_01_Linear_cla", output_dim=8, hidden_width=16, K=4, P=16),
        optimizer=dict(name="tii_joint.configure_optimizers AdamW", lr=.001, weight_decay=.0001,
                       betas=[.9, .999], eps=1e-8, scheduler=None),
        head=dict(implementation="tii_target_head.fit_head", alpha=.01, scale=2, episode_seed=1729, shots=5),
        generator=dict(duration_s=1, source_rates_hz={"full": 64, "common_only": 16}, target_rate_hz=64,
                       common_frequencies_hz=[3, 5], increment_frequencies_hz=[12, 16],
                       amplitude_uniform=[.8, 1.2], phase_uniform=[0, 2 * np.pi], gaussian_noise_std=.1,
                       common_bands_hz=[[1, 8]], increment_bands_hz=[[8, 24]], grid_rate_hz=64,
                       units="m/s²", source2_labels="one minus common latent", records_are_independent=True,
                       target_distribution="same generating distribution as full source, independent new records",
                       out_of_distribution_shift=False,
                       interpretation="independent target generalization under this generator"),
        thread_environment={key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        selection=dict(criterion="native equal-source/group source_val NLL", tie_tolerance=1e-8, tie="earliest"),
        versions=dict(python=platform.python_version(), platform=platform.platform(), numpy=np.__version__,
                      pandas=pd.__version__, scipy=scipy.__version__, torch=torch.__version__,
                      pytorch_lightning=pytorch_lightning.__version__)))
    results: list[dict[str, Any]] = []
    try:
        for task_name in spec.tasks:
            for data_seed, optimizer_seed in spec.repeats:
                repeat_dir = output / f"{task_name}_data{data_seed}_opt{optimizer_seed}"
                repeat_dir.mkdir()
                fixture = generate_fixture(task_name, data_seed, spec)
                write_csv(repeat_dir / "records.csv", fixture["rows"])
                write_csv(repeat_dir / "target_episode.csv", fixture["episode"])
                torch.save(dict(raw_blocks=fixture["raw_blocks"], inventories=fixture["inventories"]), repeat_dir / "fixture.pt")
                source_rows = [r for r in fixture["rows"] if r["dataset"] in (1, 2)]
                metadata = MetadataAccessor(pd.DataFrame([
                    dict(Id=r["file_id"], Dataset_id=r["dataset"], Label=r["label"],
                         Sample_rate=r["native_rate_hz"], Name=f"constructed_{r['dataset']}") for r in source_rows]))
                rms = fit_source_rms(fixture["inventories"]["source_train"])
                write_json(repeat_dir / "source_rms.json", dict(value=rms, fit_role="source_train", measure="equal-source/group/window", fitted_centering=False))
                paired = []
                for arm in ARMS:
                    torch.manual_seed(optimizer_seed)
                    paired.append(make_task(arm, metadata, rms, spec, optimizer_seed))
                states = [joint.network.state_dict() for joint in paired]
                matching = (states[0].keys() == states[1].keys()
                            and all(torch.equal(states[0][key], states[1][key]) for key in states[0]))
                if not matching:
                    raise ValueError("S/U must start from identical native state tensors")
                torch.save(states[0], repeat_dir / "matched_initial_state.pt")
                write_json(repeat_dir / "initialization.json", dict(all_state_tensors_exactly_equal=matching,
                           parameter_count=sum(p.numel() for p in paired[0].parameters()),
                           parameter_shapes={k: list(v.shape) for k, v in states[0].items()},
                           ordinary_first_affine=first_affine_diagnostics(paired[1].network),
                           optimizer_seed=optimizer_seed, sampling_seed=optimizer_seed,
                           sampling="same deterministic native SourceRounds per arm"))
                for arm, joint in zip(ARMS, paired):
                    print(f"RUN {task_name} data={data_seed} optimizer={optimizer_seed} arm={arm}", flush=True)
                    summary = run_fit(joint, fixture, optimizer_seed, spec, repeat_dir / arm)
                    readout = summary["readout"]
                    row = dict(task=task_name, data_seed=data_seed, optimizer_seed=optimizer_seed, arm=arm,
                               selected_step=summary["selected_step"], source_train_nll=summary["source_train_nll"],
                               source_val_nll=summary["source_val_nll"],
                               query_nll=readout["metrics"]["original"]["nll"],
                               query_accuracy=readout["metrics"]["original"]["accuracy"],
                               scaled_refit_nll=readout["metrics"]["scaled_refit"]["nll"],
                               scaled_refit_nll_change=readout["metrics"]["scaled_refit"]["nll"] - readout["metrics"]["original"]["nll"],
                               scaled_refit_decision_changes=readout["scaled_refit_decision_changes"],
                               original_head_gradient_inf=readout["original"]["gradient_inf"],
                               scaled_head_gradient_inf=readout["scaled_refit"]["gradient_inf"],
                               initial_affine_rank=summary["initial_first_affine"]["rank"],
                               selected_affine_rank=summary["selected_first_affine"]["rank"],
                               selected_affine_sigma_min=summary["selected_first_affine"]["sigma_min"],
                               inverse_weight_logits_difference=readout["logits_max_absolute_difference"],
                               wall_seconds=summary["resources"]["wall_seconds"])
                    results.append(row)
                    write_csv(output / "fit_results.csv", results)
        differences, aggregates = aggregate_results(results, spec)
        write_csv(output / "per_seed_differences.csv", differences)
        write_csv(output / "aggregate.csv", aggregates)
        write_json(output / "status.json", dict(status="completed", encoder_fits=len(results), head_fits=2 * len(results),
                   industrial_data_reads=0, gpu_workloads=0, resources=resources(started, before)))
    except Exception as error:
        write_json(output / "status.json", dict(status="failed", completed_encoder_fits=len(results),
                   error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), resources=resources(started, before)))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New derived-output directory; existing paths are refused")
    args = parser.parse_args()
    run(args.output, RunSpec(), [sys.executable, "-m", "scripts.tii_motivation_simulation", *sys.argv[1:]])


if __name__ == "__main__":
    main()
