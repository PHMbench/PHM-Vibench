"""Reanalyse frozen constructed-source S/U states without encoder retraining.

The source-training features determine one scalar per frozen encoder. The
existing fixed target episode and alpha are retained. This is a constructed
diagnostic, not an industrial transfer experiment.
"""
from __future__ import annotations

import argparse
import csv
import shlex
import json
import os
from pathlib import Path
import platform
import sys
from types import SimpleNamespace

import numpy as np
import scipy
from scipy.special import logsumexp
import torch

from src.model_factory.ISFM.backbone.B_04_Dlinear import B_04_Dlinear
from src.model_factory.ISFM.embedding.SupportConditionedTokenizer import SupportConditionedTokenizer
from src.task_factory.Components.tii_target_head import fit_head, mean_pool


ARTIFACTS = ORIGINAL = OUTPUT = AUXILIARY = None
PAPER = SOURCE_PREDICTIONS = THEORY = RANKS = None
TASKS = ("common_informative", "increment_informative")
REPEATS = ((101, 0), (202, 1), (303, 2))
ARMS = ("support", "ordinary")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_encoder(path: Path) -> tuple[SupportConditionedTokenizer, B_04_Dlinear]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    args = SimpleNamespace(**checkpoint["model_config"])
    embedding = SupportConditionedTokenizer(args).eval()
    backbone = B_04_Dlinear(args).eval()
    state = checkpoint["model_state"]
    for prefix, module in (("embedding.", embedding), ("backbone.", backbone)):
        module.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                if key.startswith(prefix)}, strict=True)
    return embedding, backbone


def metrics(logits: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, float, float]:
    log_probs = logits - logsumexp(logits, axis=1, keepdims=True)
    nll = -log_probs[np.arange(len(labels)), labels]
    entropy = -np.sum(np.exp(log_probs) * log_probs, axis=1)
    return nll, float(np.mean(logits.argmax(axis=1) == labels)), float(np.mean(entropy))


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This frozen-state diagnostic requires explicit CPU execution")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = {
        "purpose": "source-training feature RMS, fixed-alpha target head and U common-branch response",
        "source_artifact": str(ARTIFACTS),
        "original_results": str(ORIGINAL),
        "original_code_commit": "150d6976f3fd5b35f4da8c92dc5330b5859fe43f",
        "child_gitlink": "7ea774cbe6156b14c125806cbd79874db17eb09a",
        "device": "cpu", "torch_threads": 1, "alpha": 0.01,
        "fit_scope": "source_train only, two sources with 64 independent records each",
        "target_episode": "original PCG64 seed 1729, five support records per class",
        "leakage_scope": "full-band source 1 source_train, common branch before backbone",
        "encoder_updates": 0, "head_fits": 12,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "torch": torch.__version__},
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    model_rows: list[dict] = []
    prediction_rows: list[dict] = []
    leakage_rows: list[dict] = []
    heads: list[dict] = []
    for task in TASKS:
        for data_seed, optimizer_seed in REPEATS:
            repeat = f"{task}_data{data_seed}_opt{optimizer_seed}"
            fixture = torch.load(ARTIFACTS / repeat / "fixture.pt", map_location="cpu", weights_only=True)
            source = fixture["inventories"]["source_train"]
            if set(source) != {1, 2} or any(set(value["role"]) != {"source_train"} for value in source.values()):
                raise ValueError("source scale must use the original two source-training inventories")
            for arm in ARMS:
                embedding, backbone = load_encoder(ARTIFACTS / repeat / arm / "selected.pt")
                with np.load(ARTIFACTS / repeat / arm / "frozen_features.npz") as saved:
                    target_features = saved["features"].copy()
                    saved_tokens = saved["tokens"].copy()
                    roles = saved["roles"].copy()
                    labels = saved["labels"].copy()
                    groups = saved["groups"].copy()
                    ids = saved["recording_ids"].copy()
                target = fixture["inventories"]["target"][3]
                if (not np.array_equal(ids, target["recording_id"])
                        or not np.array_equal(roles, target["role"])
                        or not np.array_equal(labels, target["y"].numpy())):
                    raise ValueError("saved target features do not match the fixed episode")
                support, query = roles == "support", roles == "query"
                if (support.sum() != 10 or query.sum() != 118
                        or set(groups[support]) & set(groups[query])):
                    raise ValueError("fixed target episode does not have its recorded disjoint groups")
                with torch.inference_mode():
                    def encode(batch: dict) -> np.ndarray:
                        return backbone(embedding(batch["x"], batch["incremental"], batch["availability"])).numpy()

                    restored = encode(target)
                    source_features = np.concatenate([mean_pool(encode(source[k])) for k in sorted(source)])
                    if arm == "ordinary":
                        full = source[1]
                        c = full["x"] / embedding.source_rms.to(dtype=full["x"].dtype)
                        p = full["incremental"] / embedding.source_rms.to(dtype=full["x"].dtype)
                        common_with_increment = embedding.common_branch(c + p).double().numpy()
                        common_alone = embedding.common_branch(c).double().numpy()
                        numerator = float(np.mean(np.sum((common_with_increment - common_alone) ** 2, axis=-1)))
                        denominator = float(np.mean(np.sum(common_with_increment ** 2, axis=-1)))
                        leakage_rows.append(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed,
                                                 source=1, role="source_train", observations=len(c),
                                                 numerator=numerator, denominator=denominator,
                                                 leakage_rate=numerator / denominator))
                restore_error = float(np.max(np.abs(restored.astype(np.float64) - saved_tokens)))
                # Archived tokens are float32; CPU kernels can differ by a few
                # rounding units. The separate scale-invariance check is on NLL.
                if (not np.allclose(restored, saved_tokens, rtol=1e-5, atol=1e-6)
                        or not np.allclose(mean_pool(restored), target_features, rtol=1e-5, atol=1e-6)):
                    raise ValueError(f"restored encoder disagrees with archived frozen target features: {restore_error}")
                scale = float(np.sqrt(np.mean(source_features ** 2)))
                if not np.isfinite(scale) or scale <= 0:
                    raise ValueError("source-training feature RMS must be finite and positive")
                head = fit_head(target_features[support] / scale, labels[support], groups[support].tolist(), 0.01)
                query_features = target_features[query] / scale
                logits = head.logits(query_features)
                nll, accuracy, entropy = metrics(logits, labels[query])
                scaled_source_rms = float(np.sqrt(np.mean((2 * source_features) ** 2)))
                scaled_logits = head.logits(2 * target_features[query] / scaled_source_rms)
                scaled_nll, _, _ = metrics(scaled_logits, labels[query])
                nll_error = float(np.max(np.abs(nll - scaled_nll)))
                if nll_error >= 1e-6:
                    raise ValueError(f"global twofold scale invariance failed: {nll_error}")
                original = np.load(ARTIFACTS / repeat / arm / "heads_and_query_logits.npz")
                if not np.array_equal(original["query_ids"], ids[query]):
                    raise ValueError("original target query ordering changed")
                original_nll, original_accuracy, _ = metrics(original["original_logits"], labels[query])
                model_rows.append(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed, arm=arm,
                                       source_records=len(source_features), source_feature_rms=scale,
                                       target_support=int(support.sum()), target_query=int(query.sum()),
                                       original_nll=float(original_nll.mean()), original_accuracy=original_accuracy,
                                       normalized_nll=float(nll.mean()), normalized_accuracy=accuracy,
                                       normalized_entropy=entropy, head_iterations=head.iterations,
                                       head_gradient_inf=head.gradient_inf, head_objective=head.objective,
                                       restored_token_max_abs_error=restore_error,
                                       twofold_nll_max_abs_error=nll_error))
                heads.append(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed, arm=arm,
                                  weight=head.weight.tolist(), bias=head.bias.tolist()))
                for i, record_id in enumerate(ids[query]):
                    prediction_rows.append(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed,
                                                arm=arm, recording_id=int(record_id), group=int(groups[query][i]),
                                                label=int(labels[query][i]), logit_0=float(logits[i, 0]),
                                                logit_1=float(logits[i, 1]), nll=float(nll[i]),
                                                prediction=int(logits[i].argmax())))
    paired = []
    for task in TASKS:
        for data_seed, optimizer_seed in REPEATS:
            pair = {r["arm"]: r for r in model_rows if r["task"] == task and r["data_seed"] == data_seed}
            paired.append(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed,
                               original_s_minus_u=pair["support"]["original_nll"] - pair["ordinary"]["original_nll"],
                               normalized_s_minus_u=pair["support"]["normalized_nll"] - pair["ordinary"]["normalized_nll"]))
    write_csv(OUTPUT / "per_model.csv", model_rows)
    write_csv(OUTPUT / "per_seed_differences.csv", paired)
    write_csv(OUTPUT / "query_predictions.csv", prediction_rows)
    write_csv(OUTPUT / "leakage.csv", leakage_rows)
    (OUTPUT / "heads.json").write_text(json.dumps(heads, indent=2) + "\n", encoding="utf-8")
    print(f"Completed {len(model_rows)} fixed encoder states and {len(prediction_rows)} query predictions.")
    print(f"Maximum twofold NLL difference: {max(r['twofold_nll_max_abs_error'] for r in model_rows):.3g}")
    for task in TASKS:
        differences = [r["normalized_s_minus_u"] for r in paired if r["task"] == task]
        print(f"{task}: normalized S-U NLL mean={np.mean(differences):.9f}, sample SD={np.std(differences, ddof=1):.9f}")


def initial_leakage() -> None:
    """Reproduce the pretraining U-branch response without modifying results."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This frozen-state diagnostic requires explicit CPU execution")
    torch.set_num_threads(1)
    fields = ("task", "data_seed", "optimizer_seed", "initial_leakage_rate", "numerator", "denominator")
    writer = csv.DictWriter(sys.stdout, fieldnames=fields)
    writer.writeheader()
    for task in TASKS:
        for data_seed, optimizer_seed in REPEATS:
            repeat = ARTIFACTS / f"{task}_data{data_seed}_opt{optimizer_seed}"
            fixture = torch.load(repeat / "fixture.pt", map_location="cpu", weights_only=True)
            selected = torch.load(repeat / "ordinary/selected.pt", map_location="cpu", weights_only=True)
            state = torch.load(repeat / "matched_initial_state.pt", map_location="cpu", weights_only=True)
            embedding = SupportConditionedTokenizer(SimpleNamespace(**selected["model_config"])).eval()
            prefix = "embedding."
            embedding.load_state_dict({key[len(prefix):]: value for key, value in state.items()
                                       if key.startswith(prefix)}, strict=True)
            full = fixture["inventories"]["source_train"][1]
            with torch.inference_mode():
                rms = embedding.source_rms.to(dtype=full["x"].dtype)
                common = full["x"] / rms
                increment = full["incremental"] / rms
                full_output = embedding.common_branch(common + increment).double().numpy()
                common_output = embedding.common_branch(common).double().numpy()
                numerator = float(np.mean(np.sum((full_output - common_output) ** 2, axis=-1)))
                denominator = float(np.mean(np.sum(full_output ** 2, axis=-1)))
            writer.writerow(dict(task=task, data_seed=data_seed, optimizer_seed=optimizer_seed,
                                 initial_leakage_rate=numerator / denominator,
                                 numerator=numerator, denominator=denominator))


def cancellation_example() -> None:
    """Evaluate the fixed-state cross-band cancellation example in Appendix B."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("This mathematical example requires explicit CPU execution")
    torch.set_num_threads(1)
    states = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    common = torch.tensor([[float(t + n)] * 9 for t, n in states], dtype=torch.float64).reshape(4, 3, 3)
    increment = torch.tensor([[-float(n), 0, 0, float(n), 0, 0, 0, 0, 0]
                              for _, n in states], dtype=torch.float64).reshape(4, 3, 3)
    representations = {}
    for arm in ARMS:
        args = SimpleNamespace(token_organization=arm, patch_size_L=3, num_patches=3,
                               output_dim=2, d_model=1, source_rms=1.0)
        embedding = SupportConditionedTokenizer(args).double().eval()
        backbone = B_04_Dlinear(args).double().eval()
        with torch.no_grad():
            for branch, weights in ((embedding.common_branch, (1., 0., 0.)),
                                    (embedding.increment_branch, (0., 1., 0.))):
                branch[0].weight.copy_(torch.tensor([weights], dtype=torch.float64))
                branch[0].bias.zero_()
                branch[2].weight.fill_(1.)
                branch[2].bias.zero_()
            for layer in (backbone.Linear_Seasonal, backbone.Linear_Trend):
                layer.weight.zero_()
                layer.weight[:, 0] = 1.
                layer.bias.zero_()
            representations[arm] = mean_pool(backbone(embedding(common, increment, torch.ones(4))))
    support, ordinary = representations["support"], representations["ordinary"]
    if not np.allclose(support[1], support[2], rtol=0, atol=1e-14):
        raise ValueError("The two opposite-label support states do not collide")
    if not (np.isclose(ordinary[0, 0], ordinary[1, 0], rtol=0, atol=1e-14)
            and np.isclose(ordinary[2, 0], ordinary[3, 0], rtol=0, atol=1e-14)
            and abs(ordinary[0, 0] - ordinary[2, 0]) > 1e-14):
        raise ValueError("The union branch does not preserve the latent label")
    epsilon = 0.1
    entropy = -epsilon * np.log(epsilon) - (1 - epsilon) * np.log(1 - epsilon)
    support_bound = 0.5 * (np.log(2) + entropy)
    risk_gap_bound = support_bound - entropy
    rows = [dict(latent=t, nuisance=n, label_one_probability=1-epsilon if t == 1 else epsilon,
                 common_value=t+n, incremental_first=-n,
                 support_common=float(support[i, 0]), support_increment=float(support[i, 1]),
                 union_common=float(ordinary[i, 0]), union_increment=float(ordinary[i, 1]))
            for i, (t, n) in enumerate(states)]
    output = AUXILIARY / "cross_band_cancellation.csv"
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(scope="fixed-state constructed mathematical example, not trained-model comparison",
                  probabilities="independent uniform latent and nuisance; 10% label flip",
                  ambiguous_support_mass=0.5, union_optimal_unregularized_nll=float(entropy),
                  support_unrestricted_nll_lower_bound=float(support_bound),
                  support_minus_union_nll_lower_bound=float(risk_gap_bound),
                  command=shlex.join([sys.executable, *sys.argv]))
    with (AUXILIARY / "cross_band_cancellation.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def render_paper_tables() -> None:
    """Regenerate empirical manuscript tables from the retained result CSVs."""
    paper = PAPER
    source = SOURCE_PREDICTIONS
    with source.open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in predictions:
        logits = np.asarray(json.loads(row["logits"]), dtype=np.float64)
        loss = float(logsumexp(logits) - logits[int(row["true_label"])])
        grouped.setdefault((row["method"], row["dataset"], row["group"]), []).append(loss)
    means: dict[tuple[str, str], float] = {}
    for arm in ARMS:
        for source_id in ("1", "2"):
            means[arm, source_id] = float(np.mean([np.mean(losses) for (method, dataset, _), losses in grouped.items()
                                                    if method == arm and dataset == source_id]))
        means[arm, "equal"] = (means[arm, "1"] + means[arm, "2"]) / 2
    constructed = ["| Constructed source | U NLL | S NLL | S minus U |", "|---|---:|---:|---:|"]
    for label, source_id in (("Increment available", "1"), ("Common only", "2"), ("Equal-source mean", "equal")):
        u, s = means["ordinary", source_id], means["support", source_id]
        constructed.append(f"| {label} | {u:.6f} | {s:.6f} | {s-u:+.6f} |")
    constructed.extend(["", "Table: Descriptive source-validation risk from retained constructed-data predictions; not industrial transfer estimates. {#tab:constructed}", ""])
    with (OUTPUT / "per_seed_differences.csv").open(encoding="utf-8", newline="") as stream:
        pairs = list(csv.DictReader(stream))
    scales = ["| Label mechanism | Original S minus U NLL | Source-scale-normalized S minus U NLL |",
              "|---|---:|---:|"]
    for label, task in (("Common component", "common_informative"),
                        ("Incremental component", "increment_informative")):
        cells = []
        for column in ("original_s_minus_u", "normalized_s_minus_u"):
            values = [float(row[column]) for row in pairs if row["task"] == task]
            cells.append(f"${np.mean(values):.6f}\\pm{np.std(values, ddof=1):.6f}$")
        scales.append(f"| {label} | {cells[0]} | {cells[1]} |")
    scales.extend(["", "Table: Paired target-query NLL differences in nats, mean $\\pm$ sample standard deviation over three joint data/optimizer repetitions. Negative values favor S. The normalized diagnostic fits one scalar from each encoder's source-training features, then refits the target head with the original $\\alpha=0.01$. {#tab:scale-diagnostic}", ""])
    changes: dict[tuple[str, str], list[float]] = {}
    inverse_error, decision_changes, query_count = 0.0, 0, 0
    for task in TASKS:
        for data_seed, optimizer_seed in REPEATS:
            repeat = f"{task}_data{data_seed}_opt{optimizer_seed}"
            for arm in ARMS:
                with (ORIGINAL / repeat / arm / "query_predictions.csv").open(encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
                labels = np.asarray([int(row["label"]) for row in rows])
                logits = {kind: np.asarray([[float(row[f"{kind}_logit_{i}"]) for i in range(2)] for row in rows])
                          for kind in ("original", "inverse_weight", "scaled_refit")}
                original_nll, _, _ = metrics(logits["original"], labels)
                refit_nll, _, _ = metrics(logits["scaled_refit"], labels)
                changes.setdefault((task, arm), []).append(float(np.mean(refit_nll - original_nll)))
                inverse_error = max(inverse_error, float(np.max(np.abs(logits["inverse_weight"] - logits["original"]))))
                decision_changes += int(np.count_nonzero(logits["scaled_refit"].argmax(axis=1)
                                                         != logits["original"].argmax(axis=1)))
                query_count += len(rows)
    means = {key: float(np.mean(values)) for key, values in changes.items()}
    refit_lines = [
        f"A separate positive control used the original, unnormalized representations. Multiplying features by two "
        f"and applying the inverse weight change left saved logits unchanged to a maximum absolute error of "
        f"${inverse_error:.3g}$. Refitting the target head at the same $\\alpha=0.01$ reduced query NLL in "
        f"{sum(value < 0 for values in changes.values() for value in values)} of the "
        f"{sum(len(values) for values in changes.values())} fits and changed {decision_changes} of the "
        f"{query_count} recorded query decisions; episode records recur across arms and label tasks. "
        f"Mean refit-minus-original NLL changes for S and U were "
        f"${means['common_informative', 'support']:.6f}$ and ${means['common_informative', 'ordinary']:.6f}$ "
        f"in the common task, and ${means['increment_informative', 'support']:.6f}$ and "
        f"${means['increment_informative', 'ordinary']:.6f}$ nat in the incremental task "
        f"(three paired repetitions per task). These observations demonstrate sensitivity of the fixed-regularizer "
        f"readout; they do not quantify how much of the S/U difference is attributable to scale."
    ]
    original = paper.read_text(encoding="utf-8")
    content = original
    for name, lines in (("CONSTRUCTED TABLE", constructed), ("SCALE TABLE", scales),
                        ("SCALE-REFIT PARAGRAPH", refit_lines)):
        begin = f"<!-- BEGIN GENERATED {name} -->"
        end = f"<!-- END GENERATED {name} -->"
        before, rest = content.split(begin, 1)
        _, after = rest.split(end, 1)
        content = before + begin + "\n" + "\n".join(lines) + "\n" + end + after
    if content != original:
        paper.write_text(content, encoding="utf-8")
    theory = THEORY
    with (RANKS).open(encoding="utf-8", newline="") as stream:
        ranks = list(csv.DictReader(stream))
    lines = ["| Stored weights | Stacked rank | Smallest singular value | Spectral condition number |",
             "|---|---:|---:|---:|"]
    for row in ranks:
        lines.append(f"| {row['state']} | {row['stacked_rank']} | {float(row['smallest_singular_value']):.10f} "
                     f"| {float(row['spectral_condition_number']):.8f} |")
    lines.extend(["", "Table: First-affine matrices from the retained constructed-source states. The S row is a counterfactual U map, not the actual projected S operator. {#tab:stored-rank}", ""])
    original = theory.read_text(encoding="utf-8")
    begin, end = "<!-- BEGIN GENERATED RANK TABLE -->", "<!-- END GENERATED RANK TABLE -->"
    before, rest = original.split(begin, 1)
    _, after = rest.split(end, 1)
    content = before + begin + "\n" + "\n".join(lines) + "\n" + end + after
    if content != original:
        theory.write_text(content, encoding="utf-8")
    print("Regenerated constructed, scale and rank tables and the readout-refit paragraph from retained result CSVs.")


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact_dir', required=True, type=Path)
    parser.add_argument('--original_dir', required=True, type=Path)
    parser.add_argument('--output_dir', required=True, type=Path)
    parser.add_argument('--auxiliary_dir', required=True, type=Path)
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--source_predictions', type=Path)
    parser.add_argument('--theory', type=Path)
    parser.add_argument('--ranks', type=Path)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--render-paper-tables', action='store_true')
    actions.add_argument('--initial-leakage', action='store_true')
    actions.add_argument('--cancellation-example', action='store_true')
    args = parser.parse_args()
    global ARTIFACTS, ORIGINAL, OUTPUT, AUXILIARY, PAPER, SOURCE_PREDICTIONS, THEORY, RANKS
    ARTIFACTS, ORIGINAL, OUTPUT, AUXILIARY = (p.resolve() for p in
        (args.artifact_dir, args.original_dir, args.output_dir, args.auxiliary_dir))
    repo = Path(__file__).resolve().parents[1]
    if any(p == repo or repo in p.parents for p in (OUTPUT, AUXILIARY)):
        raise ValueError('results must be written outside the implementation repository')
    if args.render_paper_tables:
        if not all((args.paper, args.source_predictions, args.theory, args.ranks)):
            parser.error('paper-table rendering requires --paper, --source_predictions, --theory and --ranks')
        PAPER, SOURCE_PREDICTIONS, THEORY, RANKS = (p.resolve() for p in
            (args.paper, args.source_predictions, args.theory, args.ranks))
        render_paper_tables()
    elif args.initial_leakage:
        initial_leakage()
    elif args.cancellation_example:
        cancellation_example()
    else:
        main()


if __name__ == '__main__':
    cli()
