from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import yaml

from scripts.uxfd_gpu_queue import DEFAULT_QUEUE


DEFAULT_TEMPLATE_ROOT = Path("paper/UXFD_paper/results/sota_aggregate_templates")


@dataclass(frozen=True)
class SotaTemplateRecord:
    queue_id: str
    paper_id: str
    minimum_seeds: int
    baselines: int
    top_representatives: int
    template_path: str


@dataclass(frozen=True)
class SotaScaffoldReport:
    template_root: str
    records: Tuple[SotaTemplateRecord, ...]
    note: str


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _top_bindings_by_paper(queue: Mapping[str, Any]) -> Mapping[str, Tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, List[Mapping[str, Any]]] = {}
    for binding in queue.get("top_representative_bindings", ()):
        if not isinstance(binding, Mapping):
            continue
        paper_id = str(binding.get("paper_id", ""))
        if paper_id:
            grouped.setdefault(paper_id, []).append(binding)
    return {paper_id: tuple(bindings) for paper_id, bindings in grouped.items()}


def _matrix_entries(matrix_path: Path, section: str) -> Tuple[Mapping[str, Any], ...]:
    if not matrix_path.exists():
        return ()
    matrix = _load_yaml(matrix_path)
    entries = matrix.get(section, ())
    if section == "proposed" and isinstance(entries, Mapping):
        return (entries,)
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, Mapping))


def _statistics_template() -> Mapping[str, str]:
    return {
        "mean": "TODO: numeric mean",
        "std": "TODO: numeric standard deviation",
        "ci95_low": "TODO: numeric 95% CI low",
        "ci95_high": "TODO: numeric 95% CI high",
    }


def _accepted_run_ref_template(paper_id: str, entry_id: str, minimum_seeds: int) -> List[str]:
    safe_entry = entry_id or "ENTRY_ID"
    return [
        f"TODO: {paper_id}/{safe_entry}/seed_{seed_index}/run_meta.yaml"
        for seed_index in range(minimum_seeds)
    ]


def _comparison_template(
    paper_id: str,
    entry_id: str,
    label: str,
    role: str,
    minimum_seeds: int,
) -> Mapping[str, Any]:
    return {
        "entry_id": entry_id,
        "label": label,
        "role": role,
        "seed_values": f"TODO: list at least {minimum_seeds} matched integer seeds",
        "statistics": _statistics_template(),
        "effect_size_vs_proposed": (
            "TODO: finite effect size or omit if paired_test is filled"
        ),
        "paired_test": {
            "name": "TODO: e.g. paired_t_test or wilcoxon",
            "p_value": "TODO: finite p-value in [0, 1]",
        },
        "accepted_run_refs": _accepted_run_ref_template(paper_id, entry_id, minimum_seeds),
    }


def _template_payload(
    queue_item: Mapping[str, Any],
    baselines: Sequence[Mapping[str, Any]],
    top_bindings: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    paper_id = str(queue_item.get("paper_id", ""))
    minimum_seeds = int(queue_item.get("minimum_seeds", 3))
    proposed_entries = _matrix_entries(Path(str(queue_item.get("matrix_path", ""))), "proposed")
    proposed = proposed_entries[0] if proposed_entries else {}
    proposed_id = str(proposed.get("id", "P00"))
    proposed_label = str(proposed.get("label", "proposed method"))
    return {
        "template_only": True,
        "accepted_sota_evidence": False,
        "paper_id": paper_id,
        "source_queue_id": str(queue_item.get("queue_id", "")),
        "claim_scope": "TODO: exact_sota, representative_only, or bounded_non_sota",
        "primary_metric": "TODO: primary metric name",
        "protocol": {
            "minimum_seeds": minimum_seeds,
            "dataset_split": "TODO: same dataset split used by accepted runs",
            "preprocessing_signature": "TODO: sha256:<64 lowercase hex>",
            "evidence_level": "TODO: accepted_same_protocol",
            "source_tree_status": "TODO: clean",
        },
        "proposed": {
            "entry_id": proposed_id,
            "label": proposed_label,
            "seed_values": f"TODO: list at least {minimum_seeds} matched integer seeds",
            "statistics": _statistics_template(),
            "accepted_run_refs": _accepted_run_ref_template(
                paper_id,
                proposed_id,
                minimum_seeds,
            ),
        },
        "comparators": [
            _comparison_template(
                paper_id,
                str(entry.get("id", "")),
                str(entry.get("label", "")),
                "baseline",
                minimum_seeds,
            )
            for entry in baselines
            if entry.get("id")
        ],
        "top_representatives": [
            {
                "binding_id": str(binding.get("binding_id", "")),
                "external_work_id": str(binding.get("external_work_id", "")),
                "scope": "TODO: exact or representative",
                "seed_values": f"TODO: list at least {minimum_seeds} matched integer seeds",
                "statistics": _statistics_template(),
                "effect_size_vs_proposed": (
                    "TODO: finite effect size or omit if paired_test is filled"
                ),
                "paired_test": {
                    "name": "TODO: e.g. paired_t_test or wilcoxon",
                    "p_value": "TODO: finite p-value in [0, 1]",
                },
                "accepted_run_refs": _accepted_run_ref_template(
                    paper_id,
                    str(binding.get("binding_id", "")),
                    minimum_seeds,
                ),
            }
            for binding in top_bindings
        ],
        "claim_output": (
            "TODO: if proposed does not beat every accepted comparator, write "
            "bounded contribution wording and keep SOTA wording blocked"
        ),
    }


def create_scaffold(
    output_root: Path = DEFAULT_TEMPLATE_ROOT,
    queue_path: Path = DEFAULT_QUEUE,
) -> SotaScaffoldReport:
    queue = _load_yaml(queue_path)
    bindings_by_paper = _top_bindings_by_paper(queue)
    output_root.mkdir(parents=True, exist_ok=True)
    records: List[SotaTemplateRecord] = []

    for queue_item in queue.get("paper_queue", ()):
        if not isinstance(queue_item, Mapping):
            continue
        paper_id = str(queue_item.get("paper_id", ""))
        matrix_path = Path(str(queue_item.get("matrix_path", "")))
        baselines = _matrix_entries(matrix_path, "baselines")
        top_bindings = bindings_by_paper.get(paper_id, ())
        paper_dir = output_root / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        template_path = paper_dir / "sota_aggregate.template.yaml"
        template_path.write_text(
            yaml.safe_dump(
                _template_payload(queue_item, baselines, top_bindings),
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        (paper_dir / "README.md").write_text(
            "\n".join(
                [
                    "# UXFD SOTA Aggregate Template",
                    "",
                    "This directory is a scaffold, not accepted SOTA evidence.",
                    "Do not create `paper/UXFD_paper/results/sota_aggregates/<paper_id>`",
                    "until the accepted-run artifact gate passes with queue coverage.",
                    "Required preflight:",
                    "`python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage`.",
                    "After accepted run coverage exists, fill the template and copy it",
                    "to `paper/UXFD_paper/results/sota_aggregates/<paper_id>/sota_aggregate.yaml`.",
                    "Then run `python -m scripts.uxfd_sota_gate`.",
                    "Each `accepted_run_refs` item must point to an existing relative",
                    "`run_meta.yaml` under `paper/UXFD_paper/results/accepted_runs`.",
                    "",
                    f"- Paper: `{paper_id}`",
                    f"- Minimum seeds: `{int(queue_item.get('minimum_seeds', 3))}`",
                    f"- Baseline comparators: `{len(baselines)}`",
                    f"- TOP representative bindings: `{len(top_bindings)}`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        records.append(
            SotaTemplateRecord(
                queue_id=str(queue_item.get("queue_id", "")),
                paper_id=paper_id,
                minimum_seeds=int(queue_item.get("minimum_seeds", 3)),
                baselines=len(baselines),
                top_representatives=len(top_bindings),
                template_path=str(template_path),
            )
        )

    (output_root / "manifest.yaml").write_text(
        yaml.safe_dump([asdict(record) for record in records], sort_keys=False),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        render_markdown(records, output_root), encoding="utf-8"
    )
    return SotaScaffoldReport(
        template_root=str(output_root),
        records=tuple(records),
        note="templates only; not accepted SOTA evidence",
    )


def render_markdown(records: Sequence[SotaTemplateRecord], root: Path) -> str:
    lines = [
        "# UXFD SOTA Aggregate Templates",
        "",
        f"- Template root: `{root}`",
        f"- Templates: `{len(records)}`",
        "- Status: templates only; not accepted SOTA evidence.",
        "- Fill one `sota_aggregate.yaml` per paper only after accepted run coverage exists.",
        "- Activation preflight: `python -m scripts.uxfd_artifact_gate paper/UXFD_paper/results/accepted_runs --require-queue-coverage` must pass before creating `paper/UXFD_paper/results/sota_aggregates`.",
        "- Do not commit template-derived `sota_aggregate.yaml` files while `accepted_runs` has zero accepted records or incomplete queue coverage.",
        (
            "- Required statistics: per-seed values, finite mean/std/95% CI, "
            "and finite effect size or paired test p-value in [0, 1]."
        ),
        "- Required run refs: every proposed, baseline, and TOP entry lists existing relative `run_meta.yaml` paths under accepted_runs.",
        "",
        "| Queue | Paper | Minimum Seeds | Baselines | TOP Bindings | Template |",
        "|---|---|---:|---:|---:|---|",
    ]
    for record in records:
        lines.append(
            f"| `{record.queue_id}` | `{record.paper_id}` | {record.minimum_seeds} | "
            f"{record.baselines} | {record.top_representatives} | `{record.template_path}` |"
        )
    return "\n".join(lines) + "\n"


def build_payload(report: SotaScaffoldReport) -> Mapping[str, Any]:
    return asdict(report)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Create UXFD SOTA aggregate templates")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TEMPLATE_ROOT)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    report = create_scaffold(output_root=args.output_root, queue_path=args.queue)
    output = (
        json.dumps(build_payload(report), indent=2) + "\n"
        if args.format == "json"
        else render_markdown(report.records, Path(report.template_root))
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
