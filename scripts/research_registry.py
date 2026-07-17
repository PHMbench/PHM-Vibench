from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, List
from urllib.parse import urlparse


REGISTRY_COLUMNS = (
    "method_id",
    "title",
    "year",
    "publication_status",
    "venue",
    "identifier",
    "task_families",
    "contribution_kinds",
    "model_backbone",
    "objective_loss",
    "protocol_setting",
    "datasets",
    "paper_url",
    "publication_record_url",
    "code_url",
    "code_license_status",
    "code_license_expression",
    "repo_mapping",
    "implementation_maturity",
    "recommendation",
    "blockers",
    "verified_on",
)

PUBLICATION_STATUSES = {
    "peer_reviewed",
    "accepted",
    "preprint",
    "submission",
    "evidence_incomplete",
}
IMPLEMENTATION_MATURITIES = {
    "catalog_only",
    "research_only",
    "experimental_candidate",
    "exploratory_runtime",
    "benchmark_candidate",
    "benchmark_valid",
}
LICENSE_STATUSES = {"verified", "conflicting", "unknown", "not_applicable"}
RECOMMENDATIONS = {
    "implement_now",
    "implement_later",
    "catalog_only",
    "exclude_license",
    "needs_evidence",
}
TASK_FAMILIES = {
    "anomaly_detection",
    "benchmark",
    "classification",
    "domain_adaptation",
    "domain_generalization",
    "few_shot",
    "forecasting",
    "generation",
    "imputation",
    "pretraining",
    "protocol",
    "super_resolution",
}
CONTRIBUTION_KINDS = {"benchmark", "loss", "metric", "model", "sampler", "setting"}

_METHOD_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class ResearchRow:
    method_id: str
    title: str
    year: int
    publication_status: str
    venue: str
    identifier: str
    task_families: tuple[str, ...]
    contribution_kinds: tuple[str, ...]
    model_backbone: str
    objective_loss: str
    protocol_setting: str
    datasets: str
    paper_url: str
    publication_record_url: str
    code_url: str
    code_license_status: str
    code_license_expression: str
    repo_mapping: str
    implementation_maturity: str
    recommendation: str
    blockers: str
    verified_on: date


def _split_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.strip() for token in value.split(";") if token.strip())


def _is_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def read_registry(path: Path) -> List[ResearchRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = set(REGISTRY_COLUMNS) - set(fieldnames)
        extra = set(fieldnames) - set(REGISTRY_COLUMNS)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing columns: {sorted(missing)}")
            if extra:
                details.append(f"unexpected columns: {sorted(extra)}")
            raise ValueError("research registry schema mismatch: " + "; ".join(details))

        rows: List[ResearchRow] = []
        for line_number, raw in enumerate(reader, start=2):
            cell = lambda name: (raw.get(name) or "").strip()
            try:
                year = int(cell("year"))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: year must be an integer") from exc
            try:
                verified_on = date.fromisoformat(cell("verified_on"))
            except ValueError as exc:
                raise ValueError(
                    f"line {line_number}: verified_on must use YYYY-MM-DD"
                ) from exc

            rows.append(
                ResearchRow(
                    method_id=cell("method_id"),
                    title=cell("title"),
                    year=year,
                    publication_status=cell("publication_status"),
                    venue=cell("venue"),
                    identifier=cell("identifier"),
                    task_families=_split_tokens(cell("task_families")),
                    contribution_kinds=_split_tokens(cell("contribution_kinds")),
                    model_backbone=cell("model_backbone"),
                    objective_loss=cell("objective_loss"),
                    protocol_setting=cell("protocol_setting"),
                    datasets=cell("datasets"),
                    paper_url=cell("paper_url"),
                    publication_record_url=cell("publication_record_url"),
                    code_url=cell("code_url"),
                    code_license_status=cell("code_license_status"),
                    code_license_expression=cell("code_license_expression"),
                    repo_mapping=cell("repo_mapping"),
                    implementation_maturity=cell("implementation_maturity"),
                    recommendation=cell("recommendation"),
                    blockers=cell("blockers"),
                    verified_on=verified_on,
                )
            )
        return rows


def validate_rows(rows: Iterable[ResearchRow]) -> List[str]:
    errors: List[str] = []
    seen_ids: set[str] = set()
    for row in rows:
        prefix = row.method_id or "<missing method_id>"
        if not _METHOD_ID_RE.fullmatch(row.method_id):
            errors.append(f"{prefix}: method_id must match {_METHOD_ID_RE.pattern}")
        if row.method_id in seen_ids:
            errors.append(f"{prefix}: duplicate method_id")
        seen_ids.add(row.method_id)

        if not row.title:
            errors.append(f"{prefix}: title is required")
        if row.year not in {2025, 2026}:
            errors.append(f"{prefix}: year must be 2025 or 2026")
        if row.publication_status not in PUBLICATION_STATUSES:
            errors.append(f"{prefix}: unknown publication_status {row.publication_status!r}")
        if row.implementation_maturity not in IMPLEMENTATION_MATURITIES:
            errors.append(
                f"{prefix}: unknown implementation_maturity {row.implementation_maturity!r}"
            )
        if row.code_license_status not in LICENSE_STATUSES:
            errors.append(f"{prefix}: unknown code_license_status {row.code_license_status!r}")
        if row.recommendation not in RECOMMENDATIONS:
            errors.append(f"{prefix}: unknown recommendation {row.recommendation!r}")

        unknown_tasks = set(row.task_families) - TASK_FAMILIES
        if not row.task_families:
            errors.append(f"{prefix}: at least one task_family is required")
        if unknown_tasks:
            errors.append(f"{prefix}: unknown task_families {sorted(unknown_tasks)}")
        unknown_kinds = set(row.contribution_kinds) - CONTRIBUTION_KINDS
        if not row.contribution_kinds:
            errors.append(f"{prefix}: at least one contribution_kind is required")
        if unknown_kinds:
            errors.append(f"{prefix}: unknown contribution_kinds {sorted(unknown_kinds)}")

        if not _is_url(row.paper_url):
            errors.append(f"{prefix}: paper_url must be an http(s) URL")
        if row.publication_record_url and not _is_url(row.publication_record_url):
            errors.append(f"{prefix}: publication_record_url must be an http(s) URL")
        if row.code_url and not _is_url(row.code_url):
            errors.append(f"{prefix}: code_url must be an http(s) URL")

        if row.publication_status in {"peer_reviewed", "accepted"}:
            if not row.venue or not row.identifier or not row.publication_record_url:
                errors.append(
                    f"{prefix}: reviewed/accepted work requires venue, identifier, "
                    "and publication_record_url"
                )
        prepublication_overpromoted = (
            row.publication_status in {"preprint", "submission"}
            and row.implementation_maturity not in {"catalog_only", "research_only"}
        )
        if prepublication_overpromoted:
            errors.append(
                f"{prefix}: preprint/submission cannot exceed research_only maturity"
            )
        if (
            row.publication_status == "evidence_incomplete"
            and row.implementation_maturity != "catalog_only"
        ):
            errors.append(f"{prefix}: evidence_incomplete work must remain catalog_only")

        if row.code_url:
            if row.code_license_status == "not_applicable":
                errors.append(f"{prefix}: code URL cannot use not_applicable license status")
            if (
                row.code_license_status in {"verified", "conflicting"}
                and not row.code_license_expression
            ):
                errors.append(
                    f"{prefix}: verified/conflicting code license requires an expression"
                )
        else:
            if row.code_license_status != "not_applicable":
                errors.append(f"{prefix}: missing code URL must use not_applicable license status")
            if row.code_license_expression:
                errors.append(f"{prefix}: missing code URL cannot declare a code license")

        if row.code_license_status in {"unknown", "conflicting"} and not row.blockers:
            errors.append(f"{prefix}: unknown/conflicting license must be explained in blockers")
        if row.recommendation == "exclude_license" and row.code_license_status != "verified":
            errors.append(
                f"{prefix}: exclude_license requires a verified incompatible license"
            )
    return errors
