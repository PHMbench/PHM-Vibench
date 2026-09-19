"""Synthetic saved-table plotting tests, not evidence of real D1 performance."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from PIL import Image

from experiments.p01 import plot_d1


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def saved_tables(root: Path, *, split: str = "test") -> dict[str, list[dict]]:
    root.mkdir()
    scopes = ("source", "unseen") if split == "test" else ("source",)
    tables: dict[str, list[dict]] = {
        "contrast_seed_summary.csv": [], "paired_contrasts.csv": [], "seed_summary.csv": [],
        "mechanism.csv": [],
    }
    for scope in scopes:
        for metric in plot_d1.METRICS:
            base = dict(split=split, scope=scope, metric=metric)
            summary = dict(base, seeds="[42, 123, 456]", status="complete", finite_seed_mean=.1,
                           seed_sd=.02, lower=-.1, upper=.2)
            for contrast in plot_d1.CONTRASTS:
                tables["contrast_seed_summary.csv"].append(dict(summary, contrast=contrast))
                for seed, value in zip(plot_d1.SEEDS, (.08, .1, .12)):
                    tables["paired_contrasts.csv"].append(dict(base, contrast=contrast, seed=seed,
                                                             value=value, lower=-.1, upper=.2))
            for arm in plot_d1.ARMS:
                tables["seed_summary.csv"].append(dict(summary, arm=arm))
        tables["mechanism.csv"].append(dict(name="selected", arm="RC", seed=42, split=split,
                                             role="adopted" if split == "test" else "direct", alpha=0.,
                                             predictor="deployed", scope=scope, A=0., b=0., sqrt_A=0.,
                                             b_over_sqrt_A="", alpha_sqrt_A=0.))
    for name, rows in tables.items():
        write_csv(root / name, rows)
    report = dict(bootstrap_repeats=2000, analysis_seed=20260919,
                  coverage=[dict(split=split, complete_core=True)])
    (root / "analysis.json").write_text(json.dumps(report))
    (root / "inputs.json").write_text(json.dumps(dict(condition_sets={split: {scope: [str(i)]
                                                                             for i, scope in enumerate(scopes)}})))
    return tables


def test_formats_captions_and_undefined_direction_from_saved_tables(tmp_path):
    source, output = tmp_path / "analysis", tmp_path / "preview"
    saved_tables(source)
    before = {path.name: path.read_bytes() for path in source.iterdir()}
    files = plot_d1.run(source, output, split="test")
    assert len(files) == 10
    for name in ("main_contrasts", "seed_dispersion", "adoption_mechanism"):
        assert (output / f"{name}.pdf").read_bytes().startswith(b"%PDF")
        assert "<svg" in (output / f"{name}.svg").read_text()
        with Image.open(output / f"{name}.png") as image:
            assert image.info["dpi"][0] == pytest.approx(300, abs=.1)
            assert image.width > 1000
    svg = (output / "adoption_mechanism.svg").read_text()
    assert "undefined" in svg and "A = 0" in svg and "α = 0.0" in svg
    caption = (output / "CAPTIONS.md").read_text()
    for required in ("not publication-validated", "contrast_seed_summary.csv", "paired_contrasts.csv",
                     "seed_summary.csv", "mechanism.csv", "not a confidence interval", "alpha = 0",
                     "source: conditions 0", "unseen: conditions 1"):
        assert required in caption
    assert before == {path.name: path.read_bytes() for path in source.iterdir()}
    with pytest.raises(FileExistsError):
        plot_d1.run(source, output, split="test")


@pytest.mark.parametrize("problem", ["missing_contrast", "partial", "nonfinite", "missing_adoption", "zero_quality"])
def test_incomplete_or_invalid_tables_fail_before_output(tmp_path, problem):
    source, output = tmp_path / "analysis", tmp_path / "preview"
    tables = saved_tables(source)
    if problem == "missing_contrast":
        tables["contrast_seed_summary.csv"].pop()
    elif problem == "partial":
        tables["seed_summary.csv"][0]["status"] = "partial"
    elif problem == "nonfinite":
        tables["paired_contrasts.csv"][0]["value"] = "nan"
    elif problem == "missing_adoption":
        for row in tables["mechanism.csv"]:
            row["role"] = "direct"
    else:
        tables["mechanism.csv"][0]["b_over_sqrt_A"] = 0.
    for name, rows in tables.items():
        write_csv(source / name, rows)
    with pytest.raises(ValueError):
        plot_d1.run(source, output, split="test")
    assert not output.exists()


def test_validation_does_not_require_or_invent_unseen_adoption(tmp_path):
    source = tmp_path / "source_analysis"
    saved_tables(source, split="validation")
    data = plot_d1._prepare(source, "validation")
    assert data["scopes"] == ("source",)
    assert data["adopted"] == []
    with pytest.raises(ValueError, match="explicitly"):
        plot_d1._prepare(source, "assessment")


def test_percentile_interval_need_not_contain_point_estimate(tmp_path):
    source = tmp_path / "analysis"
    tables = saved_tables(source)
    # The plot must use exact interval endpoints rather than requiring nonnegative xerr.
    for row in tables["contrast_seed_summary.csv"]:
        row.update(lower=.2, upper=.3, finite_seed_mean=.1)
    write_csv(source / "contrast_seed_summary.csv", tables["contrast_seed_summary.csv"])
    data = plot_d1._prepare(source, "test")
    row = data["contrasts"][0]
    assert float(row["finite_seed_mean"]) < float(row["lower"])
