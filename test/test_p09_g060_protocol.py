from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.p09.prepare_g060_protocol import (
    Record,
    build_cell,
    build_episode_core,
    domain_blocked_split,
    expected_cell_count,
    load_records,
    standardize_window,
)


def _records(count: int = 12) -> list[Record]:
    return [
        Record(
            record_id=index,
            system_id=1,
            canonical_label=0,
            domain_id=str(index % 4),
            sample_rate=12000.0,
        )
        for index in range(count)
    ]


def test_domain_blocked_split_is_deterministic_and_disjoint() -> None:
    first_adapt, first_query = domain_blocked_split(_records(), seed=17)
    second_adapt, second_query = domain_blocked_split(_records(), seed=17)
    assert [item.record_id for item in first_adapt] == [
        item.record_id for item in second_adapt
    ]
    assert [item.record_id for item in first_query] == [
        item.record_id for item in second_query
    ]
    assert len(first_adapt) == 4
    assert {item.record_id for item in first_adapt}.isdisjoint(
        {item.record_id for item in first_query}
    )
    assert len({item.domain_id for item in first_adapt}) == 4


def test_standardize_window_is_channelwise_and_finite() -> None:
    window = np.stack(
        (np.linspace(-4.0, 4.0, 1024), np.linspace(10.0, 40.0, 1024)),
        axis=1,
    )
    transformed = standardize_window(window, 1.0e-8)
    assert transformed.shape == (1024, 2)
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(transformed.std(axis=0), 1.0, atol=1.0e-6)


def _synthetic_split() -> tuple[dict[int, dict[str, list[int]]], dict[int, list[int]]]:
    split: dict[int, dict[str, list[int]]] = {}
    starts: dict[int, list[int]] = {}
    for class_id in range(4):
        adaptation = [100 * class_id + 1, 100 * class_id + 2]
        query = [100 * class_id + 51, 100 * class_id + 52]
        split[class_id] = {"adaptation": adaptation, "query": query}
        for record_id in adaptation + query:
            starts[record_id] = list(range(32))
    return split, starts


def test_core_and_cells_preserve_nested_support_and_fold_boundary() -> None:
    split, starts = _synthetic_split()
    core = build_episode_core(
        target_system=1,
        seed=42,
        episode=0,
        split=split,
        starts=starts,
        split_seed=20260801,
        candidate_windows=32,
        max_k=20,
        query_per_class=32,
    )
    assert len(core["support_max"]["2"]) == 20
    assert len(core["support_max"]["3"]) == 20
    assert all(len(values) == 32 for values in core["query"].values())

    cells = {}
    states = ["clean", "label_noise", "outlier", "imbalance"]
    for state_index, state in enumerate(states):
        for k_shot in ([5, 10, 20] if state == "imbalance" else [1, 5, 10, 20]):
            cells[(state, k_shot)] = build_cell(
                core=core,
                state=state,
                k_shot=k_shot,
                split=split,
                starts=starts,
                split_seed=20260801,
                state_index=state_index,
                candidate_windows=32,
                label_swap_probability=0.20,
                outlier_probability=0.20,
                imbalance_ratio=4,
            )
    assert cells[("clean", 1)]["support_counts"] == {"2": 1, "3": 1}
    assert cells[("clean", 20)]["support_counts"] == {"2": 20, "3": 20}
    assert sorted(cells[("imbalance", 20)]["support_counts"].values()) == [5, 20]

    for k_shot in (1, 5, 10, 20):
        noisy = cells[("label_noise", k_shot)]
        counts = defaultdict(int)
        for label in noisy["support_labels"]:
            counts[label] += 1
        assert counts[2] == counts[3] == k_shot
        assert sum(noisy["corruption_mask"]) == 2 * len(noisy["label_swap_pairs"])
        outlier = cells[("outlier", k_shot)]
        assert all(item[1] < 200 for item in outlier["outlier_replacement_keys"])
        query_records = {
            item[0] for values in core["query"].values() for item in values
        }
        assert query_records.isdisjoint(
            {item[1] for item in outlier["outlier_replacement_keys"]}
        )


def test_expected_grid_has_fifteen_cells_per_episode() -> None:
    assert expected_cell_count(6, 5, 100, [1, 5, 10, 20], [5, 10, 20]) == 45000


def test_load_records_ignores_nan_labels(tmp_path: Path) -> None:
    rows = []
    for class_id in range(4):
        for index in range(3):
            rows.append(
                {
                    "Id": 10 * class_id + index,
                    "Dataset_id": 1,
                    "Label": float(class_id),
                    "Domain_id": float(index),
                    "Sample_rate": 12000.0,
                }
            )
    rows.append(
        {
            "Id": 999,
            "Dataset_id": 1,
            "Label": np.nan,
            "Domain_id": np.nan,
            "Sample_rate": 12000.0,
        }
    )
    path = tmp_path / "metadata.xlsx"
    pd.DataFrame(rows).to_excel(path, index=False)
    by_system, by_id = load_records(
        path,
        [1],
        {1: {0: 0, 1: 1, 2: 2, 3: 3}},
    )
    assert len(by_id) == 12
    assert 999 not in by_id
    assert all(len(by_system[1][class_id]) == 3 for class_id in range(4))
