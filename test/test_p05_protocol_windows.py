from types import SimpleNamespace

import numpy as np
import pytest

from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.protocol_transforms import exact_evenly_spaced_spans


def _args(*, count=4, window_size=10):
    return SimpleNamespace(
        split=SimpleNamespace(strategy="preassigned_metadata"),
        window_size=window_size,
        stride=window_size,
        train_ratio=0.8,
        num_window=count,
        window_sampling_strategy="evenly_spaced",
        dtype="float32",
        normalization="none",
    )


def test_integer_floor_spans_are_exact_and_nonoverlapping():
    spans = exact_evenly_spaced_spans(data_length=44, window_size=10, count=4)

    assert [(span.start, span.end) for span in spans] == [
        (0, 10),
        (11, 21),
        (22, 32),
        (34, 44),
    ]
    assert all(right.start >= left.end for left, right in zip(spans, spans[1:]))


@pytest.mark.parametrize(
    ("data_length", "window_size", "count", "message"),
    [
        (9, 10, 4, "shorter"),
        (39, 10, 4, "overlap"),
        (44, 10, 1, "at least two"),
    ],
)
def test_protocol_spans_fail_instead_of_padding_reducing_or_overlapping(
    data_length, window_size, count, message
):
    with pytest.raises(ValueError, match=message):
        exact_evenly_spaced_spans(data_length, window_size, count)


def test_dataset_exports_stable_window_provenance_and_preserves_count():
    raw = np.arange(44 * 2, dtype=np.float64).reshape(44, 2, 1)
    metadata = {
        17: {
            "Label": 1,
            "Protocol_Group": "XJTU/35Hz12kN/Bearing1_1",
        }
    }

    dataset = Default_dataset(
        {17: raw},
        metadata,
        _args(),
        SimpleNamespace(),
        mode="train",
    )

    assert len(dataset) == 4
    assert [dataset[index]["window_start"] for index in range(4)] == [0, 11, 22, 34]
    assert dataset[2]["sample_id"] == "17:22:32"
    assert dataset[2]["record_id"] == "17"
    assert dataset[2]["group_id"] == "XJTU/35Hz12kN/Bearing1_1"
    assert dataset[2]["window_index"] == 2
    assert dataset[2]["window_end"] == 32
    assert dataset[2]["x"].shape == (10, 2)
    assert dataset[2]["x"].dtype == np.float32
    assert dataset.data is None
    assert all(not np.shares_memory(dataset[index]["x"], raw) for index in range(4))


def test_dataset_rejects_wrong_cached_channel_shape():
    metadata = {17: {"Label": 1, "Protocol_Group": "XJTU/g"}}
    with pytest.raises(ValueError, match="exact raw shape"):
        Default_dataset(
            {17: np.zeros((44, 1, 1), dtype=np.float64)},
            metadata,
            _args(),
            SimpleNamespace(),
        )


def test_dataset_requires_protocol_group():
    with pytest.raises(ValueError, match="Protocol_Group"):
        Default_dataset(
            {17: np.zeros((44, 2, 1), dtype=np.float64)},
            {17: {"Label": 1}},
            _args(),
            SimpleNamespace(),
        )
