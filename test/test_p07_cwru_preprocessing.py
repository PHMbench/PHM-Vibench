from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.utils.p07_protocol.cwru_manifest import ManifestSpecimen, WindowCoordinate
from src.utils.p07_protocol.cwru_preprocessing import (
    PREPROCESSING_PROTOCOL_ID,
    materialize_manifest_windows,
    standardize_window,
)


def _specimen() -> ManifestSpecimen:
    return ManifestSpecimen(
        specimen_key="fixture",
        metadata_id=1,
        file_name="fixture.mat",
        raw_size_bytes=1,
        raw_sha256="a" * 64,
        dataset_id=1,
        dataset_name="RM_001_CWRU",
        fault_type="IR",
        label=1,
        diameter_code="007",
        diameter_mils=7,
        fault_level=7,
        domain_id=0,
        load_hp=0,
        sample_rate_hz=12000,
        channels=2,
        sample_length=8,
        file_weight=1.0,
        windows=(
            WindowCoordinate(index=0, start=0, stop=4),
            WindowCoordinate(index=1, start=4, stop=8),
        ),
    )


def test_manifest_windows_are_stateless_standardized_and_input_is_immutable() -> None:
    source = torch.tensor(
        [[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, 7.0]] * 2,
        dtype=torch.float64,
    )
    before = source.clone()
    windows = materialize_manifest_windows(source, _specimen(), dtype=torch.float64)
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    assert windows.shape == (2, 4, 2)
    torch.testing.assert_close(
        windows.mean(dim=1), torch.zeros(2, 2, dtype=torch.float64), atol=1e-12, rtol=0
    )
    torch.testing.assert_close(
        windows.square().mean(dim=1).sqrt(),
        torch.ones(2, 2, dtype=torch.float64),
        atol=1e-12,
        rtol=0,
    )
    assert "STATELESS" not in PREPROCESSING_PROTOCOL_ID


def test_numpy_reader_value_is_accepted_without_mutation() -> None:
    value = np.arange(16, dtype=np.float64).reshape(8, 2)
    before = value.copy()
    result = materialize_manifest_windows(value, _specimen(), dtype=torch.float32)
    np.testing.assert_array_equal(value, before)
    assert result.dtype == torch.float32


@pytest.mark.parametrize(
    "value,match",
    (
        (torch.ones(8, 1), "must have shape"),
        (torch.ones(7, 2), "must have shape"),
        (torch.full((8, 2), float("nan")), "non-finite"),
    ),
)
def test_recording_contract_fails_closed(value: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        materialize_manifest_windows(value, _specimen(), dtype=torch.float64)


def test_constant_channel_is_rejected() -> None:
    value = torch.column_stack(
        (torch.arange(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64))
    )
    with pytest.raises(ValueError, match="constant channel"):
        materialize_manifest_windows(value, _specimen(), dtype=torch.float64)


def test_coordinate_order_overlap_and_bounds_are_rejected() -> None:
    source = torch.arange(16, dtype=torch.float64).reshape(8, 2)
    out_of_order = replace(
        _specimen(),
        windows=(
            WindowCoordinate(index=1, start=0, stop=4),
            WindowCoordinate(index=0, start=4, stop=8),
        ),
    )
    overlapping = replace(
        _specimen(),
        windows=(
            WindowCoordinate(index=0, start=0, stop=5),
            WindowCoordinate(index=1, start=4, stop=8),
        ),
    )
    outside = replace(
        _specimen(), windows=(WindowCoordinate(index=0, start=0, stop=9),)
    )
    with pytest.raises(ValueError, match="contiguous and ordered"):
        materialize_manifest_windows(source, out_of_order, dtype=torch.float64)
    with pytest.raises(ValueError, match="overlap"):
        materialize_manifest_windows(source, overlapping, dtype=torch.float64)
    with pytest.raises(ValueError, match="outside"):
        materialize_manifest_windows(source, outside, dtype=torch.float64)


def test_standardize_window_rejects_wrong_types() -> None:
    with pytest.raises(TypeError, match="torch.Tensor"):
        standardize_window(np.ones((4, 2)))
    with pytest.raises(TypeError, match="real floating"):
        standardize_window(torch.ones(4, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="non-empty"):
        standardize_window(torch.ones(4))
