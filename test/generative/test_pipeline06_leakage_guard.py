from __future__ import annotations

import torch

from src.Pipeline_06_generative import _leakage_checks_from_train_batch


class _FailingDataFactory:
    def get_dataloader(self, split: str):  # type: ignore[no-untyped-def]
        raise RuntimeError(f"{split} dataloader unavailable")


def test_leakage_guard_error_is_not_reported_as_passed() -> None:
    fake = torch.randn(2, 2, 16)

    result = _leakage_checks_from_train_batch(
        _FailingDataFactory(),
        fake=fake,
        channels=2,
        threshold=1e-6,
    )

    assert result["split_guard_passed"] is False
    assert result["nearest_neighbor_check"] == "error"
    assert result["leakage_check_status"] == "error"
    assert "dataloader unavailable" in result["reason"]
