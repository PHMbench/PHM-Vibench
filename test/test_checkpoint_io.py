import pickle

import pytest

from src.utils import checkpoint_io


def test_safe_torch_load_does_not_fallback_on_runtime_error(tmp_path, monkeypatch):
    checkpoint = tmp_path / "bad.ckpt"
    checkpoint.write_bytes(b"not a checkpoint")
    monkeypatch.setenv("PHM_TRUSTED_CHECKPOINT_ROOTS", str(tmp_path))
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        raise RuntimeError("corrupt archive")

    monkeypatch.setattr(checkpoint_io.torch, "load", fake_load)

    with pytest.raises(RuntimeError, match="corrupt archive"):
        checkpoint_io.safe_torch_load(checkpoint)

    assert len(calls) == 1
    assert calls[0]["weights_only"] is True


def test_safe_torch_load_fallback_is_limited_to_weights_only_unpickling(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "lightning.ckpt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setenv("PHM_TRUSTED_CHECKPOINT_ROOTS", str(tmp_path))
    calls = []

    def fake_load(*args, **kwargs):
        calls.append(kwargs)
        if kwargs.get("weights_only") is True:
            raise pickle.UnpicklingError("weights-only load failed")
        return {"ok": True}

    monkeypatch.setattr(checkpoint_io.torch, "load", fake_load)

    assert checkpoint_io.safe_torch_load(checkpoint) == {"ok": True}
    assert [call["weights_only"] for call in calls] == [True, False]
