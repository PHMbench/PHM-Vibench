from types import SimpleNamespace

from src.trainer_factory import Default_trainer
from src.trainer_factory.Default_trainer import create_early_stopping_callback


def test_early_stopping_uses_configured_min_delta() -> None:
    callback = create_early_stopping_callback(
        SimpleNamespace(monitor="val_loss", patience=7, min_delta=1.0e-4)
    )

    assert callback.monitor == "val_loss"
    assert callback.patience == 7
    # Lightning stores the signed threshold internally for ``mode='min'``.
    assert abs(callback.min_delta) == 1.0e-4


def test_early_stopping_keeps_zero_default_for_legacy_configs() -> None:
    callback = create_early_stopping_callback(
        SimpleNamespace(monitor="val_loss", patience=5)
    )

    assert callback.min_delta == 0.0


def test_trainer_passes_deterministic_flag_to_lightning(monkeypatch, tmp_path) -> None:
    captured = {}

    monkeypatch.setattr(Default_trainer, "call_backs", lambda *args: [])
    monkeypatch.setattr(Default_trainer, "CSVLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        Default_trainer.pl,
        "Trainer",
        lambda **kwargs: captured.update(kwargs) or kwargs,
    )
    args_t = SimpleNamespace(
        num_epochs=1,
        gpus=1,
        pruning=0.0,
        device="cpu",
        deterministic=True,
        log_every_n_steps=1,
    )

    Default_trainer.trainer(
        SimpleNamespace(wandb=False, swanlab=False),
        args_t,
        SimpleNamespace(),
        str(tmp_path),
    )

    assert captured["deterministic"] is True
