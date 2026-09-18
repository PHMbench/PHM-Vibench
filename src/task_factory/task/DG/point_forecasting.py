"""Point forecasting on explicit context-plus-horizon raw windows.

Data Factory still selects records/windows. This task alone derives the prefix
and future target; Default_task retains loss, metrics and optimizer ownership.
"""

from collections.abc import Mapping

import torch

from ...Default_task import Default_task


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


class task(Default_task):
    def __init__(self, network, args_data, args_model, args_task,
                 args_trainer, args_environment, metadata):
        self.context_length = _positive_integer(args_model.seq_len, "model.seq_len")
        self.horizon = _positive_integer(args_model.pred_len, "model.pred_len")
        expected = self.context_length + self.horizon
        if args_data.window_size != expected:
            raise ValueError(f"point_forecasting needs data.window_size={expected} (history + horizon)")
        if getattr(args_data, "normalization", None) != "none":
            raise ValueError("point_forecasting requires data.normalization=none; whole-window statistics leak future targets")
        for field in ("noise_snr", "train_noise_snr", "evaluation_noise_snr"):
            if getattr(args_data, field, None) is not None:
                raise ValueError(f"point_forecasting does not allow data.{field}; full-window noise scaling uses future values")
        if args_task.loss != "MSE":
            raise ValueError("point_forecasting currently defines horizon MSE; configure task.loss=MSE")
        if getattr(args_task, "model_task_id", None) != "forecasting":
            raise ValueError("point_forecasting requires task.model_task_id=forecasting")
        if not args_task.metrics or set(args_task.metrics) - {"mse", "mae"}:
            raise ValueError("point_forecasting requires explicit mse and/or mae estimators")
        super().__init__(network, args_data, args_model, args_task,
                         args_trainer, args_environment, metadata)

    def _window(self, batch):
        if not isinstance(batch, Mapping):
            raise TypeError("point_forecasting batch must be a mapping")
        window = batch["x"]
        if not isinstance(window, torch.Tensor) or window.ndim != 3:
            raise ValueError("point_forecasting requires a BLC tensor window")
        if window.shape[0] < 1 or window.shape[1] != self.context_length + self.horizon:
            raise ValueError("point_forecasting window must contain the complete history and horizon")
        if window.shape[2] != self.args_model.input_dim or window.dtype != torch.float32:
            raise ValueError("point_forecasting window channel count/dtype does not match model.input_dim/float32")
        if not torch.isfinite(window).all():
            raise FloatingPointError("point_forecasting window contains NaN or Inf")
        return window

    def forward(self, batch):
        window = self._window(batch)
        prediction = self.network(window[:, :self.context_length], batch["file_id"],
                                  self._resolve_model_task_id(batch))
        expected_shape = (window.shape[0], self.horizon, window.shape[2])
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected_shape:
            raise ValueError(f"point_forecasting model must return {expected_shape}; no slicing or broadcasting repair")
        return prediction

    def _compute_metrics(self, y_hat, y, data_name, stage):
        # Scalar MSE/MAE covers every horizon/channel element, including strided tails.
        return super()._compute_metrics(y_hat.reshape(-1), y.reshape(-1), data_name, stage)

    def _shared_step(self, batch, stage, task_id=False):
        window = self._window(batch)
        local_batch = dict(batch)
        # Class labels in the record catalogue are not forecasting targets.
        local_batch["y"] = window[:, self.context_length:]
        return super()._shared_step(local_batch, stage, task_id)
