"""P01 reference predictors with explicit metadata controls.

Raw/STFT references receive the same deployable condition metadata as the proposed
model. ``tspn`` is the original TSPN without the added context head;
``tspn_metadata`` adds the same context term; ``metadata_only`` measures shortcut
strength directly.
"""
from __future__ import annotations
from types import SimpleNamespace
import torch
from torch import nn
from .P01OperatorBias import _get, Model as OperatorModel


class Model(nn.Module):
    requires_physical_metadata = True
    _physical_metadata = OperatorModel._physical_metadata

    def __init__(self, args, metadata=None):
        super().__init__()
        self.metadata = metadata
        self.fs_field = str(_get(args, "fs_field", "sample_rate_hz"))
        self.rpm_field = str(_get(args, "rpm_field", "rotation_speed_rpm"))
        self.reference_speed_hz = float(_get(args, "reference_speed_hz", 10.0))
        self.kind = str(_get(args, "reference_kind", "raw_cnn"))
        c = int(_get(args, "in_channels", 1))
        k = int(_get(args, "num_classes", 3))
        width = int(_get(args, "reference_width", 16))
        self.n_fft = int(_get(args, "n_fft", 128))
        self.hop = int(_get(args, "hop_length", 32))
        self.register_buffer("window", torch.hann_window(self.n_fft))

        if self.kind == "raw_cnn":
            self.features = nn.Sequential(
                nn.Conv1d(c, width, 9, padding=4), nn.ReLU(),
                nn.Conv1d(width, width, 9, padding=4), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            )
            self.classifier = nn.Linear(width, k)
        elif self.kind == "stft_cnn":
            self.features = nn.Sequential(
                nn.Conv2d(c, width, 3, padding=1), nn.ReLU(),
                nn.Conv2d(width, width, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.classifier = nn.Linear(width, k)
        elif self.kind in {"tspn", "tspn_metadata"}:
            from .TSPN import Model as TSPN
            settings = dict(_get(args, "tspn", {}))
            settings.update(num_classes=k, in_channels=c, device="cpu")
            self.tspn = TSPN(SimpleNamespace(**settings), metadata)
        elif self.kind == "metadata_only":
            pass
        else:
            raise ValueError(
                "unknown reference_kind; use raw_cnn, stft_cnn, tspn, "
                "tspn_metadata or metadata_only"
            )

        self.use_context = self.kind in {
            "raw_cnn", "stft_cnn", "tspn_metadata", "metadata_only"
        }
        if self.use_context:
            self.metadata_head = nn.Linear(2, k, bias=False)

    def _context(self, fs, speed):
        return torch.stack(
            (torch.log(fs / 1000.0), torch.log(speed / self.reference_speed_hz)),
            dim=-1,
        )

    def forward_details(self, x, data_id=None, task_id=None, *, physical_metadata=None):
        fs, speed = self._physical_metadata(x, data_id, physical_metadata)
        if self.kind == "raw_cnn":
            logits = self.classifier(self.features(x.transpose(1, 2)))
        elif self.kind == "stft_cnn":
            b, n, c = x.shape
            if n < self.n_fft:
                raise ValueError("signal shorter than n_fft")
            z = torch.stft(
                x.transpose(1, 2).reshape(b * c, n), self.n_fft, self.hop,
                window=self.window, center=False, return_complex=True, normalized=True
            )
            z = torch.log1p(z.abs().square()).reshape(b, c, z.shape[-2], z.shape[-1])
            logits = self.classifier(self.features(z))
        elif self.kind in {"tspn", "tspn_metadata"}:
            logits = self.tspn(x, data_id, task_id)
        else:
            logits = x.new_zeros((x.shape[0], self.metadata_head.out_features))

        context = self._context(fs, speed)
        if self.use_context:
            logits = logits + self.metadata_head(context)
        return {"logits": logits, "context_features": context}

    def forward(self, x, data_id=None, task_id=None, *, physical_metadata=None):
        return self.forward_details(
            x, data_id, task_id, physical_metadata=physical_metadata
        )["logits"]
