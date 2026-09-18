"""NLinear point forecasting, adapted from cure-lab/LTSF-Linear.

Copyright 2022 DLinear Authors. All rights reserved. Apache-2.0; see the
repository LICENSE. Adaptation: explicit BLC interface and validation, retaining
last-value detachment and shared/per-channel linear projections.
"""

import torch
from torch import nn

from .._native_forecasting import boolean, check_history, positive_int


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.pred_len = positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.individual = boolean(args, "individual")
        self.linear = (nn.ModuleList([nn.Linear(self.seq_len, self.pred_len)
                                     for _ in range(self.input_dim)])
                       if self.individual else nn.Linear(self.seq_len, self.pred_len))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        last = x[:, -1:, :].detach()
        centered = x - last
        if self.individual:
            forecast = torch.stack([layer(centered[:, :, i])
                                    for i, layer in enumerate(self.linear)], dim=-1)
        else:
            forecast = self.linear(centered.transpose(1, 2)).transpose(1, 2)
        return forecast + last
