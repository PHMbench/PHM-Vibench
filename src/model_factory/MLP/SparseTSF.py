"""SparseTSF period-segment forecasting (lss-1138/SparseTSF, Apache-2.0).

Adaptation removes unused imports and exposes PHMFactory's BLC interface. The
original residual convolution, segment rearrangement and mean restoration stay.
"""

from torch import nn

from .._native_forecasting import check_history, positive_int


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.pred_len = positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.period_len = positive_int(args, "period_len")
        if self.seq_len % self.period_len or self.pred_len % self.period_len:
            raise ValueError("SparseTSF history and horizon must be divisible by period_len")
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.model_type = args.model_type
        if self.model_type not in ("linear", "mlp"):
            raise ValueError("SparseTSF model_type must be linear or mlp")
        self.conv = nn.Conv1d(1, 1, 1 + 2 * (self.period_len // 2),
                              padding=self.period_len // 2, bias=False)
        if self.model_type == "linear":
            self.projection = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            dim = positive_int(args, "d_model")
            self.projection = nn.Sequential(nn.Linear(self.seg_num_x, dim), nn.ReLU(),
                                             nn.Linear(dim, self.seg_num_y))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        mean = x.mean(1, keepdim=True)
        centered = (x - mean).transpose(1, 2)
        aggregated = self.conv(centered.reshape(-1, 1, self.seq_len)).reshape(
            batch, self.input_dim, self.seq_len) + centered
        segments = aggregated.reshape(-1, self.seg_num_x, self.period_len).transpose(1, 2)
        predicted = self.projection(segments).transpose(1, 2)
        return predicted.reshape(batch, self.input_dim, self.pred_len).transpose(1, 2) + mean
