"""SegRNN, adapted from lss-1138/SegRNN under Apache-2.0.

Preserves RNN/GRU/LSTM, parallel/recurrent decoding, explicit channel IDs and
non-affine RevIN choices. Normalization statistics are local to each forward;
no global parameter object or mutable cross-batch RevIN state is required.
"""

import torch
from torch import nn

from .._native_forecasting import boolean, check_history, positive_int, probability


class Model(nn.Module):
    def __init__(self, args, metadata=None):
        super().__init__()
        self.seq_len = positive_int(args, "seq_len")
        self.pred_len = positive_int(args, "pred_len")
        self.input_dim = positive_int(args, "input_dim")
        self.seg_len = positive_int(args, "seg_len")
        self.d_model = positive_int(args, "d_model")
        dropout = probability(args, "dropout")
        self.channel_id = boolean(args, "channel_id")
        self.revin = boolean(args, "revin")
        self.rnn_type, self.dec_way = args.rnn_type, args.dec_way
        if self.rnn_type not in ("rnn", "gru", "lstm") or self.dec_way not in ("rmf", "pmf"):
            raise ValueError("SegRNN requires rnn_type=rnn/gru/lstm and dec_way=rmf/pmf")
        if self.seq_len % self.seg_len or self.pred_len % self.seg_len:
            raise ValueError("SegRNN history and horizon must be divisible by seg_len")
        if self.dec_way == "pmf" and self.channel_id and self.d_model % 2:
            raise ValueError("Parallel channel-ID decoding requires even d_model")
        self.seg_num_x, self.seg_num_y = self.seq_len // self.seg_len, self.pred_len // self.seg_len
        self.embedding = nn.Sequential(nn.Linear(self.seg_len, self.d_model), nn.ReLU())
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[self.rnn_type]
        self.rnn = rnn_cls(self.d_model, self.d_model, batch_first=True)
        if self.dec_way == "pmf":
            pos_dim = self.d_model // 2 if self.channel_id else self.d_model
            self.pos_emb = nn.Parameter(torch.randn(self.seg_num_y, pos_dim))
            if self.channel_id:
                self.channel_emb = nn.Parameter(torch.randn(self.input_dim, pos_dim))
        self.predict = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.d_model, self.seg_len))

    def forward(self, x, file_id=None, task_id=None, return_feature=False):
        check_history(x, self.seq_len, self.input_dim, task_id, return_feature)
        batch = x.shape[0]
        if self.revin:
            center = x.mean(1, keepdim=True).detach()
            scale = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        else:
            center, scale = x[:, -1:, :].detach(), 1.0
        segments = ((x - center) / scale).transpose(1, 2).reshape(-1, self.seg_num_x, self.seg_len)
        _, state = self.rnn(self.embedding(segments))
        if self.dec_way == "rmf":
            forecasts = []
            for _ in range(self.seg_num_y):
                hidden = state[0] if self.rnn_type == "lstm" else state
                forecast = self.predict(hidden).transpose(0, 1)
                forecasts.append(forecast)
                _, state = self.rnn(self.embedding(forecast), state)
            y = torch.cat(forecasts, dim=1).reshape(batch, self.input_dim, self.pred_len)
        else:
            if self.channel_id:
                pos = torch.cat((self.pos_emb[None].expand(self.input_dim, -1, -1),
                                 self.channel_emb[:, None].expand(-1, self.seg_num_y, -1)), dim=-1)
                pos = pos.reshape(-1, 1, self.d_model).repeat(batch, 1, 1)
            else:
                pos = self.pos_emb.repeat(batch * self.input_dim, 1).unsqueeze(1)
            def repeat_state(value):
                return value.repeat(1, 1, self.seg_num_y).reshape(1, -1, self.d_model)
            repeated = (tuple(repeat_state(value) for value in state)
                        if self.rnn_type == "lstm" else repeat_state(state))
            _, final = self.rnn(pos, repeated)
            hidden = final[0] if self.rnn_type == "lstm" else final
            y = self.predict(hidden).reshape(batch, self.input_dim, self.pred_len)
        return y.transpose(1, 2) * scale + center
