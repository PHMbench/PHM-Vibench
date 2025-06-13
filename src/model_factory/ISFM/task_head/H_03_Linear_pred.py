import torch.nn as nn

class H_03_Linear_pred(nn.Module):
    """Simple head for time series prediction."""
    def __init__(self, args, num_dict=None):
        super().__init__()
        self.pred_len = getattr(args, 'pred_len', 1)
        d_model = args.output_dim
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x, data_name=False, task_name=False):
        x = self.linear(x)
        if x.dim() == 3:
            return x[:, -self.pred_len:, :]
        return x
