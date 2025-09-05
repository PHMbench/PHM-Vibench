import torch.nn as nn

class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        super(H_01_Linear_cla, self).__init__()
        self.mutiple_fc = nn.ModuleDict()
        num_classes = args.num_classes 
        for data_name, n_class in num_classes.items():
            self.mutiple_fc[str(data_name)] = nn.Linear(args.output_dim,
                                                   n_class)

    def forward(self, x, system_id = False, return_feature = False, **kwargs):
        # x: (B, T, d_model) 先对时间维度做平均池化
        x = x.mean(dim = 1)  # (B, d_model)
        logits = self.mutiple_fc[str(system_id)](x)
        return logits
