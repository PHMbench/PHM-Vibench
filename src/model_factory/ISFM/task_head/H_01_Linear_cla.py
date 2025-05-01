import torch.nn as nn

class H_01_Linear_cla(nn.Module):
    def __init__(self, args, args_d):
        super(H_01_Linear_cla, self).__init__()
        self.mutiple_fc = nn.ModuleDict()
        for data_name, dataset_dict in args_d.task.items():
            self.mutiple_fc[data_name] = nn.Linear(args.output_dim,
                                                   dataset_dict['n_classes'])

    def forward(self, x, data_name = False, task_name = False):
        # x: (B, T, d_model) 先对时间维度做平均池化
        x = x.mean(dim = 1)  # (B, d_model)
        logits = self.mutiple_fc[data_name](x)
        return logits