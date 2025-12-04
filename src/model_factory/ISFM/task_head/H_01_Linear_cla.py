import torch
import torch.nn as nn
from src.model_factory.ISFM.system_utils import normalize_system_ids


class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        super(H_01_Linear_cla, self).__init__()
        self.mutiple_fc = nn.ModuleDict()
        num_classes = args.num_classes
        for data_name, n_class in num_classes.items():
            # data_name 一般为 Dataset_id，对外统一使用 str(key)
            self.mutiple_fc[str(data_name)] = nn.Linear(args.output_dim, n_class)

    def forward(self, x, system_id=False, return_feature=False, **kwargs):
        """
        多系统线性分类头（当前假设一个 batch 只包含单一系统）。

        - x: [B, T, D] or [B, D]，先对时间维做平均池化；
        - system_id:
          * 标量 int/str：整批同一系统；
          * list/tuple/tensor：若传入 per-sample ID，当前实现会取第一个，假设 batch 内已按 Dataset_id 分组。
        - return_feature: 是否返回特征（除了logits外）。
        """
        if x.ndim == 3:
            x = x.mean(dim=1)  # [B, D]

        B = x.size(0)
        sid_tensor = normalize_system_ids(system_id, batch_size=B, device=x.device)
        sid = int(sid_tensor[0].item())
        key = str(sid)
        if key not in self.mutiple_fc:
            raise KeyError(f"Missing head for system_id '{key}' in H_01_Linear_cla.")

        head = self.mutiple_fc[key]
        logits = head(x)  # [B, num_classes_for_this_system]

        # 支持return_feature参数
        if return_feature:
            return logits, x
        else:
            return logits
