import torch
import torch.nn as nn
from src.model_factory.ISFM.system_utils import normalize_system_ids


class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        super(H_01_Linear_cla, self).__init__()
        self.mutiple_fc = nn.ModuleDict()
        num_classes = args.num_classes
        for data_name, n_class in num_classes.items():
            self.mutiple_fc[str(data_name)] = nn.Linear(args.output_dim, n_class)

    def forward(self, x, system_id=False, return_feature=False, **kwargs):
        """Apply the classification head for one explicitly homogeneous system batch."""
        if x.ndim == 3:
            x = x.mean(dim=1)

        batch_size = x.size(0)
        sid_tensor = normalize_system_ids(
            system_id,
            batch_size=batch_size,
            device=x.device,
        )
        if sid_tensor.numel() == 1 and batch_size > 1:
            sid_tensor = sid_tensor.expand(batch_size)
        elif sid_tensor.numel() != batch_size:
            raise ValueError(
                "H_01_Linear_cla requires one system ID or one ID per sample: "
                f"received {sid_tensor.numel()} IDs for batch_size={batch_size}."
            )

        unique_systems = torch.unique(sid_tensor)
        if unique_systems.numel() != 1:
            values = [int(value) for value in unique_systems.tolist()]
            raise ValueError(
                "H_01_Linear_cla requires a single Dataset_id per batch, "
                f"but received {values}. Use a system-homogeneous sampler or a "
                "head that explicitly supports mixed-system batches."
            )

        key = str(int(unique_systems[0].item()))
        if key not in self.mutiple_fc:
            raise KeyError(f"Missing head for system_id '{key}' in H_01_Linear_cla.")

        logits = self.mutiple_fc[key](x)
        if return_feature:
            return logits, x
        return logits
