from ...Default_task import Default_task
from typing import Any

class task(Default_task):
    """
    任务模块：分类任务
    """
    def __init__(self,network,
        args_data,  # Data args (Namespace)
        args_model,  # Model args (Namespace)
        args_task,  # Training args (Namespace)
        args_trainer,  # Trainer args (Namespace)
        args_environment,  # Environment args (Namespace)
        metadata):
        super().__init__(network,
        args_data,  # Data args (Namespace)
        args_model,  # Model args (Namespace)
        args_task,  # Training args (Namespace)
        args_trainer,  # Trainer args (Namespace)
        args_environment,  # Environment args (Namespace)
        metadata) 

    def forward(self, batch):
        # Classification always evaluates the trained candidate. The fusion
        # model's deployment forward has a separate probability-mixture policy.
        if self.args_model.name == "TSPN_fusion":
            return self.network.forward_details(batch["x"])["candidate_logits"]
        return super().forward(batch)
