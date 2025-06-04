import torch
from torch import nn
from ..Default_task import Default_task

class task(Default_task):
    """Time series prediction task used during pretraining."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        self.loss_fn = nn.MSELoss()

    def _compute_loss(self, y_hat, y):
        return self.loss_fn(y_hat, y.float())
