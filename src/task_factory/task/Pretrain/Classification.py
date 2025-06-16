import torch
from torch import nn
import pytorch_lightning as pl  # Import PyTorch Lightning
from ...Default_task import Default_task  # Corrected import path


class task(Default_task):
    """Standard classification task, often used for pretraining a backbone."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
