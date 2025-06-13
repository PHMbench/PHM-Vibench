import torch
from torch import nn
import pytorch_lightning as pl  # Import PyTorch Lightning
from ...Default_task import Default_task  # Corrected import path


class task(Default_task):
    """Standard classification task, often used for pretraining a backbone."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        # The loss function is typically CrossEntropyLoss for classification
        # It might be already defined in Default_task or needs to be set here if specific to pretraining.
        # If Default_task handles classification loss and metrics, this can be very minimal.
        # self.loss_fn = nn.CrossEntropyLoss() # Example, if not in Default_task
        self.num_iterations = getattr(args_environment, 'iterations', 1)
        # Other pretraining-specific parameters can be initialized here.

    # The _compute_loss, training_step, validation_step, test_step, and configure_optimizers
    # are likely inherited from Default_task if it's designed for classification.
    # If Pretrain.py needs a different loss (e.g. MSELoss as originally shown, which is unusual for classification pretraining)
    # or different step logic, those methods should be overridden here.

    # Example of overriding if Default_task is too generic or for a specific pretraining setup:
    # def training_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self.network(x)
    #     loss = self.loss_fn(y_hat, y) # Ensure y is in the correct format for the loss
    #     acc = (y_hat.argmax(dim=1) == y).float().mean() # Example accuracy
    #     self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
    #     self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
    #     return loss

    # If `args_environment.iterations` is meant to control epochs or some outer loop
    # not directly managed by Lightning's trainer.fit() loop, that logic would be
    # handled outside this module, typically in the script that runs the training (e.g., Pipeline_02_pretrain_fewshot.py).
    # PyTorch Lightning's Trainer `max_epochs` parameter controls the number of epochs.
    # If 'iterations' means something else, like repeating the training N times with different seeds,
    # that would also be an outer loop in the script.

    # If the pretraining involves multiple iterations *within a single trainer.fit()* call in a custom way,
    # that would require more complex overriding of Lightning's training loop hooks, which is uncommon.
    # Typically, `iterations` would map to `max_epochs` or be handled by running `trainer.fit()` multiple times.

    # For now, assuming Default_task provides standard classification training_step, etc.
    # and `iterations` from `args_environment` is handled by the calling script (e.g. by setting max_epochs).
    pass
