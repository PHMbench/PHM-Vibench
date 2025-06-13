import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from ...Default_task import Default_task # Inherit from Default_task

class task(Default_task): # Changed inheritance
    """Few-shot classification task using Prototypical Networks."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata) # Call Default_task's init
        # self.network is already set by Default_task
        # self.lr is already set by Default_task (as self.args_task.lr)
        # self.loss_fn is already set by Default_task if args_task.loss is 'CE' or similar,
        # but prototypical networks use a specific way of calculating loss (via distances to prototypes)
        # So, we might not use self.loss_fn directly from Default_task in _calculate_loss_acc.
        # We'll use nn.CrossEntropyLoss for the logits derived from distances.
        self.proto_loss_fn = nn.CrossEntropyLoss()

    # forward method is inherited from Default_task if it's just self.network(x)
    # If specific logic is needed for prototypical networks beyond backbone, override it.
    # For now, assume self.network(x) from Default_task is sufficient for feature extraction.

    def _compute_prototypes(self, support_features, support_y):
        """Computes class prototypes from support features.
        Args:
            support_features: Tensor of shape [N_support, D_feature]
            support_y: Tensor of shape [N_support]
        Returns:
            prototypes: Tensor of shape [N_way, D_feature]
        """
        classes = torch.unique(support_y)
        prototypes = []
        for c in classes:
            class_features = support_features[support_y == c]
            if class_features.numel() > 0:
                prototypes.append(class_features.mean(dim=0))
            else:
                # This case should ideally be prevented by the data sampler.
                # Fallback: add a zero prototype (not ideal, but prevents crash)
                # A warning could be logged here.
                prototypes.append(torch.zeros_like(support_features[0]) if support_features.numel() > 0 else torch.zeros(self.network.output_dim , device=support_y.device)) # Assuming network has output_dim attribute or similar
        
        if not prototypes: # Should not happen if support_y is not empty
             raise ValueError("Cannot compute prototypes. Support set might be empty or invalid, or feature_dim unknown for fallback.")

        prototypes = torch.stack(prototypes)
        return prototypes

    def _calculate_loss_acc(self, query_features, prototypes, query_y):
        """Calculates loss and accuracy for query samples against prototypes.
        Args:
            query_features: Tensor of shape [N_query, D_feature]
            prototypes: Tensor of shape [N_way, D_feature]
            query_y: Tensor of shape [N_query] (0-indexed for the episode's ways)
        Returns:
            loss: Scalar tensor
            acc: Scalar tensor
        """
        dists = torch.cdist(query_features, prototypes)
        logits = -dists
        # Use the specific loss function for prototypical networks
        loss = self.proto_loss_fn(logits, query_y)
        acc = (logits.argmax(dim=1) == query_y).float().mean()
        return loss, acc

    def _shared_step(self, batch, stage):
        # Batch structure for few-shot is different: {'support_x', 'support_y', 'query_x', 'query_y', 'n_way', ...}
        # Default_task._shared_step expects {'x', 'y', 'id', ...}
        # So, we must override _shared_step completely for few-shot tasks.

        support_x, support_y = batch['support_x'], batch['support_y']
        query_x, query_y = batch['query_x'], batch['query_y']
        
        # Extract features using the network from Default_task
        support_features = self.network(support_x) # Accessing self.network
        query_features = self.network(query_x)
        
        prototypes = self._compute_prototypes(support_features, support_y)
        loss, acc = self._calculate_loss_acc(query_features, prototypes, query_y)

        # Logging using self.log from pl.LightningModule
        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True if stage != 'train' else False)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True if stage != 'train' else False)
        
        if stage == 'test':
            # Default_task.test_step expects a dictionary of metrics to be logged by _log_metrics
            # However, for few-shot, the structure might be simpler or handled directly.
            # Let's return what's expected if we were to use _log_metrics, or simplify if not.
            # For now, returning loss directly for training_step, and dict for test_step if needed by a collector.
            return {'test_loss': loss, 'test_acc': acc} 
        return loss

    def training_step(self, batch, batch_idx):
        # Overrides Default_task.training_step
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        # Overrides Default_task.validation_step
        self._shared_step(batch, 'val') 

    def test_step(self, batch, batch_idx):
        # Overrides Default_task.test_step
        return self._shared_step(batch, 'test')

    # configure_optimizers can be inherited from Default_task if Adam with self.args_task.lr is sufficient.
    # If prototypical networks require specific optimizer setups, override it.
    # For now, assume Default_task.configure_optimizers is fine.
    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.network.parameters(), lr=self.args_task.lr)
    #     return optimizer
