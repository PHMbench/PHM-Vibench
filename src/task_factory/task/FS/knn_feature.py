import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from ...Default_task import Default_task # Inherit from Default_task

class task(Default_task): # Changed inheritance
    """Few-shot classification task using KNN on features from a pretrained backbone."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        self.k = getattr(self.args_task, 'k_neighbors', 3) # Use self.args_task

        # Ensure network is not trainable for this baseline
        # This should ideally be controlled by optimizer configuration (not passing network params)
        # or by how the network is used (e.g., with torch.no_grad() in feature extraction).
        # Default_task's configure_optimizers will try to optimize self.network.parameters().
        # We need to ensure this task does not train the network.
        for param in self.network.parameters():
            param.requires_grad = False

    # forward method is inherited from Default_task

    def _shared_step(self, batch, stage):
        # KNN also uses the few-shot batch structure
        support_x, support_y = batch['support_x'], batch['support_y']
        query_x, query_y = batch['query_x'], batch['query_y']

        # Extract features with no gradient tracking for the network
        with torch.no_grad():
            support_features = self.network(support_x.to(self.device)) # Accessing self.network
            query_features = self.network(query_x.to(self.device))

        # Move features to CPU for sklearn
        support_features_np = support_features.detach().cpu().numpy()
        support_y_np = support_y.detach().cpu().numpy()
        query_features_np = query_features.detach().cpu().numpy()
        query_y_np = query_y.detach().cpu().numpy()

        # Fit KNN
        # Adjust k if n_samples_per_class in support set is less than self.k
        # This is a simplified approach; a more robust way would be to check per class.
        # For now, we assume n_samples_per_class is consistent across classes in an episode.
        n_samples_per_class_support = np.min(np.bincount(support_y_np)) if support_y_np.size > 0 else 0
        current_k = min(self.k, n_samples_per_class_support)
        
        acc = 0.0
        if current_k > 0:
            knn = KNeighborsClassifier(n_neighbors=current_k)
            knn.fit(support_features_np, support_y_np)
            query_preds = knn.predict(query_features_np)
            acc = np.mean(query_preds == query_y_np)
        
        # KNN does not have a natural loss function in the same way as gradient-based methods.
        # We can use a proxy like 0-1 loss (1 - accuracy) or skip logging loss.
        # For consistency with other methods, we'll use 1 - accuracy.
        loss = 1.0 - acc # Proxy loss

        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)
        
        if stage == 'test':
            return {'test_loss': torch.tensor(loss), 'test_acc': torch.tensor(acc)}
        # For training/validation, since no gradients are computed for the network,
        # the returned loss is for monitoring.
        return torch.tensor(loss, device=self.device, requires_grad=False)

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        # KNN baseline does not train any parameters.
        # Override Default_task's configure_optimizers.
        return None
