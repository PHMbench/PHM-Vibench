import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
import numpy as np
from ...Default_task import Default_task # Inherit from Default_task

class task(Default_task): # Changed inheritance
    """Few-shot classification task using Matching Networks.
    The backbone network is trained episodically.
    """
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        # self.lr is inherited from Default_task (self.args_task.lr)
        # self.loss_fn from Default_task might be CrossEntropy. Matching Networks use NLLLoss with log-softmax.
        self.matching_loss_fn = nn.NLLLoss() 
        
    # forward method is inherited

    def _compute_similarity_logprobs(self, support_features, query_features, n_way, n_shot, n_query):
        """Computes log probabilities for query samples based on similarity to support samples.
        Args:
            support_features: Tensor of shape [N_way * N_shot, D_feature]
            query_features: Tensor of shape [N_way * N_query, D_feature]
            n_way: Number of classes in the episode
            n_shot: Number of support samples per class
            n_query: Number of query samples per class
        Returns:
            log_probs: Tensor of shape [N_way * N_query, N_way]
        """
        # Normalize features (cosine similarity)
        support_features_norm = F.normalize(support_features, p=2, dim=1)
        query_features_norm = F.normalize(query_features, p=2, dim=1)

        # Calculate cosine similarities
        # Transpose support_features_norm to [D_feature, N_way * N_shot]
        similarities = torch.mm(query_features_norm, support_features_norm.transpose(0, 1)) # [N_way*N_query, N_way*N_shot]
        
        # Reshape similarities to [N_way*N_query, N_way, N_shot] to average over shots for each way
        # This is a simplification. Original Matching Networks might use attention/softmax over similarities to support samples of each class.
        # A common approach is to sum similarities for each class and then apply softmax.
        
        # Create one-hot labels for support set
        support_y = torch.arange(n_way).repeat_interleave(n_shot).to(support_features.device) # [N_way*N_shot]
        support_y_one_hot = F.one_hot(support_y, num_classes=n_way).float() # [N_way*N_shot, N_way]

        # Weighted sum of one-hot support labels by similarities (attention mechanism)
        # similarities shape: (N_query_total, N_support_total)
        # support_y_one_hot shape: (N_support_total, N_way)
        # Resulting attention_weights shape: (N_query_total, N_way)
        attention_weights = F.softmax(similarities * 100, dim=1) # Scaling factor can be learned or fixed
        log_probs = (torch.mm(attention_weights, support_y_one_hot) + 1e-6).log()
        
        return log_probs

    def _shared_step(self, batch, stage):
        # Matching Networks use the few-shot batch structure
        support_x, query_x = batch['support_x'], batch['query_x']
        n_way = batch['n_way']
        n_shot = batch['n_shot']
        n_query = batch['n_query']

        support_features = self.network(support_x) # Accessing self.network
        query_features = self.network(query_x)
        
        log_probs = self._compute_similarity_logprobs(support_features, query_features, n_way, n_shot, n_query)
        query_y = torch.arange(n_way, device=log_probs.device).repeat_interleave(n_query)
        
        loss = self.matching_loss_fn(log_probs, query_y)
        acc = (log_probs.argmax(dim=1) == query_y).float().mean()

        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True if stage != 'train' else False)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True if stage != 'train' else False)
        
        if stage == 'test':
            return {'test_loss': loss, 'test_acc': acc}
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    # configure_optimizers can be inherited from Default_task if Adam with self.args_task.lr is sufficient
    # for training the backbone episodically.
