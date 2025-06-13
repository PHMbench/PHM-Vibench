import torch
import torch.nn.functional as F
from torch import nn
import pytorch_lightning as pl
from ...Default_task import Default_task # Inherit from Default_task
# from .utils import model_weights_init # Assuming utils.py is in the same directory

class task(Default_task): # Changed inheritance
    """Few-shot classification task using Finetuning (last layer or whole model)."""
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__(network, args_data, args_model, args_task, args_trainer, args_environment, metadata)
        
        # Ensure feature_dim is correctly obtained. Default_task doesn't enforce network.output_dim
        # This needs to be a known quantity, e.g. from args_model or by probing the network.
        # For now, assume it's passed or handled correctly by the network's structure.
        if hasattr(self.network, 'output_dim'):
            self.feature_dim = self.network.output_dim
        elif hasattr(self.args_model, 'feature_dim'): # Check if it's in model args
             self.feature_dim = self.args_model.feature_dim
        else:
            # Attempt to infer feature dimension by doing a dummy forward pass
            # This is a common practice but requires a sample input shape.
            # This might be too complex for __init__; better to require it in config.
            raise ValueError("FinetuningTask requires 'network.output_dim' or 'args_model.feature_dim' to be defined.")

        # self.num_classes_target = self.args_task.num_classes_target # N-way for the target few-shot task (dynamic from batch['n_way'])
        # self.lr is inherited (self.args_task.lr) - this is for the meta-optimizer if used
        self.finetune_lr = getattr(self.args_task, 'finetune_lr', 1e-4)
        self.finetune_steps = getattr(self.args_task, 'finetune_steps', 10)
        self.finetune_mode = getattr(self.args_task, 'finetune_mode', 'last_layer') 
        
        # self.loss_fn is inherited (e.g., CrossEntropyLoss for the finetuned head)
        # We will use the self.loss_fn from Default_task for the finetuned head.

    # forward method is inherited

    def _finetune_and_evaluate(self, support_x, support_y, query_x, query_y, n_way):
        """ Finetunes a new head (and optionally backbone) on support set and evaluates on query set. """
        
        temp_classifier_head = nn.Linear(self.feature_dim, n_way).to(self.device)
        # Initialize weights (optional, but can be good practice)
        # temp_classifier_head.apply(model_weights_init) # Assuming model_weights_init is available

        params_to_finetune = list(temp_classifier_head.parameters())
        original_backbone_training_state = self.network.training

        if self.finetune_mode == 'whole_model':
            self.network.train() # Set backbone to train mode
            params_to_finetune.extend(self.network.parameters())
        else: # 'last_layer' or default
            self.network.eval() # Backbone features are fixed
        
        optimizer_finetune = torch.optim.Adam(params_to_finetune, lr=self.finetune_lr)

        # 3. Finetuning loop
        temp_classifier_head.train()
        for _ in range(self.finetune_steps):
            optimizer_finetune.zero_grad()
            # Use self.network (potentially from Default_task)
            support_features = self.network(support_x) 
            if self.finetune_mode == 'last_layer':
                 support_features = support_features.detach()
            
            support_logits = temp_classifier_head(support_features)
            # Use self.loss_fn (potentially from Default_task)
            support_loss = self.loss_fn(support_logits, support_y) 
            support_loss.backward()
            optimizer_finetune.step()

        # 4. Evaluation on query set
        self.network.eval() # Ensure backbone is in eval mode for consistent feature extraction
        temp_classifier_head.eval()
        with torch.no_grad():
            query_features = self.network(query_x)
            query_logits = temp_classifier_head(query_features)
            query_loss = self.loss_fn(query_logits, query_y)
            query_acc = (query_logits.argmax(dim=1) == query_y).float().mean()
            
        # Restore backbone's original training state if it was changed
        self.network.train(original_backbone_training_state)
            
        return query_loss, query_acc


    def _shared_step(self, batch, stage):
        support_x, support_y = batch['support_x'], batch['support_y']
        query_x, query_y = batch['query_x'], batch['query_y']
        n_way = batch['n_way']

        original_backbone_state_dict = None
        # For 'whole_model' finetuning, save and restore backbone weights if not in meta-training phase
        # This prevents tasks from polluting each other's backbone state during val/test.
        # During training, we WANT the backbone to be updated by the meta-optimizer based on query loss.
        if self.finetune_mode == 'whole_model' and stage != 'train':
            original_backbone_state_dict = {name: param.clone() for name, param in self.network.state_dict().items()}

        loss, acc = self._finetune_and_evaluate(support_x, support_y, query_x, query_y, n_way)

        if original_backbone_state_dict:
            self.network.load_state_dict(original_backbone_state_dict)

        self.log(f'{stage}_loss', loss, prog_bar=True, sync_dist=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, sync_dist=True)
        
        if stage == 'test':
            return {'test_loss': loss, 'test_acc': acc}
        
        # This loss (query loss after adaptation) is what a meta-optimizer (like MAML) would use
        # to update the initial parameters of self.network.
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, 'val')

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')

    def configure_optimizers(self):
        # If finetune_mode is 'whole_model', the self.network parameters are part of the
        # optimization process driven by the query loss from _shared_step.
        # The optimizer configured here by Default_task (e.g., Adam on self.network.parameters())
        # acts as the meta-optimizer.
        # If finetune_mode is 'last_layer', the backbone (self.network) is frozen.
        # In this case, self.network.parameters() should not be passed to an optimizer,
        # or their gradients should be zero.
        
        if self.finetune_mode == 'last_layer':
            # No parameters of this LightningModule (self.network) are being meta-trained.
            # The finetuning happens internally with a temporary optimizer.
            # So, no global optimizer is needed for the LightningModule itself.
            return None 
        else: # 'whole_model'
            # Inherit optimizer from Default_task, which will optimize self.network.parameters()
            # based on the loss returned by training_step.
            return super().configure_optimizers()
