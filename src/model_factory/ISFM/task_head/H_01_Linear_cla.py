import torch
import torch.nn as nn
from typing import Union, List

class H_01_Linear_cla(nn.Module):
    def __init__(self, args):
        super(H_01_Linear_cla, self).__init__()
        self.mutiple_fc = nn.ModuleDict()
        num_classes = args.num_classes 
        for data_name, n_class in num_classes.items():
            self.mutiple_fc[str(data_name)] = nn.Linear(args.output_dim,
                                                   n_class)

    def forward(self, x, system_id = False, return_feature = False, **kwargs):
        """Forward pass supporting both single and batch system_ids.
        
        Args:
            x: Input tensor (B, T, d_model)
            system_id: Single system_id or list/tensor of system_ids for batch
            return_feature: Whether to return features before classification
            
        Returns:
            Logits tensor (B, num_classes) where num_classes depends on system_id(s)
        """
        # x: (B, T, d_model) 先对时间维度做平均池化
        x = x.mean(dim=1)  # (B, d_model)
        
        if return_feature:
            return x
            
        # Check if system_id is a vector (list or tensor)
        if isinstance(system_id, (list, tuple)) or (isinstance(system_id, torch.Tensor) and system_id.numel() > 1):
            return self._batch_forward(x, system_id)
        else:
            return self._single_forward(x, system_id)
    
    def _single_forward(self, x, system_id):
        """Original single system_id logic for backward compatibility.
        Note: x should already be mean-pooled by the main forward method.
        """
        logits = self.mutiple_fc[str(system_id)](x)
        return logits
    
    def _batch_forward(self, x, system_ids):
        """New batch processing logic for mixed system_ids.
        
        Args:
            x: Features (B, d_model)
            system_ids: List or tensor of system_ids, one per sample
            
        Returns:
            Logits tensor (B, max_num_classes) with padding for different class counts
        """
        batch_size = x.shape[0]
        
        # Convert system_ids to list for consistent handling
        if isinstance(system_ids, torch.Tensor):
            system_ids = system_ids.tolist() if system_ids.dim() == 1 else [system_ids.item()]
        
        # Ensure we have one system_id per sample
        if len(system_ids) != batch_size:
            if len(system_ids) == 1:
                # Single system_id for entire batch - use original logic
                return self._single_forward(x, system_ids[0])
            else:
                raise ValueError(f"system_ids length ({len(system_ids)}) must match batch_size ({batch_size})")
        
        # Determine maximum number of classes across all systems
        max_classes = 0
        for sys_id in system_ids:
            if str(sys_id) in self.mutiple_fc:
                max_classes = max(max_classes, self.mutiple_fc[str(sys_id)].out_features)
        
        if max_classes == 0:
            raise ValueError(f"No valid FC layers found for system_ids: {system_ids}")
            
        # OPTIMIZED: Group samples by system_id and process in batches
        # This reduces O(N) forward passes to O(K) where K = unique system_ids
        
        # Initialize result tensor
        batch_logits = torch.zeros(batch_size, max_classes, device=x.device, dtype=x.dtype)
        
        # Group indices by system_id for efficient batch processing
        groups = {}
        unknown_indices = []
        
        for i, sys_id in enumerate(system_ids):
            sys_id_str = str(sys_id)
            if sys_id_str in self.mutiple_fc:
                if sys_id_str not in groups:
                    groups[sys_id_str] = []
                groups[sys_id_str].append(i)
            else:
                unknown_indices.append(i)
                print(f"Warning: Unknown system_id {sys_id}, using zero logits")
        
        # Process each group in a single forward pass (MAJOR EFFICIENCY GAIN)
        for sys_id_str, indices in groups.items():
            # Extract all features for this system_id group
            indices_tensor = torch.tensor(indices, device=x.device)
            group_x = x[indices_tensor]  # (group_size, d_model)
            
            # Single forward pass for entire group
            fc_layer = self.mutiple_fc[sys_id_str]
            group_logits = fc_layer(group_x)  # (group_size, num_classes_for_sys_id)
            
            # Place group results in correct positions with padding
            current_classes = group_logits.shape[1]
            if current_classes <= max_classes:
                batch_logits[indices_tensor, :current_classes] = group_logits
                # Zero padding is already handled by initialization
            else:
                # Truncate if necessary (shouldn't happen with correct max_classes calculation)
                batch_logits[indices_tensor, :] = group_logits[:, :max_classes]
        
        # Unknown system_ids already have zero logits from initialization
        return batch_logits
