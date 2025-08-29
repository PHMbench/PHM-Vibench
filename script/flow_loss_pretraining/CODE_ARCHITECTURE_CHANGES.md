# Code Architecture Modifications for Flow Loss Pretraining

## Overview
This document details the specific code changes needed to integrate flow loss pretraining into the PHM-Vibench framework.

## 1. New Module Structure

### 1.1 Flow Network Module
```
src/model_factory/ISFM/flow_net/
├── __init__.py
├── F_01_RectifiedFlow.py
└── base_flow.py
```

### 1.2 Enhanced Encoder Module
```
src/model_factory/ISFM/encoder/
├── __init__.py
├── E_03_ConditionalEncoder.py
└── base_encoder.py
```

### 1.3 New ISFM Model
```
src/model_factory/ISFM/
├── M_04_ISFM_Flow.py
└── (existing files...)
```

## 2. Detailed Implementation Specifications

### 2.1 Flow Network (`F_01_RectifiedFlow.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class F_01_RectifiedFlow(nn.Module):
    """
    Rectified flow network for continuous flow matching.
    
    Based on the implementation from CFL.ipynb notebook.
    Predicts velocity field for flow matching between noise and data.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.latent_dim = configs.latent_dim
        self.condition_dim = configs.condition_dim  # From encoder
        self.hidden_dim = getattr(configs, 'flow_hidden_dim', 256)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 4)
        )
        
        # Main network
        self.network = nn.Sequential(
            nn.Linear(self.latent_dim + self.condition_dim + self.hidden_dim // 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.latent_dim)
        )
        
    def forward(self, z_t, t, h):
        """
        Predict velocity at interpolated state z_t.
        
        Args:
            z_t: Interpolated state (B, latent_dim)
            t: Time parameter (B, 1)
            h: Condition/context (B, condition_dim)
            
        Returns:
            v_pred: Predicted velocity (B, latent_dim)
        """
        # Embed time
        t_embed = self.time_embed(t)
        
        # Concatenate inputs
        x = torch.cat([z_t, h, t_embed], dim=1)
        
        # Predict velocity
        v_pred = self.network(x)
        
        return v_pred
```

### 2.2 Conditional Encoder (`E_03_ConditionalEncoder.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class E_03_ConditionalEncoder(nn.Module):
    """
    Conditional encoder with domain and system embeddings.
    
    Encodes input signals with hierarchical conditioning based on
    domain and system metadata.
    """
    
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.input_dim
        self.latent_dim = configs.latent_dim
        self.num_domains = getattr(configs, 'num_domains', 2)
        self.num_systems = getattr(configs, 'num_systems', 2)
        self.cond_embed_dim = getattr(configs, 'cond_embed_dim', 16)
        
        # Domain and system embeddings
        self.domain_embed = nn.Embedding(self.num_domains, self.cond_embed_dim)
        self.system_embed = nn.Embedding(self.num_systems, self.cond_embed_dim)
        
        # Main encoding network
        total_input_dim = self.input_dim + 2 * self.cond_embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(total_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.latent_dim)
        )
        
    def forward(self, x, domain_id, system_id):
        """
        Encode input with conditional information.
        
        Args:
            x: Input signal (B, input_dim)
            domain_id: Domain indices (B,)
            system_id: System indices (B,)
            
        Returns:
            h: Encoded representation (B, latent_dim)
        """
        # Get embeddings
        domain_emb = self.domain_embed(domain_id)  # (B, cond_embed_dim)
        system_emb = self.system_embed(system_id)  # (B, cond_embed_dim)
        
        # Concatenate input with embeddings
        x_cond = torch.cat([x, domain_emb, system_emb], dim=1)
        
        # Encode
        h = self.encoder(x_cond)
        
        return h
```

### 2.3 Flow-Enhanced ISFM Model (`M_04_ISFM_Flow.py`)

```python
import torch
import torch.nn as nn
from ..flow_net import F_01_RectifiedFlow
from ..encoder import E_03_ConditionalEncoder

class Model(nn.Module):
    """
    ISFM model with flow matching capability.
    
    Integrates conditional encoding, reconstruction, and flow matching
    for hierarchical pretraining with domain/system awareness.
    """
    
    def __init__(self, args_m, metadata):
        super().__init__()
        self.args_m = args_m
        self.metadata = metadata
        
        # Components
        self.encoder = E_03_ConditionalEncoder(args_m)
        
        # Simple decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(args_m.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128), 
            nn.ReLU(),
            nn.Linear(128, args_m.input_dim)
        )
        
        # Flow network
        self.flow_net = F_01_RectifiedFlow(args_m)
        
        # Optional classifier
        if getattr(args_m, 'use_classifier', False):
            self.classifier = nn.Linear(args_m.latent_dim, args_m.num_classes)
        else:
            self.classifier = None
    
    def forward(self, x, domain_id, system_id, t=None, return_components=False):
        """
        Forward pass with optional flow prediction.
        
        Args:
            x: Input signal (B, input_dim)
            domain_id: Domain indices (B,)
            system_id: System indices (B,)
            t: Time for flow matching (B, 1) - optional
            return_components: Whether to return intermediate representations
            
        Returns:
            If return_components=False:
                x_recon: Reconstructed signal (B, input_dim)
            If return_components=True:
                (x_recon, h, v_pred, y_pred): All components
        """
        # Encode with conditioning
        h = self.encoder(x, domain_id, system_id)
        
        # Reconstruct
        x_recon = self.decoder(h)
        
        # Flow prediction (if time provided)
        v_pred = None
        if t is not None:
            # Sample noise for interpolation
            z0 = torch.randn_like(h)
            # Interpolate
            z_t = (1 - t) * z0 + t * h
            # Predict velocity
            v_pred = self.flow_net(z_t, t, h)
        
        # Classification (if enabled)
        y_pred = None
        if self.classifier is not None:
            y_pred = self.classifier(h)
        
        if return_components:
            return x_recon, h, v_pred, y_pred
        else:
            return x_recon
    
    def sample_flow(self, h, num_steps=100):
        """
        Sample from the flow model given condition h.
        
        Args:
            h: Condition/context (B, latent_dim)
            num_steps: Number of integration steps
            
        Returns:
            x_samples: Generated samples (B, input_dim)
        """
        device = h.device
        batch_size = h.shape[0]
        
        # Start from noise
        z = torch.randn_like(h)
        
        # Integrate through time
        dt = 1.0 / num_steps
        for i in range(num_steps):
            t = torch.ones(batch_size, 1, device=device) * i * dt
            v = self.flow_net(z, t, h)
            z = z + v * dt
        
        # Decode
        x_samples = self.decoder(z)
        
        return x_samples
```

### 2.4 Enhanced Pretrain Loss (`flow_pretrain_loss.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Tuple, Any

@dataclass
class FlowPretrainLossCfg:
    """Configuration for flow-based pretraining loss."""
    
    # Basic loss weights
    lambda_recon: float = 1.0
    lambda_flow: float = 1.0
    lambda_contrastive: float = 0.1
    
    # Hierarchical loss weights
    lambda_hier_domain: float = 1.0
    lambda_hier_system: float = 1.0
    lambda_hier_margin: float = 1.0
    margin: float = 0.1
    
    # Regularization
    lambda_reg: float = 0.01
    target_radius: float = 3.0
    
    # Classification (optional)
    lambda_class: float = 1.0
    use_classifier: bool = False

class FlowPretrainLoss(nn.Module):
    """
    Multi-objective loss for flow-based pretraining.
    
    Combines reconstruction, flow matching, hierarchical organization,
    and contrastive learning objectives.
    """
    
    def __init__(self, cfg: FlowPretrainLossCfg):
        super().__init__()
        self.cfg = cfg
    
    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total pretraining loss.
        
        Args:
            model: Flow-enabled ISFM model
            batch: Batch containing data and metadata
            
        Returns:
            (total_loss, loss_stats): Loss tensor and individual components
        """
        device = next(model.parameters()).device
        
        # Extract batch data
        x_batch = torch.stack([torch.as_tensor(d, dtype=torch.float32) for d in batch['data']]).to(device)
        metadata = batch['metadata']
        
        # Extract domain/system IDs
        domain_ids = torch.tensor([m['domain'] for m in metadata], dtype=torch.long, device=device)
        system_ids = torch.tensor([m['dataset'] for m in metadata], dtype=torch.long, device=device)
        
        batch_size = x_batch.shape[0]
        
        # Sample time for flow matching
        t = torch.rand(batch_size, 1, device=device)
        
        # Forward pass
        x_recon, h, v_pred, y_pred = model(x_batch, domain_ids, system_ids, t=t, return_components=True)
        
        # 1. Reconstruction loss
        loss_recon = F.mse_loss(x_recon, x_batch)
        
        # 2. Flow matching loss
        z0 = torch.randn_like(h)
        v_true = h.detach() - z0  # Target velocity
        loss_flow = F.mse_loss(v_pred, v_true)
        
        # 3. Contrastive flow loss
        negative_idx = torch.randperm(batch_size, device=device)
        v_negative = v_true[negative_idx]
        loss_contrastive = -F.mse_loss(v_pred, v_negative)
        
        # 4. Hierarchical losses
        loss_hier = self._compute_hierarchical_losses(h, domain_ids, system_ids)
        
        # 5. Regularization
        loss_reg = self._compute_regularization(h)
        
        # 6. Classification (optional)
        loss_class = torch.tensor(0.0, device=device)
        if self.cfg.use_classifier and y_pred is not None:
            # Get labels from metadata
            labels = torch.tensor([m.get('label', 0) for m in metadata], dtype=torch.long, device=device)
            loss_class = F.cross_entropy(y_pred, labels)
        
        # Total loss
        cfg = self.cfg
        total_loss = (
            cfg.lambda_recon * loss_recon +
            cfg.lambda_flow * loss_flow +
            cfg.lambda_contrastive * loss_contrastive +
            cfg.lambda_hier_domain * loss_hier['domain'] +
            cfg.lambda_hier_system * loss_hier['system'] +
            cfg.lambda_hier_margin * loss_hier['margin'] +
            cfg.lambda_reg * loss_reg +
            cfg.lambda_class * loss_class
        )
        
        # Loss statistics
        stats = {
            'loss_recon': loss_recon.item(),
            'loss_flow': loss_flow.item(),
            'loss_contrastive': loss_contrastive.item(),
            'loss_hier_domain': loss_hier['domain'].item(),
            'loss_hier_system': loss_hier['system'].item(),
            'loss_hier_margin': loss_hier['margin'].item(),
            'loss_reg': loss_reg.item(),
            'loss_class': loss_class.item(),
        }
        
        return total_loss, stats
    
    def _compute_hierarchical_losses(self, h, domain_ids, system_ids):
        """Compute hierarchical organization losses."""
        device = h.device
        
        # Domain-level separation
        domain_centers = []
        unique_domains = domain_ids.unique()
        for d in unique_domains:
            mask = domain_ids == d
            if mask.sum() > 0:
                domain_centers.append(h[mask].mean(0))
        
        loss_domain = torch.tensor(0.0, device=device)
        if len(domain_centers) > 1:
            # Encourage domain separation
            domain_dists = []
            for i, center_i in enumerate(domain_centers):
                for j, center_j in enumerate(domain_centers[i+1:], i+1):
                    dist = F.mse_loss(center_i, center_j)
                    domain_dists.append(dist)
            loss_domain = -torch.stack(domain_dists).mean()  # Negative to encourage separation
        
        # System-level cohesion
        system_centers = []
        unique_systems = system_ids.unique()
        for s in unique_systems:
            mask = system_ids == s
            if mask.sum() > 1:  # Need multiple samples for cohesion
                system_h = h[mask]
                center = system_h.mean(0)
                cohesion = F.mse_loss(system_h, center.unsqueeze(0).expand_as(system_h))
                system_centers.append(cohesion)
        
        loss_system = torch.stack(system_centers).mean() if system_centers else torch.tensor(0.0, device=device)
        
        # Hierarchical margin: domains should be more separated than systems
        avg_domain_dist = -loss_domain if len(domain_centers) > 1 else torch.tensor(0.0, device=device)
        avg_system_cohesion = loss_system
        loss_margin = F.relu(avg_system_cohesion - avg_domain_dist + self.cfg.margin)
        
        return {
            'domain': loss_domain,
            'system': loss_system, 
            'margin': loss_margin
        }
    
    def _compute_regularization(self, h):
        """Compute regularization loss."""
        # Radius constraint
        radius_loss = ((torch.linalg.norm(h, dim=1) - self.cfg.target_radius) ** 2).mean()
        
        # Centering penalty
        center_loss = h.mean(dim=0).pow(2).sum()
        
        return radius_loss + center_loss
```

## 3. Integration Points

### 3.1 Model Factory Registration

Update `src/model_factory/ISFM/__init__.py`:
```python
# Add new components to dictionaries
Embedding_dict = {
    # ... existing entries
    'E_03_ConditionalEncoder': E_03_ConditionalEncoder,
}

FlowNet_dict = {
    'F_01_RectifiedFlow': F_01_RectifiedFlow,
}

# Update Model selection to handle flow models
def get_model(model_name):
    if model_name == "M_04_ISFM_Flow":
        from .M_04_ISFM_Flow import Model
        return Model
    # ... existing model selections
```

### 3.2 Task Factory Registration

Update `src/task_factory/task_factory.py`:
```python
from .task.pretrain.flow_pretrain_task import FlowPretrainTask

task_dict = {
    # ... existing tasks
    ("pretrain", "flow_pretrain"): FlowPretrainTask,
}
```

### 3.3 Pipeline Integration

Modify `src/Pipeline_03_multitask_pretrain_finetune.py`:
```python
def run_flow_pretraining_stage(self):
    """Run flow-based pretraining with hierarchical organization."""
    print("Starting Flow-Based Pretraining Stage")
    
    # Configuration for flow pretraining
    flow_config = {
        'model': {
            'name': 'M_04_ISFM_Flow',
            'type': 'ISFM',
            'encoder': 'E_03_ConditionalEncoder',
            'flow_net': 'F_01_RectifiedFlow',
        },
        'task': {
            'name': 'flow_pretrain',
            'type': 'pretrain',
            'loss_config': FlowPretrainLossCfg(),
        }
    }
    
    # Run training
    checkpoint_path = self._run_single_pretraining(flow_config, "flow_pretrain")
    
    return checkpoint_path
```

## 4. Testing Strategy

### 4.1 Unit Tests
```python
# test_flow_network.py
def test_flow_network_forward():
    flow_net = F_01_RectifiedFlow(configs)
    z_t = torch.randn(32, 128)
    t = torch.rand(32, 1) 
    h = torch.randn(32, 128)
    v_pred = flow_net(z_t, t, h)
    assert v_pred.shape == (32, 128)

# test_hierarchical_loss.py  
def test_hierarchical_loss_computation():
    loss_fn = FlowPretrainLoss(FlowPretrainLossCfg())
    # Test with mock batch
    total_loss, stats = loss_fn(model, batch)
    assert 'loss_hier_margin' in stats
```

### 4.2 Integration Tests
```python
# test_flow_pretraining_pipeline.py
def test_full_pretraining_pipeline():
    pipeline = MultiTaskPretrainFinetunePipeline(config_path)
    checkpoint_path = pipeline.run_flow_pretraining_stage()
    assert os.path.exists(checkpoint_path)
```

## 5. Configuration Management

### 5.1 Default Parameters
```python
# Default configuration values
DEFAULT_FLOW_CONFIG = {
    'latent_dim': 128,
    'condition_dim': 128, 
    'flow_hidden_dim': 256,
    'num_domains': 2,
    'num_systems': 2,
    'target_radius': 3.0,
    'margin': 0.1,
}
```

### 5.2 Validation
```python
def validate_flow_config(config):
    required_fields = ['latent_dim', 'num_domains', 'num_systems']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
    
    if config['num_domains'] < 1 or config['num_systems'] < 1:
        raise ValueError("Domain and system counts must be positive")
```

This architecture integrates the flow loss concepts from CFL.ipynb while maintaining compatibility with PHM-Vibench's existing factory pattern and modular design.