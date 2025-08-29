# Flow Loss Pretraining Implementation Plan for PHM-Vibench

## Overview
This document outlines the comprehensive plan for integrating flow loss-based pretraining into the PHM-Vibench framework, based on the Continuous Flow Matching (CFM) and hierarchical contrastive learning approaches from the CFL.ipynb notebook.

## 1. Core Concepts from CFL.ipynb

### 1.1 Flow Matching Framework
- **Rectified Flow**: Direct linear interpolation between noise and data distributions
- **Velocity Prediction**: Network predicts velocity field v(z_t, t) for flow matching
- **Time-dependent Interpolation**: z_t = (1-t) * z_0 + t * h, where z_0 is noise, h is encoded feature

### 1.2 Hierarchical Contrastive Learning
- **Multi-level Hierarchy**: Domain > System > Instance
- **Contrastive Objectives**: InfoNCE, Triplet losses for representation learning
- **Margin Constraints**: Enforce hierarchical separation in latent space

### 1.3 Key Loss Components
```python
# From notebook analysis:
- flow_loss: MSE between predicted and target velocities
- contrastive_flow_loss: Negative sample repulsion
- hierarchical_margin_loss: Domain-system hierarchy enforcement  
- regularization_loss: Latent radius constraint + centering
- reconstruction_loss: Decoder reconstruction error
```

## 2. Architecture Modifications

### 2.1 New Model Components

#### A. Flow Network Module (`src/model_factory/ISFM/flow_net/`)
```python
# F_01_RectifiedFlow.py
class F_01_RectifiedFlow(nn.Module):
    """
    Rectified flow network for velocity prediction
    Input: z_t (interpolated state), t (time), h (condition)
    Output: v_pred (velocity vector)
    """
    def __init__(self, latent_dim, condition_dim, hidden_dim=256):
        - Time embedding network
        - Conditional MLP with SiLU activations
        - Velocity prediction head
```

#### B. Conditional Encoder (`src/model_factory/ISFM/encoder/`)
```python
# E_03_ConditionalEncoder.py
class E_03_ConditionalEncoder(nn.Module):
    """
    Encoder with domain/system conditioning
    Input: x (signal), domain_id, system_id
    Output: h (latent representation)
    """
    def __init__(self, input_dim, latent_dim, num_domains, num_systems):
        - Domain embedding layer
        - System embedding layer
        - Conditional encoding network
```

### 2.2 Extended ISFM Model

#### New Foundation Model (`src/model_factory/ISFM/M_04_ISFM_Flow.py`)
```python
class Model(nn.Module):
    """
    ISFM with flow matching pretraining capability
    Combines encoder, decoder, flow network, and optional classifier
    """
    def __init__(self, args_m, metadata):
        - Conditional encoder (E_03)
        - Decoder network
        - Flow network (F_01)
        - Optional classifier head
        
    def forward(self, x, domain_id, system_id, t=None):
        - Encode: h = encoder(x, domain, system)
        - Reconstruct: x_recon = decoder(h)
        - Flow: v_pred = flow_net(z_t, t, h) if t provided
        - Classify: y_pred = classifier(h) if enabled
        return x_recon, h, v_pred, y_pred
```

### 2.3 New Task Head (`src/model_factory/ISFM/task_head/H_10_flow_pretrain.py`)
```python
class H_10_flow_pretrain(nn.Module):
    """
    Combined head for flow-based pretraining
    Outputs: reconstruction, flow velocity, optional classification
    """
```

## 3. Loss Function Extensions

### 3.1 Enhanced Pretrain Loss (`src/task_factory/Components/flow_pretrain_loss.py`)

```python
class FlowPretrainLoss(nn.Module):
    """
    Multi-objective loss for flow-based pretraining
    """
    def __init__(self, cfg: FlowPretrainLossCfg):
        self.cfg = cfg
        
    def forward(self, model, batch):
        # 1. Reconstruction loss
        loss_recon = F.mse_loss(x_recon, x)
        
        # 2. Flow matching loss
        z0 = torch.randn_like(h)
        t = torch.rand(batch_size, 1)
        z_t = (1-t) * z0 + t * h
        v_true = h - z0
        loss_flow = F.mse_loss(v_pred, v_true)
        
        # 3. Contrastive flow loss
        negative_target = v_true[torch.randperm(batch_size)]
        loss_contrastive = -F.mse_loss(v_pred, negative_target)
        
        # 4. Hierarchical margin losses
        loss_hier_domain = compute_domain_separation(h, domain_ids)
        loss_hier_system = compute_system_cohesion(h, system_ids)
        loss_hier_margin = F.relu(dist_domain - dist_system + margin)
        
        # 5. Regularization
        loss_reg = radius_constraint(h) + centering_penalty(h)
        
        # Total loss
        total_loss = (lambda_recon * loss_recon +
                     lambda_flow * loss_flow +
                     lambda_contrastive * loss_contrastive +
                     lambda_hier * (loss_hier_domain + loss_hier_system + loss_hier_margin) +
                     lambda_reg * loss_reg)
        
        return total_loss, loss_dict
```

### 3.2 Loss Configuration
```python
@dataclass
class FlowPretrainLossCfg:
    # Basic weights
    lambda_recon: float = 1.0
    lambda_flow: float = 1.0
    lambda_contrastive: float = 0.1
    
    # Hierarchical weights
    lambda_hier_domain: float = 1.0
    lambda_hier_system: float = 1.0
    lambda_hier_margin: float = 1.0
    margin: float = 0.1
    
    # Regularization
    lambda_reg: float = 0.01
    target_radius: float = 3.0
    
    # Optional classification
    lambda_class: float = 1.0
    use_classifier: bool = False
```

## 4. Training Pipeline Integration

### 4.1 Modified Pretraining Pipeline

Update `src/Pipeline_03_multitask_pretrain_finetune.py`:

```python
def run_flow_pretraining(self, config):
    """
    Run flow-based pretraining stage
    """
    # 1. Initialize flow-enabled model
    model = build_model(args_model)  # M_04_ISFM_Flow
    
    # 2. Setup flow pretrain task
    task = FlowPretrainTask(model, args_task)
    
    # 3. Configure flow loss
    loss_fn = FlowPretrainLoss(args_task.loss_config)
    
    # 4. Training loop with flow matching
    for epoch in range(epochs):
        for batch in dataloader:
            # Extract domain/system metadata
            x, domain_ids, system_ids = process_batch(batch)
            
            # Forward pass with flow
            outputs = model(x, domain_ids, system_ids)
            
            # Compute multi-objective loss
            loss, stats = loss_fn(model, batch)
            
            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### 4.2 New Task Factory Entry

Create `src/task_factory/task/pretrain/flow_pretrain_task.py`:

```python
class FlowPretrainTask(pl.LightningModule):
    """
    PyTorch Lightning module for flow-based pretraining
    """
    def __init__(self, network, args_data, args_model, args_task, args_trainer, args_environment, metadata):
        super().__init__()
        self.network = network
        self.loss_fn = FlowPretrainLoss(args_task.loss_config)
        
    def training_step(self, batch, batch_idx):
        # Process batch with metadata
        x, metadata = batch
        domain_ids = extract_domain_ids(metadata)
        system_ids = extract_system_ids(metadata)
        
        # Forward pass
        x_recon, h, v_pred, y_pred = self.network(x, domain_ids, system_ids)
        
        # Compute loss
        loss, stats = self.loss_fn(self.network, batch)
        
        # Log metrics
        for key, value in stats.items():
            self.log(f'train_{key}', value)
            
        return loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]
```

## 5. Configuration Templates

### 5.1 Flow Pretraining Config (`configs/demo/Pretraining/flow_pretrain.yaml`)

```yaml
# Environment configuration
environment:
  seed: 42
  device: "cuda"
  
# Data configuration  
data:
  name: "ID_dataset"
  metadata_file_list: ["metadata_CWRU_split.xlsx", "metadata_THU_split.xlsx"]
  batch_size: 256
  
# Model configuration
model:
  name: "M_04_ISFM_Flow"
  type: "ISFM"
  
  # Architecture components
  encoder: "E_03_ConditionalEncoder"
  decoder: "SimpleDecoder"
  flow_net: "F_01_RectifiedFlow"
  
  # Model dimensions
  input_dim: 1
  latent_dim: 128
  hidden_dim: 256
  
  # Domain/System configuration
  num_domains: 2
  num_systems: 2
  
# Task configuration  
task:
  name: "flow_pretrain"
  type: "pretrain"
  
  # Loss configuration
  loss_config:
    # Basic losses
    lambda_recon: 1.0
    lambda_flow: 1.0
    lambda_contrastive: 0.1
    
    # Hierarchical losses
    lambda_hier_domain: 1.0
    lambda_hier_system: 1.0
    lambda_hier_margin: 1.0
    margin: 0.1
    
    # Regularization
    lambda_reg: 0.01
    target_radius: 3.0
    
    # Optional classification
    use_classifier: false
    lambda_class: 1.0
    
  # Training parameters
  epochs: 500
  lr: 1e-3
  
# Trainer configuration
trainer:
  max_epochs: 500
  accelerator: "gpu"
  devices: 1
  log_every_n_steps: 10
```

## 6. Implementation Checklist

### 6.1 New Files to Create
- [ ] `src/model_factory/ISFM/flow_net/F_01_RectifiedFlow.py`
- [ ] `src/model_factory/ISFM/flow_net/__init__.py`
- [ ] `src/model_factory/ISFM/encoder/E_03_ConditionalEncoder.py`
- [ ] `src/model_factory/ISFM/M_04_ISFM_Flow.py`
- [ ] `src/model_factory/ISFM/task_head/H_10_flow_pretrain.py`
- [ ] `src/task_factory/Components/flow_pretrain_loss.py`
- [ ] `src/task_factory/task/pretrain/flow_pretrain_task.py`
- [ ] `configs/demo/Pretraining/flow_pretrain.yaml`

### 6.2 Files to Modify
- [ ] `src/model_factory/ISFM/__init__.py` - Register new components
- [ ] `src/task_factory/task_factory.py` - Register flow pretrain task
- [ ] `src/Pipeline_03_multitask_pretrain_finetune.py` - Add flow pretraining option
- [ ] `src/data_factory/ID_dataset.py` - Ensure domain/system metadata extraction

### 6.3 Testing Requirements
- [ ] Unit test for flow network forward pass
- [ ] Unit test for hierarchical loss computation
- [ ] Integration test for full pretraining pipeline
- [ ] Validation of latent space organization
- [ ] Performance comparison with existing pretraining

## 7. Key Implementation Details

### 7.1 Flow Matching Algorithm
```python
# Core flow matching loop
def flow_matching_step(h, model):
    # 1. Sample noise
    z0 = torch.randn_like(h)
    
    # 2. Sample time
    t = torch.rand(h.shape[0], 1).to(h.device)
    
    # 3. Interpolate
    z_t = (1 - t) * z0 + t * h
    
    # 4. Predict velocity
    v_pred = model.flow_net(z_t, t, h)
    
    # 5. Target velocity
    v_true = h - z0
    
    # 6. Loss
    loss = F.mse_loss(v_pred, v_true)
    
    return loss
```

### 7.2 Hierarchical Organization
```python
def compute_hierarchical_losses(h, domain_ids, system_ids):
    # Domain-level separation
    domain_centers = []
    for d in domain_ids.unique():
        mask = domain_ids == d
        domain_centers.append(h[mask].mean(0))
    
    # System-level cohesion within domains
    system_centers = []
    for s in system_ids.unique():
        mask = system_ids == s
        system_centers.append(h[mask].mean(0))
    
    # Enforce hierarchy: domains more separated than systems
    domain_dist = compute_pairwise_distances(domain_centers)
    system_dist = compute_pairwise_distances(system_centers)
    
    margin_loss = F.relu(system_dist - domain_dist + margin)
    
    return margin_loss
```

### 7.3 Sampling Strategy
```python
def sample_from_flow(model, num_samples, device):
    # 1. Start from noise
    z = torch.randn(num_samples, latent_dim).to(device)
    
    # 2. Integrate through time
    dt = 1.0 / num_steps
    for i in range(num_steps):
        t = torch.ones(num_samples, 1).to(device) * i * dt
        v = model.flow_net(z, t, condition)
        z = z + v * dt
    
    # 3. Decode to data space
    x = model.decoder(z)
    
    return x
```

## 8. Expected Benefits

1. **Better Representation Learning**: Flow matching provides smooth interpolation between noise and data
2. **Hierarchical Organization**: Explicit domain/system structure in latent space
3. **Contrastive Learning**: Enhanced separation between different fault types
4. **Generative Capability**: Can sample new data points for augmentation
5. **Transfer Learning**: Hierarchical structure aids in cross-domain transfer

## 9. Evaluation Metrics

- **Reconstruction Error**: MSE between input and reconstructed signals
- **Flow Matching Loss**: Velocity prediction accuracy
- **Latent Space Metrics**:
  - Domain separation distance
  - System cohesion within domains
  - Instance diversity within systems
- **Downstream Task Performance**: Classification/prediction after fine-tuning

## 10. Next Steps

1. Implement core flow network module
2. Extend existing pretrain loss with flow components
3. Create configuration templates
4. Run ablation studies on loss components
5. Compare with existing pretraining baselines
6. Optimize hyperparameters for PHM domain

---

**Note**: This plan integrates advanced flow matching techniques while maintaining compatibility with PHM-Vibench's modular factory pattern architecture.