"""
Optimization Utilities for CWRU Few-Shot Learning Experiments
Enhanced models, augmentation strategies, and training utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class EnhancedContrastiveEncoder(nn.Module):
    """
    Enhanced contrastive encoder with residual connections and better architecture
    """
    def __init__(self, input_channels=2, hidden_dim=128, dropout=0.2):
        super(EnhancedContrastiveEncoder, self).__init__()

        # Enhanced architecture with residual connections
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        # Residual connection for skip paths
        self.residual_conv = nn.Conv1d(32, 128, kernel_size=1)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Enhanced projection head
        self.projection = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, C, L)

        # First conv block
        x1 = F.relu(self.bn1(self.conv1(x)))
        x1 = F.max_pool1d(x1, 2)

        # Second conv block
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x2 = F.max_pool1d(x2, 2)

        # Third conv block with residual connection
        x3 = F.relu(self.bn3(self.conv3(x2)))

        # Residual connection from x1 to x3
        residual = self.residual_conv(x1)
        residual = F.adaptive_avg_pool1d(residual, x3.size(-1))
        x3 = x3 + residual

        # Global pooling and projection
        features = self.pool(x3).squeeze(-1)
        features = self.dropout(features)
        embeddings = self.projection(features)

        return embeddings, features


class ProgressiveUnfreezing:
    """
    Progressive unfreezing strategy for fine-tuning pretrained models
    """
    def __init__(self, model: nn.Module, total_epochs: int):
        self.model = model
        self.total_epochs = total_epochs
        self.encoder_layers = []

        # Identify encoder layers (assuming encoder attribute exists)
        if hasattr(model, 'encoder'):
            for name, module in model.encoder.named_children():
                if isinstance(module, (nn.Conv1d, nn.Linear)):
                    self.encoder_layers.append(module)

    def update_freezing(self, current_epoch: int):
        """Update which layers are frozen based on current epoch"""
        progress = current_epoch / self.total_epochs

        if progress < 0.3:
            # First 30%: Freeze all encoder layers
            for layer in self.encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        elif progress < 0.6:
            # Next 30%: Unfreeze last 2 layers
            for layer in self.encoder_layers[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            # Final 40%: Unfreeze all layers
            for layer in self.encoder_layers:
                for param in layer.parameters():
                    param.requires_grad = True


class IndustrialAugmentation:
    """
    Domain-specific augmentation strategies for industrial vibration signals
    """

    @staticmethod
    def gaussian_noise(x: torch.Tensor, std: float = 0.1) -> torch.Tensor:
        """Add Gaussian noise"""
        return x + torch.randn_like(x) * std

    @staticmethod
    def amplitude_scaling(x: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """Random amplitude scaling"""
        scale = torch.uniform(scale_range[0], scale_range[1], (x.shape[0], 1, 1)).to(x.device)
        return x * scale

    @staticmethod
    def time_shifting(x: torch.Tensor, max_shift: int = 50) -> torch.Tensor:
        """Circular time shifting"""
        batch_size = x.shape[0]
        shifts = torch.randint(-max_shift, max_shift + 1, (batch_size,))

        shifted = torch.zeros_like(x)
        for i, shift in enumerate(shifts):
            shifted[i] = torch.roll(x[i], shift.item(), dims=0)
        return shifted

    @staticmethod
    def frequency_masking(x: torch.Tensor, mask_ratio: float = 0.1) -> torch.Tensor:
        """Frequency domain masking"""
        # Convert to frequency domain
        x_fft = torch.fft.fft(x, dim=1)

        # Create random mask
        freq_len = x_fft.shape[1]
        mask_len = int(freq_len * mask_ratio)

        for i in range(x.shape[0]):
            start_idx = torch.randint(0, freq_len - mask_len, (1,)).item()
            x_fft[i, start_idx:start_idx + mask_len] = 0

        # Convert back to time domain
        return torch.fft.ifft(x_fft, dim=1).real

    @staticmethod
    def apply_random_augmentation(x: torch.Tensor,
                                  augment_prob: float = 0.8,
                                  methods: Optional[List[str]] = None) -> torch.Tensor:
        """Apply random combination of augmentations"""
        if methods is None:
            methods = ['gaussian_noise', 'amplitude_scaling', 'time_shifting']

        if torch.rand(1).item() < augment_prob:
            # Choose random augmentation method
            method = np.random.choice(methods)

            if method == 'gaussian_noise':
                return IndustrialAugmentation.gaussian_noise(x)
            elif method == 'amplitude_scaling':
                return IndustrialAugmentation.amplitude_scaling(x)
            elif method == 'time_shifting':
                return IndustrialAugmentation.time_shifting(x)
            elif method == 'frequency_masking':
                return IndustrialAugmentation.frequency_masking(x)

        return x


class MultiTaskPretraining:
    """
    Multi-task pretraining combining different self-supervised objectives
    """

    @staticmethod
    def masked_signal_modeling_loss(model: nn.Module, x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        """Masked signal modeling loss (like BERT for time series)"""
        batch_size, seq_len, channels = x.shape

        # Create random mask
        mask_len = int(seq_len * mask_ratio)
        masked_x = x.clone()
        targets = []

        for i in range(batch_size):
            start_idx = torch.randint(0, seq_len - mask_len, (1,)).item()
            targets.append(x[i, start_idx:start_idx + mask_len].clone())
            masked_x[i, start_idx:start_idx + mask_len] = 0  # Zero out masked region

        # Predict masked regions
        if hasattr(model, 'decode'):
            predictions = model.decode(masked_x)
        else:
            # Use encoder + decoder head for reconstruction
            embeddings, _ = model(masked_x)
            predictions = embeddings  # Simplified - needs proper decoder

        # Compute reconstruction loss for masked regions only
        loss = 0
        for i, target in enumerate(targets):
            pred_segment = predictions[i, start_idx:start_idx + mask_len]
            loss += F.mse_loss(pred_segment, target)

        return loss / batch_size

    @staticmethod
    def temporal_order_prediction_loss(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Predict whether two segments are in correct temporal order"""
        batch_size, seq_len, channels = x.shape

        # Split signal into two halves
        mid_point = seq_len // 2
        first_half = x[:, :mid_point]
        second_half = x[:, mid_point:]

        # Create positive (correct order) and negative (reversed order) pairs
        positive_pairs = torch.cat([first_half, second_half], dim=1)
        negative_pairs = torch.cat([second_half, first_half], dim=1)

        # Combine and create labels
        combined = torch.cat([positive_pairs, negative_pairs], dim=0)
        labels = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).long().to(x.device)

        # Binary classification on temporal order
        embeddings, _ = model(combined)
        order_classifier = nn.Linear(embeddings.shape[-1], 2).to(x.device)
        logits = order_classifier(embeddings)

        return F.cross_entropy(logits, labels)


class AdaptiveOptimizer:
    """
    Adaptive optimizer with different learning rates for different components
    """

    @staticmethod
    def create_grouped_optimizer(model: nn.Module,
                                 base_lr: float = 1e-3,
                                 pretrained_lr_ratio: float = 0.1,
                                 head_lr_ratio: float = 1.0,
                                 weight_decay: float = 1e-4) -> torch.optim.Optimizer:
        """Create optimizer with grouped parameters"""

        pretrained_params = []
        head_params = []

        # Identify pretrained vs new parameters
        for name, param in model.named_parameters():
            if 'encoder' in name or 'flow' in name:
                pretrained_params.append(param)
            else:
                head_params.append(param)

        param_groups = [
            {'params': pretrained_params, 'lr': base_lr * pretrained_lr_ratio},
            {'params': head_params, 'lr': base_lr * head_lr_ratio}
        ]

        return torch.optim.Adam(param_groups, weight_decay=weight_decay)


class EarlyStopping:
    """
    Early stopping utility with patience and best model saving
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.best_state = None

    def __call__(self, loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop
        Returns True if should stop, False otherwise
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_best_weights(self, model: nn.Module):
        """Restore best model weights"""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# Example usage functions
def create_optimized_training_setup(model: nn.Module,
                                   total_epochs: int) -> Tuple[torch.optim.Optimizer,
                                                              torch.optim.lr_scheduler._LRScheduler,
                                                              ProgressiveUnfreezing,
                                                              EarlyStopping]:
    """Create complete optimized training setup"""

    optimizer = AdaptiveOptimizer.create_grouped_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-5)
    unfreezing = ProgressiveUnfreezing(model, total_epochs)
    early_stopping = EarlyStopping(patience=10)

    return optimizer, scheduler, unfreezing, early_stopping


def enhanced_contrastive_pretraining(encoder: nn.Module,
                                   dataloader: torch.utils.data.DataLoader,
                                   epochs: int = 50,
                                   use_supervised: bool = False) -> nn.Module:
    """
    Enhanced contrastive pretraining with all optimizations
    """
    device = next(encoder.parameters()).device
    optimizer, scheduler, _, early_stopping = create_optimized_training_setup(encoder, epochs)

    encoder.train()
    print(f'Starting enhanced contrastive pretraining for {epochs} epochs...')

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            # Enhanced augmentation
            augmented = IndustrialAugmentation.apply_random_augmentation(
                batch_x, methods=['gaussian_noise', 'amplitude_scaling', 'time_shifting']
            )

            # Forward pass
            embeddings1, _ = encoder(batch_x)
            embeddings2, _ = encoder(augmented)
            embeddings = torch.cat([embeddings1, embeddings2])

            # Compute loss (supervised or unsupervised)
            if use_supervised:
                combined_labels = torch.cat([batch_y, batch_y])
                loss = supervised_contrastive_loss(embeddings, combined_labels)
            else:
                loss = contrastive_loss(embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        scheduler.step()
        avg_loss = total_loss / batch_count

        # Early stopping check
        if early_stopping(avg_loss, encoder):
            print(f'Early stopping at epoch {epoch}')
            break

        if epoch % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, LR={lr:.6f}')

    # Restore best weights
    early_stopping.restore_best_weights(encoder)
    print(f'Enhanced pretraining completed! Best loss: {early_stopping.best_loss:.4f}')

    return encoder


# Loss functions (need to be defined if not already available)
def contrastive_loss(embeddings, temperature=0.5):
    """Standard contrastive loss"""
    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=1)
    similarity = torch.mm(embeddings, embeddings.t()) / temperature
    batch_size = embeddings.shape[0] // 2

    labels = torch.cat([
        torch.arange(batch_size, batch_size * 2),
        torch.arange(batch_size)
    ]).to(device)

    mask = torch.eye(similarity.shape[0]).bool().to(device)
    similarity = similarity.masked_fill(mask, -float('inf'))

    return F.cross_entropy(similarity, labels)


def supervised_contrastive_loss(embeddings, labels, temperature=0.5):
    """Supervised contrastive loss using class labels"""
    device = embeddings.device
    embeddings = F.normalize(embeddings, dim=1)
    batch_size = embeddings.shape[0]

    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    similarity = torch.div(torch.matmul(embeddings, embeddings.T), temperature)

    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask

    exp_logits = torch.exp(similarity) * logits_mask
    log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))

    mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    loss = -mean_log_prob_pos.mean()

    return loss


# ===================================================================
# Flow Matching Utilities
# ===================================================================

class FlowMatchingPretrainer:
    """
    Flow matching pretraining wrapper that integrates FlowLoss with encoder models
    """

    def __init__(self, encoder: nn.Module,
                 target_channels: int,
                 z_channels: int = 128,
                 depth: int = 4,
                 width: int = 256,
                 num_sampling_steps: int = 20):
        self.encoder = encoder
        self.device = next(encoder.parameters()).device

        # Import FlowLoss from the main codebase
        try:
            from src.task_factory.Components.flow import FlowLoss
            self.flow_model = FlowLoss(
                target_channels=target_channels,
                z_channels=z_channels,
                depth=depth,
                width=width,
                num_sampling_steps=num_sampling_steps
            ).to(self.device)
        except ImportError:
            # Fallback to simplified implementation if import fails
            print("Warning: Could not import FlowLoss, using simplified implementation")
            self.flow_model = SimpleFlowModel(target_channels, z_channels).to(self.device)

    def pretrain(self, dataloader, epochs: int = 50, lr: float = 1e-3):
        """Flow matching pretraining"""

        # Combined optimizer for encoder + flow model
        params = list(self.encoder.parameters()) + list(self.flow_model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        early_stopping = EarlyStopping(patience=15)

        self.encoder.train()
        self.flow_model.train()

        print(f'Starting flow matching pretraining for {epochs} epochs...')

        for epoch in range(epochs):
            total_loss = 0
            batch_count = 0

            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)

                # Optional augmentation
                if torch.rand(1).item() < 0.3:  # 30% chance
                    batch_x = IndustrialAugmentation.apply_random_augmentation(batch_x)

                # Get conditional representation from encoder
                if hasattr(self.encoder, 'get_rep'):
                    condition = self.encoder.get_rep(batch_x)
                else:
                    # Use encoder output directly
                    embeddings, _ = self.encoder(batch_x)
                    condition = embeddings

                # Flatten signal for flow matching
                target = batch_x.view(batch_x.size(0), -1)

                # Flow matching loss
                loss = self.flow_model(target, condition)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            scheduler.step()
            avg_loss = total_loss / batch_count

            # Early stopping check
            if early_stopping(avg_loss, self.flow_model):
                print(f'Early stopping at epoch {epoch}')
                break

            if epoch % 10 == 0:
                lr_current = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}: Flow Loss={avg_loss:.4f}, LR={lr_current:.6f}')

        # Restore best weights
        early_stopping.restore_best_weights(self.flow_model)
        print(f'Flow pretraining completed! Best loss: {early_stopping.best_loss:.4f}')

        return self.encoder

    def generate_samples(self, conditions, num_samples: int = 5):
        """Generate samples using trained flow model"""
        self.flow_model.eval()
        with torch.no_grad():
            generated = self.flow_model.sample(conditions, num_samples)
        return generated


class SimpleFlowModel(nn.Module):
    """
    Simplified flow model for cases where main FlowLoss is not available
    """

    def __init__(self, target_channels: int, z_channels: int):
        super().__init__()
        self.target_channels = target_channels

        # Simple velocity prediction network
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64)
        )

        self.cond_embed = nn.Linear(z_channels, 64)

        self.velocity_net = nn.Sequential(
            nn.Linear(target_channels + 64 + 64, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, target_channels)
        )

    def forward(self, target, condition):
        batch_size = target.shape[0]
        device = target.device

        # Random time steps
        t = torch.rand(batch_size, 1, device=device)

        # Random noise
        noise = torch.randn_like(target)

        # Linear interpolation: x_t = (1-t) * noise + t * target
        noised_target = (1 - t) * noise + t * target

        # Time and condition embeddings
        t_embed = self.time_embed(t)
        c_embed = self.cond_embed(condition)

        # Concatenate inputs
        net_input = torch.cat([noised_target, t_embed, c_embed], dim=1)

        # Predict velocity: v = target - noise
        velocity_pred = self.velocity_net(net_input)
        velocity_true = target - noise

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_true)
        return loss

    def sample(self, condition, num_samples: int = 1):
        """Simple sampling procedure"""
        self.eval()
        device = condition.device
        batch_size = condition.shape[0]

        # Start from noise
        x = torch.randn(batch_size * num_samples, self.target_channels, device=device)
        condition_expanded = condition.repeat_interleave(num_samples, dim=0)

        # Simple denoising steps
        steps = 20
        dt = 1.0 / steps

        for i in range(steps):
            t = torch.full((x.shape[0], 1), i / steps, device=device)

            t_embed = self.time_embed(t)
            c_embed = self.cond_embed(condition_expanded)

            net_input = torch.cat([x, t_embed, c_embed], dim=1)
            velocity = self.velocity_net(net_input)

            x = x + velocity * dt

        return x.view(batch_size, num_samples, -1)


class FlowDataAugmentation:
    """
    Data augmentation using flow models for generating synthetic training samples
    """

    def __init__(self, flow_pretrainer: FlowMatchingPretrainer):
        self.flow_pretrainer = flow_pretrainer

    def augment_batch(self, batch_x: torch.Tensor, augment_ratio: float = 0.5):
        """Augment batch with flow-generated samples"""
        device = batch_x.device
        batch_size = batch_x.shape[0]
        num_synthetic = int(batch_size * augment_ratio)

        if num_synthetic == 0:
            return batch_x

        # Get conditions from real samples
        with torch.no_grad():
            if hasattr(self.flow_pretrainer.encoder, 'get_rep'):
                conditions = self.flow_pretrainer.encoder.get_rep(batch_x[:num_synthetic])
            else:
                embeddings, _ = self.flow_pretrainer.encoder(batch_x[:num_synthetic])
                conditions = embeddings

            # Generate synthetic samples
            synthetic_flat = self.flow_pretrainer.generate_samples(conditions, num_samples=1)
            synthetic_samples = synthetic_flat.view(num_synthetic, *batch_x.shape[1:])

            # Combine real and synthetic
            augmented_batch = torch.cat([batch_x, synthetic_samples], dim=0)

        return augmented_batch


class HybridPretraining:
    """
    Hybrid pretraining combining contrastive learning and flow matching
    """

    def __init__(self, encoder: nn.Module, target_channels: int):
        self.encoder = encoder
        self.flow_pretrainer = FlowMatchingPretrainer(encoder, target_channels)
        self.device = next(encoder.parameters()).device

    def pretrain(self, dataloader,
                 total_epochs: int = 100,
                 flow_epochs: int = 50,
                 contrastive_epochs: int = 50,
                 flow_weight: float = 1.0,
                 contrastive_weight: float = 0.5):
        """
        Hybrid pretraining: Flow matching + Contrastive learning
        """

        print(f'Starting hybrid pretraining: {flow_epochs} flow + {contrastive_epochs} contrastive epochs')

        # Phase 1: Flow matching pretraining
        print('\n=== Phase 1: Flow Matching Pretraining ===')
        self.flow_pretrainer.pretrain(dataloader, epochs=flow_epochs)

        # Phase 2: Joint training with both objectives
        print('\n=== Phase 2: Joint Flow + Contrastive Training ===')
        params = list(self.encoder.parameters()) + list(self.flow_pretrainer.flow_model.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-4)  # Lower LR for joint training
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=contrastive_epochs)

        self.encoder.train()
        self.flow_pretrainer.flow_model.train()

        for epoch in range(contrastive_epochs):
            total_flow_loss = 0
            total_contrastive_loss = 0
            batch_count = 0

            for batch_x, batch_y in dataloader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Augmentation
                augmented = IndustrialAugmentation.apply_random_augmentation(batch_x)

                # Flow loss
                if hasattr(self.encoder, 'get_rep'):
                    condition = self.encoder.get_rep(batch_x)
                else:
                    embeddings, _ = self.encoder(batch_x)
                    condition = embeddings

                target = batch_x.view(batch_x.size(0), -1)
                flow_loss = self.flow_pretrainer.flow_model(target, condition)

                # Contrastive loss
                embeddings1, _ = self.encoder(batch_x)
                embeddings2, _ = self.encoder(augmented)
                combined_embeddings = torch.cat([embeddings1, embeddings2])
                combined_labels = torch.cat([batch_y, batch_y])

                contrastive_loss_val = supervised_contrastive_loss(combined_embeddings, combined_labels)

                # Combined loss
                total_loss = flow_weight * flow_loss + contrastive_weight * contrastive_loss_val

                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                optimizer.step()

                total_flow_loss += flow_loss.item()
                total_contrastive_loss += contrastive_loss_val.item()
                batch_count += 1

            scheduler.step()

            if epoch % 10 == 0:
                avg_flow = total_flow_loss / batch_count
                avg_contrastive = total_contrastive_loss / batch_count
                lr_current = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}: Flow={avg_flow:.4f}, Contrastive={avg_contrastive:.4f}, LR={lr_current:.6f}')

        print('Hybrid pretraining completed!')
        return self.encoder


# Utility functions for flow matching
def create_flow_optimized_training_setup(encoder: nn.Module,
                                        target_channels: int,
                                        total_epochs: int = 100,
                                        use_hybrid: bool = True) -> Tuple[nn.Module, object]:
    """Create flow-optimized training setup"""

    if use_hybrid:
        trainer = HybridPretraining(encoder, target_channels)
    else:
        trainer = FlowMatchingPretrainer(encoder, target_channels)

    return encoder, trainer


def flow_based_few_shot_learning(encoder: nn.Module,
                                flow_model: nn.Module,
                                support_x: torch.Tensor,
                                support_y: torch.Tensor,
                                query_x: torch.Tensor,
                                num_synthetic: int = 5) -> torch.Tensor:
    """
    Few-shot learning enhanced with flow-generated support samples
    """
    device = support_x.device

    # Generate synthetic support samples
    with torch.no_grad():
        if hasattr(encoder, 'get_rep'):
            conditions = encoder.get_rep(support_x)
        else:
            embeddings, _ = encoder(support_x)
            conditions = embeddings

        # Generate multiple synthetic samples per support sample
        synthetic_samples = []
        synthetic_labels = []

        for i, (condition, label) in enumerate(zip(conditions, support_y)):
            if hasattr(flow_model, 'sample'):
                synthetic_flat = flow_model.sample(condition.unsqueeze(0), num_samples=num_synthetic)
                synthetic = synthetic_flat.view(num_synthetic, *support_x.shape[1:])
            else:
                # Fallback for simple flow model
                synthetic = flow_model.sample(condition.unsqueeze(0), num_samples=num_synthetic)
                synthetic = synthetic.view(num_synthetic, *support_x.shape[1:])

            synthetic_samples.append(synthetic)
            synthetic_labels.extend([label] * num_synthetic)

        synthetic_samples = torch.cat(synthetic_samples, dim=0)
        synthetic_labels = torch.tensor(synthetic_labels, device=device)

        # Combine original and synthetic support sets
        augmented_support_x = torch.cat([support_x, synthetic_samples], dim=0)
        augmented_support_y = torch.cat([support_y, synthetic_labels], dim=0)

    # Train a simple classifier on augmented support set
    classifier = nn.Linear(encoder.encoder.projection[-1].out_features if hasattr(encoder, 'encoder') else 128,
                          len(torch.unique(support_y))).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    encoder.eval()
    classifier.train()

    # Few training steps on augmented support set
    for _ in range(20):
        embeddings, _ = encoder(augmented_support_x)
        logits = classifier(embeddings)
        loss = F.cross_entropy(logits, augmented_support_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on query set
    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        query_embeddings, _ = encoder(query_x)
        query_logits = classifier(query_embeddings)

    return query_logits