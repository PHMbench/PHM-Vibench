"""
Physics-informed Probabilistic Deep Network for Trustworthy Fault Diagnosis
Based on: "Physics-informed probabilistic deep network with interpretable mechanism
for trustworthy mechanical fault diagnosis" (MSSP, 2024)

Implementation of physics-informed neural network with uncertainty quantification
for reliable fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class PhysicsInformedLayer(nn.Module):
    """Layer that incorporates physical constraints and domain knowledge"""

    def __init__(self, input_dim: int, output_dim: int, physics_params: Dict):
        """
        Initialize physics-informed layer

        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            physics_params: Dictionary containing physics parameters
        """
        super(PhysicsInformedLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Physics parameters
        self.resonance_freq = physics_params.get('resonance_freq', 100.0)
        self.damping_ratio = physics_params.get('damping_ratio', 0.1)
        self.freq_range = physics_params.get('freq_range', [0, 1000])

        # Learnable transformation
        self.linear = nn.Linear(input_dim, output_dim)
        self.physics_weight = nn.Parameter(torch.ones(1))

        # Frequency domain constraint
        self.freq_constraint = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor, freq_domain: bool = False) -> torch.Tensor:
        """
        Forward pass with physics constraints

        Args:
            x: Input tensor
            freq_domain: Whether to apply frequency domain constraints

        Returns:
            Transformed tensor with physics information
        """
        # Standard linear transformation
        x_transformed = self.linear(x)

        if freq_domain:
            # Apply frequency domain physics constraints
            x_fft = torch.fft.rfft(x, dim=-1)
            freqs = torch.fft.rfftfreq(x.size(-1), d=1.0)

            # Resonance constraint (enhance frequencies near resonance)
            resonance_mask = torch.exp(
                -((freqs - self.resonance_freq) ** 2) / (2 * (self.damping_ratio * self.resonance_freq) ** 2)
            ).to(x.device)

            # Apply physics weighting
            x_fft_weighted = x_fft * (1 + self.physics_weight * resonance_mask.unsqueeze(0).unsqueeze(0))
            x_physics = torch.fft.irfft(x_fft_weighted, n=x.size(-1), dim=-1)

            # Combine with learned transformation
            output = x_transformed + self.freq_constraint * x_physics[:, :x_transformed.size(1)]
        else:
            output = x_transformed

        return output


class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty quantification"""

    def __init__(self, in_features: int, out_features: int):
        """
        Initialize Bayesian linear layer

        Args:
            in_features: Number of input features
            out_features: Number of output features
        """
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.zeros(out_features, in_features))

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_logvar = nn.Parameter(torch.zeros(out_features))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with stochastic weights

        Args:
            x: Input tensor

        Returns:
            Tuple of (output, kl_divergence)
        """
        # Sample weights
        weight_std = torch.exp(0.5 * self.weight_logvar)
        weight = self.weight_mu + weight_std * torch.randn_like(weight_std)

        bias_std = torch.exp(0.5 * self.bias_logvar)
        bias = self.bias_mu + bias_std * torch.randn_like(bias_std)

        # Forward pass
        output = F.linear(x, weight, bias)

        # KL divergence
        kl_weight = -0.5 * torch.sum(1 + self.weight_logvar - self.weight_mu.pow(2) - weight_std.pow(2))
        kl_bias = -0.5 * torch.sum(1 + self.bias_logvar - self.bias_mu.pow(2) - bias_std.pow(2))
        kl_divergence = (kl_weight + kl_bias) / x.size(0)

        return output, kl_divergence


class ExplainableFeatureExtractor(nn.Module):
    """
    Explainable feature extractor with physical interpretability
    """

    def __init__(self, input_dim: int, hidden_dim: int, physics_params: Dict):
        """
        Initialize feature extractor

        Args:
            input_dim: Input signal dimension
            hidden_dim: Hidden dimension
            physics_params: Physics parameters dictionary
        """
        super(ExplainableFeatureExtractor, self).__init__()

        # Physics-informed layers
        self.physics_layer1 = PhysicsInformedLayer(input_dim, hidden_dim, physics_params)
        self.physics_layer2 = PhysicsInformedLayer(hidden_dim, hidden_dim, physics_params)

        # Statistical feature extractors
        self.statistical_features = nn.ModuleDict({
            'mean': lambda x: torch.mean(x, dim=-1, keepdim=True),
            'std': lambda x: torch.std(x, dim=-1, keepdim=True),
            'rms': lambda x: torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)),
            'kurtosis': lambda x: torch.mean((x - torch.mean(x, dim=-1, keepdim=True)) ** 4, dim=-1, keepdim=True) / (torch.std(x, dim=-1, keepdim=True) ** 4 + 1e-8),
            'skewness': lambda x: torch.mean((x - torch.mean(x, dim=-1, keepdim=True)) ** 3, dim=-1, keepdim=True) / (torch.std(x, dim=-1, keepdim=True) ** 3 + 1e-8),
            'peak': lambda x: torch.max(torch.abs(x), dim=-1, keepdim=True)[0],
            'crest_factor': lambda x: torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / (torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) + 1e-8),
            'clearance_factor': lambda x: torch.max(torch.abs(x), dim=-1, keepdim=True)[0] / (torch.mean(torch.sqrt(torch.abs(x)), dim=-1, keepdim=True) ** 2 + 1e-8)
        })

        # Feature importance weights
        self.feature_importance = nn.Parameter(torch.ones(len(self.statistical_features) + 2))

        # Output projection
        self.output_dim = hidden_dim + len(self.statistical_features) + 2

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Extract features with interpretability

        Args:
            x: Input signal [batch_size, seq_length]

        Returns:
            Tuple of (features, explanations)
        """
        batch_size = x.size(0)

        # Physics-informed features
        x_freq = torch.fft.rfft(x, dim=-1)
        x_freq_mag = torch.abs(x_freq)

        physics_features1 = self.physics_layer1(x_freq_mag, freq_domain=True)
        physics_features2 = self.physics_layer2(physics_features1)

        # Global pooling
        physics_features = torch.mean(physics_features2, dim=1)  # [batch_size, hidden_dim]

        # Statistical features
        statistical_feats = []
        feature_names = []

        for name, func in self.statistical_features.items():
            feat = func(x)
            statistical_feats.append(feat.squeeze(-1))
            feature_names.append(name)

        # Concatenate features
        all_features = torch.cat([physics_features] + statistical_feats, dim=1)

        # Apply feature importance weights
        importance_weights = F.softmax(self.feature_importance, dim=0)
        weighted_features = all_features * importance_weights.unsqueeze(0)

        # Create explanations
        explanations = {
            'feature_importance': importance_weights.detach().cpu().numpy(),
            'feature_names': ['physics'] + feature_names,
            'physics_weights': self.physics_layer1.physics_weight.detach().cpu().numpy(),
            'statistical_values': {name: feat[0].detach().cpu().numpy() for name, feat in zip(feature_names, statistical_feats)}
        }

        return weighted_features, explanations


class PhysicsInformedPDN(nn.Module):
    """
    Physics-informed Probabilistic Deep Network for fault diagnosis
    """

    def __init__(self,
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_samples: int = 10,
                 physics_params: Optional[Dict] = None):
        """
        Initialize the model

        Args:
            input_dim: Input signal dimension
            num_classes: Number of fault classes
            hidden_dim: Hidden dimension
            num_samples: Number of Monte Carlo samples for uncertainty
            physics_params: Physics parameters dictionary
        """
        super(PhysicsInformedPDN, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples

        # Default physics parameters
        if physics_params is None:
            physics_params = {
                'resonance_freq': 100.0,
                'damping_ratio': 0.1,
                'freq_range': [0, 1000]
            }

        # Feature extractor
        self.feature_extractor = ExplainableFeatureExtractor(input_dim, hidden_dim, physics_params)

        # Bayesian layers for uncertainty quantification
        self.bayesian_layers = nn.ModuleList([
            BayesianLinear(hidden_dim + len(self.feature_extractor.statistical_features) + 2, hidden_dim),
            BayesianLinear(hidden_dim, hidden_dim // 2),
            BayesianLinear(hidden_dim // 2, num_classes)
        ])

        # Activation and dropout
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, return_uncertainty: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, Dict]]:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, seq_length]
            return_uncertainty: Whether to return uncertainty estimates

        Returns:
            Output logits, uncertainty, and explanations
        """
        batch_size = x.size(0)

        # Extract features with explanations
        features, explanations = self.feature_extractor(x)

        if return_uncertainty:
            # Monte Carlo sampling for uncertainty
            all_outputs = []
            total_kl = 0

            for _ in range(self.num_samples):
                x_sample = features
                kl_sum = 0

                for bayesian_layer in self.bayesian_layers:
                    x_sample, kl = bayesian_layer(x_sample)
                    kl_sum += kl

                    if bayesian_layer != self.bayesian_layers[-1]:
                        x_sample = self.activation(x_sample)
                        x_sample = self.dropout(x_sample)

                all_outputs.append(x_sample)
                total_kl += kl_sum

            # Stack predictions
            predictions = torch.stack(all_outputs)  # [num_samples, batch_size, num_classes]

            # Calculate mean and variance
            mean_pred = torch.mean(predictions, dim=0)
            variance_pred = torch.var(predictions, dim=0)
            uncertainty = torch.mean(variance_pred, dim=-1)  # [batch_size]

            # Average KL divergence
            avg_kl = total_kl / self.num_samples

            # Add uncertainty to explanations
            explanations.update({
                'prediction_uncertainty': uncertainty.detach().cpu().numpy(),
                'kl_divergence': avg_kl.item(),
                'predictive_variance': variance_pred.detach().cpu().numpy()
            })

            return mean_pred, uncertainty, explanations
        else:
            # Deterministic forward pass (use mean weights)
            x_det = features

            for i, bayesian_layer in enumerate(self.bayesian_layers):
                # Use mean weights (no sampling)
                weight = bayesian_layer.weight_mu
                bias = bayesian_layer.bias_mu
                x_det = F.linear(x_det, weight, bias)

                if i < len(self.bayesian_layers) - 1:
                    x_det = self.activation(x_det)
                    x_det = self.dropout(x_det)

            return x_det, torch.zeros(batch_size), explanations

    def get_explanation(self, x: torch.Tensor, target_class: Optional[int] = None) -> Dict:
        """
        Generate detailed explanation for predictions

        Args:
            x: Input tensor
            target_class: Target class for explanation

        Returns:
            Detailed explanation dictionary
        """
        self.eval()
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits, uncertainty, explanations = self.forward(x, return_uncertainty=True)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()

            if target_class is None:
                target_class = predicted_class

        # Enhanced explanation
        explanation = {
            'prediction': predicted_class,
            'target_class': target_class,
            'confidence': probabilities[0, target_class].item(),
            'prediction_uncertainty': float(uncertainty[0]),
            'all_probabilities': probabilities.squeeze().cpu().numpy(),
            'feature_importance': explanations['feature_importance'],
            'feature_names': explanations['feature_names'],
            'physics_weights': float(explanations['physics_weights']),
            'statistical_features': explanations['statistical_values'],
            'reliability_score': float(probabilities[0, target_class].item() * (1 - uncertainty[0])),
            'uncertainty_breakdown': {
                'predictive': float(uncertainty[0]),
                'model': float(explanations['kl_divergence'])
            }
        }

        # Interpretation based on uncertainty
        if uncertainty[0] < 0.1:
            explanation['interpretation'] = "High confidence prediction - model is certain about the diagnosis"
        elif uncertainty[0] < 0.3:
            explanation['interpretation'] = "Medium confidence - prediction is reasonable but consider additional verification"
        else:
            explanation['interpretation'] = "Low confidence - model is uncertain, recommend further inspection"

        return explanation


class PhysicsInformedPDN_XFD:
    """
    Wrapper class for Physics-informed PDN compatible with UXFD framework
    """

    def __init__(self, config: Dict):
        """
        Initialize Physics-informed PDN

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model parameters
        self.input_dim = config.get('input_dim', 4096)
        self.num_classes = config.get('num_classes', 10)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_samples = config.get('num_samples', 10)

        # Physics parameters
        self.physics_params = config.get('physics_params', {
            'resonance_freq': 100.0,
            'damping_ratio': 0.1,
            'freq_range': [0, 1000]
        })

        # Initialize model
        self.model = PhysicsInformedPDN(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_samples=self.num_samples,
            physics_params=self.physics_params
        ).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        logger.info(f"Initialized Physics-informed PDN with {sum(p.numel() for p in self.model.parameters())} parameters")

    def fit(self, train_loader, val_loader=None, epochs=100):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        self.model.train()
        train_losses = []
        val_accuracies = []
        val_uncertainties = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = data.squeeze(1).to(self.device)  # Remove channel dimension
                target = target.to(self.device)

                # Forward pass with uncertainty
                self.optimizer.zero_grad()
                logits, uncertainty, _ = self.model(data, return_uncertainty=True)

                # Classification loss
                cls_loss = self.criterion(logits, target)

                # Uncertainty regularization (encourage confident predictions)
                uncertainty_loss = torch.mean(uncertainty)

                # Total loss
                total_loss = cls_loss + 0.01 * uncertainty_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += cls_loss.item()
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                              f'Loss: {cls_loss.item():.6f}, Uncertainty: {uncertainty_loss.item():.6f}')

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # Validation
            if val_loader:
                val_acc, val_unc = self._evaluate_with_uncertainty(val_loader)
                val_accuracies.append(val_acc)
                val_uncertainties.append(val_unc)
                logger.info(f'Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Acc = {val_acc:.4f}, '
                          f'Val Uncertainty = {val_unc:.4f}')

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'val_uncertainties': val_uncertainties
        }

    def _evaluate_with_uncertainty(self, data_loader):
        """
        Evaluate model with uncertainty metrics

        Args:
            data_loader: Data loader

        Returns:
            Tuple of (accuracy, mean_uncertainty)
        """
        self.model.eval()
        correct = 0
        total = 0
        total_uncertainty = 0

        with torch.no_grad():
            for data, target in data_loader:
                data = data.squeeze(1).to(self.device)
                target = target.to(self.device)

                logits, uncertainty, _ = self.model(data, return_uncertainty=True)
                pred = logits.argmax(dim=1)

                correct += (pred == target).sum().item()
                total += target.size(0)
                total_uncertainty += uncertainty.sum().item()

        accuracy = correct / total
        mean_uncertainty = total_uncertainty / total

        return accuracy, mean_uncertainty

    def predict(self, data):
        """
        Make predictions with uncertainty

        Args:
            data: Input data tensor

        Returns:
            Tuple of (predictions, probabilities, uncertainties)
        """
        self.model.eval()
        data = data.squeeze(1).to(self.device)

        with torch.no_grad():
            logits, uncertainties, _ = self.model(data, return_uncertainty=True)
            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy(), uncertainties.cpu().numpy()

    def explain(self, data, target_class=None):
        """
        Generate explanations with physics-based interpretations

        Args:
            data: Input data tensor
            target_class: Target class for explanation

        Returns:
            List of detailed explanations
        """
        self.model.eval()
        data = data.squeeze(1).to(self.device)

        explanations = []
        for i in range(data.size(0)):
            single_input = data[i:i+1]
            explanation = self.model.get_explanation(single_input, target_class)
            explanations.append(explanation)

        return explanations

    def evaluate(self, data_loader):
        """
        Evaluate model performance

        Args:
            data_loader: Test data loader

        Returns:
            Accuracy score
        """
        accuracy, _ = self._evaluate_with_uncertainty(data_loader)
        return accuracy

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'physics_params': self.physics_params
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'physics_params' in checkpoint:
            self.physics_params = checkpoint['physics_params']
        logger.info(f"Model loaded from {path}")