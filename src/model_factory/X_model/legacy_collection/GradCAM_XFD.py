"""
1D-Grad-CAM for Explainable Fault Diagnosis
Adapted from: https://github.com/liguge/1D-Grad-CAM-for-interpretable-intelligent-fault-diagnosis

Implementation of 1D Gradient-weighted Class Activation Mapping for interpretability
in fault diagnosis using 1D vibration signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class GradCAM1D:
    """1D Grad-CAM implementation for fault diagnosis interpretability"""

    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize 1D Grad-CAM

        Args:
            model: The neural network model
            target_layer: Name of the target layer for visualization
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks"""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                break
        else:
            raise ValueError(f"Layer {self.target_layer} not found in model")

    def generate_cam(self, input_tensor: torch.Tensor,
                    class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Class Activation Map

        Args:
            input_tensor: Input signal tensor [batch_size, channels, length]
            class_idx: Target class index (if None, use predicted class)

        Returns:
            CAM array [batch_size, length]
        """
        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        class_idx_tensor = torch.tensor([class_idx], device=output.device) if isinstance(class_idx, int) else class_idx
        one_hot.scatter_(1, class_idx_tensor.unsqueeze(1), 1)
        output.backward(gradient=one_hot)

        # Get gradients and activations
        gradients = self.gradients  # [batch_size, channels, length]
        activations = self.activations  # [batch_size, channels, length]

        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=2)  # [batch_size, channels]

        # Weighted combination of activations
        cam = torch.zeros((input_tensor.size(0), activations.size(2)))
        for i in range(weights.size(0)):
            for j in range(weights.size(1)):
                cam[i] += weights[i, j] * activations[i, j]

        # ReLU to keep only positive influences
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.detach().cpu().numpy()


class ExplainableCNN(nn.Module):
    """
    Explainable 1D CNN for fault diagnosis with Grad-CAM support
    """

    def __init__(self,
                 input_channels: int = 1,
                 num_classes: int = 10,
                 seq_length: int = 4096,
                 dropout: float = 0.2):
        """
        Initialize explainable CNN

        Args:
            input_channels: Number of input channels
            num_classes: Number of fault classes
            seq_length: Input sequence length
            dropout: Dropout rate
        """
        super(ExplainableCNN, self).__init__()

        self.seq_length = seq_length

        # Convolutional layers with increasing receptive fields
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=64, stride=16, padding=32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize weights
        self._initialize_weights()

        # Store layer names for Grad-CAM
        self.layer_names = {
            'conv1': self.conv1,
            'conv2': self.conv2,
            'conv3': self.conv3,
            'conv4': self.conv4
        }

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, channels, seq_length]

        Returns:
            Output logits [batch_size, num_classes]
        """
        # Convolutional feature extraction
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def get_explanation(self,
                       input_signal: torch.Tensor,
                       target_layer: str = 'conv3',
                       class_idx: Optional[int] = None) -> Dict:
        """
        Generate model explanation using Grad-CAM

        Args:
            input_signal: Input signal [1, channels, seq_length]
            target_layer: Layer to visualize
            class_idx: Target class for explanation

        Returns:
            Dictionary containing:
            - 'prediction': Predicted class
            - 'probabilities': Class probabilities
            - 'cam': Class activation map
            - 'explanation': Text explanation
        """
        self.eval()

        with torch.no_grad():
            # Get prediction
            logits = self(input_signal)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()

        # Generate CAM
        grad_cam = GradCAM1D(self, target_layer)
        cam = grad_cam.generate_cam(input_signal, class_idx or predicted_class)

        # Create explanation
        if class_idx is None:
            class_idx = predicted_class

        # Find important regions (top 10% of CAM values)
        cam_threshold = np.percentile(cam[0], 90)
        important_regions = np.where(cam[0] > cam_threshold)[0]

        explanation = {
            'prediction': predicted_class,
            'probabilities': probabilities.squeeze().cpu().numpy(),
            'cam': cam[0],
            'important_regions': important_regions,
            'target_class': class_idx,
            'confidence': probabilities[0, class_idx].item()
        }

        return explanation


class GradCAM_XFD:
    """
    Wrapper class for 1D Grad-CAM explainable fault diagnosis model
    Compatible with UXFD framework
    """

    def __init__(self, config: Dict):
        """
        Initialize GradCAM-XFD model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model parameters
        self.input_channels = config.get('input_channels', 1)
        self.num_classes = config.get('num_classes', 10)
        self.seq_length = config.get('seq_length', 4096)
        self.dropout = config.get('dropout', 0.2)

        # Initialize model
        self.model = ExplainableCNN(
            input_channels=self.input_channels,
            num_classes=self.num_classes,
            seq_length=self.seq_length,
            dropout=self.dropout
        ).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.get('step_size', 30),
            gamma=config.get('gamma', 0.1)
        )

        logger.info(f"Initialized GradCAM-XFD with {sum(p.numel() for p in self.model.parameters())} parameters")

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

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}')

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # Validation
            if val_loader:
                val_acc = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                logger.info(f'Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Acc = {val_acc:.4f}')

            self.scheduler.step()

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies
        }

    def predict(self, data):
        """
        Make predictions

        Args:
            data: Input data tensor

        Returns:
            Predicted classes and probabilities
        """
        self.model.eval()
        data = data.to(self.device)

        with torch.no_grad():
            output = self.model(data)
            probabilities = F.softmax(output, dim=1)
            predictions = output.argmax(dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def explain(self, data, target_layer='conv3'):
        """
        Generate explanations for predictions

        Args:
            data: Input data tensor [batch_size, channels, seq_length]
            target_layer: Layer for Grad-CAM visualization

        Returns:
            List of explanations for each sample
        """
        self.model.eval()
        explanations = []

        for i in range(data.size(0)):
            single_input = data[i:i+1].to(self.device)
            explanation = self.model.get_explanation(single_input, target_layer)
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
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total

    def save_model(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")