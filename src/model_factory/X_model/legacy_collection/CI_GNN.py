"""
CI-GNN: Granger Causality-inspired Graph Neural Network for Explainable Fault Diagnosis
Based on: "CI-GNN: A Granger causality-inspired graph neural network" (Neurocomputing, 2024)

Implementation of causality-inspired GNN with built-in interpretability
for multi-sensor fault diagnosis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class CausalityLayer(nn.Module):
    """Learn causal relationships between sensors"""

    def __init__(self, num_sensors: int, hidden_dim: int):
        """
        Initialize causality layer

        Args:
            num_sensors: Number of sensors/nodes
            hidden_dim: Hidden dimension size
        """
        super(CausalityLayer, self).__init__()
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim

        # Learnable adjacency matrix for causal relationships
        self.adjacency = nn.Parameter(torch.randn(num_sensors, num_sensors))

        # Causal embedding layers
        self.sensor_embedding = nn.Linear(1, hidden_dim)
        self.temporal_embedding = nn.Linear(10, hidden_dim)  # 10 time steps

        # Causal attention
        self.causal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, num_sensors, seq_length]

        Returns:
            Tuple of (graph_features, causal_matrix)
        """
        batch_size, num_sensors, seq_length = x.shape

        # Temporal pooling to get sensor features
        time_steps = min(10, seq_length)
        x_pooled = F.adaptive_avg_pool1d(x, time_steps)  # [batch, sensors, 10]
        x_pooled = x_pooled.permute(0, 2, 1)  # [batch, 10, sensors]

        # Sensor embeddings
        sensor_emb = self.sensor_embedding(x_pooled.mean(dim=1, keepdim=True))  # [batch, 1, hidden]
        temporal_emb = self.temporal_embedding(x_pooled)  # [batch, 10, hidden]

        # Combine embeddings
        combined_emb = temporal_emb + sensor_emb  # [batch, 10, hidden]

        # Causal attention
        attended, attention_weights = self.causal_attention(
            combined_emb, combined_emb, combined_emb
        )  # [batch, 10, hidden], [batch, 10, 10]

        # Learn causal adjacency matrix
        causal_adj = torch.sigmoid(self.adjacency)  # [sensors, sensors]
        causal_adj = causal_adj * (1 - torch.eye(num_sensors, device=x.device))  # Remove self-loops

        # Aggregate features
        graph_features = attended.mean(dim=1)  # [batch, hidden]

        return graph_features, causal_adj


class ExplainableGNN(nn.Module):
    """
    Explainable Graph Neural Network for fault diagnosis
    """

    def __init__(self,
                 num_sensors: int,
                 num_classes: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.2):
        """
        Initialize Explainable GNN

        Args:
            num_sensors: Number of sensors
            num_classes: Number of fault classes
            hidden_dim: Hidden dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
        """
        super(ExplainableGNN, self).__init__()

        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Causality layer
        self.causality_layer = CausalityLayer(num_sensors, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))

        # Attention mechanism for interpretability
        self.attention_weights = nn.Parameter(torch.ones(num_layers))

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # Feature importance for each sensor
        self.sensor_importance = nn.Parameter(torch.ones(num_sensors))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass

        Args:
            x: Input tensor [batch_size, num_sensors, seq_length]

        Returns:
            Tuple of (logits, explanations)
        """
        batch_size = x.size(0)

        # Get causal relationships and initial features
        features, causal_adj = self.causality_layer(x)  # [batch, hidden], [sensors, sensors]

        # Create graph data structure
        # Node features: repeat for each sensor
        node_features = features.unsqueeze(1).repeat(1, self.num_sensors, 1)  # [batch, sensors, hidden]

        # Edge indices from causal adjacency
        edge_index = causal_adj.nonzero().t().contiguous()  # [2, num_edges]
        edge_weight = causal_adj[causal_adj > 0]  # [num_edges]

        # Apply GNN layers with attention
        layer_outputs = []

        for i, gnn_layer in enumerate(self.gnn_layers):
            # Reshape for PyTorch Geometric
            batch_idx = torch.arange(batch_size, device=x.device).repeat_interleave(self.num_sensors)
            node_flat = node_features.view(-1, self.hidden_dim)

            # Apply GNN
            if isinstance(gnn_layer, GATConv):
                node_flat = gnn_layer(node_flat, edge_index, edge_attr=edge_weight)
            else:
                node_flat = gnn_layer(node_flat, edge_index, edge_weight=edge_weight)

            # Reshape back
            node_features = node_flat.view(batch_size, self.num_sensors, self.hidden_dim)
            layer_outputs.append(node_features)

        # Weighted combination of layer outputs
        attention_weights = F.softmax(self.attention_weights, dim=0)
        final_features = sum(w * output for w, output in zip(attention_weights, layer_outputs))

        # Global pooling
        graph_representation = global_mean_pool(
            final_features.view(-1, self.hidden_dim),
            batch_idx
        )  # [batch, hidden]

        # Apply sensor importance
        sensor_weights = F.softmax(self.sensor_importance, dim=0)
        weighted_features = (final_features * sensor_weights.unsqueeze(0).unsqueeze(-1)).sum(dim=1)

        # Combine global and weighted features
        combined_features = (graph_representation + weighted_features) / 2

        # Classification
        logits = self.classifier(combined_features)

        # Create explanations
        explanations = {
            'causal_matrix': causal_adj.detach().cpu().numpy(),
            'sensor_importance': sensor_weights.detach().cpu().numpy(),
            'layer_attention': attention_weights.detach().cpu().numpy(),
            'edge_weights': edge_weight.detach().cpu().numpy() if len(edge_weight) > 0 else np.array([]),
            'node_features': final_features.detach().cpu().numpy()
        }

        return logits, explanations

    def get_explanation(self, x: torch.Tensor, target_class: Optional[int] = None) -> Dict:
        """
        Generate detailed explanation for predictions

        Args:
            x: Input tensor [batch_size, num_sensors, seq_length]
            target_class: Target class for explanation

        Returns:
            Detailed explanation dictionary
        """
        self.eval()
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits, explanations = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = logits.argmax(dim=1).item()

            if target_class is None:
                target_class = predicted_class

        # Enhanced explanation
        explanation = {
            'prediction': predicted_class,
            'target_class': target_class,
            'confidence': probabilities[0, target_class].item(),
            'all_probabilities': probabilities.squeeze().cpu().numpy(),
            'causal_relationships': explanations['causal_matrix'],
            'sensor_importance': explanations['sensor_importance'],
            'layer_importance': explanations['layer_attention'],
            'top_influential_sensors': np.argsort(-explanations['sensor_importance'])[:5],
            'causal_strength': np.max(explanations['causal_matrix'])
        }

        # Analyze causal paths
        causal_matrix = explanations['causal_matrix']
        threshold = np.percentile(causal_matrix, 90)
        strong_causal_edges = np.where(causal_matrix > threshold)

        explanation['strong_causal_paths'] = [
            (int(src), int(dst), float(causal_matrix[src, dst]))
            for src, dst in zip(strong_causal_edges[0], strong_causal_edges[1])
        ]

        return explanation


class CI_GNN_XFD:
    """
    Wrapper class for CI-GNN model compatible with UXFD framework
    """

    def __init__(self, config: Dict):
        """
        Initialize CI-GNN model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Model parameters
        self.num_sensors = config.get('num_sensors', 8)
        self.num_classes = config.get('num_classes', 10)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)

        # Initialize model
        self.model = ExplainableGNN(
            num_sensors=self.num_sensors,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer with different learning rates for different components
        self.optimizer = torch.optim.Adam([
            {'params': self.model.causality_layer.parameters(), 'lr': 0.001},
            {'params': self.model.gnn_layers.parameters(), 'lr': 0.01},
            {'params': self.model.classifier.parameters(), 'lr': 0.01},
            {'params': self.model.sensor_importance, 'lr': 0.01},
            {'params': self.model.attention_weights, 'lr': 0.01}
        ], weight_decay=config.get('weight_decay', 1e-4))

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )

        logger.info(f"Initialized CI-GNN with {sum(p.numel() for p in self.model.parameters())} parameters")

    def _prepare_data(self, data):
        """
        Prepare data for multi-sensor input

        Args:
            data: Raw data tensor

        Returns:
            Formatted data [batch_size, num_sensors, seq_length]
        """
        # If data is 1D, reshape to multi-sensor format
        if data.dim() == 3 and data.size(1) == 1:
            # Split single sensor into multiple virtual sensors
            data = data.repeat(1, self.num_sensors, 1)
        elif data.dim() == 2:
            data = data.unsqueeze(1).repeat(1, self.num_sensors, 1)

        return data

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
        best_val_acc = 0.0

        for epoch in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data = self._prepare_data(data).to(self.device)
                target = target.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                logits, _ = self.model(data)
                loss = self.criterion(logits, target)

                # L1 regularization on causal matrix for sparsity
                causal_adj = self.model.causality_layer.adjacency
                l1_loss = 0.01 * torch.sum(torch.abs(causal_adj))
                total_loss = loss + l1_loss

                # Backward pass
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if batch_idx % 100 == 0:
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}, '
                              f'Loss: {loss.item():.6f}, L1: {l1_loss.item():.6f}')

            # Average loss for epoch
            avg_loss = epoch_loss / num_batches
            train_losses.append(avg_loss)

            # Validation
            if val_loader:
                val_acc = self.evaluate(val_loader)
                val_accuracies.append(val_acc)
                self.scheduler.step(val_acc)

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    self.save_model('best_ci_gnn_model.pth', save_optimizer=False)

                logger.info(f'Epoch {epoch}: Train Loss = {avg_loss:.6f}, Val Acc = {val_acc:.4f}')

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc
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
        data = self._prepare_data(data).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(data)
            probabilities = F.softmax(logits, dim=1)
            predictions = logits.argmax(dim=1)

        return predictions.cpu().numpy(), probabilities.cpu().numpy()

    def explain(self, data, target_class=None):
        """
        Generate explanations for predictions

        Args:
            data: Input data tensor
            target_class: Target class for explanation

        Returns:
            List of explanations
        """
        self.model.eval()
        data = self._prepare_data(data).to(self.device)

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
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in data_loader:
                data = self._prepare_data(data).to(self.device)
                target = target.to(self.device)
                logits, _ = self.model(data)
                pred = logits.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        return correct / total

    def save_model(self, path, save_optimizer=True):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config
        }
        if save_optimizer:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

    def get_causal_graph(self):
        """
        Get learned causal relationships between sensors

        Returns:
            Causal adjacency matrix
        """
        self.model.eval()
        with torch.no_grad():
            causal_adj = torch.sigmoid(self.model.causality_layer.adjacency)
            return causal_adj.cpu().numpy()