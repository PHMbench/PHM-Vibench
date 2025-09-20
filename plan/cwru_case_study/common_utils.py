#!/usr/bin/env python3
"""
Common utilities for CWRU Multi-Task Few-Shot Learning Study

This module contains shared components used by all three experimental cases:
- Data loading and preprocessing
- Model architectures (UnifiedEncoder, task heads)
- Training utilities
- Evaluation metrics
- Logging utilities

Author: PHM-Vibench Development Team
Date: September 2025
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import h5py
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
import logging
import pickle
from datetime import datetime
import sys

warnings.filterwarnings("ignore")

# ==================== CONSTANTS ====================
# Data paths
DATA_DIR = '/mnt/crucial/LQ/PHM-Vibench'
METADATA_FILE = os.path.join(DATA_DIR, 'metadata_6_11.xlsx')
H5_FILE = os.path.join(DATA_DIR, 'RM_001_CWRU.h5')

# Windowing parameters
WINDOW_SIZE = 1024
STRIDE = 256
SAMPLE_RATE = 12000

# Few-shot learning parameters
N_SUPPORT = 5
N_QUERY = 15
N_CLASSES_DIAG = 4
N_CLASSES_ANOM = 2
N_CHANNELS = 2

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PRETRAIN_EPOCHS = 50
FINETUNE_EPOCHS = 30

# Tasks configuration
TASKS_TO_RUN = {
    'diagnosis': True,
    'anomaly': True,
    'prediction': True
}

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== LOGGING UTILITIES ====================
def setup_logger(name, log_file, level=logging.INFO):
    """Setup logger with both file and console handlers"""
    formatter = logging.Formatter(
        '%(asctime)s %(name)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # Logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_system_info(logger):
    """Log system and environment information"""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Random seed: 42")

# ==================== DATA LOADING AND PREPROCESSING ====================
def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def load_cwru_data(logger):
    """Load and preprocess CWRU dataset"""
    logger.info("Loading CWRU dataset...")

    # Load metadata
    if not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

    metadata = pd.read_excel(METADATA_FILE)
    cwru_meta = metadata[metadata['Name'] == 'RM_001_CWRU'].reset_index(drop=True)
    logger.info(f"Found {len(cwru_meta)} CWRU files in metadata")

    # Load H5 data
    if not os.path.exists(H5_FILE):
        raise FileNotFoundError(f"H5 file not found: {H5_FILE}")

    with h5py.File(H5_FILE, 'r') as f:
        logger.info(f"H5 file keys: {list(f.keys())}")

        signals_list = []
        labels_diag_list = []
        labels_anom_list = []
        file_ids_list = []

        for idx, row in cwru_meta.iterrows():
            file_id = row['Id']

            try:
                if str(file_id) in f:
                    signal_data = f[str(file_id)][:]

                    if signal_data.ndim == 1:
                        signal_data = signal_data.reshape(-1, 1)

                    if signal_data.shape[1] > N_CHANNELS:
                        signal_data = signal_data[:, :N_CHANNELS]
                    elif signal_data.shape[1] < N_CHANNELS:
                        signal_data = np.pad(signal_data, ((0, 0), (0, N_CHANNELS - signal_data.shape[1])), mode='constant')

                    # Create windows
                    windows = create_windows(signal_data, WINDOW_SIZE, STRIDE)

                    # Labels
                    diag_label = row.get('Label', 0)
                    anom_label = 1 if diag_label > 0 else 0

                    signals_list.append(windows)
                    labels_diag_list.extend([diag_label] * len(windows))
                    labels_anom_list.extend([anom_label] * len(windows))
                    file_ids_list.extend([file_id] * len(windows))

                    logger.debug(f"Loaded file {file_id}: {len(windows)} windows, diag={diag_label}, anom={anom_label}")

            except Exception as e:
                logger.warning(f"Failed to load file {file_id}: {e}")
                continue

    # Combine all data
    all_signals = np.concatenate(signals_list, axis=0)
    all_diag_labels = np.array(labels_diag_list)
    all_anom_labels = np.array(labels_anom_list)
    all_file_ids = np.array(file_ids_list)

    # Squeeze last dimension if it exists (remove trailing dimension of size 1)
    if all_signals.ndim == 4 and all_signals.shape[-1] == 1:
        all_signals = all_signals.squeeze(-1)

    logger.info(f"Total loaded: {len(all_signals)} windows")
    logger.info(f"Signal shape: {all_signals.shape}")
    logger.info(f"Diagnosis classes: {np.unique(all_diag_labels, return_counts=True)}")
    logger.info(f"Anomaly classes: {np.unique(all_anom_labels, return_counts=True)}")

    # Normalize signals
    scaler = StandardScaler()
    all_signals_norm = scaler.fit_transform(all_signals.reshape(-1, all_signals.shape[-1]))
    all_signals_norm = all_signals_norm.reshape(all_signals.shape)

    return all_signals_norm, all_diag_labels, all_anom_labels, all_file_ids, scaler

def create_windows(signal, window_size, stride):
    """Create sliding windows from signal"""
    if len(signal) < window_size:
        # Pad if signal is too short
        padding = window_size - len(signal)
        signal = np.pad(signal, ((0, padding), (0, 0)), mode='constant')

    windows = []
    for i in range(0, len(signal) - window_size + 1, stride):
        window = signal[i:i + window_size]
        windows.append(window)

    return np.array(windows)

def create_few_shot_episodes(signals, labels, n_support, n_query, n_classes, logger):
    """Create few-shot learning episodes"""
    logger.info(f"Creating few-shot episodes: {n_support}-shot, {n_query} query per class")

    # Filter to only include samples from available classes
    available_classes = np.unique(labels)
    if len(available_classes) < n_classes:
        logger.warning(f"Only {len(available_classes)} classes available, requested {n_classes}")
        n_classes = len(available_classes)

    selected_classes = available_classes[:n_classes]

    support_signals = []
    support_labels = []
    query_signals = []
    query_labels = []

    for class_idx, class_label in enumerate(selected_classes):
        class_mask = labels == class_label
        class_signals = signals[class_mask]

        if len(class_signals) < n_support + n_query:
            logger.warning(f"Class {class_label} has only {len(class_signals)} samples, need {n_support + n_query}")
            continue

        # Random sampling
        indices = np.random.permutation(len(class_signals))
        support_indices = indices[:n_support]
        query_indices = indices[n_support:n_support + n_query]

        support_signals.append(class_signals[support_indices])
        support_labels.extend([class_idx] * n_support)

        query_signals.append(class_signals[query_indices])
        query_labels.extend([class_idx] * n_query)

    support_x = np.concatenate(support_signals, axis=0)
    support_y = np.array(support_labels)
    query_x = np.concatenate(query_signals, axis=0)
    query_y = np.array(query_labels)

    logger.info(f"Created episode: Support {support_x.shape}, Query {query_x.shape}")

    return support_x, support_y, query_x, query_y

# ==================== MODEL ARCHITECTURES ====================
class UnifiedEncoder(nn.Module):
    """Single encoder architecture used by ALL cases for fair comparison"""
    def __init__(self, input_channels=2, feature_dim=128):
        super(UnifiedEncoder, self).__init__()
        self.feature_dim = feature_dim

        # Same CNN architecture for all cases
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Optional projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x, use_projection=False):
        """Forward pass with optional projection for contrastive learning"""
        x = x.transpose(1, 2)
        feature_map = self.conv_layers(x)
        pooled_features = self.pool(feature_map).squeeze(-1)

        if use_projection:
            embeddings = self.projection(pooled_features)
            return embeddings, pooled_features
        else:
            # Return feature_map and pooled features for compatibility with heads
            return feature_map, pooled_features

    def get_rep(self, x):
        """Get representation for flow matching compatibility"""
        _, pooled_features = self.forward(x, use_projection=False)
        return pooled_features

class ClassificationHead(nn.Module):
    """Classification head for fault diagnosis and anomaly detection"""
    def __init__(self, in_dim, n_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(in_dim, n_classes)

    def forward(self, feature_map, pooled):
        return self.fc(pooled)

class PredictionHead(nn.Module):
    """Prediction head for signal forecasting"""
    def __init__(self, feature_channels, output_channels=2):
        super(PredictionHead, self).__init__()
        # Decoder with upsampling to restore original sequence length
        # Input: 256 length -> Output: 1024 length (4x upsampling)
        self.decoder = nn.Sequential(
            nn.Conv1d(feature_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # 256 -> 512
            nn.Conv1d(64, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),  # 512 -> 1024
            nn.Conv1d(32, output_channels, kernel_size=7, padding=3),
        )

    def forward(self, feature_map, pooled):
        out = self.decoder(feature_map)
        return out.transpose(1, 2)

class DirectFewShotModel(nn.Module):
    """Multi-task model for few-shot learning"""
    def __init__(self, input_channels, tasks_config, n_classes_diag=4, n_classes_anom=2):
        super(DirectFewShotModel, self).__init__()
        self.backbone = UnifiedEncoder(input_channels)
        self.tasks_config = tasks_config
        self.heads = nn.ModuleDict()
        feature_dim = self.backbone.feature_dim

        if tasks_config.get('diagnosis', False):
            self.heads['diagnosis'] = ClassificationHead(feature_dim, n_classes_diag)
        if tasks_config.get('anomaly', False):
            self.heads['anomaly'] = ClassificationHead(feature_dim, n_classes_anom)
        if tasks_config.get('prediction', False):
            self.heads['prediction'] = PredictionHead(feature_dim, input_channels)

    def forward(self, x, task):
        if task not in self.heads:
            raise ValueError(f'Task {task} is not enabled for this model.')
        feature_map, pooled = self.backbone(x)
        return self.heads[task](feature_map, pooled)

# ==================== TRAINING UTILITIES ====================
def train_classification(model, support_x, support_y, query_x, query_y, task_name,
                        epochs=30, lr=0.001, logger=None):
    """Train classification task with few-shot learning"""
    if logger is None:
        logger = logging.getLogger(__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    losses, accuracies = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Train on support set
        support_logits = model(support_x, task_name)
        loss = criterion(support_logits, support_y)
        loss.backward()
        optimizer.step()

        # Evaluate on query set
        model.eval()
        with torch.no_grad():
            query_logits = model(query_x, task_name)
            query_preds = torch.argmax(query_logits, dim=1)
            accuracy = (query_preds == query_y).float().mean().item()

        losses.append(loss.item())
        accuracies.append(accuracy)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}: Loss {loss.item():.4f}, Accuracy {accuracy:.4f}")

    logger.info(f"Final {task_name} accuracy: {accuracies[-1]:.4f}")
    return losses, accuracies

def train_prediction(model, support_x, support_y, query_x, query_y, epochs=30, lr=0.001, logger=None):
    """Train prediction task with MSE loss"""
    if logger is None:
        logger = logging.getLogger(__name__)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    support_x, support_y = support_x.to(device), support_y.to(device)
    query_x, query_y = query_x.to(device), query_y.to(device)

    losses, mse_values = [], []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Train on support set
        support_pred = model(support_x, 'prediction')
        loss = criterion(support_pred, support_y)
        loss.backward()
        optimizer.step()

        # Evaluate on query set
        model.eval()
        with torch.no_grad():
            query_pred = model(query_x, 'prediction')
            mse = F.mse_loss(query_pred, query_y).item()

        losses.append(loss.item())
        mse_values.append(mse)

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch:3d}: Loss {loss.item():.6f}, Query MSE {mse:.6f}")

    logger.info(f"Final prediction MSE: {mse_values[-1]:.6f}")
    return losses, mse_values

def contrastive_loss(z1, z2, temperature=0.1):
    """SimCLR-style contrastive loss"""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # 2B x D

    # Normalize embeddings
    z = F.normalize(z, dim=1)

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.t()) / temperature  # 2B x 2B

    # Create labels for positive pairs
    labels = torch.cat([torch.arange(batch_size), torch.arange(batch_size)]).to(z.device)

    # Mask out self-similarities
    mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
    sim_matrix.masked_fill_(mask, -float('inf'))

    # Compute cross-entropy loss
    loss = F.cross_entropy(sim_matrix, labels)
    return loss

# ==================== FLOW MATCHING UTILITIES ====================
try:
    from src.task_factory.Components.flow import FlowLoss
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False

class SimpleFlowModel(nn.Module):
    """Simplified flow model when FlowLoss is not available"""
    def __init__(self, target_channels, z_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(target_channels + z_channels + 1, 256),  # +1 for time
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, target_channels)
        )

    def forward(self, target, condition):
        batch_size = target.size(0)
        target_flat = target.view(batch_size, -1)

        # Sample random time steps
        t = torch.rand(batch_size, 1).to(target.device)

        # Sample noise
        noise = torch.randn_like(target_flat)

        # Linear interpolation
        noised = t * target_flat + (1 - t) * noise

        # Concatenate inputs
        input_tensor = torch.cat([noised, condition, t], dim=1)

        # Predict velocity
        velocity_pred = self.net(input_tensor)

        # True velocity (target - noise)
        velocity_true = target_flat - noise

        # MSE loss
        loss = F.mse_loss(velocity_pred, velocity_true)
        return loss

# ==================== EVALUATION UTILITIES ====================
def evaluate_classification_metrics(y_true, y_pred, logger=None):
    """Compute comprehensive classification metrics"""
    if logger is None:
        logger = logging.getLogger(__name__)

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

    logger.info(f"Classification metrics: {metrics}")
    return metrics

def save_results(results, filepath, logger=None):
    """Save results to pickle file"""
    if logger is None:
        logger = logging.getLogger(__name__)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, 'wb') as f:
        pickle.dump(results, f)

    logger.info(f"Results saved to {filepath}")

def load_results(filepath, logger=None):
    """Load results from pickle file"""
    if logger is None:
        logger = logging.getLogger(__name__)

    with open(filepath, 'rb') as f:
        results = pickle.load(f)

    logger.info(f"Results loaded from {filepath}")
    return results

# ==================== INITIALIZATION ====================
def init_common_setup():
    """Initialize common setup for all cases"""
    set_random_seeds(42)
    print(f"Common utilities initialized")
    print(f"Device: {device}")
    print(f"PyTorch version: {torch.__version__}")
    return device

if __name__ == "__main__":
    # Test common utilities
    print("Testing common utilities...")

    # Test logger setup
    logger = setup_logger("test", "logs/test.log")
    log_system_info(logger)

    # Test device setup
    device = init_common_setup()

    # Test model creation
    encoder = UnifiedEncoder(N_CHANNELS)
    model = DirectFewShotModel(N_CHANNELS, TASKS_TO_RUN)

    logger.info(f"UnifiedEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    logger.info(f"DirectFewShotModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    test_input = torch.randn(4, WINDOW_SIZE, N_CHANNELS)
    with torch.no_grad():
        feature_map, pooled = encoder(test_input)
        logger.info(f"Encoder output: feature_map {feature_map.shape}, pooled {pooled.shape}")

        if 'diagnosis' in TASKS_TO_RUN and TASKS_TO_RUN['diagnosis']:
            diag_output = model(test_input, 'diagnosis')
            logger.info(f"Diagnosis output: {diag_output.shape}")

    logger.info("✅ Common utilities test completed successfully!")