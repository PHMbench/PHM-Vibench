"""
Time Series Forecasting Example for PHM-Vibench Model Factory

This example demonstrates how to use different model architectures
for time-series forecasting in industrial applications.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# Import PHM-Vibench model factory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_factory import build_model


def generate_synthetic_timeseries(num_samples=1000, seq_len=168, pred_len=24, num_features=6):
    """
    Generate synthetic multivariate time series data.
    
    Parameters:
    - num_samples: Number of time series samples
    - seq_len: Input sequence length (e.g., 168 hours = 1 week)
    - pred_len: Prediction length (e.g., 24 hours = 1 day)
    - num_features: Number of features (sensors)
    """
    np.random.seed(42)
    
    data = []
    targets = []
    
    for i in range(num_samples):
        # Generate base time series with trend, seasonality, and noise
        t = np.arange(seq_len + pred_len)
        
        # Different patterns for each feature
        series = np.zeros((seq_len + pred_len, num_features))
        
        for j in range(num_features):
            # Trend component
            trend = 0.01 * t + np.random.normal(0, 0.1)
            
            # Seasonal components (daily and weekly patterns)
            daily_season = 2 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 0.2, len(t))
            weekly_season = 1.5 * np.sin(2 * np.pi * t / (24 * 7)) + np.random.normal(0, 0.15, len(t))
            
            # Random walk component
            random_walk = np.cumsum(np.random.normal(0, 0.1, len(t)))
            
            # Combine components with feature-specific weights
            feature_weight = 1 + 0.5 * j  # Different scales for different features
            series[:, j] = feature_weight * (trend + daily_season + weekly_season + random_walk)
            
            # Add some feature-specific patterns
            if j == 0:  # Temperature-like pattern
                series[:, j] += 20 + 5 * np.sin(2 * np.pi * t / (24 * 7))
            elif j == 1:  # Vibration-like pattern
                series[:, j] += 0.5 * np.sin(2 * np.pi * t * 50 / len(t))
            elif j == 2:  # Pressure-like pattern
                series[:, j] += 100 + 10 * np.sin(2 * np.pi * t / 24)
        
        # Split into input and target
        input_seq = series[:seq_len]
        target_seq = series[seq_len:seq_len + pred_len]
        
        data.append(input_seq)
        targets.append(target_seq)
    
    return torch.FloatTensor(data), torch.FloatTensor(targets)


def create_forecasting_dataloaders(data, targets, batch_size=32, train_ratio=0.8):
    """Create train and test dataloaders for forecasting."""
    dataset = TensorDataset(data, targets)
    
    # Split into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_forecasting_model(model, train_loader, num_epochs=50, learning_rate=1e-3):
    """Train the forecasting model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    train_losses = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            
            # Handle different output shapes
            if output.dim() == 2:  # (batch_size, features)
                # Expand to match target shape
                output = output.unsqueeze(1).expand(-1, target.size(1), -1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}')
    
    return train_losses


def evaluate_forecasting_model(model, test_loader):
    """Evaluate the forecasting model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle different output shapes
            if output.dim() == 2:  # (batch_size, features)
                output = output.unsqueeze(1).expand(-1, target.size(1), -1)
            
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return predictions.numpy(), targets.numpy()


def calculate_metrics(predictions, targets):
    """Calculate forecasting metrics."""
    # Flatten for overall metrics
    pred_flat = predictions.reshape(-1)
    target_flat = targets.reshape(-1)
    
    mse = mean_squared_error(target_flat, pred_flat)
    mae = mean_absolute_error(target_flat, pred_flat)
    rmse = np.sqrt(mse)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((target_flat - pred_flat) / (target_flat + 1e-8))) * 100
    
    return {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}


def plot_forecasting_results(train_losses, predictions, targets, feature_names=None):
    """Plot forecasting results."""
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(targets.shape[-1])]
    
    num_features = targets.shape[-1]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Training loss
    axes[0].plot(train_losses)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].grid(True)
    
    # Plot predictions vs targets for each feature (first sample)
    sample_idx = 0
    for i in range(min(num_features, 5)):  # Plot up to 5 features
        ax_idx = i + 1
        if ax_idx < len(axes):
            time_steps = range(targets.shape[1])
            axes[ax_idx].plot(time_steps, targets[sample_idx, :, i], 'b-', label='Actual', linewidth=2)
            axes[ax_idx].plot(time_steps, predictions[sample_idx, :, i], 'r--', label='Predicted', linewidth=2)
            axes[ax_idx].set_title(f'{feature_names[i]} - Sample {sample_idx + 1}')
            axes[ax_idx].set_xlabel('Time Steps')
            axes[ax_idx].set_ylabel('Value')
            axes[ax_idx].legend()
            axes[ax_idx].grid(True)
    
    plt.tight_layout()
    plt.savefig('forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating forecasting with different models."""
    print("PHM-Vibench Forecasting Example")
    print("=" * 50)
    
    # Generate synthetic time series data
    print("Generating synthetic multivariate time series data...")
    seq_len = 168  # 1 week of hourly data
    pred_len = 24  # Predict next 24 hours
    num_features = 6  # 6 different sensors
    
    data, targets = generate_synthetic_timeseries(
        num_samples=800, 
        seq_len=seq_len, 
        pred_len=pred_len, 
        num_features=num_features
    )
    print(f"Data shape: {data.shape}, Targets shape: {targets.shape}")
    
    # Create dataloaders
    train_loader, test_loader = create_forecasting_dataloaders(data, targets, batch_size=32)
    
    # Test different forecasting models
    models_to_test = [
        {
            'name': 'Dlinear',
            'args': Namespace(
                model_name='Dlinear',
                input_dim=num_features,
                seq_len=seq_len,
                pred_len=pred_len,
                kernel_size=25,
                individual=False
            )
        },
        {
            'name': 'PatchTST',
            'args': Namespace(
                model_name='PatchTST',
                input_dim=num_features,
                d_model=128,
                n_heads=8,
                e_layers=3,
                patch_len=16,
                stride=8,
                seq_len=seq_len,
                pred_len=pred_len
            )
        },
        {
            'name': 'FNO',
            'args': Namespace(
                model_name='FNO',
                input_dim=num_features,
                output_dim=num_features,
                modes=32,
                width=64,
                num_layers=4
            )
        }
    ]
    
    results = {}
    feature_names = ['Temperature', 'Vibration', 'Pressure', 'Current', 'Speed', 'Torque']
    
    for model_config in models_to_test:
        print(f"\nTraining {model_config['name']}...")
        print("-" * 30)
        
        # Build model
        model = build_model(model_config['args'])
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        train_losses = train_forecasting_model(model, train_loader, num_epochs=50)
        
        # Evaluate model
        predictions, targets_eval = evaluate_forecasting_model(model, test_loader)
        
        # Calculate metrics
        metrics = calculate_metrics(predictions, targets_eval)
        
        print(f"Test Metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.6f}")
        
        # Store results
        results[model_config['name']] = {
            'metrics': metrics,
            'train_losses': train_losses,
            'predictions': predictions,
            'targets': targets_eval
        }
    
    # Plot results for the best model (lowest RMSE)
    best_model = min(results.keys(), key=lambda k: results[k]['metrics']['RMSE'])
    print(f"\nBest model: {best_model} with RMSE: {results[best_model]['metrics']['RMSE']:.6f}")
    
    plot_forecasting_results(
        results[best_model]['train_losses'],
        results[best_model]['predictions'],
        results[best_model]['targets'],
        feature_names
    )
    
    # Summary comparison
    print("\nModel Comparison (RMSE):")
    print("-" * 40)
    for model_name, result in results.items():
        rmse = result['metrics']['RMSE']
        mae = result['metrics']['MAE']
        mape = result['metrics']['MAPE']
        print(f"{model_name:15}: RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
    
    # Feature-wise analysis for best model
    print(f"\nFeature-wise RMSE for {best_model}:")
    print("-" * 40)
    best_predictions = results[best_model]['predictions']
    best_targets = results[best_model]['targets']
    
    for i, feature_name in enumerate(feature_names):
        feature_rmse = np.sqrt(mean_squared_error(
            best_targets[:, :, i].flatten(),
            best_predictions[:, :, i].flatten()
        ))
        print(f"{feature_name:12}: {feature_rmse:.6f}")


if __name__ == "__main__":
    main()
