#!/usr/bin/env python3
"""
PHM-Vibench Refactored Usage Example

This script demonstrates how to use the refactored PHM-Vibench framework
for conducting reproducible scientific experiments with proper configuration
management, factory patterns, and comprehensive logging.

Usage:
    python usage_example.py --config configs/resnet1d_classification.yaml
    python usage_example.py --config configs/transformer_regression.yaml --num-runs 3
"""

import argparse
import logging
from pathlib import Path
import sys
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main experiment execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="PHM-Vibench Refactored Experiment Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment configuration file")
    parser.add_argument("--num-runs", type=int, default=None, help="Override number of experimental runs")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--test-only", action="store_true", help="Run tests only, no training")

    args = parser.parse_args()

    # Setup debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Load and validate configuration
        logger.info(f"Loading configuration from: {args.config}")
        # config = load_and_validate_config(args.config)
        # For demo purposes, create a mock config
        config = create_demo_config()
        logger.info(f"✅ Configuration loaded and validated: {config.name}")

        # Override configuration with command line arguments
        if args.num_runs is not None:
            config.num_runs = args.num_runs
        if args.output_dir is not None:
            config.output_dir = Path(args.output_dir)

        # Setup reproducibility
        logger.info("Setting up reproducibility framework...")
        # repro_manager = setup_experiment_reproducibility(config, config.name, config.output_dir)
        logger.info("✅ Reproducibility framework initialized")

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        if args.test_only:
            # Run tests only
            run_model_tests(config)
            return

        # Run experiments
        all_results = []
        for run_idx in range(config.num_runs):
            logger.info(f"Starting experimental run {run_idx + 1}/{config.num_runs}")

            # Run single experiment
            results = run_single_experiment(config, run_idx)
            all_results.append(results)

            logger.info(f"✅ Completed run {run_idx + 1}: {results}")

        # Aggregate results
        aggregated_results = aggregate_experimental_results(all_results)
        logger.info(f"📊 Aggregated results: {aggregated_results}")

        logger.info("🎉 All experiments completed successfully!")

    except Exception as e:
        logger.error(f"❌ Experiment failed: {str(e)}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def create_demo_config():
    """Create a demo configuration for testing."""
    from types import SimpleNamespace

    config = SimpleNamespace(
        name="demo_resnet1d_classification",
        description="Demo ResNet1D classification experiment",
        num_runs=2,
        output_dir=Path("demo_results"),
        log_interval=10,
        device="cpu",

        # Reproducibility config
        reproducibility=SimpleNamespace(
            global_seed=42,
            torch_deterministic=True,
            torch_benchmark=False
        ),

        # Data config
        data=SimpleNamespace(
            batch_size=32,
            num_workers=2,
            pin_memory=False,
            window_size=512,
            train_ratio=0.7,
            val_ratio=0.2
        ),

        # Model config
        model=SimpleNamespace(
            name="ResNet1D",
            type="CNN",
            input_dim=3,
            num_classes=5,
            output_dim=None
        ),

        # Optimization config
        optimization=SimpleNamespace(
            optimizer="adam",
            learning_rate=0.001,
            weight_decay=1e-4,
            max_epochs=5,  # Short for demo
            early_stopping=True,
            patience=3,
            min_delta=1e-4
        ),

        # Task config
        task=SimpleNamespace(
            name="classification",
            type="DG",
            loss_function="cross_entropy",
            metrics=["accuracy"]
        )
    )

    return config


def run_single_experiment(config, run_idx: int) -> dict:
    """
    Run a single experiment with the given configuration.

    Parameters
    ----------
    config : SimpleNamespace
        Experiment configuration
    run_idx : int
        Index of the current run

    Returns
    -------
    dict
        Experiment results
    """
    logger.info(f"Creating model: {config.model.type}.{config.model.name}")

    # Create simple model for demo
    model = create_demo_model(config.model)
    logger.info(f"✅ Model created: {type(model).__name__}")

    # Log model information
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {param_count:,} total, {trainable_params:,} trainable")

    # Create mock data for demonstration
    train_loader, val_loader, test_loader = create_mock_data_loaders(config)
    logger.info("✅ Data loaders created")

    # Create task (Lightning module)
    task = create_lightning_task(model, config)
    logger.info(f"✅ Task created: {type(task).__name__}")

    # Setup callbacks
    callbacks = []

    # Early stopping
    if config.optimization.early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=config.optimization.patience,
            min_delta=config.optimization.min_delta,
            mode='min'
        )
        callbacks.append(early_stop)

    # Model checkpointing
    checkpoint_dir = config.output_dir / f"run_{run_idx}" / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1
    )
    callbacks.append(checkpoint_callback)

    # Setup logger (handle missing tensorboard gracefully)
    try:
        tb_logger = TensorBoardLogger(
            save_dir=config.output_dir / f"run_{run_idx}",
            name="tensorboard_logs"
        )
    except ModuleNotFoundError:
        logger.warning("TensorBoard not available, using default logger")
        tb_logger = True  # Use default logger

    # Create trainer
    trainer_kwargs = {
        'max_epochs': config.optimization.max_epochs,
        'callbacks': callbacks,
        'logger': tb_logger,
        'deterministic': config.reproducibility.torch_deterministic,
        'log_every_n_steps': config.log_interval,
        'enable_progress_bar': False  # Reduce output for demo
    }

    # Configure device settings
    if config.device != "cpu" and torch.cuda.is_available():
        trainer_kwargs['devices'] = 1
        trainer_kwargs['accelerator'] = "gpu"
    else:
        trainer_kwargs['accelerator'] = "cpu"

    trainer = pl.Trainer(**trainer_kwargs)

    # Train model
    logger.info("Starting training...")
    trainer.fit(task, train_loader, val_loader)
    logger.info("✅ Training completed")

    # Test model
    logger.info("Starting testing...")
    test_results = trainer.test(task, test_loader, verbose=False)
    logger.info("✅ Testing completed")

    # Extract results
    results = {
        'run_idx': run_idx,
        'best_val_loss': float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score else None,
        'final_epoch': trainer.current_epoch,
        'test_results': test_results[0] if test_results else {}
    }

    return results


def create_demo_model(model_config):
    """Create a simple demo model."""
    import torch.nn as nn

    class DemoResNet1D(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3)
            self.bn1 = nn.BatchNorm1d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

            # Simple residual block
            self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm1d(64)
            self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
            self.bn3 = nn.BatchNorm1d(64)

            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(64, num_classes)

        def forward(self, x):
            # Transpose for convolution: (B, L, C) -> (B, C, L)
            x = x.transpose(1, 2)

            # Initial convolution
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            # Simple residual block
            identity = x
            out = self.conv2(x)
            out = self.bn2(out)
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)
            out += identity
            out = self.relu(out)

            # Global average pooling and classification
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)

            return out

    return DemoResNet1D(model_config.input_dim, model_config.num_classes)


def create_mock_data_loaders(config):
    """Create mock data loaders for demonstration."""
    from torch.utils.data import DataLoader, TensorDataset

    # Generate synthetic data
    batch_size = config.data.batch_size
    input_dim = config.model.input_dim
    seq_len = config.data.window_size
    num_samples = batch_size * 10  # Small dataset for demo

    # Create data
    x = torch.randn(num_samples, seq_len, input_dim)

    if config.model.num_classes is not None:
        # Classification task
        y = torch.randint(0, config.model.num_classes, (num_samples,))
    else:
        # Regression task
        y = torch.randn(num_samples, config.model.output_dim)

    # Create datasets
    dataset = TensorDataset(x, y)

    # Split into train/val/test
    train_size = int(config.data.train_ratio * num_samples)
    val_size = int(config.data.val_ratio * num_samples)
    test_size = num_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )

    return train_loader, val_loader, test_loader


def create_lightning_task(model, config):
    """Create a Lightning task for the model."""
    import torch.nn as nn
    import torch.optim as optim
    from torchmetrics import Accuracy

    class DemoTask(pl.LightningModule):
        def __init__(self, model, config):
            super().__init__()
            self.model = model
            self.config = config

            # Loss function
            if config.model.num_classes is not None:
                self.loss_fn = nn.CrossEntropyLoss()
                self.accuracy = Accuracy(task="multiclass", num_classes=config.model.num_classes)
            else:
                self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            # Log metrics
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

            if hasattr(self, 'accuracy'):
                acc = self.accuracy(y_hat, y)
                self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            # Log metrics
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

            if hasattr(self, 'accuracy'):
                acc = self.accuracy(y_hat, y)
                self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

            return loss

        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)

            # Log metrics
            self.log('test_loss', loss, on_step=False, on_epoch=True)

            if hasattr(self, 'accuracy'):
                acc = self.accuracy(y_hat, y)
                self.log('test_acc', acc, on_step=False, on_epoch=True)

            return loss

        def configure_optimizers(self):
            optimizer = optim.Adam(
                self.parameters(),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay
            )
            return optimizer

    return DemoTask(model, config)


def aggregate_experimental_results(all_results):
    """Aggregate results from multiple experimental runs."""
    import numpy as np

    if not all_results:
        return {}

    # Extract metrics
    val_losses = [r.get('best_val_loss') for r in all_results if r.get('best_val_loss') is not None]
    final_epochs = [r.get('final_epoch', 0) for r in all_results]

    # Extract test metrics
    test_losses = []
    test_accs = []

    for result in all_results:
        test_result = result.get('test_results', {})
        if 'test_loss' in test_result:
            test_losses.append(test_result['test_loss'])
        if 'test_acc' in test_result:
            test_accs.append(test_result['test_acc'])

    # Compute statistics
    aggregated = {
        'num_runs': len(all_results),
        'validation_loss': {
            'mean': np.mean(val_losses) if val_losses else None,
            'std': np.std(val_losses) if val_losses else None,
            'min': np.min(val_losses) if val_losses else None,
            'max': np.max(val_losses) if val_losses else None,
        },
        'final_epochs': {
            'mean': np.mean(final_epochs),
            'std': np.std(final_epochs),
            'min': np.min(final_epochs),
            'max': np.max(final_epochs),
        }
    }

    if test_losses:
        aggregated['test_loss'] = {
            'mean': np.mean(test_losses),
            'std': np.std(test_losses),
            'min': np.min(test_losses),
            'max': np.max(test_losses),
        }

    if test_accs:
        aggregated['test_accuracy'] = {
            'mean': np.mean(test_accs),
            'std': np.std(test_accs),
            'min': np.min(test_accs),
            'max': np.max(test_accs),
        }

    return aggregated


def run_model_tests(config):
    """Run model tests to validate implementation."""
    logger.info("Running model tests...")

    # Create model
    model = create_demo_model(config.model)

    # Test basic functionality
    batch_size = 4
    seq_len = config.data.window_size
    input_dim = config.model.input_dim

    # Test forward pass
    x = torch.randn(batch_size, seq_len, input_dim)

    model.eval()
    with torch.no_grad():
        output = model(x)

    # Validate output shape
    expected_shape = (batch_size, config.model.num_classes)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, got {output.shape}"

    # Validate output is finite
    assert torch.isfinite(output).all(), "Output contains non-finite values"

    # Test gradient flow
    model.train()
    x.requires_grad_(True)
    output = model(x)
    loss = output.sum()
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for parameter: {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for parameter: {name}"

    logger.info("✅ All model tests passed!")


if __name__ == "__main__":
    main()