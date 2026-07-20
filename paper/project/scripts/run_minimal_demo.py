#!/usr/bin/env python3
"""
Minimal 1D-2D Fusion Demo Script
Simple training and evaluation script for early fusion model
"""
import os
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Add code directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, '../code')
if code_dir not in sys.path:
    sys.path.append(code_dir)

from models import EarlyFusionModel
from utils import get_1d2d_dataloaders


class SimpleTrainer:
    """
    Simple trainer for 1D-2D fusion model
    """

    def __init__(self, model, train_loader, val_loader, test_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config

        # Setup optimizer and loss
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 1e-4)
        )
        self.criterion = nn.CrossEntropyLoss()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.output_root = Path(config['output_root'])
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.output_root / 'best_model.pth'

        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch_idx, (signals, labels) in enumerate(self.train_loader):
            signals = signals.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, _, _ = self.model(signals)
            loss = self.criterion(logits, labels)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Print progress
            if batch_idx % 10 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy

    def evaluate(self, loader):
        """Evaluate model on given dataloader"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for signals, labels in loader:
                signals = signals.to(self.device)
                labels = labels.to(self.device)

                logits, _, _ = self.model(signals)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')

        return avg_loss, accuracy, f1, all_labels, all_preds

    def train(self):
        """Main training loop"""
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

        best_val_acc = -1.0
        patience_counter = 0
        max_patience = self.config.get('patience', 5)

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")

            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            # Validate
            val_loss, val_acc, val_f1, _, _ = self.evaluate(self.val_loader)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.best_model_path)
                print("New best model saved!")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping after {epoch + 1} epochs")
                break

        print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.4f}")

        # Load best model for final evaluation
        self.model.load_state_dict(torch.load(self.best_model_path))

    def final_evaluation(self):
        """Final evaluation on test set"""
        print("\n=== Final Test Evaluation ===")
        test_loss, test_acc, test_f1, test_labels, test_preds = self.evaluate(self.test_loader)

        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro F1: {test_f1:.4f}")

        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(test_labels, test_preds, zero_division=0))

        return test_acc, test_f1

    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)

        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        figure_path = self.output_root / 'figures' / 'training_history.png'
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {figure_path}")
        # plt.show()  # Comment out for headless environments


def main():
    parser = argparse.ArgumentParser(description='Minimal 1D-2D Fusion Demo')
    parser.add_argument('--data_dir', type=str,
                       default='/home/user/data/PHMbenchdata/PHM-Vibench',
                       help='Path to data directory')
    parser.add_argument('--dataset_task', type=str, default='THU_018_basic',
                       help='Dataset task name')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--num_classes', type=int, default=10,
                       help='Number of classes')
    parser.add_argument('--input_dim', type=int, default=4096,
                       help='Input signal dimension')
    parser.add_argument('--use_dummy', action='store_true',
                       help='Use dummy dataset for testing')
    parser.add_argument('--output_root', type=str, default=None,
                       help='Directory for demo artifacts')
    parser.add_argument('--max_records', type=int, default=20,
                       help='Maximum PHM-Vibench records for paper-local smoke runs')
    parser.add_argument('--windows_per_record', type=int, default=2,
                       help='Number of deterministic windows per PHM-Vibench record')

    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent.parent
    output_root = Path(args.output_root) if args.output_root else (script_root / 'results')
    (output_root / 'figures').mkdir(parents=True, exist_ok=True)

    # Configuration
    config = {
        'data_dir': args.data_dir if not args.use_dummy else '/tmp/test_data',
        'dataset_task': args.dataset_task,
        'batch_size': args.batch_size,
        'num_workers': 0,
        'pin_memory': False,
        'learning_rate': args.learning_rate,
        'weight_decay': 1e-4,
        'num_epochs': args.num_epochs,
        'patience': 5,
        'num_classes': args.num_classes,
        'input_dim': args.input_dim,
        'output_root': str(output_root),
        'use_dummy': args.use_dummy,
        'max_records': args.max_records,
        'windows_per_record': args.windows_per_record,
        'seed': 0,
    }

    print("=== 1D-2D Fusion Minimal Demo ===")
    print(f"Configuration: {config}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_1d2d_dataloaders(config)
    print(f"Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches, {len(test_loader)} test batches")

    # Create model
    print("\nCreating model...")
    model = EarlyFusionModel(
        input_dim_1d=config['input_dim'],
        spectrogram_size=(128, 128),
        num_classes=config['num_classes'],
        hidden_dim=128
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer and train
    print("\nStarting training...")
    trainer = SimpleTrainer(model, train_loader, val_loader, test_loader, config)
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Final evaluation
    test_acc, test_f1 = trainer.final_evaluation()

    # Plot training history
    trainer.plot_training_history()

    # Save results
    results = {
        'test_accuracy': test_acc,
        'test_f1_macro': test_f1,
        'training_time': training_time,
        'config': config
    }

    import json
    results_path = output_root / 'demo_results.json'
    with results_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {results_path}")
    print("=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()
