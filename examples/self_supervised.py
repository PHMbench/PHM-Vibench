"""
Self-Supervised Learning Example for PHM-Vibench Model Factory

This example demonstrates how to use foundation models for self-supervised
pre-training followed by downstream task fine-tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import seaborn as sns

# Import PHM-Vibench model factory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_factory import build_model


def generate_unlabeled_data(num_samples=5000, seq_len=128, num_features=3):
    """Generate large amount of unlabeled data for pre-training."""
    np.random.seed(42)
    
    data = []
    for i in range(num_samples):
        # Generate diverse signal patterns
        t = np.linspace(0, 1, seq_len)
        
        # Random frequency components
        freqs = np.random.uniform(10, 100, 5)
        amplitudes = np.random.uniform(0.1, 1.0, 5)
        phases = np.random.uniform(0, 2*np.pi, 5)
        
        signal = np.zeros(seq_len)
        for freq, amp, phase in zip(freqs, amplitudes, phases):
            signal += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add noise and trends
        signal += 0.1 * np.random.randn(seq_len)
        signal += np.random.uniform(-0.5, 0.5) * t  # Random trend
        
        # Create multi-channel data
        multi_channel = np.stack([
            signal,
            signal + 0.2 * np.random.randn(seq_len),
            0.8 * signal + 0.1 * np.random.randn(seq_len)
        ], axis=1)
        
        data.append(multi_channel)
    
    return torch.FloatTensor(data)


def generate_labeled_data(num_samples=500, seq_len=128, num_features=3, num_classes=4):
    """Generate smaller amount of labeled data for fine-tuning."""
    np.random.seed(123)
    
    data = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples // num_classes):
            t = np.linspace(0, 1, seq_len)
            
            # Class-specific patterns
            if class_id == 0:  # Normal operation
                signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(seq_len)
            elif class_id == 1:  # Fault type 1
                signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t) + 0.1 * np.random.randn(seq_len)
            elif class_id == 2:  # Fault type 2
                signal = np.sin(2 * np.pi * 50 * t) + 0.3 * np.sin(2 * np.pi * 80 * t) + 0.2 * np.random.randn(seq_len)
            else:  # Fault type 3
                signal = 0.7 * np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 150 * t) + 0.15 * np.random.randn(seq_len)
            
            # Create multi-channel data
            multi_channel = np.stack([
                signal,
                signal + 0.1 * np.random.randn(seq_len),
                0.9 * signal + 0.05 * np.random.randn(seq_len)
            ], axis=1)
            
            data.append(multi_channel)
            labels.append(class_id)
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def pretrain_contrastive_model(model, unlabeled_data, num_epochs=100, batch_size=64, learning_rate=1e-3):
    """Pre-train model using contrastive learning."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create dataloader for unlabeled data
    dataset = TensorDataset(unlabeled_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    pretrain_losses = []
    
    model.train()
    print("Starting contrastive pre-training...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass in contrastive mode
            output = model(data, mode='contrastive')
            loss = output['loss']
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        pretrain_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f'Pre-train Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
    
    return pretrain_losses


def finetune_downstream_task(model, labeled_data, labels, num_epochs=50, batch_size=32, learning_rate=1e-4):
    """Fine-tune pre-trained model on downstream classification task."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Split data into train and test
    dataset = TensorDataset(labeled_data, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Use smaller learning rate for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    finetune_losses = []
    finetune_accuracies = []
    
    model.train()
    print("Starting downstream task fine-tuning...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass in downstream mode
            output = model(data, mode='downstream')
            loss = criterion(output, target)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        finetune_losses.append(avg_loss)
        finetune_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Fine-tune Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    # Evaluate on test set
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data, mode='downstream')
            _, predicted = torch.max(output, 1)
            
            test_predictions.extend(predicted.cpu().numpy())
            test_targets.extend(target.cpu().numpy())
    
    test_accuracy = (np.array(test_predictions) == np.array(test_targets)).mean() * 100
    
    return finetune_losses, finetune_accuracies, test_predictions, test_targets, test_accuracy


def train_from_scratch_baseline(labeled_data, labels, num_epochs=100, batch_size=32, learning_rate=1e-3):
    """Train a model from scratch as baseline comparison."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create a simple baseline model
    baseline_args = Namespace(
        model_name='ResNetMLP',
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        num_classes=4,
        dropout=0.1
    )
    
    baseline_model = build_model(baseline_args)
    baseline_model.to(device)
    
    # Split data
    dataset = TensorDataset(labeled_data, labels)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    optimizer = optim.Adam(baseline_model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    baseline_model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = baseline_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    # Evaluate baseline
    baseline_model.eval()
    baseline_predictions = []
    baseline_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = baseline_model(data)
            _, predicted = torch.max(output, 1)
            
            baseline_predictions.extend(predicted.cpu().numpy())
            baseline_targets.extend(target.cpu().numpy())
    
    baseline_accuracy = (np.array(baseline_predictions) == np.array(baseline_targets)).mean() * 100
    
    return baseline_accuracy, baseline_predictions, baseline_targets


def plot_ssl_results(pretrain_losses, finetune_losses, finetune_accuracies, 
                    ssl_predictions, ssl_targets, baseline_predictions, baseline_targets):
    """Plot self-supervised learning results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Pre-training loss
    axes[0, 0].plot(pretrain_losses)
    axes[0, 0].set_title('Pre-training Contrastive Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Fine-tuning loss
    axes[0, 1].plot(finetune_losses)
    axes[0, 1].set_title('Fine-tuning Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].grid(True)
    
    # Fine-tuning accuracy
    axes[0, 2].plot(finetune_accuracies)
    axes[0, 2].set_title('Fine-tuning Accuracy')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].grid(True)
    
    # SSL confusion matrix
    from sklearn.metrics import confusion_matrix
    class_names = ['Normal', 'Fault 1', 'Fault 2', 'Fault 3']
    
    cm_ssl = confusion_matrix(ssl_targets, ssl_predictions)
    sns.heatmap(cm_ssl, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('SSL Model Confusion Matrix')
    
    # Baseline confusion matrix
    cm_baseline = confusion_matrix(baseline_targets, baseline_predictions)
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Reds', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 1])
    axes[1, 1].set_title('Baseline Model Confusion Matrix')
    
    # Accuracy comparison
    ssl_acc = (np.array(ssl_predictions) == np.array(ssl_targets)).mean() * 100
    baseline_acc = (np.array(baseline_predictions) == np.array(baseline_targets)).mean() * 100
    
    models = ['SSL Model', 'Baseline']
    accuracies = [ssl_acc, baseline_acc]
    colors = ['blue', 'red']
    
    bars = axes[1, 2].bar(models, accuracies, color=colors, alpha=0.7)
    axes[1, 2].set_title('Model Comparison')
    axes[1, 2].set_ylabel('Test Accuracy (%)')
    axes[1, 2].set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        axes[1, 2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{acc:.1f}%', ha='center', va='bottom')
    
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ssl_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating self-supervised learning."""
    print("PHM-Vibench Self-Supervised Learning Example")
    print("=" * 50)
    
    # Generate unlabeled data for pre-training
    print("Generating unlabeled data for pre-training...")
    unlabeled_data = generate_unlabeled_data(num_samples=2000, seq_len=128, num_features=3)
    print(f"Unlabeled data shape: {unlabeled_data.shape}")
    
    # Generate labeled data for fine-tuning
    print("Generating labeled data for fine-tuning...")
    labeled_data, labels = generate_labeled_data(num_samples=400, seq_len=128, num_features=3, num_classes=4)
    print(f"Labeled data shape: {labeled_data.shape}, Labels shape: {labels.shape}")
    
    # Configure contrastive SSL model
    ssl_args = Namespace(
        model_name='ContrastiveSSL',
        input_dim=3,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        projection_dim=64,
        temperature=0.1,
        dropout=0.1,
        num_classes=4
    )
    
    # Build SSL model
    ssl_model = build_model(ssl_args)
    print(f"SSL model parameters: {sum(p.numel() for p in ssl_model.parameters()):,}")
    
    # Phase 1: Self-supervised pre-training
    print("\n" + "="*50)
    print("Phase 1: Self-Supervised Pre-training")
    print("="*50)
    
    pretrain_losses = pretrain_contrastive_model(
        ssl_model, unlabeled_data, 
        num_epochs=80, batch_size=64, learning_rate=1e-3
    )
    
    # Phase 2: Downstream task fine-tuning
    print("\n" + "="*50)
    print("Phase 2: Downstream Task Fine-tuning")
    print("="*50)
    
    finetune_losses, finetune_accuracies, ssl_predictions, ssl_targets, ssl_test_accuracy = finetune_downstream_task(
        ssl_model, labeled_data, labels,
        num_epochs=50, batch_size=32, learning_rate=1e-4
    )
    
    print(f"SSL Model Test Accuracy: {ssl_test_accuracy:.2f}%")
    
    # Baseline: Train from scratch
    print("\n" + "="*50)
    print("Baseline: Training from Scratch")
    print("="*50)
    
    baseline_accuracy, baseline_predictions, baseline_targets = train_from_scratch_baseline(
        labeled_data, labels, num_epochs=100, batch_size=32, learning_rate=1e-3
    )
    
    print(f"Baseline Model Test Accuracy: {baseline_accuracy:.2f}%")
    
    # Results comparison
    print("\n" + "="*50)
    print("Results Comparison")
    print("="*50)
    print(f"SSL Model (Pre-train + Fine-tune): {ssl_test_accuracy:.2f}%")
    print(f"Baseline Model (From scratch):     {baseline_accuracy:.2f}%")
    print(f"Improvement: {ssl_test_accuracy - baseline_accuracy:.2f} percentage points")
    
    # Classification reports
    class_names = ['Normal', 'Fault 1', 'Fault 2', 'Fault 3']
    
    print("\nSSL Model Classification Report:")
    print(classification_report(ssl_targets, ssl_predictions, target_names=class_names))
    
    print("\nBaseline Model Classification Report:")
    print(classification_report(baseline_targets, baseline_predictions, target_names=class_names))
    
    # Plot results
    plot_ssl_results(
        pretrain_losses, finetune_losses, finetune_accuracies,
        ssl_predictions, ssl_targets, baseline_predictions, baseline_targets
    )


if __name__ == "__main__":
    main()
