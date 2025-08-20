"""
Basic Classification Example for PHM-Vibench Model Factory

This example demonstrates how to use different model architectures
for bearing fault classification using the CWRU dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from argparse import Namespace
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import PHM-Vibench model factory
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.model_factory import build_model


def generate_synthetic_data(num_samples=1000, seq_len=1024, num_classes=4):
    """Generate synthetic bearing vibration data for demonstration."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    data = []
    labels = []
    
    for class_id in range(num_classes):
        for _ in range(num_samples // num_classes):
            # Generate synthetic vibration signal
            t = np.linspace(0, 1, seq_len)
            
            # Base signal with different characteristics for each class
            if class_id == 0:  # Normal
                signal = np.sin(2 * np.pi * 50 * t) + 0.1 * np.random.randn(seq_len)
            elif class_id == 1:  # Inner race fault
                signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 162 * t) + 0.1 * np.random.randn(seq_len)
            elif class_id == 2:  # Outer race fault
                signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 107 * t) + 0.1 * np.random.randn(seq_len)
            else:  # Ball fault
                signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 142 * t) + 0.1 * np.random.randn(seq_len)
            
            # Create 3-channel data (simulating 3-axis accelerometer)
            vibration_data = np.stack([
                signal,
                signal + 0.1 * np.random.randn(seq_len),  # Y-axis with slight variation
                0.5 * signal + 0.1 * np.random.randn(seq_len)  # Z-axis with different amplitude
            ], axis=1)
            
            data.append(vibration_data)
            labels.append(class_id)
    
    return torch.FloatTensor(data), torch.LongTensor(labels)


def create_dataloaders(data, labels, batch_size=32, train_ratio=0.8):
    """Create train and test dataloaders."""
    dataset = TensorDataset(data, labels)
    
    # Split into train and test
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def train_model(model, train_loader, num_epochs=50, learning_rate=1e-3):
    """Train the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    train_losses = []
    train_accuracies = []
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, train_accuracies


def evaluate_model(model, test_loader):
    """Evaluate the model on test data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_targets)


def plot_results(train_losses, train_accuracies, predictions, targets):
    """Plot training results and confusion matrix."""
    class_names = ['Normal', 'Inner Race', 'Outer Race', 'Ball Fault']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(train_losses)
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True)
    
    # Training accuracy
    axes[0, 1].plot(train_accuracies)
    axes[0, 1].set_title('Training Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].grid(True)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # Classification report
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
    metrics_data = []
    for class_name in class_names:
        metrics_data.append([
            report[class_name]['precision'],
            report[class_name]['recall'],
            report[class_name]['f1-score']
        ])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    axes[1, 1].bar(x - width, [m[0] for m in metrics_data], width, label='Precision')
    axes[1, 1].bar(x, [m[1] for m in metrics_data], width, label='Recall')
    axes[1, 1].bar(x + width, [m[2] for m in metrics_data], width, label='F1-Score')
    
    axes[1, 1].set_xlabel('Classes')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('Classification Metrics')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names, rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function demonstrating classification with different models."""
    print("PHM-Vibench Classification Example")
    print("=" * 50)
    
    # Generate synthetic data
    print("Generating synthetic bearing vibration data...")
    data, labels = generate_synthetic_data(num_samples=1000, seq_len=1024, num_classes=4)
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(data, labels, batch_size=32)
    
    # Test different models
    models_to_test = [
        {
            'name': 'ResNetMLP',
            'args': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=256,
                num_layers=6,
                num_classes=4,
                dropout=0.1
            )
        },
        {
            'name': 'AttentionLSTM',
            'args': Namespace(
                model_name='AttentionLSTM',
                input_dim=3,
                hidden_dim=128,
                num_layers=2,
                num_classes=4,
                dropout=0.1,
                bidirectional=True
            )
        },
        {
            'name': 'ResNet1D',
            'args': Namespace(
                model_name='ResNet1D',
                input_dim=3,
                num_classes=4,
                block_type='basic',
                layers=[2, 2, 2, 2],
                dropout=0.1
            )
        }
    ]
    
    results = {}
    
    for model_config in models_to_test:
        print(f"\nTraining {model_config['name']}...")
        print("-" * 30)
        
        # Build model
        model = build_model(model_config['args'])
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        train_losses, train_accuracies = train_model(model, train_loader, num_epochs=30)
        
        # Evaluate model
        predictions, targets = evaluate_model(model, test_loader)
        accuracy = (predictions == targets).mean() * 100
        
        print(f"Test Accuracy: {accuracy:.2f}%")
        
        # Store results
        results[model_config['name']] = {
            'accuracy': accuracy,
            'train_losses': train_losses,
            'train_accuracies': train_accuracies,
            'predictions': predictions,
            'targets': targets
        }
        
        # Print classification report
        class_names = ['Normal', 'Inner Race', 'Outer Race', 'Ball Fault']
        print("\nClassification Report:")
        print(classification_report(targets, predictions, target_names=class_names))
    
    # Plot results for the best model
    best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
    print(f"\nBest model: {best_model} with {results[best_model]['accuracy']:.2f}% accuracy")
    
    plot_results(
        results[best_model]['train_losses'],
        results[best_model]['train_accuracies'],
        results[best_model]['predictions'],
        results[best_model]['targets']
    )
    
    # Summary
    print("\nModel Comparison:")
    print("-" * 40)
    for model_name, result in results.items():
        print(f"{model_name:15}: {result['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
