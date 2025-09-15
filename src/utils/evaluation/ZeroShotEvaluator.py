"""
ZeroShotEvaluator for evaluating pretrained models on downstream tasks.

This evaluator implements linear probe evaluation on frozen pretrained backbones
to measure zero-shot transfer learning performance across different datasets.

Features:
- Linear probe evaluation on frozen pretrained backbones
- Per-dataset zero-shot performance measurement  
- Universal representation quality scoring
- Comparison with random baseline and dataset-specific training
- Comprehensive self-test with mock pretrained models
- Statistical significance testing
- Detailed performance analysis and reporting

Author: PHM-Vibench Team
Date: 2025-09-12
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import logging
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Warning: seaborn not available, some visualization features disabled")

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.utils.registry import Registry
    from torch.utils.data import DataLoader
except ImportError as e:
    print(f"Warning: Could not import PHM-Vibench components: {e}")


EVALUATOR_REGISTRY = Registry()

def register_evaluator(name: str):
    """Decorator to register an evaluator implementation."""
    return EVALUATOR_REGISTRY.register(name)


class LinearProbeClassifier(nn.Module):
    """
    Linear probe classifier for zero-shot evaluation.
    
    Simple linear layer on top of frozen pretrained features.
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout_rate: float = 0.1):
        """
        Initialize linear probe classifier.
        
        Args:
            input_dim: Dimension of input features
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through linear classifier."""
        return self.classifier(x)


class RepresentationQualityAnalyzer:
    """
    Analyzer for measuring representation quality of pretrained features.
    
    Provides various metrics to assess the quality of learned representations.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize representation quality analyzer."""
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_intrinsic_dimension(self, features: np.ndarray, variance_threshold: float = 0.95) -> int:
        """
        Compute intrinsic dimension using PCA.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            variance_threshold: Variance threshold for PCA
            
        Returns:
            Intrinsic dimension
        """
        pca = PCA()
        pca.fit(features)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        intrinsic_dim = np.argmax(cumsum_var >= variance_threshold) + 1
        
        return intrinsic_dim
    
    def compute_separability_score(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute class separability score using logistic regression.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            
        Returns:
            Separability score (0-1, higher is better)
        """
        try:
            # Use logistic regression for separability
            clf = LogisticRegression(random_state=42, max_iter=1000)
            clf.fit(features, labels)
            
            # Cross-validation score as separability measure
            from sklearn.model_selection import cross_val_score
            scores = cross_val_score(clf, features, labels, cv=5)
            separability = np.mean(scores)
            
            return separability
        except Exception as e:
            self.logger.warning(f"Failed to compute separability score: {e}")
            return 0.0
    
    def compute_cluster_quality(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Compute cluster quality metrics.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            
        Returns:
            Dictionary of cluster quality metrics
        """
        try:
            from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
            
            # Compute clustering metrics
            silhouette = silhouette_score(features, labels)
            calinski_harabasz = calinski_harabasz_score(features, labels)
            davies_bouldin = davies_bouldin_score(features, labels)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': calinski_harabasz,
                'davies_bouldin_score': davies_bouldin
            }
        except Exception as e:
            self.logger.warning(f"Failed to compute cluster quality: {e}")
            return {
                'silhouette_score': 0.0,
                'calinski_harabasz_score': 0.0,
                'davies_bouldin_score': float('inf')
            }
    
    def analyze_representation_quality(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive representation quality analysis.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            labels: Class labels (n_samples,)
            
        Returns:
            Dictionary of quality metrics
        """
        self.logger.info("Analyzing representation quality...")
        
        quality_metrics = {
            'feature_dimension': features.shape[1],
            'num_samples': features.shape[0],
            'num_classes': len(np.unique(labels))
        }
        
        # Intrinsic dimension
        try:
            intrinsic_dim = self.compute_intrinsic_dimension(features)
            quality_metrics['intrinsic_dimension'] = intrinsic_dim
            quality_metrics['compression_ratio'] = intrinsic_dim / features.shape[1]
        except Exception as e:
            self.logger.warning(f"Failed to compute intrinsic dimension: {e}")
            quality_metrics['intrinsic_dimension'] = features.shape[1]
            quality_metrics['compression_ratio'] = 1.0
        
        # Separability
        separability = self.compute_separability_score(features, labels)
        quality_metrics['separability_score'] = separability
        
        # Cluster quality
        cluster_quality = self.compute_cluster_quality(features, labels)
        quality_metrics.update(cluster_quality)
        
        # Feature statistics
        quality_metrics['feature_stats'] = {
            'mean_norm': np.mean(np.linalg.norm(features, axis=1)),
            'std_norm': np.std(np.linalg.norm(features, axis=1)),
            'mean_feature': np.mean(features),
            'std_feature': np.std(features)
        }
        
        # Overall quality score (0-1, higher is better)
        overall_quality = (
            separability * 0.4 +  # Separability (most important)
            min(1.0, quality_metrics['silhouette_score'] + 1.0) * 0.3 +  # Silhouette (normalized)
            (1.0 - quality_metrics['compression_ratio']) * 0.2 +  # Compression (lower is better)
            min(1.0, quality_metrics['calinski_harabasz_score'] / 100.0) * 0.1  # CH index
        )
        
        quality_metrics['overall_quality_score'] = max(0.0, min(1.0, overall_quality))
        
        return quality_metrics


@register_evaluator("zero_shot")
class ZeroShotEvaluator:
    """
    Zero-shot evaluator for pretrained models.
    
    Evaluates pretrained models using linear probe evaluation on frozen
    backbones to measure transfer learning performance.
    """
    
    def __init__(self,
                 output_dir: Optional[Union[str, Path]] = None,
                 device: Optional[torch.device] = None):
        """
        Initialize zero-shot evaluator.
        
        Args:
            output_dir: Directory for saving evaluation results
            device: Device to use for evaluation (auto-detect if None)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("evaluation_outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device or self._detect_device()
        
        # Initialize components
        self.representation_analyzer = RepresentationQualityAnalyzer()
        
        # Storage for evaluation results
        self.evaluation_results = {}
        self.baseline_results = {}
        
        # Setup logging
        self.setup_logging()
        
        self.logger.info("🎯 ZeroShotEvaluator initialized")
        self.logger.info(f"📱 Device: {self.device}")
        self.logger.info(f"📁 Output directory: {self.output_dir}")
    
    def _detect_device(self) -> torch.device:
        """Auto-detect the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def setup_logging(self):
        """Setup comprehensive logging for evaluation."""
        log_file = self.output_dir / f"zero_shot_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Setup logger
        self.logger = logging.getLogger(f"{__name__}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Prevent duplicate logs
        self.logger.propagate = False
    
    def extract_features(self,
                        model: nn.Module,
                        data_loader: DataLoader,
                        feature_extractor_fn: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features from pretrained model.
        
        Args:
            model: Pretrained model (will be set to eval mode)
            data_loader: Data loader for feature extraction
            feature_extractor_fn: Optional function to extract features from model output
            
        Returns:
            Tuple of (features, labels) as numpy arrays
        """
        self.logger.info("Extracting features from pretrained model...")
        
        model.eval()
        model = model.to(self.device)
        
        all_features = []
        all_labels = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                # Handle different batch formats
                if isinstance(batch, dict):
                    inputs = batch['data'].to(self.device)
                    labels = batch['label'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    labels = batch[1].to(self.device)
                else:
                    raise ValueError(f"Unsupported batch format: {type(batch)}")
                
                # Forward pass
                outputs = model(inputs)
                
                # Extract features
                if feature_extractor_fn is not None:
                    features = feature_extractor_fn(outputs)
                else:
                    # Use the output as features (assume it's already features)
                    features = outputs
                
                # Handle different output formats
                if isinstance(features, dict):
                    # Assume 'features' key contains the actual features
                    features = features.get('features', features.get('embeddings', outputs))
                
                # Ensure features are 2D
                if features.dim() > 2:
                    features = features.view(features.size(0), -1)  # Flatten
                
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                
                if batch_idx % 100 == 0:
                    self.logger.info(f"Processed {batch_idx + 1} batches...")
        
        # Concatenate all features and labels
        features = np.concatenate(all_features, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        self.logger.info(f"Extracted {features.shape[0]} features with dimension {features.shape[1]}")
        
        return features, labels
    
    def train_linear_probe(self,
                          train_features: np.ndarray,
                          train_labels: np.ndarray,
                          num_classes: int,
                          max_epochs: int = 100,
                          learning_rate: float = 0.01,
                          batch_size: int = 256) -> nn.Module:
        """
        Train linear probe classifier on extracted features.
        
        Args:
            train_features: Training features (n_samples, n_features)
            train_labels: Training labels (n_samples,)
            num_classes: Number of classes
            max_epochs: Maximum training epochs
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            
        Returns:
            Trained linear probe classifier
        """
        self.logger.info(f"Training linear probe classifier...")
        self.logger.info(f"Features: {train_features.shape}, Classes: {num_classes}")
        
        # Create classifier
        input_dim = train_features.shape[1]
        classifier = LinearProbeClassifier(input_dim, num_classes)
        classifier = classifier.to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(train_features).float().to(self.device)
        labels_tensor = torch.from_numpy(train_labels).long().to(self.device)
        
        # Training loop
        classifier.train()
        best_accuracy = 0.0
        
        for epoch in range(max_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            # Mini-batch training
            num_batches = (len(features_tensor) + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(features_tensor))
                
                batch_features = features_tensor[start_idx:end_idx]
                batch_labels = labels_tensor[start_idx:end_idx]
                
                # Forward pass
                optimizer.zero_grad()
                outputs = classifier(batch_features)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
            
            accuracy = correct / total
            avg_loss = total_loss / num_batches
            
            # Track best accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
            
            if epoch % 20 == 0 or epoch == max_epochs - 1:
                self.logger.info(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}, Acc={accuracy:.4f}")
        
        self.logger.info(f"Linear probe training completed. Best accuracy: {best_accuracy:.4f}")
        
        return classifier
    
    def evaluate_linear_probe(self,
                             classifier: nn.Module,
                             test_features: np.ndarray,
                             test_labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate trained linear probe classifier.
        
        Args:
            classifier: Trained classifier
            test_features: Test features (n_samples, n_features)
            test_labels: Test labels (n_samples,)
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating linear probe classifier...")
        
        classifier.eval()
        classifier = classifier.to(self.device)
        
        # Convert to tensors
        features_tensor = torch.from_numpy(test_features).float().to(self.device)
        labels_tensor = torch.from_numpy(test_labels).long().to(self.device)
        
        with torch.no_grad():
            outputs = classifier(features_tensor)
            _, predicted = torch.max(outputs, 1)
            
            predicted_np = predicted.cpu().numpy()
            labels_np = labels_tensor.cpu().numpy()
        
        # Compute metrics
        accuracy = accuracy_score(labels_np, predicted_np)
        f1 = f1_score(labels_np, predicted_np, average='weighted')
        precision = precision_score(labels_np, predicted_np, average='weighted')
        recall = recall_score(labels_np, predicted_np, average='weighted')
        
        # Per-class metrics
        unique_labels = np.unique(labels_np)
        per_class_accuracy = {}
        for label in unique_labels:
            mask = labels_np == label
            if mask.sum() > 0:
                per_class_accuracy[int(label)] = accuracy_score(labels_np[mask], predicted_np[mask])
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'per_class_accuracy': per_class_accuracy
        }
        
        self.logger.info(f"Evaluation results: Acc={accuracy:.4f}, F1={f1:.4f}")
        
        return metrics
    
    def compute_random_baseline(self, test_labels: np.ndarray) -> Dict[str, float]:
        """
        Compute random baseline performance.
        
        Args:
            test_labels: Test labels for baseline computation
            
        Returns:
            Dictionary of random baseline metrics
        """
        unique_labels = np.unique(test_labels)
        num_classes = len(unique_labels)
        
        # Generate random predictions
        random_predictions = np.random.choice(unique_labels, size=len(test_labels))
        
        # Compute metrics
        accuracy = accuracy_score(test_labels, random_predictions)
        f1 = f1_score(test_labels, random_predictions, average='weighted')
        precision = precision_score(test_labels, random_predictions, average='weighted', zero_division=0)
        recall = recall_score(test_labels, random_predictions, average='weighted', zero_division=0)
        
        # Theoretical random accuracy
        theoretical_accuracy = 1.0 / num_classes
        
        baseline_metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'theoretical_accuracy': theoretical_accuracy,
            'num_classes': num_classes
        }
        
        self.logger.info(f"Random baseline: Acc={accuracy:.4f} (theoretical: {theoretical_accuracy:.4f})")
        
        return baseline_metrics
    
    def evaluate_dataset(self,
                        model: nn.Module,
                        train_loader: DataLoader,
                        test_loader: DataLoader,
                        dataset_name: str,
                        num_classes: int,
                        feature_extractor_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate model on a single dataset with zero-shot linear probe.
        
        Args:
            model: Pretrained model
            train_loader: Training data loader (for linear probe training)
            test_loader: Test data loader (for evaluation)
            dataset_name: Name of the dataset
            num_classes: Number of classes in the dataset
            feature_extractor_fn: Optional function to extract features
            
        Returns:
            Dictionary of evaluation results
        """
        self.logger.info(f"🎯 Evaluating dataset: {dataset_name}")
        
        results = {
            'dataset_name': dataset_name,
            'num_classes': num_classes,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Extract features from training set
            train_features, train_labels = self.extract_features(model, train_loader, feature_extractor_fn)
            
            # Extract features from test set
            test_features, test_labels = self.extract_features(model, test_loader, feature_extractor_fn)
            
            results['train_samples'] = len(train_features)
            results['test_samples'] = len(test_features)
            results['feature_dimension'] = train_features.shape[1]
            
            # Representation quality analysis
            self.logger.info("Analyzing representation quality...")
            quality_metrics = self.representation_analyzer.analyze_representation_quality(
                train_features, train_labels
            )
            results['representation_quality'] = quality_metrics
            
            # Train linear probe
            classifier = self.train_linear_probe(
                train_features, train_labels, num_classes
            )
            
            # Evaluate linear probe
            linear_probe_results = self.evaluate_linear_probe(
                classifier, test_features, test_labels
            )
            results['linear_probe'] = linear_probe_results
            
            # Compute random baseline
            random_baseline = self.compute_random_baseline(test_labels)
            results['random_baseline'] = random_baseline
            
            # Compute improvement over random
            improvement_over_random = {
                'accuracy_improvement': linear_probe_results['accuracy'] - random_baseline['accuracy'],
                'f1_improvement': linear_probe_results['f1_score'] - random_baseline['f1_score'],
                'relative_improvement': (linear_probe_results['accuracy'] - random_baseline['accuracy']) / random_baseline['accuracy'] if random_baseline['accuracy'] > 0 else 0.0
            }
            results['improvement_over_random'] = improvement_over_random
            
            # Statistical significance test
            if len(test_labels) > 30:  # Minimum sample size for t-test
                try:
                    # Simple significance test comparing accuracies
                    observed_accuracy = linear_probe_results['accuracy']
                    random_accuracy = random_baseline['theoretical_accuracy']
                    
                    # Approximate standard error for accuracy
                    n = len(test_labels)
                    se = np.sqrt(observed_accuracy * (1 - observed_accuracy) / n)
                    
                    # Z-test for proportion
                    z_score = (observed_accuracy - random_accuracy) / se if se > 0 else 0
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed test
                    
                    results['statistical_significance'] = {
                        'z_score': z_score,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                    
                except Exception as e:
                    self.logger.warning(f"Failed to compute statistical significance: {e}")
                    results['statistical_significance'] = None
            
            # Overall performance score (0-1, higher is better)
            overall_score = (
                linear_probe_results['accuracy'] * 0.4 +
                linear_probe_results['f1_score'] * 0.3 +
                quality_metrics['overall_quality_score'] * 0.2 +
                min(1.0, improvement_over_random['relative_improvement']) * 0.1
            )
            results['overall_performance_score'] = max(0.0, min(1.0, overall_score))
            
            self.logger.info(f"✅ Dataset {dataset_name} evaluation completed")
            self.logger.info(f"   Accuracy: {linear_probe_results['accuracy']:.4f}")
            self.logger.info(f"   F1-Score: {linear_probe_results['f1_score']:.4f}")
            self.logger.info(f"   Overall Score: {overall_score:.4f}")
            
        except Exception as e:
            self.logger.error(f"❌ Dataset {dataset_name} evaluation failed: {e}")
            results['error'] = str(e)
            results['success'] = False
            return results
        
        results['success'] = True
        return results
    
    def evaluate_multiple_datasets(self,
                                  model: nn.Module,
                                  dataset_loaders: Dict[str, Dict[str, DataLoader]],
                                  num_classes_per_dataset: Dict[str, int],
                                  feature_extractor_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Evaluate model on multiple datasets.
        
        Args:
            model: Pretrained model
            dataset_loaders: Dict mapping dataset names to {'train': loader, 'test': loader}
            num_classes_per_dataset: Dict mapping dataset names to number of classes
            feature_extractor_fn: Optional function to extract features
            
        Returns:
            Dictionary of all evaluation results
        """
        self.logger.info(f"🎯 Evaluating model on {len(dataset_loaders)} datasets...")
        
        all_results = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'total_datasets': len(dataset_loaders),
            'datasets': {},
            'summary': {}
        }
        
        dataset_results = []
        
        for dataset_name, loaders in dataset_loaders.items():
            num_classes = num_classes_per_dataset.get(dataset_name, 10)  # Default to 10
            
            result = self.evaluate_dataset(
                model,
                loaders['train'],
                loaders['test'],
                dataset_name,
                num_classes,
                feature_extractor_fn
            )
            
            all_results['datasets'][dataset_name] = result
            
            if result.get('success', False):
                dataset_results.append(result)
        
        # Compute summary statistics
        if dataset_results:
            accuracies = [r['linear_probe']['accuracy'] for r in dataset_results]
            f1_scores = [r['linear_probe']['f1_score'] for r in dataset_results]
            quality_scores = [r['representation_quality']['overall_quality_score'] for r in dataset_results]
            overall_scores = [r['overall_performance_score'] for r in dataset_results]
            
            summary = {
                'successful_evaluations': len(dataset_results),
                'failed_evaluations': len(dataset_loaders) - len(dataset_results),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': np.min(accuracies),
                'max_accuracy': np.max(accuracies),
                'mean_f1_score': np.mean(f1_scores),
                'std_f1_score': np.std(f1_scores),
                'mean_quality_score': np.mean(quality_scores),
                'mean_overall_score': np.mean(overall_scores),
                'dataset_ranking': sorted(
                    [(r['dataset_name'], r['overall_performance_score']) for r in dataset_results],
                    key=lambda x: x[1],
                    reverse=True
                )
            }
            
            all_results['summary'] = summary
            
            self.logger.info(f"🏁 Multi-dataset evaluation completed:")
            self.logger.info(f"   Mean Accuracy: {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
            self.logger.info(f"   Mean F1-Score: {summary['mean_f1_score']:.4f} ± {summary['std_f1_score']:.4f}")
            self.logger.info(f"   Mean Quality Score: {summary['mean_quality_score']:.4f}")
        
        # Save results
        self.save_evaluation_results(all_results)
        
        return all_results
    
    def save_evaluation_results(self, results: Dict[str, Any]):
        """Save evaluation results to JSON file."""
        results_file = self.output_dir / f"zero_shot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"📊 Results saved to: {results_file}")
    
    def create_mock_pretrained_model(self, input_dim: int = 1024, feature_dim: int = 256) -> nn.Module:
        """
        Create a mock pretrained model for testing.
        
        Args:
            input_dim: Input dimension
            feature_dim: Feature dimension
            
        Returns:
            Mock pretrained model
        """
        class MockPretrainedModel(nn.Module):
            def __init__(self, input_dim, feature_dim):
                super().__init__()
                self.feature_extractor = nn.Sequential(
                    nn.Linear(input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, feature_dim)
                )
                
                # Initialize with reasonable weights
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)
            
            def forward(self, x):
                # Flatten input if needed
                if x.dim() > 2:
                    x = x.view(x.size(0), -1)
                
                features = self.feature_extractor(x)
                return features
        
        return MockPretrainedModel(input_dim, feature_dim)
    
    def run_self_test(self) -> bool:
        """
        Run comprehensive self-test with mock pretrained models.
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("🧪 Running ZeroShotEvaluator self-test...")
        
        test_results = []
        
        try:
            # Test 1: Mock model creation
            self.logger.info("Test 1: Mock model creation")
            mock_model = self.create_mock_pretrained_model(input_dim=512, feature_dim=128)
            if mock_model is not None:
                test_results.append(("Mock model", True, "Mock pretrained model created"))
            else:
                test_results.append(("Mock model", False, "Failed to create mock model"))
            
            # Test 2: Feature extraction
            self.logger.info("Test 2: Feature extraction")
            try:
                # Create mock data loader
                from torch.utils.data import TensorDataset, DataLoader
                
                mock_data = torch.randn(100, 512)  # 100 samples, 512 features
                mock_labels = torch.randint(0, 5, (100,))  # 5 classes
                mock_dataset = TensorDataset(mock_data, mock_labels)
                mock_loader = DataLoader(mock_dataset, batch_size=16, shuffle=False)
                
                features, labels = self.extract_features(mock_model, mock_loader)
                
                if features.shape == (100, 128) and labels.shape == (100,):
                    test_results.append(("Feature extraction", True, f"Features shape: {features.shape}"))
                else:
                    test_results.append(("Feature extraction", False, f"Unexpected shapes: {features.shape}, {labels.shape}"))
                    
            except Exception as e:
                test_results.append(("Feature extraction", False, f"Feature extraction failed: {e}"))
            
            # Test 3: Linear probe training
            self.logger.info("Test 3: Linear probe training")
            try:
                classifier = self.train_linear_probe(features, labels, num_classes=5, max_epochs=10)
                if classifier is not None:
                    test_results.append(("Linear probe training", True, "Linear probe trained successfully"))
                else:
                    test_results.append(("Linear probe training", False, "Failed to train linear probe"))
            except Exception as e:
                test_results.append(("Linear probe training", False, f"Training failed: {e}"))
            
            # Test 4: Linear probe evaluation
            self.logger.info("Test 4: Linear probe evaluation")
            try:
                eval_results = self.evaluate_linear_probe(classifier, features, labels)
                required_keys = ['accuracy', 'f1_score', 'precision', 'recall']
                if all(key in eval_results for key in required_keys):
                    test_results.append(("Linear probe evaluation", True, f"Accuracy: {eval_results['accuracy']:.4f}"))
                else:
                    test_results.append(("Linear probe evaluation", False, "Missing evaluation metrics"))
            except Exception as e:
                test_results.append(("Linear probe evaluation", False, f"Evaluation failed: {e}"))
            
            # Test 5: Random baseline computation
            self.logger.info("Test 5: Random baseline")
            try:
                baseline = self.compute_random_baseline(labels)
                if 'accuracy' in baseline and 'theoretical_accuracy' in baseline:
                    test_results.append(("Random baseline", True, f"Baseline accuracy: {baseline['accuracy']:.4f}"))
                else:
                    test_results.append(("Random baseline", False, "Missing baseline metrics"))
            except Exception as e:
                test_results.append(("Random baseline", False, f"Baseline computation failed: {e}"))
            
            # Test 6: Representation quality analysis
            self.logger.info("Test 6: Representation quality analysis")
            try:
                quality_metrics = self.representation_analyzer.analyze_representation_quality(features, labels)
                if 'overall_quality_score' in quality_metrics:
                    test_results.append(("Quality analysis", True, f"Quality score: {quality_metrics['overall_quality_score']:.4f}"))
                else:
                    test_results.append(("Quality analysis", False, "Missing quality metrics"))
            except Exception as e:
                test_results.append(("Quality analysis", False, f"Quality analysis failed: {e}"))
            
            # Test 7: Full dataset evaluation
            self.logger.info("Test 7: Full dataset evaluation")
            try:
                # Create train/test split
                train_data = mock_data[:70]
                train_labels_data = mock_labels[:70]
                test_data = mock_data[70:]
                test_labels_data = mock_labels[70:]
                
                train_dataset = TensorDataset(train_data, train_labels_data)
                test_dataset = TensorDataset(test_data, test_labels_data)
                
                train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
                
                dataset_result = self.evaluate_dataset(
                    mock_model, train_loader, test_loader, "MockDataset", 5
                )
                
                if dataset_result.get('success', False):
                    test_results.append(("Dataset evaluation", True, "Full evaluation completed"))
                else:
                    error_msg = dataset_result.get('error', 'Unknown error')
                    test_results.append(("Dataset evaluation", False, f"Evaluation failed: {error_msg}"))
                    
            except Exception as e:
                test_results.append(("Dataset evaluation", False, f"Dataset evaluation failed: {e}"))
            
        except Exception as e:
            test_results.append(("Self-test", False, f"Unexpected error: {e}"))
        
        # Print results
        self.logger.info("🧪 Self-test results:")
        all_passed = True
        for test_name, passed, message in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        self.logger.info(f"🧪 Overall result: {overall_status}")
        
        return all_passed


def main():
    """Main entry point for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ZeroShotEvaluator - Zero-shot evaluation for pretrained models")
    parser.add_argument("--output_dir", type=str, default="evaluation_outputs",
                       help="Output directory for evaluation results")
    parser.add_argument("--device", type=str, choices=['cpu', 'cuda', 'auto'], default='auto',
                       help="Device to use for evaluation")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = None  # Auto-detect
    else:
        device = torch.device(args.device)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create evaluator
    print("🎯 Creating ZeroShotEvaluator...")
    evaluator = ZeroShotEvaluator(
        output_dir=args.output_dir,
        device=device
    )
    
    # Run self-test
    success = evaluator.run_self_test()
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())