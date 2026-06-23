"""
Base class for explainable fault diagnosis models in UXFD
"""

import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


class BaseExplainableModel(ABC):
    """
    Abstract base class for explainable fault diagnosis models
    """

    def __init__(self, config: Dict):
        """
        Initialize the base model

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.is_trained = False

    @abstractmethod
    def fit(self, train_loader, val_loader=None, epochs=100):
        """
        Train the model

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        pass

    @abstractmethod
    def predict(self, data) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions

        Args:
            data: Input data tensor

        Returns:
            Tuple of (predictions, probabilities)
        """
        pass

    @abstractmethod
    def explain(self, data, target_class: Optional[int] = None) -> List[Dict]:
        """
        Generate explanations for predictions

        Args:
            data: Input data tensor
            target_class: Target class for explanation

        Returns:
            List of explanations
        """
        pass

    @abstractmethod
    def evaluate(self, data_loader) -> float:
        """
        Evaluate model performance

        Args:
            data_loader: Test data loader

        Returns:
            Accuracy score
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """Save model checkpoint"""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Load model checkpoint"""
        pass

    def get_model_info(self) -> Dict:
        """
        Get model information

        Returns:
            Dictionary containing model details
        """
        if self.model:
            num_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        else:
            num_params = 0
            trainable_params = 0

        return {
            'model_type': self.__class__.__name__,
            'total_parameters': num_params,
            'trainable_parameters': trainable_params,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'config': self.config
        }

    def visualize_explanation(self, explanation: Dict, save_path: Optional[str] = None):
        """
        Visualize model explanation

        Args:
            explanation: Explanation dictionary
            save_path: Path to save visualization
        """
        plt.figure(figsize=(15, 10))

        # Create subplots based on explanation type
        if 'cam' in explanation:
            # Grad-CAM visualization
            plt.subplot(3, 1, 1)
            cam = explanation['cam']
            plt.plot(cam)
            plt.title('Grad-CAM Heatmap')
            plt.xlabel('Time Steps')
            plt.ylabel('Attention Weight')
            plt.colorbar(label='Activation Intensity')

        if 'sensor_importance' in explanation:
            # Sensor importance
            plt.subplot(3, 1, 2)
            importance = explanation['sensor_importance']
            sensor_names = explanation.get('sensor_names', [f'Sensor {i}' for i in range(len(importance))])
            plt.bar(sensor_names, importance)
            plt.title('Sensor Importance')
            plt.xlabel('Sensors')
            plt.ylabel('Importance Weight')
            plt.xticks(rotation=45)

        if 'feature_importance' in explanation:
            # Feature importance
            plt.subplot(3, 1, 3)
            feat_importance = explanation['feature_importance']
            feat_names = explanation.get('feature_names', [f'Feature {i}' for i in range(len(feat_importance))])
            plt.bar(feat_names, feat_importance)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Importance Weight')
            plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Explanation visualization saved to {save_path}")

        plt.show()

    def generate_explanation_report(self, explanations: List[Dict]) -> str:
        """
        Generate a textual explanation report

        Args:
            explanations: List of explanation dictionaries

        Returns:
            Formatted explanation report
        """
        report = "Model Explanation Report\n"
        report += "=" * 50 + "\n\n"

        for i, exp in enumerate(explanations):
            report += f"Sample {i+1}:\n"
            report += f"  Predicted Class: {exp.get('prediction', 'N/A')}\n"
            report += f"  Confidence: {exp.get('confidence', 0):.4f}\n"

            if 'prediction_uncertainty' in exp:
                report += f"  Prediction Uncertainty: {exp.get('prediction_uncertainty', 0):.4f}\n"

            if 'interpretation' in exp:
                report += f"  Interpretation: {exp['interpretation']}\n"

            if 'top_influential_sensors' in exp:
                top_sensors = exp['top_influential_sensors']
                report += f"  Top Influential Sensors: {top_sensors[:5]}\n"

            if 'strong_causal_paths' in exp:
                causal_paths = exp['strong_causal_paths']
                report += f"  Strong Causal Paths: {len(causal_paths)} detected\n"

            report += "\n"

        return report

    def evaluate_explainability(self, test_loader, num_samples: int = 100) -> Dict:
        """
        Evaluate model explainability

        Args:
            test_loader: Test data loader
            num_samples: Number of samples to evaluate

        Returns:
            Explainability metrics
        """
        self.model.eval()
        explanations = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            for i, (data, target) in enumerate(test_loader):
                if i >= num_samples:
                    break

                # Get explanation
                sample_explanations = self.explain(data)
                explanations.extend(sample_explanations)

                # Get predictions
                pred, prob = self.predict(data)
                predictions.extend(pred)
                true_labels.extend(target.numpy())

        # Calculate explainability metrics
        metrics = {
            'num_samples': len(explanations),
            'avg_confidence': np.mean([exp.get('confidence', 0) for exp in explanations]),
            'confidence_std': np.std([exp.get('confidence', 0) for exp in explanations])
        }

        # Add uncertainty metrics if available
        if 'prediction_uncertainty' in explanations[0]:
            uncertainties = [exp.get('prediction_uncertainty', 0) for exp in explanations]
            metrics.update({
                'avg_uncertainty': np.mean(uncertainties),
                'uncertainty_std': np.std(uncertainties)
            })

        # Classification metrics
        metrics.update({
            'accuracy': np.mean(np.array(predictions) == np.array(true_labels)),
            'classification_report': classification_report(true_labels, predictions, output_dict=True)
        })

        return metrics

    def compare_models(self, other_models: List['BaseExplainableModel'], test_loader) -> Dict:
        """
        Compare with other explainable models

        Args:
            other_models: List of other models to compare
            test_loader: Test data loader

        Returns:
            Comparison results
        """
        results = {
            'models': [self.__class__.__name__] + [m.__class__.__name__ for m in other_models],
            'accuracy': [],
            'num_parameters': [],
            'explainability_metrics': []
        }

        # Evaluate self
        self_acc = self.evaluate(test_loader)
        self_metrics = self.get_model_info()
        self_explain = self.evaluate_explainability(test_loader)

        results['accuracy'].append(self_acc)
        results['num_parameters'].append(self_metrics['total_parameters'])
        results['explainability_metrics'].append(self_explain)

        # Evaluate other models
        for model in other_models:
            acc = model.evaluate(test_loader)
            metrics = model.get_model_info()
            explain = model.evaluate_explainability(test_loader)

            results['accuracy'].append(acc)
            results['num_parameters'].append(metrics['total_parameters'])
            results['explainability_metrics'].append(explain)

        # Create comparison summary
        results['summary'] = {
            'best_accuracy': results['models'][np.argmax(results['accuracy'])],
            'least_params': results['models'][np.argmin(results['num_parameters'])],
            'most_confident': results['models'][np.argmax([m['avg_confidence'] for m in results['explainability_metrics']])]
        }

        return results

    def export_explanations(self, explanations: List[Dict], format: str = 'json', save_path: str = None):
        """
        Export explanations to file

        Args:
            explanations: List of explanations
            format: Export format ('json', 'csv', 'txt')
            save_path: Path to save file
        """
        import json
        import pandas as pd

        if format == 'json':
            with open(save_path or 'explanations.json', 'w') as f:
                json.dump(explanations, f, indent=2, default=str)

        elif format == 'csv':
            # Flatten explanations for CSV
            flattened = []
            for exp in explanations:
                flat_exp = {}
                for key, value in exp.items():
                    if isinstance(value, (list, np.ndarray)):
                        flat_exp[key] = str(value)
                    else:
                        flat_exp[key] = value
                flattened.append(flat_exp)

            df = pd.DataFrame(flattened)
            df.to_csv(save_path or 'explanations.csv', index=False)

        elif format == 'txt':
            report = self.generate_explanation_report(explanations)
            with open(save_path or 'explanations.txt', 'w') as f:
                f.write(report)

        logger.info(f"Explanations exported to {save_path or f'explanations.{format}'}")


def register_explainable_model(name: str, model_class: BaseExplainableModel):
    """
    Decorator to register new explainable models

    Args:
        name: Model name
        model_class: Model class
    """
    # This can be used to create a registry of explainable models
    if not hasattr(register_explainable_model, 'registry'):
        register_explainable_model.registry = {}

    register_explainable_model.registry[name] = model_class
    return model_class