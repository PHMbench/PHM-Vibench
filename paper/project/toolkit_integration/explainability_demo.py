"""
Explainability Toolkit Demo

Demonstrates basic usage of the explainability toolkit for fault diagnosis.
This example shows how to create explanations using both intrinsic and post-hoc methods.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainability import UnifiedExplainer
from model.TSPN_explainable import Transparent_Signal_Processing_Network_Explainable, create_explainable_tspn


def create_demo_model():
    """Create a demo TSPN model for testing."""
    # Demo configuration
    class DemoArgs:
        def __init__(self):
            self.in_channels = 2
            self.out_channels = 64
            self.scale = 4
            self.skip_connection = True
            self.num_classes = 5
            self.device = 'cpu'

    args = DemoArgs()

    # Create demo signal processing modules
    signal_processing_modules = []
    for i in range(2):  # 2 layers
        layer_modules = {}
        layer_modules['FFT'] = torch.nn.Identity()  # Simplified for demo
        layer_modules['I'] = torch.nn.Identity()
        signal_processing_modules.append(layer_modules)

    # Create demo feature extractor modules
    feature_extractor_modules = {}
    feature_types = ['Mean', 'Std', 'Max', 'Min', 'RMS']
    for feat_type in feature_types:
        feature_extractor_modules[feat_type] = torch.nn.Identity()  # Simplified for demo

    # Create model
    model = create_explainable_tspn(signal_processing_modules, feature_extractor_modules, args)
    model.eval()

    return model


def create_demo_signal(length=1000, noise_level=0.1):
    """Create a demo vibration signal."""
    # Create a synthetic vibration signal with some frequency components
    t = np.linspace(0, 1, length)

    # Add multiple frequency components
    signal = (
        1.0 * np.sin(2 * np.pi * 10 * t) +  # 10 Hz component
        0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz component
        0.3 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz component
        noise_level * np.random.randn(length)  # Add noise
    )

    # Convert to tensor and add batch and channel dimensions
    signal_tensor = torch.FloatTensor(signal).unsqueeze(0).unsqueeze(-1)
    # Duplicate for 2 channels
    signal_tensor = signal_tensor.repeat(1, 1, 2)

    return signal_tensor


def demo_signal_path_explanation():
    """Demonstrate signal path explanation."""
    print("=== Signal Path Explanation Demo ===")

    # Create model and data
    model = create_demo_model()
    signal_data = create_demo_signal()

    # Create explainer
    explainer = UnifiedExplainer(model, method='signal_path')

    # Generate explanation
    explanation = explainer.explain(signal_data)

    print(f"Method: {explanation.get_method_name()}")
    print(f"Model: {explanation.get_model_name()}")
    print(f"Data shape: {explanation.get_meta('input_shape')}")

    # Get signal path information
    signal_path = explanation.get_data('signal_path')
    if signal_path:
        print(f"Number of transformations: {len(signal_path)}")
        for i, step in enumerate(signal_path):
            print(f"  Step {i}: {step['layer_name']} ({step['operator_type']})")
            if 'input_stats' in step:
                input_energy = step['input_stats'].get('energy', 0)
                output_energy = step['output_stats'].get('energy', 0)
                print(f"    Energy change: {input_energy:.4f} -> {output_energy:.4f}")

    # Get transformation summary
    summary = explanation.get_data('transformation_summary')
    if summary:
        print(f"Overall energy change: {summary.get('overall_energy_change', 0):.4f}")
        print(f"Number of layers: {summary.get('total_layers', 0)}")

    # Visualize
    try:
        fig = explanation.visualize(mode='path')
        plt.title('Signal Path Explanation')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")

    return explanation


def demo_integrated_gradients_explanation():
    """Demonstrate Integrated Gradients explanation."""
    print("\n=== Integrated Gradients Explanation Demo ===")

    # Create model and data
    model = create_demo_model()
    signal_data = create_demo_signal()

    # Create explainer with Integrated Gradients
    config = {
        'method': 'integrated_gradients',
        'n_steps': 20,
        'baseline': 'zero'
    }
    explainer = UnifiedExplainer(model, method='integrated_gradients', **config)

    # Generate explanation
    explanation = explainer.explain(signal_data)

    print(f"Method: {explanation.get_method_name()}")
    print(f"Target class: {explanation.get_data('target_class')}")

    # Get attribution statistics
    attribution_stats = explanation.get_metrics()
    print("Attribution statistics:")
    for metric, value in attribution_stats.items():
        print(f"  {metric}: {value:.4f}")

    # Visualize
    try:
        fig = explanation.visualize(mode='attribution')
        plt.title('Integrated Gradients Explanation')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization error: {e}")

    return explanation


def demo_method_comparison():
    """Demonstrate comparison of different methods."""
    print("\n=== Method Comparison Demo ===")

    # Create model and data
    model = create_demo_model()
    signal_data = create_demo_signal()

    # Create explainer
    explainer = UnifiedExplainer(model, method='auto')

    # Compare available methods
    try:
        explanations = explainer.compare_methods(signal_data)

        print("Comparison results:")
        for method, explanation in explanations.items():
            if explanation is not None:
                metrics = explanation.get_metrics()
                print(f"  {method}:")
                for metric, value in metrics.items():
                    print(f"    {metric}: {value:.4f}")
            else:
                print(f"  {method}: Failed to generate explanation")

    except Exception as e:
        print(f"Comparison error: {e}")


def demo_model_capabilities():
    """Demonstrate getting model explainability information."""
    print("\n=== Model Capabilities Demo ===")

    model = create_demo_model()

    # Get model explainability info
    if hasattr(model, 'get_model_explainability_info'):
        info = model.get_model_explainability_info()
        print(f"Model type: {info['model_type']}")
        print(f"Supported methods: {info['supported_methods']}")
        print("Explainability features:")
        for feature in info['explainability_features']:
            print(f"  - {feature}")

    # Get operator graph
    if hasattr(model, 'get_operator_graph'):
        try:
            graph = model.get_operator_graph()
            print(f"\nArchitecture: {graph['architecture']}")
            print(f"Number of signal processing layers: {len(graph['signal_processing_modules'])}")
        except Exception as e:
            print(f"Could not get operator graph: {e}")


def demo_unified_api():
    """Demonstrate the unified API for quick usage."""
    print("\n=== Unified API Demo ===")

    model = create_demo_model()
    signal_data = create_demo_signal()

    try:
        # Using the quick function
        from explainability.core.unified_explainer import explain_model

        explanation = explain_model(model, signal_data, method='signal_path')
        print(f"Quick explanation generated with method: {explanation.get_method_name()}")

        # Using the factory method
        explainer = UnifiedExplainer.create_explainer(
            model,
            method='signal_path',
            include_frequency_analysis=True
        )
        explanation = explainer.explain(signal_data)
        print(f"Factory explainer method: {explanation.get_method_name()}")

    except Exception as e:
        print(f"Unified API error: {e}")


def main():
    """Run all demos."""
    print("Fault Diagnosis Explainability Toolkit Demo")
    print("=" * 50)

    # Run individual demos
    demo_signal_path_explanation()
    demo_integrated_gradients_explanation()
    demo_method_comparison()
    demo_model_capabilities()
    demo_unified_api()

    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()