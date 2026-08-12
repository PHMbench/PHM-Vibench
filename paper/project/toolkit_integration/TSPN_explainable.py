"""
Explainable Transparent Signal Processing Network

Extended version of TSPN with built-in explainability support.
This version inherits from ExplainableMixin to provide standardized
explainability interfaces.
"""

from scipy import optimize
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Tuple

from .explainable_base import ExplainableMixin, ExplainableModel

# Import original TSPN components (we'll extend them)
from .TSPN import (
    CustomBatchNorm,
    SignalProcessingLayer,
    FeatureExtractorlayer,
    Classifier
)


class ExplainableSignalProcessingLayer(SignalProcessingLayer):
    """
    Extended Signal Processing Layer with explainability features.
    """

    def __init__(self, signal_processing_modules, input_channels, output_channels, skip_connection=True, layer_index=0):
        super().__init__(signal_processing_modules, input_channels, output_channels, skip_connection)
        self.layer_index = layer_index
        self._layer_outputs = {}  # Store intermediate outputs for explanation

    def forward(self, x):
        # Store input for explanation
        self._layer_outputs['input'] = x.clone()

        # Original forward pass with intermediate tracking
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')

        # Store normalized input
        self._layer_outputs['normalized_input'] = normed_x.clone()

        # Weight connection with softmax
        self.weight_connection.weight.data = F.softmax((1.0 / self.temperature) *
                                                       self.weight_connection.weight.data, dim=0)
        weight_out = self.weight_connection(normed_x)

        # Store weight connection output
        self._layer_outputs['weight_output'] = weight_out.clone()

        # Split and process through modules
        splits = torch.split(weight_out, weight_out.size(2) // self.module_num, dim=2)
        outputs = []
        module_outputs = {}

        for i, (module_name, module) in enumerate(self.signal_processing_modules.items()):
            split = splits[i]
            module_output = module(split)
            outputs.append(module_output)
            module_outputs[module_name] = {
                'input': split.clone(),
                'output': module_output.clone(),
                'module_type': type(module).__name__
            }

        x = torch.cat(outputs, dim=2)

        # Store module outputs for explanation
        self._layer_outputs['module_outputs'] = module_outputs
        self._layer_outputs['pre_skip_output'] = x.clone()

        # Skip connection
        if hasattr(self, 'skip_connection'):
            skip_output = self.skip_connection(normed_x)
            self._layer_outputs['skip_output'] = skip_output.clone()
            x = x + skip_output

        # Store final output
        self._layer_outputs['output'] = x.clone()
        self._layer_outputs['weight_matrix'] = self.weight_connection.weight.data.clone()

        return x

    def get_layer_explanation(self) -> Dict[str, Any]:
        """Get explanation information for this layer."""
        return self._layer_outputs.copy()

    def get_module_importance(self) -> Dict[str, float]:
        """Get importance scores for each signal processing module."""
        if 'module_outputs' not in self._layer_outputs:
            return {}

        importance_scores = {}
        for module_name, module_info in self._layer_outputs['module_outputs'].items():
            # Compute importance based on output variance/energy
            output = module_info['output']
            importance = torch.var(output).item()
            importance_scores[module_name] = importance

        return importance_scores


class ExplainableFeatureExtractorLayer(FeatureExtractorlayer):
    """
    Extended Feature Extractor Layer with explainability features.
    """

    def __init__(self, feature_extractor_modules, in_channels=1, out_channels=1):
        super().__init__(feature_extractor_modules, in_channels, out_channels)
        self._feature_outputs = {}

    def forward(self, x):
        # Store input
        self._feature_outputs['input'] = x.clone()

        # Original processing
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.pre_norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')

        weight_out = self.weight_connection(normed_x)
        x = rearrange(weight_out, 'b l c -> b c l')

        outputs = {}
        for module_name, module in self.feature_extractor_modules.items():
            feature_output = module(x)
            outputs[module_name] = {
                'output': feature_output.clone(),
                'feature_name': module_name,
                'module_type': type(module).__name__
            }

        self._feature_outputs['feature_outputs'] = outputs
        self._feature_outputs['weight_output'] = weight_out.clone()

        res = torch.cat([output['output'] for output in outputs.values()], dim=1).squeeze()
        res = self.norm(res)

        self._feature_outputs['final_features'] = res.clone()
        return res

    def get_feature_explanation(self) -> Dict[str, Any]:
        """Get explanation for feature extraction."""
        return self._feature_outputs.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get importance scores for each feature."""
        if 'feature_outputs' not in self._feature_outputs:
            return {}

        importance_scores = {}
        for feature_name, feature_info in self._feature_outputs['feature_outputs'].items():
            # Compute importance based on feature magnitude
            feature_output = feature_info['output']
            importance = torch.mean(torch.abs(feature_output)).item()
            importance_scores[feature_name] = importance

        return importance_scores


class Transparent_Signal_Processing_Network_Explainable(nn.Module, ExplainableMixin):
    """
    Explainable version of Transparent Signal Processing Network.

    This class extends the original TSPN with comprehensive explainability
    support while maintaining the same architecture and functionality.
    """

    def __init__(self, signal_processing_modules, feature_extractor, args):
        nn.Module.__init__(self)
        ExplainableMixin.__init__(self)

        self.layer_num = len(signal_processing_modules)
        self.signal_processing_modules = signal_processing_modules
        self.feature_extractor_modules = feature_extractor
        self.args = args

        self._network_outputs = {}  # Store network-level outputs

        self.init_signal_processing_layers()
        self.init_feature_extractor_layers()
        self.init_classifier()

    def init_signal_processing_layers(self):
        """Initialize signal processing layers with explainability."""
        print('# build explainable signal processing layers')
        in_channels = self.args.in_channels
        out_channels = int(self.args.out_channels * self.args.scale)

        self.signal_processing_layers = nn.ModuleList()
        for i in range(self.layer_num):
            layer = ExplainableSignalProcessingLayer(
                self.signal_processing_modules[i],
                in_channels,
                out_channels,
                self.args.skip_connection,
                layer_index=i
            ).to(self.args.device)

            self.signal_processing_layers.append(layer)
            in_channels = out_channels
            assert out_channels % self.signal_processing_layers[i].module_num == 0

        self.channel_for_feature = out_channels

    def init_feature_extractor_layers(self):
        """Initialize feature extractor layer with explainability."""
        print('# build explainable feature extractor layer')
        self.feature_extractor_layers = ExplainableFeatureExtractorLayer(
            self.feature_extractor_modules,
            self.channel_for_feature,
            self.channel_for_feature
        ).to(self.args.device)

        len_feature = len(self.feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature

    def init_classifier(self):
        """Initialize classifier."""
        print('# build classifier')
        self.clf = Classifier(self.channel_for_classifier, self.args.num_classes).to(self.args.device)

    def forward(self, x):
        """Forward pass with explainability tracking."""
        self._network_outputs = {}
        self._network_outputs['input'] = x.clone()

        # Signal processing layers
        layer_outputs = []
        for i, layer in enumerate(self.signal_processing_layers):
            x = layer(x)
            layer_outputs.append({
                'layer_index': i,
                'layer_name': f'signal_processing_{i}',
                'output': x.clone(),
                'explanation': layer.get_layer_explanation()
            })

        self._network_outputs['signal_processing_outputs'] = layer_outputs
        self._network_outputs['post_signal_processing'] = x.clone()

        # Feature extraction
        features = self.feature_extractor_layers(x)
        self._network_outputs['features'] = features.clone()
        self._network_outputs['feature_explanation'] = self.feature_extractor_layers.get_feature_explanation()

        # Classification
        logits = self.clf(features)
        self._network_outputs['logits'] = logits.clone()

        return logits

    def get_signal_path(self, input_data: torch.Tensor) -> List[Dict[str, Any]]:
        """
        Get the signal transformation path through the model.

        Returns detailed information about how the signal is transformed
        at each layer, including physical interpretations.
        """
        # Forward pass to collect outputs
        with torch.no_grad():
            _ = self.forward(input_data)

        signal_path = []

        # Add input signal information
        signal_path.append({
            'layer_name': 'input',
            'layer_type': 'input',
            'operator_type': 'raw_signal',
            'input_signal': input_data.clone(),
            'output_signal': input_data.clone(),
            'input_stats': self._compute_signal_stats(input_data),
            'output_stats': self._compute_signal_stats(input_data),
            'parameters': {}
        })

        # Add signal processing layers
        for layer_info in self._network_outputs['signal_processing_outputs']:
            layer = self.signal_processing_layers[layer_info['layer_index']]
            layer_explanation = layer.get_layer_explanation()

            path_entry = {
                'layer_index': layer_info['layer_index'],
                'layer_name': layer_info['layer_name'],
                'layer_type': 'SignalProcessingLayer',
                'operator_type': 'multi_operator_processing',
                'input_signal': layer_explanation.get('input'),
                'output_signal': layer_explanation.get('output'),
                'input_stats': self._compute_signal_stats(layer_explanation.get('input')),
                'output_stats': self._compute_signal_stats(layer_explanation.get('output')),
                'parameters': {
                    'weight_matrix_shape': layer_explanation.get('weight_matrix', torch.tensor([])).shape,
                    'module_count': len(layer.signal_processing_modules),
                    'modules': list(layer.signal_processing_modules.keys())
                },
                'module_outputs': layer_explanation.get('module_outputs', {}),
                'module_importance': layer.get_module_importance()
            }
            signal_path.append(path_entry)

        # Add feature extraction layer
        feature_explanation = self._network_outputs['feature_explanation']
        signal_path.append({
            'layer_name': 'feature_extractor',
            'layer_type': 'FeatureExtractorLayer',
            'operator_type': 'statistical_feature_extraction',
            'input_signal': feature_explanation.get('input'),
            'output_signal': feature_explanation.get('final_features'),
            'input_stats': self._compute_signal_stats(feature_explanation.get('input')),
            'output_stats': self._compute_signal_stats(feature_explanation.get('final_features')),
            'parameters': {
                'feature_count': len(self.feature_extractor_modules),
                'features': list(self.feature_extractor_modules.keys())
            },
            'feature_outputs': feature_explanation.get('feature_outputs', {}),
            'feature_importance': self.feature_extractor_layers.get_feature_importance()
        })

        return signal_path

    def get_operator_graph(self) -> Dict[str, Any]:
        """
        Get the operator graph structure of the model.

        Returns information about how different operators are connected
        and their roles in the signal processing pipeline.
        """
        graph_info = {
            'model_type': 'Transparent_Signal_Processing_Network_Explainable',
            'architecture': {
                'signal_processing_layers': self.layer_num,
                'feature_extractor_features': len(self.feature_extractor_modules),
                'num_classes': self.args.num_classes,
                'input_channels': self.args.in_channels,
                'output_channels': self.args.out_channels
            },
            'operator_connections': [],
            'signal_processing_modules': [],
            'feature_extraction_modules': list(self.feature_extractor_modules.keys())
        }

        # Build signal processing module information
        for i, layer_modules in enumerate(self.signal_processing_modules):
            layer_info = {
                'layer_index': i,
                'layer_name': f'signal_processing_{i}',
                'modules': {}
            }

            for module_name, module in layer_modules.items():
                layer_info['modules'][module_name] = {
                    'module_type': type(module).__name__,
                    'position_in_layer': list(layer_modules.keys()).index(module_name)
                }

            graph_info['signal_processing_modules'].append(layer_info)

        # Build operator connections
        for i in range(self.layer_num):
            if i == 0:
                input_layer = 'input'
            else:
                input_layer = f'signal_processing_{i-1}'

            current_layer = f'signal_processing_{i}'

            connection = {
                'from': input_layer,
                'to': current_layer,
                'connection_type': 'sequential',
                'has_skip_connection': self.args.skip_connection
            }
            graph_info['operator_connections'].append(connection)

        # Add feature extractor connection
        if self.layer_num > 0:
            last_signal_layer = f'signal_processing_{self.layer_num-1}'
            graph_info['operator_connections'].append({
                'from': last_signal_layer,
                'to': 'feature_extractor',
                'connection_type': 'feature_extraction'
            })

        # Add classifier connection
        graph_info['operator_connections'].append({
            'from': 'feature_extractor',
            'to': 'classifier',
            'connection_type': 'classification'
        })

        return graph_info

    def get_attention_maps(self, input_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get attention weights/maps from the model.

        For TSPN, this includes the weight matrices that determine
        the importance of different signal processing modules.
        """
        # Forward pass to collect outputs
        with torch.no_grad():
            _ = self.forward(input_data)

        attention_maps = {}

        # Collect weight connection matrices from signal processing layers
        for i, layer in enumerate(self.signal_processing_layers):
            layer_explanation = layer.get_layer_explanation()
            weight_matrix = layer_explanation.get('weight_matrix')

            if weight_matrix is not None:
                attention_maps[f'signal_processing_{i}_weights'] = weight_matrix.clone()

                # Convert to attention weights (normalized)
                attention_weights = F.softmax(weight_matrix, dim=0)
                attention_maps[f'signal_processing_{i}_attention'] = attention_weights.clone()

        return attention_maps

    def get_model_explainability_info(self) -> Dict[str, Any]:
        """Get information about the explainability capabilities of this model."""
        return {
            'model_type': 'Transparent_Signal_Processing_Network_Explainable',
            'supported_methods': ['signal_path', 'operator_graph', 'attention', 'intermediate'],
            'architecture': self.get_operator_graph()['architecture'],
            'explainability_features': [
                'Signal transformation path tracking',
                'Operator importance analysis',
                'Module-level attention weights',
                'Feature importance scoring',
                'Physical interpretation of signal processing'
            ]
        }

    def _compute_signal_stats(self, signal: Optional[torch.Tensor]) -> Dict[str, float]:
        """Compute basic signal statistics."""
        if signal is None:
            return {}

        if isinstance(signal, torch.Tensor):
            signal_np = signal.detach().cpu().numpy()
        else:
            signal_np = np.array(signal)

        return {
            'mean': float(np.mean(signal_np)),
            'std': float(np.std(signal_np)),
            'rms': float(np.sqrt(np.mean(signal_np ** 2))),
            'max': float(np.max(signal_np)),
            'min': float(np.min(signal_np)),
            'energy': float(np.sum(signal_np ** 2))
        }


# Convenience function to create explainable TSPN
def create_explainable_tspn(signal_processing_modules, feature_extractor, args):
    """
    Factory function to create an explainable TSPN model.

    Args:
        signal_processing_modules: Signal processing module configurations
        feature_extractor: Feature extractor module configurations
        args: Model arguments

    Returns:
        Explainable TSPN model instance
    """
    return Transparent_Signal_Processing_Network_Explainable(
        signal_processing_modules, feature_extractor, args
    )