"""
Comprehensive test suite for PHM-Vibench Model Factory

This module tests all implemented models across different categories
to ensure they work correctly with various configurations.
"""

import pytest
import torch
import numpy as np
from argparse import Namespace
from typing import Dict, Any, List, Tuple
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_factory import build_model


class TestModelFactory:
    """Test suite for model factory functionality."""
    
    @pytest.fixture
    def device(self):
        """Get available device for testing."""
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        return {
            'classification': {
                'x': torch.randn(8, 64, 3),  # (batch, seq_len, features)
                'y': torch.randint(0, 4, (8,))  # (batch,)
            },
            'regression': {
                'x': torch.randn(8, 64, 3),
                'y': torch.randn(8, 24, 3)  # (batch, pred_len, features)
            },
            'multimodal': {
                'x': {
                    'vibration': torch.randn(8, 64, 3),
                    'acoustic': torch.randn(8, 64, 1),
                    'thermal': torch.randn(8, 2)
                },
                'y': torch.randint(0, 4, (8,))
            }
        }
    
    def test_model_factory_import(self):
        """Test that model factory can be imported."""
        from src.model_factory import build_model
        assert build_model is not None
    
    def test_invalid_model_name(self):
        """Test that invalid model names raise appropriate errors."""
        args = Namespace(
            model_name='NonExistentModel',
            input_dim=3
        )
        
        with pytest.raises((KeyError, AttributeError, ImportError)):
            build_model(args)


class TestMLPModels:
    """Test suite for MLP model family."""
    
    @pytest.fixture
    def mlp_models(self):
        """MLP model configurations for testing."""
        return {
            'ResNetMLP': Namespace(
                model_name='ResNetMLP',
                input_dim=3,
                hidden_dim=64,
                num_layers=3,
                dropout=0.1,
                num_classes=4
            ),
            'MLPMixer': Namespace(
                model_name='MLPMixer',
                input_dim=3,
                patch_size=8,
                hidden_dim=64,
                num_layers=4,
                mlp_ratio=2.0,
                dropout=0.1,
                num_classes=4
            ),
            'gMLP': Namespace(
                model_name='gMLP',
                input_dim=3,
                hidden_dim=64,
                num_layers=3,
                seq_len=64,
                dropout=0.1,
                num_classes=4
            ),
            'DenseNetMLP': Namespace(
                model_name='DenseNetMLP',
                input_dim=3,
                growth_rate=16,
                num_layers=3,
                hidden_dim=64,
                dropout=0.1,
                num_classes=4
            ),
            'Dlinear': Namespace(
                model_name='Dlinear',
                input_dim=3,
                seq_len=64,
                pred_len=24,
                kernel_size=25,
                individual=False
            )
        }
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'MLPMixer', 'gMLP', 'DenseNetMLP', 'Dlinear'])
    def test_mlp_model_creation(self, mlp_models, model_name):
        """Test MLP model creation."""
        args = mlp_models[model_name]

        # Import and instantiate model directly to avoid metadata issues
        if model_name == 'ResNetMLP':
            from src.model_factory.MLP.ResNetMLP import Model
        elif model_name == 'MLPMixer':
            from src.model_factory.MLP.MLPMixer import Model
        elif model_name == 'gMLP':
            from src.model_factory.MLP.gMLP import Model
        elif model_name == 'DenseNetMLP':
            from src.model_factory.MLP.DenseNetMLP import Model
        elif model_name == 'Dlinear':
            from src.model_factory.MLP.Dlinear import Model

        model = Model(args)
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.parametrize("model_name", ['ResNetMLP', 'MLPMixer', 'gMLP', 'DenseNetMLP'])
    def test_mlp_classification_forward(self, mlp_models, model_name):
        """Test MLP models forward pass for classification."""
        args = mlp_models[model_name]

        # Import and instantiate model directly
        if model_name == 'ResNetMLP':
            from src.model_factory.MLP.ResNetMLP import Model
        elif model_name == 'MLPMixer':
            from src.model_factory.MLP.MLPMixer import Model
        elif model_name == 'gMLP':
            from src.model_factory.MLP.gMLP import Model
        elif model_name == 'DenseNetMLP':
            from src.model_factory.MLP.DenseNetMLP import Model

        model = Model(args)

        x = torch.randn(4, 64, 3)
        output = model(x)

        assert output.shape == (4, 4)  # (batch_size, num_classes)
        assert not torch.isnan(output).any()
    
    def test_dlinear_regression_forward(self, mlp_models):
        """Test Dlinear model forward pass for regression."""
        args = mlp_models['Dlinear']

        from src.model_factory.MLP.Dlinear import Model
        model = Model(args)

        x = torch.randn(4, 64, 3)
        output = model(x)

        assert output.shape == (4, 24, 3)  # (batch_size, pred_len, features)
        assert not torch.isnan(output).any()


class TestNeuralOperators:
    """Test suite for Neural Operator models."""
    
    @pytest.fixture
    def no_models(self):
        """Neural Operator model configurations."""
        return {
            'FNO': Namespace(
                model_name='FNO',
                input_dim=3,
                output_dim=3,
                modes=8,
                width=32,
                num_layers=2
            ),
            'DeepONet': Namespace(
                model_name='DeepONet',
                input_dim=3,
                branch_depth=3,
                trunk_depth=3,
                width=64,
                trunk_dim=1,
                output_dim=3
            ),
            'NeuralODE': Namespace(
                model_name='NeuralODE',
                input_dim=3,
                hidden_dim=32,
                num_layers=2,
                solver='euler',
                rtol=1e-2,
                atol=1e-3,
                num_classes=4
            ),
            'GraphNO': Namespace(
                model_name='GraphNO',
                input_dim=3,
                hidden_dim=32,
                num_layers=2,
                num_eigenvectors=16,
                graph_size=64,
                num_classes=4
            ),
            'WaveletNO': Namespace(
                model_name='WaveletNO',
                input_dim=3,
                hidden_dim=32,
                num_layers=2,
                wavelet='db2',
                num_levels=2,
                num_classes=4
            )
        }
    
    @pytest.mark.parametrize("model_name", ['FNO', 'DeepONet', 'NeuralODE', 'GraphNO', 'WaveletNO'])
    def test_no_model_creation(self, no_models, model_name):
        """Test Neural Operator model creation."""
        args = no_models[model_name]
        model = build_model(args)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_fno_forward(self, no_models):
        """Test FNO forward pass."""
        args = no_models['FNO']
        model = build_model(args)
        
        x = torch.randn(4, 64, 3)
        output = model(x)
        
        assert output.shape == (4, 64, 3)
        assert not torch.isnan(output).any()
    
    def test_deeponet_forward(self, no_models):
        """Test DeepONet forward pass."""
        args = no_models['DeepONet']
        model = build_model(args)
        
        branch_input = torch.randn(4, 32, 3)
        trunk_input = torch.randn(4, 16, 1)
        output = model(branch_input, trunk_input)
        
        assert output.shape == (4, 16, 3)
        assert not torch.isnan(output).any()


class TestTransformerModels:
    """Test suite for Transformer models."""
    
    @pytest.fixture
    def transformer_models(self):
        """Transformer model configurations."""
        return {
            'Informer': Namespace(
                model_name='Informer',
                input_dim=3,
                d_model=64,
                n_heads=4,
                e_layers=2,
                d_layers=1,
                d_ff=128,
                factor=3,
                dropout=0.1,
                num_classes=4
            ),
            'Autoformer': Namespace(
                model_name='Autoformer',
                input_dim=3,
                d_model=64,
                n_heads=4,
                e_layers=2,
                d_layers=1,
                moving_avg=25,
                dropout=0.1,
                num_classes=4
            ),
            'PatchTST': Namespace(
                model_name='PatchTST',
                input_dim=3,
                d_model=64,
                n_heads=4,
                e_layers=2,
                patch_len=8,
                stride=4,
                seq_len=64,
                pred_len=24,
                dropout=0.1
            ),
            'Linformer': Namespace(
                model_name='Linformer',
                input_dim=3,
                d_model=64,
                n_heads=4,
                num_layers=2,
                seq_len=64,
                k=32,
                dropout=0.1,
                num_classes=4
            ),
            'ConvTransformer': Namespace(
                model_name='ConvTransformer',
                input_dim=3,
                d_model=64,
                n_heads=4,
                num_layers=2,
                kernel_size=3,
                dropout=0.1,
                num_classes=4
            )
        }
    
    @pytest.mark.parametrize("model_name", ['Informer', 'Autoformer', 'PatchTST', 'Linformer', 'ConvTransformer'])
    def test_transformer_model_creation(self, transformer_models, model_name):
        """Test Transformer model creation."""
        args = transformer_models[model_name]
        model = build_model(args)
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.parametrize("model_name", ['Informer', 'Autoformer', 'Linformer', 'ConvTransformer'])
    def test_transformer_classification_forward(self, transformer_models, model_name):
        """Test Transformer models forward pass for classification."""
        args = transformer_models[model_name]
        model = build_model(args)
        
        x = torch.randn(4, 64, 3)
        output = model(x)
        
        assert output.shape == (4, 4)  # (batch_size, num_classes)
        assert not torch.isnan(output).any()
    
    def test_patchtst_regression_forward(self, transformer_models):
        """Test PatchTST forward pass for regression."""
        args = transformer_models['PatchTST']
        model = build_model(args)
        
        x = torch.randn(4, 64, 3)
        output = model(x)
        
        assert output.shape == (4, 24, 3)  # (batch_size, pred_len, features)
        assert not torch.isnan(output).any()


class TestRNNModels:
    """Test suite for RNN models."""
    
    @pytest.fixture
    def rnn_models(self):
        """RNN model configurations."""
        return {
            'AttentionLSTM': Namespace(
                model_name='AttentionLSTM',
                input_dim=3,
                hidden_dim=32,
                num_layers=2,
                dropout=0.1,
                bidirectional=True,
                num_classes=4
            ),
            'ConvLSTM': Namespace(
                model_name='ConvLSTM',
                input_dim=3,
                hidden_dim=32,
                kernel_size=3,
                num_layers=2,
                dropout=0.1,
                num_classes=4
            ),
            'ResidualRNN': Namespace(
                model_name='ResidualRNN',
                input_dim=3,
                hidden_dim=32,
                num_layers=3,
                rnn_type='LSTM',
                dropout=0.1,
                num_classes=4
            ),
            'AttentionGRU': Namespace(
                model_name='AttentionGRU',
                input_dim=3,
                hidden_dim=32,
                num_layers=2,
                num_heads=4,
                dropout=0.1,
                num_classes=4
            ),
            'TransformerRNN': Namespace(
                model_name='TransformerRNN',
                input_dim=3,
                hidden_dim=32,
                transformer_layers=2,
                rnn_layers=2,
                num_heads=4,
                dropout=0.1,
                num_classes=4
            )
        }
    
    @pytest.mark.parametrize("model_name", ['AttentionLSTM', 'ConvLSTM', 'ResidualRNN', 'AttentionGRU', 'TransformerRNN'])
    def test_rnn_model_creation(self, rnn_models, model_name):
        """Test RNN model creation."""
        args = rnn_models[model_name]
        model = build_model(args)
        assert model is not None
        assert hasattr(model, 'forward')
    
    @pytest.mark.parametrize("model_name", ['AttentionLSTM', 'ConvLSTM', 'ResidualRNN', 'AttentionGRU', 'TransformerRNN'])
    def test_rnn_forward(self, rnn_models, model_name):
        """Test RNN models forward pass."""
        args = rnn_models[model_name]
        model = build_model(args)

        x = torch.randn(4, 64, 3)
        output = model(x)

        assert output.shape == (4, 4)  # (batch_size, num_classes)
        assert not torch.isnan(output).any()


class TestCNNModels:
    """Test suite for CNN models."""

    @pytest.fixture
    def cnn_models(self):
        """CNN model configurations."""
        return {
            'ResNet1D': Namespace(
                model_name='ResNet1D',
                input_dim=3,
                block_type='basic',
                layers=[2, 2, 2],
                dropout=0.1,
                num_classes=4
            ),
            'TCN': Namespace(
                model_name='TCN',
                input_dim=3,
                num_channels=[16, 16, 16],
                kernel_size=3,
                dropout=0.1,
                num_classes=4
            ),
            'AttentionCNN': Namespace(
                model_name='AttentionCNN',
                input_dim=3,
                base_channels=16,
                num_blocks=3,
                reduction_ratio=4,
                dropout=0.1,
                num_classes=4
            ),
            'MobileNet1D': Namespace(
                model_name='MobileNet1D',
                input_dim=3,
                width_multiplier=0.5,
                inverted_residual_setting=[
                    [1, 16, 1, 1],
                    [6, 24, 1, 2]
                ],
                dropout=0.1,
                num_classes=4
            ),
            'MultiScaleCNN': Namespace(
                model_name='MultiScaleCNN',
                input_dim=3,
                multiscale_channels=[32, 64],
                scales=[3, 5],
                dropout=0.1,
                num_classes=4
            )
        }

    @pytest.mark.parametrize("model_name", ['ResNet1D', 'TCN', 'AttentionCNN', 'MobileNet1D', 'MultiScaleCNN'])
    def test_cnn_model_creation(self, cnn_models, model_name):
        """Test CNN model creation."""
        args = cnn_models[model_name]
        model = build_model(args)
        assert model is not None
        assert hasattr(model, 'forward')

    @pytest.mark.parametrize("model_name", ['ResNet1D', 'TCN', 'AttentionCNN', 'MobileNet1D', 'MultiScaleCNN'])
    def test_cnn_forward(self, cnn_models, model_name):
        """Test CNN models forward pass."""
        args = cnn_models[model_name]
        model = build_model(args)

        x = torch.randn(4, 64, 3)
        output = model(x)

        assert output.shape == (4, 4)  # (batch_size, num_classes)
        assert not torch.isnan(output).any()


class TestISFMModels:
    """Test suite for Industrial Signal Foundation Models."""

    @pytest.fixture
    def isfm_models(self):
        """ISFM model configurations."""
        return {
            'ContrastiveSSL': Namespace(
                model_name='ContrastiveSSL',
                input_dim=3,
                hidden_dim=64,
                num_layers=3,
                num_heads=4,
                projection_dim=32,
                temperature=0.1,
                dropout=0.1,
                num_classes=4
            ),
            'MaskedAutoencoder': Namespace(
                model_name='MaskedAutoencoder',
                input_dim=3,
                patch_size=8,
                embed_dim=64,
                decoder_embed_dim=32,
                num_layers=3,
                decoder_num_layers=2,
                num_heads=4,
                decoder_num_heads=4,
                mask_ratio=0.75,
                dropout=0.1,
                max_seq_len=64,
                num_classes=4
            ),
            'MultiModalFM': Namespace(
                model_name='MultiModalFM',
                modality_dims={'vibration': 3, 'acoustic': 1, 'thermal': 2},
                hidden_dim=64,
                num_layers=2,
                fusion_type='attention',
                dropout=0.1,
                num_classes=4
            ),
            'SignalLanguageFM': Namespace(
                model_name='SignalLanguageFM',
                input_dim=3,
                vocab_size=1000,
                hidden_dim=64,
                signal_layers=3,
                text_layers=2,
                num_heads=4,
                temperature=0.07,
                max_text_len=32,
                num_classes=4
            ),
            'TemporalDynamicsSSL': Namespace(
                model_name='TemporalDynamicsSSL',
                input_dim=3,
                hidden_dim=64,
                num_layers=3,
                num_heads=4,
                ssl_tasks=['next_step', 'permutation', 'mask'],
                crop_ratio=0.8,
                permute_segments=4,
                num_classes=4
            )
        }

    @pytest.mark.parametrize("model_name", ['ContrastiveSSL', 'MaskedAutoencoder', 'MultiModalFM', 'SignalLanguageFM', 'TemporalDynamicsSSL'])
    def test_isfm_model_creation(self, isfm_models, model_name):
        """Test ISFM model creation."""
        args = isfm_models[model_name]
        model = build_model(args)
        assert model is not None
        assert hasattr(model, 'forward')

    def test_contrastive_ssl_modes(self, isfm_models):
        """Test ContrastiveSSL different modes."""
        args = isfm_models['ContrastiveSSL']
        model = build_model(args)

        x = torch.randn(4, 32, 3)

        # Test contrastive mode
        output_contrastive = model(x, mode='contrastive')
        assert 'loss' in output_contrastive
        assert 'features' in output_contrastive
        assert 'projections' in output_contrastive

        # Test downstream mode
        output_downstream = model(x, mode='downstream')
        assert output_downstream.shape == (4, 4)

    def test_masked_autoencoder_modes(self, isfm_models):
        """Test MaskedAutoencoder different modes."""
        args = isfm_models['MaskedAutoencoder']
        model = build_model(args)

        x = torch.randn(4, 64, 3)

        # Test pretrain mode
        output_pretrain = model(x, mode='pretrain')
        assert 'pred' in output_pretrain
        assert 'mask' in output_pretrain
        assert 'latent' in output_pretrain

        # Test downstream mode
        output_downstream = model(x, mode='downstream')
        assert output_downstream.shape == (4, 4)

    def test_multimodal_fm_forward(self, isfm_models):
        """Test MultiModalFM forward pass."""
        args = isfm_models['MultiModalFM']
        model = build_model(args)

        x = {
            'vibration': torch.randn(4, 32, 3),
            'acoustic': torch.randn(4, 32, 1),
            'thermal': torch.randn(4, 2)
        }
        output = model(x)

        assert output.shape == (4, 4)
        assert not torch.isnan(output).any()

    def test_signal_language_fm_modes(self, isfm_models):
        """Test SignalLanguageFM different modes."""
        args = isfm_models['SignalLanguageFM']
        model = build_model(args)

        signals = torch.randn(4, 32, 3)
        texts = torch.randint(0, 1000, (4, 16))
        attention_mask = torch.ones(4, 16)

        # Test contrastive mode
        output_contrastive = model(signals, texts, attention_mask, mode='contrastive')
        assert 'loss' in output_contrastive
        assert 'signal_features' in output_contrastive
        assert 'text_features' in output_contrastive

        # Test downstream mode
        output_downstream = model(signals, mode='downstream')
        assert output_downstream.shape == (4, 4)

    def test_temporal_dynamics_ssl_modes(self, isfm_models):
        """Test TemporalDynamicsSSL different modes."""
        args = isfm_models['TemporalDynamicsSSL']
        model = build_model(args)

        x = torch.randn(4, 32, 3)

        # Test SSL mode
        output_ssl = model(x, mode='ssl')
        assert 'total_loss' in output_ssl
        assert 'ssl_losses' in output_ssl
        assert 'features' in output_ssl

        # Test downstream mode
        output_downstream = model(x, mode='downstream')
        assert output_downstream.shape == (4, 4)
