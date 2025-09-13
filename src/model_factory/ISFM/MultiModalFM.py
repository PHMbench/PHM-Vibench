"""Multi-Modal Foundation Model for Industrial Signals."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math


class ModalityEncoder(nn.Module):
    """Encoder for a specific modality.
    
    Parameters
    ----------
    input_dim : int
        Input dimension for this modality
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of layers
    modality_type : str
        Type of modality ('vibration', 'acoustic', 'thermal', etc.)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 4, 
                 modality_type: str = 'vibration'):
        super(ModalityEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.modality_type = modality_type
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Modality-specific processing
        if modality_type in ['vibration', 'acoustic']:
            # Use 1D CNN for temporal signals
            self.encoder = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        elif modality_type == 'thermal':
            # Use MLP for thermal data
            layers = []
            for _ in range(num_layers):
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
            self.encoder = nn.Sequential(*layers)
        else:
            # Default transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                batch_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim) or (B, input_dim)
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, hidden_dim)
        """
        # Input projection
        x = self.input_projection(x)  # (B, L, hidden_dim) or (B, hidden_dim)
        
        if self.modality_type in ['vibration', 'acoustic']:
            # For temporal signals
            if x.dim() == 3:
                x = x.transpose(1, 2)  # (B, hidden_dim, L)
                x = self.encoder(x).squeeze(-1)  # (B, hidden_dim)
            else:
                x = x.unsqueeze(-1)  # (B, hidden_dim, 1)
                x = self.encoder(x).squeeze(-1)  # (B, hidden_dim)
        
        elif self.modality_type == 'thermal':
            # For thermal data (typically scalar or low-dimensional)
            if x.dim() == 3:
                x = x.mean(dim=1)  # Average over time: (B, hidden_dim)
            x = self.encoder(x)  # (B, hidden_dim)
        
        else:
            # For other modalities using transformer
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (B, 1, hidden_dim)
            x = self.encoder(x)  # (B, L, hidden_dim)
            x = x.mean(dim=1)  # (B, hidden_dim)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class CrossModalAttention(nn.Module):
    """Cross-modal attention mechanism.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    num_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super(CrossModalAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        query : torch.Tensor
            Query tensor of shape (B, N_q, hidden_dim)
        key : torch.Tensor
            Key tensor of shape (B, N_k, hidden_dim)
        value : torch.Tensor
            Value tensor of shape (B, N_v, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Attended output of shape (B, N_q, hidden_dim)
        """
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # Project to Q, K, V
        Q = self.query_proj(query).view(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key_proj(key).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value_proj(value).view(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, N_q, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, N_q, self.hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class ModalityFusion(nn.Module):
    """Multi-modal fusion module.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    num_modalities : int
        Number of modalities
    fusion_type : str
        Type of fusion ('attention', 'concat', 'add')
    """
    
    def __init__(self, hidden_dim: int, num_modalities: int, fusion_type: str = 'attention'):
        super(ModalityFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.fusion_type = fusion_type
        
        if fusion_type == 'attention':
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)
        
        elif fusion_type == 'concat':
            # Concatenation + projection
            self.fusion_proj = nn.Linear(hidden_dim * num_modalities, hidden_dim)
        
        elif fusion_type == 'add':
            # Simple addition (no additional parameters)
            pass
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, modality_features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        modality_features : List[torch.Tensor]
            List of modality features, each of shape (B, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Fused features of shape (B, hidden_dim)
        """
        if self.fusion_type == 'attention':
            # Stack modality features
            stacked = torch.stack(modality_features, dim=1)  # (B, num_modalities, hidden_dim)
            
            # Self-attention across modalities
            attended = self.cross_attention(stacked, stacked, stacked)  # (B, num_modalities, hidden_dim)
            
            # Global pooling
            fused = attended.mean(dim=1)  # (B, hidden_dim)
            fused = self.norm(fused)
        
        elif self.fusion_type == 'concat':
            # Concatenate and project
            concatenated = torch.cat(modality_features, dim=-1)  # (B, hidden_dim * num_modalities)
            fused = self.fusion_proj(concatenated)  # (B, hidden_dim)
        
        elif self.fusion_type == 'add':
            # Simple addition
            fused = torch.stack(modality_features, dim=0).sum(dim=0)  # (B, hidden_dim)
        
        return fused


class Model(nn.Module):
    """Multi-Modal Foundation Model for Industrial Signals.
    
    Processes multiple modalities (vibration, acoustic, thermal, etc.)
    and fuses them for comprehensive industrial signal analysis.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - modality_dims : dict, input dimensions for each modality
        - hidden_dim : int, hidden dimension (default: 256)
        - num_layers : int, number of layers per encoder (default: 4)
        - fusion_type : str, fusion method (default: 'attention')
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for classification)
        - output_dim : int, output dimension (for regression)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : dict
        Dictionary of modality inputs, each of shape (batch_size, seq_len, modality_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For classification: (batch_size, num_classes)
        For regression: (batch_size, output_dim)
        
    References
    ----------
    Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" ICML 2021.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Nagrani et al. "Attention Bottlenecks for Multimodal Fusion" NeurIPS 2021.
    Adapted for multi-modal industrial signals with cross-modal attention fusion for vibration, acoustic, and thermal data.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.modality_dims = getattr(args, 'modality_dims', {'vibration': 3, 'acoustic': 1, 'thermal': 1})
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.num_layers = getattr(args, 'num_layers', 4)
        self.fusion_type = getattr(args, 'fusion_type', 'attention')
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', 1)
        
        # Modality encoders
        self.modality_encoders = nn.ModuleDict()
        for modality, input_dim in self.modality_dims.items():
            encoder = ModalityEncoder(
                input_dim=input_dim,
                hidden_dim=self.hidden_dim,
                num_layers=self.num_layers,
                modality_type=modality
            )
            self.modality_encoders[modality] = encoder
        
        # Fusion module
        self.fusion = ModalityFusion(
            hidden_dim=self.hidden_dim,
            num_modalities=len(self.modality_dims),
            fusion_type=self.fusion_type
        )
        
        # Output layers
        if self.num_classes is not None:
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression head
            self.regressor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
            self.task_type = 'regression'
    
    def forward(self, x: Dict[str, torch.Tensor], data_id=None, task_id=None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : Dict[str, torch.Tensor]
            Dictionary of modality inputs
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
            
        Returns
        -------
        torch.Tensor
            Output tensor shape depends on task type
        """
        # Encode each modality
        modality_features = []
        for modality, data in x.items():
            if modality in self.modality_encoders:
                features = self.modality_encoders[modality](data)
                modality_features.append(features)
        
        # Fuse modalities
        fused_features = self.fusion(modality_features)
        
        # Output prediction
        if self.task_type == 'classification':
            output = self.classifier(fused_features)
        else:
            output = self.regressor(fused_features)
        
        return output


if __name__ == "__main__":
    # Test Multi-Modal Foundation Model
    import torch
    from argparse import Namespace
    
    def test_multimodal_fm():
        """Test Multi-Modal Foundation Model."""
        print("Testing Multi-Modal Foundation Model...")
        
        # Test configuration
        args = Namespace(
            modality_dims={'vibration': 3, 'acoustic': 1, 'thermal': 2},
            hidden_dim=128,
            num_layers=3,
            fusion_type='attention',
            dropout=0.1,
            num_classes=4
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64
        x = {
            'vibration': torch.randn(batch_size, seq_len, 3),
            'acoustic': torch.randn(batch_size, seq_len, 1),
            'thermal': torch.randn(batch_size, 2)  # Scalar thermal data
        }
        
        # Forward pass
        with torch.no_grad():
            output = model(x)
        
        print(f"Input modalities: {list(x.keys())}")
        print(f"Input shapes: {[v.shape for v in x.values()]}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (batch_size, args.num_classes)
        
        # Test with different fusion type
        args.fusion_type = 'concat'
        model_concat = Model(args)
        
        with torch.no_grad():
            output_concat = model_concat(x)
        
        print(f"Concat fusion output shape: {output_concat.shape}")
        assert output_concat.shape == (batch_size, args.num_classes)
        
        print("✅ Multi-Modal Foundation Model tests passed!")
        return True
    
    test_multimodal_fm()
