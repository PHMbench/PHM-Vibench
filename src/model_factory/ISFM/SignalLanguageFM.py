"""Signal-Language Foundation Model for Industrial Signals with Text Descriptions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import math


class SignalEncoder(nn.Module):
    """Encoder for industrial signal data.
    
    Parameters
    ----------
    input_dim : int
        Input signal dimension
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 6, num_heads: int = 8):
        super(SignalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input signal of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Signal features of shape (B, hidden_dim)
        """
        B, L, _ = x.shape
        
        # Input projection
        x = self.input_projection(x)  # (B, L, hidden_dim)
        
        # Add positional encoding
        if L <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :L, :]
        else:
            # Interpolate for longer sequences
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2), 
                size=L, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)  # (B, L, hidden_dim)
        
        # Global pooling
        x = x.transpose(1, 2)  # (B, hidden_dim, L)
        x = self.global_pool(x).squeeze(-1)  # (B, hidden_dim)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class TextEncoder(nn.Module):
    """Simple text encoder for equipment descriptions.
    
    Parameters
    ----------
    vocab_size : int
        Vocabulary size
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads
    max_seq_len : int
        Maximum sequence length
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, num_layers: int = 4, 
                 num_heads: int = 8, max_seq_len: int = 128):
        super(TextEncoder, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input token IDs of shape (B, L)
        attention_mask : torch.Tensor, optional
            Attention mask of shape (B, L)
            
        Returns
        -------
        torch.Tensor
            Text features of shape (B, hidden_dim)
        """
        B, L = x.shape
        
        # Token embedding
        x = self.token_embedding(x)  # (B, L, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :L, :]
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for positions to mask)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)  # (B, L, hidden_dim)
        
        # Use [CLS] token or mean pooling
        if attention_mask is not None:
            # Mean pooling over valid tokens
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x)
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            # Simple mean pooling
            x = x.mean(dim=1)  # (B, hidden_dim)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class ContrastiveLoss(nn.Module):
    """Contrastive loss for signal-text alignment."""
    
    def __init__(self, temperature: float = 0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, signal_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """Compute contrastive loss.
        
        Parameters
        ----------
        signal_features : torch.Tensor
            Signal features of shape (B, hidden_dim)
        text_features : torch.Tensor
            Text features of shape (B, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Contrastive loss
        """
        # Normalize features
        signal_features = F.normalize(signal_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(signal_features, text_features.t()) / self.temperature
        
        # Labels (diagonal elements are positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute loss in both directions
        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.t(), labels)
        
        return (loss_s2t + loss_t2s) / 2


class Model(nn.Module):
    """Signal-Language Foundation Model for Industrial Signals.
    
    Learns joint representations of industrial signals and their textual
    descriptions for enhanced understanding and zero-shot capabilities.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input signal dimension
        - vocab_size : int, text vocabulary size (default: 10000)
        - hidden_dim : int, hidden dimension (default: 256)
        - signal_layers : int, number of signal encoder layers (default: 6)
        - text_layers : int, number of text encoder layers (default: 4)
        - num_heads : int, number of attention heads (default: 8)
        - temperature : float, contrastive loss temperature (default: 0.07)
        - max_text_len : int, maximum text sequence length (default: 128)
        - num_classes : int, number of output classes (for downstream tasks)
        - output_dim : int, output dimension (for regression tasks)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    signals : torch.Tensor
        Signal tensor of shape (batch_size, seq_len, input_dim)
    texts : torch.Tensor
        Text token IDs of shape (batch_size, text_len)
        
    Output Shape
    ------------
    torch.Tensor or dict
        For contrastive learning: dict with loss and features
        For downstream tasks: depends on task type
        
    References
    ----------
    Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" ICML 2021.
    Vaswani et al. "Attention Is All You Need" NeurIPS 2017.
    Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" ICML 2020.
    Adapted for industrial signal-text pairs with contrastive learning for joint signal-language representation learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.vocab_size = getattr(args, 'vocab_size', 10000)
        self.hidden_dim = getattr(args, 'hidden_dim', 256)
        self.signal_layers = getattr(args, 'signal_layers', 6)
        self.text_layers = getattr(args, 'text_layers', 4)
        self.num_heads = getattr(args, 'num_heads', 8)
        self.temperature = getattr(args, 'temperature', 0.07)
        self.max_text_len = getattr(args, 'max_text_len', 128)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', 1)
        
        # Signal encoder
        self.signal_encoder = SignalEncoder(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.signal_layers,
            num_heads=self.num_heads
        )
        
        # Text encoder
        self.text_encoder = TextEncoder(
            vocab_size=self.vocab_size,
            hidden_dim=self.hidden_dim,
            num_layers=self.text_layers,
            num_heads=self.num_heads,
            max_seq_len=self.max_text_len
        )
        
        # Contrastive loss
        self.contrastive_loss = ContrastiveLoss(temperature=self.temperature)
        
        # Downstream task heads
        if self.num_classes is not None:
            # Classification head
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression head
            self.regressor = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.hidden_dim // 2, self.output_dim)
            )
            self.task_type = 'regression'
    
    def forward(self, signals: torch.Tensor, texts: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None, data_id=None, task_id=None,
                mode: str = 'contrastive') -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        signals : torch.Tensor
            Signal tensor of shape (B, L, input_dim)
        texts : torch.Tensor, optional
            Text token IDs of shape (B, text_len)
        attention_mask : torch.Tensor, optional
            Text attention mask of shape (B, text_len)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
        mode : str
            Forward mode: 'contrastive' for signal-text alignment,
            'downstream' for supervised tasks
            
        Returns
        -------
        torch.Tensor or dict
            Output depends on mode and task type
        """
        # Encode signals
        signal_features = self.signal_encoder(signals)
        
        if mode == 'contrastive' and texts is not None:
            # Encode texts
            text_features = self.text_encoder(texts, attention_mask)
            
            # Compute contrastive loss
            loss = self.contrastive_loss(signal_features, text_features)
            
            return {
                'loss': loss,
                'signal_features': signal_features,
                'text_features': text_features
            }
        
        else:  # downstream mode
            if self.task_type == 'classification':
                output = self.classifier(signal_features)
            else:
                output = self.regressor(signal_features)
            
            return output


if __name__ == "__main__":
    # Test Signal-Language Foundation Model
    import torch
    from argparse import Namespace
    
    def test_signal_language_fm():
        """Test Signal-Language Foundation Model."""
        print("Testing Signal-Language Foundation Model...")
        
        # Test configuration
        args = Namespace(
            input_dim=3,
            vocab_size=5000,
            hidden_dim=128,
            signal_layers=4,
            text_layers=3,
            num_heads=4,
            temperature=0.07,
            max_text_len=64,
            num_classes=4
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 64
        text_len = 32
        
        signals = torch.randn(batch_size, seq_len, args.input_dim)
        texts = torch.randint(0, args.vocab_size, (batch_size, text_len))
        attention_mask = torch.ones(batch_size, text_len)
        
        # Test contrastive mode
        with torch.no_grad():
            output_contrastive = model(signals, texts, attention_mask, mode='contrastive')
        
        print(f"Contrastive mode - Signals: {signals.shape}, Texts: {texts.shape}")
        print(f"Loss: {output_contrastive['loss'].item():.4f}")
        print(f"Signal features: {output_contrastive['signal_features'].shape}")
        print(f"Text features: {output_contrastive['text_features'].shape}")
        
        # Test downstream mode
        with torch.no_grad():
            output_downstream = model(signals, mode='downstream')
        
        print(f"Downstream mode - Signals: {signals.shape}, Output: {output_downstream.shape}")
        assert output_downstream.shape == (batch_size, args.num_classes)
        
        print("✅ Signal-Language Foundation Model tests passed!")
        return True
    
    test_signal_language_fm()
