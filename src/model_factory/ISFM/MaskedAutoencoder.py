"""Masked Autoencoder for Industrial Signal Foundation Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any
import math


class PatchEmbedding(nn.Module):
    """Patch embedding for time-series data.
    
    Parameters
    ----------
    input_dim : int
        Input feature dimension
    patch_size : int
        Size of each patch
    embed_dim : int
        Embedding dimension
    """
    
    def __init__(self, input_dim: int, patch_size: int, embed_dim: int):
        super(PatchEmbedding, self).__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Linear projection for patches
        self.projection = nn.Linear(patch_size * input_dim, embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
            
        Returns
        -------
        torch.Tensor
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, L, C = x.shape
        
        # Ensure sequence length is divisible by patch size
        if L % self.patch_size != 0:
            pad_len = self.patch_size - (L % self.patch_size)
            x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            L = x.size(1)
        
        # Create patches
        num_patches = L // self.patch_size
        patches = x.view(B, num_patches, self.patch_size * C)  # (B, num_patches, patch_size * C)
        
        # Project patches
        embeddings = self.projection(patches)  # (B, num_patches, embed_dim)
        
        return embeddings


class PositionalEncoding(nn.Module):
    """Learnable positional encoding."""
    
    def __init__(self, num_patches: int, embed_dim: int):
        super(PositionalEncoding, self).__init__()
        
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding."""
        return x + self.pos_embed[:, :x.size(1), :]


class MaskedAutoencoderEncoder(nn.Module):
    """Encoder for Masked Autoencoder.
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension
    num_layers : int
        Number of transformer layers
    num_heads : int
        Number of attention heads
    mlp_ratio : float
        MLP expansion ratio
    dropout : float
        Dropout probability
    """
    
    def __init__(self, embed_dim: int, num_layers: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(MaskedAutoencoderEncoder, self).__init__()
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (B, N, embed_dim)
        mask : torch.Tensor, optional
            Attention mask
            
        Returns
        -------
        torch.Tensor
            Encoded features of shape (B, N, embed_dim)
        """
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.norm(x)
        return x


class MaskedAutoencoderDecoder(nn.Module):
    """Decoder for Masked Autoencoder.
    
    Parameters
    ----------
    embed_dim : int
        Encoder embedding dimension
    decoder_embed_dim : int
        Decoder embedding dimension
    num_layers : int
        Number of decoder layers
    num_heads : int
        Number of attention heads
    mlp_ratio : float
        MLP expansion ratio
    dropout : float
        Dropout probability
    """
    
    def __init__(self, embed_dim: int, decoder_embed_dim: int, num_layers: int = 8,
                 num_heads: int = 16, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(MaskedAutoencoderDecoder, self).__init__()
        
        self.decoder_embed_dim = decoder_embed_dim
        
        # Projection from encoder to decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=num_heads,
            dim_feedforward=int(decoder_embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(decoder_embed_dim)
    
    def forward(self, x: torch.Tensor, ids_restore: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Encoded visible patches of shape (B, N_vis, embed_dim)
        ids_restore : torch.Tensor
            Indices to restore original order
            
        Returns
        -------
        torch.Tensor
            Decoded features of shape (B, N, decoder_embed_dim)
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # No cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # Unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # Append cls token
        
        # Apply transformer
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class Model(nn.Module):
    """Masked Autoencoder for Industrial Signal Foundation Model.
    
    Implements masked autoencoding for self-supervised learning on
    industrial time-series signals using patch-based masking.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - patch_size : int, size of each patch (default: 16)
        - embed_dim : int, encoder embedding dimension (default: 768)
        - decoder_embed_dim : int, decoder embedding dimension (default: 512)
        - num_layers : int, number of encoder layers (default: 12)
        - decoder_num_layers : int, number of decoder layers (default: 8)
        - num_heads : int, number of attention heads (default: 12)
        - decoder_num_heads : int, number of decoder heads (default: 16)
        - mask_ratio : float, masking ratio (default: 0.75)
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for downstream tasks)
        - output_dim : int, output dimension (for regression tasks)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For pretraining: reconstructed patches
        For downstream tasks: depends on task type
        
    References
    ----------
    He et al. "Masked Autoencoders Are Scalable Vision Learners" CVPR 2022.
    Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" NAACL 2019.
    Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" ICLR 2021.
    Adapted for time-series industrial signals with patch-based masking and reconstruction for self-supervised learning.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.patch_size = getattr(args, 'patch_size', 16)
        self.embed_dim = getattr(args, 'embed_dim', 768)
        self.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
        self.num_layers = getattr(args, 'num_layers', 12)
        self.decoder_num_layers = getattr(args, 'decoder_num_layers', 8)
        self.num_heads = getattr(args, 'num_heads', 12)
        self.decoder_num_heads = getattr(args, 'decoder_num_heads', 16)
        self.mask_ratio = getattr(args, 'mask_ratio', 0.75)
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Calculate number of patches (assuming max sequence length)
        self.max_seq_len = getattr(args, 'max_seq_len', 1024)
        self.num_patches = self.max_seq_len // self.patch_size
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(self.input_dim, self.patch_size, self.embed_dim)
        
        # Positional encoding
        self.pos_embed = PositionalEncoding(self.num_patches + 1, self.embed_dim)  # +1 for cls token
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Encoder
        self.encoder = MaskedAutoencoderEncoder(
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # Decoder
        self.decoder = MaskedAutoencoderDecoder(
            embed_dim=self.embed_dim,
            decoder_embed_dim=self.decoder_embed_dim,
            num_layers=self.decoder_num_layers,
            num_heads=self.decoder_num_heads,
            dropout=self.dropout
        )
        
        # Decoder positional encoding (separate from encoder)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, self.decoder_embed_dim) * 0.02)
        
        # Prediction head for reconstruction
        self.decoder_pred = nn.Linear(self.decoder_embed_dim, self.patch_size * self.input_dim)
        
        # Downstream task heads
        if self.num_classes is not None:
            # Classification head
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression head
            self.regressor = nn.Linear(self.embed_dim, self.output_dim)
            self.task_type = 'regression'
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform random masking by per-sample shuffling.
        
        Parameters
        ----------
        x : torch.Tensor
            Input sequence of shape (B, N, D)
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Masked sequence, mask, and restore indices
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)  # Noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None, 
                mode: str = 'pretrain') -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
        mode : str
            Forward mode: 'pretrain' for masked autoencoding,
            'downstream' for supervised tasks
            
        Returns
        -------
        torch.Tensor or dict
            Output depends on mode and task type
        """
        if mode == 'pretrain':
            # Patch embedding
            x = self.patch_embed(x)  # (B, N, embed_dim)
            
            # Add positional encoding
            x = self.pos_embed(x)
            
            # Add class token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
            # Random masking
            x_masked, mask, ids_restore = self.random_masking(x)
            
            # Encoder
            latent = self.encoder(x_masked)
            
            # Decoder
            pred = self.decoder(latent, ids_restore)

            # Add decoder positional encoding
            if pred.size(1) <= self.decoder_pos_embed.size(1):
                pred = pred + self.decoder_pos_embed[:, :pred.size(1), :]
            
            # Remove class token
            pred = pred[:, 1:, :]
            
            # Prediction
            pred = self.decoder_pred(pred)  # (B, N, patch_size * input_dim)
            
            return {
                'pred': pred,
                'mask': mask,
                'latent': latent
            }
        
        else:  # downstream mode
            # Patch embedding
            x = self.patch_embed(x)  # (B, N, embed_dim)
            
            # Add positional encoding
            x = self.pos_embed(x)
            
            # Add class token
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            
            # Encoder (no masking)
            features = self.encoder(x)
            
            # Use class token for downstream tasks
            cls_features = features[:, 0, :]  # (B, embed_dim)
            
            if self.task_type == 'classification':
                output = self.classifier(cls_features)
            else:
                output = self.regressor(cls_features)
            
            return output


if __name__ == "__main__":
    # Test Masked Autoencoder model
    import torch
    from argparse import Namespace
    
    def test_masked_autoencoder():
        """Test Masked Autoencoder model."""
        print("Testing Masked Autoencoder model...")
        
        # Test configuration
        args = Namespace(
            input_dim=3,
            patch_size=8,
            embed_dim=256,
            decoder_embed_dim=128,
            num_layers=6,
            decoder_num_layers=4,
            num_heads=8,
            decoder_num_heads=8,
            mask_ratio=0.75,
            dropout=0.1,
            max_seq_len=128,
            num_classes=5
        )
        
        model = Model(args)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 128
        x = torch.randn(batch_size, seq_len, args.input_dim)
        
        # Test pretrain mode
        with torch.no_grad():
            output_pretrain = model(x, mode='pretrain')
        
        print(f"Pretrain mode - Input: {x.shape}")
        print(f"Prediction: {output_pretrain['pred'].shape}")
        print(f"Mask: {output_pretrain['mask'].shape}")
        print(f"Latent: {output_pretrain['latent'].shape}")
        
        # Test downstream mode
        with torch.no_grad():
            output_downstream = model(x, mode='downstream')
        
        print(f"Downstream mode - Input: {x.shape}, Output: {output_downstream.shape}")
        assert output_downstream.shape == (batch_size, args.num_classes)
        
        print("✅ Masked Autoencoder model tests passed!")
        return True
    
    test_masked_autoencoder()
