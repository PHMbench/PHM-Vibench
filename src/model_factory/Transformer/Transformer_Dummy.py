import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """Simplified transformer encoder for sequence classification.

    Parameters
    ----------
    args : Namespace
        Should define ``input_dim``, ``hidden_dim``, ``num_heads``,
        ``num_layers`` and ``num_classes``.
    metadata : Any, optional
        Unused.

    Notes
    -----
    Input tensor is ``(B, L, C)`` and the output is ``(B, num_classes)``.
    """

    def __init__(self, args, metadata=None):
        super(Model, self).__init__()

        input_dim = args.input_dim
        hidden_dim= args.hidden_dim
        num_heads= args.num_heads
        num_layers= args.num_layers
        num_classes= args.num_classes
        dropout= args.dropout
        
        self.input_projection = nn.Linear(input_dim, hidden_dim) # shape [C, hidden_dim] wight 
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads,
            dim_feedforward=hidden_dim*4,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x,data_id = None,task_id = None):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape [L, C] where L is sequence length and C is feature dimension
            
        Returns:
            Output tensor of shape [L, num_classes]
        """
        # If input doesn't have batch dimension, add it
        if x.dim() == 2:
            x = x.unsqueeze(2)  # [L, C] -> [L, 1, C]
            single_sample = True
        else:
            single_sample = False
            # x = x.permute(1, 0, 2)  # [B, L, C] -> [L, B, C]
        x = x.float()  # Ensure input is float
        # Project input to hidden dimension
        
        if x.shape[2] != self.input_projection.in_features: # Traditional expect the unique dimension of input
            x = torch.nn.functional.pad(x, (0, self.input_projection.in_features - x.shape[2]), mode='constant', value=0)
        
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)  # Global average pooling
        # Apply classification head
        output = self.classifier(x)
        
        # Remove batch dimension if input didn't have one
        if single_sample:
            output = output.squeeze(1)
            
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(max_len, 1, d_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pos_encoding[:x.size(0), :]
        return self.dropout(x)
