"""ConvLSTM for spatial-temporal sequence modeling."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell.
    
    Parameters
    ----------
    input_dim : int
        Number of input channels
    hidden_dim : int
        Number of hidden channels
    kernel_size : int or tuple
        Size of the convolutional kernel
    bias : bool
        Whether to add bias
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        # Convolutional layers for input-to-state and state-to-state transitions
        self.conv = nn.Conv1d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,  # i, f, o, g gates
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
    
    def forward(self, input_tensor: torch.Tensor, cur_state: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of ConvLSTM cell.
        
        Parameters
        ----------
        input_tensor : torch.Tensor
            Input tensor of shape (B, input_dim, L)
        cur_state : Tuple[torch.Tensor, torch.Tensor]
            Current hidden and cell states
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            New hidden and cell states
        """
        h_cur, c_cur = cur_state
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # (B, input_dim + hidden_dim, L)
        
        # Apply convolution
        combined_conv = self.conv(combined)  # (B, 4 * hidden_dim, L)
        
        # Split into gates
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        # Apply gate functions
        i = torch.sigmoid(cc_i)  # Input gate
        f = torch.sigmoid(cc_f)  # Forget gate
        o = torch.sigmoid(cc_o)  # Output gate
        g = torch.tanh(cc_g)     # Candidate values
        
        # Update cell state
        c_next = f * c_cur + i * g
        
        # Update hidden state
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size: int, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden and cell states.
        
        Parameters
        ----------
        batch_size : int
            Batch size
        seq_len : int
            Sequence length
        device : torch.device
            Device to place tensors on
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Initial hidden and cell states
        """
        h = torch.zeros(batch_size, self.hidden_dim, seq_len, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, seq_len, device=device)
        return h, c


class Model(nn.Module):
    """ConvLSTM for spatial-temporal sequence modeling.
    
    Combines convolutional operations with LSTM to capture both
    spatial and temporal dependencies in sequence data.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dims : list, hidden dimensions for each layer (default: [64, 64])
        - kernel_sizes : list, kernel sizes for each layer (default: [3, 3])
        - num_layers : int, number of ConvLSTM layers (default: 2)
        - dropout : float, dropout probability (default: 0.1)
        - return_all_layers : bool, whether to return all layer outputs (default: False)
        - num_classes : int, number of output classes (for classification)
        - output_dim : int, output dimension (for regression, default: input_dim)
    metadata : Any, optional
        Dataset metadata
        
    Input Shape
    -----------
    x : torch.Tensor
        Input tensor of shape (batch_size, seq_len, input_dim)
        
    Output Shape
    ------------
    torch.Tensor
        For classification: (batch_size, num_classes)
        For regression: (batch_size, seq_len, output_dim)
        
    References
    ----------
    Shi et al. "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting" NIPS 2015.
    Hochreiter and Schmidhuber "Long Short-Term Memory" Neural Computation 1997.
    LeCun et al. "Gradient-based learning applied to document recognition" Proceedings of the IEEE 1998.
    Adapted for time-series industrial signals with convolutional operations for spatial-temporal pattern modeling.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dims = getattr(args, 'hidden_dims', [64, 64])
        self.kernel_sizes = getattr(args, 'kernel_sizes', [3, 3])
        self.num_layers = getattr(args, 'num_layers', len(self.hidden_dims))
        self.dropout = getattr(args, 'dropout', 0.1)
        self.return_all_layers = getattr(args, 'return_all_layers', False)
        self.num_classes = getattr(args, 'num_classes', None)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        
        # Ensure consistent dimensions
        if len(self.hidden_dims) != self.num_layers:
            self.hidden_dims = self.hidden_dims[:self.num_layers] + [self.hidden_dims[-1]] * (self.num_layers - len(self.hidden_dims))
        if len(self.kernel_sizes) != self.num_layers:
            self.kernel_sizes = self.kernel_sizes[:self.num_layers] + [self.kernel_sizes[-1]] * (self.num_layers - len(self.kernel_sizes))
        
        # Build ConvLSTM layers
        self.cell_list = nn.ModuleList()
        for i in range(self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            self.cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=self.hidden_dims[i],
                    kernel_size=self.kernel_sizes[i]
                )
            )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Output layers
        final_hidden_dim = self.hidden_dims[-1]
        
        if self.num_classes is not None:
            # Classification: global pooling + classifier
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(final_hidden_dim, final_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(final_hidden_dim // 2, self.num_classes)
            )
            self.task_type = 'classification'
        else:
            # Regression: sequence-to-sequence
            self.output_projection = nn.Conv1d(final_hidden_dim, self.output_dim, kernel_size=1)
            self.task_type = 'regression'
    
    def forward(self, x: torch.Tensor, data_id=None, task_id=None) -> torch.Tensor:
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, L, input_dim)
        data_id : Any, optional
            Data identifier (unused)
        task_id : Any, optional
            Task identifier (unused)
            
        Returns
        -------
        torch.Tensor
            Output tensor shape depends on task type:
            - Classification: (B, num_classes)
            - Regression: (B, L, output_dim)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Transpose for convolution: (B, L, C) -> (B, C, L)
        x = x.transpose(1, 2)
        
        # Initialize hidden states for all layers
        hidden_state = []
        for i in range(self.num_layers):
            h, c = self.cell_list[i].init_hidden(batch_size, seq_len, device)
            hidden_state.append((h, c))
        
        # Process through ConvLSTM layers
        layer_output_list = []
        cur_layer_input = x
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            
            # Forward through ConvLSTM cell
            h, c = self.cell_list[layer_idx](cur_layer_input, (h, c))
            
            # Apply dropout
            h = self.dropout_layer(h)
            
            layer_output_list.append(h)
            cur_layer_input = h
        
        # Get final layer output
        final_output = layer_output_list[-1]  # (B, hidden_dim, L)
        
        if self.task_type == 'classification':
            # Classification
            output = self.classifier(final_output)  # (B, num_classes)
        else:
            # Regression: project to output dimension
            output = self.output_projection(final_output)  # (B, output_dim, L)
            output = output.transpose(1, 2)  # (B, L, output_dim)
        
        return output


if __name__ == "__main__":
    # Test ConvLSTM
    import torch
    from argparse import Namespace
    
    def test_conv_lstm():
        """Test ConvLSTM with different configurations."""
        print("Testing ConvLSTM...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=4,
            hidden_dims=[32, 64],
            kernel_sizes=[3, 5],
            num_layers=2,
            dropout=0.1,
            return_all_layers=False,
            output_dim=4
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 32
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=4,
            hidden_dims=[32, 64],
            kernel_sizes=[3, 5],
            num_layers=2,
            dropout=0.1,
            return_all_layers=False,
            num_classes=6
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ ConvLSTM tests passed!")
        return True
    
    test_conv_lstm()
