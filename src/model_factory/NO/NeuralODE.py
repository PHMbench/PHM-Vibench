"""Neural ODE for continuous-time modeling of time-series data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable
import math


class ODEFunc(nn.Module):
    """ODE function network that defines the dynamics.
    
    Parameters
    ----------
    hidden_dim : int
        Hidden dimension
    num_layers : int
        Number of layers in the ODE function
    activation : str
        Activation function name
    dropout : float
        Dropout probability
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 3, 
                 activation: str = 'relu', dropout: float = 0.1):
        super(ODEFunc, self).__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU() if activation == 'relu' else nn.GELU(),
                    nn.Dropout(dropout)
                ])
        
        self.net = nn.Sequential(*layers)
        self.nfe = 0  # Number of function evaluations
    
    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of ODE function.
        
        Parameters
        ----------
        t : torch.Tensor
            Time tensor (scalar or vector)
        x : torch.Tensor
            State tensor of shape (B, hidden_dim) or (B, L, hidden_dim)
            
        Returns
        -------
        torch.Tensor
            Time derivative dx/dt of same shape as x
        """
        self.nfe += 1
        return self.net(x)


class ODESolver:
    """Simple Euler ODE solver for Neural ODE.
    
    Parameters
    ----------
    ode_func : nn.Module
        ODE function network
    step_size : float
        Integration step size
    """
    
    def __init__(self, ode_func: nn.Module, step_size: float = 0.1):
        self.ode_func = ode_func
        self.step_size = step_size
    
    def solve(self, x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """Solve ODE using Euler method.
        
        Parameters
        ----------
        x0 : torch.Tensor
            Initial state of shape (B, hidden_dim)
        t_span : torch.Tensor
            Time points to evaluate at, shape (num_steps,)
            
        Returns
        -------
        torch.Tensor
            Solution trajectory of shape (B, num_steps, hidden_dim)
        """
        trajectory = [x0]
        x = x0
        
        for i in range(len(t_span) - 1):
            dt = t_span[i + 1] - t_span[i]
            dx_dt = self.ode_func(t_span[i], x)
            x = x + dt * dx_dt
            trajectory.append(x)
        
        return torch.stack(trajectory, dim=1)  # (B, num_steps, hidden_dim)


class Model(nn.Module):
    """Neural ODE for continuous-time modeling of time-series.
    
    Neural ODEs model the continuous dynamics of hidden states using
    ordinary differential equations, enabling modeling of irregular
    time-series and continuous-time processes.
    
    Parameters
    ----------
    args : Namespace
        Configuration object containing:
        - input_dim : int, input feature dimension
        - hidden_dim : int, hidden dimension (default: 64)
        - output_dim : int, output dimension (default: input_dim)
        - ode_layers : int, number of layers in ODE function (default: 3)
        - step_size : float, ODE integration step size (default: 0.1)
        - num_steps : int, number of integration steps (default: 10)
        - activation : str, activation function (default: 'relu')
        - dropout : float, dropout probability (default: 0.1)
        - num_classes : int, number of output classes (for classification)
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
    Chen et al. "Neural Ordinary Differential Equations" NeurIPS 2018.
    Rubanova et al. "Latent ODEs for Irregularly-Sampled Time Series" NeurIPS 2019.
    Kidger et al. "Neural Controlled Differential Equations for Irregular Time Series" NeurIPS 2020.
    Adapted for time-series industrial signals with continuous-time dynamics modeling for irregular sampling patterns.
    """
    
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        
        # Extract parameters
        self.input_dim = args.input_dim
        self.hidden_dim = getattr(args, 'hidden_dim', 64)
        self.output_dim = getattr(args, 'output_dim', self.input_dim)
        self.ode_layers = getattr(args, 'ode_layers', 3)
        self.step_size = getattr(args, 'step_size', 0.1)
        self.num_steps = getattr(args, 'num_steps', 10)
        self.activation = getattr(args, 'activation', 'relu')
        self.dropout = getattr(args, 'dropout', 0.1)
        self.num_classes = getattr(args, 'num_classes', None)
        
        # Input encoder
        self.input_encoder = nn.Linear(self.input_dim, self.hidden_dim)
        
        # ODE function
        self.ode_func = ODEFunc(
            hidden_dim=self.hidden_dim,
            num_layers=self.ode_layers,
            activation=self.activation,
            dropout=self.dropout
        )
        
        # ODE solver
        self.solver = ODESolver(self.ode_func, self.step_size)
        
        # Output layers
        if self.num_classes is not None:
            # Classification: use final state
            self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
            self.task_type = 'classification'
        else:
            # Regression: decode all states
            self.output_decoder = nn.Linear(self.hidden_dim, self.output_dim)
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
        
        # Encode input to hidden space
        h = self.input_encoder(x)  # (B, L, hidden_dim)
        
        if self.task_type == 'classification':
            # For classification, use the last time step as initial condition
            h0 = h[:, -1, :]  # (B, hidden_dim)
            
            # Create time span for integration
            t_span = torch.linspace(0, 1, self.num_steps + 1, device=x.device)
            
            # Solve ODE
            trajectory = self.solver.solve(h0, t_span)  # (B, num_steps+1, hidden_dim)
            
            # Use final state for classification
            final_state = trajectory[:, -1, :]  # (B, hidden_dim)
            output = self.classifier(final_state)  # (B, num_classes)
            
        else:
            # For regression, process each time step
            outputs = []
            
            for t in range(seq_len):
                h_t = h[:, t, :]  # (B, hidden_dim)
                
                # Short integration for each time step
                t_span = torch.linspace(0, self.step_size, 3, device=x.device)
                trajectory = self.solver.solve(h_t, t_span)  # (B, 3, hidden_dim)
                
                # Use final integrated state
                h_integrated = trajectory[:, -1, :]  # (B, hidden_dim)
                output_t = self.output_decoder(h_integrated)  # (B, output_dim)
                outputs.append(output_t)
            
            output = torch.stack(outputs, dim=1)  # (B, L, output_dim)
        
        return output


if __name__ == "__main__":
    # Test Neural ODE
    import torch
    from argparse import Namespace
    
    def test_neural_ode():
        """Test Neural ODE with different configurations."""
        print("Testing Neural ODE...")
        
        # Test regression configuration
        args_reg = Namespace(
            input_dim=3,
            hidden_dim=32,
            output_dim=3,
            ode_layers=2,
            step_size=0.1,
            num_steps=5,
            activation='relu',
            dropout=0.1
        )
        
        model_reg = Model(args_reg)
        print(f"Regression model parameters: {sum(p.numel() for p in model_reg.parameters()):,}")
        
        # Test data
        batch_size = 4
        seq_len = 16  # Smaller for faster testing
        x = torch.randn(batch_size, seq_len, args_reg.input_dim)
        
        # Forward pass
        with torch.no_grad():
            output_reg = model_reg(x)
        
        print(f"Regression - Input: {x.shape}, Output: {output_reg.shape}")
        assert output_reg.shape == (batch_size, seq_len, args_reg.output_dim)
        
        # Test classification configuration
        args_cls = Namespace(
            input_dim=3,
            hidden_dim=32,
            ode_layers=2,
            step_size=0.1,
            num_steps=5,
            activation='relu',
            dropout=0.1,
            num_classes=4
        )
        
        model_cls = Model(args_cls)
        print(f"Classification model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")
        
        with torch.no_grad():
            output_cls = model_cls(x)
        
        print(f"Classification - Input: {x.shape}, Output: {output_cls.shape}")
        assert output_cls.shape == (batch_size, args_cls.num_classes)
        
        print("✅ Neural ODE tests passed!")
        return True
    
    test_neural_ode()
