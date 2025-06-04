import torch
from torch import nn

class Model(nn.Module):
    """Simple prototype network encoder."""
    def __init__(self, args_model, metadata=None):
        super().__init__()
        input_dim = getattr(args_model, 'input_dim', 4096)
        hidden_dim = getattr(args_model, 'hidden_dim', 128)
        embedding_dim = getattr(args_model, 'embedding_dim', 64)
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )

    def forward(self, x):
        return self.encoder(x)
