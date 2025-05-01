import torch.nn as nn

import torch.nn.functional as F



class B_01_basic_transformer(nn.Module):
    def __init__(self, args):
        super(B_01_basic_transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model= args.output_dim, nhead = args.nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

