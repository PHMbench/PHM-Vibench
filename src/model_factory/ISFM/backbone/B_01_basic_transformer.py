import torch.nn as nn

import torch.nn.functional as F



class B_01_basic_transformer(nn.Module):
    def __init__(self, args):
        super(B_01_basic_transformer, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model= args.output_dim, nhead = args.num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
    def forward(self, x):
        x = self.transformer_encoder(x)
        return x

if __name__ == "__main__":
    # 创建配置
    import torch
    from argparse import Namespace

    class Config:
        def __init__(self):
            self.output_dim = 64  # 输出维度
            self.num_heads = 4  # 注意力头数
            self.num_layers = 2  # Transformer层数

    configs = Config()
    model = B_01_basic_transformer(configs)
    x_enc = torch.randn(32, 128, configs.output_dim)  # [B, L, D]
    output = model(x_enc)
    print(output.shape)  # 应该是 [B, L, D]
