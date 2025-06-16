import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, configs):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len),
            nn.Dropout(configs.dropout)
        )

        self.channel = nn.Sequential(
            nn.Linear(configs.enc_in, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.enc_in),
            nn.Dropout(configs.dropout)
        )

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


class B_07_TSMixer(nn.Module):
    def __init__(self, configs):
        super(B_07_TSMixer, self).__init__()
        self.layer = configs.e_layers
        self.model = nn.ModuleList([ResBlock(configs)
                                    for _ in range(configs.e_layers)])

    def forward(self, x_enc):
        # x_enc: [B, L, D]
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        return x_enc

if __name__ == "__main__":
    # 创建配置
    import torch
    from argparse import Namespace

    class Config:
        def __init__(self):
            self.seq_len = 128
            self.enc_in = 64
            self.d_model = 128
            self.e_layers = 3
            self.dropout = 0.1

    configs = Config()
    model = B_07_TSMixer(configs)
    x_enc = torch.randn(32, configs.seq_len, configs.enc_in)  # [B, L, D]
    output = model(x_enc)
    print(output.shape)  # 应该是 [B, L, D]
