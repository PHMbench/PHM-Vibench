import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    """
    1D傅里叶谱卷积层。
    通过对输入信号进行FFT，在频域中应用线性变换，然后通过IFFT变换回时域。
    """
    def __init__(self, in_channels, out_channels, modes):
        """
        :param in_channels: 输入通道数
        :param out_channels: 输出通道数
        :param modes: 要保留的傅里叶模式数。只使用低频模式。
        """
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes

        # 缩放因子，用于保持梯度幅度稳定
        self.scale = (1 / (in_channels * out_channels))
        # 学习权重，形状为 (in_channels, out_channels, modes)，使用复数
        self.weights = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes, dtype=torch.cfloat))

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (B, C, L)
        :return: 经过谱卷积后的张量
        """
        # x - b, c, l
        B, C, L = x.shape
        out_ft = torch.zeros(B, self.out_channels, L // 2 + 1, device=x.device, dtype=torch.cfloat)
        x_ft = torch.fft.rfft(x)
        # Fix for RuntimeError: Slice weights to match input channels C
        # This handles cases where the input feature dimension does not match
        # the dimension the model was initialized with.
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modes], self.weights[:C, :, :])

        x = torch.fft.irfft(out_ft, n=L)
        return x


class Model(nn.Module):
    """
    1D 傅里叶神经算子 (FNO) 主模型
    """
    def __init__(self, modes, width, n_layers=4, channels=1):
        """
        :param modes: 要保留的傅里叶模式数
        :param width: 隐藏层的通道数 (宽度)
        :param n_layers: FNO层的数量
        """
        super(Model, self).__init__()
        self.modes = modes
        self.width = width
        self.n_layers = n_layers
        self.channels = channels

        # 1. 输入层：将输入通道提升到指定的宽度 (width)
        self.fc0 = nn.Linear(self.channels, self.width) # 假设输入通道数为1，可以按需修改

        # 2. FNO谱卷积层和普通卷积层
        self.spectral_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.spectral_layers.append(SpectralConv1d(self.width, self.width, self.modes))
            self.conv_layers.append(nn.Conv1d(self.width, self.width, 1))

        # 3. 输出层：将宽度映射回输出通道数
        self.fc1 = nn.Linear(self.width, self.channels) # 假设输出通道数为1

    def forward(self, x):
        """
        :param x: 输入张量，形状为 (B, L, C)
        :return: 输出张量，形状为 (B, L, C)
        """
        # 输入张量形状: (B, L, C)
        
        # 提升到隐藏维度
        x = self.fc0(x) # (B, L, C) -> (B, L, width)

        # FNO要求输入形状为 (B, C, L)，所以需要进行维度重排
        x = x.permute(0, 2, 1) # (B, L, width) -> (B, width, L)

        # 迭代应用FNO层
        for i in range(self.n_layers):
            x1 = self.spectral_layers[i](x)
            x2 = self.conv_layers[i](x)
            x = x1 + x2 # 残差连接
            x = F.gelu(x) # 应用激活函数

        # 将维度重排回 (B, L, C)
        x = x.permute(0, 2, 1) # (B, width, L) -> (B, L, width)

        # 映射回输出维度
        x = self.fc1(x)  # (B, L, width) -> (B, L, C)
        return x
