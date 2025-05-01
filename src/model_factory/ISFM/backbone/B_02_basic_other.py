# TODO replace all the embedding
import torch.nn as nn

import math
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange



############################################################################################################
class MLPModel(nn.Module):
    def __init__(self, input_dim, num_classes, use_embedding=False, embedding_params=None):
        super(MLPModel, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            embedding_dim = self.embedding.output_dim
        else:
            embedding_dim = input_dim  # input_dim = L * C
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, L, C)
        if self.use_embedding:
            x = self.embedding(x)  # Shape: (B, num_patches, output_dim)
            x = x.mean(dim=1)  # Aggregate over patches
        else:
            x = x.view(x.size(0), -1)  # Flatten
        out = self.fc(x)
        return out
############################################################################################################
class CNNModel(nn.Module):
    def __init__(self, input_channels, num_classes, use_embedding=False, embedding_params=None):
        super(CNNModel, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            embedding_dim = self.embedding.output_dim
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        else:
            # Input channels = C (should be 1)
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_embedding:
            x = self.embedding(x)  # (B, num_patches, embedding_dim)
            x = x.permute(0, 2, 1)  # (B, embedding_dim, num_patches)
        else:
            x = x.permute(0, 2, 1)  # (B, C, L)
        x = self.conv(x)  # (B, 64, 1)
        x = x.squeeze(-1)  # (B, 64)
        out = self.fc(x)
        return out
############################################################################################################
class LSTMModel(nn.Module):
    def __init__(self, input_dim, num_classes, use_embedding=False, embedding_params=None):
        super(LSTMModel, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            input_dim = self.embedding.output_dim
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=64, num_layers=1, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_embedding:
            x = self.embedding(x)  # (B, num_patches, output_dim)
        # else x remains (B, L, C)
        output, (hn, cn) = self.lstm(x)  # hn: (num_layers, B, hidden_size)
        out = self.fc(hn[-1])  # Take the last layer's hidden state
        return out
############################################################################################################
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_classes, use_embedding=False, embedding_params=None):
        super(TransformerModel, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            input_dim = self.embedding.output_dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_embedding:
            x = self.embedding(x)  # (B, num_patches, input_dim)
            x = x.permute(1, 0, 2)  # (seq_len, batch_size, input_dim)
        else:
            x = x.permute(1, 0, 2)  # (L, B, C)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # (B, L, input_dim)
        x = x.mean(dim=0)  # (batch_size, input_dim)
        out = self.fc(x)
        return out



############################################################################################################


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.kernels = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i) 
            for i in range(num_kernels)
        ])
        if init_weight:
            self._initialize_weights()
            
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res = torch.stack([kernel(x) for kernel in self.kernels], dim=-1).mean(-1)
        return res


class TimesBlock(nn.Module):
    def __init__(self, seq_len=1024, top_k=3, d_model=64, d_ff=32, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        xf = torch.fft.rfft(x, dim=1)
        frequency = abs(xf).mean(0).mean(-1)
        frequency[0] = 0
        _, top = torch.topk(frequency, self.k)
        period = x.shape[1] // top
        res = []
        for i in range(self.k):
            p = period[i]
            padding = (p - (T % p)) if T % p != 0 else 0
            if padding > 0:
                out = F.pad(x, (0, 0, 0, padding), "constant", 0)
            else:
                out = x
            out = out.view(B, -1, p, N).permute(0, 3, 1, 2).contiguous()
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)[:, :T, :]
            res.append(out)
        res = torch.stack(res, dim=-1)
        weight = F.softmax(abs(xf).mean(-1)[:, top], dim=1).unsqueeze(1).unsqueeze(1)
        res = torch.sum(res * weight, -1) + x
        return res

def FeedForward(dim, ff_mult):
    dim_hidden = int(dim * ff_mult)
    return nn.Sequential(
        nn.Linear(dim, dim_hidden),
        nn.GELU(),
        nn.Linear(dim_hidden, dim)
)

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1
        self.tokenConv = nn.Conv1d(c_in, d_model, kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        nn.init.kaiming_normal_(self.tokenConv.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
    
class Timesnet(nn.Module):
    def __init__(self, channels, dim, lenth, depth, dropout, num_classes, use_embedding=False, embedding_params=None):
        super().__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            embedding_dim = self.embedding.output_dim
        else:
            self.embedding = TokenEmbedding(channels, dim)
            embedding_dim = dim

        self.layers = nn.ModuleList([
            nn.ModuleList([
                TimesBlock(seq_len=lenth, top_k=3, d_model=embedding_dim, d_ff=32, num_kernels=6),
                nn.LayerNorm(embedding_dim)
            ]) for _ in range(depth)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, mask=None):
        if self.use_embedding:
            x = self.embedding(x)
        else:
            x = self.embedding(x)
        for layer, norm in self.layers:
            x = layer(x)
            x = norm(x)
        x = x.mean(dim=1)
        out = self.classifier(x)
        return out
    
    
### Resnet-18

# 更新后的 ResNet1D，使用 RandomPatchMixer
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # 残差连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果尺寸不同，进行下采样
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=10, input_dim=1, use_embedding=False, embedding_params=None):
        super(ResNet1D, self).__init__()
        self.use_embedding = use_embedding
        if self.use_embedding:
            self.embedding = RandomPatchMixer(**embedding_params)
            in_channels = self.embedding.output_dim
            self.embedding_num_patches = self.embedding.num_patches
        else:
            in_channels = input_dim  # 输入维度，即通道数

        self.in_channels = 64
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])  # layers[0]=2
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # layers[1]=2
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # layers[2]=2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # layers[3]=2

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, L, C)
        if self.use_embedding:
            x = self.embedding(x)  # x: (B, num_patches, output_dim)
            # 将嵌入后的输出调整为 (B, C, L)
            x = rearrange(x, 'b p c -> b c p')  # 将 num_patches 视为序列长度 L
        else:
            x = x.permute(0, 2, 1)  # 转换为 (B, C, L)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)

        return out
    
    

__all__ = ['CNNModel', 'LSTMModel', 'MLPModel', 'TransformerModel', 'Timesnet', 'ResNet1D', 'get_resnet1d']