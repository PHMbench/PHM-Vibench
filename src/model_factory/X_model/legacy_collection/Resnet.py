
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import math
from einops import rearrange
import numpy as np

def conv3x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Morlet(p):
    C = pow(math.pi, 0.25)
    # p = 0.03 * p
    y = C * torch.exp(-torch.pow(p, 2) / 2) * torch.cos(2 * math.pi * p)
    return y

def sinc(band, t_right):
    y_right = torch.sin(2 * math.pi * band * t_right) / ((2 * math.pi * band * t_right) + 1e-6)
    y_left = torch.flip(y_right, [0])
    y = torch.cat([y_left, torch.ones(1).to(t_right.device), y_right])
    return y

class SincConv_fast(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        if in_channels != 1:
            raise ValueError(f"SincConv only supports one input channel (here, in_channels = {in_channels})")

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

    def forward(self, waveforms):
        half_kernel = self.kernel_size // 2
        time_disc = torch.linspace(-half_kernel, half_kernel, steps=self.kernel_size).to(waveforms.device)
        self.a_ = self.a_.to(waveforms.device)
        self.b_ = self.b_.to(waveforms.device)
        
        filters = []
        for i in range(self.out_channels):
            band = self.a_[i]
            t_right = time_disc - self.b_[i]
            filter = sinc(band, t_right)
            filters.append(filter)

        filters = torch.stack(filters)
        self.filters = filters.view(self.out_channels, 1, -1)

        return F.conv1d(waveforms, self.filters, stride=1, padding=half_kernel, dilation=1, bias=None, groups=1)


class Morlet_fast(nn.Module):

    def __init__(self, num_classs, kernel_size, in_channels=1):

        super(Morlet_fast, self).__init__()

        if in_channels != 1:

            msg = "MexhConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.num_classs = num_classs
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, num_classs)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, num_classs)).view(-1, 1)
        
        # self.register_parameter('param_a', self.a_) # should
        # self.register_parameter('param_b', self.b_)

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right - self.b_ / self.a_
        p2 = time_disc_left - self.b_ / self.a_
        
        p1 = p1.to(waveforms.device)
        p2 = p2.to(waveforms.device)

        Morlet_right = Morlet(p1)
        Morlet_left = Morlet(p2)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.num_classs, 1, self.kernel_size).to(waveforms.device)

        return F.conv1d(waveforms, self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1)


class SincConv_multiple_channel(nn.Module):
    def __init__(self, out_channels, kernel_size, in_channels=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size % 2 == 0:
            self.kernel_size += 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)
        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

        self.channel_linear = conv1x1(int(in_channels * out_channels), 64) # 6*10 ， 64  

    def forward(self, waveforms):
        half_kernel = self.kernel_size // 2
        time_disc = torch.linspace(-half_kernel, half_kernel, steps=self.kernel_size).to(waveforms.device)
        self.a_ = self.a_.to(waveforms.device)
        self.b_ = self.b_.to(waveforms.device)
        
        filters = []
        for i in range(self.out_channels):
            band = self.a_[i]
            t_right = time_disc - self.b_[i]
            filter = sinc(band, t_right)
            filters.append(filter)

        filters = torch.stack(filters)
        self.filters = filters.view(self.out_channels, 1, -1)

        output = []
        for i in range(self.in_channels):
            output.append(F.conv1d(waveforms[:, i:i+1], self.filters, stride=1, padding=half_kernel, dilation=1, bias=None, groups=1))

        res = torch.cat(output, dim=1)
        res = self.channel_linear(res)

        return res


class Morlet_multiple_channel(nn.Module):

    def __init__(self, out_channels, kernel_size, in_channels=1):

        super(Morlet_multiple_channel, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size - 1

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.a_ = nn.Parameter(torch.linspace(1, 10, out_channels)).view(-1, 1)

        self.b_ = nn.Parameter(torch.linspace(0, 10, out_channels)).view(-1, 1)

        self.channel_linear = conv1x1(int(in_channels * out_channels), 64) # 6*10 ， 64  

    def forward(self, waveforms):

        time_disc_right = torch.linspace(0, (self.kernel_size / 2) - 1,
                                         steps=int((self.kernel_size / 2)))

        time_disc_left = torch.linspace(-(self.kernel_size / 2) + 1, -1,
                                        steps=int((self.kernel_size / 2)))

        p1 = time_disc_right - self.b_ / self.a_
        p2 = time_disc_left - self.b_ / self.a_

        Morlet_right = Morlet(p1).to(waveforms.device)
        Morlet_left = Morlet(p2).to(waveforms.device)

        Morlet_filter = torch.cat([Morlet_left, Morlet_right], dim=1)  # 40x1x250

        self.filters = (Morlet_filter).view(self.out_channels, 1, self.kernel_size).to(waveforms.device)# .cuda()

        output = []
        for i in range(self.in_channels):
            output.append(F.conv1d(waveforms[:, i:i+1], self.filters, stride=1, padding=1, dilation=1, bias=None, groups=1))

        res = self.channel_linear(torch.cat(output, dim=1))

        return res


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = conv3x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False, first_kernel = 'conv'):
        super(ResNet, self).__init__()
        
        FIRST_CONV_DICT = {
            'Morlet': Morlet_fast(64, 16),
            'Sinc': SincConv_fast(64, 16),
            'conv': nn.Conv1d(in_channel, 64, kernel_size=7, stride=2, padding=3, bias=False),
            'Sinc_m': SincConv_multiple_channel(int(64//in_channel), 16, in_channel),
            'Morlet_m': Morlet_multiple_channel(int(64//in_channel), 16, in_channel),
        }
        
        self.inplanes = 64
        self.conv1 = FIRST_CONV_DICT[first_kernel]
        
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        
        x = rearrange(x, 'b c n -> b n c')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
