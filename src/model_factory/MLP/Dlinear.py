import torch
import torch.nn as nn
from torch.nn import functional as F
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x
    
class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class B_04_Dlinear(nn.Module):
    """
    DLinear Backbone
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    """

    def __init__(self, configs, individual=False):
        """
        individual: Bool, whether shared model among different variates.
        """
        super(B_04_Dlinear, self).__init__()
        self.patch_size_L = configs.patch_size_L
        # Series decomposition block from Autoformer
        self.decompsition = series_decomp(65)
        self.individual = individual
        self.channels = configs.patch_size_C

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.patch_size_L, self.patch_size_L))
                self.Linear_Trend.append(nn.Linear(self.patch_size_L, self.patch_size_L))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
        else:
            self.Linear_Seasonal = nn.Linear(self.patch_size_L, self.patch_size_L)
            self.Linear_Trend = nn.Linear(self.patch_size_L, self.patch_size_L)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.patch_size_L) * torch.ones([self.patch_size_L, self.patch_size_L]))

    def forward(self, x):
        # x: [B, L, C]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)  # [B, C, L]
            
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.patch_size_L],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.patch_size_L],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
        output = seasonal_output + trend_output
        
        output = F.silu(output)
        
        return output.permute(0, 2, 1)  # [B, L, C]
    


if __name__ == "__main__":
# 创建配置
    import torch
    import numpy as np
    from argparse import Namespace
    class Config:
        def __init__(self):
            self.patch_size_L = 4096
            self.patch_size_C = 1  # 输入通道数/特征维度
            self.moving_avg = 125

    # DLinear模型测试
    def test_dlinear_backbone():
        # 创建配置
        configs = Config()
        
        # 创建模型实例
        model = Model(configs, individual=False)
        
        # 创建一批测试数据
        batch_size = 32
        x = torch.randn(batch_size, configs.patch_size_L, configs.patch_size_C)  # [B, L, C]
        
        # 运行模型
        output = model(x)
        
        # 验证输出形状
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        assert output.shape == (batch_size, configs.patch_size_L, configs.patch_size_C), "输出形状应为 [B, L, C]"
        print("测试通过！输入输出形状一致: [B, L, C]")
        
        return output

    test_dlinear_backbone()