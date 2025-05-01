import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# import models.NLinear as DLinear

class B_03_FITS(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(B_03_FITS, self).__init__()
        self.seq_len = 4096# configs.patch_size_L
        self.pred_len = 0 # configs.pred_len
        self.individual = True # configs.individual
        self.channels = configs.patch_size_C

        self.dominance_freq = 65# configs.cut_freq # 720/24
        self.length_ratio = 1 # (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upcampling]
        # configs.pred_len=configs.seq_len+configs.pred_len
        # #self.Dlinear=DLinear.Model(configs)
        # configs.pred_len=self.pred_len


    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var=torch.var(x, dim=1, keepdim=True)+ 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:,self.dominance_freq:]=0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF
        # print(low_specx.permute(0,2,1))
        if self.individual:
            low_specxy_ = torch.zeros([low_specx.size(0),int(self.dominance_freq*self.length_ratio),low_specx.size(2)],dtype=low_specx.dtype).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:,:,i]=self.freq_upsampler[i](low_specx[:,:,i].permute(0,1)).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0,2,1)).permute(0,2,1)
        # print(low_specxy_)
        low_specxy = torch.zeros([low_specxy_.size(0),int((self.seq_len+self.pred_len)/2+1),low_specxy_.size(2)],dtype=low_specxy_.dtype).to(low_specxy_.device)
        low_specxy[:,0:low_specxy_.size(1),:]=low_specxy_ # zero padding
        low_xy=torch.fft.irfft(low_specxy, dim=1)
        low_xy=low_xy * self.length_ratio # compemsate the length change
        # dom_x=x-low_x
        
        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy=(low_xy) * torch.sqrt(x_var) +x_mean
        return xy # , low_xy* torch.sqrt(x_var)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # 创建配置类
    class Config:
        def __init__(self, seq_len, pred_len, enc_in, cut_freq, individual):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.enc_in = enc_in
            self.cut_freq = cut_freq
            self.individual = individual

    def test_fits_model():
        print("开始测试FITS模型...")
        
        # 基本参数
        batch_size = 16
        seq_len = 4096
        pred_len = 0
        channels = 7
        cut_freq = 100
        
        # 测试1：共享参数模式
        print("\n测试1：共享参数模式 (individual=False)")
        configs = Config(seq_len, pred_len, channels, cut_freq, False)
        model = Model(configs)
        
        # 创建随机输入
        x = torch.randn(batch_size, seq_len, channels)
        
        # 前向传播
        output, low_freq_output = model(x)
        
        # 检查输出
        print(f"输入形状: {x.shape}")
        print(f"输出形状: {output.shape}")
        print(f"预期输出形状: [{batch_size}, {seq_len + pred_len}, {channels}]")
        
        assert output.shape == (batch_size, seq_len + pred_len, channels), "输出形状错误！"
        assert low_freq_output.shape == (batch_size, seq_len + pred_len, channels), "低频输出形状错误！"
        assert not torch.isnan(output).any(), "输出包含NaN值！"
        
        # 测试2：独立参数模式
        print("\n测试2：独立参数模式 (individual=True)")
        configs = Config(seq_len, pred_len, channels, cut_freq, True)
        model = Model(configs)
        
        # 前向传播
        output, low_freq_output = model(x)
        
        # 检查输出
        print(f"输出形状: {output.shape}")
        assert output.shape == (batch_size, seq_len + pred_len, channels), "输出形状错误！"
        assert not torch.isnan(output).any(), "输出包含NaN值！"
        
        # 测试3：正弦波预测
        print("\n测试3：正弦波预测")
        configs = Config(seq_len, pred_len, 1, 10, False)  # 单通道，较小的截断频率
        model = Model(configs)
        
        # 创建正弦波输入
        t = np.linspace(0, 4*np.pi, seq_len)
        sine_wave = np.sin(t).reshape(-1, 1)
        x = torch.FloatTensor(sine_wave).unsqueeze(0)  # [1, seq_len, 1]
        
        # 前向传播
        output, _ = model(x)
        
        # 检查输出并计算MSE
        mse_input_region = torch.mean((output[0, :seq_len, 0] - x[0, :, 0])**2).item()
        print(f"输入区域MSE: {mse_input_region:.6f}")
        print(f"输出总长度: {output.shape[1]}")
        
        # # 可视化预测结果
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(seq_len), x[0, :, 0].numpy(), 'b-', label='输入序列')
        # plt.plot(range(seq_len, seq_len + pred_len), output[0, seq_len:, 0].detach().numpy(), 'r-', label='预测序列')
        # plt.plot(range(seq_len), output[0, :seq_len, 0].detach().numpy(), 'g--', label='重构序列')
        # plt.axvline(x=seq_len-1, color='k', linestyle='--')
        # plt.legend()
        # plt.title('FITS模型：正弦波预测')
        # plt.savefig('fits_sine_prediction.png')
        # print("预测图表已保存为'fits_sine_prediction.png'")
        
        print("\n所有测试通过！")
    test_fits_model()