from numpy import identity
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fs
import sys
import numpy as np
import os
from einops import rearrange
# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 将当前目录添加到 sys.path
sys.path.append(current_dir)
# from utils import convlutional_operator, signal_filter_, FRE # TODO
# base class for signal processing modules
class SignalProcessingBase(torch.nn.Module):
    def __init__(self, args):
        super(SignalProcessingBase, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.device = args.device
        self.to(self.device)


    def forward(self, x):
        # x should be B,L,C first
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device) 
        output = self.forward(test_input)
        assert output.shape == (2, self.out_dim, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_dim, self.out_channels)}"

class SignalProcessingBase2Arity(torch.nn.Module):
    def __init__(self, args):
        super(SignalProcessingBase2Arity, self).__init__()
        self.args = args
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels //2
        self.device = args.device
        self.to(self.device)

    def split_input(self, x):
        # 拆分输入信号
        half_channels = x.shape[-1] // 2
        x1 = x[:, :, :half_channels]
        x2 = x[:, :, half_channels:]
        return x1, x2
    
    def repeat_output(self, x):
        # 合并输出信号
        return torch.cat((x, x), dim=-1)

    def forward(self, x):
        x1, x2 = self.split_input(x)
        x = self.operation(x1, x2)
        x = self.repeat_output(x)
        return x

    def operation(self, x1, x2):
        raise NotImplementedError("This method should be implemented by subclass.")
    
    def test_forward(self):
        test_input = torch.randn(2, self.in_dim, self.in_channels).to(self.device)
        output = self.forward(test_input)
        assert output.shape == (2, self.out_dim, self.out_channels), f"\
        input shape is {test_input.shape}, \n\
        Output shape is {output.shape}, \n\
        expected {(2, self.out_dim, self.out_channels)}"


class SignalProcessingModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_dict):
        super(SignalProcessingModuleDict, self).__init__(module_dict)

    def forward(self, x, key):
        if key in self:
            return self[key](x)
        else:
            raise KeyError(f"No signal processing module found for key: {key}")
        
    def test_forward(self):
        for key in self.keys():
            self[key].test_forward()

# TODO
# SignalProcessingModuleDict_2_arity
# subclass for FFT module
# 1
class FFTSignalProcessing(SignalProcessingBase):
    '''
    args:
    input_dim: 输入信号的长度
    '''
    def __init__(self, args):
        # FFT 不改变通道数，只改变长度，因此 output_dim = input_dim // 2
        super(FFTSignalProcessing, self).__init__(args)
        self.name = "FFT"
    def forward(self, x):
        # 假设 x 的形状为 [B, L, C]
        fft_result = torch.fft.rfft(x, dim=1, norm='ortho')  # 对长度L进行FFT
        return fft_result

# 2 ############################################## subclass for Hilbert module###############################################  
class HilbertTransform(SignalProcessingBase):
    def __init__(self, args):
        # 希尔伯特变换不改变维度
        super(HilbertTransform, self).__init__(args)
        self.name = "HT"
    def forward(self, x):
        # N = x.shape[-1]
        x = rearrange(x, 'b l c -> b c l')
        N = x.shape[-1] # 对length进行Hilbert变换
        Xf = torch.fft.fft(x, dim=2)  # 对length维度执行FFT
        if (N % 2 == 0):
            Xf[..., 1:N // 2] *= 2
            Xf[..., N // 2 + 1:] = 0
        else:
            Xf[..., 1:(N + 1) // 2] *= 2
            Xf[..., (N + 1) // 2:] = 0
        Hilbert_x = torch.fft.ifft(Xf, dim=2).abs()
        x = rearrange(Hilbert_x, 'b c l -> b l c')
        return x
    #%%
# 3 ##############################################  WaveFilters module  ############################################## 
class WaveFilters(SignalProcessingBase): # TII中的实现
    def __init__(self, args):
        super(WaveFilters, self).__init__(args)

        self.name = "WF"
        self.device = args.device
        self.to(self.device)
        in_channels = args.scale # large enough to avoid setting

        
        # 初始化频率和带宽参数
        self.f_c = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        self.f_b = nn.Parameter(torch.empty(1, 1,in_channels, device=self.device))
        
        # 自定义参数初始化
        self.initialize_parameters()
        
        # 预生成滤波器

    def initialize_parameters(self):
        # 根据提供的参数初始化f_c和f_b
        nn.init.normal_(self.f_c, mean=self.args.f_c_mu, std=self.args.f_c_sigma)
        nn.init.normal_(self.f_b, mean=self.args.f_b_mu, std=self.args.f_b_sigma)

    # TODO add other filter
        
    def filter_generator(self, in_channels, freq_length): 
        omega = torch.linspace(0, 0.5, freq_length, device=self.device).view(1, -1, 1)
        
        self.omega = omega # .reshape(1, freq_length, 1).repeat([1, 1, in_channels])
        
        filters = torch.exp(-((self.omega - self.f_c) / (2 * self.f_b)) ** 2)
        return filters

    def forward(self, x): 
        in_dim, in_channels = x.shape[-2],x.shape[-1] # B,L,C
        freq = torch.fft.rfft(x, dim=1, norm='ortho')
        
        self.filters = self.filter_generator(in_channels, in_dim//2 + 1)
        # 应用滤波器到所有通道
        filtered_freq = freq * self.filters[:,:,:in_channels] # B,L//2,C * 1,L//2,c
        
        x_hat = torch.fft.irfft(filtered_freq, dim=1, norm='ortho')
        return x_hat.real
#%%
# 4 ##############################################  小波滤波器的通用实现  ##############################################
class WaveFiltersBase(nn.Module):
    def __init__(self, args):
        super(WaveFiltersBase, self).__init__()
        self.name = "WF"
        self.device = args.device
        self.to(self.device)
        self.in_channels = args.scale  # Number of input channels

    def fft_convolution(self, x, wavelet):
        # x: (B, L, C)
        # wavelet: (C, L)
        # Rearrange x to (B, C, L) for FFT
        x = rearrange(x, 'b l c -> b c l')
        x_fft = torch.fft.fft(x, n=x.size(-1))
        wavelet_fft = torch.fft.fft(wavelet, n=x.size(-1))
        # Rearrange wavelet_fft to (1, C, L) for broadcasting
        wavelet_fft = rearrange(wavelet_fft, 'c l -> 1 c l')
        # Element-wise multiplication in frequency domain
        filtered_fft = x_fft * wavelet_fft
        # Inverse FFT to get back to time domain
        filtered_signal = torch.fft.ifft(filtered_fft).real
        # Rearrange back to (B, L, C)
        filtered_signal = rearrange(filtered_signal, 'b c l -> b l c')
        return filtered_signal

class RickerWaveletFilter(WaveFiltersBase):
    def __init__(self, args):
        super(RickerWaveletFilter, self).__init__(args)
        # Initialize sigma for each channel
        self.sigma = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        nn.init.normal_(self.sigma, mean=args.Ricker_sigma, std=1.0)

    def ricker_wavelet(self, t):
        # t: (L)
        # Compute Ricker wavelet for each channel
        self.sigma_exp = torch.exp(self.sigma)  # Exponentiate sigma
        t = rearrange(t, 'l -> 1 l')  # (1, L)
        sigma = rearrange(self.sigma_exp, 'c -> c 1')  # (C, 1)
        term1 = 2 / (torch.sqrt(3 * sigma + 1e-6) * torch.pi**0.25)
        term2 = 1 - (t**2 / (sigma**2 + 1e-6))
        term3 = torch.exp(-t**2 / (2 * (sigma**2 + 1e-6)))
        return term1 * term2 * term3  # (C, L)

    def forward(self, x):
        # x: (B, L, C)
        length = x.shape[1]
        t = torch.linspace(-1, 1, length, device=self.device)
        wavelet = self.ricker_wavelet(t)  # (C, L)
        filtered_signal = self.fft_convolution(x, wavelet)
        return filtered_signal

class ChirpletWaveletFilter(WaveFiltersBase):
    def __init__(self, args):
        super(ChirpletWaveletFilter, self).__init__(args)
        # Initialize parameters for each channel
        self.sigma = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        self.omega = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        self.alpha = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        nn.init.normal_(self.sigma, mean=args.Chirplet_sigma, std=1.0)
        nn.init.normal_(self.omega, mean=args.Chirplet_omega, std=1.0)
        nn.init.normal_(self.alpha, mean=args.Chirplet_alpha, std=1.0)

    def chirplet_wavelet(self, t):
        # t: (L)
        # Compute Chirplet wavelet for each channel
        t = rearrange(t, 'l -> 1 l')  # (1, L)
        self.sigma_exp = torch.exp(self.sigma)  # Exponentiate sigma
        sigma = rearrange(self.sigma_exp, 'c -> c 1')  # (C, 1)
        omega = rearrange(self.omega, 'c -> c 1')  # (C, 1)
        alpha = rearrange(self.alpha, 'c -> c 1')  # (C, 1)
        term1 = 1 / (sigma + 1e-6)
        term2 = torch.exp(-0.5 * (t / sigma)**2)
        term3 = torch.exp(-1j * (0.5 * alpha * t**2 + omega * t))
        return term1 * term2 * term3  # (C, L)

    def forward(self, x):
        # x: (B, L, C)
        length = x.shape[1]
        t = torch.linspace(-1, 1, length, device=self.device)
        wavelet = self.chirplet_wavelet(t)  # (C, L)
        filtered_signal = self.fft_convolution(x, wavelet)
        return filtered_signal

class LaplaceWaveletFilter(WaveFiltersBase):
    def __init__(self, args):
        super(LaplaceWaveletFilter, self).__init__(args)
        # Initialize Laplace wavelet parameters for each channel
        self.A = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        self.ep = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        self.tal = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        self.f = nn.Parameter(torch.empty(self.in_channels, device=self.device))
        nn.init.normal_(self.A, mean=args.Laplace_A, std=1.0)
        nn.init.normal_(self.ep, mean=args.Laplace_ep, std=1.0)
        nn.init.normal_(self.tal, mean=args.Laplace_tal, std=1.0)
        nn.init.normal_(self.f, mean=args.Laplace_f, std=1.0)

    def laplace_wavelet(self, p):
        # p: (L)
        p = rearrange(p, 'l -> 1 l')  # (1, L)
        A = rearrange(self.A, 'c -> c 1')  # (C, 1)
        ep = rearrange(self.ep, 'c -> c 1')  # (C, 1)
        ep_sigmoid = torch.sigmoid(ep)  # Sigmoid to ensure 0 < ep < 1
        tal = rearrange(self.tal, 'c -> c 1')  # (C, 1)
        # tal_sigmoid = - torch.sigmoid(tal)  # Sigmoid to ensure tal < 0
        f = rearrange(self.f, 'c -> c 1')  # (C, 1)
        w = 2 * torch.pi * f  # Angular frequency, (C, 1)
        q = 1 - ep_sigmoid**2
        q = torch.clamp(q, min=1e-6)
        term = (-ep_sigmoid / torch.sqrt(q)) * (w * (p - tal))
        return A * torch.exp(term) * torch.sin(w * (p - tal))  # (C, L)

    def forward(self, x):
        # x: (B, L, C)
        length = x.shape[1]
        p = torch.linspace(0, 1, length, device=self.device)
        wavelet = self.laplace_wavelet(p)  # (C, L)
        filtered_signal = self.fft_convolution(x, wavelet)
        return filtered_signal

class MorletWaveletFilter(WaveFiltersBase):
    def __init__(self, args):
        super(MorletWaveletFilter, self).__init__(args)
        # Initialize parameters for each channel
        self.f_b = nn.Parameter(torch.empty(self.in_channels, device=self.device))  # Bandwidth parameter
        self.f_c = nn.Parameter(torch.empty(self.in_channels, device=self.device))  # Center frequency
        nn.init.normal_(self.f_b, mean=args.Morlet_f_b, std=1.0)
        nn.init.normal_(self.f_c, mean=args.Morlet_f_c, std=1.0)

    def complex_gaussian_wavelet(self, n):
        # n: (L)
        n = rearrange(n, 'l -> 1 l')  # (1, L)
        f_b = rearrange(self.f_b, 'c -> c 1')  # (C, 1)
        f_c = rearrange(self.f_c, 'c -> c 1')  # (C, 1)
        gaussian_envelope = (f_b / torch.sqrt(torch.tensor(math.pi, device=self.device))) * torch.exp(-(f_b ** 2) * (n ** 2))
        complex_exponent = torch.exp(1j * 2 * math.pi * f_c * n)
        return gaussian_envelope * complex_exponent  # (C, L)

    def forward(self, x):
        # x: (B, L, C)
        length = x.shape[1]
        n = torch.linspace(-1, 1, length, device=self.device)
        wavelet = self.complex_gaussian_wavelet(n)  # (C, L)
        filtered_signal = self.fft_convolution(x, wavelet)
        return filtered_signal

# 4 ############################################# Identity module #######################################
class Identity(SignalProcessingBase):
    def __init__(self, args):
        super(Identity, self).__init__(args)
        self.name = "I"
    def forward(self, x):
        return x


#%% 5 

class Morlet(SignalProcessingBase):
    def __init__(self, args):
        super(Morlet, self).__init__(args)
        self.name = "Morlet"
        self.convolution_operator = convlutional_operator('Morlet', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.convolution_operator(x)
        return x_transformed
    #%% Laplace
class Laplace(SignalProcessingBase):
    def __init__(self, args):
        super(Laplace, self).__init__(args)
        self.name = "Laplace"
        self.convolution_operator = convlutional_operator('Laplace', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.convolution_operator(x)
        return x_transformed
    #%% Order1MAFilter
class Order1MAFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order1MAFilter, self).__init__(args)
        self.name = "order1_MA"
        self.filter_operator = signal_filter_('order1_MA', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed
    #%% Order2MAFilter
class Order2MAFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order2MAFilter, self).__init__(args)
        self.name = "order2_MA"
        self.filter_operator = signal_filter_('order2_MA', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class Order1DFFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order1DFFilter, self).__init__(args)
        self.name = "order1_DF"
        self.filter_operator = signal_filter_('order1_DF', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class Order2DFFilter(SignalProcessingBase):
    def __init__(self, args):
        super(Order2DFFilter, self).__init__(args)
        self.name = "order2_DF"
        self.filter_operator = signal_filter_('order2_DF', in_channels=args.scale, device=args.device)

    def forward(self, x):
        x_transformed = self.filter_operator(x)
        return x_transformed

class LogOperation(SignalProcessingBase):
    def __init__(self, args):
        super(LogOperation, self).__init__(args)
        self.name = "Log"

    def forward(self, x):
        return torch.log(x)
class SquOperation(SignalProcessingBase):
    def __init__(self, args):
        super(SquOperation, self).__init__(args)
        self.name = "Squ"

    def forward(self, x):
        return x ** 2

class SinOperation(SignalProcessingBase):
    def __init__(self, args):
        super(SinOperation, self).__init__(args)
        self.name = "sin"
        self.fre = FRE # TODO learbable

    def forward(self, x):
        return torch.sin(self.fre * x)
    
class Laplace_neural_operator(SignalProcessingBase):
    # LNO
    def __init__(self, args):
        super(Laplace_neural_operator, self).__init__(args)
        self.name = "LNO"
        self.modes1 = args.scale * args.scale
        self.scale = (1 / (args.scale*args.scale))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(args.scale, args.scale, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(args.scale, args.scale, self.modes1, dtype=torch.cfloat))
       
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, -Hw) 
        return output_residue1,output_residue2    

    def forward(self, x):
        x = rearrange(x, 'b l c -> b c l')
        # t = grid_x_train.cuda()
        # x.shape = (batch_size, width, length),20,4,2048
        t = torch.linspace(0, 1, steps=x.shape[-1], dtype=x.dtype, device=x.device)
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
    
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2 = torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1 = torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2 = torch.exp(term1)
        x2 = torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2 = torch.real(x2)
        x2 = x2/x.size(-1)
        x = x1+x2
        # Norm = nn.InstanceNorm1d(x.size(1))
        # x = Norm(x)
        x = rearrange(x, 'b c l -> b l c')
        return x


###############################################2 arity###################################################
class AddOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(AddOperation, self).__init__(args)
        self.name = "add"

    def operation(self, x1, x2):
        return x1 + x2
    
class MulOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(MulOperation, self).__init__(args)
        self.name = "mul"

    def operation(self, x1, x2):
        return x1 * x2
    
class DivOperation(SignalProcessingBase2Arity):
    def __init__(self, args):
        super(DivOperation, self).__init__(args)
        self.name = "div"

    def operation(self, x1, x2):
        return x1 / (x2 + 1e-8)


if __name__ == "__main__":
    # 测试模块
    pass