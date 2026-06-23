from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
import ptwt
import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableWaveletLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(LearnableWaveletLayer, self).__init__()
        
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Initialize learnable parameters
        self.w = nn.Parameter(torch.randn(out_channels, 1))
        self.fc = nn.Parameter(torch.randn(out_channels, 1))
        
        # self.register_parameter('w', self.w)
        # self.register_parameter('fc', self.fc)
        
    def forward(self, x):
        # Compute wavelet coefficients
        
        n = torch.arange(-self.kernel_size//2, self.kernel_size//2, dtype=torch.float32, device=x.device)
        psi = self.w / torch.sqrt(torch.tensor(torch.pi, device=x.device)) * torch.exp(-self.w**2 * n**2) * torch.exp(1j * 2 * torch.pi * self.fc * n) #
        psi = psi.view(self.out_channels, 1, self.kernel_size)
        psi = psi.repeat(1, self.in_channels, 1)
        # psi = psi.view(self.out_channels*self.in_channels, 1, self.kernel_size)
        

        x = nn.functional.conv1d(x, psi.real, groups=self.out_channels*self.in_channels)

        
        return x

def env_hilbert(x):
    '''
        Perform Hilbert transform along the last axis of x.
        
        Parameters:
        -------------
        x (Tensor) : The signal data. 
                     The Hilbert transform is performed along last dimension of `x`.
        
        Returns:
        -------------
        analytic (Tensor): A complex tensor with the same shape of `x`,
                           representing its analytic signal. 
                           
        Hilbert transform @ wiki: https://en.wikipedia.org/wiki/Hilbert_transform
            
    '''
    
    x = torch.as_tensor(x) # .double()
    
    N = x.shape[-1]
    Xf = torch.fft.fft(x)
    if (N % 2 == 0):
        Xf[..., 1 : N//2] *= 2
        Xf[..., N//2+1 : ] = 0
    else:
        Xf[..., 1 : (N+1)//2] *= 2
        Xf[..., (N+1)//2 : ] = 0
    return torch.fft.ifft(Xf).abs() # squ()


class ELM(nn.Module):
    def __init__(self, input_size, h_size, num_classes):
        super(ELM, self).__init__()
        self._input_size = input_size
        self._h_size = h_size
        self._output_size = num_classes

        self._alpha = nn.Parameter(torch.empty(self._input_size, self._h_size).uniform_(-1., 1.))
        self._beta = nn.Parameter(torch.empty(self._h_size, self._output_size).uniform_(-1., 1.))
        self._bias = nn.Parameter(torch.zeros(self._h_size))

        self._activation = nn.Sigmoid()
        
        self.register_parameter('alpha', self._alpha)
        self.register_parameter('beta', self._beta)
        self.register_parameter('bias', self._bias)
        
        # self.bn = nn.BatchNorm1d(self._h_size)

    def forward(self, x):
        h = self._activation(torch.add(x @ self._alpha, self._bias))
        
        # h = self.bn(h)
        
        out = h @ self._beta

        return out.view(-1, self._output_size)

class Dong_ELM(nn.Module):
    def __init__(self, num_class = 4):
        super(Dong_ELM, self).__init__()
        
        # self.DWT0= DWT1DForward(J=1, wave='morlet').cuda()
        self.wave_t = LearnableWaveletLayer(1,1, 32)
        
        self.hilbert_t = env_hilbert
        
        self.mean = lambda x: torch.mean(x,dim=-1, keepdim = True)
        
        self.entropy = lambda x: (x * torch.log(x)).mean(dim=-1, keepdim = True)
       
        self.kurtosis = lambda x: (((x - torch.mean(x,dim=-1, keepdim = True)) ** 4).mean(dim=-1, keepdim = True)) / ((torch.var(x,dim=-1, keepdim = True) ** 2).mean(dim=-1, keepdim = True))
        
        self.gini = lambda x:  (torch.sort(x)[0] * (-2 * x.shape[-1] + 2 * torch.arange(x.shape[-1], device=x.device) - 1) / x.shape[-1]).mean(dim=-1, keepdim = True)
        # self.gini = lambda x:  torch.sum(torch.sort(x)[0] * (-2 * x.shape[-1] + 2 * torch.arange(x.shape[-1], device=x.device) - 1) / x.shape[-1], keepdim=True)
        
        def reciprocal_smoothness(x):
            # 除了NSE，还需要原始信号计算系数
            x = x**2
            NSES = x / torch.sum(x, dim=-1, keepdim=True)
            B = x.shape[0]
            a = (x / torch.pow(torch.prod(x, dim=-1, keepdim=True) + 1e-6, 1/B).mean(dim=-1, keepdim=True))
            res = torch.sum(NSES * a, dim=-1, keepdim=True)
            return res
        
        self.reciprocal_smoothness = reciprocal_smoothness
        

        
        self.elm = ELM(4, 1024, num_class)
        
    def forward(self, input):

        wave = self.wave_t(input)
        env = self.hilbert_t(wave)
        NSE = env**2 / torch.sum(env**2, dim=-1, keepdim=True)
        
        kurtosis = self.kurtosis(NSE)
        entropy = self.entropy(NSE)
        gini = self.gini(NSE)
        reciprocal_smoothness = self.reciprocal_smoothness(env)
        
        features = torch.cat([kurtosis, entropy, gini, reciprocal_smoothness], dim=-1)
        
        features = F.relu(features)
        
        output = self.elm(features)
        
        return output
    
if __name__ == '__main__':
    from torchinfo import summary
    model = Dong_ELM()
    summary(model.cuda(), (1, 64))
    x = torch.randn(2,1,1024).cuda()
    y = model(x)
    print(y.shape)

# https://github.com/qthequartermasterman/torch_pso