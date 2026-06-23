import torch
import torch.nn as nn
from pytorch_wavelets import DWT1DForward, DWT1DInverse  # or simply DWT1D, IDWT1D
import ptwt
from einops import rearrange
#%% wanghuan‘ work


class A_cSE(nn.Module):
    
    def __init__(self, in_ch):
        super(A_cSE, self).__init__()
        
        self.conv0 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_ch, int(in_ch/2), kernel_size=1, padding=0),
            nn.BatchNorm1d(int(in_ch/2)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(int(in_ch/2), in_ch, kernel_size=1, padding=0),
            nn.BatchNorm1d(in_ch)
        )
        
    def forward(self, in_x):
        
        x = self.conv0(in_x)
        x = nn.AvgPool1d(x.size()[2:])(x)
        #print('channel',x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.sigmoid(x)
        
        return in_x * x + in_x

class SConv_1D(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, kernel, pad):
        super(SConv_1D, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel, padding=pad),
            nn.GroupNorm(6, out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x = self.conv(x)
        return x

numf =12

class Huan_net(nn.Module):
    def __init__(self,input_size = 1,num_class = 4):
        super(Huan_net, self).__init__()
    
        
        self.DWT0= DWT1DForward(J=1, wave='db16').cuda()
        
        self.SConv1 = SConv_1D(input_size*2, numf, 3, 0)
        self.DWT1= DWT1DForward(J=1, wave='db16').cuda()
        self.dropout1 = nn.Dropout(p=0.1)
        self.cSE1 = A_cSE(numf*2)
        
        self.SConv2 = SConv_1D(numf*2, numf*2, 3, 0)
        self.DWT2= DWT1DForward(J=1, wave='db16').cuda() 
        self.dropout2 = nn.Dropout(p=0.1)
        self.cSE2 = A_cSE(numf*4)
        
        self.SConv3 = SConv_1D(numf*4, numf*4, 3, 0)
        self.DWT3= DWT1DForward(J=1, wave='db16').cuda()       
        self.dropout3 = nn.Dropout(p=0.1)
        self.cSE3 = A_cSE(numf*8)
        
        self.SConv6 = SConv_1D(numf*8, numf*8, 3, 0)              
        
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(numf*8, num_class)

        
    def forward(self, input):
        
        input = rearrange(input, 'b l c -> b c l')
        DMT_yl,DMT_yh = self.DWT0(input)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1)
        
        output = self.SConv1(output)
        DMT_yl,DMT_yh = self.DWT1(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1)
        output = self.dropout1(output)
        output = self.cSE1(output)
        
        output = self.SConv2(output)
        DMT_yl,DMT_yh = self.DWT2(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
        output = self.dropout2(output)
        output = self.cSE2(output)
        
        output = self.SConv3(output)
        DMT_yl,DMT_yh = self.DWT3(output)
        output = torch.cat([DMT_yl,DMT_yh[0]], dim=1) 
        output = self.dropout3(output)
        output = self.cSE3(output)
        
        output = self.SConv6(output)             
            
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        
        return output
    
if __name__ == '__main__':
    model = Huan_net(input_size=2, num_class=4).cuda()
    print(model)
    input = torch.rand(2,4096,2).cuda()
    output = model(input)
    print(output.size())
    print(output)