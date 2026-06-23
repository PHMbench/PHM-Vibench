import sys
sys.path.append('/home/user/LQ/B_Signal/Transparent_information_fusion/model_collection')
import numpy as np
from torchinfo import summary
import torch 

from Resnet import ResNet, BasicBlock
from Sincnet import Sincnet,Sinc_net_m
from WKN import WKN,WKN_m
from EELM import Dong_ELM
from MWA_CNN import A_cSE,Huan_net
from TFN.Models.TFN import TFN_Morlet
from MCN.models import MCN_GFK, MultiChannel_MCN_GFK
from MCN.models import MCN_WFK,MultiChannel_MCN_WFK


ff = np.arange(0, 2049) / 2049


MODEL_DICT = {
    'Resnet': ResNet(BasicBlock, [2, 2, 2, 2],in_channel =2, num_class = 4),
    # 'Sincnet': Sincnet(BasicBlock, [2, 2, 2, 2],num_class = 4),
    # 'WKN': WKN(BasicBlock, [2, 2, 2, 2],num_class = 4),
    'WKN_m': WKN_m(BasicBlock, [2, 2, 2, 2],in_channel =2 ,num_class = 4),
    'Sinc_net_m': Sinc_net_m(BasicBlock, [2, 2, 2, 2],in_channel =2,num_class = 4),
    # 'Dong_ELM': Dong_ELM(num_class = 4), # 
    'Huan_net': Huan_net(input_size = 2,num_class = 4),
    'TFN_Morlet': TFN_Morlet(in_channels=2, out_channels=4),
    
    'MCN_GFK':MultiChannel_MCN_GFK(ff=ff, in_channels=2, num_MFKs=8, num_classes=4),
    # 'MCN_WFK':MultiChannel_MCN_WFK(ff=ff, in_channels=2, num_MFKs=8, num_classes=4)
    # 'MCN_GFK': MCN_GFK,
    # 'MCN_WFK': MCN_WFK
}

if __name__ == '__main__':
    test_x = torch.randn(3, 2, 4096).cuda()
    for model_name, model in MODEL_DICT.items():
        model = model.cuda()
        y = model(test_x)
        net_summaary= summary(model.cuda(),(2,2,4096),device = "cuda")
        print(net_summaary)
        with open(f'save/{model_name}.txt','w') as f:
            f.write(str(net_summaary))  
        