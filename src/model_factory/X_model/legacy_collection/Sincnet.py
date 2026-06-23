from .Resnet import ResNet
import torch
class Sincnet(ResNet):
    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False):
        super().__init__(block, layers, in_channel, num_class, zero_init_residual, first_kernel='Sinc')
        
class Sinc_net_m(ResNet):
    def __init__(self, block, layers, in_channel=1, num_class=4, zero_init_residual=False):
        super().__init__(block, layers, in_channel, num_class, zero_init_residual, first_kernel='Sinc_m')


if __name__ == '__main__':
    from Resnet import BasicBlock
    model = Sinc_net_m(BasicBlock, [2, 2, 2, 2], in_channel=6, num_class=7)
    test_x = torch.randn(2,2048,6)
    print(model(test_x).shape)
