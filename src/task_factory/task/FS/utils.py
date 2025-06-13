\
import torch
from torch.nn import init

def model_weights_init(m):
    """
    Initializes model weights.
    :param m: model module
    """
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        # For linear layers, it's also common to use kaiming or xavier.
        # init.xavier_normal_(m.weight.data)
        init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias.data, 0)

# PyTorch Lightning handles device management (GPU/CPU) automatically via the Trainer.
# The get_avaliable_gpu function from the original code is generally not needed
# when using PyTorch Lightning.
