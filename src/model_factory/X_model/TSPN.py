"""
@file TSPN.py
@brief Transparent Signal Processing Network (TSPN) for time series classification.
@details This module implements a Transparent Signal Processing Network that consists of signal processing layers, feature extractor layers, and a classifier.
@date 2025-06-07
@version 1.0
@author Qi Li, Xuan Li
"""
#TODO: 2D signal processing, Logic_inference
from scipy import optimize
import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from collections import OrderedDict
from .Signal_processing import *
from .Feature_extract import *

class Model(nn.Module):
    """
    Transparent Signal Processing Network (TSPN) for time series classification.
    This model consists of multiple signal processing layers, feature extractor layers, and a classifier.
    Args:
        args: Arguments containing model configuration such as input channels, output channels, scale, skip connection, and number of classes.
        metadata: Optional metadata for the model (not used in this implementation).
    """
    def __init__(self, args, metadata=None):
        super(Model, self).__init__()
        self.signal_processing_modules, self.feature_extractor_modules = self.config_network(args)
        self.layer_num = len(self.signal_processing_modules)
        self.args = args

        self.init_signal_processing_layers()
        self.init_feature_extractor_layers()
        self.init_classifier()

    def config_network(self, args):
        """
        input: config,args
        putput: signal_processing_modules,feature_extractor_modules
        function: 从配置文件中构建信号处理模块和特征提取模块。
        """
        signal_processing_modules = []
        for layer in args.signal_processing_configs.values():
            signal_module = OrderedDict()
            for module_name in layer:
                
                module_class = ALL_SP[module_name]
                
                module_name = get_unique_module_name(signal_module.keys(), module_name)
                signal_module[module_name] = module_class(args)  # 假设所有模块的构造函数不需要参数
            signal_processing_modules.append(SignalProcessingModuleDict(signal_module))

        feature_extractor_modules = OrderedDict()
        for feature_name in args.feature_extractor_configs:
            module_class = ALL_FE[feature_name]
            feature_extractor_modules[feature_name] = module_class()  # 假设所有模块的构造函数不需要参数
        
        # TODO logic
        
        return signal_processing_modules,feature_extractor_modules
    
    def init_signal_processing_layers(self):
        print('# build signal processing layers')
        in_channels = self.args.in_channels
        out_channels = int(self.args.out_channels * self.args.scale)

        self.signal_processing_layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.signal_processing_layers.append(SignalProcessingLayer(self.signal_processing_modules[i],
                                                                       in_channels,
                                                                         out_channels,
                                                                         self.args.skip_connection).to(self.args.device))
            in_channels = out_channels 
            assert out_channels % self.signal_processing_layers[i].module_num == 0 
            # out_channels = int(out_channels * self.args.scale)
        self.channel_for_feature = out_channels # // self.args.scale

    def init_feature_extractor_layers(self):
        print('# build feature extractor layers')
        self.feature_extractor_layers = FeatureExtractorlayer(self.feature_extractor_modules,self.channel_for_feature,self.channel_for_feature).to(self.args.device)
        len_feature = len(self.feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature


    def init_classifier(self):
        print('# build classifier')
        self.clf = Classifier(self.channel_for_classifier, self.args.num_classes).to(self.args.device)

    def forward(self, x, data_id = None,task_id = None):
        # TODO: data_id,task_id
        for layer in self.signal_processing_layers:
            x = layer(x)
        x = self.feature_extractor_layers(x)
        
        x = self.clf(x)
        return x

class CustomBatchNorm(nn.Module):
    def __init__(self, num_features, eps=0.1):
        super(CustomBatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('running_mean', torch.zeros(1,num_features))
        self.register_buffer('running_var', torch.ones(1,num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            self.running_mean = (1 - self.eps) * self.running_mean + self.eps * mean
            self.running_var = (1 - self.eps) * self.running_var + self.eps * var
            out = (x - mean) / (var.sqrt() + self.eps)
        else:
            out = (x - self.running_mean) / (self.running_var.sqrt() + self.eps)
        return out

class SignalProcessingLayer(nn.Module):
    # TODO op first then weight connection -> attention
    def __init__(self, signal_processing_modules, input_channels, output_channels,skip_connection=True):
        super(SignalProcessingLayer, self).__init__()
        self.norm = nn.InstanceNorm1d(input_channels)
        self.weight_connection = nn.Linear(input_channels, output_channels)
        self.signal_processing_modules = signal_processing_modules
        self.module_num = len(signal_processing_modules)
        self.temperature = 0.1
        
        if skip_connection:
            self.skip_connection = nn.Linear(input_channels, output_channels)
    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        # 通过线性层
        
        self.weight_connection.weight.data = F.softmax((1.0 / self.temperature) *
                                                       self.weight_connection.weight.data, dim=0)
        x = self.weight_connection(normed_x)

        # 按模块数拆分
        splits = torch.split(x, x.size(2) // self.module_num, dim=2)

        # 通过模块计算
        outputs = []
        for module, split in zip(self.signal_processing_modules.values(), splits):
            outputs.append(module(split))
        x = torch.cat(outputs, dim=2)
        # 添加skip connection
        if hasattr(self, 'skip_connection'):
            # self.skip_connection.weight.data = F.softmax((1.0 / self.temperature) *
            #                                             self.skip_connection.weight.data, dim=0)
            x = x + self.skip_connection(normed_x)
        return x
    
class FeatureExtractorlayer(nn.Module):
    def __init__(self, feature_extractor_modules,in_channels=1, out_channels=1):
        super(FeatureExtractorlayer, self).__init__()
        self.weight_connection = nn.Linear(in_channels, out_channels)
        self.feature_extractor_modules = feature_extractor_modules
        
        out_channels = int(len(feature_extractor_modules) * out_channels)
        
        self.pre_norm = nn.InstanceNorm1d(in_channels)
        self.norm = CustomBatchNorm(out_channels)
        
        # self.temperature = 1
    # def norm(self,x): # feature normalization
    #     mean = x.mean(dim = 0,keepdim = True)
    #     std = x.std(dim = 0,keepdim = True)
    #     out = (x-mean)/(std + 1e-10)
    #     return out
           
    def forward(self, x):
        # TODO # self.weight_connection.weight.data = F.softmax((1.0 / self.temperature) *
        #                                                self.weight_connection.weight.data, dim=0)
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.pre_norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        
        x = self.weight_connection(normed_x)
        x = rearrange(x, 'b l c -> b c l')
        outputs = []
        for module in self.feature_extractor_modules.values():
            outputs.append(module(x))
        res = torch.cat(outputs, dim=1).squeeze() # B,C
        return self.norm(res)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes): # TODO logic
        super(Classifier, self).__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
            
        )
        # self.clf = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.clf(x)

def get_unique_module_name(existing_names, module_name):
    """
    根据已存在的模块名列表，为新模块生成一个唯一的名称。
    
    :param existing_names: 已存在的模块名称的集合或列表。
    :param module_name: 要检查的模块名称。
    :return: 唯一的模块名称。
    """
    if module_name not in existing_names:
        # 如果模块名不存在，则直接返回
        return module_name
    else:
        # 如果模块名已存在，尝试添加序号直到找到一个唯一的名字
        index = 1
        unique_name = f"{module_name}_{index}"
        while unique_name in existing_names:
            index += 1
            unique_name = f"{module_name}_{index}"
        return unique_name
        
ALL_SP = {
    'FFT': FFTSignalProcessing,
    'HT': HilbertTransform,
    'WF': WaveFilters,
    'I': Identity,
    'LNO': Laplace_neural_operator,
    'RWF':RickerWaveletFilter,
    'LWF':LaplaceWaveletFilter,
    'CWF':ChirpletWaveletFilter,
    'MWF':MorletWaveletFilter,
    
    'Morlet':Morlet, # 'Morlet':Morlet,
    'Laplace':Laplace,
    'Order1MAFilter':Order1MAFilter,
    'Order2MAFilter':Order2MAFilter,
    'Order1DFFilter':Order1DFFilter,
    'Order2DFFilter':Order2DFFilter,
    'Log':LogOperation,
    'Squ':SquOperation,
    'Sin':SinOperation,
    # 2arity
    'Add':AddOperation,
    'Mul':MulOperation,
    'Div':DivOperation  
}

ALL_FE = {
    'Mean': MeanFeature,
    'Std': StdFeature,
    'Var': VarFeature,
    'Entropy': EntropyFeature,
    'Max': MaxFeature,
    'Min': MinFeature,
    'AbsMean': AbsMeanFeature,
    'Kurtosis': KurtosisFeature,
    'RMS': RMSFeature,
    'CrestFactor': CrestFactorFeature,
    'Skewness': SkewnessFeature,
    'ClearanceFactor': ClearanceFactorFeature,
    'ShapeFactor': ShapeFactorFeature,
}

if __name__ == '__main__':
    pass