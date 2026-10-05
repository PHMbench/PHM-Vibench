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
from numbers import Integral
import math
from .Signal_processing import *
from .Feature_extract import *

class Model(nn.Module):
    """Transparent Signal Processing Network (TSPN).

    Parameters
    ----------
    args : Namespace
        Defines the module composition and ``num_classes``.
    metadata : Any, optional
        Unused placeholder for compatibility.

    Notes
    -----
    Accepts an input tensor ``(B, L, C)`` and returns logits of shape
    ``(B, num_classes)``.
    """
    def __init__(self, args, metadata=None):
        """Build network modules from configuration.

        Args:
            args: 实验配置，包含信号处理与特征提取模块定义。
            metadata: 数据集元信息，可选。
        """
        super(Model, self).__init__()
        self.signal_processing_modules, self.feature_extractor_modules = self.config_network(args)
        self.layer_num = len(self.signal_processing_modules)
        self.args = args
        self.internal_instance_normalization = bool(
            getattr(args, "internal_instance_normalization", True)
        )

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
        feature_definitions = dict(getattr(args, "feature_definitions", {}))
        unsupported = set(feature_definitions) - {"Entropy", "Kurtosis"}
        inactive = set(feature_definitions) - set(args.feature_extractor_configs)
        if unsupported or inactive:
            raise ValueError(
                "feature_definitions only supports active Entropy/Kurtosis features; "
                f"unsupported={sorted(unsupported)}, inactive={sorted(inactive)}"
            )
        for feature_name in args.feature_extractor_configs:
            module_class = ALL_FE[feature_name]
            options = ({"definition": feature_definitions[feature_name]}
                       if feature_name in feature_definitions else {})
            if feature_name in {"Entropy", "Kurtosis"}:
                options["epsilon"] = getattr(args, "feature_epsilon", 1e-12)
            feature_extractor_modules[feature_name] = module_class(**options)
        
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
                                                                       self.args.skip_connection,
                                                                       self.internal_instance_normalization,
                                                                       gate_parameterization=getattr(self.args, "gate_parameterization", "softmax"),
                                                                       gate_temperature=getattr(self.args, "gate_temperature", 0.1),
                                                                       gate_bias=getattr(self.args, "gate_bias", True),
                                                                       skip_bias=getattr(self.args, "skip_bias", True),
                                                                       ).to(self.args.device))
            in_channels = out_channels 
            assert out_channels % self.signal_processing_layers[i].module_num == 0 
            # out_channels = int(out_channels * self.args.scale)
        self.channel_for_feature = out_channels # // self.args.scale

    def init_feature_extractor_layers(self):
        print('# build feature extractor layers')
        self.feature_extractor_layers = FeatureExtractorlayer(
            self.feature_extractor_modules,
            self.channel_for_feature,
            self.channel_for_feature,
            self.internal_instance_normalization,
            mixing=getattr(self.args, "feature_mixing", "shared"),
            mixing_bias=getattr(self.args, "feature_mixing_bias", True),
        ).to(self.args.device)
        len_feature = len(self.feature_extractor_modules)
        self.channel_for_classifier = self.channel_for_feature * len_feature


    def init_classifier(self):
        print('# build classifier')
        self.clf = Classifier(
            self.channel_for_classifier,
            self.args.num_classes,
            hidden_dims=getattr(self.args, "classifier_hidden_dims", (128,)),
            activation=getattr(self.args, "classifier_activation", "relu"),
            bias=getattr(self.args, "classifier_bias", True),
        ).to(self.args.device)

    def forward(self, x, data_id = None,task_id = None):
        """Compute logits for a batch.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, L, C)``.
        data_id : Any, optional
            Unused.
        task_id : Any, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
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
            # Running statistics are source-fitted state, not an autograd history
            # spanning training batches. Preserve the existing numerical update.
            with torch.no_grad():
                self.running_mean.copy_((1 - self.eps) * self.running_mean + self.eps * mean)
                self.running_var.copy_((1 - self.eps) * self.running_var + self.eps * var)
            # sqrt(var)'s derivative is infinite at a constant feature; masking
            # only after sqrt still produces 0 * inf in backward. Preserve the
            # exact forward value while choosing the zero subgradient there.
            zero_variance = var == 0
            safe_var = torch.where(zero_variance, torch.ones_like(var), var)
            std = torch.where(zero_variance, torch.zeros_like(var), safe_var.sqrt())
            out = (x - mean) / (std + self.eps)
        else:
            out = (x - self.running_mean) / (self.running_var.sqrt() + self.eps)
        return out

class SignalProcessingLayer(nn.Module):
    # TODO op first then weight connection -> attention
    def __init__(
        self,
        signal_processing_modules,
        input_channels,
        output_channels,
        skip_connection=True,
        internal_instance_normalization=True,
        *,
        gate_parameterization: str = "softmax",
        gate_temperature: float = 0.1,
        gate_bias: bool = True,
        skip_bias: bool = True,
    ):
        super(SignalProcessingLayer, self).__init__()
        self.norm = (
            nn.InstanceNorm1d(input_channels)
            if internal_instance_normalization
            else nn.Identity()
        )
        if gate_parameterization not in {"softmax", "raw"}:
            raise ValueError("gate_parameterization must be 'softmax' or 'raw'")
        if not math.isfinite(gate_temperature) or gate_temperature <= 0:
            raise ValueError("gate_temperature must be finite and positive")
        self.gate_parameterization = gate_parameterization
        self.weight_connection = nn.Linear(input_channels, output_channels, bias=gate_bias)
        self.signal_processing_modules = signal_processing_modules
        self.module_num = len(signal_processing_modules)
        self.temperature = float(gate_temperature)
        
        if skip_connection:
            self.skip_connection = nn.Linear(input_channels, output_channels, bias=skip_bias)
    def forward(self, x):
        # 信号标准化
        x = rearrange(x, 'b l c -> b c l')
        normed_x = self.norm(x)
        normed_x = rearrange(normed_x, 'b c l -> b l c')
        # 通过线性层
        
        # Normalize for this forward without mutating the parameter. In-place
        # ``weight.data`` replacement made consecutive interventions consume
        # different backbones and bypassed autograd's actual parameter path.
        normalized_weight = self.weight_connection.weight
        if self.gate_parameterization == "softmax":
            normalized_weight = F.softmax(normalized_weight / self.temperature, dim=0)
        x = F.linear(normed_x, normalized_weight, self.weight_connection.bias)

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
    def __init__(
        self,
        feature_extractor_modules,
        in_channels=1,
        out_channels=1,
        internal_instance_normalization=True,
        *,
        mixing: str = "shared",
        mixing_bias: bool = True,
    ):
        super(FeatureExtractorlayer, self).__init__()
        if mixing not in {"shared", "per_feature"}:
            raise ValueError("feature_mixing must be 'shared' or 'per_feature'")
        self.mixing = mixing
        if mixing == "shared":
            # Keep the original module name for strict restoration of p0.
            self.weight_connection = nn.Linear(in_channels, out_channels, bias=mixing_bias)
        else:
            self.weight_connections = nn.ModuleDict({
                name: nn.Linear(in_channels, out_channels, bias=mixing_bias)
                for name in feature_extractor_modules
            })
        self.feature_extractor_modules = feature_extractor_modules
        
        out_channels = int(len(feature_extractor_modules) * out_channels)
        
        self.pre_norm = (
            nn.InstanceNorm1d(in_channels)
            if internal_instance_normalization
            else nn.Identity()
        )
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
        
        outputs = []
        if self.mixing == "shared":
            mixed = self.weight_connection(normed_x).transpose(1, 2)
            outputs = [module(mixed) for module in self.feature_extractor_modules.values()]
        else:
            for name, module in self.feature_extractor_modules.items():
                mixed = self.weight_connections[name](normed_x).transpose(1, 2)
                outputs.append(module(mixed))
        res = torch.cat(outputs, dim=1).squeeze(-1) # B,C
        return self.norm(res)

class Classifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        *,
        hidden_dims=(128,),
        activation: str = "relu",
        bias: bool = True,
    ):
        super(Classifier, self).__init__()
        if activation not in {"relu", "identity"}:
            raise ValueError("classifier_activation must be 'relu' or 'identity'")
        if not isinstance(hidden_dims, (list, tuple)) or any(
            isinstance(width, bool) or not isinstance(width, Integral) or width <= 0
            for width in hidden_dims
        ):
            raise ValueError("classifier_hidden_dims must be a list of positive integers (or [])")
        dimensions = [in_channels, *hidden_dims, num_classes]
        layers = []
        for index, (width_in, width_out) in enumerate(zip(dimensions, dimensions[1:])):
            layers.append(nn.Linear(width_in, width_out, bias=bias))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU() if activation == "relu" else nn.Identity())
        self.clf = nn.Sequential(*layers)
        
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
