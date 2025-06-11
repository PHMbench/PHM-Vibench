import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy
from sympy.abc import x


class FeatureExtractionBase(nn.Module):
    def __init__(self, method_name):
        super(FeatureExtractionBase, self).__init__()
        self.method_name = method_name
        self.method_function = None
        self.symbolic_function = None

    def register_feature_method(self, torch_func):
        self.method_function = torch_func

    def forward(self, x): # x: [B, C, L] -> [B, C, 1] 
        if self.method_function is None:
            raise NotImplementedError("Feature method function is not registered.")
        return self.method_function(x)
    
class FeatureExtractionModuleDict(torch.nn.ModuleDict):
    def __init__(self, module_dict):
        super(FeatureExtractionModuleDict, self).__init__(module_dict)

    def forward(self, x, key):
        if key in self:
            return self[key](x)
        else:
            raise KeyError(f"No feature extraction module found for key: {key}")


# Mean
class MeanFeature(FeatureExtractionBase):
    def __init__(self):
        super(MeanFeature, self).__init__("mean")
        self.register_feature_method(
            lambda x: torch.mean(x, dim=-1, keepdim=True),
        )
        self.name = "Mean"
        
# Std        
class StdFeature(FeatureExtractionBase):
    def __init__(self):
        super(StdFeature, self).__init__("std")
        self.register_feature_method(
            lambda x: torch.std(x, dim=-1, keepdim=True)
        )
        self.name = "Std"
# Var
class VarFeature(FeatureExtractionBase):
    def __init__(self):
        super(VarFeature, self).__init__("var")
        self.register_feature_method(
            lambda x: torch.var(x, dim=-1, keepdim=True)
        )
        self.name = "Var"
#  Entropy
class EntropyFeature(FeatureExtractionBase):
    def __init__(self):
        super(EntropyFeature, self).__init__("entropy")
        self.register_feature_method(
            lambda x: (x * torch.log(torch.softmax(x, dim=-1))).mean(dim=-1, keepdim=True)
        )
        self.name = "Entropy"
# Max
class MaxFeature(FeatureExtractionBase):
    def __init__(self):
        super(MaxFeature, self).__init__("max")
        self.register_feature_method(
            lambda x: torch.max(x, dim=-1, keepdim=True)[0]  # 返回值和索引，这里只取值
        )
        self.name = "Max"
# Min
class MinFeature(FeatureExtractionBase):
    def __init__(self):
        super(MinFeature, self).__init__("min")
        self.register_feature_method(
            lambda x: torch.min(x, dim=-1, keepdim=True)[0]  # 返回值和索引，这里只取值
        )
        self.name = "Min"
# AbsMax        
class AbsMeanFeature(FeatureExtractionBase):
    def __init__(self):
        super(AbsMeanFeature, self).__init__("abs_mean")
        self.register_feature_method(
            lambda x: torch.mean(torch.abs(x), dim=-1, keepdim=True)
        )
        self.name = "AbsMean"
# Kurtosis        
class KurtosisFeature(FeatureExtractionBase):
    def __init__(self):
        super(KurtosisFeature, self).__init__("kurtosis")
        self.register_feature_method(
            lambda x: (((x - torch.mean(x, dim=-1, keepdim=True)) ** 4).mean(dim=-1, keepdim=True)) /
                      (torch.var(x, dim=-1, keepdim=True) ** 2)
        )
        self.name = "Kurtosis"
# RMS
class RMSFeature(FeatureExtractionBase):
    def __init__(self):
        super(RMSFeature, self).__init__("rms")
        self.register_feature_method(
            lambda x: torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        )
        self.name = "RMS"
# CrestFactor
class CrestFactorFeature(FeatureExtractionBase):
    def __init__(self):
        super(CrestFactorFeature, self).__init__("crest_factor")
        self.register_feature_method(
            lambda x: torch.max(x, dim=-1, keepdim=True)[0] / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        )
        self.name = "CrestFactor"
# ClearanceFactor        
class ClearanceFactorFeature(FeatureExtractionBase):
    def __init__(self):
        super(ClearanceFactorFeature, self).__init__("clearance_factor")
        self.register_feature_method(
            lambda x: torch.max(x, dim=-1, keepdim=True)[0] / torch.mean(torch.abs(x), dim=-1, keepdim=True)
        )
        self.name = "ClearanceFactor"
# Skewness
class SkewnessFeature(FeatureExtractionBase):
    def __init__(self):
        super(SkewnessFeature, self).__init__("skewness")
        self.register_feature_method(
            lambda x: ((x - torch.mean(x, dim=-1, keepdim=True)) ** 3).mean(dim=-1, keepdim=True) /
                      (torch.std(x, dim=-1, keepdim=True) ** 3)
        )
        self.name = "Skewness"
# ShapeFactor
class ShapeFactorFeature(FeatureExtractionBase):
    def __init__(self):
        super(ShapeFactorFeature, self).__init__("shape_factor")
        self.register_feature_method(
            lambda x: torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True)) / torch.mean(torch.abs(x), dim=-1, keepdim=True)
        )
        self.name = "ShapeFactor"
# CrestFactorDelta
class CrestFactorDeltaFeature(FeatureExtractionBase):
    def __init__(self):
        super(CrestFactorDeltaFeature, self).__init__("crest_factor_delta")
        self.register_feature_method(
            lambda x: torch.sqrt(torch.mean(torch.pow(x[:,:, 1:] - x[:,:, :-1], 2), dim=-1, keepdim=True)) / torch.mean(torch.abs(x), dim=-1, keepdim=True)
        )
        self.name = "CrestFactorDelta"
# KurtosisDelta
class KurtosisDeltaFeature(FeatureExtractionBase):
    def __init__(self):
        super(KurtosisDeltaFeature, self).__init__("kurtosis_delta")
        self.register_feature_method(
            lambda x: (((x[:,:, 1:] - x[:,:, :-1] - torch.mean(x[:,:, 1:] - x[:,:, :-1], dim=-1, keepdim=True)) ** 4).mean(dim=-1, keepdim=True)) /
                      (((x[:,:, 1:] - x[:,:, :-1] - torch.mean(x[:,:, 1:] - x[:,:, :-1], dim=-1, keepdim=True)) ** 2).mean(dim=-1, keepdim=True) ** 2)
        )
        self.name = "KurtosisDelta"



if __name__ == "__main__":


# 创建三维信号
    signal = torch.randn(2,3, 100)

    # 创建特征提取器字典
    feature_extractors = {
        "Mean": MeanFeature(),
        "Std": StdFeature(),
        "Var": VarFeature(),
        "Entropy": EntropyFeature(),
        "Max": MaxFeature(),
        "Min": MinFeature(),
        "AbsMean": AbsMeanFeature(),
        "Kurtosis": KurtosisFeature(),
        "RMS": RMSFeature(),
        "CrestFactor": CrestFactorFeature(),
        "ClearanceFactor": ClearanceFactorFeature(),
        "Skewness": SkewnessFeature(),
        "ShapeFactor": ShapeFactorFeature(),
        "CrestFactorDelta": CrestFactorDeltaFeature(),
        "KurtosisDelta": KurtosisDeltaFeature(),
    }

    # 对每个维度计算特征并打印
    for i in range(signal.shape[1]):
        print(f"Dimension: {i+1}")
        for feature_name, feature in feature_extractors.items():
            print(f"{feature_name}: {feature(signal)}")
        print()
    features = FeatureExtractionModuleDict(feature_extractors)
    print(features(signal, "Mean").shape)
