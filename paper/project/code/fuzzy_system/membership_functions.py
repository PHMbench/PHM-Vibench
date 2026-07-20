"""
隶属度函数模块

实现常用的模糊隶属度函数，包括三角形、梯形、高斯型隶属函数，
以及模糊变量和模糊集合的定义。

基于1阶谓词逻辑框架，每个模糊集合都可以视为谓词的解释。
"""

import numpy as np
import torch
from typing import Dict, List, Union, Tuple, Optional
from abc import ABC, abstractmethod


class MembershipFunction(ABC):
    """隶属度函数抽象基类"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        计算隶属度值

        Args:
            x: 输入值或数组

        Returns:
            隶属度值（0-1之间）
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, float]:
        """获取隶属度函数参数"""
        pass

    @abstractmethod
    def set_parameters(self, **params):
        """设置隶属度函数参数"""
        pass


class TriangularMembershipFunction(MembershipFunction):
    """三角形隶属度函数"""

    def __init__(self, name: str, a: float, b: float, c: float):
        """
        初始化三角形隶属度函数

        Args:
            name: 名称
            a: 左顶点
            b: 顶点
            c: 右顶点
        """
        super().__init__(name)
        self.a = min(a, b, c)
        self.b = max(min(a, b), min(b, c))  # 确保b在中间
        self.c = max(a, b, c)

    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return torch.clamp(
                torch.max(
                    torch.min((x - self.a) / (self.b - self.a),
                             (self.c - x) / (self.c - self.b)),
                    torch.tensor(0.0)
                ),
                0, 1
            )
        else:
            x = np.array(x)
            result = np.zeros_like(x, dtype=float)
            mask1 = (x >= self.a) & (x <= self.b)
            mask2 = (x > self.b) & (x <= self.c)
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
            result[mask2] = (self.c - x[mask2]) / (self.c - self.b)
            return np.clip(result, 0, 1)

    def get_parameters(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}

    def set_parameters(self, **params):
        if "a" in params:
            self.a = params["a"]
        if "b" in params:
            self.b = params["b"]
        if "c" in params:
            self.c = params["c"]
        # 确保顺序正确
        values = sorted([self.a, self.b, self.c])
        self.a, self.b, self.c = values


class GaussianMembershipFunction(MembershipFunction):
    """高斯型隶属度函数"""

    def __init__(self, name: str, center: float, sigma: float):
        """
        初始化高斯隶属度函数

        Args:
            name: 名称
            center: 中心点
            sigma: 标准差
        """
        super().__init__(name)
        self.center = center
        self.sigma = max(sigma, 1e-6)  # 避免除零

    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            return torch.exp(-((x - self.center) ** 2) / (2 * self.sigma ** 2))
        else:
            x = np.array(x)
            return np.exp(-((x - self.center) ** 2) / (2 * self.sigma ** 2))

    def get_parameters(self) -> Dict[str, float]:
        return {"center": self.center, "sigma": self.sigma}

    def set_parameters(self, **params):
        if "center" in params:
            self.center = params["center"]
        if "sigma" in params:
            self.sigma = max(params["sigma"], 1e-6)


class TrapezoidalMembershipFunction(MembershipFunction):
    """梯形隶属度函数"""

    def __init__(self, name: str, a: float, b: float, c: float, d: float):
        """
        初始化梯形隶属度函数

        Args:
            name: 名称
            a: 左下角
            b: 左上角
            c: 右上角
            d: 右下角
        """
        super().__init__(name)
        self.a, self.b, self.c, self.d = sorted([a, b, c, d])

    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        if isinstance(x, torch.Tensor):
            result = torch.zeros_like(x)
            mask1 = (x >= self.a) & (x < self.b)
            mask2 = (x >= self.b) & (x <= self.c)
            mask3 = (x > self.c) & (x <= self.d)
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
            result[mask2] = torch.tensor(1.0)
            result[mask3] = (self.d - x[mask3]) / (self.d - self.c)
            return torch.clamp(result, 0, 1)
        else:
            x = np.array(x)
            result = np.zeros_like(x, dtype=float)
            mask1 = (x >= self.a) & (x < self.b)
            mask2 = (x >= self.b) & (x <= self.c)
            mask3 = (x > self.c) & (x <= self.d)
            result[mask1] = (x[mask1] - self.a) / (self.b - self.a)
            result[mask2] = 1.0
            result[mask3] = (self.d - x[mask3]) / (self.d - self.c)
            return np.clip(result, 0, 1)

    def get_parameters(self) -> Dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}

    def set_parameters(self, **params):
        if "a" in params:
            self.a = params["a"]
        if "b" in params:
            self.b = params["b"]
        if "c" in params:
            self.c = params["c"]
        if "d" in params:
            self.d = params["d"]
        # 确保顺序正确
        values = sorted([self.a, self.b, self.c, self.d])
        self.a, self.b, self.c, self.d = values


class FuzzySet:
    """模糊集合定义"""

    def __init__(self, name: str, membership_func: MembershipFunction):
        """
        初始化模糊集合

        Args:
            name: 模糊集合名称（如"low", "medium", "high"）
            membership_func: 隶属度函数
        """
        self.name = name
        self.membership_func = membership_func

    def get_membership(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """计算隶属度"""
        return self.membership_func(x)

    def __call__(self, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        return self.get_membership(x)


class FuzzyVariable:
    """模糊变量定义"""

    def __init__(self, name: str, universe: Tuple[float, float], fuzzy_sets: List[FuzzySet]):
        """
        初始化模糊变量

        Args:
            name: 变量名称
            universe: 论域范围 [min, max]
            fuzzy_sets: 模糊集合列表
        """
        self.name = name
        self.universe = universe
        self.fuzzy_sets = {}
        for fs in fuzzy_sets:
            self.fuzzy_sets[fs.name] = fs

    def get_membership(self, set_name: str, x: Union[float, np.ndarray, torch.Tensor]) -> Union[float, np.ndarray, torch.Tensor]:
        """
        计算特定模糊集合的隶属度

        Args:
            set_name: 模糊集合名称
            x: 输入值

        Returns:
            隶属度值
        """
        if set_name not in self.fuzzy_sets:
            raise ValueError(f"Fuzzy set '{set_name}' not found in variable '{self.name}'")
        return self.fuzzy_sets[set_name].get_membership(x)

    def get_all_memberships(self, x: Union[float, np.ndarray, torch.Tensor]) -> Dict[str, Union[float, np.ndarray, torch.Tensor]]:
        """
        计算所有模糊集合的隶属度

        Args:
            x: 输入值

        Returns:
            所有模糊集合的隶属度字典
        """
        return {name: fs.get_membership(x) for name, fs in self.fuzzy_sets.items()}

    def add_fuzzy_set(self, fuzzy_set: FuzzySet):
        """添加模糊集合"""
        self.fuzzy_sets[fuzzy_set.name] = fuzzy_set

    def get_fuzzy_set(self, set_name: str) -> FuzzySet:
        """获取模糊集合"""
        if set_name not in self.fuzzy_sets:
            raise ValueError(f"Fuzzy set '{set_name}' not found")
        return self.fuzzy_sets[set_name]


def create_triangular_sets(name: str, universe: Tuple[float, float], num_sets: int = 3) -> FuzzyVariable:
    """
    快速创建基于三角形隶属度函数的模糊变量

    Args:
        name: 变量名称
        universe: 论域范围
        num_sets: 模糊集合数量（通常为3: low, medium, high）

    Returns:
        FuzzyVariable实例
    """
    min_val, max_val = universe
    step = (max_val - min_val) / (num_sets - 1)

    if num_sets == 3:
        # 创建low, medium, high三个集合
        low = TriangularMembershipFunction(f"{name}_low", min_val, min_val, min_val + step)
        medium = TriangularMembershipFunction(f"{name}_medium", min_val, min_val + step, max_val)
        high = TriangularMembershipFunction(f"{name}_high", min_val + step, max_val, max_val)

        fuzzy_sets = [
            FuzzySet("low", low),
            FuzzySet("medium", medium),
            FuzzySet("high", high)
        ]
    else:
        # 通用情况
        fuzzy_sets = []
        for i in range(num_sets):
            if i == 0:
                a, b, c = min_val, min_val, min_val + step
            elif i == num_sets - 1:
                a, b, c = max_val - step, max_val, max_val
            else:
                a = min_val + (i - 1) * step
                b = min_val + i * step
                c = min_val + (i + 1) * step

            mf = TriangularMembershipFunction(f"{name}_set_{i}", a, b, c)
            fuzzy_sets.append(FuzzySet(f"set_{i}", mf))

    return FuzzyVariable(name, universe, fuzzy_sets)