"""
对比学习配置模板和工具函数
=================================

提供标准化的对比学习配置模板，简化用户配置过程。
支持新旧格式的自动转换和验证。

作者: PHM-Vibench Team
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union
from types import SimpleNamespace

from .config_utils import ConfigWrapper


def create_single_contrastive_config(
    loss_type: str = "INFONCE",
    temperature: float = 0.07,
    margin: float = 0.3,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建单策略对比学习配置模板

    Args:
        loss_type: 损失类型，支持 "INFONCE", "SUPCON", "TRIPLET", "PROTOTYPICAL", "BARLOWTWINS", "VICREG"
        temperature: 温度参数 (InfoNCE/SupCon需要)
        margin: 边际参数 (Triplet需要)
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度 (None表示使用模型默认值)

    Returns:
        对比学习配置字典
    """
    config = {
        "type": "single",
        "loss_type": loss_type,
        "augmentation_noise_std": augmentation_noise_std
    }

    # 根据损失类型添加特定参数
    if loss_type in ["INFONCE", "SUPCON"]:
        config["temperature"] = temperature
    elif loss_type == "TRIPLET":
        config["margin"] = margin

    if projection_dim is not None:
        config["projection_dim"] = projection_dim

    return config


def create_ensemble_contrastive_config(
    losses: List[Dict[str, Any]],
    auto_normalize_weights: bool = True,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建集成策略对比学习配置模板

    Args:
        losses: 损失配置列表，每个元素包含 loss_type, weight 和特定参数
        auto_normalize_weights: 是否自动归一化权重
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度 (None表示使用模型默认值)

    Returns:
        对比学习配置字典
    """
    if not losses:
        raise ValueError("损失列表不能为空")

    # 权重归一化
    if auto_normalize_weights:
        total_weight = sum(loss.get("weight", 1.0) for loss in losses)
        for loss in losses:
            loss["weight"] = loss.get("weight", 1.0) / total_weight

    config = {
        "type": "ensemble",
        "losses": losses,
        "augmentation_noise_std": augmentation_noise_std
    }

    if projection_dim is not None:
        config["projection_dim"] = projection_dim

    return config


def create_infonce_config(
    temperature: float = 0.07,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建InfoNCE对比学习配置的便捷函数

    Args:
        temperature: 温度参数
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度

    Returns:
        InfoNCE配置字典
    """
    return create_single_contrastive_config(
        loss_type="INFONCE",
        temperature=temperature,
        augmentation_noise_std=augmentation_noise_std,
        projection_dim=projection_dim
    )


def create_supcon_config(
    temperature: float = 0.07,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建SupCon对比学习配置的便捷函数

    Args:
        temperature: 温度参数
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度

    Returns:
        SupCon配置字典
    """
    return create_single_contrastive_config(
        loss_type="SUPCON",
        temperature=temperature,
        augmentation_noise_std=augmentation_noise_std,
        projection_dim=projection_dim
    )


def create_triplet_config(
    margin: float = 0.3,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建Triplet对比学习配置的便捷函数

    Args:
        margin: 三元组边际参数
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度

    Returns:
        Triplet配置字典
    """
    return create_single_contrastive_config(
        loss_type="TRIPLET",
        margin=margin,
        augmentation_noise_std=augmentation_noise_std,
        projection_dim=projection_dim
    )


def create_hse_infonce_supcon_ensemble(
    infonce_weight: float = 0.6,
    supcon_weight: float = 0.4,
    temperature: float = 0.07,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建HSE专用的InfoNCE+SupCon集成配置

    结合InfoNCE的自监督能力和SupCon的监督对比能力，
    专为HSE对比学习任务优化。

    Args:
        infonce_weight: InfoNCE损失权重
        supcon_weight: SupCon损失权重
        temperature: 共享温度参数
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度

    Returns:
        HSE集成配置字典
    """
    losses = [
        {
            "loss_type": "INFONCE",
            "weight": infonce_weight,
            "temperature": temperature
        },
        {
            "loss_type": "SUPCON",
            "weight": supcon_weight,
            "temperature": temperature
        }
    ]

    return create_ensemble_contrastive_config(
        losses=losses,
        auto_normalize_weights=True,
        augmentation_noise_std=augmentation_noise_std,
        projection_dim=projection_dim
    )


def create_adaptive_contrastive_config(
    base_strategy: str = "INFONCE",
    adaptive_temperature: bool = True,
    temperature_range: tuple = (0.05, 0.15),
    adaptive_weights: bool = False,
    augmentation_noise_std: float = 0.1,
    projection_dim: Optional[int] = None
) -> Dict[str, Any]:
    """创建自适应对比学习配置

    支持训练过程中的自适应参数调整，为高级用户提供灵活性。

    Args:
        base_strategy: 基础策略类型
        adaptive_temperature: 是否使用自适应温度
        temperature_range: 温度调整范围
        adaptive_weights: 是否使用自适应权重 (仅集成策略)
        augmentation_noise_std: 数据增强噪声标准差
        projection_dim: 投影头维度

    Returns:
        自适应对比学习配置字典
    """
    config = create_single_contrastive_config(
        loss_type=base_strategy,
        temperature=0.07,  # 初始温度
        augmentation_noise_std=augmentation_noise_std,
        projection_dim=projection_dim
    )

    # 添加自适应配置
    config["adaptive"] = {
        "temperature": adaptive_temperature,
        "temperature_range": temperature_range,
        "weights": adaptive_weights
    }

    return config


def add_contrastive_to_config(
    base_config: Union[Dict, ConfigWrapper],
    contrastive_config: Dict[str, Any],
    contrast_weight: float = 0.15,
    use_system_sampling: bool = True,
    cross_system_contrast: bool = True
) -> ConfigWrapper:
    """将对比学习配置添加到基础配置中

    Args:
        base_config: 基础配置对象
        contrastive_config: 对比学习配置
        contrast_weight: 对比损失权重
        use_system_sampling: 是否使用系统采样
        cross_system_contrast: 是否使用跨系统对比

    Returns:
        包含对比学习配置的ConfigWrapper
    """
    from .config_utils import load_config

    # 加载基础配置
    if isinstance(base_config, dict):
        config = load_config(base_config)
    else:
        config = load_config(base_config.copy())

    # 设置任务为对比学习
    config.task.name = "hse_contrastive"
    config.task.type = "CDDG"

    # 添加对比学习配置
    config.task.contrastive_strategy = contrastive_config
    config.task.contrast_weight = contrast_weight
    config.task.use_system_sampling = use_system_sampling
    config.task.cross_system_contrast = cross_system_contrast

    # 确保有投影头维度
    if "projection_dim" in contrastive_config:
        config.model.projection_dim = contrastive_config["projection_dim"]

    return config


def upgrade_legacy_contrastive_config(legacy_config: Union[Dict, ConfigWrapper]) -> ConfigWrapper:
    """将旧版对比学习配置升级为新格式

    Args:
        legacy_config: 旧版配置对象

    Returns:
        升级后的ConfigWrapper
    """
    from .config_utils import load_config

    # 加载配置
    if isinstance(legacy_config, dict):
        config = load_config(legacy_config)
    else:
        config = load_config(legacy_config.copy())

    # 检查是否需要升级
    if not hasattr(config.task, 'name') or config.task.name != 'hse_contrastive':
        return config

    task = config.task

    # 检查是否已经是新格式
    if hasattr(task, 'contrastive_strategy'):
        return config

    # 检查是否有旧版对比学习参数
    if not hasattr(task, 'contrast_loss'):
        return config

    # 提取旧版参数
    old_loss_type = getattr(task, 'contrast_loss', 'INFONCE')
    old_temperature = getattr(task, 'temperature', 0.07)
    old_margin = getattr(task, 'margin', 0.3)
    old_prompt_weight = getattr(task, 'prompt_weight', 0.1)

    # 创建新格式配置
    contrastive_config = create_single_contrastive_config(
        loss_type=old_loss_type,
        temperature=old_temperature,
        margin=old_margin
    )

    # 移除旧版参数
    old_params = ['contrast_loss', 'temperature', 'margin', 'prompt_weight']
    for param in old_params:
        if hasattr(task, param):
            delattr(task, param)

    # 添加新格式配置
    task.contrastive_strategy = contrastive_config

    return config


def validate_contrastive_config_safely(config: Union[Dict, ConfigWrapper]) -> tuple[bool, List[str]]:
    """安全验证对比学习配置

    Args:
        config: 配置对象

    Returns:
        (is_valid, error_messages): 验证结果和错误信息列表
    """
    try:
        from .config_utils import load_config

        # 加载配置
        if isinstance(config, dict):
            cfg = load_config(config)
        else:
            cfg = load_config(config.copy())

        # 触发验证
        # 这将调用 _validate_contrastive_config 函数
        from .config_utils import _validate_contrastive_config
        _validate_contrastive_config(cfg)

        return True, []

    except Exception as e:
        return False, [str(e)]


# ==================== 预定义配置模板 ====================

CONTRASTIVE_TEMPLATES = {
    "infonce_basic": create_infonce_config(),
    "infonce_strong": create_infonce_config(temperature=0.05, augmentation_noise_std=0.15),
    "supcon_basic": create_supcon_config(),
    "triplet_basic": create_triplet_config(),
    "hse_ensemble": create_hse_infonce_supcon_ensemble(),
    "adaptive_infonce": create_adaptive_contrastive_config("INFONCE"),
    "adaptive_ensemble": create_adaptive_contrastive_config("ENSEMBLE")
}


def get_contrastive_template(template_name: str) -> Dict[str, Any]:
    """获取预定义的对比学习配置模板

    Args:
        template_name: 模板名称

    Returns:
        对比学习配置字典

    Raises:
        KeyError: 模板不存在时
    """
    if template_name not in CONTRASTIVE_TEMPLATES:
        available = list(CONTRASTIVE_TEMPLATES.keys())
        raise KeyError(f"对比学习模板 '{template_name}' 不存在。可用模板: {available}")

    return CONTRASTIVE_TEMPLATES[template_name].copy()


def list_contrastive_templates() -> List[str]:
    """列出所有可用的对比学习配置模板

    Returns:
        模板名称列表
    """
    return list(CONTRASTIVE_TEMPLATES.keys())


__all__ = [
    # 配置创建函数
    "create_single_contrastive_config",
    "create_ensemble_contrastive_config",
    "create_infonce_config",
    "create_supcon_config",
    "create_triplet_config",
    "create_hse_infonce_supcon_ensemble",
    "create_adaptive_contrastive_config",

    # 配置操作函数
    "add_contrastive_to_config",
    "upgrade_legacy_contrastive_config",
    "validate_contrastive_config_safely",

    # 模板管理
    "get_contrastive_template",
    "list_contrastive_templates",
    "CONTRASTIVE_TEMPLATES"
]