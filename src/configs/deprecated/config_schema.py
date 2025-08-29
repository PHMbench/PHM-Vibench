"""
PHM-Vibench Pydantic配置模型
=============================

基于Pydantic的智能配置系统，提供：
- 🔍 自动类型验证
- 📝 IDE自动补全支持
- ⚙️ 智能默认值管理
- 📚 自动文档生成
- 🔗 配置继承和组合

使用方式：
    from src.configs.config_schema import PHMConfig
    
    config = PHMConfig(
        experiment_name="my_experiment",
        model__d_model=256,
        trainer__num_epochs=100
    )

作者: PHM-Vibench Team
"""

from typing import Dict, List, Optional, Union, Literal, Any
from pathlib import Path
from pydantic import BaseModel, Field, field_validator, model_validator
import os
import torch


# ==================== 基础配置类 ====================

class EnvironmentConfig(BaseModel):
    """环境配置 - 控制实验运行环境"""
    
    # 实验元信息
    experiment_name: str = Field(default="phm_experiment", description="实验名称")
    project: str = Field(default="phm_vibench", description="项目名称")  
    notes: str = Field(default="", description="实验备注")
    
    # 随机性控制
    seed: int = Field(default=42, description="随机种子", ge=0, le=2**32-1)
    iterations: int = Field(default=1, description="实验重复次数", ge=1, le=100)
    
    # 日志和监控
    wandb: bool = Field(default=False, description="启用WandB日志")
    swanlab: bool = Field(default=False, description="启用SwanLab日志")
    WANDB_MODE: str = Field(default="disabled", description="WandB模式")
    
    # 路径配置
    VBENCH_HOME: Optional[str] = Field(default=None, description="项目根目录")
    output_dir: str = Field(default="save", description="输出目录")
    
    class Config:
        extra = "allow"  # 允许额外字段


class DataConfig(BaseModel):
    """数据配置 - 控制数据加载和预处理"""
    
    # 数据源
    data_dir: str = Field(..., description="数据根目录")
    metadata_file: str = Field(..., description="元数据文件名")
    
    # 数据加载
    batch_size: int = Field(default=32, description="批次大小", ge=1, le=1024)
    num_workers: int = Field(default=4, description="数据加载进程数", ge=0, le=32)
    pin_memory: bool = Field(default=True, description="启用内存固定")
    persistent_workers: bool = Field(default=False, description="保持工作进程")
    
    # 数据划分
    train_ratio: float = Field(default=0.7, description="训练集比例", ge=0.1, le=0.9)
    val_ratio: float = Field(default=0.15, description="验证集比例", ge=0.1, le=0.5)
    
    # 信号处理
    normalization: Union[bool, str] = Field(default=True, description="归一化方式")
    window_size: int = Field(default=1024, description="窗口大小", ge=32)
    stride: int = Field(default=512, description="滑动步长", ge=1)
    truncate_lenth: int = Field(default=8192, description="最大长度限制", ge=32)
    
    # 数据类型
    dtype: str = Field(default="float32", description="数据类型")
    num_window: Optional[int] = Field(default=None, description="窗口数量")
    
    @field_validator('stride')
    @classmethod
    def validate_stride(cls, v, info):
        if hasattr(info, 'data') and 'window_size' in info.data and v > info.data['window_size']:
            raise ValueError("stride不能大于window_size")
        return v


class ModelConfig(BaseModel):
    """模型配置 - 控制模型架构和参数"""
    
    # 基础信息
    name: str = Field(..., description="模型名称")
    type: str = Field(..., description="模型类型")
    
    # 通用参数
    input_dim: int = Field(default=1, description="输入维度", ge=1)
    num_classes: Optional[int] = Field(default=None, description="分类类别数", ge=2)
    dropout: float = Field(default=0.1, description="Dropout概率", ge=0.0, le=1.0)
    activation: str = Field(default="relu", description="激活函数")
    
    # Transformer参数
    d_model: int = Field(default=128, description="模型维度", ge=16)
    num_heads: int = Field(default=8, description="注意力头数", ge=1)
    num_layers: int = Field(default=6, description="层数", ge=1, le=50)
    d_ff: Optional[int] = Field(default=None, description="前馈网络维度")
    
    # ISFM特有参数
    embedding: Optional[str] = Field(default=None, description="嵌入层类型")
    backbone: Optional[str] = Field(default=None, description="骨干网络类型")
    task_head: Optional[str] = Field(default=None, description="任务头类型")
    
    # Patch参数
    patch_size_L: int = Field(default=16, description="时间维度patch大小", ge=1)
    patch_size_C: int = Field(default=1, description="通道维度patch大小", ge=1)
    num_patches: int = Field(default=64, description="patch数量", ge=1)
    output_dim: int = Field(default=128, description="输出维度", ge=16)
    
    # CNN参数
    depth: Optional[int] = Field(default=None, description="网络深度", ge=1)
    in_channels: Optional[int] = Field(default=None, description="输入通道数", ge=1)
    hidden_dim: Optional[int] = Field(default=None, description="隐藏层维度")
    
    @field_validator('d_ff')
    @classmethod
    def set_d_ff_default(cls, v, info):
        if v is None and hasattr(info, 'data') and 'd_model' in info.data:
            return info.data['d_model'] * 4
        return v
    
    @model_validator(mode='after')
    def validate_isfm_config(self):
        """验证ISFM模型配置完整性"""
        if self.type == 'ISFM':
            required_fields = ['embedding', 'backbone', 'task_head']
            missing = [f for f in required_fields if not getattr(self, f, None)]
            if missing:
                raise ValueError(f"ISFM模型缺少必需字段: {missing}")
        return self


class TaskConfig(BaseModel):
    """任务配置 - 控制学习任务和训练参数"""
    
    # 任务定义
    name: str = Field(..., description="任务名称")
    type: str = Field(..., description="任务类型")
    
    # 数据设置
    target_system_id: Optional[List[int]] = Field(default=None, description="目标系统ID")
    source_domain_id: Optional[List[int]] = Field(default=None, description="源域ID")
    target_domain_id: Optional[List[int]] = Field(default=None, description="目标域ID")
    target_domain_num: Optional[int] = Field(default=None, description="目标域数量")
    
    # 训练参数
    epochs: int = Field(default=50, description="训练轮数", ge=1, le=1000)
    lr: float = Field(default=0.001, description="学习率", ge=1e-6, le=1.0)
    weight_decay: float = Field(default=0.0001, description="权重衰减", ge=0.0, le=1.0)
    optimizer: str = Field(default="adam", description="优化器")
    
    # 损失和指标
    loss: str = Field(default="CE", description="损失函数")
    metrics: List[str] = Field(default=["acc"], description="评估指标")
    
    # 调度器
    scheduler: bool = Field(default=False, description="启用学习率调度器")
    scheduler_type: str = Field(default="step", description="调度器类型")
    step_size: int = Field(default=10, description="调度器步长", ge=1)
    gamma: float = Field(default=0.5, description="学习率衰减因子", ge=0.1, le=1.0)
    
    # 早停
    early_stopping: bool = Field(default=True, description="启用早停")
    es_patience: int = Field(default=10, description="早停耐心值", ge=1, le=100)
    
    # 数据加载（任务级覆盖）
    batch_size: Optional[int] = Field(default=None, description="任务特定批次大小")
    num_workers: Optional[int] = Field(default=None, description="任务特定工作进程数")
    pin_memory: Optional[bool] = Field(default=None, description="任务特定内存固定")
    shuffle: bool = Field(default=True, description="打乱数据")
    log_interval: int = Field(default=50, description="日志间隔", ge=1)
    
    # 多任务特有
    task_list: Optional[List[str]] = Field(default=None, description="多任务列表")
    loss_weights: Optional[Dict[str, float]] = Field(default=None, description="损失权重")
    
    # Few-shot特有
    num_support: Optional[int] = Field(default=None, description="支撑集大小", ge=1)
    num_query: Optional[int] = Field(default=None, description="查询集大小", ge=1)
    num_episodes: Optional[int] = Field(default=None, description="训练episodes", ge=1)


class TrainerConfig(BaseModel):
    """训练器配置 - 控制训练过程和硬件设置"""
    
    # 基础设置
    name: str = Field(default="Default_trainer", description="训练器名称")
    num_epochs: int = Field(default=50, description="训练轮数", ge=1, le=1000)
    
    # 硬件设置
    gpus: Union[int, List[int]] = Field(default=1, description="GPU设置")
    device: str = Field(default="auto", description="计算设备")
    accelerator: str = Field(default="auto", description="加速器类型")
    
    # 训练优化
    mixed_precision: bool = Field(default=False, description="混合精度训练")
    gradient_clip_val: Optional[float] = Field(default=None, description="梯度裁剪", ge=0.0)
    accumulate_grad_batches: int = Field(default=1, description="梯度累积", ge=1)
    
    # 验证和检查点
    check_val_every_n_epoch: int = Field(default=1, description="验证频率", ge=1)
    val_check_interval: Union[int, float] = Field(default=1.0, description="验证间隔")
    enable_checkpointing: bool = Field(default=True, description="启用检查点")
    save_top_k: int = Field(default=3, description="保存最佳k个模型", ge=1)
    monitor_metric: str = Field(default="val_loss", description="监控指标")
    mode: str = Field(default="min", description="监控模式")
    
    # 早停
    early_stopping: bool = Field(default=True, description="启用早停")
    patience: int = Field(default=10, description="早停耐心值", ge=1)
    min_delta: float = Field(default=0.001, description="最小变化量", ge=0.0)
    
    # 日志和监控
    wandb: bool = Field(default=False, description="启用WandB")
    swanlab: bool = Field(default=False, description="启用SwanLab")
    log_every_n_steps: int = Field(default=50, description="日志频率", ge=1)
    enable_progress_bar: bool = Field(default=True, description="显示进度条")
    
    # 高级功能
    pruning: bool = Field(default=False, description="启用模型剪枝")
    profiler: Optional[str] = Field(default=None, description="性能分析器")
    auto_scale_batch_size: bool = Field(default=False, description="自动批次大小")
    auto_lr_find: bool = Field(default=False, description="自动学习率搜索")
    
    @field_validator('device')
    @classmethod
    def set_device_auto(cls, v):
        if v == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return v


# ==================== 主配置类 ====================

class PHMConfig(BaseModel):
    """PHM-Vibench主配置类"""
    
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    data: DataConfig
    model: ModelConfig  
    task: TaskConfig
    trainer: TrainerConfig = Field(default_factory=TrainerConfig)
    
    class Config:
        # 允许通过 model__d_model 方式设置嵌套属性
        allow_population_by_field_name = True
        validate_assignment = True
    
    def __init__(self, **kwargs):
        """支持双下划线语法设置嵌套参数"""
        # 处理双下划线语法
        nested_updates = {}
        regular_kwargs = {}
        
        for key, value in kwargs.items():
            if '__' in key:
                section, param = key.split('__', 1)
                if section not in nested_updates:
                    nested_updates[section] = {}
                nested_updates[section][param] = value
            else:
                regular_kwargs[key] = value
        
        # 合并嵌套更新
        for section, updates in nested_updates.items():
            if section in regular_kwargs:
                if isinstance(regular_kwargs[section], dict):
                    regular_kwargs[section].update(updates)
                else:
                    # 如果已经是对象，需要转换
                    section_dict = regular_kwargs[section].dict() if hasattr(regular_kwargs[section], 'dict') else {}
                    section_dict.update(updates)
                    regular_kwargs[section] = section_dict
            else:
                regular_kwargs[section] = updates
        
        super().__init__(**regular_kwargs)
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """验证配置间的一致性"""
        
        # 验证训练轮数一致性
        task_epochs = self.task.epochs if hasattr(self.task, 'epochs') else None
        trainer_epochs = self.trainer.num_epochs if hasattr(self.trainer, 'num_epochs') else None
        
        if task_epochs and trainer_epochs and task_epochs != trainer_epochs:
            # 自动同步为trainer的值
            if hasattr(self.task, 'epochs'):
                self.task.epochs = trainer_epochs
        
        # 验证批次大小一致性
        data_batch = self.data.batch_size if hasattr(self.data, 'batch_size') else None
        task_batch = getattr(self.task, 'batch_size', None)
        
        if task_batch and data_batch and task_batch != data_batch:
            # 任务级别的批次大小优先
            if hasattr(self.data, 'batch_size'):
                self.data.batch_size = task_batch
        
        return self
    
    def to_legacy_dict(self) -> Dict[str, Any]:
        """转换为旧格式字典，用于向后兼容"""
        return {
            'environment': self.environment.dict(),
            'data': self.data.dict(), 
            'model': self.model.dict(),
            'task': self.task.dict(),
            'trainer': self.trainer.dict()
        }
    
    def save_yaml(self, path: Union[str, Path], minimal: bool = False) -> None:
        """保存为YAML格式"""
        import yaml
        
        config_dict = self.to_legacy_dict()
        
        if minimal:
            # 只保存非默认值
            config_dict = self._filter_defaults(config_dict)
        
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    
    def _filter_defaults(self, config_dict: Dict) -> Dict:
        """过滤默认值，只保留修改过的参数"""
        # 创建默认配置用于比较
        default_env = EnvironmentConfig().dict()
        default_trainer = TrainerConfig().dict()
        
        filtered = {}
        
        # 环境配置过滤
        env_filtered = {k: v for k, v in config_dict['environment'].items() 
                       if k not in default_env or v != default_env[k]}
        if env_filtered:
            filtered['environment'] = env_filtered
        
        # 数据和模型配置通常都需要保留（因为有必需字段）
        filtered['data'] = config_dict['data']
        filtered['model'] = config_dict['model'] 
        filtered['task'] = config_dict['task']
        
        # 训练器配置过滤
        trainer_filtered = {k: v for k, v in config_dict['trainer'].items()
                           if k not in default_trainer or v != default_trainer[k]}
        if trainer_filtered:
            filtered['trainer'] = trainer_filtered
        
        return filtered


# ==================== 辅助函数 ====================

def get_model_choices() -> Dict[str, List[str]]:
    """获取可用的模型选择"""
    return {
        'CNN': ['ResNet1D', 'AttentionCNN', 'MultiScaleCNN', 'MobileNet1D', 'TCN'],
        'RNN': ['AttentionLSTM', 'AttentionGRU', 'ConvLSTM', 'ResidualRNN'],
        'Transformer': ['PatchTST', 'Autoformer', 'Informer', 'Linformer'],
        'ISFM': ['M_01_ISFM', 'M_02_ISFM'],  # M_03不推荐
        'MLP': ['Dlinear', 'MLPMixer', 'ResNetMLP', 'DenseNetMLP'],
        'NO': ['FNO', 'DeepONet', 'GraphNO', 'NeuralODE'],
        'FewShot': ['ProtoNet', 'Matching']
    }


def get_task_choices() -> Dict[str, List[str]]:
    """获取可用的任务选择"""
    return {
        'DG': ['classification', 'prediction'],
        'CDDG': ['classification'],
        'FS': ['classification'],
        'GFS': ['classification'],
        'Pretrain': ['pretraining', 'prediction'],
        'Multitask': ['multitask']
    }


def validate_config(config: PHMConfig) -> List[str]:
    """验证配置并返回警告信息"""
    warnings = []
    
    # 检查模型选择
    model_choices = get_model_choices()
    if config.model.type in model_choices:
        if config.model.name not in model_choices[config.model.type]:
            warnings.append(f"模型组合可能无效: {config.model.type}.{config.model.name}")
    
    # 检查任务选择
    task_choices = get_task_choices()
    if config.task.type in task_choices:
        if config.task.name not in task_choices[config.task.type]:
            warnings.append(f"任务组合可能无效: {config.task.type}.{config.task.name}")
    
    # 性能建议
    if config.data.num_workers < 4:
        warnings.append("建议增加data.num_workers到4-8以提升性能")
    
    if not config.data.pin_memory:
        warnings.append("建议启用data.pin_memory以加速GPU训练")
    
    if config.model.dropout == 0:
        warnings.append("建议设置model.dropout>0以防止过拟合")
    
    return warnings


# ==================== 使用示例 ====================

if __name__ == "__main__":
    # 创建基础配置
    config = PHMConfig(
        data=DataConfig(
            data_dir="./data",
            metadata_file="metadata.xlsx"
        ),
        model=ModelConfig(
            name="ResNet1D",
            type="CNN",
            num_classes=4
        ),
        task=TaskConfig(
            name="classification",
            type="DG"
        )
    )
    
    print("✅ 基础配置创建成功")
    print(f"模型: {config.model.type}.{config.model.name}")
    print(f"任务: {config.task.type}.{config.task.name}")
    
    # 使用双下划线语法
    advanced_config = PHMConfig(
        data__data_dir="./data",
        data__metadata_file="metadata.xlsx",
        model__name="M_01_ISFM",
        model__type="ISFM",
        model__embedding="E_01_HSE",
        model__backbone="B_08_PatchTST",
        model__task_head="H_01_Linear_cla",
        model__d_model=256,
        task__name="classification",
        task__type="DG",
        trainer__num_epochs=100
    )
    
    print("\n✅ 高级配置创建成功")
    print(f"模型维度: {advanced_config.model.d_model}")
    print(f"训练轮数: {advanced_config.trainer.num_epochs}")
    
    # 验证配置
    warnings = validate_config(config)
    if warnings:
        print(f"\n⚠️  配置警告: {warnings}")
    else:
        print("\n✅ 配置验证通过")