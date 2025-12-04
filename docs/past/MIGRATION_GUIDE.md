# 配置系统迁移指南 🔄

> **从复杂YAML到简洁Python的完整迁移方案**

## 迁移概述

PHM-Vibench 2.0配置系统提供了从旧YAML配置到新Pydantic配置的无缝迁移方案，确保现有项目能够平滑升级。

### 📊 迁移前后对比

| 方面 | 旧YAML系统 | 新Pydantic系统 | 改进 |
|------|-----------|----------------|------|
| **配置长度** | 110+ 行 | 5-20 行 | 减少80%+ |
| **错误检测** | 运行时发现 | 编码时发现 | 提前发现问题 |
| **IDE支持** | 无智能提示 | 完整自动补全 | 开发效率提升 |
| **类型安全** | 字符串容易出错 | 强类型检查 | 可靠性提升 |
| **配置复用** | 复制粘贴 | 继承组合 | 维护性提升 |

## 🚀 快速迁移

### 第一步：安装新配置系统

新配置系统已集成在PHM-Vibench中，无需额外安装：

```python
from src.configs import PHMConfig, load_config, ConfigManager
```

### 第二步：识别迁移场景

根据你的使用情况选择迁移方式：

#### 场景A：完全替换（推荐）
```python
# 旧方式
python main.py --config configs/demo/CWRU.yaml

# 新方式  
python main.py --config quickstart
```

#### 场景B：渐进迁移
```python
# 保留现有YAML，使用新系统加载
config = load_config("configs/demo/CWRU.yaml")  # 兼容旧YAML
```

#### 场景C：混合使用
```python
# YAML基础配置 + Python覆盖
config = load_config("old_config.yaml", {
    "model__d_model": 256,
    "task__epochs": 100
})
```

## 📋 具体迁移步骤

### 步骤1：分析现有配置

使用分析工具了解现有配置：

```python
from src.configs.config_manager import ConfigManager

manager = ConfigManager()

# 分析现有YAML配置
config = manager.load("configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml")
print("当前配置信息:")
print(f"  实验名: {config.environment.experiment_name}")
print(f"  模型: {config.model.type}.{config.model.name}")
print(f"  任务: {config.task.type}")
print(f"  数据: {config.data.metadata_file}")
```

### 步骤2：选择合适的预设

找到最接近你需求的预设：

```python
from src.configs.presets import list_presets

# 查看所有预设
presets = list_presets()
for name, desc in presets.items():
    print(f"{name}: {desc}")

# 测试不同预设
test_configs = ["quickstart", "basic", "isfm", "research"]
for preset in test_configs:
    config = load_config(preset)
    print(f"\n{preset} 预设:")
    print(f"  模型: {config.model.name}")
    print(f"  任务: {config.task.name}")
    print(f"  数据: {config.data.metadata_file}")
```

### 步骤3：创建迁移配置

基于选择的预设创建新配置：

```python
# 原始复杂YAML配置的Python替代
original_config = PHMConfig(
    # 环境设置
    environment__experiment_name="CWRU_THU_ISFM_experiment",
    environment__project="multi_domain_generalization", 
    environment__seed=42,
    
    # 数据设置  
    data__data_dir="/home/user/data/PHMbenchdata/PHM-Vibench",
    data__metadata_file="metadata_6_1.xlsx",
    data__batch_size=32,
    
    # 模型设置
    model__name="M_01_ISFM",
    model__type="ISFM",
    model__d_model=64,
    model__num_heads=4,
    
    # 任务设置
    task__name="classification", 
    task__type="CDDG",
    task__epochs=10,
    task__lr=0.001,
    
    # 训练设置
    trainer__num_epochs=10,
    trainer__gpus=1
)
```

### 步骤4：验证迁移结果

确保迁移的配置正确：

```python
# 验证新配置
is_valid, errors, warnings = manager.validate(original_config)

if is_valid:
    print("✅ 配置验证通过")
else:
    print("❌ 配置验证失败:")
    for error in errors:
        print(f"  - {error}")

if warnings:
    print("⚠️  注意事项:")
    for warning in warnings:
        print(f"  - {warning}")
```

### 步骤5：保存新配置

将新配置保存为不同格式：

```python
# 保存为简化YAML（推荐）
manager.save(original_config, "migrated_config.yaml", minimal=True)

# 保存为Python配置
manager.save(original_config, "migrated_config.py", format="py")

# 保存为JSON格式
manager.save(original_config, "migrated_config.json", format="json")

print("✅ 迁移完成！新配置文件已生成")
```

## 🔄 自动迁移工具

### 批量迁移脚本

创建自动迁移脚本：

```python
#!/usr/bin/env python3
"""
批量迁移YAML配置到Pydantic配置
"""

import os
from pathlib import Path
from src.configs.config_manager import ConfigManager

def migrate_configs(input_dir, output_dir):
    """批量迁移配置文件"""
    manager = ConfigManager()
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    yaml_files = list(input_path.glob("**/*.yaml")) + list(input_path.glob("**/*.yml"))
    
    print(f"🔍 找到 {len(yaml_files)} 个YAML配置文件")
    
    successful = 0
    failed = 0
    
    for yaml_file in yaml_files:
        try:
            # 加载YAML配置
            config = manager.load(yaml_file)
            
            # 生成输出文件名
            relative_path = yaml_file.relative_to(input_path)
            output_file = output_path / relative_path.with_suffix('.py')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为Python配置
            manager.save(config, output_file, format="py", add_comments=True)
            
            print(f"✅ {yaml_file.name} -> {output_file.name}")
            successful += 1
            
        except Exception as e:
            print(f"❌ {yaml_file.name}: {e}")
            failed += 1
    
    print(f"\n📊 迁移完成: {successful} 成功, {failed} 失败")

if __name__ == "__main__":
    # 迁移所有demo配置
    migrate_configs("configs/demo", "configs/migrated")
```

### 配置差异分析

比较迁移前后的配置：

```python
def analyze_migration(old_yaml, new_config):
    """分析迁移前后的配置差异"""
    manager = ConfigManager()
    
    # 加载两个配置
    old_config = manager.load(old_yaml)  
    
    # 比较差异
    diff = manager.compare(old_config, new_config)
    
    print(f"📊 配置迁移分析:")
    print(f"  总差异: {diff['total_differences']} 项")
    print(f"  新增: {diff['summary']['added']} 项")
    print(f"  修改: {diff['summary']['modified']} 项") 
    print(f"  删除: {diff['summary']['removed']} 项")
    
    if diff['differences']:
        print(f"\n📋 详细差异:")
        for d in diff['differences'][:5]:  # 显示前5个差异
            if d['type'] == 'modified':
                print(f"  修改 {d['path']}: {d['old_value']} -> {d['new_value']}")
            elif d['type'] == 'added':
                print(f"  新增 {d['path']}: {d['value']}")
            elif d['type'] == 'removed':
                print(f"  删除 {d['path']}: {d['value']}")
```

## 📚 迁移示例

### 示例1：基础CNN配置迁移

**旧YAML配置 (50+行):**
```yaml
environment:
  project: "basic_cnn_experiment" 
  seed: 42
  output_dir: "results/cnn"
  
data:
  data_dir: "./data"
  metadata_file: "metadata.xlsx"
  batch_size: 32
  num_workers: 4
  window_size: 4096
  
model:
  name: "ResNet1D"
  type: "CNN"
  input_dim: 2
  num_classes: 4
  
task:
  name: "classification"
  type: "DG" 
  epochs: 50
  lr: 0.001
  
trainer:
  num_epochs: 50
  gpus: 1
  device: "cuda"
```

**新Python配置 (5行):**
```python
config = load_config("basic", {
    "environment__project": "basic_cnn_experiment",
    "data__window_size": 4096,
    "task__epochs": 50,
    "trainer__device": "cuda"
})
```

### 示例2：ISFM高级配置迁移

**旧YAML配置 (110+行):**
```yaml
# 大量复杂配置...
environment:
  project: "ISFM_research"
  # ... 20+ 行环境配置
  
data:  
  # ... 30+ 行数据配置
  
model:
  name: "M_01_ISFM"
  type: "ISFM"
  # ... 40+ 行模型配置
  
task:
  # ... 20+ 行任务配置
```

**新Python配置 (8行):**
```python
config = load_config("isfm", {
    "environment__project": "ISFM_research",
    "model__d_model": 256,
    "model__num_heads": 8, 
    "task__target_system_id": [6],
    "task__epochs": 100,
    "trainer__gpus": 2
})
```

### 示例3：多数据集配置迁移

**旧方式：多个独立YAML**
```
configs/
├── CWRU.yaml       (110+ 行)
├── XJTU.yaml       (110+ 行)  
├── FEMTO.yaml      (110+ 行)
└── THU.yaml        (110+ 行)
```

**新方式：统一配置+动态参数**
```python
# 单一配置模板
datasets = ["CWRU", "XJTU", "FEMTO", "THU"] 

for dataset in datasets:
    config = load_config("benchmark", {
        "data__metadata_file": f"{dataset}_metadata.xlsx",
        "environment__experiment_name": f"benchmark_{dataset}",
        "task__target_system_id": DATASET_MAPPING[dataset]
    })
    # 运行实验...
```

## ⚠️ 注意事项

### 兼容性说明

1. **向后兼容**: 新系统完全支持加载现有YAML配置
2. **文件格式**: 支持 .yaml, .yml, .json, .py 格式 
3. **参数命名**: 保持与原YAML相同的参数名称
4. **默认值**: 新系统提供更智能的默认值

### 潜在问题

1. **路径问题**: 
   ```python
   # 可能需要调整相对路径
   config.data.data_dir = str(Path.cwd() / "data")
   ```

2. **类型转换**:
   ```python
   # YAML中的字符串数字在新系统中会自动转换
   # "42" -> 42 (int)
   # "0.001" -> 0.001 (float)  
   ```

3. **参数验证**:
   ```python
   # 新系统会验证参数范围和类型
   config.task.lr = 0.001  # ✅ 有效
   config.task.lr = -0.001 # ❌ 无效，学习率必须为正
   ```

## 🎯 迁移策略建议

### 新项目（推荐）
直接使用新Pydantic配置系统：
```python
config = load_config("quickstart")  # 立即开始
```

### 现有项目（渐进式）
1. **第一阶段**: 使用新系统加载现有YAML
   ```python
   config = load_config("existing_config.yaml")
   ```

2. **第二阶段**: 简化为预设+覆盖
   ```python
   config = load_config("isfm", overrides="my_overrides.yaml")
   ```

3. **第三阶段**: 完全迁移到Python配置
   ```python
   config = PHMConfig(model__name="MyModel", task__epochs=100)
   ```

### 团队项目（混合式）
- **配置模板**: 团队共享预设配置
- **个人定制**: 个人覆盖参数
- **版本管理**: 配置文件纳入Git管理

```python
# 团队基础配置
base_config = load_config("team_standard")

# 个人实验配置  
my_config = load_config("team_standard", {
    "model__d_model": 512,  # 个人调优
    "task__lr": 0.0005,     # 实验参数
    "environment__experiment_name": "zhang_experiment_v2"
})
```

## 🆘 故障排除

### 常见迁移错误

1. **配置文件找不到**
   ```python
   # 问题
   config = load_config("config.yaml")  # FileNotFoundError
   
   # 解决
   config = manager.load("./configs/config.yaml")  # 使用完整路径
   ```

2. **参数类型错误**
   ```python
   # 问题  
   config = PHMConfig(task__epochs="100")  # 字符串
   
   # 解决
   config = PHMConfig(task__epochs=100)    # 整数
   ```

3. **嵌套参数错误**
   ```python
   # 问题
   config.model.d_model = 256  # 可能失败
   
   # 解决  
   config = PHMConfig(model__d_model=256)  # 使用双下划线
   ```

### 获取帮助

- **配置验证**: `manager.validate(config)` 
- **参数选项**: `get_model_choices()`, `get_task_choices()`
- **预设列表**: `list_presets()`
- **配置比较**: `manager.compare(config1, config2)`

## 🎉 迁移完成检查清单

- [ ] 识别所有现有YAML配置文件
- [ ] 选择合适的预设配置
- [ ] 创建简化的Python配置  
- [ ] 验证配置正确性
- [ ] 保存新配置文件
- [ ] 更新实验脚本
- [ ] 测试运行实验
- [ ] 清理旧配置文件（可选）

完成迁移后，你将享受：
- 🚀 5-10倍的配置编写效率
- 🛡️ 类型安全和自动验证
- 💡 IDE智能提示和自动补全  
- 🔧 强大的配置管理功能

**立即开始迁移，体验现代化的PHM配置管理！**