# PHM-Vibench Config Manager Skill

## 功能概述

PHM-Vibench配置管理核心Skill，完全拥抱v5.0配置系统的极简而强大的实验管理能力。

**核心优势**:
- **统一接口**: 单一`load_config()`函数处理所有配置需求
- **4×4灵活性**: 支持预设/文件/字典/ConfigWrapper × 4种覆盖方式
- **智能合并**: 递归合并嵌套配置，点号展开自动处理
- **链式操作**: 支持copy().update()链式配置构建
- **100%兼容**: 所有现有Pipeline无需修改即可使用
- **Vbench标准**: 自动验证E_01_HSE_v2、B_08_PatchTST等标准命名

## 主要功能

### 1. Vbench标准配置预设加载
```bash
skill: "config-manager"
```

**支持预设**:
- `quickstart`: 快速开始配置
- `isfm`: ISFM基础模型配置
- `gfs`: 广义少样本学习配置
- `hse_prompt`: HSE-Prompt实验配置
- `baseline`: 基线实验配置

### 2. HSE-Prompt实验配置生成
自动创建符合Vbench标准的HSE-Prompt实验配置，包括：
- 模型配置: M_02_ISFM_Prompt + E_01_HSE_v2 + 标准骨干网络
- 任务配置: hse_contrastive + 距离分类头
- 数据配置: target_system_id跨域设置
- 训练配置: 优化的超参数

### 3. Vbench标准命名验证
自动验证配置文件是否符合PHM-Vibench v5.0标准：
- ✅ `embedding: "E_01_HSE_v2"` (正确)
- ❌ `embedding: "HSE_Prompt"` (错误，缺少E_01_前缀)
- ✅ `backbone: "B_08_PatchTST"` (正确)
- ❌ `backbone: "PatchTST"` (错误，缺少B_08_前缀)

### 4. 配置文件自动修复
检测到非标准命名时，自动提供修复建议：
- 组件名称标准化
- 缺失字段自动补充
- 不兼容组合检测和建议

### 5. 论文实验配置生成
为paper/2025-10_foundation_model_0_metric/自动生成标准化配置：
- Table1基线实验配置
- Table2小样本实验配置
- Table3鲁棒性实验配置
- Table4消融实验配置

## 使用方法

### 基本用法

#### 加载Vbench预设配置
```python
# 自然语言方式
"加载ISFM预设配置"
"创建快速开始实验配置"
"加载广义少样本学习配置"
```

#### 创建HSE-Prompt配置
```python
# 中文交互示例
"创建一个HSE-Prompt跨域实验配置，使用CWRU到THU的数据集"
"生成HSE-Prompt小样本学习配置，目标87.6%准确率"
"创建符合Vbench标准的E_01_HSE_v2实验配置"
```

#### 验证配置文件
```python
# 验证示例
"验证这个配置是否符合Vbench标准命名"
"检查HSE-Prompt配置是否有命名错误"
"自动修复配置文件中的组件命名"
```

### 高级配置

#### 自定义配置生成
```python
# 高级示例
config = skill.create_custom_config(
    task_type="cross_domain",
    model_components={
        "embedding": "E_01_HSE_v2",
        "backbone": "B_08_PatchTST",
        "task_head": "H_02_distance_cla"
    },
    data_config={
        "target_system_id": [1, 2, 6, 5, 12],
        "cross_domain": True
    },
    training_config={
        "lr": 0.001,
        "batch_size": 64,
        "max_epochs": 100
    }
)
```

#### 论文实验配置
```python
# 论文实验示例
"生成论文Table1的基线实验配置"
"创建Table2的小样本学习配置，5-shot设置"
"设置Table3的鲁棒性测试，噪声级别[0.1, 0.2, 0.3]"
"配置Table4的HSE-Prompt消融实验"
```

## 资源需求

- **内存**: 最低2GB，推荐4GB
- **磁盘**: 100MB (配置文件和模板)
- **网络**: 不需要 (本地配置管理)
- **依赖**: yaml, pyyaml, pathlib

## 输出格式

### 配置文件输出
```yaml
# 标准Vbench配置示例
data:
  target_system_id: [1, 2, 6, 5, 12]  # CWRU, XJTU, THU, Ottawa, JNU

model:
  name: "M_02_ISFM_Prompt"
  embedding: "E_01_HSE_v2"      # ✅ Vbench标准命名
  backbone: "B_08_PatchTST"     # ✅ Vbench标准命名
  task_head: "H_02_distance_cla" # ✅ Vbench标准命名

task:
  name: "hse_contrastive"
  type: "classification"
  lr: 0.001
  batch_size: 64

trainer:
  name: "default_trainer"
  max_epochs: 100
  gpu_ids: [0]
```

### 验证报告输出
```python
# 配置验证结果
{
    "status": "valid",  # valid | warning | error
    "vbench_compliant": True,
    "issues": [],
    "suggestions": [
        "建议使用E_01_HSE_v2而不是HSE_Prompt",
        "考虑添加B_08_PatchTST作为骨干网络"
    ],
    "auto_fixed": [
        "embedding: 'HSE_Prompt' -> 'E_01_HSE_v2'",
        "backbone: 'PatchTST' -> 'B_08_PatchTST'"
    ]
}
```

## 依赖关系

### 必需的父Skills
- 无 (基础Skill)

### 嵌套的子Skills
- `vbench-validator`: Vbench标准验证
- `naming-convention-checker`: 命名规范检查
- `config-template-generator`: 配置模板生成

### 外部依赖
- `src.configs`: PHM-Vibench配置系统
- `yaml`: YAML文件处理
- `pathlib`: 路径处理

## 最佳实践

### 1. 配置文件组织
```
configs/
├── vbench_standard/          # Vbench标准配置
│   ├── isfm.yaml            # ISFM基础模型配置
│   ├── hse_prompt.yaml      # HSE-Prompt实验配置
│   └── baseline/            # 基线实验配置
├── paper_experiments/        # 论文实验配置
│   ├── table1/              # Table1基线实验
│   ├── table2/              # Table2小样本实验
│   ├── table3/              # Table3鲁棒性实验
│   └── table4/              # Table4消融实验
└── custom/                  # 自定义配置
```

### 2. 配置命名规范
- 使用Vbench标准组件命名
- 遵循YAML最佳实践
- 添加必要的注释说明
- 保持配置文件的模块化结构

### 3. 版本管理
- 为重要实验配置添加版本号
- 保留实验配置的历史记录
- 使用git管理配置文件变更
- 为不同实验阶段创建专门配置

## 错误处理

### 常见问题及解决方案

1. **组件命名错误**
   ```
   错误: embedding: "HSE_Prompt"
   修复: embedding: "E_01_HSE_v2"
   ```

2. **数据集配置错误**
   ```
   错误: target_domains: ["CWRU", "XJTU"]
   修复: target_system_id: [1, 2]
   ```

3. **模型组合不兼容**
   ```
   检测: ISFM_Prompt + 不兼容的task_head
   建议: 使用H_02_distance_cla或H_01_Linear_cla
   ```

## 更新日志

### v1.0.0 (2025-01-13)
- 初始版本发布
- 支持Vbench标准配置验证
- 实现HSE-Prompt配置生成
- 添加论文实验配置支持
- 集成中文自然语言交互

---

**注意**: 此Skill完全兼容PHM-Vibench v5.0配置系统，确保100%向后兼容性和标准化的实验配置管理。