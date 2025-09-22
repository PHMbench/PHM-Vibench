# 阶段2: 数据准备指南

工业振动数据集准备、验证和预处理的完整指南。

## 📋 本阶段目标

- [x] 验证数据集完整性和格式
- [x] 检查ContrastiveIDTask兼容性
- [x] 分析数据质量和分布
- [x] 生成数据准备报告

## 🚀 快速开始

### 1. 单数据集验证
```bash
python data_validation.py --dataset CWRU --metadata_path /path/to/metadata_CWRU.xlsx
```

### 2. 多数据集批量验证
```bash
# 验证多个数据集
for dataset in CWRU XJTU PU FEMTO; do
    python data_validation.py --dataset $dataset --quick
done
```

### 3. 快速兼容性检查
```bash
python data_validation.py --dataset CWRU --compatibility_only
```

## 🛠️ 核心功能详解

### data_validation.py
**主要功能**: 全面的数据集验证和分析工具

#### 基本验证
```bash
# 标准验证流程
python data_validation.py --dataset CWRU

# 快速检查（跳过详细分析）
python data_validation.py --dataset CWRU --quick

# 只检查兼容性
python data_validation.py --dataset CWRU --compatibility_only
```

#### 高级分析
```bash
# 详细数据分析
python data_validation.py --dataset CWRU --analyze --verbose

# 生成可视化报告
python data_validation.py --dataset CWRU --visualize --output_dir reports/

# 多数据集对比分析
python data_validation.py --datasets CWRU,XJTU,PU --compare
```

## 📊 验证检查项目

### 🔍 基础完整性检查
- [x] **H5文件存在性**: 检查所有H5数据文件是否存在
- [x] **元数据一致性**: 验证metadata.xlsx与实际数据匹配
- [x] **文件完整性**: 检查H5文件是否可正常读取
- [x] **数据结构**: 验证信号维度和格式

### 📈 数据质量分析
- [x] **信号长度分布**: 分析时间序列长度统计
- [x] **通道数验证**: 确认信号通道数一致性
- [x] **数值范围检查**: 检测异常值和数据范围
- [x] **标签分布**: 分析类别标签的平衡性

### 🧪 ContrastiveIDTask兼容性
- [x] **最小长度验证**: 确保信号长度≥window_size
- [x] **窗口采样测试**: 验证窗口生成功能
- [x] **批处理测试**: 测试批次准备流程
- [x] **内存需求估算**: 预估训练内存占用

## 📋 验证报告解读

### ✅ 正常输出示例
```
📊 数据集分析报告: CWRU
=====================================
✅ 基础检查:
   - H5文件: 2,400个 (100%完整)
   - 元数据匹配: 2,400/2,400
   - 平均信号长度: 121,945 ± 15,234
   - 通道数: 2 (一致)

✅ 质量分析:
   - 数值范围: [-5.23, 4.87] (正常)
   - 异常值比例: 0.02% (可接受)
   - 标签分布: 均衡 (最大偏差<10%)

✅ ContrastiveIDTask兼容性:
   - 最小长度: 8,192 > window_size(256) ✓
   - 窗口采样: 成功率100%
   - 批处理测试: 通过
   - 预估内存: 1.2GB (batch_size=32)

🎉 数据集验证通过，可用于ContrastiveIDTask训练
```

### ⚠️ 问题报告示例
```
⚠️ 数据集问题报告: EXAMPLE_DATASET
=====================================
❌ 发现问题:
   1. 缺失H5文件: 15个样本无对应数据文件
   2. 长度不足样本: 23个样本 < 最小窗口大小
   3. 异常值过多: Channel_1中8.3%数据为NaN
   4. 标签不平衡: Class_0占83.2%，建议重新平衡

📝 修复建议:
   - 补充缺失的H5文件或从元数据中移除
   - 调整window_size≤2048适应短信号
   - 清理或插值处理NaN值
   - 考虑数据增强或重采样平衡标签
```

## 🔧 高级功能使用

### 批量数据集处理
```bash
# 创建数据集处理脚本
cat > validate_all.sh << EOF
#!/bin/bash
datasets=("CWRU" "XJTU" "PU" "FEMTO" "IMS" "MFPT")
for dataset in "\${datasets[@]}"; do
    echo "验证数据集: \$dataset"
    python data_validation.py --dataset \$dataset --analyze --output_dir "reports/\$dataset/"
done
EOF

chmod +x validate_all.sh
./validate_all.sh
```

### 自定义验证规则
```python
# custom_validation.py
from data_validation import DatasetValidator

# 创建自定义验证器
validator = DatasetValidator(
    min_signal_length=512,
    max_nan_ratio=0.05,
    required_channels=2,
    min_samples_per_class=100
)

# 运行自定义验证
result = validator.validate_dataset("CUSTOM_DATASET")
print(result.summary())
```

### 跨数据集兼容性分析
```bash
# 分析多数据集的兼容性
python data_validation.py \
    --datasets CWRU,XJTU,PU \
    --cross_compatibility \
    --output_report cross_dataset_analysis.json
```

## 📊 数据预处理建议

### 信号长度标准化
```python
# 根据分析结果调整参数
recommended_config = {
    'window_size': 256,    # 基于最短信号长度
    'stride': 128,         # 50%重叠
    'truncate_length': 4096,  # 基于95%分位数
}
```

### 批大小优化
```bash
# 基于内存分析调整批大小
python data_validation.py --dataset CWRU --memory_analysis --batch_sizes 8,16,32,64
```

**输出示例**:
```
💾 内存使用分析:
   batch_size=8:  0.3GB (推荐)
   batch_size=16: 0.6GB (推荐)
   batch_size=32: 1.2GB (可行)
   batch_size=64: 2.4GB (需要>4GB GPU)
```

## 🎯 数据质量优化

### 异常值处理
```python
# 检测和处理异常值
from data_validation import DataCleaner

cleaner = DataCleaner()

# 检测异常值
outliers = cleaner.detect_outliers(dataset_path, method='iqr')

# 清理策略选择
clean_dataset = cleaner.clean(
    dataset_path,
    strategy='interpolate',  # 'remove', 'interpolate', 'clip'
    outlier_threshold=3.0
)
```

### 数据增强建议
```python
# 根据分析结果制定增强策略
augmentation_config = {
    'noise_injection': 0.02,      # 基于SNR分析
    'time_warping': 0.1,          # 基于长度变异性
    'frequency_masking': 0.15,     # 基于频域特征
    'mixup_alpha': 0.2            # 标签平衡策略
}
```

## 🔍 故障排除

### ❌ H5文件读取失败
```bash
# 检查H5文件完整性
python -c "
import h5py
try:
    with h5py.File('problem_file.h5', 'r') as f:
        print(f'Keys: {list(f.keys())}')
except Exception as e:
    print(f'Error: {e}')
"

# 修复损坏的H5文件
python data_validation.py --dataset DATASET --repair_h5
```

### ❌ 内存不足
```bash
# 使用流式处理模式
python data_validation.py --dataset LARGE_DATASET --streaming --chunk_size 100
```

### ❌ 元数据不匹配
```bash
# 生成新的元数据文件
python data_validation.py --dataset DATASET --generate_metadata --output metadata_new.xlsx
```

## 📈 性能优化

### 并行处理
```bash
# 多进程验证
python data_validation.py --dataset LARGE_DATASET --parallel --num_workers 4

# GPU加速分析
python data_validation.py --dataset DATASET --use_gpu --gpu_id 0
```

### 缓存机制
```python
# 启用验证结果缓存
export DATA_VALIDATION_CACHE=1
python data_validation.py --dataset DATASET  # 首次运行，建立缓存
python data_validation.py --dataset DATASET  # 后续运行使用缓存
```

## 🎯 进入下一阶段

### 检查清单
- [ ] 所有目标数据集验证通过
- [ ] ContrastiveIDTask兼容性确认
- [ ] 数据质量问题已修复
- [ ] 最优参数配置已确定

### 下一步行动
```bash
# 进入实验执行阶段
cd ../03_experiments/

# 使用验证后的配置运行实验
python multi_dataset_runner.py \
    --datasets CWRU \
    --config validated_config.yaml
```

## 📚 深入学习

### 数据集特性参考
| 数据集 | 样本数 | 信号长度 | 通道数 | 故障类型 | 特点 |
|--------|--------|----------|--------|----------|------|
| CWRU | 2,400 | ~120K | 2 | 4 | 标准基准数据集 |
| XJTU | 15,000 | ~32K | 2 | 5 | 真实工况数据 |
| PU | 26,400 | ~64K | 2 | 6 | 多工况组合 |
| FEMTO | 17,000 | ~2.5K | 2 | 3 | 加速寿命试验 |

### 相关技术文档
- [H5DataDict文档](../docs/technical_guide.md#h5datadict) - 数据加载机制
- [BaseReader模式](../docs/technical_guide.md#basereader) - 数据读取器
- [数据工厂架构](../docs/technical_guide.md#data-factory) - 整体数据处理

---

**🎉 恭喜！您的数据已准备就绪。**

数据质量直接影响模型性能，好的开始是成功的一半！

让我们进入[实验执行阶段](../03_experiments/README.md)开始训练模型。