# HSE Industrial Contrastive Learning 用户执行计划

## 🎯 执行概览

本文档为用户提供了HSE Industrial Contrastive Learning系统的完整执行计划，从环境设置到生产部署的详细步骤指南。

## 📋 执行前检查清单

### 环境要求检查
- [ ] Python 3.8+
- [ ] PyTorch 2.6.0+
- [ ] CUDA 11.8+ (可选，用于GPU加速)
- [ ] 16GB+ RAM
- [ ] 10GB+ 可用磁盘空间

### 依赖验证
```bash
# 检查关键依赖
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import wandb, swanlab; print('✅ 实验跟踪工具已安装')"
```

### 数据准备检查
- [ ] PHM数据目录配置正确
- [ ] metadata文件存在
- [ ] H5缓存文件可访问
- [ ] 读写权限正确

## 🚀 快速开始 (15分钟)

### 步骤1: 环境激活
```bash
# 激活conda环境
conda activate P

# 验证安装
python scripts/hse_synthetic_demo.py --quick-test
```

### 步骤2: 基础验证
```bash
# 运行合成数据演示 (推荐首次使用)
python scripts/hse_synthetic_demo.py

# 预期输出:
# ✅ 系统提示编码: 成功
# ✅ 样本提示编码: 成功
# ✅ 提示融合: 成功
# ✅ 对比学习: 成功 (准确度提升: 14.3%)
# ✅ 验证测试: 成功
```

### 步骤3: Pipeline测试
```bash
# 运行Pipeline_03集成测试
python scripts/test_pipeline03_integration.py

# 预期结果: 5/9测试通过 (55.6%成功率)
```

## 📊 分阶段执行计划

### 阶段1: 系统验证 (30分钟)

#### 1.1 核心组件验证
```bash
# OneEpochValidator验证
python -c "
from src.utils.validation.OneEpochValidator import OneEpochValidator
from src.configs import load_config
config = load_config('quickstart')
validator = OneEpochValidator(config)
result = validator.run_validation()
print(f'验证结果: {result}')
"
```

#### 1.2 数据加载验证
```bash
# 验证UnifiedDataLoader
python -c "
from src.data_factory.UnifiedDataLoader import UnifiedDataLoader
from src.configs import load_config
config = load_config('quickstart')
loader = UnifiedDataLoader(config)
print('✅ 数据加载器创建成功')
"
```

#### 1.3 模型组件验证
```bash
# 验证HSE核心组件
python -c "
from src.model_factory.ISFM_Prompt.components.SystemPromptEncoder import SystemPromptEncoder
from src.model_factory.ISFM_Prompt.components.PromptFusion import PromptFusion
print('✅ HSE提示系统组件可用')
"
```

### 阶段2: 基线实验 (1小时)

#### 2.1 CWRU基线实验
```bash
# 运行CWRU基线实验
python main.py --config configs/baseline/cwru_baseline.yaml

# 预期结果:
# - 训练完成无错误
# - 准确度 > 70%
# - 内存使用 < 1GB
```

#### 2.2 HSE演示实验
```bash
# 运行HSE对比学习演示
python main.py --config configs/demo/HSE_Contrastive/hse_cddg.yaml

# 预期结果:
# - HSE准确度 > 基线准确度
# - 提示系统正常工作
# - 对比学习收敛
```

### 阶段3: 完整Pipeline (2-3小时)

#### 3.1 两阶段训练
```bash
# 阶段1: 预训练
python scripts/run_hse_prompt_pipeline03.py \
  --stage pretrain \
  --config configs/demo/HSE_Contrastive/hse_prompt_pretrain.yaml

# 阶段2: 微调
python scripts/run_hse_prompt_pipeline03.py \
  --stage finetune \
  --config configs/demo/HSE_Contrastive/hse_prompt_finetune.yaml
```

#### 3.2 多任务实验
```bash
# 运行多任务预训练-微调pipeline
python main.py --pipeline Pipeline_03 \
  --config configs/pipeline_03/hse_prompt_multitask_config.yaml
```

### 阶段4: 生产验证 (4-6小时)

#### 4.1 多数据集验证
```bash
# 依次在5个数据集上验证
datasets=("CWRU" "XJTU" "THU" "Ottawa" "JNU")
for dataset in "${datasets[@]}"; do
    echo "验证数据集: $dataset"
    python main.py --config configs/demo/Single_DG/${dataset}.yaml
done
```

#### 4.2 消融研究
```bash
# 运行消融研究实验
ablation_configs=(
    "configs/pipeline_03/ablation/hse_no_prompt_baseline.yaml"
    "configs/pipeline_03/ablation/hse_system_prompt_only.yaml"
    "configs/pipeline_03/ablation/hse_sample_prompt_only.yaml"
)

for config in "${ablation_configs[@]}"; do
    echo "运行消融实验: $config"
    python main.py --config "$config"
done
```

## 🔧 故障排除指南

### 常见问题解决方案

#### 问题1: ConfigWrapper兼容性错误
```bash
# 症状: TypeError: 'ConfigWrapper' object is not iterable
# 解决方案:
git checkout main src/configs/config_utils.py
# 或者
pip install --upgrade pyyaml
```

#### 问题2: H5数据加载失败
```bash
# 症状: FileNotFoundError: *.h5
# 解决方案:
export DATA_DIR=/path/to/your/data
python scripts/check_data_paths.py
# 确保data/目录包含正确的metadata文件
```

#### 问题3: CUDA内存不足
```bash
# 症状: RuntimeError: CUDA out of memory
# 解决方案:
# 1. 减小batch_size
sed -i 's/batch_size: 32/batch_size: 16/g' your_config.yaml
# 2. 启用gradient checkpointing
echo "gradient_checkpointing: true" >> your_config.yaml
```

#### 问题4: seaborn导入错误
```bash
# 症状: ModuleNotFoundError: No module named 'seaborn'
# 解决方案:
pip install seaborn
# 或者忽略(系统已做容错处理)
```

#### 问题5: reformer_pytorch缺失
```bash
# 症状: ModuleNotFoundError: No module named 'reformer_pytorch'
# 解决方案:
pip install reformer-pytorch
```

### 性能优化指南

#### GPU优化
```bash
# 启用混合精度训练
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
python main.py --config your_config.yaml --precision 16
```

#### CPU优化
```bash
# 优化CPU工作线程
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
python main.py --config your_config.yaml
```

#### 内存优化
```bash
# 减少数据预取
export PYTHONHASHSEED=0
python main.py --config your_config.yaml --data.num_workers 4
```

## 📊 执行监控和验证

### 实时监控指标

#### 必须监控的指标
1. **训练损失**: 应单调递减
2. **验证准确度**: 应逐步提升
3. **内存使用**: 应 < 8GB
4. **GPU利用率**: 应 > 80%
5. **数据加载速度**: 应 > 1000 samples/sec

#### 监控命令
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 监控内存使用
htop

# 监控实验进度
tail -f logs/training.log
```

### 验证检查点

#### 阶段1完成检查
- [ ] 所有核心组件无错误加载
- [ ] 合成数据演示成功
- [ ] OneEpochValidator通过
- [ ] 内存使用 < 1GB

#### 阶段2完成检查
- [ ] CWRU基线实验完成
- [ ] 准确度达到预期 (> 70%)
- [ ] HSE对比实验显示提升
- [ ] 无内存泄漏

#### 阶段3完成检查
- [ ] 两阶段训练成功
- [ ] 检查点保存/加载正常
- [ ] 多任务学习收敛
- [ ] Pipeline_03集成无错误

#### 阶段4完成检查
- [ ] 至少3个数据集验证成功
- [ ] 跨域泛化效果显著
- [ ] 消融研究结果合理
- [ ] 系统稳定运行

## 📈 结果收集和分析

### 实验结果收集
```bash
# 收集所有实验结果
python script/unified_metric/collect_results.py \
  --experiment_dir save/ \
  --output_format markdown \
  --include_plots

# 生成综合报告
python script/unified_metric/generate_summary_report.py \
  --results_dir reports/ \
  --output comprehensive_results.md
```

### 关键指标提取
```bash
# 提取关键性能指标
python -c "
import json
import glob

results = []
for file in glob.glob('save/*/metrics.json'):
    with open(file) as f:
        data = json.load(f)
        results.append({
            'experiment': file.split('/')[1],
            'accuracy': data.get('test_accuracy', 0),
            'training_time': data.get('training_time', 0),
            'memory_peak': data.get('memory_peak_gb', 0)
        })

# 排序并输出最佳结果
results.sort(key=lambda x: x['accuracy'], reverse=True)
print('🏆 最佳实验结果:')
for i, r in enumerate(results[:3]):
    print(f'{i+1}. {r[\"experiment\"]}: {r[\"accuracy\"]:.1%} 准确度')
"
```

### 可视化结果
```bash
# 生成性能对比图表
python script/unified_metric/paper_visualization.py \
  --results_dir save/ \
  --output_dir plots/ \
  --comparison_baseline

# 生成论文级别图表
python script/unified_metric/sota_comparison.py \
  --output plots/sota_comparison.png
```

## 🎯 成功标准和验收

### 最低验收标准
- [ ] **基础功能**: 合成数据演示成功
- [ ] **性能提升**: 相比基线准确度提升 > 5%
- [ ] **计算效率**: 内存使用 < 2GB
- [ ] **稳定性**: 连续3次实验成功

### 推荐验收标准
- [ ] **显著提升**: 相比基线准确度提升 > 10%
- [ ] **高效计算**: 内存使用 < 1GB, 速度 > 1000 s/s
- [ ] **跨域泛化**: 至少2个数据集上验证成功
- [ ] **生产就绪**: Pipeline_03集成测试通过

### 优秀验收标准
- [ ] **卓越性能**: 相比基线准确度提升 > 15%
- [ ] **极致效率**: 内存使用 < 0.5GB, 速度 > 1500 s/s
- [ ] **全面验证**: 所有5个数据集验证成功
- [ ] **完整功能**: 所有消融研究完成

## 📝 执行报告模板

### 日执行报告
```markdown
# HSE执行日报 - YYYY-MM-DD

## 执行进度
- 计划阶段: [ ] 阶段1 [ ] 阶段2 [ ] 阶段3 [ ] 阶段4
- 完成进度: X/Y 项目完成
- 主要成果: [简述当天主要成果]

## 关键指标
- 最佳准确度: XX.X%
- 平均训练时间: XX分钟
- 峰值内存使用: X.XGB
- 成功实验数: X/Y

## 问题和解决
- 遇到问题: [描述]
- 解决方案: [描述]
- 状态: [已解决/进行中/待解决]

## 明日计划
- [具体计划项目1]
- [具体计划项目2]
```

### 周执行总结
```markdown
# HSE执行周报 - 第X周

## 总体进展
- 阶段完成情况: [详述]
- 里程碑达成: [列举]
- 技术突破: [说明]

## 性能总结
- 最佳实验结果: [详述]
- 性能趋势分析: [说明]
- 技术问题解决: [统计]

## 下周重点
- 主要目标: [列举]
- 关键任务: [详述]
- 风险评估: [说明]
```

## 🚀 部署和交付

### 部署清单
- [ ] 代码库整理和文档完善
- [ ] 依赖项清单和安装脚本
- [ ] 配置文件模板和示例
- [ ] 用户指南和FAQ
- [ ] 性能基准和对比报告

### 交付物清单
1. **核心代码库**: 完整的HSE实现代码
2. **配置文件**: 各种实验配置模板
3. **文档体系**: 完整的技术文档
4. **验证报告**: 全面的实验验证结果
5. **部署指南**: 生产环境部署说明

### 维护计划
- **Bug修复**: 响应用户反馈，及时修复问题
- **性能优化**: 持续优化算法和系统性能
- **功能扩展**: 根据需求增加新功能
- **文档更新**: 保持文档和代码同步

---

**执行计划版本**: v1.0
**最后更新**: 2025年9月15日
**适用范围**: HSE Industrial Contrastive Learning v2.0

**重要提醒**: 执行过程中如遇到问题，请优先参考故障排除指南，或联系开发团队获取支持。