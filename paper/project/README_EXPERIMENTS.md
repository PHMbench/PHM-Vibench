# 1D-2D Fusion Explainable Paper 实验补充说明

本文档说明如何执行1D-2D Fusion论文的补充实验。

## 📁 目录结构

```
Paper/1D-2D_fusion_explainable/
├── configs/                    # 实验配置文件
│   ├── config_CWRU.yaml       # CWRU数据集配置
│   ├── config_XJTU.yaml       # XJTU数据集配置
│   ├── config_THU_006.yaml     # THU_006数据集配置
│   ├── ablation/              # 消融实验配置
│   │   ├── config_1D_only.yaml
│   │   ├── config_2D_only.yaml
│   │   └── config_no_statistical.yaml
│   └── noise/                 # 噪声鲁棒性配置
│       ├── config_snr0.yaml
│       ├── config_snr5.yaml
│       ├── config_snr10.yaml
│       └── config_snr20.yaml
├── model/
│   └── Fusion1D2D_ablation.py   # 支持消融的模型版本
├── scripts/
│   ├── run_multi_dataset_experiments.sh    # 多数据集验证脚本
│   ├── run_ablation_studies.sh            # 消融研究脚本
│   ├── run_noise_robustness.sh            # 噪声鲁棒性脚本
│   └── collect_all_results.py             # 结果收集和可视化
└── results/                        # 实验结果
    ├── multi_dataset/
    ├── ablation/
    └── noise_robustness/
```

## 🚀 快速开始

### 1. 运行多数据集验证

```bash
cd Paper/1D-2D_fusion_explainable
./scripts/run_multi_dataset_experiments.sh
```

这将同时在3个数据集上运行实验：
- CWRU (PHM-Vibench)
- XJTU (PHM-Vibench)
- THU_006

### 2. 运行消融实验

```bash
./scripts/run_ablation_studies.sh
```

测试各组件贡献：
- 1D-only (仅1D分支)
- 2D-only (仅2D分支)
- no-statistical (无统计特征)

### 3. 运行噪声鲁棒性测试

```bash
./scripts/run_noise_robustness.sh
```

测试不同信噪比下的性能：
- SNR=0dB (纯噪声)
- SNR=5dB (低信噪比)
- SNR=10dB (中等信噪比)
- SNR=20dB (高信噪比)

### 4. 收集结果并生成可视化

```bash
python scripts/collect_all_results.py \
    --results_dir Paper/1D-2D_fusion_explainable/results \
    --output Paper/1D-2D_fusion_explainable/results/comprehensive
```

将生成：
- 多数据集对比图
- 消融实验雷达图
- 噪声鲁棒性曲线
- 综合实验报告

## 📊 预期结果

### 多数据集验证
- CWRU数据集：预期90%+
- XJTU数据集：预期85%+
- THU_006数据集：预期95%+

### 消融实验
- Full Fusion (完整融合)：99.57%
- 1D-only：约85-90%
- 2D-only：约80-85%
- No Statistical：约95%

### 噪声鲁棒性
- SNR=20dB：预期95%+
- SNR=10dB：预期90%+
- SNR=5dB：预期80%+
- SNR=0dB：预期60%+

## ⚠️ 注意事项

1. **数据集路径**
   - 确保PHM-Vibench数据集在正确路径：`/home/user/data/PHMbenchdata/PHM-Vibench/`
   - THU数据集路径：`/home/user/data/a_bearing/`

2. **GPU资源**
   - 多数据集实验需要3个GPU
   - 消融实验需要1个GPU
   - 噪声测试需要4个GPU

3. **模型兼容性**
   - 消融实验使用修改版的Fusion1D2D_ablation模型
   - 原始Fusion1D2D_simple模型不受影响

4. **实验时间**
   - 每个实验约需2-3小时
   - 完整实验约需1天

## 🔍 故障排除

### 配置文件问题
如果配置文件中数据集路径不正确，请检查并修改：
```yaml
data_dir: '/your/correct/data/path'
```

### 模型导入错误
确保PYTHONPATH包含项目根目录：
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### GPU内存不足
减少batch_size或使用更少GPU：
```bash
# 修改配置文件中的batch_size
batch_size: 32  # 从64改为32
```

## 📈 实验监控

所有实验都集成了Weights & Biases (WandB)跟踪，可以实时查看：
```bash
wandb status
```

项目名称：`PHM_bench/THU_018_basic`

## 📋 结果文件

运行完成后，结果保存在：
- `results/multi_dataset/` - 多数据集验证结果
- `results/ablation/` - 消融实验结果
- `results/noise_robustness/` - 噪声鲁棒性测试结果
- `results/comprehensive/` - 综合分析报告和可视化

## 🤝 贡献

欢迎提交Issue和Pull Request来改进实验脚本和可视化效果。

---

**创建时间**: 2025-12-02
**最后更新**: 2025-12-02