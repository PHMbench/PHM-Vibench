# 🧪 Flow预训练验证清单

> **版本**: v2.0
> **更新**: 2025年9月
> **用途**: 确保Flow预训练系统的正确性和可靠性

---

## ✅ 环境验证

### GPU和硬件要求
- [ ] **CUDA可用性**: `nvidia-smi` 正常运行
- [ ] **显存充足**: 至少8GB可用显存
- [ ] **PyTorch版本**: >= 2.0 且支持CUDA
- [ ] **内存充足**: 至少16GB RAM

```bash
# 验证命令
nvidia-smi
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.__version__}')"
free -h
```

### Python环境
- [ ] **Python版本**: >= 3.8
- [ ] **必要包安装**: PyTorch, Lightning, pandas, numpy
- [ ] **PHM-Vibench导入**: 核心模块可正常导入
- [ ] **Flow模型导入**: M_04_ISFM_Flow可正常导入

```bash
# 验证命令
python --version
python -c "from src.model_factory.ISFM.M_04_ISFM_Flow import Model; print('✅ Flow模型导入成功')"
```

---

## 📊 数据验证

### 数据文件完整性
- [ ] **元数据文件**: `data/metadata_6_11.xlsx` 存在
- [ ] **H5数据文件**: `data/data.h5` 存在（可选）
- [ ] **数据格式**: 元数据包含必要字段 (Id, Dataset_id, Domain_id, Label)
- [ ] **数据类型**: 信号数据为数值类型

```bash
# 验证命令
ls -la data/metadata_6_11.xlsx
python -c "import pandas as pd; df=pd.read_excel('data/metadata_6_11.xlsx'); print(f'数据条目: {len(df)}')"
```

### 数据预处理
- [ ] **窗口大小**: 1024样本窗口可正常创建
- [ ] **归一化**: 数据标准化无异常
- [ ] **批次加载**: DataLoader可正常工作
- [ ] **内存使用**: 批次加载不会耗尽内存

---

## 🔧 模型验证

### Flow模型核心功能
- [ ] **模型初始化**: 无错误创建FlowModel实例
- [ ] **前向传播**: 正确处理(B,L,C)输入格式
- [ ] **ODE求解**: 流模型计算无数值不稳定
- [ ] **条件编码**: 条件信息正确编码

```bash
# 验证命令
cd script/flow_loss_pretraining/tests/
python test_flow_model.py
```

### 训练功能
- [ ] **损失计算**: Flow损失和对比学习损失正常
- [ ] **梯度流**: 反向传播无梯度爆炸/消失
- [ ] **优化器**: Adam/AdamW优化器正常工作
- [ ] **学习率调度**: 余弦退火调度器正常

### 生成功能
- [ ] **采样生成**: 可生成合理的振动信号
- [ ] **异常检测**: 异常分数计算无异常
- [ ] **条件生成**: 不同条件生成不同信号特征
- [ ] **质量评估**: 生成信号统计特性合理

---

## ⚙️ 配置验证

### YAML配置文件
- [ ] **语法正确**: 所有YAML文件语法无误
- [ ] **参数完整**: 必要参数都已设置
- [ ] **路径正确**: 文件路径指向正确位置
- [ ] **类型匹配**: 参数类型与代码期望一致

```bash
# 验证命令
python -c "import yaml; yaml.safe_load(open('experiments/configs/quick_validation.yaml'))"
```

### 实验配置
- [ ] **快速验证**: 配置可在30分钟内完成
- [ ] **基线实验**: 配置合理，资源需求明确
- [ ] **完整研究**: 参数设置适合论文级实验
- [ ] **消融研究**: 变体配置覆盖关键组件

---

## 🧪 实验验证

### 快速验证实验
- [ ] **执行成功**: 快速验证实验无错误完成
- [ ] **损失收敛**: 训练和验证损失正常下降
- [ ] **指标合理**: 准确率等指标在合理范围
- [ ] **时间预期**: 完成时间符合预期

```bash
# 验证命令
python main.py --config script/flow_loss_pretraining/experiments/configs/quick_validation.yaml
```

### 结果保存
- [ ] **目录结构**: 结果保存在正确目录
- [ ] **检查点**: 模型检查点正常保存
- [ ] **日志文件**: 训练日志完整记录
- [ ] **配置备份**: 使用的配置文件已备份

### 性能基准
- [ ] **基线对比**: Flow方法优于简单基线
- [ ] **收敛速度**: 训练在合理轮次内收敛
- [ ] **内存使用**: 显存使用在预期范围内
- [ ] **计算效率**: 每轮训练时间合理

---

## 📈 质量验证

### 代码质量
- [ ] **单元测试**: 关键组件有测试覆盖
- [ ] **集成测试**: 端到端流程测试通过
- [ ] **错误处理**: 异常情况有适当处理
- [ ] **日志完整**: 重要操作有日志记录

### 实验可重现性
- [ ] **随机种子**: 已设置固定随机种子
- [ ] **确定性计算**: 结果可重现
- [ ] **环境记录**: 环境信息已记录
- [ ] **版本控制**: 代码版本已记录

### 文档完整性
- [ ] **README更新**: 文档反映当前状态
- [ ] **配置说明**: 参数含义清晰
- [ ] **使用示例**: 提供完整使用示例
- [ ] **故障排除**: 常见问题有解决方案

---

## 🚀 部署验证

### 脚本功能
- [ ] **执行权限**: run_experiments.sh有执行权限
- [ ] **参数解析**: 命令行参数正确解析
- [ ] **错误处理**: 脚本错误时正确退出
- [ ] **日志输出**: 脚本执行过程有清晰日志

```bash
# 验证命令
chmod +x script/flow_loss_pretraining/experiments/scripts/run_experiments.sh
bash script/flow_loss_pretraining/experiments/scripts/run_experiments.sh --quick
```

### 结果分析
- [ ] **结果收集**: collect_results.py正常运行
- [ ] **统计分析**: statistical_analysis.py正常运行
- [ ] **图表生成**: 可视化脚本正常工作
- [ ] **LaTeX输出**: 表格生成格式正确

---

## 📋 最终检查清单

在提交实验结果或发布论文前，确保以下所有项目都已完成：

### 实验完整性
- [ ] **所有基线**: 对比方法实验已完成
- [ ] **消融研究**: 关键组件贡献已分析
- [ ] **跨域评估**: 泛化性能已测试
- [ ] **统计显著性**: 结果差异已验证

### 文档和可重现性
- [ ] **完整文档**: 所有操作步骤已记录
- [ ] **代码注释**: 关键代码有充分注释
- [ ] **配置归档**: 实验配置已完整保存
- [ ] **结果备份**: 重要结果已多处备份

### 论文准备
- [ ] **图表质量**: 所有图表清晰美观
- [ ] **表格格式**: LaTeX表格格式正确
- [ ] **数据完整**: 论文中的数据有对应实验
- [ ] **代码开源**: 代码已准备公开发布

---

## 🆘 常见问题解决

### GPU内存不足
```bash
# 减少批次大小
sed -i 's/batch_size: 64/batch_size: 32/' config.yaml
# 启用梯度累积
sed -i 's/accumulate_grad_batches: 1/accumulate_grad_batches: 2/' config.yaml
```

### 模型不收敛
```yaml
# 调整学习率
learning_rate: 1e-4  # 降低学习率
# 增加预热轮次
warmup_epochs: 20
# 调整Flow参数
num_steps: 50        # 减少ODE步数
sigma: 0.01         # 增加噪声
```

### 数据加载慢
```yaml
# 增加工作线程
dataloader:
  num_workers: 8
  pin_memory: true
  persistent_workers: true
```

### 导入错误
```bash
# 检查Python路径
export PYTHONPATH="${PYTHONPATH}:/path/to/PHM-Vibench-flow"
# 或在代码中添加
import sys
sys.path.append('/path/to/PHM-Vibench-flow')
```

---

## 📞 支持和帮助

如果验证过程中遇到问题：

1. **查看错误日志**: 详细阅读错误信息
2. **检查GitHub Issues**: 搜索相似问题
3. **联系维护者**: 发送问题详情和日志
4. **社区求助**: 在PHM社区提问

**记住**: 彻底的验证是高质量研究的基础！ 🎯