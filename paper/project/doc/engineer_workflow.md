# Explainable FD Toolkit - 工程使用流程指南

**文档版本**: v1.0
**目标读者**: 故障诊断工程师、设备维护专家、工业AI部署工程师
**更新时间**: 2025年12月2日

---

## 🎯 概述

Explainable FD Toolkit为故障诊断工程师提供了从信号采集到解释生成的完整工作流程。本指南将帮助您：

1. **快速上手**: 5分钟内完成第一次可解释性诊断
2. **标准流程**: 掌握工业级故障诊断的标准操作流程
3. **结果解读**: 理解解释结果的含义和工程价值
4. **故障排查**: 解决常见问题和配置错误

---

## 🚀 快速开始 (5分钟体验)

### 第1步: 环境准备
```bash
# 激活环境
conda activate UXFD

# 进入工具目录
cd Paper/Explainable_FD_Toolkit

# 检查依赖
python -c "import torch, numpy, matplotlib; print('环境就绪')"
```

### 第2步: 加载信号数据
```python
from toolkit_integration.fd_explainability_toolkit import load_signal_data

# 加载振动信号 (可以是.npy, .csv, .mat格式)
signal_data = load_signal_data('data/vibration_signal.npy')

print(f"信号长度: {len(signal_data)}")
print(f"信号范围: [{signal_data.min():.3f}, {signal_data.max():.3f}]")
```

### 第3步: 创建解释器
```python
# 创建TSPN模型解释器 (推荐用于工业应用)
explainer = create_model_explainer('TSPN')

print("✅ TSPN解释器创建成功")
```

### 第4步: 生成解释
```python
# 生成诊断解释
result = explainer.explain(signal_data, model_name='TSPN')

# 显示诊断结果
print(f"故障类型: {result['fault_type']}")
print(f"置信度: {result['confidence']:.2%}")
print(f"处理时间: {result['computation_time']:.4f}秒")
```

### 第5步: 生成报告
```python
# 生成专业维护报告
report = explainer.generate_report(result, output_format='html')

print(f"📄 报告已生成: {report['file_path']}")
```

**恭喜！您已在5分钟内完成了第一次可解释性故障诊断。**

---

## 📊 完整工作流程

### 阶段1: 数据准备与预处理

#### 1.1 信号采集
```python
# 支持的信号格式
signal_formats = {
    '.npy': 'NumPy数组格式',
    '.csv': 'CSV表格格式',
    '.mat': 'MATLAB格式',
    '.txt': '文本格式'
}

# 采集参数
sampling_params = {
    'sampling_rate': '1000 Hz  # 推荐采样率',
    'duration': '4s  # 信号时长',
    'channels': '1-8  # 传感器通道数'
}
```

#### 1.2 信号预处理
```python
from toolkit_integration.signal_processor import SignalProcessor

processor = SignalProcessor()

# 标准预处理流程
processed_signal = processor.preprocess(
    signal_data,
    steps=[
        'denoising',      # 降噪处理
        'normalization',  # 归一化
        'segmentation',   # 信号分段
        'filtering'       # 滤波处理
    ]
)
```

### 阶段2: 模型选择与配置

#### 2.1 模型选择指南
```python
model_selection_guide = {
    'TSPN': {
        '适用场景': '工业设备常规诊断',
        '优势': '解释清晰、实时性好、工业验证',
        '准确率': '99%+',
        '资源需求': '中等'
    },
    'FuzzyLogic': {
        '适用场景': '安全关键系统、边缘设备',
        '优势': '规则透明、轻量级、安全兜底',
        '准确率': '70%+',
        '资源需求': '极低'
    },
    'Fusion1D2D': {
        '适用场景': '高精度需求场景',
        '优势': '多模态融合、性能最佳',
        '准确率': '99.5%+',
        '资源需求': '较高'
    }
}
```

#### 2.2 模型配置
```python
# TSPN配置示例
tspn_config = {
    'signal_layers': ['FFT', 'HT'],  # 信号处理层
    'feature_dim': 13,                # 特征维度
    'classification_head': 'softmax',  # 分类头
    'explanation_method': 'intrinsic'  # 解释方法
}

explainer = create_model_explainer('TSPN', config=tspn_config)
```

### 阶段3: 可解释性诊断

#### 3.1 实时诊断模式
```python
# 实时诊断 (适合在线监测)
real_time_results = explainer.explain_realtime(
    signal_stream,
    window_size=4096,
    overlap=0.5
)

# 实时告警
if real_time_results['confidence'] > 0.9:
    alert_level = 'HIGH'
    maintenance_action = '立即检查设备'
elif real_time_results['confidence'] > 0.7:
    alert_level = 'MEDIUM'
    maintenance_action = '加强监控'
```

#### 3.2 批量诊断模式
```python
# 批量分析 (适合历史数据分析)
batch_results = explainer.explain_batch(
    signal_batch,
    parallel=True,
    save_intermediate=True
)

# 批量报告生成
batch_report = explainer.generate_batch_report(batch_results)
```

### 阶段4: 结果解读与分析

#### 4.1 解释结果结构
```python
# 典型的解释结果
explanation_result = {
    # 基础诊断信息
    'fault_type': 'IF',              # 故障类型 (IF/OF/BF/N)
    'confidence': 0.92,              # 置信度
    'processing_time': 0.005,       # 处理时间

    # 详细解释信息
    'signal_path': [                 # 信号处理路径
        'FFT变换: 将时域信号转换为频域',
        '统计特征提取: 计算13维特征向量',
        '分类决策: 基于特征模式匹配'
    ],

    'key_features': {                # 关键特征
        'rms': 0.156,               # 均方根值
        'peak_freq': 157.3,         # 峰值频率
        'kurtosis': 2.34            # 峰度
    },

    'attention_weights': [           # 注意力权重
        0.45, 0.32, 0.23           # 各处理步骤的重要性
    ]
}
```

#### 4.2 结果解读指南
```python
# 置信度解读
confidence_interpretation = {
    (0.9, 1.0): '高度确信 - 立即采取维护行动',
    (0.7, 0.9): '中度确信 - 安排计划性维护',
    (0.5, 0.7): '低度确信 - 加强监控，收集更多数据',
    (0.0, 0.5): '不确定 - 检查传感器和数据质量'
}

# 故障类型说明
fault_descriptions = {
    'IF': '内圈故障 - 通常由轴承内圈磨损或损伤引起',
    'OF': '外圈故障 - 通常由轴承外圈疲劳引起',
    'BF': '滚动体故障 - 通常由滚子损伤引起',
    'N': '正常状态 - 设备运行正常'
}
```

### 阶段5: 报告生成与输出

#### 5.1 HTML报告生成
```python
# 生成专业的HTML维护报告
html_report = explainer.generate_html_report(
    result=explanation_result,
    template='maintenance',          # 使用维护模板
    include_charts=True,            # 包含图表
    language='zh'                   # 中文输出
)

print(f"📄 HTML报告: {html_report['file_path']}")
```

#### 5.2 JSON结构化数据
```python
# 生成JSON格式数据 (适合系统集成)
json_data = explainer.generate_json_report(
    result=explanation_result,
    schema='industry_standard'       # 行业标准格式
)

# 集成到维护系统
maintenance_system.import_diagnosis(json_data)
```

#### 5.3 实时告警生成
```python
# 生成维护告警
if explanation_result['confidence'] > 0.9:
    alert = explainer.generate_alert(
        result=explanation_result,
        alert_level='HIGH',
        actions=['立即停机检查', '联系维修团队', '准备备件']
    )

    # 发送告警
    send_maintenance_alert(alert)
```

---

## 🔧 高级配置与优化

### 模型参数调优
```python
# 性能优化配置
optimization_config = {
    'batch_size': 64,              # 批处理大小
    'compute_precision': 'float32', # 计算精度
    'gpu_acceleration': True,       # GPU加速
    'memory_limit': '2GB'          # 内存限制
}

explainer.set_optimization(optimization_config)
```

### 自定义解释方法
```python
# 自定义解释器
class CustomExplainer(BaseExplainer):
    def explain(self, signal_data, **kwargs):
        # 实现自定义解释逻辑
        pass

explainer.register_custom_method('custom', CustomExplainer())
```

### 多模型集成
```python
# 多模型集成诊断
ensemble_results = explainer.ensemble_explain(
    signal_data,
    models=['TSPN', 'FuzzyLogic', 'Fusion1D2D'],
    voting_method='weighted_average'
)
```

---

## 📈 性能指标与监控

### 实时性能指标
```python
# 获取系统性能
performance_metrics = explainer.get_performance_stats()

print(f"诊断延迟: {performance_metrics['latency_ms']}ms")
print(f"内存占用: {performance_metrics['memory_mb']}MB")
print(f"GPU利用率: {performance_metrics['gpu_util']}%")
print(f"诊断准确率: {performance_metrics['accuracy']:.2%}")
```

### 质量监控
```python
# 诊断质量监控
quality_monitor = explainer.get_quality_monitor()

if quality_monitor['confidence_drift'] > 0.1:
    print("⚠️ 检测到置信度漂移，建议重新校准模型")

if quality_monitor['data_quality_score'] < 0.8:
    print("⚠️ 数据质量下降，检查传感器状态")
```

---

## 🚨 故障排查指南

### 常见问题与解决方案

#### 1. 环境问题
```python
# 检查环境
try:
    import torch
    print("✅ PyTorch环境正常")
except ImportError:
    print("❌ 请安装PyTorch: pip install torch")

# 检查CUDA
if torch.cuda.is_available():
    print(f"✅ CUDA可用，GPU数量: {torch.cuda.device_count()}")
else:
    print("⚠️ CUDA不可用，将使用CPU模式")
```

#### 2. 数据问题
```python
# 数据质量检查
def validate_signal_data(signal):
    issues = []

    if len(signal) == 0:
        issues.append("信号为空")
    if np.isnan(signal).any():
        issues.append("信号包含NaN值")
    if np.isinf(signal).any():
        issues.append("信号包含无穷值")
    if np.std(signal) < 1e-6:
        issues.append("信号方差过小")

    return issues

# 检查数据
issues = validate_signal_data(signal_data)
if issues:
    print("❌ 数据问题:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✅ 数据质量正常")
```

#### 3. 模型加载问题
```python
# 模型检查
def validate_model(model_path):
    if not os.path.exists(model_path):
        return False, "模型文件不存在"

    try:
        # 尝试加载模型
        model = torch.load(model_path)
        return True, "模型加载成功"
    except Exception as e:
        return False, f"模型加载失败: {str(e)}"
```

#### 4. 性能问题
```python
# 性能优化建议
performance_tips = {
    '诊断速度慢': '启用GPU加速或减小批处理大小',
    '内存占用高': '使用float32精度或减小batch_size',
    '准确率下降': '检查信号质量，重新校准模型',
    '置信度过低': '增加信号长度或改善数据预处理'
}
```

---

## 📚 最佳实践

### 1. 工业部署建议
- **数据质量**: 确保传感器定期校准和数据清洗
- **模型更新**: 定期使用新数据重新训练模型
- **监控机制**: 建立实时监控和异常检测机制
- **备份方案**: 准备备用诊断系统和应急处理流程

### 2. 解释结果应用
- **预防性维护**: 基于置信度趋势制定维护计划
- **故障根因**: 利用解释特征定位故障根本原因
- **知识积累**: 建立故障案例库和专家知识库
- **持续改进**: 根据现场反馈优化解释方法

### 3. 团队协作
- **工程师培训**: 定期培训操作工程师使用工具
- **跨部门协作**: 建立维护、生产、技术部门协作机制
- **文档管理**: 维护完整的操作文档和故障记录
- **技术支持**: 建立技术支持团队解决复杂问题

---

## 📞 技术支持

### 联系方式
- **技术文档**: [在线文档链接]
- **问题反馈**: [GitHub Issues]
- **技术交流**: [微信群/QQ群]
- **商务合作**: [邮箱地址]

### 版本更新
- **当前版本**: v1.0
- **更新频率**: 每月发布新版本
- **获取更新**: `pip install --upgrade explainable-fd-toolkit`

---

**文档维护**: 本文档会随着产品更新持续完善
**适用版本**: v1.0及以上版本

*最后更新: 2025年12月2日*