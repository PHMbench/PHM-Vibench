# LLM增强可解释性系统

## 概述

本系统在原有故障诊断可解释性框架的基础上，集成了大语言模型(LLM)能力，提供了自然语言解释、交互式对话和知识增强诊断功能。

## 核心特性

### 🔍 技术解释增强
- **自然语言解释**: 将技术分析结果转换为易懂的自然语言描述
- **对话式交互**: 支持多轮对话，实时回答工程师的技术问题
- **上下文感知**: 结合设备信息和历史数据提供个性化解释

### 📚 领域知识集成
- **故障知识图谱**: 包含机械故障的特征、原因和维护策略
- **术语标准化**: 统一技术术语，确保沟通的准确性
- **上下文处理**: 结合运行环境、设备状态等上下文信息

### 🤖 智能对话系统
- **多轮对话管理**: 维护对话状态和上下文信息
- **查询意图识别**: 理解用户查询的具体意图和需求
- **个性化响应**: 根据用户背景和问题类型定制回答

### 📊 反馈与分析
- **用户体验收集**: 系统性收集用户反馈用于持续改进
- **对话质量分析**: 分析对话效果和用户满意度
- **性能指标监控**: 跟踪系统性能和诊断准确性

## 架构设计

```
LLM Enhanced Explainability System
├── Core Components (核心组件)
│   ├── LLMExplainer: LLM解释生成器
│   ├── SignalEncoder: 信号编码器
│   ├── PromptManager: 提示管理器
│   ├── LLMInterface: LLM统一接口
│   └── ResponseParser: 响应解析器
├── Knowledge Enhancement (知识增强)
│   ├── FaultKnowledgeGraph: 故障知识图谱
│   ├── TerminologyMapper: 术语映射器
│   └── ContextProcessor: 上下文处理器
├── Conversation System (对话系统)
│   ├── ConversationEngine: 对话引擎
│   ├── QueryProcessor: 查询处理器
│   └── FeedbackCollector: 反馈收集器
└── Model Integration (模型集成)
    └── TSPN_LLM_Enhanced: 增强版TSPN模型
```

## 快速开始

### 1. 基本使用

```python
from model.TSPN_LLM_Enhanced import TSPN_LLM_Enhanced
import torch

# 配置模型
config = {
    'layer1': 'FFT', 'layer2': 'WF', 'layer3': 'HT', 'layer4': 'I',
    'in_channels': 1, 'out_channels': 1, 'in_dim': 4096, 'out_dim': 10,
    'scale': 4, 'skip_connection': True, 'num_classes': 10
}

# 配置LLM (需要API密钥)
llm_config = {
    'providers': {
        'openai': {
            'type': 'openai',
            'api_key': 'your-api-key',
            'model': 'gpt-3.5-turbo'
        }
    }
}

# 创建模型
model = TSPN_LLM_Enhanced(config, llm_config)

# 加载振动数据
input_data = torch.randn(1, 4096, 1)

# 获取带解释的预测结果
result = model.predict_with_explanation(
    input_data,
    user_query="请详细解释这个故障的机理",
    include_llm=True
)

print(f"故障类型: {result['prediction']['class_name']}")
print(f"置信度: {result['prediction']['confidence']}")
print(f"LLM解释: {result['llm_explanation']}")
```

### 2. 交互式对话

```python
# 开始诊断对话
greeting = model.start_diagnostic_conversation(
    input_data,
    device_info={
        "device_type": "电机",
        "operating_speed": 1800,
        "criticality_level": "high"
    }
)
print(greeting)

# 继续对话
response1 = model.continue_conversation(session_id, "这个故障的主要原因是什么？")
print(response1)

response2 = model.continue_conversation(session_id, "应该如何维修？")
print(response2)
```

### 3. 知识增强查询

```python
from explainability.knowledge.fault_knowledge_graph import FaultKnowledgeGraph, FaultType

# 创建知识图谱
knowledge_graph = FaultKnowledgeGraph()

# 获取故障模式
fault_pattern = knowledge_graph.get_fault_pattern(FaultType.INNER_RACE_FAULT)
print(f"症状: {fault_pattern.symptoms}")
print(f"常见原因: {fault_pattern.common_causes}")

# 获取特征频率
char_freqs = knowledge_graph.get_characteristic_frequencies(
    FaultType.INNER_RACE_FAULT, shaft_speed=1800
)
print(f"特征频率: {char_freqs}")
```

## 组件详解

### LLMExplainer
- **功能**: 核心LLM解释生成器
- **输入**: 技术分析结果、用户查询、上下文信息
- **输出**: 自然语言解释、建议和回答
- **特性**: 支持多种LLM提供商、上下文增强、结构化输出

### PromptManager
- **功能**: 管理和生成LLM提示
- **模板**: 预定义领域特定提示模板
- **特性**: 多语言支持、动态提示构建、领域知识注入

### LLMInterface
- **功能**: 统一的LLM访问接口
- **支持**: OpenAI GPT、Anthropic Claude、本地模型
- **特性**: 自动故障转移、负载均衡、错误处理

### FaultKnowledgeGraph
- **功能**: 机械故障领域知识图谱
- **内容**: 故障模式、特征频率、维护策略
- **特性**: 故障关系推理、严重程度评估、维护规划

### ConversationEngine
- **功能**: 管理交互式对话
- **特性**: 多轮对话、状态跟踪、上下文管理
- **支持**: 对话历史、会话恢复、反馈收集

## 配置说明

### LLM配置

```python
llm_config = {
    'providers': {
        'openai': {
            'type': 'openai',
            'api_key': 'sk-...',
            'model': 'gpt-4',
            'base_url': 'https://api.openai.com/v1'
        },
        'anthropic': {
            'type': 'anthropic',
            'api_key': 'sk-ant-...',
            'model': 'claude-3-sonnet'
        },
        'local': {
            'type': 'local',
            'model_path': '/path/to/model',
            'api_url': 'http://localhost:8000/v1'
        }
    }
}
```

### 模型配置

```python
model_config = {
    # 信号处理层配置
    'layer1': 'FFT',  # FFT, WF, HT, I, LNO
    'layer2': 'WF',
    'layer3': 'HT',
    'layer4': 'I',

    # 网络参数
    'in_channels': 1,
    'out_channels': 1,
    'in_dim': 4096,
    'out_dim': 10,
    'scale': 4,
    'skip_connection': True,
    'num_classes': 10
}
```

## 使用案例

### 案例1: 故障诊断与解释
```python
# 工程师检测到异常振动，需要详细解释
result = model.predict_with_explanation(
    vibration_data,
    user_query="这个内圈故障的频谱特征是什么意思？如何确认故障？"
)

# 系统提供：
# 1. 技术诊断结果
# 2. 自然语言解释
# 3. 确认方法建议
# 4. 相关风险分析
```

### 案例2: 维修方案制定
```python
# 通过对话制定维修方案
session_id = model.start_diagnostic_conversation(vibration_data)
response = model.continue_conversation(
    session_id,
    "针对这个不对中故障，请提供详细的维修步骤和注意事项"
)
```

### 案例3: 知识查询
```python
# 查询特定故障的知识
knowledge_graph = FaultKnowledgeGraph()
maintenance_actions = knowledge_graph.get_maintenance_actions(
    FaultType.BEARING_WEAR, SeverityLevel.HIGH
)
```

## 性能优化

### 1. LLM调用优化
- **缓存机制**: 缓存常见问题的LLM响应
- **批量处理**: 批量处理多个查询
- **本地模型**: 支持本地部署降低延迟

### 2. 对话管理优化
- **上下文压缩**: 智能压缩长对话历史
- **会话池**: 复用对话连接
- **状态持久化**: 支持会话恢复

### 3. 知识检索优化
- **索引优化**: 快速故障模式检索
- **缓存策略**: 热点知识缓存
- **增量更新**: 支持知识库动态更新

## 扩展开发

### 添加新的LLM提供商
```python
from explainability.llm.llm_interface import LLMProvider

class CustomLLMProvider(LLMProvider):
    def generate_response(self, prompt, **kwargs):
        # 实现自定义LLM调用
        pass

    def is_available(self):
        return True

    def get_model_info(self):
        return {"provider": "custom", "model": "custom-model"}
```

### 扩展故障知识库
```python
# 添加新的故障模式
new_fault_pattern = FaultPattern(
    fault_type=FaultType.NEW_FAULT,
    characteristic_frequencies={"new_freq": 2.5},
    # ... 其他参数
)
knowledge_graph.fault_patterns[FaultType.NEW_FAULT] = new_fault_pattern
```

### 自定义对话流程
```python
from explainability.conversation.conversation_engine import DialogueState

# 添加新的对话状态
class CustomState(DialogueState):
    CUSTOM_DIAGNOSIS = "custom_diagnosis"

# 扩展状态转换逻辑
conversation_engine.state_transitions[CustomState.CUSTOM_DIAGNOSIS] = {
    "next_state": DialogueState.MAINTENANCE_PLANNING,
    "triggers": ["维修", "处理"]
}
```

## 注意事项

### 1. API密钥安全
- 不要在代码中硬编码API密钥
- 使用环境变量或安全配置文件
- 定期轮换API密钥

### 2. 成本控制
- 设置LLM调用频率限制
- 监控API使用量和成本
- 优先使用缓存减少重复调用

### 3. 数据隐私
- 不向LLM发送敏感数据
- 使用匿名化处理
- 遵守数据保护法规

### 4. 错误处理
- 实现优雅降级机制
- 提供备用的规则基础响应
- 记录和分析错误模式

## 故障排除

### 常见问题

1. **LLM API调用失败**
   - 检查API密钥和网络连接
   - 确认API配额和限制
   - 使用备用提供商或本地模型

2. **对话会话丢失**
   - 检查会话ID有效性
   - 确认会话超时设置
   - 实现会话持久化

3. **知识库查询慢**
   - 优化索引结构
   - 启用查询缓存
   - 考虑数据分片

4. **内存使用过高**
   - 限制对话历史长度
   - 定期清理过期会话
   - 优化数据结构

## 贡献指南

欢迎为项目贡献代码和改进建议：

1. Fork项目仓库
2. 创建功能分支
3. 提交代码变更
4. 编写测试用例
5. 提交Pull Request

## 许可证

本项目遵循MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@example.com

---

**注意**: 本系统仍在持续开发中，部分功能可能需要完整的环境配置才能正常运行。建议在生产环境使用前进行充分测试。