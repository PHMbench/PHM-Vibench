#!/usr/bin/env python3
"""
LLM增强解释示例

本示例展示如何使用LLM增强的解释系统，
为故障诊断生成自然语言解释和对话式交互。

主要功能：
1. 技术解释转换为自然语言
2. 对话式解释系统
3. 个性化解释定制
4. 多层级解释生成
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from toolkit_integration.explainability.core.explanation import Explanation


class DemoLLMExplainer:
    """演示用的LLM增强解释器"""

    def __init__(self, llm_config=None):
        self.config = llm_config or {
            "model": "demo-llm",
            "language": "zh",
            "max_tokens": 500,
            "temperature": 0.7
        }
        self.fault_knowledge = self._initialize_fault_knowledge()

    def _initialize_fault_knowledge(self):
        """初始化故障知识库"""
        return {
            "inner_race": {
                "description": "内圈故障是轴承内圈表面出现的疲劳剥落或裂纹",
                "symptoms": ["高频振动", "旋转频率的谐波", "包络谱特征"],
                "causes": ["过载", "润滑不足", "安装不当"],
                "severity": "中等到严重",
                "recommendation": "建议尽快停机检查，更换轴承"
            },
            "outer_race": {
                "description": "外圈故障是轴承外圈表面出现的疲劳损伤",
                "symptoms": ["明显的高频成分", "特定的故障频率", "温度升高"],
                "causes": ["载荷分布不均", "材料疲劳", "腐蚀"],
                "severity": "中等",
                "recommendation": "安排计划性维护，监控运行状态"
            },
            "ball": {
                "description": "滚动体故障是轴承滚珠或滚子的表面损伤",
                "symptoms": ["随机性冲击", "宽带高频", "不规则振动"],
                "causes": ["材料缺陷", "过载", "污染"],
                "severity": "严重",
                "recommendation": "立即停机检查，防止进一步损坏"
            }
        }

    def generate_natural_explanation(self, explanation, target_audience='engineer'):
        """生成自然语言解释"""

        fault_type = explanation.get_meta('fault_type', 'unknown')
        method = explanation.get_method_name()
        metrics = explanation.get_metrics()

        # 获取故障知识
        fault_info = self.fault_knowledge.get(fault_type, {})

        # 根据受众调整解释内容
        if target_audience == 'engineer':
            return self._generate_engineer_explanation(explanation, fault_info, method, metrics)
        elif target_audience == 'manager':
            return self._generate_manager_explanation(explanation, fault_info, method, metrics)
        elif target_audience == 'researcher':
            return self._generate_researcher_explanation(explanation, fault_info, method, metrics)
        else:
            return self._generate_general_explanation(explanation, fault_info, method, metrics)

    def _generate_engineer_explanation(self, explanation, fault_info, method, metrics):
        """生成面向工程师的解释"""

        explanation_text = f"""
## 故障诊断报告 (工程师版本)

### 📊 诊断结果
- **故障类型**: {explanation.get_meta('fault_type', '未知')}
- **置信度**: {metrics.get('attribution_max', 0):.2%}
- **解释方法**: {method}

### 🔍 技术分析
使用{method}方法的分析结果如下:

1. **信号特征识别**:
   - 模型识别出{metrics.get('attribution_mean', 0):.3f}的平均归因强度
   - 关键特征集中在信号的特定频段
   - 归因稀疏度为{metrics.get('attribution_sparsity', 0):.2%}

2. **故障机制分析**:
   {fault_info.get('description', '未找到故障描述')}

3. **关键症状**:
   {', '.join(fault_info.get('symptoms', ['无明显症状']))}

### ⚙️ 技术建议
{fault_info.get('recommendation', '建议进一步检查')}

### 📈 解释质量指标
- **完整性**: 85%
- **可信度**: 90%
- **技术可靠性**: 高

---
*此报告基于AI分析生成，请结合工程经验进行最终判断*
        """

        return explanation_text.strip()

    def _generate_manager_explanation(self, explanation, fault_info, method, metrics):
        """生成面向管理者的解释"""

        severity = fault_info.get('severity', '未知')
        confidence = metrics.get('attribution_max', 0)

        # 根据置信度和严重程度确定行动级别
        if confidence > 0.8 and '严重' in severity:
            action_level = "🔴 立即行动"
        elif confidence > 0.6:
            action_level = "🟡 计划维护"
        else:
            action_level = "🟢 持续监控"

        explanation_text = f"""
## 设备状态报告 (管理层版本)

### 📋 执行摘要
- **设备状态**: {explanation.get_meta('fault_type', '未知故障')}
- **严重程度**: {severity}
- **AI置信度**: {confidence:.1%}
- **建议行动**: {action_level}

### 🎯 影响评估
{fault_info.get('description', '需要进一步分析')}

### 💡 风险评估
- **技术风险**: {severity}
- **置信度**: {confidence:.1%}
- **预测准确性**: 高

### 📊 关键指标
- 诊断方法: {method}
- 分析可靠性: {metrics.get('attribution_std', 0):.3f}
- 特征一致性: 优秀

### 🚀 建议措施
{fault_info.get('recommendation', '建议咨询技术专家')}

---
*基于AI故障诊断系统分析 - 生成时间: {self._get_current_time()}*
        """

        return explanation_text.strip()

    def _generate_researcher_explanation(self, explanation, fault_info, method, metrics):
        """生成面向研究者的解释"""

        explanation_text = f"""
## 技术研究报告 (研究者版本)

### 🧪 方法论分析
- **解释方法**: {method}
- **模型架构**: {explanation.get_meta('model_name', '未知模型')}
- **输入特征维度**: {explanation.get_meta('input_shape', '未知')}

### 📊 定量分析结果
```
归因统计:
- 均值: {metrics.get('attribution_mean', 0):.6f}
- 标准差: {metrics.get('attribution_std', 0):.6f}
- 最大值: {metrics.get('attribution_max', 0):.6f}
- 稀疏度: {metrics.get('attribution_sparsity', 0):.6f}
```

### 🔬 技术细节
1. **算法性能**:
   - 解释方法: {method}
   - 计算复杂度: O(n)
   - 内存效率: 优秀

2. **诊断质量**:
   - 特征识别准确性: 高
   - 多方法一致性: {metrics.get('attribution_std', 0) < 0.1}
   - 解释稳定性: 良好

3. **故障机理**:
   - 类型: {explanation.get_meta('fault_type', '未知')}
   - 物理机制: {fault_info.get('description', '未详细分析')}
   - 相关症状: {len(fault_info.get('symptoms', []))}个

### 📈 研究价值
- 方法创新性: 中等
- 实用性: 高
- 可扩展性: 优秀

---
*研究数据 - 可用于进一步方法改进和验证*
        """

        return explanation_text.strip()

    def _generate_general_explanation(self, explanation, fault_info, method, metrics):
        """生成通用解释"""

        explanation_text = f"""
## 故障诊断解释

### 检测结果
系统检测到{explanation.get_meta('fault_type', '故障')}，使用{method}方法进行分析。

### 主要发现
{fault_info.get('description', '检测到异常信号模式')}

### 建议措施
{fault_info.get('recommendation', '建议进行专业检查')}

### 诊断置信度
{metrics.get('attribution_max', 0):.1%}
        """

        return explanation_text.strip()

    def conversational_explain(self, query, context=None):
        """对话式解释功能"""

        query_lower = query.lower()

        if '为什么' in query and '故障' in query:
            return self._explain_fault_cause(context)
        elif '如何' in query and '修复' in query:
            return self._explain_repair_method(context)
        elif '严重' in query or '风险' in query:
            return self._assess_severity(context)
        elif '预防' in query:
            return self._provide_prevention_tips(context)
        elif '置信度' in query or '准确' in query:
            return self._explain_confidence(context)
        else:
            return self._general_response(query, context)

    def _explain_fault_cause(self, context):
        """解释故障原因"""
        return """
故障原因分析:

根据诊断结果，故障主要由以下因素造成:

1. **机械因素**: 长期运行导致的材料疲劳
2. **环境因素**: 工作环境中的振动和冲击
3. **维护因素**: 定期维护不够充分

建议进行详细的设备检查，特别关注易损部件的磨损情况。
        """.strip()

    def _explain_repair_method(self, context):
        """解释修复方法"""
        return """
修复建议:

1. **立即措施**:
   - 停机检查，防止故障扩大
   - 记录故障现象和运行参数

2. **维修步骤**:
   - 更换损坏部件
   - 清洁和润滑相关部件
   - 校准设备参数

3. **预防措施**:
   - 建立定期检查制度
   - 优化运行参数
   - 改善工作环境

需要专业技术人员进行操作。
        """.strip()

    def _assess_severity(self, context):
        """评估严重程度"""
        return """
严重程度评估:

基于AI分析结果:

🔴 **风险评估**: 中等到严重
- 短期影响: 可能导致生产中断
- 长期影响: 设备寿命缩短

⏰ **时间敏感性**: 建议在下次维护窗口期内处理

💰 **成本影响**:
- 及时处理: 低成本
- 延迟处理: 可能增加50%以上成本

建议制定详细的维修计划。
        """.strip()

    def _provide_prevention_tips(self, context):
        """提供预防建议"""
        return """
预防性维护建议:

1. **日常监控**:
   - 定期检查振动信号
   - 监控温度变化
   - 记录异常声音

2. **定期维护**:
   - 按照制造商建议更换润滑剂
   - 定期校准设备
   - 检查连接部件的紧固情况

3. **运行优化**:
   - 避免超负荷运行
   - 保持良好的工作环境
   - 培训操作人员

4. **数据分析**:
   - 建立历史数据档案
   - 分析故障模式趋势
   - 优化维护策略

通过这些措施，可以显著降低故障发生概率。
        """.strip()

    def _explain_confidence(self, context):
        """解释置信度"""
        return """
诊断置信度说明:

🎯 **技术指标**:
- 信号特征匹配度: 92%
- 多方法一致性: 88%
- 历史验证准确率: 85%

🔍 **不确定性来源**:
- 环境噪声影响: 轻微
- 设备状态变化: 已考虑
- 传感器精度: 良好

📊 **置信度评估**: 高

这个置信度意味着模型有85%以上的把握认为诊断结果是正确的。
建议结合人工检查进行最终确认。
        """.strip()

    def _general_response(self, query, context):
        """通用回答"""
        return """
我可以帮助您理解故障诊断的结果，包括:

✅ **故障原因分析**
✅ **修复方法建议**
✅ **严重程度评估**
✅ **预防措施建议**
✅ **技术细节解释**

请告诉我您想了解哪个方面的信息，我会提供详细的解释。
        """.strip()

    def _get_current_time(self):
        """获取当前时间"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def demo_llm_enhanced_explanation():
    """演示LLM增强解释功能"""
    print("=" * 60)
    print("1. LLM增强解释系统演示")
    print("=" * 60)

    # 创建演示解释数据
    demo_explanation_data = {
        'attributions': [0.1, 0.8, 0.3, 0.9, 0.2],
        'signal_path': [
            {'layer': 'Layer 1', 'output': '频域特征'},
            {'layer': 'Layer 2', 'output': '包络特征'}
        ]
    }

    demo_meta_data = {
        'method': 'integrated_gradients',
        'model_name': 'NNSPN',
        'fault_type': 'inner_race',
        'input_shape': [1, 1000, 2]
    }

    explanation = Explanation(demo_explanation_data, demo_meta_data)

    # 创建LLM解释器
    llm_explainer = DemoLLMExplainer()

    print("✓ LLM增强解释器初始化完成")
    print(f"  配置: {llm_explainer.config}")
    print(f"  故障知识库条目: {len(llm_explainer.fault_knowledge)}")

    return explanation, llm_explainer


def demo_audience_specific_explanations(explanation, llm_explainer):
    """演示面向不同受众的解释"""
    print("\n" + "=" * 60)
    print("2. 面向不同受众的解释生成")
    print("=" * 60)

    audiences = ['engineer', 'manager', 'researcher']
    audience_names = ['工程师', '管理者', '研究者']

    for audience, name in zip(audiences, audience_names):
        print(f"\n🎯 {name}版本解释:")
        print("-" * 40)

        natural_explanation = llm_explainer.generate_natural_explanation(
            explanation, target_audience=audience
        )

        print(natural_explanation[:200] + "..." if len(natural_explanation) > 200 else natural_explanation)
        print()


def demo_conversational_interface(llm_explainer):
    """演示对话式解释接口"""
    print("\n" + "=" * 60)
    print("3. 对话式解释接口演示")
    print("=" * 60)

    # 模拟对话场景
    demo_queries = [
        "为什么会发生这个故障？",
        "如何修复这个问题？",
        "故障严重吗？有什么风险？",
        "如何预防类似问题？",
        "诊断的置信度如何？"
    ]

    context = {
        'fault_type': 'inner_race',
        'confidence': 0.85,
        'method': 'integrated_gradients'
    }

    for i, query in enumerate(demo_queries, 1):
        print(f"\n💬 用户问题 {i}: {query}")
        print("-" * 30)

        response = llm_explainer.conversational_explain(query, context)
        print(response)
        print()


def demo_multi_level_explanations(explanation, llm_explainer):
    """演示多层级解释"""
    print("\n" + "=" * 60)
    print("4. 多层级解释生成")
    print("=" * 60)

    # 定义不同详细程度的解释级别
    explanation_levels = {
        'brief': '简要概述',
        'detailed': '详细分析',
        'comprehensive': '全面报告'
    }

    for level, description in explanation_levels.items():
        print(f"\n📊 {description}级别:")
        print("-" * 30)

        # 模拟不同级别的解释长度
        if level == 'brief':
            brief_text = f"检测到{explanation.get_meta('fault_type')}故障，置信度85%，建议尽快检查。"
            print(brief_text)
        elif level == 'detailed':
            detailed_text = f"""通过{explanation.get_meta('method')}方法分析，识别出{explanation.get_meta('fault_type')}故障。
主要原因包括材料疲劳和运行载荷过大。关键特征包括高频振动和特定频率成分。
建议安排维护并考虑更换相关部件。置信度: 85%。"""
            print(detailed_text)
        else:  # comprehensive
            comprehensive_text = llm_explainer.generate_natural_explanation(
                explanation, target_audience='engineer'
            )
            print(comprehensive_text[:300] + "..." if len(comprehensive_text) > 300 else comprehensive_text)

        print()


def save_llm_explanations(explanation, llm_explainer):
    """保存LLM生成的解释"""
    print("\n" + "=" * 60)
    print("5. 保存LLM解释结果")
    print("=" * 60)

    output_dir = Path('output/llm_explanations')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为不同受众生成并保存解释
    audiences = ['engineer', 'manager', 'researcher']

    for audience in audiences:
        natural_explanation = llm_explainer.generate_natural_explanation(
            explanation, target_audience=audience
        )

        save_path = output_dir / f'explanation_{audience}.md'
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(natural_explanation)

        print(f"✓ {audience}版本解释已保存到: {save_path}")

    # 保存对话记录示例
    conversation_log = []
    demo_queries = [
        "为什么会发生这个故障？",
        "如何修复这个问题？",
        "故障严重吗？"
    ]

    context = {'fault_type': 'inner_race'}

    for query in demo_queries:
        response = llm_explainer.conversational_explain(query, context)
        conversation_log.append({
            'timestamp': llm_explainer._get_current_time(),
            'query': query,
            'response': response
        })

    # 保存对话记录
    import json
    conversation_path = output_dir / 'conversation_log.json'
    with open(conversation_path, 'w', encoding='utf-8') as f:
        json.dump(conversation_log, f, ensure_ascii=False, indent=2)

    print(f"✓ 对话记录已保存到: {conversation_path}")


def main():
    """主函数：运行完整的LLM增强解释演示"""
    print("LLM增强解释系统演示")
    print("=" * 80)

    # 1. 初始化系统
    explanation, llm_explainer = demo_llm_enhanced_explanation()

    # 2. 面向不同受众的解释
    demo_audience_specific_explanations(explanation, llm_explainer)

    # 3. 对话式解释
    demo_conversational_interface(llm_explainer)

    # 4. 多层级解释
    demo_multi_level_explanations(explanation, llm_explainer)

    # 5. 保存解释结果
    save_llm_explanations(explanation, llm_explainer)

    print("\n" + "=" * 80)
    print("演示完成！")
    print("\n关键功能:")
    print("1. 🎯 面向不同受众的个性化解释生成")
    print("2. 💬 智能对话式解释系统")
    print("3. 📊 多层级详细程度控制")
    print("4. 💾 结构化解释结果保存")
    print("5. 🔗 技术分析与知识库融合")
    print("\n输出文件保存在 'output/llm_explanations/' 目录中")


if __name__ == "__main__":
    main()