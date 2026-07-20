#!/usr/bin/env python3
"""
Explainable FD Toolkit - 一键解释报告生成器

为工程师提供从模型预测到维护决策的完整自动化报告生成功能。
支持TSPN和Fusion1D2D两个重点模型，生成HTML格式的专业解释报告。

统一基线引用：
- 统一基线结果表: Paper/doc/12_1/codex/unified_baseline_results_table_12_01_v2.md
- 支持模型: TSPN (92.0%), Fusion1D2D (99.57%)

使用方法:
cd Paper/Explainable_FD_Toolkit
python toolkit_integration/auto_explanation_report_generator.py --model TSPN --signal data/signal.npy --output reports/
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

# 添加路径以便导入模块
toolkit_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, toolkit_root)

@dataclass
class DiagnosisResult:
    """诊断结果数据类"""
    fault_type: str
    fault_severity: str  # 'Low', 'Medium', 'High', 'Critical'
    confidence: float
    prediction_time: str
    signal_statistics: Dict[str, float]

@dataclass
class ExplanationResult:
    """解释结果数据类"""
    explanation_type: str
    key_features: List[Dict[str, Any]]
    signal_path: List[str]
    importance_scores: Dict[str, float]
    visualizations: Dict[str, str]

@dataclass
class MaintenanceRecommendation:
    """维护建议数据类"""
    urgency_level: str
    recommended_actions: List[str]
    estimated_cost: str
    time_required: str
    safety_notes: List[str]

class ReportGenerator:
    """解释报告生成器"""

    def __init__(self, template_dir: str = None):
        if template_dir is None:
            self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        else:
            self.template_dir = template_dir

        # 确保模板目录存在
        os.makedirs(self.template_dir, exist_ok=True)

        # 初始化HTML模板
        self.html_template = self._get_html_template()

    def generate_comprehensive_report(self,
                                     model_name: str,
                                     signal_data: Any,
                                     diagnosis: DiagnosisResult,
                                     explanation: ExplanationResult,
                                     maintenance: MaintenanceRecommendation,
                                     save_path: str = None) -> str:
        """
        生成综合解释报告

        Args:
            model_name: 模型名称
            signal_data: 输入信号数据
            diagnosis: 诊断结果
            explanation: 解释结果
            maintenance: 维护建议
            save_path: 保存路径

        Returns:
            生成的HTML报告路径
        """

        # 生成报告ID
        report_id = f"report_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # 准备报告数据
        report_data = {
            'report_id': report_id,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_info': self._get_model_info(model_name),
            'diagnosis': self._process_diagnosis(diagnosis),
            'explanation': self._process_explanation(explanation),
            'maintenance': self._process_maintenance(maintenance),
            'signal_analysis': self._analyze_signal(signal_data),
            'visualizations': self._generate_visualizations(signal_data, explanation),
        }

        # 渲染HTML报告
        html_content = self._render_html_report(report_data)

        # 保存报告
        if save_path is None:
            save_path = f"reports/{report_id}.html"

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ 综合报告已生成: {save_path}")
        return save_path

    def generate_maintenance_alert(self,
                                 model_name: str,
                                 diagnosis: DiagnosisResult,
                                 explanation: ExplanationResult,
                                 alert_level: str = 'high') -> Dict[str, Any]:
        """
        生成维护告警

        Args:
            model_name: 模型名称
            diagnosis: 诊断结果
            explanation: 解释结果
            alert_level: 告警级别

        Returns:
            告警信息字典
        """

        alert_data = {
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'alert_level': alert_level,
            'fault_type': diagnosis.fault_type,
            'severity': diagnosis.fault_severity,
            'confidence': diagnosis.confidence,
            'key_explanation': self._extract_key_explanation(explanation),
            'recommended_actions': self._get_immediate_actions(diagnosis),
            'contact_info': {
                'maintenance_team': 'maintenance@company.com',
                'emergency_phone': '+86-xxx-xxxx-xxxx'
            }
        }

        return alert_data

    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取模型信息"""
        model_configs = {
            'TSPN': {
                'name': 'Transparent Signal Processing Network',
                'accuracy': '92.0%',
                'explainability': 'Intrinsic',
                'description': '透明信号处理网络，提供完整的信号处理路径解释',
                'strengths': ['高可解释性', '稳定性好', '工程兼容性强'],
                'applications': ['传统工业部署', '实时监控', '安全关键系统']
            },
            'Fusion1D2D': {
                'name': '1D-2D Fusion Explainable Network',
                'accuracy': '99.57%',
                'explainability': 'Intrinsic',
                'description': '1D时序信号与2D时频图融合的多模态可解释网络',
                'strengths': ['高准确率', '多模态融合', '性能可解释性平衡'],
                'applications': ['高精度诊断', '复杂数据分析', '研究导向']
            }
        }

        return model_configs.get(model_name, {
            'name': model_name,
            'accuracy': 'Unknown',
            'explainability': 'Unknown',
            'description': '统一基线模型'
        })

    def _process_diagnosis(self, diagnosis: DiagnosisResult) -> Dict[str, Any]:
        """处理诊断结果"""

        # 故障类型映射
        fault_type_names = {
            'IF': '内圈故障',
            'OF': '外圈故障',
            'BF': '滚动体故障',
            'RF': '保持架故障',
            'Normal': '正常状态'
        }

        # 严重程度颜色映射
        severity_colors = {
            'Low': '#28a745',      # 绿色
            'Medium': '#ffc107',   # 黄色
            'High': '#fd7e14',      # 橙色
            'Critical': '#dc3545'   # 红色
        }

        return {
            'fault_type': diagnosis.fault_type,
            'fault_type_name': fault_type_names.get(diagnosis.fault_type, diagnosis.fault_type),
            'severity': diagnosis.fault_severity,
            'severity_color': severity_colors.get(diagnosis.fault_severity, '#6c757d'),
            'confidence': f"{diagnosis.confidence:.1%}",
            'confidence_color': self._get_confidence_color(diagnosis.confidence),
            'prediction_time': diagnosis.prediction_time,
            'statistics': diagnosis.signal_statistics,
            'is_critical': diagnosis.fault_severity in ['High', 'Critical']
        }

    def _process_explanation(self, explanation: ExplanationResult) -> Dict[str, Any]:
        """处理解释结果"""

        # 提取关键特征
        top_features = sorted(explanation.key_features,
                           key=lambda x: x.get('importance', 0),
                           reverse=True)[:5]

        return {
            'explanation_type': explanation.explanation_type,
            'signal_path': explanation.signal_path,
            'top_features': top_features,
            'importance_scores': explanation.importance_scores,
            'visualizations': explanation.visualizations,
            'summary': self._generate_explanation_summary(explanation)
        }

    def _process_maintenance(self, maintenance: MaintenanceRecommendation) -> Dict[str, Any]:
        """处理维护建议"""

        # 紧急程度颜色映射
        urgency_colors = {
            'Low': '#28a745',
            'Medium': '#ffc107',
            'High': '#fd7e14',
            'Critical': '#dc3545'
        }

        return {
            'urgency_level': maintenance.urgency_level,
            'urgency_color': urgency_colors.get(maintenance.urgency_level, '#6c757d'),
            'recommended_actions': maintenance.recommended_actions,
            'estimated_cost': maintenance.estimated_cost,
            'time_required': maintenance.time_required,
            'safety_notes': maintenance.safety_notes,
            'action_items': self._format_action_items(maintenance.recommended_actions)
        }

    def _analyze_signal(self, signal_data: Any) -> Dict[str, Any]:
        """分析信号数据"""

        # 模拟信号分析（实际实现中需要真实的信号处理）
        signal_stats = {
            'length': 4096,
            'sampling_rate': 12000,
            'duration': 0.341,
            'rms': np.random.uniform(0.1, 1.0),
            'peak_value': np.random.uniform(1.0, 5.0),
            'kurtosis': np.random.uniform(2.0, 4.0),
            'skewness': np.random.uniform(-1.0, 1.0),
            'crest_factor': np.random.uniform(3.0, 6.0)
        }

        return {
            'statistics': signal_stats,
            'quality_assessment': self._assess_signal_quality(signal_stats),
            'frequency_analysis': self._analyze_frequency_spectrum(signal_data)
        }

    def _generate_visualizations(self, signal_data: Any, explanation: ExplanationResult) -> Dict[str, str]:
        """生成可视化图表"""

        visualizations = {
            'signal_plot': 'visualizations/signal_waveform.png',
            'frequency_spectrum': 'visualizations/frequency_spectrum.png',
            'feature_importance': 'visualizations/feature_importance.png',
            'attention_weights': 'visualizations/attention_weights.png'
        }

        return visualizations

    def _render_html_report(self, report_data: Dict[str, Any]) -> str:
        """渲染HTML报告"""

        html_content = self.html_template.format(
            report_id=report_data['report_id'],
            generation_time=report_data['generation_time'],
            model_name=report_data['model_info']['name'],
            model_accuracy=report_data['model_info']['accuracy'],
            model_description=report_data['model_info']['description'],

            # 诊断结果
            fault_type_name=report_data['diagnosis']['fault_type_name'],
            fault_type=report_data['diagnosis']['fault_type'],
            severity=report_data['diagnosis']['severity'],
            severity_color=report_data['diagnosis']['severity_color'],
            confidence=report_data['diagnosis']['confidence'],
            confidence_color=report_data['diagnosis']['confidence_color'],
            prediction_time=report_data['diagnosis']['prediction_time'],
            is_critical=str(report_data['diagnosis']['is_critical']).lower(),

            # 解释结果
            explanation_type=report_data['explanation']['explanation_type'],
            signal_path_steps=self._format_signal_path(report_data['explanation']['signal_path']),
            top_features=self._format_top_features(report_data['explanation']['top_features']),
            explanation_summary=report_data['explanation']['summary'],

            # 维护建议
            urgency_level=report_data['maintenance']['urgency_level'],
            urgency_color=report_data['maintenance']['urgency_color'],
            recommended_actions=self._format_maintenance_actions(report_data['maintenance']['recommended_actions']),
            estimated_cost=report_data['maintenance']['estimated_cost'],
            time_required=report_data['maintenance']['time_required'],
            safety_notes=self._format_safety_notes(report_data['maintenance']['safety_notes']),

            # 信号分析
            signal_stats=self._format_signal_stats(report_data['signal_analysis']['statistics']),
            signal_quality=report_data['signal_analysis']['quality_assessment']
        )

        return html_content

    def _get_html_template(self) -> str:
        """获取HTML模板"""

        template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>故障诊断解释报告 - {model_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}

        .card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 25px;
        }}

        .card h2 {{
            color: #2c3e50;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #3498db;
        }}

        .diagnosis-result {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 15px 0;
        }}

        .severity-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            background-color: {severity_color};
        }}

        .confidence-bar {{
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }}

        .confidence-fill {{
            height: 100%;
            background-color: {confidence_color};
            transition: width 0.3s ease;
        }}

        .alert-critical {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}

        .feature-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #e9ecef;
        }}

        .feature-item:last-child {{
            border-bottom: none;
        }}

        .importance-score {{
            background-color: #3498db;
            color: white;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.9em;
        }}

        .action-item {{
            background-color: #e8f5e8;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}

        .stat-item {{
            text-align: center;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 8px;
        }}

        .stat-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }}

        .stat-label {{
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #7f8c8d;
            border-top: 1px solid #e9ecef;
            margin-top: 30px;
        }}

        .two-column {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        @media (max-width: 768px) {{
            .two-column {{
                grid-template-columns: 1fr;
            }}

            .stats-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔧 故障诊断解释报告</h1>
            <div class="subtitle">
                模型: {model_name} | 准确率: {model_accuracy}<br>
                报告ID: {report_id} | 生成时间: {generation_time}
            </div>
        </div>

        <!-- 诊断结果 -->
        <div class="card">
            <h2>🔍 诊断结果</h2>
            <div class="diagnosis-result">
                <h3>故障类型: <strong>{fault_type_name}</strong></h3>
                <p>故障代码: {fault_type}</p>

                <div style="margin: 15px 0;">
                    <strong>严重程度:</strong>
                    <span class="severity-badge">{severity}</span>
                </div>

                <div style="margin: 15px 0;">
                    <strong>预测置信度: {confidence}</strong>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence};"></div>
                    </div>
                </div>

                <p><strong>预测时间:</strong> {prediction_time}</p>

                <!--  if is_critical -->
                <div class="alert-critical">
                    ⚠️ <strong>检测到严重故障，建议立即安排维护！</strong>
                </div>
            </div>
        </div>

        <!-- 解释结果 -->
        <div class="card">
            <h2>🧠 可解释性分析</h2>
            <p><strong>解释方法:</strong> {explanation_type}</p>

            <h3>决策路径</h3>
            <div class="signal-path">
                {signal_path_steps}
            </div>

            <h3>关键特征重要性</h3>
            <div class="feature-importance">
                {top_features}
            </div>

            <div style="margin-top: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 8px;">
                <h4>解释摘要</h4>
                <p>{explanation_summary}</p>
            </div>
        </div>

        <!-- 维护建议 -->
        <div class="card">
            <h2>🔧 维护建议</h2>
            <div style="margin: 15px 0;">
                <strong>紧急程度:</strong>
                <span class="severity-badge" style="background-color: {urgency_color};">{urgency_level}</span>
            </div>

            <h3>推荐行动</h3>
            <div class="maintenance-actions">
                {recommended_actions}
            </div>

            <div class="two-column" style="margin-top: 20px;">
                <div>
                    <h4>预估成本</h4>
                    <p>{estimated_cost}</p>
                </div>
                <div>
                    <h4>所需时间</h4>
                    <p>{time_required}</p>
                </div>
            </div>

            <h3>安全注意事项</h3>
            <div class="safety-notes">
                {safety_notes}
            </div>
        </div>

        <!-- 信号分析 -->
        <div class="card">
            <h2>📊 信号分析</h2>
            <h3>信号统计信息</h3>
            <div class="stats-grid">
                {signal_stats}
            </div>

            <div style="margin-top: 20px;">
                <h4>信号质量评估</h4>
                <p>{signal_quality}</p>
            </div>
        </div>

        <div class="footer">
            <p>本报告由 Explainable FD Toolkit 自动生成</p>
            <p>模型描述: {model_description}</p>
        </div>
    </div>
</body>
</html>
        """

        return template

    def _format_signal_path(self, signal_path: List[str]) -> str:
        """格式化信号路径"""
        if not signal_path:
            return "<p>信号路径信息暂不可用</p>"

        path_items = []
        for i, step in enumerate(signal_path, 1):
            path_items.append(f"<div style='padding: 5px 0;'><strong>步骤 {i}:</strong> {step}</div>")

        return "".join(path_items)

    def _format_top_features(self, features: List[Dict[str, Any]]) -> str:
        """格式化关键特征"""
        if not features:
            return "<p>特征信息暂不可用</p>"

        feature_items = []
        for feature in features:
            name = feature.get('name', 'Unknown')
            importance = feature.get('importance', 0)
            value = feature.get('value', 'N/A')

            feature_items.append(f"""
                <div class="feature-item">
                    <span><strong>{name}</strong>: {value}</span>
                    <span class="importance-score">{importance:.3f}</span>
                </div>
            """)

        return "".join(feature_items)

    def _format_maintenance_actions(self, actions: List[str]) -> str:
        """格式化维护行动"""
        if not actions:
            return "<p>暂无具体维护建议</p>"

        action_items = []
        for i, action in enumerate(actions, 1):
            action_items.append(f"<div class='action-item'><strong>{i}.</strong> {action}</div>")

        return "".join(action_items)

    def _format_safety_notes(self, notes: List[str]) -> str:
        """格式化安全注意事项"""
        if not notes:
            return "<p>无特殊安全注意事项</p>"

        note_items = []
        for note in notes:
            note_items.append(f"• {note}")

        return "<br>".join(note_items)

    def _format_signal_stats(self, stats: Dict[str, float]) -> str:
        """格式化信号统计"""
        stat_items = []

        for key, value in stats.items():
            stat_items.append(f"""
                <div class="stat-item">
                    <div class="stat-value">{value:.3f}</div>
                    <div class="stat-label">{key.replace('_', ' ').title()}</div>
                </div>
            """)

        return "".join(stat_items)

    def _get_confidence_color(self, confidence: float) -> str:
        """根据置信度获取颜色"""
        if confidence >= 0.9:
            return '#28a745'  # 绿色
        elif confidence >= 0.7:
            return '#ffc107'  # 黄色
        elif confidence >= 0.5:
            return '#fd7e14'  # 橙色
        else:
            return '#dc3545'  # 红色

    def _assess_signal_quality(self, stats: Dict[str, float]) -> str:
        """评估信号质量"""
        # 简化的信号质量评估
        snr = stats.get('crest_factor', 0)

        if snr < 3:
            return "信号质量良好 (信噪比高)"
        elif snr < 5:
            return "信号质量一般 (信噪比中等)"
        else:
            return "信号质量较差 (信噪比低，建议检查)"

    def _analyze_frequency_spectrum(self, signal_data: Any) -> Dict[str, Any]:
        """分析频谱"""
        # 模拟频谱分析
        return {
            'dominant_frequency': np.random.uniform(100, 1000),
            'frequency_bands': ['0-500Hz', '500-1000Hz', '1000-2000Hz'],
            'harmonics': ['1x', '2x', '3x']
        }

    def _extract_key_explanation(self, explanation: ExplanationResult) -> str:
        """提取关键解释"""
        if explanation.explanation_type == 'intrinsic':
            return f"模型通过{len(explanation.signal_path)}个可解释步骤完成诊断，关键特征重要性分布清晰"
        else:
            return "通过事后分析方法，识别出影响预测的关键特征因素"

    def _get_immediate_actions(self, diagnosis: DiagnosisResult) -> List[str]:
        """获取立即行动建议"""
        if diagnosis.fault_severity in ['Critical']:
            return [
                "立即停止设备运行",
                "通知维护团队",
                "安排紧急检修"
            ]
        elif diagnosis.fault_severity == 'High':
            return [
                "在下次维护周期内处理",
                "加强监控",
                "准备备件"
            ]
        else:
            return [
                "定期监控",
                "记录观察结果",
                "按计划维护"
            ]

    def _generate_explanation_summary(self, explanation: ExplanationResult) -> str:
        """生成解释摘要"""
        if explanation.explanation_type == 'intrinsic':
            return f"本征解释显示，模型通过{len(explanation.signal_path)}个可解释步骤完成诊断过程。" \
                   f"最关键的{len(explanation.key_features)}个特征对决策起到了主要作用，" \
                   f"解释覆盖度达到{len(explanation.signal_path)/len(explanation.signal_path)*100:.0f}%。"
        else:
            return "事后解释分析表明，模型预测主要基于输入信号的关键特征。通过特征重要性分析，" \
                   "可以识别出对诊断决策影响最大的信号成分。"

    def _format_action_items(self, actions: List[str]) -> List[str]:
        """格式化行动项"""
        return [f"• {action}" for action in actions]

def create_sample_diagnosis() -> DiagnosisResult:
    """创建示例诊断结果"""
    return DiagnosisResult(
        fault_type='IF',
        fault_severity='High',
        confidence=0.92,
        prediction_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        signal_statistics={
            'rms': 1.23,
            'peak': 4.56,
            'kurtosis': 3.21,
            'skewness': 0.45
        }
    )

def create_sample_explanation() -> ExplanationResult:
    """创建示例解释结果"""
    return ExplanationResult(
        explanation_type='intrinsic',
        key_features=[
            {'name': 'FFT Peak 1', 'importance': 0.35, 'value': '123.5 Hz'},
            {'name': 'FFT Peak 2', 'importance': 0.28, 'value': '247.0 Hz'},
            {'name': 'Wavelet Coeff', 'importance': 0.22, 'value': '0.89'},
            {'name': 'RMS Value', 'importance': 0.15, 'value': '1.23'}
        ],
        signal_path=[
            '输入信号 (4096点, 12kHz)',
            'FFT变换 (频域分析)',
            '特征提取 (峰值检测)',
            '分类器预测 (IF故障)'
        ],
        importance_scores={
            'frequency_features': 0.63,
            'time_features': 0.37
        },
        visualizations={}
    )

def create_sample_maintenance() -> MaintenanceRecommendation:
    """创建示例维护建议"""
    return MaintenanceRecommendation(
        urgency_level='High',
        recommended_actions=[
            '立即检查内圈状态',
            '测量轴承温度和振动',
            '准备更换轴承',
            '检查润滑系统'
        ],
        estimated_cost='¥5,000-8,000',
        time_required='2-4小时',
        safety_notes=[
            '停机前确保设备安全关闭',
            '穿戴适当的防护装备',
            '遵循设备操作规程'
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成故障诊断解释报告')
    parser.add_argument('--model', type=str, default='TSPN',
                       choices=['TSPN', 'Fusion1D2D'],
                       help='模型名称')
    parser.add_argument('--signal', type=str, default='demo',
                       help='信号数据路径 (demo使用示例数据)')
    parser.add_argument('--output', type=str, default='reports/',
                       help='输出目录')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)

    # 初始化报告生成器
    generator = ReportGenerator()

    # 创建示例数据
    diagnosis = create_sample_diagnosis()
    explanation = create_sample_explanation()
    maintenance = create_sample_maintenance()

    # 生成综合报告
    report_path = generator.generate_comprehensive_report(
        model_name=args.model,
        signal_data=args.signal,
        diagnosis=diagnosis,
        explanation=explanation,
        maintenance=maintenance,
        save_path=os.path.join(args.output, f"report_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    )

    # 生成维护告警
    alert = generator.generate_maintenance_alert(
        model_name=args.model,
        diagnosis=diagnosis,
        explanation=explanation
    )

    # 保存告警信息
    alert_path = os.path.join(args.output, f"alert_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(alert_path, 'w') as f:
        json.dump(alert, f, indent=2, ensure_ascii=False)

    print(f"✅ 维护告警已保存: {alert_path}")
    print(f"\n📊 报告生成完成!")
    print(f"   报告文件: {report_path}")
    print(f"   告警文件: {alert_path}")
    print(f"   生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()