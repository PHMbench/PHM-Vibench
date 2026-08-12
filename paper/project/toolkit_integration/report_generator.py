"""
可解释性报告自动生成器

提供完整的可解释性报告生成功能，包括技术分析、可视化、
评估指标、建议措施等多个维度的报告内容。

作者: Explainable_FD_Toolkit开发团队
版本: 1.0.0
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from jinja2 import Template, Environment, FileSystemLoader
import base64
from io import BytesIO

# 导入项目相关模块
from toolkit_integration.explainability.core.explanation import Explanation
from toolkit_integration.llm_interface import LLMInterfaceManager, AudienceType, ExplanationLevel


class ReportFormat(Enum):
    """报告格式类型"""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    WORD = "word"


class ReportType(Enum):
    """报告类型"""
    SINGLE_EXPLANATION = "single_explanation"    # 单个解释报告
    BATCH_COMPARISON = "batch_comparison"        # 批量比较报告
    MODEL_EVALUATION = "model_evaluation"        # 模型评估报告
    TECHNICAL_ANALYSIS = "technical_analysis"    # 技术分析报告
    EXECUTIVE_SUMMARY = "executive_summary"      # 执行摘要
    USER_FRIENDLY = "user_friendly"              # 用户友好报告


@dataclass
class ReportConfig:
    """报告配置"""
    report_type: ReportType = ReportType.SINGLE_EXPLANATION
    format: ReportFormat = ReportFormat.HTML
    output_dir: str = "reports"
    include_visualizations: bool = True
    include_raw_data: bool = False
    include_metrics: bool = True
    include_recommendations: bool = True
    audience: AudienceType = AudienceType.ENGINEER
    language: str = "zh"
    template_dir: Optional[str] = None
    custom_css: Optional[str] = None
    logo_path: Optional[str] = None
    company_info: Optional[Dict[str, str]] = None
    auto_timestamp: bool = True


@dataclass
class ReportSection:
    """报告章节"""
    title: str
    content: str
    order: int
    visualizations: List[str] = field(default_factory=list)
    subsections: List['ReportSection'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportData:
    """报告数据"""
    explanations: List[Explanation]
    model_info: Dict[str, Any]
    evaluation_metrics: Optional[Dict[str, Any]] = None
    comparison_results: Optional[Dict[str, Any]] = None
    user_context: Optional[Dict[str, Any]] = None
    generation_time: Optional[datetime] = None
    custom_data: Dict[str, Any] = field(default_factory=dict)


class VisualizationGenerator:
    """可视化生成器"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_attribution_plot(self, explanation: Explanation, save_path: Optional[str] = None) -> str:
        """生成归因图"""
        attribution = explanation.get_attribution()
        if attribution is None:
            return ""

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 原始信号
        if 'original_signal' in explanation.data:
            signal = explanation.data['original_signal']
            if hasattr(signal, 'detach'):
                signal = signal.detach().cpu().numpy()
            if signal.ndim > 1:
                signal = signal.flatten()

            ax1.plot(signal, alpha=0.8, color='blue', linewidth=1.5)
            ax1.set_title('原始信号', fontsize=12, fontweight='bold')
            ax1.set_xlabel('时间点')
            ax1.set_ylabel('幅值')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, '原始信号数据不可用', ha='center', va='center', transform=ax1.transAxes)

        # 归因值
        ax2.plot(attribution.flatten(), color='red', linewidth=2, alpha=0.8)
        ax2.set_title(f'归因值 ({explanation.get_method_name()})', fontsize=12, fontweight='bold')
        ax2.set_xlabel('时间点')
        ax2.set_ylabel('归因强度')
        ax2.grid(True, alpha=0.3)

        # 标记高归因值区域
        threshold = np.percentile(np.abs(attribution), 90)
        high_attr_indices = np.where(np.abs(attribution.flatten()) > threshold)[0]
        if len(high_attr_indices) > 0:
            ax2.scatter(high_attr_indices, attribution.flatten()[high_attr_indices],
                       color='red', s=30, alpha=0.7, zorder=5, label='高归因区域')
            ax2.legend()

        plt.tight_layout()

        if save_path is None:
            save_path = f"attribution_{explanation.get_method_name()}_{int(time.time())}.png"
            save_path = self.output_dir / save_path

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def generate_metrics_comparison(self, explanations: List[Explanation], save_path: Optional[str] = None) -> str:
        """生成指标比较图"""
        if len(explanations) < 2:
            return ""

        # 提取指标数据
        methods = []
        metrics_data = {
            'attribution_mean': [],
            'attribution_max': [],
            'attribution_sparsity': []
        }

        for exp in explanations:
            methods.append(exp.get_method_name())
            metrics = exp.get_metrics()
            for key in metrics_data.keys():
                metrics_data[key].append(metrics.get(key, 0))

        # 创建雷达图
        fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

        # 准备数据
        categories = list(metrics_data.keys())
        N = len(categories)

        # 角度
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # 为每个方法绘制雷达图
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        for i, method in enumerate(methods):
            values = [metrics_data[cat][i] for cat in categories]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.25, color=colors[i % len(colors)])

        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['归因均值', '最大归因值', '归因稀疏度'])
        ax.set_ylim(0, 1)
        ax.set_title('解释方法指标比较', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

        plt.tight_layout()

        if save_path is None:
            save_path = f"metrics_comparison_{int(time.time())}.png"
            save_path = self.output_dir / save_path

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def generate_method_comparison_chart(self, comparison_results: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """生成方法比较图"""
        if not comparison_results:
            return ""

        # 提取性能数据
        methods = list(comparison_results.keys())
        times = []
        qualities = []

        for method, result in comparison_results.items():
            if result:
                times.append(result.get('single_time', 0))
                qualities.append(result.get('metrics', {}).get('attribution_mean', 0))
            else:
                times.append(float('inf'))
                qualities.append(0)

        # 创建散点图
        fig, ax = plt.subplots(figsize=(10, 6))

        # 过滤有效数据
        valid_indices = [i for i, t in enumerate(times) if t != float('inf')]
        valid_methods = [methods[i] for i in valid_indices]
        valid_times = [times[i] for i in valid_indices]
        valid_qualities = [qualities[i] for i in valid_indices]

        scatter = ax.scatter(valid_times, valid_qualities, s=100, alpha=0.7, c=range(len(valid_methods)), cmap='viridis')

        # 标注点
        for i, method in enumerate(valid_methods):
            ax.annotate(method, (valid_times[i], valid_qualities[i]),
                       xytext=(5, 5), textcoords='offset points')

        ax.set_xlabel('计算时间 (秒)')
        ax.set_ylabel('解释质量')
        ax.set_title('解释方法性能比较', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path is None:
            save_path = f"method_comparison_{int(time.time())}.png"
            save_path = self.output_dir / save_path

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        return str(save_path)

    def embed_image_base64(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        if not os.path.exists(image_path):
            return ""

        with open(image_path, 'rb') as img_file:
            img_data = img_file.read()
            base64_str = base64.b64encode(img_data).decode('utf-8')

        return f"data:image/png;base64,{base64_str}"


class ReportTemplateManager:
    """报告模板管理器"""

    def __init__(self, template_dir: Optional[str] = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"

        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)

        # 初始化Jinja2环境
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=True
        )

        # 创建默认模板
        self._create_default_templates()

    def get_template(self, template_name: str) -> Template:
        """获取模板"""
        try:
            return self.jinja_env.get_template(template_name)
        except Exception:
            # 如果模板不存在，使用默认模板
            return self.jinja_env.from_string(self._get_fallback_template())

    def _create_default_templates(self):
        """创建默认模板"""
        # HTML报告模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #2E86AB;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .section {
            margin-bottom: 30px;
            padding: 20px;
            border-left: 4px solid #2E86AB;
            background-color: #f8f9fa;
        }
        .section h2 {
            color: #2E86AB;
            margin-top: 0;
        }
        .visualization {
            text-align: center;
            margin: 20px 0;
        }
        .visualization img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background: white;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2E86AB;
        }
        .metric-label {
            color: #666;
            font-size: 14px;
        }
        .footer {
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #666;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        th {
            background-color: #2E86AB;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>生成时间: {{ generation_time }}</p>
            {% if company_info %}
            <p>{{ company_info.name }} - {{ company_info.department }}</p>
            {% endif %}
        </div>

        {% for section in sections %}
        <div class="section">
            <h2>{{ section.title }}</h2>
            {{ section.content | safe }}

            {% for viz in section.visualizations %}
            {% if viz %}
            <div class="visualization">
                <img src="{{ viz }}" alt="{{ section.title }} 可视化">
            </div>
            {% endif %}
            {% endfor %}
        </div>
        {% endfor %}

        {% if metrics %}
        <div class="section">
            <h2>评估指标</h2>
            <div class="metrics-grid">
                {% for metric_name, metric_value in metrics.items() %}
                <div class="metric-card">
                    <div class="metric-value">{{ "%.3f"|format(metric_value) }}</div>
                    <div class="metric-label">{{ metric_name }}</div>
                </div>
                {% endfor %}
            </div>
        </div>
        {% endif %}

        <div class="footer">
            <p>本报告由 Explainable_FD_Toolkit 自动生成</p>
            <p>版本: {{ version }} | 联系: {{ contact }}</p>
        </div>
    </div>
</body>
</html>
        """

        template_path = self.template_dir / "report.html"
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(html_template)

    def _get_fallback_template(self) -> str:
        """获取备用模板"""
        return """
        <html>
        <head><title>{{ title }}</title></head>
        <body>
            <h1>{{ title }}</h1>
            {% for section in sections %}
            <h2>{{ section.title }}</h2>
            <div>{{ section.content }}</div>
            {% endfor %}
        </body>
        </html>
        """


class ReportGenerator:
    """报告生成器"""

    def __init__(self, config: ReportConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.viz_generator = VisualizationGenerator(str(self.output_dir / "visualizations"))
        self.template_manager = ReportTemplateManager(config.template_dir)

    def generate_report(self, report_data: ReportData) -> str:
        """生成报告"""
        # 设置生成时间
        if report_data.generation_time is None:
            report_data.generation_time = datetime.now()

        # 生成报告章节
        sections = self._generate_sections(report_data)

        # 生成可视化
        visualizations = self._generate_visualizations(report_data)

        # 选择模板并渲染
        template = self.template_manager.get_template("report.html")

        report_content = template.render(
            title=self._get_report_title(),
            generation_time=report_data.generation_time.strftime("%Y-%m-%d %H:%M:%S"),
            sections=sections,
            visualizations=visualizations,
            metrics=report_data.evaluation_metrics,
            model_info=report_data.model_info,
            company_info=self.config.company_info,
            version="1.0.0",
            contact="explainable-toolkit@example.com"
        )

        # 保存报告
        report_path = self._save_report(report_content)

        return report_path

    def _generate_sections(self, report_data: ReportData) -> List[ReportSection]:
        """生成报告章节"""
        sections = []

        # 1. 执行摘要
        if self.config.report_type in [ReportType.EXECUTIVE_SUMMARY, ReportType.TECHNICAL_ANALYSIS]:
            sections.append(self._create_executive_summary(report_data))

        # 2. 技术分析
        if self.config.report_type != ReportType.EXECUTIVE_SUMMARY:
            sections.append(self._create_technical_analysis(report_data))

        # 3. 解释详情
        if len(report_data.explanations) > 0:
            sections.append(self._create_explanation_details(report_data))

        # 4. 评估指标
        if report_data.evaluation_metrics:
            sections.append(self._create_evaluation_section(report_data))

        # 5. 比较分析
        if report_data.comparison_results:
            sections.append(self._create_comparison_section(report_data))

        # 6. 建议和结论
        sections.append(self._create_recommendations_section(report_data))

        return sections

    def _create_executive_summary(self, report_data: ReportData) -> ReportSection:
        """创建执行摘要"""
        content = f"""
        <h3>概述</h3>
        <p>本报告对故障诊断模型的解释性进行了全面分析。使用了 {len(report_data.explanations)} 个解释方法，
        评估了模型决策的透明度和可信度。</p>

        <h3>关键发现</h3>
        <ul>
            <li>模型类型: {report_data.model_info.get('model_name', '未知')}</li>
            <li>评估方法: {', '.join([exp.get_method_name() for exp in report_data.explanations])}</li>
            <li>生成时间: {report_data.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</li>
        </ul>

        <h3>主要建议</h3>
        <p>基于分析结果，建议在实际应用中结合多种解释方法，提高决策的可靠性和透明度。</p>
        """

        return ReportSection(
            title="执行摘要",
            content=content,
            order=1
        )

    def _create_technical_analysis(self, report_data: ReportData) -> ReportSection:
        """创建技术分析章节"""
        content = f"""
        <h3>模型信息</h3>
        <table>
            <tr><th>属性</th><th>值</th></tr>
            <tr><td>模型类型</td><td>{report_data.model_info.get('model_name', '未知')}</td></tr>
            <tr><td>参数数量</td><td>{report_data.model_info.get('parameters', '未知'):,}</td></tr>
            <tr><td>输入维度</td><td>{report_data.model_info.get('input_shape', '未知')}</td></tr>
            <tr><td>输出类别</td><td>{report_data.model_info.get('num_classes', '未知')}</td></tr>
        </table>

        <h3>解释方法分析</h3>
        <p>本次分析使用了以下解释方法:</p>
        <ul>
        """

        for exp in report_data.explanations:
            method = exp.get_method_name()
            metrics = exp.get_metrics()
            content += f"<li><strong>{method}</strong>: 均值={metrics.get('attribution_mean', 0):.4f}, "
            content += f"最大值={metrics.get('attribution_max', 0):.4f}, "
            content += f"稀疏度={metrics.get('attribution_sparsity', 0):.4f}</li>"

        content += "</ul>"

        return ReportSection(
            title="技术分析",
            content=content,
            order=2
        )

    def _create_explanation_details(self, report_data: ReportData) -> ReportSection:
        """创建解释详情章节"""
        content = "<h3>详细解释结果</h3>"

        for i, exp in enumerate(report_data.explanations):
            method = exp.get_method_name()
            content += f"<h4>{i+1}. {method} 方法</h4>"

            # 基本信息
            content += f"<p><strong>方法描述:</strong> {self._get_method_description(method)}</p>"
            content += f"<p><strong>适用场景:</strong> {self._get_method_use_case(method)}</p>"

            # 指标
            metrics = exp.get_metrics()
            content += f"<p><strong>性能指标:</strong> "
            content += f"均值={metrics.get('attribution_mean', 0):.4f}, "
            content += f"标准差={metrics.get('attribution_std', 0):.4f}, "
            content += f"最大值={metrics.get('attribution_max', 0):.4f}</p>"

        return ReportSection(
            title="解释详情",
            content=content,
            order=3
        )

    def _create_evaluation_section(self, report_data: ReportData) -> ReportSection:
        """创建评估章节"""
        content = "<h3>评估指标详情</h3>"

        if report_data.evaluation_metrics:
            for metric_name, metric_value in report_data.evaluation_metrics.items():
                content += f"<p><strong>{metric_name}:</strong> {metric_value:.4f}</p>"
        else:
            content += "<p>暂无评估指标数据。</p>"

        return ReportSection(
            title="评估分析",
            content=content,
            order=4
        )

    def _create_comparison_section(self, report_data: ReportData) -> ReportSection:
        """创建比较分析章节"""
        content = "<h3>方法比较分析</h3>"

        if report_data.comparison_results:
            content += "<table><tr><th>方法</th><th>计算时间(s)</th><th>质量分数</th><th>推荐指数</th></tr>"

            for method, result in report_data.comparison_results.items():
                if result:
                    time_val = result.get('single_time', 0)
                    quality = result.get('metrics', {}).get('attribution_mean', 0)
                    recommendation = self._get_recommendation_score(quality, time_val)

                    content += f"<tr><td>{method}</td><td>{time_val:.3f}</td><td>{quality:.4f}</td><td>{recommendation}</td></tr>"

            content += "</table>"
        else:
            content += "<p>暂无比较数据。</p>"

        return ReportSection(
            title="方法比较",
            content=content,
            order=5
        )

    def _create_recommendations_section(self, report_data: ReportData) -> ReportSection:
        """创建建议章节"""
        content = """
        <h3>使用建议</h3>
        <ul>
            <li><strong>实时应用:</strong> 推荐使用计算效率高的方法，如梯度显著性分析</li>
            <li><strong>详细分析:</strong> 推荐使用积分梯度或DeepLift等更准确的方法</li>
            <li><strong>混合使用:</strong> 结合多种方法可以获得更全面的解释</li>
        </ul>

        <h3>技术改进建议</h3>
        <ul>
            <li>考虑引入用户反馈机制，持续优化解释质量</li>
            <li>开发针对特定领域的解释模板和术语库</li>
            <li>增强交互式解释功能，支持用户深究</li>
        </ul>
        """

        return ReportSection(
            title="建议与结论",
            content=content,
            order=6
        )

    def _generate_visualizations(self, report_data: ReportData) -> List[str]:
        """生成可视化图表"""
        visualizations = []

        if not self.config.include_visualizations:
            return visualizations

        # 单个解释的可视化
        for exp in report_data.explanations:
            viz_path = self.viz_generator.generate_attribution_plot(exp)
            if viz_path:
                # 转换为相对路径
                rel_path = os.path.relpath(viz_path, str(self.output_dir))
                base64_path = self.viz_generator.embed_image_base64(viz_path)
                visualizations.append(base64_path)

        # 多个解释的比较图
        if len(report_data.explanations) > 1:
            comp_path = self.viz_generator.generate_metrics_comparison(report_data.explanations)
            if comp_path:
                base64_path = self.viz_generator.embed_image_base64(comp_path)
                visualizations.append(base64_path)

        # 方法比较图
        if report_data.comparison_results:
            method_comp_path = self.viz_generator.generate_method_comparison_chart(report_data.comparison_results)
            if method_comp_path:
                base64_path = self.viz_generator.embed_image_base64(method_comp_path)
                visualizations.append(base64_path)

        return visualizations

    def _get_report_title(self) -> str:
        """获取报告标题"""
        titles = {
            ReportType.SINGLE_EXPLANATION: "可解释性故障诊断报告",
            ReportType.BATCH_COMPARISON: "解释方法比较分析报告",
            ReportType.MODEL_EVALUATION: "模型可解释性评估报告",
            ReportType.TECHNICAL_ANALYSIS: "技术分析报告",
            ReportType.EXECUTIVE_SUMMARY: "执行摘要报告",
            ReportType.USER_FRIENDLY: "故障诊断解释报告"
        }

        base_title = titles.get(self.config.report_type, "可解释性分析报告")

        if self.config.auto_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_title += f"_{timestamp}"

        return base_title

    def _save_report(self, content: str) -> str:
        """保存报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.config.format == ReportFormat.HTML:
            filename = f"report_{timestamp}.html"
            file_path = self.output_dir / filename

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

        elif self.config.format == ReportFormat.JSON:
            filename = f"report_{timestamp}.json"
            file_path = self.output_dir / filename

            # 将HTML内容保存为JSON
            json_data = {
                'content': content,
                'generation_time': datetime.now().isoformat(),
                'config': self.config.__dict__
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)

        elif self.config.format == ReportFormat.MARKDOWN:
            filename = f"report_{timestamp}.md"
            file_path = self.output_dir / filename

            # 简单的HTML到Markdown转换
            markdown_content = self._html_to_markdown(content)

            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

        else:
            raise ValueError(f"不支持的报告格式: {self.config.format}")

        return str(file_path)

    def _html_to_markdown(self, html_content: str) -> str:
        """简单的HTML到Markdown转换"""
        # 这是一个简化的转换，实际应用中可能需要更复杂的处理
        import re

        # 移除HTML标签，保留文本内容
        text = re.sub(r'<[^>]+>', '\n', html_content)
        text = re.sub(r'\n+', '\n\n', text)  # 合并多个换行

        return text.strip()

    def _get_method_description(self, method: str) -> str:
        """获取方法描述"""
        descriptions = {
            'signal_path': '追踪信号在模型中的变换过程，提供透明性最高的解释',
            'integrated_gradients': '通过积分路径计算特征贡献，具有良好的理论基础',
            'deeplift': '基于激活传播的解释方法，计算效率较高',
            'saliency': '简单的梯度显著性分析，计算速度最快'
        }
        return descriptions.get(method, '未知解释方法')

    def _get_method_use_case(self, method: str) -> str:
        """获取方法适用场景"""
        use_cases = {
            'signal_path': '适用于需要理解模型内部机制的场景',
            'integrated_gradients': '适用于需要高精度解释的科研应用',
            'deeplift': '适用于需要平衡精度和效率的工业应用',
            'saliency': '适用于需要实时解释的在线系统'
        }
        return use_cases.get(method, '通用场景')

    def _get_recommendation_score(self, quality: float, time_cost: float) -> str:
        """计算推荐指数"""
        # 简单的评分逻辑
        if time_cost == 0:
            return "N/A"

        efficiency = quality / time_cost
        if efficiency > 0.5:
            return "★★★★★"
        elif efficiency > 0.3:
            return "★★★★☆"
        elif efficiency > 0.1:
            return "★★★☆☆"
        elif efficiency > 0.05:
            return "★★☆☆☆"
        else:
            return "★☆☆☆☆"


# 便捷函数
def generate_single_explanation_report(
    explanation: Explanation,
    model_info: Dict[str, Any],
    config: Optional[ReportConfig] = None
) -> str:
    """生成单个解释报告的便捷函数"""
    if config is None:
        config = ReportConfig(
            report_type=ReportType.SINGLE_EXPLANATION,
            format=ReportFormat.HTML
        )

    report_data = ReportData(
        explanations=[explanation],
        model_info=model_info
    )

    generator = ReportGenerator(config)
    return generator.generate_report(report_data)


def generate_comparison_report(
    explanations: List[Explanation],
    model_info: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None,
    config: Optional[ReportConfig] = None
) -> str:
    """生成比较报告的便捷函数"""
    if config is None:
        config = ReportConfig(
            report_type=ReportType.BATCH_COMPARISON,
            format=ReportFormat.HTML
        )

    report_data = ReportData(
        explanations=explanations,
        model_info=model_info,
        comparison_results=comparison_results
    )

    generator = ReportGenerator(config)
    return generator.generate_report(report_data)


# 使用示例
if __name__ == "__main__":
    from toolkit_integration.explainability.core.explanation import Explanation

    # 创建示例数据
    explanation_data = {
        'attributions': np.random.rand(1000),
        'original_signal': np.random.randn(1000)
    }
    explanation_meta = {
        'method': 'integrated_gradients',
        'model_name': 'TSPN'
    }
    explanation = Explanation(explanation_data, explanation_meta)

    model_info = {
        'model_name': 'TSPN',
        'parameters': 1000000,
        'input_shape': [1000, 2],
        'num_classes': 5
    }

    # 生成报告
    config = ReportConfig(
        report_type=ReportType.SINGLE_EXPLANATION,
        format=ReportFormat.HTML,
        include_visualizations=True
    )

    report_path = generate_single_explanation_report(explanation, model_info, config)
    print(f"报告已生成: {report_path}")