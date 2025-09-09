"""
Markdown指标报告生成器
生成详细的系统级指标分析报告
"""
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


class MetricsMarkdownReporter:
    """生成Markdown格式的指标报告"""
    
    def __init__(self, save_dir: str = "metrics_reports"):
        """初始化报告生成器
        
        Args:
            save_dir: 报告保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_report(self, 
                       system_metrics: Dict[str, Dict[str, float]],
                       global_metrics: Dict[str, float] = None,
                       phase: str = 'test',
                       experiment_name: str = 'multi_task_phm',
                       config_info: Dict[str, Any] = None) -> Path:
        """生成完整的Markdown报告
        
        Args:
            system_metrics: 系统级指标 {system_id: {metric: value}}
            global_metrics: 全局指标 {metric: value}
            phase: 实验阶段 (train/val/test)
            experiment_name: 实验名称
            config_info: 配置信息
            
        Returns:
            报告文件路径
        """
        
        report_lines = []
        
        # 报告头部
        report_lines.append(f"# Multi-Task PHM Metrics Report")
        report_lines.append(f"\n**Experiment**: {experiment_name}")
        report_lines.append(f"**Phase**: {phase}")
        report_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 添加配置信息
        if config_info:
            report_lines.append(f"\n**Configuration**:")
            for key, value in config_info.items():
                report_lines.append(f"- {key}: {value}")
                
        report_lines.append("\n---\n")
        
        # 1. 执行摘要
        report_lines.append("## 📋 Executive Summary\n")
        report_lines.extend(self._generate_executive_summary(system_metrics, global_metrics))
        
        # 2. 全局指标汇总
        if global_metrics:
            report_lines.append("\n## 📊 Global Metrics Summary\n")
            report_lines.extend(self._format_global_metrics(global_metrics))
        
        # 3. 系统级指标详情
        if system_metrics:
            report_lines.append("\n## 🔍 System-Level Metrics\n")
            report_lines.extend(self._format_system_metrics(system_metrics))
            
            # 4. 任务性能对比
            report_lines.append("\n## 📈 Task Performance Comparison\n")
            report_lines.extend(self._format_task_comparison(system_metrics))
            
            # 5. 系统性能排名
            report_lines.append("\n## 🏆 System Performance Ranking\n")
            report_lines.extend(self._format_system_ranking(system_metrics))
            
            # 6. 问题诊断
            report_lines.append("\n## ⚠️ Diagnostic Insights\n")
            report_lines.extend(self._generate_diagnostics(system_metrics))
        
        # 7. 详细数据
        report_lines.append("\n## 📈 Detailed Data\n")
        report_lines.extend(self._format_detailed_data(system_metrics))
        
        # 保存报告
        filename = f"{experiment_name}_{phase}_{self.timestamp}.md"
        filepath = self.save_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✅ Metrics report saved to: {filepath}")
        return filepath
    
    def _generate_executive_summary(self, system_metrics: Dict, global_metrics: Dict = None) -> List[str]:
        """生成执行摘要"""
        lines = []
        
        if not system_metrics:
            lines.append("*No system metrics available for analysis.*")
            return lines
        
        # 统计基本信息
        num_systems = len(system_metrics)
        all_metrics = set()
        for metrics in system_metrics.values():
            all_metrics.update(metrics.keys())
        
        # 识别任务
        tasks = set()
        for metric in all_metrics:
            if '_' in metric:
                task = metric.split('_')[0]
                tasks.add(task)
        
        lines.append(f"**Systems Evaluated**: {num_systems}")
        lines.append(f"**Tasks Analyzed**: {', '.join(sorted(tasks))}")
        lines.append(f"**Total Metrics**: {len(all_metrics)}")
        
        # 快速洞察
        lines.append(f"\n**Key Findings**:")
        
        # 找出最佳和最差系统
        if len(system_metrics) > 1:
            system_scores = self._calculate_system_scores(system_metrics)
            if system_scores:
                best_system = max(system_scores.items(), key=lambda x: x[1])
                worst_system = min(system_scores.items(), key=lambda x: x[1])
                lines.append(f"- 🥇 Best performing system: **{best_system[0]}** (score: {best_system[1]:.3f})")
                lines.append(f"- 🔴 Worst performing system: **{worst_system[0]}** (score: {worst_system[1]:.3f})")
        
        # 检查异常指标
        anomalies = self._detect_metric_anomalies(system_metrics)
        if anomalies:
            lines.append(f"- ⚠️ **{len(anomalies)} anomalies detected** requiring attention")
        else:
            lines.append(f"- ✅ No significant anomalies detected")
        
        return lines
    
    def _format_global_metrics(self, metrics: Dict[str, float]) -> List[str]:
        """格式化全局指标为Markdown表格"""
        lines = []
        
        # 按任务分组
        task_metrics = {}
        for key, value in metrics.items():
            # 尝试从key中提取任务名
            if '_' in key:
                # 假设格式为 task_metric 或 phase_task_metric
                parts = key.split('_')
                if len(parts) >= 2:
                    # 找到任务名（跳过phase前缀如'test_'）
                    task_start = 1 if parts[0] in ['train', 'val', 'test'] else 0
                    if task_start < len(parts) - 1:
                        task = parts[task_start]
                        metric = '_'.join(parts[task_start + 1:])
                        if task not in task_metrics:
                            task_metrics[task] = {}
                        task_metrics[task][metric] = value
        
        if not task_metrics:
            # 如果无法分组，直接显示
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    lines.append(f"| {key} | {value:.4f} |")
                else:
                    lines.append(f"| {key} | {value} |")
        else:
            # 创建分组表格
            lines.append("| Task | Metric | Value | Status |")
            lines.append("|------|--------|-------|--------|")
            
            for task, task_metrics_dict in sorted(task_metrics.items()):
                for metric, value in sorted(task_metrics_dict.items()):
                    if isinstance(value, float):
                        status = self._get_metric_status(metric, value)
                        lines.append(f"| {task} | {metric} | {value:.4f} | {status} |")
                    else:
                        lines.append(f"| {task} | {metric} | {value} | - |")
        
        return lines
    
    def _format_system_metrics(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """格式化系统级指标为详细表格"""
        lines = []
        
        if not system_metrics:
            lines.append("*No system-level metrics available*")
            return lines
        
        # 获取所有指标名称
        all_metrics = set()
        for sys_metrics in system_metrics.values():
            all_metrics.update(sys_metrics.keys())
        
        # 按任务分组指标
        task_groups = {}
        for metric in all_metrics:
            task = metric.split('_')[0] if '_' in metric else 'general'
            if task not in task_groups:
                task_groups[task] = []
            task_groups[task].append(metric)
        
        # 为每个任务创建表格
        for task, metrics in sorted(task_groups.items()):
            lines.append(f"\n### {task.capitalize()} Task Metrics\n")
            
            # 表头
            header = "| System ID |"
            separator = "|-----------|"
            for metric in sorted(metrics):
                metric_name = metric.replace(f"{task}_", "").replace('_', ' ').title()
                header += f" {metric_name} |"
                separator += "-----------|"
            header += " Status |"
            separator += "--------|"
            
            lines.append(header)
            lines.append(separator)
            
            # 数据行
            for sys_id, sys_metrics in sorted(system_metrics.items()):
                row = f"| **{sys_id}** |"
                system_status = "✅"
                
                for metric in sorted(metrics):
                    value = sys_metrics.get(metric, 'N/A')
                    if isinstance(value, float):
                        row += f" {value:.4f} |"
                        # 检查异常值
                        if self._is_metric_anomaly(metric, value):
                            system_status = "⚠️"
                    else:
                        row += f" {value} |"
                
                row += f" {system_status} |"
                lines.append(row)
        
        return lines
    
    def _format_task_comparison(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """创建任务间性能对比"""
        lines = []
        
        # 计算每个任务的统计信息
        task_stats = {}
        
        for sys_metrics in system_metrics.values():
            for metric, value in sys_metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    task = metric.split('_')[0] if '_' in metric else 'general'
                    if task not in task_stats:
                        task_stats[task] = []
                    task_stats[task].append(value)
        
        # 计算统计值
        lines.append("| Task | Avg Performance | Std Dev | Min | Max | Status |")
        lines.append("|------|----------------|---------|-----|-----|--------|")
        
        for task, values in sorted(task_stats.items()):
            if values:
                avg = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # 判断状态
                if avg > 0.8:
                    status = "✅ Excellent"
                elif avg > 0.6:
                    status = "🟡 Good"
                elif avg > 0.4:
                    status = "🟠 Fair"
                else:
                    status = "🔴 Poor"
                
                lines.append(f"| {task} | {avg:.4f} | {std:.4f} | {min_val:.4f} | {max_val:.4f} | {status} |")
        
        return lines
    
    def _format_system_ranking(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """创建系统性能排名"""
        lines = []
        
        # 计算每个系统的综合得分
        system_scores = self._calculate_system_scores(system_metrics)
        
        if not system_scores:
            lines.append("*Cannot calculate system ranking due to insufficient data*")
            return lines
        
        # 排序
        sorted_systems = sorted(system_scores.items(), key=lambda x: x[1], reverse=True)
        
        lines.append("| Rank | System ID | Overall Score | Performance Level |")
        lines.append("|------|-----------|---------------|-------------------|")
        
        for rank, (sys_id, score) in enumerate(sorted_systems, 1):
            # 添加奖牌
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else ""
            
            # 性能等级
            if score > 0.8:
                level = "Excellent"
            elif score > 0.6:
                level = "Good"
            elif score > 0.4:
                level = "Fair"
            else:
                level = "Needs Improvement"
            
            lines.append(f"| {medal} {rank} | **{sys_id}** | {score:.4f} | {level} |")
        
        return lines
    
    def _generate_diagnostics(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """生成诊断建议"""
        lines = []
        issues = []
        recommendations = set()
        
        # 检查各种问题
        for sys_id, metrics in system_metrics.items():
            for metric, value in metrics.items():
                if not isinstance(value, (int, float)):
                    continue
                
                # 检查AUROC异常
                if 'auroc' in metric.lower() and value < 0.5:
                    issues.append(f"- **System {sys_id}**: {metric} = {value:.4f} (< 0.5) - possible label inversion or severe class imbalance")
                    recommendations.add("Review anomaly detection labels and class balance")
                
                # 检查R2异常
                if 'r2' in metric.lower() and value < -1:
                    issues.append(f"- **System {sys_id}**: {metric} = {value:.4f} (< -1) - poor regression performance")
                    recommendations.add("Check RUL target scaling and data quality")
                
                # 检查准确率异常
                if 'acc' in metric.lower() and value < 0.3:
                    issues.append(f"- **System {sys_id}**: {metric} = {value:.4f} (< 0.3) - very poor classification performance")
                    recommendations.add("Review classification model and data preprocessing")
                
                # 检查F1异常
                if 'f1' in metric.lower() and value < 0.2:
                    issues.append(f"- **System {sys_id}**: {metric} = {value:.4f} (< 0.2) - poor precision/recall balance")
                    recommendations.add("Address class imbalance or model calibration")
                
                # 检查极值
                if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                    issues.append(f"- **System {sys_id}**: {metric} = {value} - invalid value detected")
                    recommendations.add("Investigate computational errors in metric calculation")
        
        if issues:
            lines.append("### 🚨 Issues Detected:\n")
            lines.extend(issues)
            lines.append("\n### 💡 Recommendations:\n")
            for rec in sorted(recommendations):
                lines.append(f"- {rec}")
        else:
            lines.append("### ✅ No Major Issues Detected\n")
            lines.append("All systems are performing within acceptable ranges.")
        
        # 添加通用建议
        lines.append("\n### 📋 General Recommendations:\n")
        lines.append("- Monitor system-specific trends across multiple epochs")
        lines.append("- Consider per-system hyperparameter optimization")
        lines.append("- Implement early stopping based on system-level performance")
        lines.append("- Use ensemble methods to leverage strengths of different systems")
        
        return lines
    
    def _format_detailed_data(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """格式化详细数据为可导出格式"""
        lines = []
        
        lines.append("### Raw Data (CSV Format)\n")
        lines.append("```csv")
        
        if system_metrics:
            # 获取所有指标名
            all_metrics = set()
            for metrics in system_metrics.values():
                all_metrics.update(metrics.keys())
            
            # 创建CSV头部
            header = "System_ID," + ",".join(sorted(all_metrics))
            lines.append(header)
            
            # 添加数据行
            for sys_id, metrics in sorted(system_metrics.items()):
                row = [sys_id]
                for metric in sorted(all_metrics):
                    value = metrics.get(metric, 'N/A')
                    if isinstance(value, float):
                        row.append(f"{value:.6f}")
                    else:
                        row.append(str(value))
                lines.append(",".join(row))
        
        lines.append("```")
        return lines
    
    def _calculate_system_scores(self, system_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """计算系统综合得分"""
        system_scores = {}
        
        for sys_id, metrics in system_metrics.items():
            valid_scores = []
            
            for metric, value in metrics.items():
                if isinstance(value, (int, float)) and not (np.isnan(value) or np.isinf(value)):
                    # 对不同类型的指标进行归一化
                    if 'auroc' in metric.lower() or 'acc' in metric.lower() or 'f1' in metric.lower():
                        # 这些指标越大越好，范围[0,1]
                        normalized = max(0, min(1, value))
                    elif 'loss' in metric.lower() or 'mae' in metric.lower() or 'mse' in metric.lower():
                        # 这些指标越小越好，使用倒数
                        if value > 0:
                            normalized = 1 / (1 + value)  # 范围[0,1]
                        else:
                            normalized = 0
                    elif 'r2' in metric.lower():
                        # R²理论上可以是负数，但通常期望正值
                        normalized = max(0, min(1, (value + 1) / 2))  # 将[-1,1]映射到[0,1]
                    else:
                        # 其他指标假设在[0,1]范围内
                        normalized = max(0, min(1, value))
                    
                    valid_scores.append(normalized)
            
            if valid_scores:
                system_scores[sys_id] = sum(valid_scores) / len(valid_scores)
        
        return system_scores
    
    def _detect_metric_anomalies(self, system_metrics: Dict[str, Dict[str, float]]) -> List[str]:
        """检测指标异常"""
        anomalies = []
        
        for sys_id, metrics in system_metrics.items():
            for metric, value in metrics.items():
                if self._is_metric_anomaly(metric, value):
                    anomalies.append(f"{sys_id}_{metric}")
        
        return anomalies
    
    def _is_metric_anomaly(self, metric: str, value: Any) -> bool:
        """判断是否为异常指标"""
        if not isinstance(value, (int, float)):
            return True
        
        if np.isnan(value) or np.isinf(value):
            return True
        
        metric_lower = metric.lower()
        
        # 检查各种异常模式
        if 'auroc' in metric_lower and value < 0.5:
            return True
        if 'r2' in metric_lower and value < -1:
            return True
        if ('acc' in metric_lower or 'f1' in metric_lower) and value < 0.3:
            return True
        if ('loss' in metric_lower or 'mae' in metric_lower or 'mse' in metric_lower) and value > 10:
            return True
        
        return False
    
    def _get_metric_status(self, metric: str, value: float) -> str:
        """获取指标状态"""
        if self._is_metric_anomaly(metric, value):
            return "⚠️"
        elif 'auroc' in metric.lower() or 'acc' in metric.lower() or 'f1' in metric.lower():
            return "✅" if value > 0.7 else "🟡" if value > 0.5 else "🔴"
        elif 'r2' in metric.lower():
            return "✅" if value > 0.5 else "🟡" if value > 0 else "🔴"
        else:
            return "✅"


if __name__ == '__main__':
    """单元测试"""
    print("=== Testing MetricsMarkdownReporter ===")
    
    # 创建测试数据
    system_metrics = {
        'system_1': {
            'classification_acc': 0.95,
            'classification_f1': 0.93,
            'anomaly_auroc': 0.87,
            'rul_mae': 0.45,
            'signal_r2': 0.78
        },
        'system_5': {
            'classification_acc': 0.85,
            'classification_f1': 0.82,
            'anomaly_auroc': 0.02,  # 异常值
            'rul_mae': 1.23,
            'signal_r2': -0.45
        },
        'system_13': {
            'classification_acc': 0.98,
            'classification_f1': 0.97,
            'anomaly_auroc': 0.91,
            'rul_mae': 0.34,
            'signal_r2': 0.89
        }
    }
    
    global_metrics = {
        'test_classification_acc': 0.926,
        'test_anomaly_auroc': 0.600,
        'test_rul_mae': 0.674,
        'test_signal_r2': 0.407
    }
    
    # 创建报告生成器
    reporter = MetricsMarkdownReporter(save_dir="test_reports")
    
    # 生成报告
    report_path = reporter.generate_report(
        system_metrics=system_metrics,
        global_metrics=global_metrics,
        phase='test',
        experiment_name='test_experiment',
        config_info={
            'model': 'M_01_ISFM',
            'tasks': 'classification, anomaly_detection, rul_prediction, signal_prediction',
            'batch_size': 32
        }
    )
    
    print(f"✅ Test report generated: {report_path}")
    print("\n✓ MetricsMarkdownReporter test completed!")