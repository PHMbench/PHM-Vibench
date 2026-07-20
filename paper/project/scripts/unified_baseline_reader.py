#!/usr/bin/env python3
"""
统一基线结果表读取器

用于从统一基线结果表中自动读取模型配置和性能数据
支持动态更新模型信息，避免硬编码
"""

import os
import re
import json
from typing import Dict, List, Any
from pathlib import Path

class UnifiedBaselineReader:
    """统一基线结果表读取器"""

    def __init__(self, baseline_table_path: str = None):
        """
        初始化读取器

        Args:
            baseline_table_path: 统一基线结果表路径
        """
        if baseline_table_path is None:
            # 默认路径
            self.baseline_table_path = "../../doc/12_1/codex/unified_baseline_results_table_12_01_v2.md"
        else:
            self.baseline_table_path = baseline_table_path

        self.models_info = {}
        self.last_modified = None

    def load_baseline_table(self) -> Dict[str, Any]:
        """
        加载统一基线结果表

        Returns:
            models_info: 模型信息字典
        """
        # 检查文件是否存在
        if not os.path.exists(self.baseline_table_path):
            print(f"⚠️  统一基线结果表不存在: {self.baseline_table_path}")
            return self._get_default_models()

        # 检查文件修改时间
        current_modified = os.path.getmtime(self.baseline_table_path)
        if self.last_modified == current_modified and self.models_info:
            return self.models_info

        print(f"📖 读取统一基线结果表: {self.baseline_table_path}")

        try:
            with open(self.baseline_table_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 解析表格内容
            self.models_info = self._parse_baseline_table(content)
            self.last_modified = current_modified

            print(f"✅ 成功加载 {len(self.models_info)} 个模型的配置")
            return self.models_info

        except Exception as e:
            print(f"❌ 读取统一基线结果表失败: {e}")
            print("🔄 使用默认模型配置")
            return self._get_default_models()

    def _parse_baseline_table(self, content: str) -> Dict[str, Any]:
        """
        解析基线结果表格

        Args:
            content: Markdown文件内容

        Returns:
            models_info: 解析后的模型信息
        """
        models_info = {}

        # 查找表格部分
        table_pattern = r'\|.*\|.*\|.*\|.*\|.*\|'
        table_matches = re.findall(table_pattern, content)

        if not table_matches:
            print("⚠️  未找到表格内容")
            return self._get_default_models()

        # 解析表头
        headers = self._parse_table_row(table_matches[0])

        # 解析模型行
        for i, row in enumerate(table_matches[2:], 2):  # 跳过表头和分隔符
            if i >= len(table_matches):
                break

            cells = self._parse_table_row(row)
            if len(cells) < 3:
                continue

            model_name = cells[0].strip()

            # 提取准确率（可能包含百分号）
            accuracy_str = cells[1].strip()
            accuracy = self._extract_accuracy(accuracy_str)

            # 构建模型信息
            models_info[model_name] = {
                'accuracy': accuracy,
                'config': f"configs/unified_baseline/config_{model_name.replace(' ', '_')}.yaml",
                'model_type': self._infer_model_type(model_name),
                'explainability': self._infer_explainability(model_name, accuracy),
                'status': cells[2].strip() if len(cells) > 2 else 'unknown',
                'notes': cells[3].strip() if len(cells) > 3 else ''
            }

        return models_info

    def _parse_table_row(self, row: str) -> List[str]:
        """解析表格行"""
        # 移除首尾的|，然后按|分割
        row = row.strip('|')
        cells = [cell.strip() for cell in row.split('|')]
        return cells

    def _extract_accuracy(self, accuracy_str: str) -> float:
        """从字符串中提取准确率数值"""
        # 移除百分号和其他字符
        accuracy_clean = re.sub(r'[^\d.]', '', accuracy_str)
        if accuracy_clean:
            return float(accuracy_clean)
        return 0.0

    def _infer_model_type(self, model_name: str) -> str:
        """推断模型类型（本征/事后）"""
        # 根据模型名称推断类型
        intrinsic_models = ['TSPN', 'Fusion1D2D', 'OperatorAttention', 'FuzzyLogic']
        posthoc_models = ['MoE']

        if model_name in intrinsic_models:
            return 'intrinsic'
        elif model_name in posthoc_models:
            return 'posthoc'
        else:
            return 'unknown'

    def _infer_explainability(self, model_name: str, accuracy: float) -> str:
        """推断可解释性水平"""
        if model_name == 'OperatorAttention':
            return 'very_high'
        elif accuracy > 90:
            return 'high'
        elif accuracy > 60:
            return 'medium'
        else:
            return 'high'  # 理论模型通常有高可解释性

    def _get_default_models(self) -> Dict[str, Any]:
        """获取默认模型配置（当无法读取表格时使用）"""
        return {
            'TSPN': {
                'accuracy': 92.0,
                'config': 'configs/unified_baseline/config_TSPN.yaml',
                'model_type': 'intrinsic',
                'explainability': 'high',
                'status': '✅ 可靠基线',
                'notes': '透明信号处理'
            },
            'Fusion1D2D': {
                'accuracy': 99.57,
                'config': 'configs/unified_baseline/config_Fusion1D2D.yaml',
                'model_type': 'intrinsic',
                'explainability': 'high',
                'status': '✅ 业界领先',
                'notes': '多模态融合'
            },
            'MoE': {
                'accuracy': 63.04,
                'config': 'configs/unified_baseline/config_MoE.yaml',
                'model_type': 'posthoc',
                'explainability': 'medium',
                'status': '✅ 概念验证',
                'notes': '物理约束专家系统'
            },
            'OperatorAttention': {
                'accuracy': 20.0,
                'config': 'configs/unified_baseline/config_OperatorAttention.yaml',
                'model_type': 'intrinsic',
                'explainability': 'very_high',
                'status': '🔄 优化中',
                'notes': '算子级注意力机制'
            },
            'FuzzyLogic': {
                'accuracy': 20.0,
                'config': 'configs/unified_baseline/config_FuzzyLogic.yaml',
                'model_type': 'intrinsic',
                'explainability': 'high',
                'status': '⚠️ 待优化',
                'notes': '模糊逻辑推理系统'
            }
        }

    def get_model_list(self) -> List[str]:
        """获取所有模型名称列表"""
        models_info = self.load_baseline_table()
        return list(models_info.keys())

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取特定模型的信息

        Args:
            model_name: 模型名称

        Returns:
            model_info: 模型信息字典
        """
        models_info = self.load_baseline_table()
        return models_info.get(model_name, {})

    def get_models_by_type(self, model_type: str) -> List[str]:
        """
        根据类型获取模型列表

        Args:
            model_type: 模型类型 ('intrinsic' 或 'posthoc')

        Returns:
            models: 指定类型的模型列表
        """
        models_info = self.load_baseline_table()
        return [name for name, info in models_info.items()
                if info.get('model_type') == model_type]

    def export_config(self, output_path: str):
        """
        导出模型配置为JSON文件

        Args:
            output_path: 输出文件路径
        """
        models_info = self.load_baseline_table()

        # 创建输出目录
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 保存配置
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(models_info, f, indent=2, ensure_ascii=False)

        print(f"✅ 模型配置已导出到: {output_path}")

    def print_summary(self):
        """打印模型摘要信息"""
        models_info = self.load_baseline_table()

        print("\n" + "="*60)
        print("📊 统一基线模型摘要")
        print("="*60)

        print(f"总模型数: {len(models_info)}")

        # 按类型统计
        intrinsic_count = len([m for m in models_info.values()
                             if m.get('model_type') == 'intrinsic'])
        posthoc_count = len([m for m in models_info.values()
                           if m.get('model_type') == 'posthoc'])

        print(f"本征解释模型: {intrinsic_count}")
        print(f"事后解释模型: {posthoc_count}")

        # 性能统计
        accuracies = [m.get('accuracy', 0) for m in models_info.values()
                     if m.get('accuracy', 0) > 0]
        if accuracies:
            avg_accuracy = sum(accuracies) / len(accuracies)
            max_accuracy = max(accuracies)
            print(f"平均准确率: {avg_accuracy:.2f}%")
            print(f"最高准确率: {max_accuracy:.2f}%")

        print("\n模型详情:")
        print("-" * 60)
        for name, info in models_info.items():
            status = info.get('status', 'unknown')
            accuracy = info.get('accuracy', 0)
            exp_type = info.get('model_type', 'unknown')
            notes = info.get('notes', '')
            print(f"{name:20} | {accuracy:6.2f}% | {exp_type:10} | {status:15} | {notes}")
        print("-" * 60)


def main():
    """测试函数"""
    # 创建读取器
    reader = UnifiedBaselineReader()

    # 加载配置
    models_info = reader.load_baseline_table()

    # 打印摘要
    reader.print_summary()

    # 导出配置
    output_dir = "Paper/Explainable_FD_Toolkit/configs"
    os.makedirs(output_dir, exist_ok=True)
    reader.export_config(f"{output_dir}/unified_baseline_models.json")

    # 测试获取特定模型信息
    print("\n🔍 TSPN模型信息:")
    tspn_info = reader.get_model_info('TSPN')
    for key, value in tspn_info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()