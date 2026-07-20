#!/usr/bin/env python3
"""
Explainable FD Toolkit - 实时故障诊断演示

展示从信号输入到维护决策的完整流程，演示Explainable FD Toolkit
在工业实时场景中的应用价值。

支持的模型：
- TSPN: 透明信号处理网络 (92.0%准确率)
- Fusion1D2D: 1D-2D融合网络 (99.57%准确率)

演示场景：
1. 实时信号流诊断
2. 批量信号分析
3. 维护决策支持
4. 解释报告生成

使用方法:
cd Paper/Explainable_FD_Toolkit
python demos/real_time_diagnosis_demo.py --model TSPN --mode realtime
python demos/real_time_diagnosis_demo.py --model Fusion1D2D --mode batch
"""

import os
import sys
import time
import json
import argparse
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

# 添加路径以便导入模块
toolkit_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, toolkit_root)

# 导入工具模块
try:
    from scripts.run_unified_explain_eval import UnifiedBaselineExplainer
    from toolkit_integration.auto_explanation_report_generator import (
        DiagnosisResult, ExplanationResult, MaintenanceRecommendation, ReportGenerator
    )
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保在正确的目录下运行此脚本")
    sys.exit(1)

class RealTimeDiagnosticSystem:
    """实时故障诊断系统"""

    def __init__(self, model_name: str = 'TSPN'):
        self.model_name = model_name
        self.model_config = self._get_model_config(model_name)
        self.explainer = UnifiedBaselineExplainer(model_name, self.model_config)
        self.report_generator = ReportGenerator()
        self.alert_queue = queue.Queue()
        self.diagnosis_history = []
        self.is_running = False

        # 模拟实时数据流
        self.signal_generator = SignalDataGenerator()

        print(f"🚀 初始化{model_name}实时诊断系统...")

    def _get_model_config(self, model_name: str) -> Dict[str, Any]:
        """获取模型配置"""
        model_configs = {
            'TSPN': {
                'accuracy': 92.0,
                'explainability': 'intrinsic',
                'description': '透明信号处理网络',
                'signal_types': ['vibration', 'temperature', 'acoustic']
            },
            'Fusion1D2D': {
                'accuracy': 99.57,
                'explainability': 'intrinsic',
                'description': '1D-2D融合网络',
                'signal_types': ['vibration', 'temperature', 'acoustic', 'time_frequency']
            }
        }
        return model_configs.get(model_name, {})

    def start_real_time_mode(self, duration_seconds: int = 60):
        """启动实时诊断模式"""
        print(f"\n🎯 启动实时诊断模式 (持续{duration_seconds}秒)")
        print("=" * 60)

        self.is_running = True
        start_time = time.time()

        try:
            while time.time() - start_time < duration_seconds and self.is_running:
                # 生成模拟实时信号
                signal_data = self.signal_generator.generate_signal()

                # 实时诊断
                self._process_real_time_signal(signal_data)

                # 模拟实时采样率 (1Hz)
                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\n⏹️ 用户中断，停止实时诊断")
        finally:
            self.is_running = False
            self._display_session_summary()

    def _process_real_time_signal(self, signal_data: np.ndarray):
        """处理实时信号数据"""

        # 执行诊断
        diagnosis = self._diagnose_signal(signal_data)

        # 生成解释
        explanation = self._explain_diagnosis(diagnosis, signal_data)

        # 生成维护建议
        maintenance = self._generate_maintenance_plan(diagnosis)

        # 记录诊断历史
        diagnosis_record = {
            'timestamp': datetime.now().isoformat(),
            'diagnosis': diagnosis,
            'explanation': explanation,
            'maintenance': maintenance
        }
        self.diagnosis_history.append(diagnosis_record)

        # 检查是否需要告警
        if diagnosis.confidence > 0.85 and diagnosis.fault_severity in ['High', 'Critical']:
            self._generate_alert(diagnosis, explanation)

        # 显示实时结果
        self._display_real_time_result(diagnosis_record)

    def run_batch_analysis(self, signal_files: List[str], output_dir: str = 'batch_results'):
        """运行批量分析模式"""
        print(f"\n📊 批量分析模式 - {len(signal_files)}个信号文件")
        print("=" * 60)

        os.makedirs(output_dir, exist_ok=True)
        batch_results = []

        for i, signal_file in enumerate(signal_files, 1):
            print(f"🔍 处理文件 {i}/{len(signal_files)}: {os.path.basename(signal_file)}")

            try:
                # 加载信号数据 (模拟)
                signal_data = self._load_signal_file(signal_file)

                # 执行诊断
                diagnosis = self._diagnose_signal(signal_data)

                # 生成解释
                explanation = self._explain_diagnosis(diagnosis, signal_data)

                # 生成维护建议
                maintenance = self._generate_maintenance_plan(diagnosis)

                # 生成详细报告
                report_path = os.path.join(output_dir, f"batch_report_{i}_{os.path.basename(signal_file).replace('.npy', '.html')}")
                self.report_generator.generate_comprehensive_report(
                    model_name=self.model_name,
                    signal_data=signal_data,
                    diagnosis=diagnosis,
                    explanation=explanation,
                    maintenance=maintenance,
                    save_path=report_path
                )

                batch_results.append({
                    'file': signal_file,
                    'diagnosis': diagnosis,
                    'report_path': report_path
                })

                print(f"  ✅ 报告生成: {report_path}")

            except Exception as e:
                print(f"  ❌ 处理失败: {e}")

        # 生成批量分析报告
        self._generate_batch_summary(batch_results, output_dir)

    def _diagnose_signal(self, signal_data: np.ndarray) -> DiagnosisResult:
        """诊断信号"""

        # 模拟模型预测 (实际实现中需要真实模型)
        fault_types = ['Normal', 'IF', 'OF', 'BF', 'RF']
        severities = ['Low', 'Medium', 'High', 'Critical']

        # 随机生成故障类型（偏向于特定模式）
        if self.model_name == 'TSPN':
            # TSPN倾向于检测某些类型故障
            fault_probs = [0.7, 0.1, 0.1, 0.05, 0.05]  # Normal概率较高
        else:
            # Fusion1D2D整体性能更好
            fault_probs = [0.6, 0.1, 0.1, 0.1, 0.1]

        fault_idx = np.random.choice(len(fault_types), p=fault_probs)
        fault_type = fault_types[fault_idx]

        # 根据故障类型确定严重程度
        severity_map = {
            'Normal': 'Low',
            'IF': 'High',
            'OF': 'High',
            'BF': 'Medium',
            'RF': 'Medium'
        }

        return DiagnosisResult(
            fault_type=fault_type,
            fault_severity=severity_map.get(fault_type, 'Medium'),
            confidence=np.random.uniform(0.8, 0.98),
            prediction_time=datetime.now().strftime('%H:%M:%S'),
            signal_statistics=self._calculate_signal_stats(signal_data)
        )

    def _explain_diagnosis(self, diagnosis: DiagnosisResult, signal_data: np.ndarray) -> ExplanationResult:
        """解释诊断结果"""

        # 生成解释特征
        if self.model_name == 'TSPN':
            key_features = self._generate_tspn_features(signal_data, diagnosis)
            signal_path = self._generate_tspn_signal_path(diagnosis)
        else:
            key_features = self._generate_fusion_features(signal_data, diagnosis)
            signal_path = self._generate_fusion_signal_path(diagnosis)

        return ExplanationResult(
            explanation_type='intrinsic',
            key_features=key_features,
            signal_path=signal_path,
            importance_scores={
                'frequency_domain': 0.7,
                'time_domain': 0.3
            },
            visualizations={}
        )

    def _generate_maintenance_plan(self, diagnosis: DiagnosisResult) -> MaintenanceRecommendation:
        """生成维护计划"""

        # 根据故障类型和严重程度生成维护建议
        maintenance_actions = {
            'Normal': ['继续正常运行', '定期监测信号'],
            'IF': [
                '检查轴承内圈状态',
                '测量轴承温度',
                '准备更换轴承',
                '检查润滑系统'
            ],
            'OF': [
                '检查轴承外圈',
                '测量轴承温度',
                '准备更换轴承',
                '检查对中状态'
            ],
            'BF': [
                '检查滚动体状况',
                '测量轴承温度',
                '准备更换轴承',
                '检查润滑质量'
            ],
            'RF': [
                '检查保持架完整性',
                '测量轴承温度',
                '准备更换轴承',
                '检查安装精度'
            ]
        }

        # 成本估算
        cost_estimates = {
            'Normal': '无成本',
            'IF': '¥5,000-8,000',
            'OF': '¥4,500-7,000',
            'BF': '¥4,000-6,000',
            'RF': '¥3,500-5,500'
        }

        # 时间估算
        time_estimates = {
            'Normal': '0分钟',
            'IF': '3-5小时',
            'OF': '3-5小时',
            'BF': '2-4小时',
            'RF': '2-4小时'
        }

        return MaintenanceRecommendation(
            urgency_level=diagnosis.fault_severity,
            recommended_actions=maintenance_actions.get(diagnosis.fault_type, []),
            estimated_cost=cost_estimates.get(diagnosis.fault_type, 'Unknown'),
            time_required=time_estimates.get(diagnosis.fault_type, 'Unknown'),
            safety_notes=self._get_safety_notes(diagnosis)
        )

    def _generate_alert(self, diagnosis: DiagnosisResult, explanation: ExplanationResult):
        """生成告警"""

        alert = {
            'alert_id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'fault_type': diagnosis.fault_type,
            'severity': diagnosis.fault_severity,
            'confidence': diagnosis.confidence,
            'key_explanation': f"基于{len(explanation.signal_path)}步决策路径的诊断，"
                             f"关键特征重要性分布清晰",
            'recommended_actions': self._get_immediate_actions(diagnosis),
            'status': 'ACTIVE'
        }

        self.alert_queue.put(alert)
        self._display_alert(alert)

    def _display_real_time_result(self, diagnosis_record: Dict[str, Any]):
        """显示实时诊断结果"""
        diagnosis = diagnosis_record['diagnosis']

        print(f"\n⏰ [{diagnosis.prediction_time}] 新的诊断结果:")
        print(f"  📊 故障类型: {diagnosis.fault_type}")
        print(f"  ⚠️  严重程度: {diagnosis.fault_severity}")
        print(f"  🎯 置信度: {diagnosis.confidence:.1%}")

        if diagnosis.fault_type != 'Normal':
            print(f"  🔧 维护建议: {self.model_config['description']}已生成详细报告")

    def _display_alert(self, alert: Dict[str, Any]):
        """显示告警"""
        print(f"\n🚨 高优先级告警 [{alert['timestamp'][:19]}]")
        print(f"  模型: {alert['model']}")
        print(f"  故障: {alert['fault_type']} (严重程度: {alert['severity']})")
        print(f"  置信度: {alert['confidence']:.1%}")
        print(f"  建议行动: {'; '.join(alert['recommended_actions'][:2])}...")

    def _display_session_summary(self):
        """显示会话总结"""
        print("\n" + "=" * 60)
        print("📈 实时诊断会话总结")
        print("=" * 60)
        print(f"🤖 模型: {self.model_name}")
        print(f"📊 总诊断数: {len(self.diagnosis_history)}")

        if self.diagnosis_history:
            # 统计故障分布
            fault_counts = {}
            severity_counts = {}

            for record in self.diagnosis_history:
                diagnosis = record['diagnosis']
                fault_counts[diagnosis.fault_type] = fault_counts.get(diagnosis.fault_type, 0) + 1
                severity_counts[diagnosis.fault_severity] = severity_counts.get(diagnosis.fault_severity, 0) + 1

            print("\n📋 故障类型分布:")
            for fault_type, count in fault_counts.items():
                print(f"  {fault_type}: {count}次 ({count/len(self.diagnosis_history)*100:.1f}%)")

            print("\n⚠️ 严重程度分布:")
            for severity, count in severity_counts.items():
                print(f"  {severity}: {count}次 ({count/len(self.diagnosis_history)*100:.1f}%)")

            # 计算平均置信度
            avg_confidence = np.mean([r['diagnosis'].confidence for r in self.diagnosis_history])
            print(f"\n🎯 平均置信度: {avg_confidence:.1%}")

    def _generate_batch_summary(self, batch_results: List[Dict], output_dir: str):
        """生成批量分析总结"""

        total_files = len(batch_results)
        fault_counts = {}

        for result in batch_results:
            diagnosis = result['diagnosis']
            fault_counts[diagnosis.fault_type] = fault_counts.get(diagnosis.fault_type, 0) + 1

        # 生成总结报告
        summary_path = os.path.join(output_dir, 'batch_analysis_summary.json')

        summary = {
            'analysis_time': datetime.now().isoformat(),
            'model': self.model_name,
            'total_files': total_files,
            'fault_distribution': fault_counts,
            'reports_generated': len(batch_results),
            'model_accuracy': self.model_config['accuracy']
        }

        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"\n📊 批量分析总结:")
        print(f"  总文件数: {total_files}")
        print(f"  模型准确率: {self.model_config['accuracy']}%")
        print(f"  生成报告: {len(batch_results)}个")
        print(f"  故障分布: {fault_counts}")
        print(f"  总结文件: {summary_path}")

    def _calculate_signal_stats(self, signal_data: np.ndarray) -> Dict[str, float]:
        """计算信号统计特征"""
        return {
            'length': len(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak': np.max(np.abs(signal_data)),
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'kurtosis': self._calculate_kurtosis(signal_data),
            'skewness': self._calculate_skewness(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2))
        }

    def _calculate_kurtosis(self, signal_data: np.ndarray) -> float:
        """计算峭度"""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            return 0
        return np.mean(((signal_data - mean) / std) ** 4)

    def _calculate_skewness(self, signal_data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(signal_data)
        std = np.std(signal_data)
        if std == 0:
            return 0
        return np.mean(((signal_data - mean) / std) ** 3)

    def _generate_tspn_features(self, signal_data: np.ndarray, diagnosis: DiagnosisResult) -> List[Dict[str, Any]]:
        """生成TSPN特征"""
        # 模拟FFT特征
        fft_data = np.fft.fft(signal_data)
        freq_bins = np.fft.fftfreq(len(fft_data))

        # 找到主要频率峰值
        peaks = []
        for i in range(10, len(fft_data)//2):
            if np.abs(fft_data[i]) > np.mean(np.abs(fft_data)) * 5:
                peaks.append({
                    'name': f'FFT_Peak_{len(peaks)+1}',
                    'importance': np.abs(fft_data[i]) / np.max(np.abs(fft_data)),
                    'value': f"{abs(freq_bins[i]):.1f} Hz"
                })

        return peaks[:4]  # 返回前4个主要峰值

    def _generate_fusion_features(self, signal_data: np.ndarray, diagnosis: DiagnosisResult) -> List[Dict[str, Any]]:
        """生成Fusion1D2D特征"""
        features = []

        # 1D时域特征
        features.append({
            'name': 'Time_Domain_RMS',
            'importance': 0.3,
            'value': f"{np.sqrt(np.mean(signal_data**2)):.3f}"
        })

        features.append({
            'name': 'Time_Domain_Peak',
            'importance': 0.2,
            'value': f"{np.max(np.abs(signal_data)):.3f}"
        })

        # 2D时频图特征 (模拟)
        features.append({
            'name': 'Time_Freq_Peak_1D',
            'importance': 0.25,
            'value': "123.5 Hz"
        })

        features.append({
            'name': 'Time_Freq_Peak_2D',
            'importance': 0.25,
            'value': "247.0 Hz"
        })

        return features

    def _generate_tspn_signal_path(self, diagnosis: DiagnosisResult) -> List[str]:
        """生成TSPN信号处理路径"""
        if diagnosis.fault_type == 'Normal':
            return [
                '输入信号采集 (12kHz, 4096点)',
                'FFT变换 → 频谱分析',
                '特征提取 (时域+频域)',
                '分类器预测 (正常状态)'
            ]
        else:
            return [
                '输入信号采集 (12kHz, 4096点)',
                'FFT变换 → 频谱分析',
                '故障特征检测',
                '特征提取 (峰值+能量)',
                '分类器预测 (故障诊断)'
            ]

    def _generate_fusion_signal_path(self, diagnosis: DiagnosisResult) -> List[str]:
        """生成Fusion1D2D信号处理路径"""
        return [
            '输入信号采集 (多传感器)',
            '1D时序信号处理 (RMS, Peak, Crest Factor)',
            '2D时频图生成 (STFT, Wavelet)',
            '统计特征提取 (Entropy, Skewness, Kurtosis)',
            '多模态特征融合',
            '注意力加权融合',
            '分类器预测 (高精度诊断)'
        ]

    def _get_immediate_actions(self, diagnosis: DiagnosisResult) -> List[str]:
        """获取立即行动建议"""
        if diagnosis.fault_severity == 'Critical':
            return [
                '立即停止设备运行',
                '通知维护团队',
                '安排紧急检修'
            ]
        elif diagnosis.fault_severity == 'High':
            return [
                '在下次维护周期内处理',
                '加强监控频率',
                '准备备件'
            ]
        elif diagnosis.fault_severity == 'Medium':
            return [
                '计划性检查',
                '增加监测频率',
                '记录观察结果'
            ]
        else:
            return [
                '正常运行',
                '继续监测'
            ]

    def _get_safety_notes(self, diagnosis: DiagnosisResult) -> List[str]:
        """获取安全注意事项"""
        if diagnosis.fault_type == 'Normal':
            return [
                '设备运行正常',
                '定期维护检查'
            ]
        else:
            return [
                '停机前确保安全断电',
                '佩戴适当的防护装备',
                '遵循设备操作规程',
                '检查相关安全装置'
            ]

    def _load_signal_file(self, file_path: str) -> np.ndarray:
        """加载信号文件"""
        # 模拟加载信号文件
        return np.random.randn(4096) * np.random.uniform(0.1, 2.0)

class SignalDataGenerator:
    """信号数据生成器"""

    def __init__(self):
        self.time_index = 0
        self.base_patterns = self._create_base_patterns()

    def _create_base_patterns(self) -> Dict[str, np.ndarray]:
        """创建基准信号模式"""
        length = 4096

        # 正常信号
        normal_signal = np.random.randn(length) * 0.1

        # 内圈故障信号 (添加特定频率成分)
        if_signal = normal_signal.copy()
        bpfi = 150  # 内圈故障频率
        for i in range(1, 4):  # 添加谐波
            if_signal += 0.3 * np.sin(2 * np.pi * bpfi * i * np.linspace(0, 1, length))

        # 外圈故障信号
        of_signal = normal_signal.copy()
        bpfo = 100
        for i in range(1, 3):
            of_signal += 0.2 * np.sin(2 * np.pi * bpfo * i * np.linspace(0, 1, length))

        return {
            'Normal': normal_signal,
            'IF': if_signal,
            'OF': of_signal,
            'BF': normal_signal * 1.5,  # 滚动体故障模拟
            'RF': normal_signal + np.random.randn(length) * 0.5  # 保持架故障
        }

    def generate_signal(self, fault_type: Optional[str] = None) -> np.ndarray:
        """生成信号数据"""
        self.time_index += 1

        if fault_type is None:
            # 随机选择故障类型
            fault_types = ['Normal', 'Normal', 'Normal', 'IF', 'IF', 'OF', 'BF', 'RF']
            fault_type = np.random.choice(fault_types)

        base_signal = self.base_patterns[fault_type]

        # 添加时间变化和噪声
        time_factor = 1 + 0.1 * np.sin(2 * np.pi * 0.01 * self.time_index)
        noise = np.random.randn(len(base_signal)) * 0.05

        signal = base_signal * time_factor + noise

        return signal

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='实时故障诊断演示')
    parser.add_argument('--model', type=str, default='TSPN',
                       choices=['TSPN', 'Fusion1D2D'],
                       help='诊断模型')
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['realtime', 'batch', 'demo'],
                       help='运行模式')
    parser.add_argument('--duration', type=int, default=30,
                       help='实时模式持续时间(秒)')
    parser.add_argument('--files', type=str, nargs='*',
                       help='批量模式信号文件路径')
    parser.add_argument('--output', type=str, default='demo_results',
                       help='输出目录')

    args = parser.parse_args()

    print("🚀 Explainable FD Toolkit - 实时故障诊断演示")
    print("=" * 60)
    print(f"🤖 诊断模型: {args.model}")
    print(f"🎯 运行模式: {args.mode}")
    print("=" * 60)

    # 初始化诊断系统
    diagnostic_system = RealTimeDiagnosticSystem(args.model)

    if args.mode == 'realtime':
        # 实时诊断模式
        diagnostic_system.start_real_time_mode(args.duration)

    elif args.mode == 'batch':
        # 批量分析模式
        if not args.files:
            # 生成模拟文件列表
            args.files = [f'signal_{i}.npy' for i in range(1, 6)]

        diagnostic_system.run_batch_analysis(args.files, args.output)

    elif args.mode == 'demo':
        # 演示模式 - 快速展示所有功能
        print("\n🎬 演示模式 - 快速展示所有功能")
        print("-" * 40)

        # 演示实时诊断 (5秒)
        print("1. 实时诊断演示 (5秒)...")
        diagnostic_system.start_real_time_mode(5)

        # 演示批量分析
        print("\n2. 批量分析演示...")
        demo_files = [f'demo_signal_{i}.npy' for i in range(1, 4)]
        diagnostic_system.run_batch_analysis(demo_files, 'demo_results')

        # 演示报告生成
        print("\n3. 报告生成演示...")
        signal_data = diagnostic_system.signal_generator.generate_signal('IF')
        diagnosis = diagnostic_system._diagnose_signal(signal_data)
        explanation = diagnostic_system._explain_diagnosis(diagnosis, signal_data)
        maintenance = diagnostic_system._generate_maintenance_plan(diagnosis)

        demo_report_path = diagnostic_system.report_generator.generate_comprehensive_report(
            model_name=args.model,
            signal_data=signal_data,
            diagnosis=diagnosis,
            explanation=explanation,
            maintenance=maintenance,
            save_path=f'demo_report_{args.model}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        )

        print(f"✅ 演示报告: {demo_report_path}")

        # 演示告警功能
        print("\n4. 告警功能演示...")
        diagnostic_system._generate_alert(diagnosis, explanation)

        print("\n🎉 演示完成！")
        print("   - 实时诊断: 自动处理信号流并生成告警")
        print("   - 批量分析: 处理多个信号文件并生成报告")
        print("   - 报告生成: 创建HTML格式的专业诊断报告")
        print("   - 告警系统: 高置信度严重故障自动告警")

        print(f"\n📊 演示统计:")
        print(f"   模型: {args.model}")
        print(f"   准确率: {diagnostic_system.model_config['accuracy']}%")
        print(f"   可解释性: {diagnostic_system.model_config['explainability']}")
        print(f"   应用场景: {diagnostic_system.model_config['description']}")
        print(f"   支持信号: {', '.join(diagnostic_system.model_config.get('signal_types', ['Unknown']))}")

if __name__ == "__main__":
    main()