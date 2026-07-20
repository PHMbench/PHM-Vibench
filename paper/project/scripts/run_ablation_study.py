#!/usr/bin/env python3
"""
Paper 1: 1D-2D Fusion Explainable - Ablation Study
目标：通过消融实验验证1D、2D、统计特征各组件的贡献度
"""

import os
import sys
import subprocess
import json
import time
import shlex
from datetime import datetime
import argparse
import yaml
from pathlib import Path

# 定义消融实验配置
ABLATION_CONFIGS = {
    'Full_Fusion': {
        'description': '完整1D+2D+统计特征融合',
        'signal_layers': ['HT','WF','I'],
        'feature_extractors': ['Mean', 'Std', 'Var', 'Entropy','Max', 'Min', 'AbsMean', 'Kurtosis', 'RMS', 'CrestFactor','Skewness', 'ClearanceFactor', 'ShapeFactor'],
        'use_1d': True,
        'use_2d': True,
        'use_statistical': True
    },
    '1D_only': {
        'description': '仅1D信号处理',
        'signal_layers': ['HT','WF','I'],
        'feature_extractors': [],  # 不使用统计特征
        'use_1d': True,
        'use_2d': False,
        'use_statistical': False
    },
    '2D_only': {
        'description': '仅2D频谱分析',
        'signal_layers': ['FFT','I','I'],  # 主要使用FFT生成2D频谱
        'feature_extractors': [],  # 不使用统计特征
        'use_1d': False,
        'use_2d': True,
        'use_statistical': False
    },
    'No_Statistical': {
        'description': '1D+2D融合但无统计特征',
        'signal_layers': ['HT','WF','I'],
        'feature_extractors': [],  # 不使用统计特征
        'use_1d': True,
        'use_2d': True,
        'use_statistical': False
    },
    'Minimal_1D': {
        'description': '最简1D（仅基础处理）',
        'signal_layers': ['I','I','I'],  # 仅恒等变换
        'feature_extractors': ['Mean', 'Std'],  # 仅基础统计特征
        'use_1d': True,
        'use_2d': False,
        'use_statistical': True
    }
}

MODEL_NAME = "Fusion1D2D"
BASE_SEED = 42
LOCAL_GPU_IDS = (0, 1)
GPU_ID = 0
REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN_ENTRYPOINT = REPO_ROOT / "main.py"
TEMPLATE_CONFIG = (
    REPO_ROOT
    / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml"
)

def create_ablation_config(config_name, config_info, output_dir):
    """为每个消融实验创建专用配置文件"""

    # 读取当前 PHM-Vibench 最小配置，避免依赖旧 unified_baseline 路径。
    with TEMPLATE_CONFIG.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    config.setdefault('pipeline', 'Pipeline_01_default')
    config.setdefault('environment', {})
    config.setdefault('data', {})
    config.setdefault('task', {})
    config.setdefault('trainer', {})
    config.setdefault('model', {})

    # 修改消融实验参数
    config['environment'].update({
        'project': f'uxfd_1d2d_ablation_{config_name}',
        'seed': BASE_SEED,
        'output_dir': str(Path(output_dir) / config_name),
        'notes': f"Fusion1D2D ablation - {config_name}: {config_info['description']}",
    })
    config['trainer'].update({
        'device': 'cuda',
        'gpus': 1,
        'paper_id': '1D-2D_fusion_explainable',
        'preset_version': f'paper02-ablation-{config_name}',
    })
    config['model'].update({
        'device': 'cuda',
        'signal_processing_configs': {
            'layer1': config_info['signal_layers'],
        },
        'feature_extractor_configs': config_info['feature_extractors'],
    })
    config['model'].setdefault('signal_processing_2d', {})
    config['model']['signal_processing_2d']['enable'] = bool(config_info['use_2d'])

    # The current NSN proxy has no separate 1D-branch switch; record the intended
    # component switches as metadata for the accepted-run collector.
    config['model']['ablation_flags'] = {
        'use_1d': config_info['use_1d'],
        'use_2d': config_info['use_2d'],
        'use_statistical': config_info['use_statistical'],
    }

    config_path = f"{output_dir}/config_Fusion1D2D_ablation_{config_name}.yaml"

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    return config_path

def run_ablation_experiment(config_name, config_path, gpu_id):
    """运行单个消融实验"""
    print(f"\n{'='*50}")
    print(f"开始运行消融实验: {config_name}")
    print(f"描述: {ABLATION_CONFIGS[config_name]['description']}")
    print(f"配置文件: {config_path}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*50}")

    # 设置CUDA设备和conda环境
    inner = (
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate LQ_signal && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} "
        f"python {shlex.quote(str(MAIN_ENTRYPOINT))} --config {shlex.quote(config_path)}"
    )
    cmd = f"bash -lc {shlex.quote(inner)}"

    print(f"执行命令: {cmd}")
    start_time = time.time()

    try:
        # 运行实验
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True,
                              cwd=str(REPO_ROOT))

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ {config_name} 消融实验完成！")
        print(f"⏱️  用时: {duration/60:.1f} 分钟")

        return True, duration, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"❌ {config_name} 消融实验失败！")
        print(f"错误信息: {e.stderr}")
        return False, 0, e.stderr

def extract_metrics_from_log(log_output, config_name):
    """从日志输出中提取关键指标"""
    metrics = {
        'config': config_name,
        'description': ABLATION_CONFIGS[config_name]['description'],
        'components': {
            'use_1d': ABLATION_CONFIGS[config_name]['use_1d'],
            'use_2d': ABLATION_CONFIGS[config_name]['use_2d'],
            'use_statistical': ABLATION_CONFIGS[config_name]['use_statistical']
        },
        'test_acc': None,
        'val_acc': None,
        'test_loss': None,
        'val_loss': None,
        'best_epoch': None
    }

    # 解析日志输出寻找测试准确率
    lines = log_output.split('\n')
    for line in lines:
        if 'test_acc' in line and ':' in line:
            try:
                # 寻找类似 "test_acc': 0.9957" 的模式
                if 'test_acc' in line:
                    # 提取数字
                    import re
                    numbers = re.findall(r'\d+\.?\d*', line)
                    if numbers:
                        metrics['test_acc'] = float(numbers[-1])
            except:
                pass

        if 'val_acc' in line and ':' in line:
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    metrics['val_acc'] = float(numbers[-1])
            except:
                pass

        if 'test_loss' in line and ':' in line:
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    metrics['test_loss'] = float(numbers[-1])
            except:
                pass

        if 'val_loss' in line and ':' in line:
            try:
                import re
                numbers = re.findall(r'\d+\.?\d*', line)
                if numbers:
                    metrics['val_loss'] = float(numbers[-1])
            except:
                pass

    return metrics

def analyze_component_contributions(results):
    """分析各组件贡献度"""
    contributions = {}

    # 获取Full Fusion作为基准
    baseline_result = None
    for result in results:
        if result['config'] == 'Full_Fusion' and result['metrics']['test_acc']:
            baseline_result = result
            break

    if not baseline_result:
        return None

    baseline_acc = baseline_result['metrics']['test_acc']

    # 分析每个组件的贡献
    contributions['baseline'] = {
        'config': 'Full_Fusion',
        'accuracy': baseline_acc,
        'description': '完整系统性能'
    }

    # 1D贡献度
    for result in results:
        if (result['config'] == '1D_only' and
            result['metrics']['test_acc']):
            acc_1d = result['metrics']['test_acc']
            contributions['1d_contribution'] = {
                'config': '1D_only',
                'accuracy': acc_1d,
                'contribution': acc_1d / baseline_acc,
                'description': '1D信号处理贡献度'
            }
            break

    # 2D贡献度
    for result in results:
        if (result['config'] == '2D_only' and
            result['metrics']['test_acc']):
            acc_2d = result['metrics']['test_acc']
            contributions['2d_contribution'] = {
                'config': '2D_only',
                'accuracy': acc_2d,
                'contribution': acc_2d / baseline_acc,
                'description': '2D频谱分析贡献度'
            }
            break

    # 统计特征贡献度
    for result in results:
        if (result['config'] == 'No_Statistical' and
            result['metrics']['test_acc']):
            acc_no_stat = result['metrics']['test_acc']
            stat_contribution = (baseline_acc - acc_no_stat) / baseline_acc
            contributions['statistical_contribution'] = {
                'config': 'No_Statistical',
                'accuracy': acc_no_stat,
                'contribution': stat_contribution,
                'description': '统计特征贡献度'
            }
            break

    return contributions

def save_ablation_results(results, contributions, output_dir):
    """保存消融实验结果"""
    # 保存JSON格式结果
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'model': MODEL_NAME,
        'seed': BASE_SEED,
        'ablation_results': results,
        'component_contributions': contributions
    }

    results_file = os.path.join(output_dir, f"ablation_study_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # 创建结果摘要
    summary_file = os.path.join(output_dir, "ablation_study_summary.md")

    with open(summary_file, 'w') as f:
        f.write("# Paper 1: Fusion1D2D 消融实验报告\n\n")
        f.write(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {MODEL_NAME}\n")
        f.write(f"**基准种子**: {BASE_SEED}\n\n")

        f.write("## 消融实验结果\n\n")
        f.write("| 配置 | 描述 | 1D | 2D | 统计特征 | 测试准确率 | 验证准确率 | 测试损失 | 相对性能 |\n")
        f.write("|------|------|----|----|----------|------------|------------|----------|----------|\n")

        baseline_acc = None
        total_time = 0

        # 首先找到基线准确率
        for result in results:
            if result['config'] == 'Full_Fusion' and result['metrics']['test_acc']:
                baseline_acc = result['metrics']['test_acc']
                break

        for result in results:
            config_name = result['config']
            config_info = ABLATION_CONFIGS[config_name]
            metrics = result['metrics']
            duration = result['duration']
            status = result['status']

            total_time += duration

            if status == 'success':
                test_acc = f"{metrics['test_acc']*100:.2f}%" if metrics['test_acc'] else "N/A"
                val_acc = f"{metrics['val_acc']*100:.2f}%" if metrics['val_acc'] else "N/A"
                test_loss = f"{metrics['test_loss']:.4f}" if metrics['test_loss'] else "N/A"

                # 计算相对性能
                if metrics['test_acc'] and baseline_acc:
                    relative_perf = f"{metrics['test_acc']/baseline_acc:.3f}"
                else:
                    relative_perf = "N/A"
            else:
                test_acc = val_acc = test_loss = relative_perf = "FAILED"

            f.write(f"| {config_name} | {config_info['description']} | "
                   f"{'✅' if config_info['use_1d'] else '❌'} | "
                   f"{'✅' if config_info['use_2d'] else '❌'} | "
                   f"{'✅' if config_info['use_statistical'] else '❌'} | "
                   f"{test_acc} | {val_acc} | {test_loss} | {relative_perf} |\n")

        f.write(f"\n## 组件贡献度分析\n\n")

        if contributions:
            f.write("### 关键发现\n\n")

            for key, contrib in contributions.items():
                if key == 'baseline':
                    f.write(f"- **基准性能**: {contrib['accuracy']*100:.2f}% ({contrib['description']})\n")
                elif 'contribution' in contrib['description']:
                    if '1d' in key or '2d' in key:
                        f.write(f"- **{contrib['description']}**: {contrib['accuracy']*100:.2f}% (相对基准: {contrib['contribution']:.3f})\n")
                    else:
                        f.write(f"- **{contrib['description']}**: {(contrib['contribution']*100):.1f}%\n")

        f.write(f"\n## 实验信息\n\n")
        f.write(f"- **总用时**: {total_time/60:.1f} 分钟\n")
        f.write(f"- **平均用时**: {total_time/len(results)/60:.1f} 分钟/实验\n")
        f.write(f"- **成功率**: {sum(1 for r in results if r['status'] == 'success')}/{len(results)} ({sum(1 for r in results if r['status'] == 'success')/len(results)*100:.1f}%)\n")

    print(f"\n📊 消融实验结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   摘要报告: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Paper 1: Fusion1D2D Ablation Study')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='paper/UXFD_paper/1D-2D_fusion_explainable/experiments/ablation_study',
    )
    parser.add_argument('--gpu_id', type=int, default=GPU_ID, choices=LOCAL_GPU_IDS)
    parser.add_argument('--configs', nargs='+', default=list(ABLATION_CONFIGS.keys()),
                       help='Ablation configs to test (default: all)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate configs and print launch commands without running experiments')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 过滤指定的配置
    test_configs = {k: v for k, v in ABLATION_CONFIGS.items() if k in args.configs}

    print(f"🚀 Paper 1: {MODEL_NAME} 消融实验开始")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试配置: {list(test_configs.keys())}")
    print(f"💾 输出目录: {args.output_dir}")

    results = []
    dry_run_results = []

    # 运行所有消融实验
    for i, (config_name, config_info) in enumerate(test_configs.items()):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(test_configs)}")
        print(f"当前配置: {config_name}")
        print(f"描述: {config_info['description']}")
        print(f"组件: 1D={config_info['use_1d']}, 2D={config_info['use_2d']}, 统计={config_info['use_statistical']}")
        print(f"{'='*60}")

        # 创建消融实验配置
        config_path = create_ablation_config(config_name, config_info, args.output_dir)

        if args.dry_run:
            dry_run_results.append({
                'config': config_name,
                'config_path': config_path,
                'gpu_id': args.gpu_id,
                'entrypoint': str(MAIN_ENTRYPOINT),
                'repo_root': str(REPO_ROOT),
            })
            continue

        # 运行实验
        success, duration, log_output = run_ablation_experiment(config_name, config_path, args.gpu_id)

        # 提取指标
        if success:
            metrics = extract_metrics_from_log(log_output, config_name)
        else:
            metrics = {'config': config_name, 'description': config_info['description']}

        # 保存结果
        results.append({
            'config': config_name,
            'config_path': config_path,
            'status': 'success' if success else 'failed',
            'duration': duration,
            'metrics': metrics,
            'log_file': os.path.join(args.output_dir, f"ablation_{config_name}_log.txt")
        })

        # 保存日志
        with open(results[-1]['log_file'], 'w') as f:
            f.write(f"{config_name} 消融实验日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置: {config_path}\n")
            f.write(f"GPU: {args.gpu_id}\n")
            f.write(f"状态: {'成功' if success else '失败'}\n")
            f.write(f"用时: {duration:.1f} 秒\n")
            f.write("-" * 50 + "\n")
            f.write(log_output)

    if args.dry_run:
        print(json.dumps({
            'dry_run': True,
            'allowed_gpu_ids': list(LOCAL_GPU_IDS),
            'configs': dry_run_results,
        }, ensure_ascii=False, indent=2))
        return

    # 分析组件贡献度
    contributions = analyze_component_contributions(results)

    # 保存所有结果
    save_ablation_results(results, contributions, args.output_dir)

    print(f"\n🎉 消融实验完成！")
    print(f"📊 结果摘要:")

    successful_count = sum(1 for r in results if r['status'] == 'success')
    print(f"   ✅ 成功: {successful_count}/{len(results)}")

    if contributions and 'baseline' in contributions:
        baseline_acc = contributions['baseline']['accuracy']
        print(f"   📈 基准性能: {baseline_acc*100:.2f}%")

        if '1d_contribution' in contributions:
            acc_1d = contributions['1d_contribution']['accuracy']
            print(f"   📊 1D贡献: {acc_1d*100:.2f}% ({acc_1d/baseline_acc:.3f})")

        if '2d_contribution' in contributions:
            acc_2d = contributions['2d_contribution']['accuracy']
            print(f"   📊 2D贡献: {acc_2d*100:.2f}% ({acc_2d/baseline_acc:.3f})")

if __name__ == "__main__":
    main()
