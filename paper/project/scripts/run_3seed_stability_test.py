#!/usr/bin/env python3
"""
Paper 1: 1D-2D Fusion Explainable - 3-Seed Stability Test
目标：验证Fusion1D2D模型在3个不同随机种子下的性能稳定性
"""

import os
import sys
import subprocess
import json
import time
import shlex
import csv
from datetime import datetime
import argparse
import yaml
from pathlib import Path

# 定义实验参数
SEEDS = [42, 123, 456]  # 三个稳定性测试种子
MODEL_NAME = "Fusion1D2D"
BASE_CONFIG = "paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml"
LOCAL_GPU_IDS = (0, 1)
GPU_ID = 0
REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN_ENTRYPOINT = REPO_ROOT / "main.py"
TEMPLATE_CONFIG = REPO_ROOT / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml"

def create_seed_config(seed, output_dir):
    """为每个种子创建专用配置文件"""
    with TEMPLATE_CONFIG.open('r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    config.setdefault('pipeline', 'Pipeline_01_default')
    config.setdefault('environment', {})
    config.setdefault('data', {})
    config.setdefault('task', {})
    config.setdefault('trainer', {})
    config.setdefault('model', {})
    config['base_configs'] = {
        'environment': str(REPO_ROOT / 'configs/base/environment/base.yaml'),
        'data': str(REPO_ROOT / 'configs/base/data/base_cross_domain.yaml'),
        'model': str(REPO_ROOT / 'configs/base/model/backbone_dlinear.yaml'),
        'task': str(REPO_ROOT / 'configs/base/task/dg.yaml'),
        'trainer': str(REPO_ROOT / 'configs/base/trainer/default_single_gpu.yaml'),
    }

    seed_run_dir = Path(output_dir) / f"seed_{seed}"
    config['environment'].update({
        'project': f'uxfd_1d2d_stability_seed_{seed}',
        'seed': seed,
        'output_dir': str(seed_run_dir),
        'notes': f'Fusion1D2D stability test for seed {seed}',
    })
    config['data'].update({
        'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
        'metadata_file': 'metadata.xlsx',
        'batch_size': 64,
        'num_workers': 8,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'normalization': 'standardization',
        'window_size': 4096,
        'stride': 5,
        'num_window': 64,
        'dtype': 'float32',
        'pin_memory': True,
    })
    config['task'].update({
        'name': 'classification',
        'type': 'DG',
        'target_system_id': [1],
        'target_domain_num': 1,
        'loss': 'CE',
        'metrics': ['acc'],
        'optimizer': 'adam',
        'batch_size': 64,
        'num_workers': 8,
        'pin_memory': True,
        'shuffle': True,
        'epochs': 3,
        'lr': 0.001,
        'weight_decay': 0.0001,
    })
    config['trainer'].update({
        'num_epochs': 3,
        'patience': 5,
        'gpus': 1,
        'device': 'cuda',
        'monitor': 'val_loss',
    })
    config['model'].update({
        'name': 'TSPN_UXFD',
        'type': 'X_model',
        'device': 'cuda',
        'scale': 1,
        'skip_connection': True,
        'in_channels': 2,
        'f_c_mu': 0,
        'f_c_sigma': 0.1,
        'f_b_mu': 0,
        'f_b_sigma': 0.1,
        'signal_processing_configs': {
            'layer1': ['I'],
        },
        'feature_extractor_configs': ['Mean', 'Std'],
        'uxfd': {
            'enable_sp2d': True,
            'sp2d': {
                'n_fft': 128,
                'hop_length': 64,
            },
            'fusion': {
                'type': 'gated',
            },
        },
        'out_channels': 4,
    })

    config_path = Path(output_dir) / f"config_Fusion1D2D_seed{seed}.yaml"
    with config_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    return str(config_path)

def run_experiment(config_path, seed, gpu_id):
    """运行单个实验"""
    print(f"\n{'='*50}")
    print(f"开始运行 Seed {seed} 实验...")
    print(f"配置文件: {config_path}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*50}")

    # 设置CUDA设备和conda环境
    inner = (
        "source ~/anaconda3/etc/profile.d/conda.sh && "
        "conda activate LQ_signal && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} python {shlex.quote(str(MAIN_ENTRYPOINT))} --config {shlex.quote(config_path)}"
    )
    cmd = f"bash -lc {shlex.quote(inner)}"

    print(f"执行命令: {cmd}")
    start_time = time.time()

    try:
        # 运行实验
        result = subprocess.run(cmd, shell=True, check=True,
                              capture_output=True, text=True,
                              cwd=os.getcwd())

        end_time = time.time()
        duration = end_time - start_time

        print(f"✅ Seed {seed} 实验完成！")
        print(f"⏱️  用时: {duration/60:.1f} 分钟")

        return True, duration, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"❌ Seed {seed} 实验失败！")
        print(f"错误信息: {e.stderr}")
        return False, 0, e.stderr

def extract_metrics_from_log(log_output, seed):
    """从日志输出中提取关键指标"""
    metrics = {
        'seed': seed,
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

def extract_metrics_from_result_artifacts(seed_root):
    """从结果产物中提取关键指标，优先读取 test_result_*.csv。"""
    seed_root = Path(seed_root)
    candidates = sorted(seed_root.rglob("test_result_*.csv"))
    if not candidates:
        candidates = sorted(seed_root.rglob("all_results.csv"))
    if not candidates:
        return None

    metrics = {}
    for csv_path in candidates:
        try:
            with csv_path.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                row = next(reader, None)
            if not row:
                continue
            for key in ("test_acc", "val_acc", "test_loss", "val_loss", "test_total_loss"):
                if key in row and row[key] not in ("", None):
                    try:
                        metrics[key] = float(row[key])
                    except ValueError:
                        pass
            for key in row:
                if key.startswith("test_acc_") and row[key] not in ("", None):
                    try:
                        metrics["test_acc"] = float(row[key])
                    except ValueError:
                        pass
                if key.startswith("test_loss_") and row[key] not in ("", None):
                    try:
                        metrics["test_loss"] = float(row[key])
                    except ValueError:
                        pass
                if key.startswith("val_acc_") and row[key] not in ("", None):
                    try:
                        metrics["val_acc"] = float(row[key])
                    except ValueError:
                        pass
                if key.startswith("val_loss_") and row[key] not in ("", None):
                    try:
                        metrics["val_loss"] = float(row[key])
                    except ValueError:
                        pass
            if metrics:
                metrics["source_csv"] = str(csv_path)
                return metrics
        except Exception:
            continue
    return None

def save_results(results, output_dir):
    """保存实验结果"""
    # 保存JSON格式结果
    results_file = os.path.join(output_dir, f"stability_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    successful_results = [result for result in results if result.get('status') == 'success']
    successful_accs = [
        result.get('metrics', {}).get('test_acc')
        for result in successful_results
        if result.get('metrics', {}).get('test_acc') is not None
    ]
    mean_acc = None
    std_acc = None
    ci95_acc = None
    cv_percent = None
    if successful_accs:
        import statistics
        mean_acc = statistics.mean(successful_accs)
        std_acc = statistics.stdev(successful_accs) if len(successful_accs) > 1 else 0.0
        ci95_acc = 1.96 * std_acc / (len(successful_accs) ** 0.5) if len(successful_accs) > 1 else 0.0
        cv_percent = (std_acc / mean_acc * 100.0) if mean_acc else None
    metrics_summary = {
        'requested_seeds': SEEDS,
        'success_count': len(successful_results),
        'failed_count': len(results) - len(successful_results),
        'successful_seeds': [result['seed'] for result in successful_results],
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'ci95_accuracy': ci95_acc,
        'cv_percent': cv_percent,
        'results_file': results_file,
    }
    metrics_summary_file = os.path.join(output_dir, "stability_metrics_summary.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # 创建结果摘要
    summary_file = os.path.join(output_dir, "stability_test_summary.md")

    with open(summary_file, 'w') as f:
        f.write("# Paper 1: Fusion1D2D 稳定性测试报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {MODEL_NAME}\n")
        f.write(f"**测试种子**: {SEEDS}\n\n")

        f.write("## 实验结果汇总\n\n")
        f.write("| Seed | 测试准确率 | 验证准确率 | 测试损失 | 验证损失 | 状态 |\n")
        f.write("|------|------------|------------|----------|----------|------|\n")

        successful_results = []
        total_time = 0

        for result in results:
            seed = result['seed']
            metrics = result['metrics']
            duration = result['duration']
            status = result['status']

            total_time += duration

            if status == 'success':
                test_acc = f"{metrics['test_acc']*100:.2f}%" if metrics['test_acc'] else "N/A"
                val_acc = f"{metrics['val_acc']*100:.2f}%" if metrics['val_acc'] else "N/A"
                test_loss = f"{metrics['test_loss']:.4f}" if metrics['test_loss'] else "N/A"
                val_loss = f"{metrics['val_loss']:.4f}" if metrics['val_loss'] else "N/A"

                if metrics['test_acc']:
                    successful_results.append(metrics['test_acc'])
            else:
                test_acc = val_acc = test_loss = val_loss = "FAILED"

            f.write(f"| {seed} | {test_acc} | {val_acc} | {test_loss} | {val_loss} | {'✅' if status == 'success' else '❌'} |\n")

        f.write(f"\n## 统计分析\n\n")
        if successful_results:
            import statistics
            mean_acc = statistics.mean(successful_results)
            std_acc = statistics.stdev(successful_results) if len(successful_results) > 1 else 0

            f.write(f"- **平均测试准确率**: {mean_acc*100:.2f}%\n")
            f.write(f"- **标准差**: {std_acc*100:.2f}%\n")
            f.write(f"- **最高准确率**: {max(successful_results)*100:.2f}%\n")
            f.write(f"- **最低准确率**: {min(successful_results)*100:.2f}%\n")
            f.write(f"- **成功率**: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)\n")

            # 稳定性评估
            if std_acc < 0.005:  # 0.5%
                stability = "🟢 优秀"
            elif std_acc < 0.01:  # 1%
                stability = "🟡 良好"
            else:
                stability = "🔴 需要改进"

            f.write(f"- **稳定性评估**: {stability}\n")
        else:
            f.write("- ❌ 所有实验均失败\n")

        f.write(f"\n## 实验信息\n\n")
        f.write(f"- **总用时**: {total_time/60:.1f} 分钟\n")
        f.write(f"- **平均用时**: {total_time/len(results)/60:.1f} 分钟/实验\n")
        f.write(f"- **配置文件**: {BASE_CONFIG}\n")

    print(f"\n📊 结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   摘要报告: {summary_file}")
    print(f"   机器摘要: {metrics_summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Paper 1: Fusion1D2D 3-Seed Stability Test')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='paper/UXFD_paper/1D-2D_fusion_explainable/experiments/stability_test',
    )
    parser.add_argument('--gpu_id', type=int, default=GPU_ID, choices=LOCAL_GPU_IDS)

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Paper 1: {MODEL_NAME} 3-Seed 稳定性测试开始")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试种子: {SEEDS}")
    print(f"💾 输出目录: {args.output_dir}")

    results = []

    # 运行所有种子实验
    for i, seed in enumerate(SEEDS):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(SEEDS)}")
        print(f"当前种子: {seed}")
        print(f"{'='*60}")

        # 创建种子专用配置
        config_path = create_seed_config(seed, args.output_dir)

        # 运行实验
        success, duration, log_output = run_experiment(config_path, seed, args.gpu_id)

        # 提取指标
        if success:
            metrics = extract_metrics_from_result_artifacts(Path(args.output_dir) / f"seed_{seed}")
            if metrics is None:
                metrics = extract_metrics_from_log(log_output, seed)
        else:
            metrics = {'seed': seed}

        # 保存结果
        results.append({
            'seed': seed,
            'config_path': config_path,
            'status': 'success' if success else 'failed',
            'duration': duration,
            'metrics': metrics,
            'artifact_metrics_found': success and metrics.get("source_csv") is not None,
            'log_file': os.path.join(args.output_dir, f"seed_{seed}_log.txt")
        })

        # 保存日志
        with open(results[-1]['log_file'], 'w') as f:
            f.write(f"Seed {seed} 实验日志\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"配置: {config_path}\n")
            f.write(f"GPU: {args.gpu_id}\n")
            f.write(f"状态: {'成功' if success else '失败'}\n")
            f.write(f"用时: {duration:.1f} 秒\n")
            f.write("-" * 50 + "\n")
            f.write(log_output)

    # 保存所有结果
    save_results(results, args.output_dir)

    print(f"\n🎉 稳定性测试完成！")
    print(f"📊 结果摘要:")

    successful_count = sum(1 for r in results if r['status'] == 'success')
    print(f"   ✅ 成功: {successful_count}/{len(results)}")

    if successful_count > 0:
        successful_accs = [r['metrics']['test_acc'] for r in results
                          if r['status'] == 'success' and r['metrics']['test_acc']]
        if successful_accs:
            import statistics
            mean_acc = statistics.mean(successful_accs)
            std_acc = statistics.stdev(successful_accs) if len(successful_accs) > 1 else 0
            print(f"   📈 平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")

if __name__ == "__main__":
    main()
