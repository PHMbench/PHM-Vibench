#!/usr/bin/env python3
"""
Paper 1: 1D-2D Fusion Explainable - Multi-Dataset Validation
目标：验证Fusion1D2D模型在不同数据集上的泛化性能
"""

import os
import sys
import subprocess
import json
import time
import copy
import shlex
import csv
from datetime import datetime
import argparse
import yaml
from pathlib import Path

# 定义实验参数
DATASETS = {
    'CWRU': {
        'dataset_id': 1,
        'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
        'metadata_file': 'metadata.xlsx',
        'target_domain_num': 1,
        'expected_acc': 0.90,  # 期望准确率90%+
        'description': '凯斯西储大学轴承数据集'
    },
    'XJTU': {
        'dataset_id': 2,
        'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
        'metadata_file': 'metadata.xlsx',
        'target_domain_num': 1,
        'expected_acc': 0.85,  # 期望准确率85%+
        'description': '西安交通大学轴承数据集'
    },
    'THU_006': {
        'dataset_id': 6,
        'data_dir': '/home/user/data/PHMbenchdata/PHM-Vibench',
        'metadata_file': 'metadata.xlsx',
        'target_domain_num': 1,
        'expected_acc': 0.95,  # 期望准确率95%+
        'description': '清华大学006实验台数据集'
    }
}

MODEL_NAME = "Fusion1D2D"
BASE_SEED = 42
LOCAL_GPU_IDS = (0, 1)
GPU_ID = 0
REPO_ROOT = Path(__file__).resolve().parents[4]
MAIN_ENTRYPOINT = REPO_ROOT / "main.py"
TEMPLATE_CONFIG = REPO_ROOT / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml"
PAPER_CONFIGS = {
    'CWRU': REPO_ROOT / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_CWRU.yaml",
    'XJTU': REPO_ROOT / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_XJTU.yaml",
    'THU_006': REPO_ROOT / "paper/UXFD_paper/1D-2D_fusion_explainable/configs/config_THU_006.yaml",
}
RUNTIME_MODEL_ALIASES = {
    # The paper names the method Fusion1D2D, while the maintained executable entrypoint
    # is the NSN wrapper over TSPN_UXFD in the unified runtime.
    'Fusion1D2D': 'NSN',
}


def _load_yaml(path):
    with Path(path).open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _paper_profile(dataset_name):
    profile_path = PAPER_CONFIGS[dataset_name]
    profile = _load_yaml(profile_path)
    if not profile:
        raise ValueError(f"Empty paper profile: {profile_path}")
    return profile


def _resolve_protocol(requested_protocol, datasets, required_test_acc):
    if requested_protocol != "auto":
        return requested_protocol
    # The strengthened 98% gate is intended to be in-domain, not cross-domain DG.
    if required_test_acc >= 0.98 and len(datasets) == 1:
        return "id"
    return "dg"


def create_dataset_config(dataset_name, dataset_info, output_dir, protocol):
    """为每个数据集创建专用配置文件"""
    config = _load_yaml(TEMPLATE_CONFIG)
    profile = _paper_profile(dataset_name)
    profile_args = profile.get('args') or {}
    vbench_config = profile_args.get('vbench_config') or {}

    dataset_run_dir = Path(output_dir) / dataset_name
    batch_size = int(profile_args.get('batch_size', 64))
    num_workers = int(profile_args.get('num_workers', 8))
    window_size = int(profile_args.get('in_dim', 4096))
    num_epochs = int(profile_args.get('num_epochs', 50))
    patience = int(profile_args.get('patience', 15))
    learning_rate = float(profile_args.get('learning_rate', 0.001))
    weight_decay = float(profile_args.get('weight_decay', 0.0001))
    data_dir = str(vbench_config.get('data_dir') or profile_args.get('data_dir') or dataset_info['data_dir']).rstrip('/')
    metadata_file = vbench_config.get('metadata_file') or dataset_info['metadata_file']
    target_system_id = vbench_config.get('dataset_ids') or [dataset_info['dataset_id']]
    signal_processing_configs = copy.deepcopy(profile.get('signal_processing_configs') or {})
    feature_extractor_configs = copy.deepcopy(profile.get('feature_extractor_configs') or [])
    paper_model_name = str(profile_args.get('model', MODEL_NAME))
    runtime_model_name = RUNTIME_MODEL_ALIASES.get(paper_model_name, paper_model_name)

    protocol = str(protocol)
    is_id_protocol = protocol == "id"

    config.setdefault('pipeline', 'Pipeline_01_default')
    config.setdefault('environment', {})
    config.setdefault('data', {})
    config.setdefault('task', {})
    config.setdefault('trainer', {})
    config.setdefault('model', {})
    base_configs = {
        'environment': str(REPO_ROOT / 'configs/base/environment/base.yaml'),
        'data': str(
            REPO_ROOT / (
                'configs/base/data/base_classification.yaml'
                if is_id_protocol else
                'configs/base/data/base_cross_domain.yaml'
            )
        ),
        'model': str(REPO_ROOT / 'configs/base/model/backbone_dlinear.yaml'),
        'trainer': str(REPO_ROOT / 'configs/base/trainer/default_single_gpu.yaml'),
    }
    if not is_id_protocol:
        base_configs['task'] = str(REPO_ROOT / 'configs/base/task/dg.yaml')
    config['base_configs'] = base_configs
    config['pipeline'] = 'Pipeline_ID' if is_id_protocol else 'Pipeline_01_default'

    config['environment'].update({
        'project': f'uxfd_1d2d_{dataset_name.lower()}_multi_dataset_validation_{protocol}',
        'seed': BASE_SEED,
        'output_dir': str(dataset_run_dir / (f"{dataset_name}_ID" if is_id_protocol else "")) if is_id_protocol else str(dataset_run_dir),
        'notes': (
            f'Fusion1D2D validation on {dataset_name} using paper profile overlay '
            f'(runtime_model={runtime_model_name}, paper_model={paper_model_name}, protocol={protocol})'
        ),
    })

    config['data'].update({
        'data_dir': data_dir,
        'metadata_file': metadata_file,
        'batch_size': batch_size,
        'num_workers': 0 if is_id_protocol else num_workers,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'normalization': 'standardization',
        'window_size': window_size,
        'stride': 5,
        'num_window': 64,
        'dtype': 'float32',
        'pin_memory': False if is_id_protocol else True,
    })
    if is_id_protocol:
        config['data']['factory_name'] = 'id'

    config['task'].update({
        'name': 'ID_task' if is_id_protocol else 'classification',
        'type': 'Default_task' if is_id_protocol else 'DG',
        'target_system_id': target_system_id,
        'loss': 'CE',
        'metrics': ['acc'],
        'optimizer': 'adam',
        'batch_size': batch_size,
        'num_workers': 0 if is_id_protocol else num_workers,
        'pin_memory': False if is_id_protocol else True,
        'shuffle': True,
        'epochs': num_epochs,
        'lr': learning_rate,
        'weight_decay': weight_decay,
    })
    if not is_id_protocol:
        config['task']['target_domain_num'] = dataset_info['target_domain_num']

    config['trainer'].update({
        'num_epochs': num_epochs,
        'patience': patience,
        'gpus': 1,
        'device': str(profile_args.get('device', 'cuda')),
        'monitor': str(profile_args.get('monitor', 'val_loss')),
        'paper_id': '1D-2D_fusion_explainable',
        'preset_version': f'paper-profile-v2-{protocol}',
    })

    config['model'].update({
        'name': runtime_model_name,
        'paper_alias': paper_model_name,
        'type': 'X_model',
        'device': str(profile_args.get('device', 'cuda')),
        'in_dim': int(profile_args.get('in_dim', window_size)),
        'out_dim': int(profile_args.get('out_dim', window_size)),
        'scale': int(profile_args.get('scale', 4)),
        'skip_connection': bool(profile_args.get('skip_connection', True)),
        'in_channels': int(profile_args.get('in_channels', 2)),
        'out_channels': int(profile_args.get('out_channels', 3)),
        'f_c_mu': float(profile_args.get('f_c_mu', 0)),
        'f_c_sigma': float(profile_args.get('f_c_sigma', 0.1)),
        'f_b_mu': float(profile_args.get('f_b_mu', 0)),
        'f_b_sigma': float(profile_args.get('f_b_sigma', 0.1)),
        'signal_processing_configs': signal_processing_configs,
        'feature_extractor_configs': feature_extractor_configs,
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
        'signal_processing_2d': {
            'enable': True,
            'stft': {
                'n_fft': 128,
                'hop_length': 64,
            },
            'fusion': {
                'type': 'gated',
            },
        },
    })

    config_path = Path(output_dir) / f"config_Fusion1D2D_{dataset_name}{'_id' if is_id_protocol else ''}.yaml"
    with config_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    return str(config_path)

def run_dataset_experiment(dataset_name, config_path, gpu_id):
    """运行单个数据集实验"""
    print(f"\n{'='*50}")
    print(f"开始运行 {dataset_name} 数据集实验...")
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

        print(f"✅ {dataset_name} 数据集实验完成！")
        print(f"⏱️  用时: {duration/60:.1f} 分钟")

        return True, duration, result.stdout

    except subprocess.CalledProcessError as e:
        print(f"❌ {dataset_name} 数据集实验失败！")
        print(f"错误信息: {e.stderr}")
        return False, 0, e.stderr

def extract_metrics_from_log(log_output, dataset_name):
    """从日志输出中提取关键指标"""
    metrics = {
        'dataset': dataset_name,
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

def extract_metrics_from_result_artifacts(dataset_root):
    """从结果产物中提取关键指标，优先读取 test_result_*.csv。"""
    dataset_root = Path(dataset_root)
    candidates = sorted(dataset_root.rglob("test_result_*.csv"))
    if not candidates:
        candidates = sorted(dataset_root.rglob("all_results.csv"))
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

def save_dataset_results(results, output_dir, required_test_acc):
    """保存多数据集验证结果"""
    # 保存JSON格式结果
    results_file = os.path.join(output_dir, f"multi_dataset_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    successful_results = [result for result in results if result.get('status') == 'success']
    successful_datasets = [result['dataset'] for result in successful_results]
    successful_accs = [
        result.get('metrics', {}).get('test_acc')
        for result in successful_results
        if result.get('metrics', {}).get('test_acc') is not None
    ]
    threshold_pass_datasets = [
        result['dataset']
        for result in successful_results
        if result.get('metrics', {}).get('test_acc') is not None
        and result.get('metrics', {}).get('test_acc') >= required_test_acc
    ]
    threshold_failed_datasets = [
        result['dataset']
        for result in results
        if result.get('status') != 'success'
        or result.get('metrics', {}).get('test_acc') is None
        or result.get('metrics', {}).get('test_acc') < required_test_acc
    ]
    mean_test_acc = sum(successful_accs) / len(successful_accs) if successful_accs else None
    generalization_gap = (max(successful_accs) - min(successful_accs)) if len(successful_accs) >= 2 else None
    metrics_summary = {
        'requested_datasets': [result['dataset'] for result in results],
        'success_count': len(successful_results),
        'failed_count': len(results) - len(successful_results),
        'successful_datasets': successful_datasets,
        'failed_datasets': [result['dataset'] for result in results if result.get('status') != 'success'],
        'mean_test_acc': mean_test_acc,
        'generalization_gap': generalization_gap,
        'required_test_acc': required_test_acc,
        'threshold_pass_datasets': threshold_pass_datasets,
        'threshold_failed_datasets': threshold_failed_datasets,
        'threshold_pass': len(threshold_failed_datasets) == 0 and bool(successful_results),
        'results_file': results_file,
    }
    metrics_summary_file = os.path.join(output_dir, "multi_dataset_validation_metrics_summary.json")
    with open(metrics_summary_file, 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    # 创建结果摘要
    summary_file = os.path.join(output_dir, "multi_dataset_validation_summary.md")

    with open(summary_file, 'w') as f:
        f.write("# Paper 1: Fusion1D2D 多数据集验证报告\n\n")
        f.write(f"**测试时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**模型**: {MODEL_NAME}\n")
        f.write(f"**基准种子**: {BASE_SEED}\n\n")
        f.write(f"**阈值门**: test_acc >= {required_test_acc:.2%}\n\n")

        f.write("## 数据集验证结果\n\n")
        f.write("| 数据集 | 描述 | 测试准确率 | 期望准确率 | 验证准确率 | 测试损失 | 状态 |\n")
        f.write("|--------|------|------------|------------|------------|----------|------|\n")

        successful_results = []
        total_time = 0

        for result in results:
            dataset = result['dataset']
            dataset_info = DATASETS[dataset]
            metrics = result['metrics']
            duration = result['duration']
            status = result['status']

            total_time += duration

            if status == 'success':
                test_acc = f"{metrics['test_acc']*100:.2f}%" if metrics['test_acc'] else "N/A"
                val_acc = f"{metrics['val_acc']*100:.2f}%" if metrics['val_acc'] else "N/A"
                test_loss = f"{metrics['test_loss']:.4f}" if metrics['test_loss'] else "N/A"

                # 判断是否达到期望
                expected = f"{required_test_acc*100:.0f}%"
                if metrics['test_acc']:
                    meet_expectation = "✅" if metrics['test_acc'] >= required_test_acc else "⚠️"
                    if metrics['test_acc']:
                        successful_results.append((dataset, metrics['test_acc']))
                else:
                    meet_expectation = "❌"
            else:
                test_acc = val_acc = test_loss = "FAILED"
                expected = f"{required_test_acc*100:.0f}%"
                meet_expectation = "❌"

            f.write(f"| {dataset} | {dataset_info['description']} | {test_acc} | {expected} | {val_acc} | {test_loss} | {meet_expectation if status == 'success' else '❌'} |\n")

        f.write(f"\n## 性能分析\n\n")

        if successful_results:
            # 计算平均性能
            accs = [acc for _, acc in successful_results]
            mean_acc = sum(accs) / len(accs)

            f.write(f"- **平均测试准确率**: {mean_acc*100:.2f}%\n")
            f.write(f"- **最佳性能数据集**: {max(successful_results, key=lambda x: x[1])[0]} ({max(accs)*100:.2f}%)\n")
            f.write(f"- **最差性能数据集**: {min(successful_results, key=lambda x: x[1])[0]} ({min(accs)*100:.2f}%)\n")
            f.write(f"- **成功率**: {len(successful_results)}/{len(results)} ({len(successful_results)/len(results)*100:.1f}%)\n")

            # 泛化能力评估
            performance_variance = max(accs) - min(accs)
            if performance_variance < 0.05:  # 5%
                generalization = "🟢 优秀"
            elif performance_variance < 0.10:  # 10%
                generalization = "🟡 良好"
            else:
                generalization = "🔴 需要改进"

            f.write(f"- **泛化能力评估**: {generalization} (差异: {performance_variance*100:.2f}%)\n")
        else:
            f.write("- ❌ 所有数据集验证均失败\n")

        f.write(f"\n## 实验信息\n\n")
        f.write(f"- **总用时**: {total_time/60:.1f} 分钟\n")
        f.write(f"- **平均用时**: {total_time/len(results)/60:.1f} 分钟/数据集\n")
        f.write(f"- **测试数据集**: {list(DATASETS.keys())}\n")

    print(f"\n📊 多数据集验证结果已保存:")
    print(f"   详细结果: {results_file}")
    print(f"   摘要报告: {summary_file}")
    print(f"   机器摘要: {metrics_summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Paper 1: Fusion1D2D Multi-Dataset Validation')
    parser.add_argument(
        '--output_dir',
        type=str,
        default='paper/UXFD_paper/1D-2D_fusion_explainable/experiments/multi_dataset',
    )
    parser.add_argument('--gpu_id', type=int, default=GPU_ID, choices=LOCAL_GPU_IDS)
    parser.add_argument('--datasets', nargs='+', default=list(DATASETS.keys()),
                       help='Datasets to validate (default: all)')
    parser.add_argument('--required-test-acc', type=float, default=0.98,
                       help='Machine gate for in-domain acceptance')
    parser.add_argument('--protocol', choices=['auto', 'dg', 'id'], default='auto',
                       help='Execution protocol. auto uses ID for single-dataset >=0.98 tickets.')
    parser.add_argument('--dry-run', action='store_true',
                       help='Generate configs and print the resolved protocol without launching experiments')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 过滤指定的数据集
    test_datasets = {k: v for k, v in DATASETS.items() if k in args.datasets}
    resolved_protocol = _resolve_protocol(args.protocol, list(test_datasets.keys()), args.required_test_acc)

    print(f"🚀 Paper 1: {MODEL_NAME} 多数据集验证开始")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 测试数据集: {list(test_datasets.keys())}")
    print(f"🧭 协议: {resolved_protocol}")
    print(f"💾 输出目录: {args.output_dir}")

    results = []

    # 运行所有数据集实验
    for i, (dataset_name, dataset_info) in enumerate(test_datasets.items()):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(test_datasets)}")
        print(f"当前数据集: {dataset_name}")
        print(f"描述: {dataset_info['description']}")
        print(f"期望准确率: {dataset_info['expected_acc']*100:.0f}%")
        print(f"{'='*60}")

        # 创建数据集专用配置
        config_path = create_dataset_config(dataset_name, dataset_info, args.output_dir, resolved_protocol)

        if args.dry_run:
            results.append({
                'dataset': dataset_name,
                'config_path': config_path,
                'protocol': resolved_protocol,
                'status': 'dry_run',
            })
            continue

        # 运行实验
        success, duration, log_output = run_dataset_experiment(dataset_name, config_path, args.gpu_id)

        # 提取指标
        if success:
            metrics = extract_metrics_from_result_artifacts(Path(args.output_dir) / dataset_name)
            if metrics is None:
                metrics = extract_metrics_from_log(log_output, dataset_name)
        else:
            metrics = {'dataset': dataset_name}

        # 保存结果
        results.append({
            'dataset': dataset_name,
            'config_path': config_path,
            'status': 'success' if success else 'failed',
            'duration': duration,
            'metrics': metrics,
            'artifact_metrics_found': success and metrics.get("source_csv") is not None,
            'log_file': os.path.join(args.output_dir, f"{dataset_name}_log.txt")
        })

        # 保存日志
        with open(results[-1]['log_file'], 'w') as f:
            f.write(f"{dataset_name} 数据集验证日志\n")
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
            'protocol': resolved_protocol,
            'datasets': results,
        }, ensure_ascii=False, indent=2))
        return

    # 保存所有结果
    save_dataset_results(results, args.output_dir, args.required_test_acc)

    print(f"\n🎉 多数据集验证完成！")
    print(f"📊 结果摘要:")

    successful_count = sum(1 for r in results if r['status'] == 'success')
    print(f"   ✅ 成功: {successful_count}/{len(results)}")

    if successful_count > 0:
        successful_accs = [r['metrics']['test_acc'] for r in results
                          if r['status'] == 'success' and r['metrics']['test_acc']]
        if successful_accs:
            mean_acc = sum(successful_accs) / len(successful_accs)
            print(f"   📈 平均准确率: {mean_acc*100:.2f}%")

            # 泛化能力评估
            performance_variance = max(successful_accs) - min(successful_accs)
            print(f"   🌐 泛化差异: {performance_variance*100:.2f}%")

    threshold_pass_datasets = [
        r['dataset']
        for r in results
        if r['status'] == 'success'
        and r['metrics'].get('test_acc') is not None
        and r['metrics']['test_acc'] >= args.required_test_acc
    ]
    machine_summary = {
        'required_test_acc': args.required_test_acc,
        'threshold_pass': len(threshold_pass_datasets) == len(results) and len(results) > 0,
        'threshold_pass_datasets': threshold_pass_datasets,
        'successful_datasets': [r['dataset'] for r in results if r['status'] == 'success'],
        'mean_test_acc': (
            sum(r['metrics']['test_acc'] for r in results if r['status'] == 'success' and r['metrics'].get('test_acc') is not None)
            / len([r for r in results if r['status'] == 'success' and r['metrics'].get('test_acc') is not None])
            if [r for r in results if r['status'] == 'success' and r['metrics'].get('test_acc') is not None]
            else None
        ),
    }
    print(json.dumps(machine_summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
