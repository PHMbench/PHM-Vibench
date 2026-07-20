#!/usr/bin/env python3
"""
MoE专家数量消融实验执行脚本
测试3、5、8专家配置下的性能对比
"""

import os
import sys
import subprocess
import yaml
import time
from datetime import datetime

# 添加项目根目录
sys.path.append('../../../')

def create_expert_configs():
    """创建不同专家数量的配置文件"""
    base_config = {
        'signal_processing_configs': {
            'layer1': ['HT', 'WF', 'I'],
            'layer2': ['HT', 'WF', 'I'],
            'layer3': ['I', 'WF', 'I'],
            'layer4': ['I', 'WF', 'I']
        },
        'feature_extractor_configs': [
            'Mean', 'Std', 'Var', 'Entropy', 'Max', 'Min',
            'AbsMean', 'Kurtosis', 'RMS', 'CrestFactor',
            'Skewness', 'ClearanceFactor', 'ShapeFactor'
        ]
    }

    # 专家数量配置
    expert_configs = [
        {'num_experts': 3, 'name': 'MoE_3experts_seed20'},
        {'num_experts': 5, 'name': 'MoE_5experts_seed20'},
        {'num_experts': 8, 'name': 'MoE_8experts_seed20'}
    ]

    for config in expert_configs:
        args = base_config.copy()
        args['args'] = {
            'model': 'MoE',
            'skip_connection': True,
            'scale': 4,
            'l1_norm': 0.0001,
            'num_epochs': 100,
            'patience': 15,
            'gpus': 1,

            # 基础参数
            'device': 'cuda',
            'data_dir': '/home/user/data/a_bearing/a_018_THU24_pro/',
            'dataset_task': 'THU_018_basic',
            'target': 'IF',
            'k_shot': 64,
            'num_classes': 5,
            'in_dim': 4096,
            'out_dim': 4096,
            'in_channels': 2,
            'out_channels': 3,
            'f_c_mu': 0,
            'f_c_sigma': 0.1,
            'f_b_mu': 0,
            'f_b_sigma': 0.1,
            'learning_rate': 0.001,
            'batch_size': 64,
            'weight_decay': 0.0001,
            'num_workers': 8,
            'seed': 20,  # 使用Seed 20成功配置
            'monitor': 'val_loss',
            'pruning': None,
            'snr': 0,
            'experiment_type': 'unified_baseline',
            'description': f'MoE混合专家网络{config["num_experts"]}专家Seed 20实验'
        }

        # 保存配置文件
        config_path = f'../../../configs/unified_baseline/config_{config["name"]}.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(args, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ 创建配置文件: {config_path}")

def run_experiment(config_name, gpu_id):
    """运行单个实验"""
    print(f"\n🚀 启动实验: {config_name} (GPU {gpu_id})")

    cmd = [
        'source', '/home/user/anaconda3/etc/profile.d/conda.sh',
        '&&',
        'conda', 'activate', 'LQ_signal',
        '&&',
        'cd', '/home/user/LQ/B_Signal/Unified_X_fault_diagnosis',
        '&&',
        f'CUDA_VISIBLE_DEVICES={gpu_id}',
        'python', 'main.py',
        f'--config_dir', f'configs/unified_baseline/config_{config_name}.yaml'
    ]

    # 使用subprocess运行
    process = subprocess.Popen(
        ' '.join(cmd),
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    return process

def main():
    """主函数"""
    print("🎯 MoE专家数量消融实验")
    print("=" * 50)

    # 创建配置文件
    print("\n📝 第一步: 创建配置文件...")
    create_expert_configs()

    # 实验配置
    experiments = [
        {'name': 'MoE_3experts_seed20', 'gpu': 0},
        {'name': 'MoE_5experts_seed20', 'gpu': 1},
        {'name': 'MoE_8experts_seed20', 'gpu': 2}
    ]

    # 启动实验
    print("\n🚀 第二步: 启动消融实验...")
    processes = []

    for exp in experiments:
        process = run_experiment(exp['name'], exp['gpu'])
        processes.append({
            'name': exp['name'],
            'process': process,
            'start_time': datetime.now()
        })
        time.sleep(5)  # 避免同时启动冲突

    # 监控进度
    print("\n📊 第三步: 监控实验进度...")

    while processes:
        for i, proc_info in enumerate(processes):
            process = proc_info['process']

            # 检查进程状态
            if process.poll() is not None:
                # 进程结束
                returncode = process.returncode
                elapsed = datetime.now() - proc_info['start_time']

                if returncode == 0:
                    print(f"✅ {proc_info['name']} 完成 (耗时: {elapsed})")
                else:
                    print(f"❌ {proc_info['name']} 失败 (返回码: {returncode})")
                    # 输出错误信息
                    stderr_output = process.stderr.read()
                    if stderr_output:
                        print(f"错误信息: {stderr_output[:500]}...")

                # 从列表中移除
                processes.pop(i)
                break

        time.sleep(60)  # 每分钟检查一次

    print("\n🎉 所有实验完成!")
    print("\n📈 生成结果分析...")

    # 这里可以添加结果分析代码
    print("📊 实验结果汇总:")
    print("- 3专家: 等待结果...")
    print("- 5专家: 等待结果...")
    print("- 8专家: 等待结果...")

if __name__ == "__main__":
    main()