#!/usr/bin/env python3
"""
Flow预训练批量实验运行器
自动运行多个Flow实验并汇总结果
"""

import os
import sys
import subprocess
import argparse
import time
import json
from datetime import datetime
from pathlib import Path


class FlowExperimentBatch:
    """Flow实验批量运行管理器"""
    
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.experiment_script = self.script_dir / "run_flow_experiments.sh"
        self.results = []
        
    def run_experiment(self, exp_type, gpu_id=0, notes="", enable_wandb=False, dry_run=False):
        """运行单个实验"""
        print(f"🚀 开始实验: {exp_type}")
        print(f"   GPU: {gpu_id}, 备注: {notes}")
        
        # 构建命令
        cmd = [str(self.experiment_script), exp_type]
        cmd.extend(["--gpu", str(gpu_id)])
        
        if notes:
            cmd.extend(["--notes", notes])
        
        if enable_wandb:
            cmd.append("--wandb")
            
        if dry_run:
            cmd.append("--dry-run")
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行实验
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # 计算运行时间
            duration = time.time() - start_time
            
            # 记录结果
            experiment_result = {
                'experiment': exp_type,
                'gpu': gpu_id,
                'notes': notes,
                'status': 'success',
                'duration': duration,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'stdout': result.stdout[-1000:] if result.stdout else "",  # 保留最后1000字符
                'stderr': result.stderr[-1000:] if result.stderr else ""
            }
            
            print(f"✅ 实验完成: {exp_type} (耗时: {duration/60:.1f}分钟)")
            
        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            experiment_result = {
                'experiment': exp_type,
                'gpu': gpu_id,
                'notes': notes,
                'status': 'failed',
                'duration': duration,
                'start_time': datetime.fromtimestamp(start_time).isoformat(),
                'error': str(e),
                'stdout': e.stdout[-1000:] if e.stdout else "",
                'stderr': e.stderr[-1000:] if e.stderr else ""
            }
            
            print(f"❌ 实验失败: {exp_type} (耗时: {duration/60:.1f}分钟)")
            print(f"   错误: {e}")
        
        self.results.append(experiment_result)
        return experiment_result
    
    def run_validation_suite(self, gpu_id=0, enable_wandb=False):
        """运行完整验证套件"""
        print("🎯 Flow预训练验证套件")
        print("=" * 50)
        
        experiments = [
            ("quick", "快速功能验证"),
            ("baseline", "基线性能测试"), 
            ("contrastive", "Flow+对比学习验证")
        ]
        
        total_start = time.time()
        
        for exp_type, description in experiments:
            print(f"\n📋 实验 {len(self.results)+1}/{len(experiments)}: {description}")
            self.run_experiment(exp_type, gpu_id=gpu_id, notes=description, enable_wandb=enable_wandb)
            
            # 实验间休息
            if exp_type != experiments[-1][0]:
                print("⏸️  实验间休息 30 秒...")
                time.sleep(30)
        
        total_duration = time.time() - total_start
        
        print(f"\n🎉 验证套件完成! 总耗时: {total_duration/60:.1f}分钟")
        self.print_summary()
        
    def run_research_pipeline(self, gpu_id=0, enable_wandb=True):
        """运行研究级实验管道"""
        print("🔬 Flow预训练研究级管道")
        print("=" * 50)
        
        experiments = [
            ("baseline", "建立基线"),
            ("contrastive", "对比学习增强"),
            ("pipeline02", "Pipeline_02预训练"),
            ("research", "完整研究实验")
        ]
        
        total_start = time.time()
        
        for exp_type, description in experiments:
            print(f"\n📋 实验 {len(self.results)+1}/{len(experiments)}: {description}")
            self.run_experiment(exp_type, gpu_id=gpu_id, notes=description, enable_wandb=enable_wandb)
            
            # 长实验后的休息时间
            if exp_type in ['baseline', 'contrastive', 'pipeline02']:
                print("⏸️  实验间休息 60 秒...")
                time.sleep(60)
        
        total_duration = time.time() - total_start
        
        print(f"\n🎉 研究管道完成! 总耗时: {total_duration/3600:.1f}小时")
        self.print_summary()
        
    def print_summary(self):
        """打印实验结果摘要"""
        print("\n📊 实验结果摘要")
        print("=" * 50)
        
        successful = len([r for r in self.results if r['status'] == 'success'])
        failed = len([r for r in self.results if r['status'] == 'failed'])
        total_time = sum(r['duration'] for r in self.results)
        
        print(f"总实验数: {len(self.results)}")
        print(f"成功: {successful} ✅")
        print(f"失败: {failed} ❌") 
        print(f"成功率: {successful/len(self.results)*100:.1f}%")
        print(f"总耗时: {total_time/3600:.1f}小时")
        
        print("\n📋 详细结果:")
        for i, result in enumerate(self.results, 1):
            status_emoji = "✅" if result['status'] == 'success' else "❌"
            duration_str = f"{result['duration']/60:.1f}min"
            print(f"{i:2d}. {result['experiment']:12s} {status_emoji} ({duration_str}) - {result['notes']}")
        
    def save_results(self, filename=None):
        """保存实验结果到JSON文件"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flow_experiment_results_{timestamp}.json"
        
        filepath = self.script_dir / filename
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(self.results),
            'successful': len([r for r in self.results if r['status'] == 'success']),
            'failed': len([r for r in self.results if r['status'] == 'failed']),
            'total_duration': sum(r['duration'] for r in self.results),
            'experiments': self.results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存到: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Flow预训练批量实验运行器")
    
    parser.add_argument('mode', choices=['validation', 'research', 'custom'], 
                       help='运行模式: validation(验证套件), research(研究管道), custom(自定义)')
    
    parser.add_argument('--gpu', type=int, default=0, help='GPU编号 (默认: 0)')
    parser.add_argument('--wandb', action='store_true', help='启用WandB跟踪')
    parser.add_argument('--save-results', type=str, help='保存结果到指定文件')
    
    # 自定义模式参数
    parser.add_argument('--experiments', nargs='+', 
                       choices=['quick', 'baseline', 'contrastive', 'pipeline02', 'research'],
                       help='自定义模式: 指定要运行的实验')
    
    args = parser.parse_args()
    
    # 创建批量运行器
    batch_runner = FlowExperimentBatch()
    
    print(f"🚀 Flow预训练批量实验运行器")
    print(f"模式: {args.mode}")
    print(f"GPU: {args.gpu}")
    print(f"WandB: {'启用' if args.wandb else '禁用'}")
    print("=" * 50)
    
    try:
        if args.mode == 'validation':
            batch_runner.run_validation_suite(gpu_id=args.gpu, enable_wandb=args.wandb)
            
        elif args.mode == 'research':
            batch_runner.run_research_pipeline(gpu_id=args.gpu, enable_wandb=args.wandb)
            
        elif args.mode == 'custom':
            if not args.experiments:
                print("❌ 自定义模式需要指定 --experiments")
                return 1
            
            print(f"📋 自定义实验序列: {args.experiments}")
            
            for exp_type in args.experiments:
                batch_runner.run_experiment(
                    exp_type, 
                    gpu_id=args.gpu, 
                    notes=f"自定义批量: {exp_type}",
                    enable_wandb=args.wandb
                )
                
                # 实验间休息
                if exp_type != args.experiments[-1]:
                    print("⏸️  实验间休息 30 秒...")
                    time.sleep(30)
            
            batch_runner.print_summary()
        
        # 保存结果
        result_file = batch_runner.save_results(args.save_results)
        
        print(f"\n✨ 批量实验完成!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️  用户中断批量实验")
        if batch_runner.results:
            print("📊 已完成的实验:")
            batch_runner.print_summary()
            batch_runner.save_results()
        return 1
    
    except Exception as e:
        print(f"\n❌ 批量实验异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())