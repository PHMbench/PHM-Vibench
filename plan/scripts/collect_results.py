#!/usr/bin/env python3
"""
结果收集和汇总脚本
自动收集所有Flow实验结果并生成汇总报告
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

def collect_experiment_results(experiment_dir="results/", output_prefix="experiment"):
    """汇总所有实验结果"""
    results = []
    
    print(f"📊 收集实验结果从: {experiment_dir}")
    
    for exp_path in Path(experiment_dir).glob("*/"):
        if exp_path.is_dir():
            print(f"  检查: {exp_path.name}")
            
            # 查找指标文件
            metrics_files = [
                exp_path / "metrics.json",
                exp_path / "lightning_logs" / "version_0" / "metrics.csv",
                exp_path / "results.json"
            ]
            
            experiment_data = {
                'experiment': exp_path.name,
                'path': str(exp_path),
                'timestamp': exp_path.stat().st_mtime
            }
            
            # 尝试从不同文件加载指标
            metrics_loaded = False
            
            for metrics_file in metrics_files:
                if metrics_file.exists():
                    try:
                        if metrics_file.suffix == '.json':
                            with open(metrics_file) as f:
                                metrics = json.load(f)
                            experiment_data.update(metrics)
                            metrics_loaded = True
                            break
                            
                        elif metrics_file.suffix == '.csv':
                            # PyTorch Lightning CSV格式
                            df = pd.read_csv(metrics_file)
                            if not df.empty:
                                # 取最后一个epoch的指标
                                last_metrics = df.iloc[-1].to_dict()
                                # 清理指标名称
                                clean_metrics = {}
                                for k, v in last_metrics.items():
                                    if not pd.isna(v) and k not in ['epoch', 'step']:
                                        clean_metrics[k] = v
                                experiment_data.update(clean_metrics)
                                metrics_loaded = True
                                break
                                
                    except Exception as e:
                        print(f"    警告: 无法读取 {metrics_file}: {e}")
                        continue
            
            if not metrics_loaded:
                print(f"    警告: 未找到有效的指标文件 for {exp_path.name}")
            
            # 添加默认值
            for key in ['accuracy', 'test_accuracy', 'val_accuracy']:
                if key not in experiment_data:
                    experiment_data[key] = None
            
            for key in ['f1_score', 'test_f1', 'val_f1']:
                if key not in experiment_data:
                    experiment_data[key] = None
                    
            for key in ['training_time', 'model_params']:
                if key not in experiment_data:
                    experiment_data[key] = None
            
            results.append(experiment_data)
    
    if not results:
        print("❌ 未找到任何实验结果!")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 数据清理和标准化
    df = clean_and_standardize_results(df)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = f"{output_prefix}_results_{timestamp}.csv"
    excel_file = f"{output_prefix}_results_{timestamp}.xlsx"
    
    df.to_csv(csv_file, index=False)
    df.to_excel(excel_file, index=False)
    
    print(f"✅ 结果已保存:")
    print(f"   CSV: {csv_file}")
    print(f"   Excel: {excel_file}")
    
    # 打印摘要统计
    print_summary_statistics(df)
    
    return df

def clean_and_standardize_results(df):
    """清理和标准化结果数据"""
    
    # 提取实验类型
    df['experiment_type'] = df['experiment'].apply(extract_experiment_type)
    
    # 标准化准确率列
    accuracy_cols = ['accuracy', 'test_accuracy', 'val_accuracy', 'test_acc', 'val_acc']
    df['final_accuracy'] = None
    
    for _, row in df.iterrows():
        for col in accuracy_cols:
            if col in row and row[col] is not None and not pd.isna(row[col]):
                df.loc[df['experiment'] == row['experiment'], 'final_accuracy'] = row[col]
                break
    
    # 标准化F1分数列
    f1_cols = ['f1_score', 'test_f1', 'val_f1', 'f1', 'test_f1_score', 'val_f1_score']
    df['final_f1'] = None
    
    for _, row in df.iterrows():
        for col in f1_cols:
            if col in row and row[col] is not None and not pd.isna(row[col]):
                df.loc[df['experiment'] == row['experiment'], 'final_f1'] = row[col]
                break
    
    # 转换时间戳为可读格式
    if 'timestamp' in df.columns:
        df['date_created'] = pd.to_datetime(df['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M')
    
    # 按时间戳排序
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp', ascending=False)
    
    return df

def extract_experiment_type(experiment_name):
    """从实验名称提取实验类型"""
    name_lower = experiment_name.lower()
    
    if 'flow_quick' in name_lower or 'quick' in name_lower:
        return 'quick_validation'
    elif 'flow_baseline' in name_lower or 'baseline' in name_lower:
        return 'baseline'
    elif 'flow_contrastive' in name_lower or 'contrastive' in name_lower:
        return 'contrastive'
    elif 'flow_pipeline02' in name_lower or 'pipeline02' in name_lower:
        return 'pipeline02'
    elif 'flow_research' in name_lower or 'research' in name_lower:
        return 'research'
    elif 'ablation' in name_lower:
        return 'ablation'
    elif 'comparison' in name_lower:
        return 'comparison'
    else:
        return 'other'

def print_summary_statistics(df):
    """打印摘要统计信息"""
    
    print("\n📈 实验结果摘要:")
    print("=" * 50)
    
    print(f"总实验数: {len(df)}")
    
    if 'experiment_type' in df.columns:
        print(f"\n实验类型分布:")
        type_counts = df['experiment_type'].value_counts()
        for exp_type, count in type_counts.items():
            print(f"  {exp_type}: {count}")
    
    if 'final_accuracy' in df.columns and df['final_accuracy'].notna().sum() > 0:
        acc_stats = df['final_accuracy'].dropna()
        print(f"\n准确率统计:")
        print(f"  平均值: {acc_stats.mean():.4f}")
        print(f"  标准差: {acc_stats.std():.4f}")
        print(f"  最大值: {acc_stats.max():.4f}")
        print(f"  最小值: {acc_stats.min():.4f}")
        print(f"  中位数: {acc_stats.median():.4f}")
    
    if 'final_f1' in df.columns and df['final_f1'].notna().sum() > 0:
        f1_stats = df['final_f1'].dropna()
        print(f"\nF1分数统计:")
        print(f"  平均值: {f1_stats.mean():.4f}")
        print(f"  标准差: {f1_stats.std():.4f}")
        print(f"  最大值: {f1_stats.max():.4f}")
        print(f"  最小值: {f1_stats.min():.4f}")
    
    # 最佳性能实验
    if 'final_accuracy' in df.columns and df['final_accuracy'].notna().sum() > 0:
        best_acc_idx = df['final_accuracy'].idxmax()
        best_exp = df.loc[best_acc_idx]
        print(f"\n🏆 最佳准确率实验:")
        print(f"  实验名: {best_exp['experiment']}")
        print(f"  准确率: {best_exp['final_accuracy']:.4f}")
        if 'final_f1' in best_exp and not pd.isna(best_exp['final_f1']):
            print(f"  F1分数: {best_exp['final_f1']:.4f}")

def generate_latex_summary_table(df, output_file="results_table.tex"):
    """生成LaTeX格式的结果汇总表"""
    
    # 按实验类型分组
    if 'experiment_type' not in df.columns:
        print("警告: 未找到实验类型信息，跳过LaTeX表格生成")
        return
    
    # 计算每种实验类型的统计信息
    summary_stats = []
    
    for exp_type in df['experiment_type'].unique():
        type_df = df[df['experiment_type'] == exp_type]
        
        acc_data = type_df['final_accuracy'].dropna()
        f1_data = type_df['final_f1'].dropna()
        
        if len(acc_data) > 0:
            acc_mean = acc_data.mean()
            acc_std = acc_data.std() if len(acc_data) > 1 else 0
        else:
            acc_mean = acc_std = None
            
        if len(f1_data) > 0:
            f1_mean = f1_data.mean()
            f1_std = f1_data.std() if len(f1_data) > 1 else 0
        else:
            f1_mean = f1_std = None
        
        summary_stats.append({
            'type': exp_type,
            'count': len(type_df),
            'acc_mean': acc_mean,
            'acc_std': acc_std,
            'f1_mean': f1_mean,
            'f1_std': f1_std
        })
    
    # 生成LaTeX表格
    latex_content = """\\begin{table}[h]
\\centering
\\caption{Flow Pretraining Experiment Results Summary}
\\label{tab:flow_results_summary}
\\begin{tabular}{lcccc}
\\toprule
Experiment Type & Count & Accuracy (\\%) & F1-Score & Notes \\\\
\\midrule
"""
    
    for stat in summary_stats:
        exp_type = stat['type'].replace('_', '\\_')
        count = stat['count']
        
        if stat['acc_mean'] is not None:
            if stat['acc_std'] is not None and stat['acc_std'] > 0:
                acc_str = f"{stat['acc_mean']*100:.2f} $\\pm$ {stat['acc_std']*100:.2f}"
            else:
                acc_str = f"{stat['acc_mean']*100:.2f}"
        else:
            acc_str = "N/A"
            
        if stat['f1_mean'] is not None:
            if stat['f1_std'] is not None and stat['f1_std'] > 0:
                f1_str = f"{stat['f1_mean']:.3f} $\\pm$ {stat['f1_std']:.3f}"
            else:
                f1_str = f"{stat['f1_mean']:.3f}"
        else:
            f1_str = "N/A"
        
        latex_content += f"{exp_type} & {count} & {acc_str} & {f1_str} & \\\\\n"
    
    latex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"✅ LaTeX表格已保存至: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="收集和分析Flow实验结果")
    parser.add_argument('--results_dir', type=str, default='results/', 
                       help='实验结果目录')
    parser.add_argument('--output_prefix', type=str, default='flow_experiment',
                       help='输出文件前缀')
    parser.add_argument('--generate_latex', action='store_true',
                       help='生成LaTeX格式表格')
    
    args = parser.parse_args()
    
    # 收集结果
    df = collect_experiment_results(args.results_dir, args.output_prefix)
    
    if df is not None and args.generate_latex:
        generate_latex_summary_table(df, f"{args.output_prefix}_summary.tex")
    
    print("\n🎉 结果收集完成!")

if __name__ == "__main__":
    main()