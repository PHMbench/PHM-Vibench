#!/usr/bin/env python3
"""
统计分析脚本
对实验结果进行统计显著性检验和效应大小分析
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import ttest_ind, mannwhitneyu, wilcoxon, kruskal, f_oneway
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse

class StatisticalAnalyzer:
    """统计分析类"""
    
    def __init__(self, alpha=0.05):
        """
        参数:
            alpha: 显著性水平，默认0.05
        """
        self.alpha = alpha
        self.results = {}
    
    def compare_two_groups(self, group1_scores, group2_scores, 
                          group1_name="Method 1", group2_name="Method 2",
                          alternative='two-sided'):
        """比较两组方法的统计显著性"""
        
        print(f"\n📊 比较分析: {group1_name} vs {group2_name}")
        print("=" * 50)
        
        # 基本统计
        stats1 = self._compute_descriptive_stats(group1_scores, group1_name)
        stats2 = self._compute_descriptive_stats(group2_scores, group2_name)
        
        print(f"\n📈 描述性统计:")
        print(f"{group1_name}: μ={stats1['mean']:.4f}, σ={stats1['std']:.4f}, n={stats1['n']}")
        print(f"{group2_name}: μ={stats2['mean']:.4f}, σ={stats2['std']:.4f}, n={stats2['n']}")
        
        # 正态性检验
        normality1 = self._test_normality(group1_scores, group1_name)
        normality2 = self._test_normality(group2_scores, group2_name)
        
        # 方差齐性检验
        homoscedasticity = self._test_homoscedasticity(group1_scores, group2_scores)
        
        # 选择合适的统计检验
        if normality1['normal'] and normality2['normal']:
            if homoscedasticity['equal_var']:
                # 独立样本t检验 (等方差)
                test_result = self._independent_ttest(
                    group1_scores, group2_scores, equal_var=True, alternative=alternative
                )
                test_name = "Independent t-test (equal variance)"
            else:
                # Welch's t检验 (不等方差)
                test_result = self._independent_ttest(
                    group1_scores, group2_scores, equal_var=False, alternative=alternative
                )
                test_name = "Welch's t-test (unequal variance)"
        else:
            # 非参数检验: Mann-Whitney U检验
            test_result = self._mann_whitney_test(
                group1_scores, group2_scores, alternative=alternative
            )
            test_name = "Mann-Whitney U test"
        
        # 效应大小
        effect_size = self._compute_effect_size(group1_scores, group2_scores)
        
        # 置信区间
        confidence_interval = self._compute_confidence_interval(group1_scores, group2_scores)
        
        # 汇总结果
        comparison_result = {
            'group1_name': group1_name,
            'group2_name': group2_name,
            'group1_stats': stats1,
            'group2_stats': stats2,
            'normality_test': {
                'group1': normality1,
                'group2': normality2
            },
            'homoscedasticity_test': homoscedasticity,
            'statistical_test': {
                'name': test_name,
                'statistic': test_result['statistic'],
                'p_value': test_result['p_value'],
                'significant': test_result['p_value'] < self.alpha,
                'interpretation': self._interpret_p_value(test_result['p_value'])
            },
            'effect_size': effect_size,
            'confidence_interval': confidence_interval
        }
        
        # 打印结果
        self._print_comparison_results(comparison_result)
        
        return comparison_result
    
    def compare_multiple_groups(self, groups_dict, group_names=None):
        """比较多组方法的统计显著性"""
        
        if group_names is None:
            group_names = list(groups_dict.keys())
        
        print(f"\n📊 多组比较分析")
        print(f"组数: {len(groups_dict)}")
        print("=" * 50)
        
        # 准备数据
        groups_data = []
        groups_labels = []
        
        for name in group_names:
            if name in groups_dict:
                scores = groups_dict[name]
                groups_data.append(scores)
                groups_labels.append(name)
                
                # 打印描述性统计
                stats = self._compute_descriptive_stats(scores, name)
                print(f"{name}: μ={stats['mean']:.4f}, σ={stats['std']:.4f}, n={stats['n']}")
        
        # 正态性检验
        print(f"\n🔍 正态性检验:")
        normality_results = []
        for i, (scores, name) in enumerate(zip(groups_data, groups_labels)):
            normality = self._test_normality(scores, name)
            normality_results.append(normality['normal'])
            print(f"{name}: {'正态分布' if normality['normal'] else '非正态分布'} (p={normality['p_value']:.4f})")
        
        # 方差齐性检验 (Levene检验)
        print(f"\n🔍 方差齐性检验:")
        levene_stat, levene_p = stats.levene(*groups_data)
        equal_variances = levene_p >= self.alpha
        print(f"Levene检验: F={levene_stat:.4f}, p={levene_p:.4f}")
        print(f"方差{'齐性' if equal_variances else '不齐性'}")
        
        # 选择合适的统计检验
        all_normal = all(normality_results)
        
        if all_normal and equal_variances:
            # 单因素方差分析 (ANOVA)
            f_stat, p_value = f_oneway(*groups_data)
            test_name = "One-way ANOVA"
            post_hoc = self._tukey_hsd_test(groups_data, groups_labels) if p_value < self.alpha else None
        else:
            # 非参数检验: Kruskal-Wallis检验
            h_stat, p_value = kruskal(*groups_data)
            f_stat = h_stat  # 为了保持一致性
            test_name = "Kruskal-Wallis H test"
            post_hoc = self._dunn_test(groups_data, groups_labels) if p_value < self.alpha else None
        
        # 效应大小 (eta squared)
        eta_squared = self._compute_eta_squared(groups_data)
        
        # 汇总结果
        multiple_comparison_result = {
            'groups': groups_labels,
            'group_stats': {
                name: self._compute_descriptive_stats(scores, name) 
                for name, scores in zip(groups_labels, groups_data)
            },
            'normality_test': {
                name: self._test_normality(scores, name)
                for name, scores in zip(groups_labels, groups_data)
            },
            'levene_test': {
                'statistic': levene_stat,
                'p_value': levene_p,
                'equal_variances': equal_variances
            },
            'statistical_test': {
                'name': test_name,
                'statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'interpretation': self._interpret_p_value(p_value)
            },
            'effect_size': {
                'eta_squared': eta_squared,
                'interpretation': self._interpret_eta_squared(eta_squared)
            },
            'post_hoc': post_hoc
        }
        
        # 打印结果
        self._print_multiple_comparison_results(multiple_comparison_result)
        
        return multiple_comparison_result
    
    def paired_comparison(self, before_scores, after_scores, 
                         method1_name="Before", method2_name="After"):
        """配对比较 (如同一数据集上不同方法的比较)"""
        
        if len(before_scores) != len(after_scores):
            raise ValueError("配对数据长度必须相等")
        
        print(f"\n📊 配对比较分析: {method1_name} vs {method2_name}")
        print("=" * 50)
        
        # 计算差值
        differences = np.array(after_scores) - np.array(before_scores)
        
        # 描述性统计
        print(f"\n📈 配对差值统计:")
        print(f"平均差值: {np.mean(differences):.4f}")
        print(f"差值标准差: {np.std(differences, ddof=1):.4f}")
        print(f"配对数: {len(differences)}")
        
        # 差值的正态性检验
        diff_normality = self._test_normality(differences, "差值")
        
        # 选择合适的检验
        if diff_normality['normal']:
            # 配对t检验
            t_stat, p_value = stats.ttest_rel(after_scores, before_scores)
            test_name = "Paired t-test"
        else:
            # Wilcoxon符号秩检验
            w_stat, p_value = wilcoxon(after_scores, before_scores)
            t_stat = w_stat  # 为了保持一致性
            test_name = "Wilcoxon signed-rank test"
        
        # 效应大小 (Cohen's d for paired data)
        effect_size = np.mean(differences) / np.std(differences, ddof=1)
        
        # 配对结果
        paired_result = {
            'method1_name': method1_name,
            'method2_name': method2_name,
            'n_pairs': len(differences),
            'mean_difference': np.mean(differences),
            'std_difference': np.std(differences, ddof=1),
            'difference_normality': diff_normality,
            'statistical_test': {
                'name': test_name,
                'statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'interpretation': self._interpret_p_value(p_value)
            },
            'effect_size': {
                'cohens_d': effect_size,
                'interpretation': self._interpret_cohens_d(effect_size)
            }
        }
        
        # 打印结果
        self._print_paired_comparison_results(paired_result)
        
        return paired_result
    
    def _compute_descriptive_stats(self, scores, name):
        """计算描述性统计"""
        return {
            'name': name,
            'n': len(scores),
            'mean': np.mean(scores),
            'std': np.std(scores, ddof=1),
            'min': np.min(scores),
            'max': np.max(scores),
            'median': np.median(scores),
            'q25': np.percentile(scores, 25),
            'q75': np.percentile(scores, 75)
        }
    
    def _test_normality(self, scores, name):
        """测试正态性 (Shapiro-Wilk检验)"""
        if len(scores) < 3:
            return {'name': name, 'normal': True, 'p_value': 1.0, 'test': 'insufficient_data'}
        
        if len(scores) <= 5000:
            # Shapiro-Wilk检验 (适用于小样本)
            stat, p_value = stats.shapiro(scores)
            test_name = 'Shapiro-Wilk'
        else:
            # D'Agostino和Pearson检验 (适用于大样本)
            stat, p_value = stats.normaltest(scores)
            test_name = "D'Agostino-Pearson"
        
        return {
            'name': name,
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            'normal': p_value >= self.alpha
        }
    
    def _test_homoscedasticity(self, group1_scores, group2_scores):
        """方差齐性检验 (Levene检验)"""
        stat, p_value = stats.levene(group1_scores, group2_scores)
        
        return {
            'test': 'Levene',
            'statistic': stat,
            'p_value': p_value,
            'equal_var': p_value >= self.alpha
        }
    
    def _independent_ttest(self, group1_scores, group2_scores, equal_var=True, alternative='two-sided'):
        """独立样本t检验"""
        t_stat, p_value = ttest_ind(group1_scores, group2_scores, equal_var=equal_var, alternative=alternative)
        
        return {
            'statistic': t_stat,
            'p_value': p_value
        }
    
    def _mann_whitney_test(self, group1_scores, group2_scores, alternative='two-sided'):
        """Mann-Whitney U检验"""
        u_stat, p_value = mannwhitneyu(group1_scores, group2_scores, alternative=alternative)
        
        return {
            'statistic': u_stat,
            'p_value': p_value
        }
    
    def _compute_effect_size(self, group1_scores, group2_scores):
        """计算效应大小 (Cohen's d)"""
        n1, n2 = len(group1_scores), len(group2_scores)
        mean1, mean2 = np.mean(group1_scores), np.mean(group2_scores)
        var1, var2 = np.var(group1_scores, ddof=1), np.var(group2_scores, ddof=1)
        
        # 合并标准差
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean1 - mean2) / pooled_std
        
        return {
            'cohens_d': cohens_d,
            'interpretation': self._interpret_cohens_d(cohens_d),
            'pooled_std': pooled_std
        }
    
    def _compute_confidence_interval(self, group1_scores, group2_scores, confidence=0.95):
        """计算均值差的置信区间"""
        n1, n2 = len(group1_scores), len(group2_scores)
        mean1, mean2 = np.mean(group1_scores), np.mean(group2_scores)
        var1, var2 = np.var(group1_scores, ddof=1), np.var(group2_scores, ddof=1)
        
        # 均值差
        mean_diff = mean1 - mean2
        
        # 标准误
        se = np.sqrt(var1/n1 + var2/n2)
        
        # 自由度 (Welch-Satterthwaite方程)
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        
        # 临界值
        alpha = 1 - confidence
        t_critical = stats.t.ppf(1 - alpha/2, df)
        
        # 置信区间
        margin_of_error = t_critical * se
        ci_lower = mean_diff - margin_of_error
        ci_upper = mean_diff + margin_of_error
        
        return {
            'mean_difference': mean_diff,
            'confidence_level': confidence,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'margin_of_error': margin_of_error
        }
    
    def _compute_eta_squared(self, groups_data):
        """计算eta squared (效应大小)"""
        # 计算组间平方和和总平方和
        all_data = np.concatenate(groups_data)
        grand_mean = np.mean(all_data)
        
        ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 for group in groups_data])
        ss_total = sum([(x - grand_mean)**2 for x in all_data])
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return eta_squared
    
    def _tukey_hsd_test(self, groups_data, groups_labels):
        """Tukey HSD事后检验"""
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            
            # 准备数据
            all_data = []
            all_labels = []
            
            for data, label in zip(groups_data, groups_labels):
                all_data.extend(data)
                all_labels.extend([label] * len(data))
            
            # 执行Tukey HSD
            tukey_result = pairwise_tukeyhsd(all_data, all_labels, alpha=self.alpha)
            
            return {
                'test': 'Tukey HSD',
                'summary': str(tukey_result)
            }
            
        except ImportError:
            print("⚠️  statsmodels未安装，跳过Tukey HSD检验")
            return None
    
    def _dunn_test(self, groups_data, groups_labels):
        """Dunn事后检验 (非参数)"""
        # 简化版的Dunn检验
        comparisons = []
        
        for i in range(len(groups_data)):
            for j in range(i+1, len(groups_data)):
                u_stat, p_value = mannwhitneyu(groups_data[i], groups_data[j])
                comparisons.append({
                    'group1': groups_labels[i],
                    'group2': groups_labels[j],
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'significant': p_value < (self.alpha / len(comparisons))  # Bonferroni校正
                })
        
        return {
            'test': 'Dunn (simplified)',
            'comparisons': comparisons
        }
    
    def _interpret_p_value(self, p_value):
        """解释p值"""
        if p_value < 0.001:
            return "极显著 (p < 0.001)"
        elif p_value < 0.01:
            return "高度显著 (p < 0.01)"
        elif p_value < 0.05:
            return "显著 (p < 0.05)"
        elif p_value < 0.1:
            return "边缘显著 (p < 0.1)"
        else:
            return "不显著 (p ≥ 0.1)"
    
    def _interpret_cohens_d(self, cohens_d):
        """解释Cohen's d效应大小"""
        abs_d = abs(cohens_d)
        
        if abs_d < 0.2:
            return "可忽略效应"
        elif abs_d < 0.5:
            return "小效应"
        elif abs_d < 0.8:
            return "中等效应"
        else:
            return "大效应"
    
    def _interpret_eta_squared(self, eta_squared):
        """解释eta squared效应大小"""
        if eta_squared < 0.01:
            return "可忽略效应"
        elif eta_squared < 0.06:
            return "小效应"
        elif eta_squared < 0.14:
            return "中等效应"
        else:
            return "大效应"
    
    def _print_comparison_results(self, result):
        """打印两组比较结果"""
        print(f"\n🔍 统计检验结果:")
        print(f"检验方法: {result['statistical_test']['name']}")
        print(f"检验统计量: {result['statistical_test']['statistic']:.4f}")
        print(f"p值: {result['statistical_test']['p_value']:.6f}")
        print(f"显著性: {result['statistical_test']['interpretation']}")
        
        print(f"\n📏 效应大小:")
        print(f"Cohen's d: {result['effect_size']['cohens_d']:.4f}")
        print(f"效应大小: {result['effect_size']['interpretation']}")
        
        print(f"\n📊 置信区间 ({result['confidence_interval']['confidence_level']*100:.0f}%):")
        print(f"均值差: {result['confidence_interval']['mean_difference']:.4f}")
        print(f"置信区间: [{result['confidence_interval']['ci_lower']:.4f}, {result['confidence_interval']['ci_upper']:.4f}]")
        
        print(f"\n📋 结论:")
        if result['statistical_test']['significant']:
            print(f"✅ {result['group1_name']}和{result['group2_name']}之间存在显著差异")
            if result['confidence_interval']['ci_lower'] > 0:
                print(f"   {result['group1_name']} 显著优于 {result['group2_name']}")
            elif result['confidence_interval']['ci_upper'] < 0:
                print(f"   {result['group2_name']} 显著优于 {result['group1_name']}")
        else:
            print(f"❌ {result['group1_name']}和{result['group2_name']}之间无显著差异")
    
    def _print_multiple_comparison_results(self, result):
        """打印多组比较结果"""
        print(f"\n🔍 多组统计检验结果:")
        print(f"检验方法: {result['statistical_test']['name']}")
        print(f"检验统计量: {result['statistical_test']['statistic']:.4f}")
        print(f"p值: {result['statistical_test']['p_value']:.6f}")
        print(f"显著性: {result['statistical_test']['interpretation']}")
        
        print(f"\n📏 效应大小:")
        print(f"Eta squared: {result['effect_size']['eta_squared']:.4f}")
        print(f"效应大小: {result['effect_size']['interpretation']}")
        
        if result['post_hoc']:
            print(f"\n🔍 事后检验:")
            if 'comparisons' in result['post_hoc']:
                for comp in result['post_hoc']['comparisons']:
                    sig_mark = "✅" if comp['significant'] else "❌"
                    print(f"  {comp['group1']} vs {comp['group2']}: p={comp['p_value']:.4f} {sig_mark}")
    
    def _print_paired_comparison_results(self, result):
        """打印配对比较结果"""
        print(f"\n🔍 配对统计检验结果:")
        print(f"检验方法: {result['statistical_test']['name']}")
        print(f"检验统计量: {result['statistical_test']['statistic']:.4f}")
        print(f"p值: {result['statistical_test']['p_value']:.6f}")
        print(f"显著性: {result['statistical_test']['interpretation']}")
        
        print(f"\n📏 效应大小:")
        print(f"Cohen's d: {result['effect_size']['cohens_d']:.4f}")
        print(f"效应大小: {result['effect_size']['interpretation']}")


def load_experiment_results(results_csv):
    """从CSV文件加载实验结果"""
    
    if not Path(results_csv).exists():
        raise FileNotFoundError(f"结果文件不存在: {results_csv}")
    
    df = pd.read_csv(results_csv)
    
    # 按实验类型分组
    results_by_type = {}
    
    if 'experiment_type' in df.columns and 'final_accuracy' in df.columns:
        for exp_type in df['experiment_type'].unique():
            if pd.notna(exp_type):
                type_df = df[df['experiment_type'] == exp_type]
                accuracies = type_df['final_accuracy'].dropna().values
                if len(accuracies) > 0:
                    results_by_type[exp_type] = accuracies
    
    return results_by_type


def main():
    parser = argparse.ArgumentParser(description="Flow实验结果统计分析")
    parser.add_argument('--results_file', type=str, required=True,
                       help='实验结果CSV文件路径')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='显著性水平')
    parser.add_argument('--output_dir', type=str, default='statistical_analysis',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建统计分析器
    analyzer = StatisticalAnalyzer(alpha=args.alpha)
    
    # 加载结果
    try:
        results = load_experiment_results(args.results_file)
        print(f"📁 已加载实验结果: {args.results_file}")
        print(f"发现实验类型: {list(results.keys())}")
        
    except Exception as e:
        print(f"❌ 加载结果失败: {e}")
        return
    
    if len(results) < 2:
        print("⚠️  需要至少2种实验类型进行比较分析")
        return
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 多组比较
    if len(results) > 2:
        print(f"\n🎯 进行多组比较分析...")
        multiple_result = analyzer.compare_multiple_groups(results)
        
        # 保存多组比较结果
        with open(output_dir / 'multiple_comparison.json', 'w') as f:
            json.dump(multiple_result, f, indent=2, default=str)
    
    # 两两比较
    methods = list(results.keys())
    pairwise_results = {}
    
    for i in range(len(methods)):
        for j in range(i+1, len(methods)):
            method1, method2 = methods[i], methods[j]
            
            print(f"\n🔍 两两比较: {method1} vs {method2}")
            comparison_result = analyzer.compare_two_groups(
                results[method1], results[method2], method1, method2
            )
            
            pairwise_results[f"{method1}_vs_{method2}"] = comparison_result
    
    # 保存两两比较结果
    with open(output_dir / 'pairwise_comparisons.json', 'w') as f:
        json.dump(pairwise_results, f, indent=2, default=str)
    
    # 生成汇总报告
    generate_analysis_report(results, multiple_result if len(results) > 2 else None, 
                           pairwise_results, output_dir)
    
    print(f"\n✅ 统计分析完成！结果保存在: {output_dir}")


def generate_analysis_report(results, multiple_result, pairwise_results, output_dir):
    """生成统计分析报告"""
    
    report_file = output_dir / 'statistical_analysis_report.md'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Flow实验统计分析报告\n\n")
        f.write(f"生成时间: {pd.Timestamp.now()}\n\n")
        
        # 数据摘要
        f.write("## 数据摘要\n\n")
        f.write("| 实验类型 | 样本数 | 均值 | 标准差 | 最小值 | 最大值 |\n")
        f.write("|----------|--------|------|--------|--------|--------|\n")
        
        for method_name, scores in results.items():
            f.write(f"| {method_name} | {len(scores)} | {np.mean(scores):.4f} | {np.std(scores, ddof=1):.4f} | {np.min(scores):.4f} | {np.max(scores):.4f} |\n")
        
        # 统计检验结果
        if multiple_result:
            f.write(f"\n## 多组比较结果\n\n")
            f.write(f"**检验方法:** {multiple_result['statistical_test']['name']}\n")
            f.write(f"**统计量:** {multiple_result['statistical_test']['statistic']:.4f}\n")
            f.write(f"**p值:** {multiple_result['statistical_test']['p_value']:.6f}\n")
            f.write(f"**显著性:** {multiple_result['statistical_test']['interpretation']}\n")
            f.write(f"**效应大小 (η²):** {multiple_result['effect_size']['eta_squared']:.4f} ({multiple_result['effect_size']['interpretation']})\n\n")
        
        # 两两比较
        f.write(f"## 两两比较结果\n\n")
        for comparison_name, result in pairwise_results.items():
            f.write(f"### {result['group1_name']} vs {result['group2_name']}\n\n")
            f.write(f"- **检验方法:** {result['statistical_test']['name']}\n")
            f.write(f"- **p值:** {result['statistical_test']['p_value']:.6f}\n")
            f.write(f"- **显著性:** {result['statistical_test']['interpretation']}\n")
            f.write(f"- **效应大小 (Cohen's d):** {result['effect_size']['cohens_d']:.4f} ({result['effect_size']['interpretation']})\n")
            f.write(f"- **95%置信区间:** [{result['confidence_interval']['ci_lower']:.4f}, {result['confidence_interval']['ci_upper']:.4f}]\n\n")
        
        f.write("## 结论建议\n\n")
        f.write("根据统计分析结果，建议在论文中报告以下内容：\n")
        f.write("1. 描述性统计 (均值、标准差、样本数)\n")
        f.write("2. 统计检验结果 (检验方法、p值、显著性)\n")
        f.write("3. 效应大小 (Cohen's d 或 η²)\n")
        f.write("4. 置信区间 (用于估计实际效应大小范围)\n")
        f.write("5. 实际显著性解释 (不仅仅依赖p值)\n")
    
    print(f"📄 分析报告已生成: {report_file}")


if __name__ == "__main__":
    main()