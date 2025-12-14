#!/usr/bin/env python3
"""
快速分析和triage扫描结果
使用方法：python docs/LQ_fix_12_14/quick_triage.py
"""

import os
import re
from collections import defaultdict, Counter
from datetime import datetime

def load_scan_results(scan_dir):
    """加载扫描结果"""
    results = {}

    # 加载各类扫描结果
    scan_files = {
        'todo': 'rg_todo_fixme_hack.txt',
        'except_bare': 'rg_except_bare.txt',
        'except_exception': 'rg_except_exception.txt',
        'except_baseexception': 'rg_except_baseexception.txt',
        'except_pass': 'rg_except_pass.txt',
        'assert_raise': 'rg_assert_raise.txt',
        'paths': 'rg_paths_envs.txt'
    }

    for category, filename in scan_files.items():
        filepath = os.path.join(scan_dir, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[category] = [line.strip() for line in f if line.strip()]
        else:
            results[category] = []

    return results

def analyze_priority(items, category):
    """分析项目优先级"""
    priority_items = {
        'P0': [],  # 致命
        'P1': [],  # 严重
        'P2': [],  # 中等
        'P3': []   # 轻微
    }

    for item in items:
        # 解析文件路径和行号
        if ':' in item:
            parts = item.split(':', 2)
            if len(parts) >= 2:
                filepath = parts[0]
                line_num = parts[1]
                content = parts[2] if len(parts) > 2 else ""
        else:
            continue

        # 根据类别和内容判断优先级
        priority = 'P3'  # 默认轻微

        if category in ['except_bare', 'except_baseexception']:
            # 裸except是高风险
            priority = 'P0' if 'src/' in filepath else 'P1'

        elif category == 'except_pass':
            # 静默忽略异常
            if 'load_state_dict' in content or 'torch.load' in content:
                priority = 'P0'  # 加载模型失败但不报错
            elif 'src/data_factory' in filepath:
                priority = 'P1'  # 数据加载失败
            else:
                priority = 'P2'

        elif category in ['todo', 'FIXME']:
            if 'FIXME' in item:
                priority = 'P1' if 'src/' in filepath else 'P2'
            elif 'TODO' in item and ('implement' in content.lower() or 'fix' in content.lower()):
                priority = 'P2'

        elif category == 'paths':
            if '/home/' in item:
                priority = 'P1'  # 硬编码绝对路径
            else:
                priority = 'P2'

        elif category == 'assert_raise':
            if 'assert' in item and ('user' in content or 'config' in content):
                priority = 'P1'  # 用户输入可能触发断言
            else:
                priority = 'P2'

        priority_items[priority].append({
            'file': filepath,
            'line': line_num,
            'content': content,
            'full': item
        })

    return priority_items

def generate_triage_report(results, output_dir):
    """生成triage报告"""
    report = []
    report.append("# 快速Triage报告\n")
    report.append(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 统计总数
    total_items = sum(len(items) for items in results.values())
    report.append(f"## 扫描结果统计\n")
    report.append(f"- 总计发现 {total_items} 个潜在问题\n")

    for category, items in results.items():
        if items:
            report.append(f"- {category}: {len(items)} 条\n")

    report.append("\n## 高优先级问题（需要立即关注）\n\n")

    # 分析并排序所有问题
    all_prioritized = {
        'P0': [],
        'P1': [],
        'P2': [],
        'P3': []
    }

    for category, items in results.items():
        prioritized = analyze_priority(items, category)
        for priority in ['P0', 'P1', 'P2', 'P3']:
            all_prioritized[priority].extend([(category, item) for item in prioritized[priority]])

    # 输出P0和P1问题
    for priority in ['P0', 'P1']:
        items = all_prioritized[priority]
        if items:
            priority_name = {'P0': '致命', 'P1': '严重'}[priority]
            report.append(f"### {priority} - {priority_name}级问题 ({len(items)}个)\n\n")

            for category, item in items[:10]:  # 只显示前10个
                report.append(f"**{category}** - `{item['file']}:{item['line']}`\n")
                report.append(f"```python\n{item['content'][:200]}\n```\n\n")

            if len(items) > 10:
                report.append(f"...还有 {len(items) - 10} 个{priority_name}问题\n\n")

    # 生成详细检查清单
    report.append("\n## 详细检查清单\n\n")

    for category, items in results.items():
        if items:
            report.append(f"### {category} ({len(items)}条)\n\n")

            # 按文件分组
            by_file = defaultdict(list)
            for item in items:
                if ':' in item:
                    parts = item.split(':', 2)
                    if len(parts) >= 2:
                        filepath = parts[0]
                        by_file[filepath].append(item)

            for filepath, file_items in sorted(by_file.items()):
                report.append(f"**文件：{filepath}**\n")
                for item in file_items[:5]:  # 每个文件最多显示5条
                    report.append(f"  - Line {item.split(':')[1]}: {item[:100]}...\n")
                report.append("\n")

    # 保存报告
    report_path = os.path.join(output_dir, 'quick_triage_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))

    return report_path

def main():
    """主函数"""
    # 目录设置
    base_dir = "/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench"
    scan_dir = os.path.join(base_dir, "docs/LQ_fix/12_14/reports/scan_logs")
    output_dir = os.path.join(base_dir, "docs/LQ_fix/12_14/reports")

    print("=== PHM-Vibench 快速Triage分析 ===\n")

    # 加载扫描结果
    print("1. 加载扫描结果...")
    results = load_scan_results(scan_dir)

    # 生成报告
    print("2. 生成triage报告...")
    report_path = generate_triage_report(results, output_dir)

    # 输出摘要
    print("\n3. 分析摘要：")
    total = sum(len(items) for items in results.values())
    print(f"   - 总计发现 {total} 个潜在问题")

    # 统计P0/P1
    p0_count = 0
    p1_count = 0
    for category, items in results.items():
        prioritized = analyze_priority(items, category)
        p0_count += len(prioritized['P0'])
        p1_count += len(prioritized['P1'])

    print(f"   - P0（致命）：{p0_count} 个")
    print(f"   - P1（严重）：{p1_count} 个")
    print(f"   - 详细报告：{report_path}")

    # 建议下一步
    print("\n4. 建议下一步操作：")
    if p0_count > 0:
        print("   - 立即检查P0级问题（可能导致系统崩溃）")
    if p1_count > 0:
        print("   - 优先处理P1级问题（影响核心功能）")
    print("   - 查看完整报告了解所有问题")
    print("   - 对每个问题进行深入调查并创建Bug记录")

if __name__ == "__main__":
    main()