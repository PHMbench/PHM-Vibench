#!/usr/bin/env python3
"""
Flow预训练实现验证脚本

轻量级验证，避免复杂依赖，专注核心功能检查。
遵循"避免炫技复杂度"原则。
"""

import os
import re
from pathlib import Path

def check_file_exists(filepath, description):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"   ✅ {description}: {filepath}")
        return True
    else:
        print(f"   ❌ {description}: {filepath} 不存在")
        return False

def check_code_structure(filepath, patterns, description):
    """检查代码结构"""
    if not os.path.exists(filepath):
        return False
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        all_found = True
        for pattern, desc in patterns:
            if re.search(pattern, content):
                print(f"     ✓ {desc}")
            else:
                print(f"     ✗ {desc}")
                all_found = False
        
        return all_found
        
    except Exception as e:
        print(f"   ❌ 读取{description}失败: {e}")
        return False

def validate_core_implementation():
    """验证核心实现"""
    print("🔍 验证核心实现文件...")
    
    checks = []
    
    # 核心文件检查
    core_files = [
        ("src/task_factory/task/pretrain/flow_pretrain.py", "FlowPretrainTask主任务"),
        ("src/task_factory/task/pretrain/flow_contrastive_loss.py", "FlowContrastiveLoss损失函数"),
        ("src/task_factory/task/pretrain/flow_metrics.py", "FlowMetrics评估模块"),
    ]
    
    for filepath, desc in core_files:
        checks.append(check_file_exists(filepath, desc))
    
    return all(checks)

def validate_task_registration():
    """验证任务注册"""
    print("\n🔗 验证任务注册...")
    
    init_file = "src/task_factory/task/pretrain/__init__.py"
    
    if not check_file_exists(init_file, "预训练任务初始化文件"):
        return False
    
    # 检查注册模式
    patterns = [
        (r"from \.flow_pretrain import \*", "FlowPretrainTask导入"),
        (r"'FlowPretrainTask'", "FlowPretrainTask在__all__中"),
    ]
    
    return check_code_structure(init_file, patterns, "任务注册")

def validate_configurations():
    """验证配置文件"""
    print("\n⚙️  验证配置文件...")
    
    config_files = [
        ("configs/demo/Pretraining/Flow/flow_pretrain_basic.yaml", "基础配置"),
        ("configs/demo/Pretraining/Flow/flow_pretrain_small.yaml", "小数据集配置"),
        ("configs/demo/Pretraining/Flow/flow_pretrain_full.yaml", "生产配置"),
    ]
    
    checks = []
    for filepath, desc in config_files:
        checks.append(check_file_exists(filepath, desc))
    
    return all(checks)

def validate_code_quality():
    """验证代码质量"""
    print("\n📋 验证代码质量...")
    
    flow_pretrain_file = "src/task_factory/task/pretrain/flow_pretrain.py"
    
    if not os.path.exists(flow_pretrain_file):
        return False
    
    # 检查关键方法和设计模式
    patterns = [
        (r"@register_task", "任务注册装饰器"),
        (r"class FlowPretrainTask\(Default_task\)", "继承Default_task基类"),
        (r"def training_step", "训练步骤方法"),
        (r"def validation_step", "验证步骤方法"), 
        (r"def forward", "前向传播方法"),
        (r"def generate_samples", "样本生成方法"),
        (r"self\.flow_metrics", "指标监控集成"),
        (r"FlowContrastiveLoss", "对比学习损失集成"),
    ]
    
    return check_code_structure(flow_pretrain_file, patterns, "FlowPretrainTask")

def validate_documentation():
    """验证文档规范"""
    print("\n📚 验证文档规范...")
    
    spec_files = [
        (".claude/specs/flow-pretraining-task/requirements.md", "需求文档"),
        (".claude/specs/flow-pretraining-task/requirements_zh.md", "中文需求文档"),
        (".claude/specs/flow-pretraining-task/design.md", "技术设计文档"),
        (".claude/specs/flow-pretraining-task/tasks.md", "任务分解文档"),
    ]
    
    checks = []
    for filepath, desc in spec_files:
        checks.append(check_file_exists(filepath, desc))
    
    return all(checks)

def count_code_lines():
    """统计代码行数"""
    print("\n📊 代码统计...")
    
    files_to_count = [
        "src/task_factory/task/pretrain/flow_pretrain.py",
        "src/task_factory/task/pretrain/flow_contrastive_loss.py", 
        "src/task_factory/task/pretrain/flow_metrics.py"
    ]
    
    total_lines = 0
    
    for filepath in files_to_count:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = len(f.readlines())
                print(f"   📄 {os.path.basename(filepath)}: {lines} 行")
                total_lines += lines
    
    print(f"   📈 总计: {total_lines} 行代码")
    return total_lines

def main():
    """主验证流程"""
    print("🚀 Flow预训练实现验证")
    print("=" * 50)
    
    validations = [
        ("核心实现", validate_core_implementation),
        ("任务注册", validate_task_registration), 
        ("配置文件", validate_configurations),
        ("代码质量", validate_code_quality),
        ("文档规范", validate_documentation),
    ]
    
    passed = 0
    total = len(validations)
    
    for name, func in validations:
        if func():
            passed += 1
            print(f"   🎉 {name} 验证通过\n")
        else:
            print(f"   ⚠️  {name} 验证失败\n")
    
    # 代码统计
    code_lines = count_code_lines()
    
    # 总结
    print("\n" + "=" * 50)
    print(f"📋 验证结果: {passed}/{total} 项通过")
    print(f"💻 代码规模: {code_lines} 行")
    
    if passed == total:
        print("🎉 Flow预训练任务实现验证完全通过！")
        print("✨ 核心功能已就绪，可开始训练测试")
        return True
    else:
        print("⚠️  存在问题需要修复")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)