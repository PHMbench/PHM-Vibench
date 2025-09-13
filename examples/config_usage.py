#!/usr/bin/env python3
"""
PHM-Vibench配置系统使用示例
===============================

这个文件展示了如何使用新的Pydantic配置系统：
- 🚀 从20+行YAML减少到5行Python代码
- ✅ 类型安全和IDE智能提示
- 🔧 配置继承和组合
- 📝 自动验证和错误提示

运行方式:
    cd /home/lq/LQcode/2_project/PHMBench/PHM-Vibench
    python examples/config_usage.py

作者: PHM-Vibench Team
日期: 2024-12-20
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.configs import PHMConfig, load_config, create_config
from src.configs.config_manager import ConfigManager

def main():
    print("🎯 PHM-Vibench配置系统使用示例")
    print("=" * 50)
    
    # ===========================================
    # 示例1: 快速创建基础配置 (替代110+行YAML)
    # ===========================================
    print("\n📋 示例1: 创建基础配置")
    print("-" * 30)
    
    # 老方式: 需要110+行YAML文件
    print("❌ 老方式需要110+行YAML配置文件")
    
    # 新方式: 5行搞定！
    print("✅ 新方式: 仅需5行代码!")
    config = PHMConfig(
        data__data_dir="./data",
        model__name="ResNet1D", 
        model__type="CNN",
        task__name="classification",
        trainer__num_epochs=50
    )
    print(f"  实验名称: {config.environment.experiment_name}")
    print(f"  数据目录: {config.data.data_dir}")
    print(f"  模型类型: {config.model.type}.{config.model.name}")
    print(f"  任务类型: {config.task.name}")
    print(f"  训练轮数: {config.trainer.num_epochs}")
    
    # ===========================================
    # 示例2: 使用预设配置 (快速启动)
    # ===========================================
    print("\n🚀 示例2: 使用预设配置")
    print("-" * 30)
    
    # 直接使用预设，秒速配置！
    quickstart_config = load_config("quickstart")
    print(f"  快速配置: {quickstart_config.environment.experiment_name}")
    print(f"  推荐新手: ResNet1D + CWRU数据")
    
    # ISFM高级配置
    isfm_config = load_config("isfm")  
    print(f"  高级配置: {isfm_config.model.name}")
    print(f"  研究专用: Transformer + 多数据集")
    
    # 生产环境配置
    production_config = load_config("production")
    print(f"  生产配置: {production_config.environment.project}")
    print(f"  稳定可靠: 优化的超参数")
    
    # ===========================================
    # 示例3: 配置继承和组合 (强大功能)
    # ===========================================
    print("\n🔧 示例3: 配置继承和组合")
    print("-" * 30)
    
    # 基于基础配置进行定制
    custom_config = load_config("basic", {
        "environment": {
            "experiment_name": "我的自定义实验", 
            "project": "PHM研究项目"
        },
        "model": {
            "d_model": 256,  # 增大模型维度
            "num_heads": 8   # 增加注意力头数
        },
        "task": {
            "epochs": 100,   # 延长训练
            "lr": 0.0005     # 降低学习率  
        }
    })
    print(f"  定制配置: {custom_config.environment.experiment_name}")
    print(f"  模型维度: {custom_config.model.d_model}")
    print(f"  学习率: {custom_config.task.lr}")
    
    # ===========================================
    # 示例4: 配置管理器 (企业级功能)
    # ===========================================
    print("\n🛠️ 示例4: 配置管理器")
    print("-" * 30)
    
    manager = ConfigManager()
    
    # 加载和保存
    config = manager.load("research")
    manager.save(config, project_root / "temp_config.yaml", minimal=True)
    print("  ✅ 配置已保存到 temp_config.yaml")
    
    # 配置比较
    config1 = manager.load("quickstart")
    config2 = manager.load("isfm")  
    diff = manager.compare(config1, config2)
    print(f"  🔍 两个配置共有 {diff['total_differences']} 处差异")
    print(f"  📊 统计: 新增{diff['summary']['added']}, 修改{diff['summary']['modified']}, 删除{diff['summary']['removed']}")
    
    # 配置验证
    is_valid, errors, warnings = manager.validate(config)
    print(f"  ✅ 配置验证: {'通过' if is_valid else '失败'}")
    if warnings:
        print(f"  ⚠️  警告数量: {len(warnings)}")
    
    # ===========================================
    # 示例5: 从YAML迁移 (向后兼容)
    # ===========================================
    print("\n🔄 示例5: 从YAML迁移")
    print("-" * 30)
    
    # 尝试加载现有的YAML配置
    yaml_config_path = project_root / "configs/demo/Multiple_DG/CWRU_THU_using_ISFM.yaml"
    if yaml_config_path.exists():
        try:
            legacy_config = manager.load(yaml_config_path)
            print(f"  ✅ 成功加载YAML配置: {legacy_config.model.name}")
            print(f"  🎯 目标系统: {legacy_config.task.target_system_id}")
            
            # 转换为新格式保存
            manager.save(legacy_config, project_root / "migrated_config.py", format="py")
            print("  🔄 已转换为Python格式保存")
            
        except Exception as e:
            print(f"  ❌ YAML配置加载失败: {e}")
    else:
        print("  ℹ️  未找到示例YAML配置文件")
    
    # ===========================================
    # 示例6: IDE智能提示演示 (开发者福利)
    # ===========================================
    print("\n💡 示例6: IDE智能提示")
    print("-" * 30)
    
    config = PHMConfig()
    # IDE会自动提示所有可用选项！
    print("  📝 IDE自动补全:")
    print("    config.model.  -> name, type, d_model, num_heads...")
    print("    config.data.   -> data_dir, batch_size, num_workers...")
    print("    config.task.   -> name, type, epochs, lr...")
    print("    config.trainer.-> num_epochs, gpus, device...")
    
    # 类型检查
    print("  ✅ 类型安全:")
    print("    config.trainer.num_epochs = 50     # ✅ 正确")
    print("    config.trainer.num_epochs = '50'   # ❌ 类型错误")
    
    # ===========================================
    # 示例7: 实验配置最佳实践
    # ===========================================
    print("\n🎓 示例7: 实验配置最佳实践")
    print("-" * 30)
    
    print("  快速原型开发:")
    print("    config = load_config('quickstart')")
    
    print("  深入研究:")  
    print("    config = load_config('research', {'model__d_model': 512})")
    
    print("  生产部署:")
    print("    config = load_config('production', {'trainer__gpus': 4})")
    
    print("  消融实验:")
    print("    for lr in [0.001, 0.0005, 0.0001]:")
    print("        config = load_config('isfm', {'task__lr': lr})")
    
    print("  多数据集验证:")
    print("    config = load_config('benchmark', {'task__target_system_id': [1,2,3]})")
    
    # ===========================================
    # 总结
    # ===========================================
    print("\n🎉 配置系统优势总结")
    print("=" * 50)
    
    print("✨ 效率提升:")
    print("  • 从110+行YAML → 5行Python")  
    print("  • 预设配置秒速启动")
    print("  • 智能默认值管理")
    
    print("🛡️ 可靠性:")
    print("  • 类型安全，运行前发现错误")
    print("  • 自动验证配置完整性") 
    print("  • IDE智能提示避免拼写错误")
    
    print("🔧 灵活性:")
    print("  • 配置继承和组合")
    print("  • 支持Python/YAML/JSON")
    print("  • 向后兼容现有配置")
    
    print("🚀 企业级:")
    print("  • 配置版本管理")
    print("  • 差异分析比较")
    print("  • 自动化配置生成")
    
    print("\n💡 开始使用:")
    print("  1. 快速上手: load_config('quickstart')")
    print("  2. 研究实验: load_config('isfm')")  
    print("  3. 自定义配置: PHMConfig(model__name='YourModel')")
    
    # 清理临时文件
    temp_files = [
        project_root / "temp_config.yaml",
        project_root / "migrated_config.py",
        project_root / "example_config.yaml", 
        project_root / "example_config.py"
    ]
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()

if __name__ == "__main__":
    main()