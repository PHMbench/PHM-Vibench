#!/usr/bin/env python3
"""
ContrastiveIDTask研究流程测试运行器
统一管理和执行所有测试套件
"""

import sys
import os
import time
import argparse
from pathlib import Path

# 添加项目路径
script_dir = Path(__file__).parent
sys.path.append(str(script_dir.parents[3]))

import warnings
warnings.filterwarnings("ignore")

# 导入测试套件
from unit_tests import run_all_tests as run_unit_tests
from integration_tests import run_integration_tests
from performance_tests import run_performance_tests


class TestSuiteRunner:
    """测试套件运行器"""

    def __init__(self):
        self.results = {}
        self.start_time = None
        self.total_time = None

    def print_banner(self, title, char="=", width=70):
        """打印横幅"""
        print(f"\n{char * width}")
        print(f"{title:^{width}}")
        print(f"{char * width}")

    def run_suite(self, suite_name, suite_function, required=True):
        """运行单个测试套件"""
        self.print_banner(f"🧪 {suite_name} 测试套件", char="-", width=50)

        start_time = time.time()

        try:
            success = suite_function()
            end_time = time.time()
            duration = end_time - start_time

            status = "✅ 通过" if success else "❌ 失败"
            self.results[suite_name] = {
                'success': success,
                'duration': duration,
                'status': status,
                'required': required
            }

            print(f"\n{status} - {suite_name} ({duration:.2f}s)")
            return success

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time

            self.results[suite_name] = {
                'success': False,
                'duration': duration,
                'status': f"❌ 异常: {e}",
                'required': required,
                'exception': str(e)
            }

            print(f"\n❌ 异常 - {suite_name}: {e}")
            return False

    def check_environment(self):
        """检查测试环境"""
        print("🔍 检查测试环境...")

        checks = []

        # Python版本
        python_version = sys.version_info
        py_ok = python_version >= (3, 7)
        checks.append(("Python >= 3.7", f"{python_version.major}.{python_version.minor}", py_ok))

        # PyTorch
        try:
            import torch
            torch_ok = True
            torch_version = torch.__version__
        except ImportError:
            torch_ok = False
            torch_version = "未安装"
        checks.append(("PyTorch", torch_version, torch_ok))

        # NumPy
        try:
            import numpy as np
            numpy_ok = True
            numpy_version = np.__version__
        except ImportError:
            numpy_ok = False
            numpy_version = "未安装"
        checks.append(("NumPy", numpy_version, numpy_ok))

        # CUDA
        if torch_ok:
            cuda_available = torch.cuda.is_available()
            cuda_info = f"可用 ({torch.cuda.get_device_name(0)})" if cuda_available else "不可用"
        else:
            cuda_available = False
            cuda_info = "PyTorch未安装"
        checks.append(("CUDA", cuda_info, True))  # CUDA不是必需的

        # 打印检查结果
        print("\n环境检查结果:")
        print("-" * 40)
        all_required_ok = True
        for name, info, ok in checks:
            status = "✅" if ok else "❌"
            print(f"{status} {name}: {info}")
            if name in ["Python >= 3.7", "PyTorch", "NumPy"] and not ok:
                all_required_ok = False

        return all_required_ok

    def generate_report(self):
        """生成测试报告"""
        self.print_banner("📊 测试报告", char="=")

        print(f"📅 测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ 总用时: {self.total_time:.2f}秒")
        print()

        # 详细结果
        print("📋 详细结果:")
        print("-" * 50)
        total_suites = len(self.results)
        passed_suites = 0
        required_failed = 0

        for suite_name, result in self.results.items():
            duration = result['duration']
            status = result['status']
            required = result['required']
            req_str = " (必需)" if required else " (可选)"

            print(f"{status} {suite_name}{req_str} - {duration:.2f}s")

            if result['success']:
                passed_suites += 1
            elif required:
                required_failed += 1

            # 如果有异常，显示详细信息
            if 'exception' in result:
                print(f"   💥 异常详情: {result['exception']}")

        # 总结
        print("-" * 50)
        success_rate = passed_suites / total_suites * 100 if total_suites > 0 else 0
        print(f"📈 成功率: {passed_suites}/{total_suites} ({success_rate:.1f}%)")

        # 判断整体结果
        if required_failed == 0:
            overall_status = "✅ 总体通过"
            overall_color = "🟢"
        else:
            overall_status = f"❌ {required_failed}个必需套件失败"
            overall_color = "🔴"

        print(f"{overall_color} {overall_status}")

        return required_failed == 0

    def run_all_tests(self, include_performance=True, include_integration=True):
        """运行所有测试套件"""
        self.print_banner("🚀 ContrastiveIDTask 研究流程测试套件")

        self.start_time = time.time()

        # 环境检查
        if not self.check_environment():
            print("❌ 环境检查失败，无法继续测试")
            return False

        print(f"\n🎯 测试计划:")
        print(f"  • 单元测试 (必需)")
        if include_integration:
            print(f"  • 集成测试 (必需)")
        if include_performance:
            print(f"  • 性能测试 (可选)")

        # 运行测试套件
        suite_configs = [
            ("单元测试", run_unit_tests, True),
        ]

        if include_integration:
            suite_configs.append(("集成测试", run_integration_tests, True))

        if include_performance:
            suite_configs.append(("性能测试", run_performance_tests, False))

        # 执行所有套件
        overall_success = True
        for suite_name, suite_func, required in suite_configs:
            success = self.run_suite(suite_name, suite_func, required)
            if required and not success:
                overall_success = False

        self.total_time = time.time() - self.start_time

        # 生成报告
        report_success = self.generate_report()

        return report_success and overall_success

    def run_specific_suite(self, suite_name):
        """运行特定测试套件"""
        suite_map = {
            'unit': ("单元测试", run_unit_tests),
            'integration': ("集成测试", run_integration_tests),
            'performance': ("性能测试", run_performance_tests),
        }

        if suite_name not in suite_map:
            print(f"❌ 未知的测试套件: {suite_name}")
            print(f"可用套件: {', '.join(suite_map.keys())}")
            return False

        display_name, suite_func = suite_map[suite_name]

        self.print_banner(f"🎯 运行 {display_name}")
        self.start_time = time.time()

        success = self.run_suite(display_name, suite_func, True)
        self.total_time = time.time() - self.start_time

        print(f"\n📊 {display_name} 结果: {'✅ 通过' if success else '❌ 失败'}")
        print(f"⏱️ 用时: {self.total_time:.2f}秒")

        return success


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="ContrastiveIDTask研究流程测试运行器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_tests.py                    # 运行所有测试
  python run_tests.py --fast             # 快速测试（跳过性能测试）
  python run_tests.py --suite unit       # 只运行单元测试
  python run_tests.py --suite performance # 只运行性能测试
  python run_tests.py --no-integration   # 跳过集成测试
        """
    )

    parser.add_argument(
        '--suite',
        choices=['unit', 'integration', 'performance'],
        help='运行特定测试套件'
    )

    parser.add_argument(
        '--fast',
        action='store_true',
        help='快速模式，跳过性能测试'
    )

    parser.add_argument(
        '--no-integration',
        action='store_true',
        help='跳过集成测试'
    )

    parser.add_argument(
        '--no-performance',
        action='store_true',
        help='跳过性能测试'
    )

    args = parser.parse_args()

    runner = TestSuiteRunner()

    try:
        if args.suite:
            # 运行特定套件
            success = runner.run_specific_suite(args.suite)
        else:
            # 运行所有或部分套件
            include_performance = not (args.fast or args.no_performance)
            include_integration = not args.no_integration

            success = runner.run_all_tests(
                include_performance=include_performance,
                include_integration=include_integration
            )

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⚠️ 用户中断测试")
        return 130
    except Exception as e:
        print(f"\n💥 测试运行器异常: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())