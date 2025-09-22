#!/usr/bin/env python3
"""
ContrastiveIDTask环境检查工具

快速诊断和验证ContrastiveIDTask运行环境的完整性，包括：
- Python环境和依赖包检查
- PyTorch和CUDA环境验证
- PHM-Vibench组件可用性测试
- 数据路径和配置文件检查
- 系统资源评估

Usage:
    # 快速环境检查
    python environment_check.py

    # 详细检查包含数据验证
    python environment_check.py --detailed

    # 自动修复常见问题
    python environment_check.py --fix

Author: PHM-Vibench Team
Version: 1.0 (Research Environment Validator)
"""

import os
import sys
import subprocess
import importlib
import platform
from pathlib import Path
import argparse
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

class EnvironmentChecker:
    """环境检查器

    提供全面的开发环境诊断，确保ContrastiveIDTask能够正常运行。
    """

    def __init__(self, detailed: bool = False, auto_fix: bool = False):
        self.detailed = detailed
        self.auto_fix = auto_fix
        self.check_results = []
        self.warnings = []
        self.errors = []

        # 必需的依赖包
        self.required_packages = {
            'torch': '>=2.0.0',
            'numpy': '>=1.20.0',
            'pandas': '>=1.3.0',
            'matplotlib': '>=3.5.0',
            'seaborn': '>=0.11.0',
            'scipy': '>=1.7.0',
            'scikit-learn': '>=1.0.0',
            'h5py': '>=3.1.0',
            'openpyxl': '>=3.0.0',
            'PyYAML': '>=6.0',
            'tqdm': '>=4.60.0',
            'psutil': '>=5.8.0'
        }

        # 可选依赖包
        self.optional_packages = {
            'pytorch-lightning': '>=1.8.0',
            'wandb': '>=0.12.0',
            'tensorboard': '>=2.8.0',
            'plotly': '>=5.0.0',
            'memory_profiler': '>=0.60.0'
        }

        # PHM-Vibench核心组件路径
        self.phm_components = [
            'src/configs/__init__.py',
            'src/data_factory/__init__.py',
            'src/model_factory/__init__.py',
            'src/task_factory/__init__.py',
            'src/trainer_factory/__init__.py',
            'src/task_factory/task/pretrain/ContrastiveIDTask.py'
        ]

        # 默认配置文件
        self.config_files = [
            'configs/id_contrastive/debug.yaml',
            'configs/id_contrastive/production.yaml',
            'configs/id_contrastive/ablation.yaml'
        ]

        print("🔍 ContrastiveIDTask环境检查工具")
        print("=" * 60)

    def check_python_environment(self) -> Dict[str, Any]:
        """检查Python环境"""
        print("📋 检查Python环境...")

        result = {
            'category': 'Python Environment',
            'checks': []
        }

        # Python版本检查
        python_version = sys.version_info
        python_check = {
            'name': 'Python版本',
            'status': 'pass' if python_version >= (3, 8) else 'fail',
            'details': f'{python_version.major}.{python_version.minor}.{python_version.micro}',
            'recommendation': 'Python 3.8+是推荐版本' if python_version >= (3, 8) else '请升级到Python 3.8或更高版本'
        }
        result['checks'].append(python_check)

        # 系统平台检查
        platform_info = {
            'name': '系统平台',
            'status': 'info',
            'details': f'{platform.system()} {platform.release()} ({platform.machine()})',
            'recommendation': '支持Linux, Windows, macOS'
        }
        result['checks'].append(platform_info)

        # 虚拟环境检查
        venv_active = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
        venv_check = {
            'name': '虚拟环境',
            'status': 'pass' if venv_active else 'warning',
            'details': '已激活' if venv_active else '未使用虚拟环境',
            'recommendation': '建议使用虚拟环境以避免依赖冲突'
        }
        result['checks'].append(venv_check)

        return result

    def check_required_packages(self) -> Dict[str, Any]:
        """检查必需的依赖包"""
        print("📦 检查必需依赖包...")

        result = {
            'category': 'Required Packages',
            'checks': []
        }

        for package_name, min_version in self.required_packages.items():
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')

                # 简单的版本比较（实际项目中应使用packaging库）
                status = 'pass'
                if version != 'unknown' and min_version.startswith('>='):
                    required_version = min_version[2:]
                    if self._compare_versions(version, required_version) < 0:
                        status = 'fail'

                check = {
                    'name': package_name,
                    'status': status,
                    'details': f'版本 {version}',
                    'recommendation': f'需要 {min_version}' if status == 'fail' else '✓'
                }

            except ImportError:
                check = {
                    'name': package_name,
                    'status': 'fail',
                    'details': '未安装',
                    'recommendation': f'请安装: pip install {package_name}'
                }

            result['checks'].append(check)

        return result

    def check_pytorch_cuda(self) -> Dict[str, Any]:
        """检查PyTorch和CUDA环境"""
        print("🔥 检查PyTorch和CUDA环境...")

        result = {
            'category': 'PyTorch & CUDA',
            'checks': []
        }

        try:
            import torch

            # PyTorch版本检查
            torch_check = {
                'name': 'PyTorch版本',
                'status': 'pass',
                'details': torch.__version__,
                'recommendation': '✓'
            }
            result['checks'].append(torch_check)

            # CUDA可用性检查
            cuda_available = torch.cuda.is_available()
            cuda_check = {
                'name': 'CUDA可用性',
                'status': 'pass' if cuda_available else 'warning',
                'details': '可用' if cuda_available else '不可用',
                'recommendation': 'GPU训练需要CUDA支持' if not cuda_available else '✓'
            }
            result['checks'].append(cuda_check)

            # GPU信息检查
            if cuda_available:
                gpu_count = torch.cuda.device_count()
                gpu_info = []

                for i in range(gpu_count):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info.append(f'GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)')

                gpu_check = {
                    'name': 'GPU信息',
                    'status': 'pass',
                    'details': '; '.join(gpu_info),
                    'recommendation': '✓'
                }
                result['checks'].append(gpu_check)

            # 简单的张量运算测试
            try:
                device = torch.device('cuda' if cuda_available else 'cpu')
                test_tensor = torch.randn(10, 10, device=device)
                test_result = torch.mm(test_tensor, test_tensor.t())

                tensor_check = {
                    'name': 'PyTorch功能测试',
                    'status': 'pass',
                    'details': f'在{device}上成功执行张量运算',
                    'recommendation': '✓'
                }
                result['checks'].append(tensor_check)

            except Exception as e:
                tensor_check = {
                    'name': 'PyTorch功能测试',
                    'status': 'fail',
                    'details': f'错误: {str(e)}',
                    'recommendation': '检查PyTorch安装'
                }
                result['checks'].append(tensor_check)

        except ImportError:
            torch_check = {
                'name': 'PyTorch',
                'status': 'fail',
                'details': '未安装',
                'recommendation': '请安装PyTorch: pip install torch'
            }
            result['checks'].append(torch_check)

        return result

    def check_phm_vibench_components(self) -> Dict[str, Any]:
        """检查PHM-Vibench组件"""
        print("⚙️ 检查PHM-Vibench组件...")

        result = {
            'category': 'PHM-Vibench Components',
            'checks': []
        }

        project_root = Path(__file__).parent.parent.parent.parent

        # 检查核心组件文件
        for component_path in self.phm_components:
            full_path = project_root / component_path
            component_check = {
                'name': component_path,
                'status': 'pass' if full_path.exists() else 'fail',
                'details': '存在' if full_path.exists() else '缺失',
                'recommendation': '✓' if full_path.exists() else f'检查文件: {full_path}'
            }
            result['checks'].append(component_check)

        # 尝试导入核心模块
        try:
            from src.configs import load_config
            config_check = {
                'name': '配置系统导入',
                'status': 'pass',
                'details': '成功',
                'recommendation': '✓'
            }
            result['checks'].append(config_check)

            # 测试配置加载
            try:
                test_config = {
                    'data': {'batch_size': 32},
                    'model': {'d_model': 128},
                    'task': {'name': 'test'}
                }
                from src.configs.config_utils import ConfigWrapper
                config_obj = ConfigWrapper(test_config)

                config_test_check = {
                    'name': '配置加载测试',
                    'status': 'pass',
                    'details': '配置系统正常工作',
                    'recommendation': '✓'
                }
                result['checks'].append(config_test_check)

            except Exception as e:
                config_test_check = {
                    'name': '配置加载测试',
                    'status': 'fail',
                    'details': f'错误: {str(e)}',
                    'recommendation': '检查配置系统实现'
                }
                result['checks'].append(config_test_check)

        except ImportError as e:
            config_check = {
                'name': '配置系统导入',
                'status': 'fail',
                'details': f'导入失败: {str(e)}',
                'recommendation': '检查PYTHONPATH和项目结构'
            }
            result['checks'].append(config_check)

        # 尝试导入ContrastiveIDTask
        try:
            from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask
            task_check = {
                'name': 'ContrastiveIDTask导入',
                'status': 'pass',
                'details': '成功',
                'recommendation': '✓'
            }
            result['checks'].append(task_check)

        except ImportError as e:
            task_check = {
                'name': 'ContrastiveIDTask导入',
                'status': 'fail',
                'details': f'导入失败: {str(e)}',
                'recommendation': '检查ContrastiveIDTask实现'
            }
            result['checks'].append(task_check)

        return result

    def check_configuration_files(self) -> Dict[str, Any]:
        """检查配置文件"""
        print("📁 检查配置文件...")

        result = {
            'category': 'Configuration Files',
            'checks': []
        }

        project_root = Path(__file__).parent.parent.parent.parent

        for config_path in self.config_files:
            full_path = project_root / config_path

            if full_path.exists():
                # 尝试加载YAML文件
                try:
                    import yaml
                    with open(full_path, 'r') as f:
                        config_data = yaml.safe_load(f)

                    # 检查关键字段
                    required_keys = ['data', 'model', 'task']
                    missing_keys = [key for key in required_keys if key not in config_data]

                    if not missing_keys:
                        config_check = {
                            'name': config_path,
                            'status': 'pass',
                            'details': '配置完整',
                            'recommendation': '✓'
                        }
                    else:
                        config_check = {
                            'name': config_path,
                            'status': 'warning',
                            'details': f'缺少键: {missing_keys}',
                            'recommendation': '补充缺失的配置项'
                        }

                except Exception as e:
                    config_check = {
                        'name': config_path,
                        'status': 'fail',
                        'details': f'解析错误: {str(e)}',
                        'recommendation': '检查YAML语法'
                    }

            else:
                config_check = {
                    'name': config_path,
                    'status': 'fail',
                    'details': '文件不存在',
                    'recommendation': f'创建配置文件: {full_path}'
                }

            result['checks'].append(config_check)

        return result

    def check_data_paths(self) -> Dict[str, Any]:
        """检查数据路径"""
        print("💾 检查数据路径...")

        result = {
            'category': 'Data Paths',
            'checks': []
        }

        project_root = Path(__file__).parent.parent.parent.parent

        # 检查数据目录
        data_dir = project_root / 'data'
        data_check = {
            'name': '数据目录',
            'status': 'pass' if data_dir.exists() else 'warning',
            'details': '存在' if data_dir.exists() else '不存在',
            'recommendation': '✓' if data_dir.exists() else '创建data目录用于存储数据集'
        }
        result['checks'].append(data_check)

        # 检查metadata文件
        if data_dir.exists():
            metadata_files = list(data_dir.glob('metadata_*.xlsx'))
            metadata_check = {
                'name': 'Metadata文件',
                'status': 'pass' if metadata_files else 'warning',
                'details': f'找到{len(metadata_files)}个文件' if metadata_files else '未找到',
                'recommendation': '✓' if metadata_files else '请放置metadata_*.xlsx文件到data目录'
            }
            result['checks'].append(metadata_check)

            # 检查H5数据文件
            h5_files = list(data_dir.glob('*.h5'))
            h5_check = {
                'name': 'H5数据文件',
                'status': 'pass' if h5_files else 'warning',
                'details': f'找到{len(h5_files)}个文件' if h5_files else '未找到',
                'recommendation': '✓' if h5_files else 'H5文件用于高效数据加载'
            }
            result['checks'].append(h5_check)

        return result

    def check_system_resources(self) -> Dict[str, Any]:
        """检查系统资源"""
        print("💻 检查系统资源...")

        result = {
            'category': 'System Resources',
            'checks': []
        }

        try:
            import psutil

            # CPU信息
            cpu_count = psutil.cpu_count()
            cpu_check = {
                'name': 'CPU核心数',
                'status': 'pass' if cpu_count >= 4 else 'warning',
                'details': f'{cpu_count}核',
                'recommendation': '推荐4核以上' if cpu_count < 4 else '✓'
            }
            result['checks'].append(cpu_check)

            # 内存信息
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            memory_check = {
                'name': '系统内存',
                'status': 'pass' if memory_gb >= 8 else 'warning',
                'details': f'{memory_gb:.1f}GB (可用: {memory.available / (1024**3):.1f}GB)',
                'recommendation': '推荐8GB以上' if memory_gb < 8 else '✓'
            }
            result['checks'].append(memory_check)

            # 磁盘空间
            disk = psutil.disk_usage('/')
            disk_free_gb = disk.free / (1024**3)
            disk_check = {
                'name': '磁盘空间',
                'status': 'pass' if disk_free_gb >= 10 else 'warning',
                'details': f'可用: {disk_free_gb:.1f}GB',
                'recommendation': '推荐10GB以上可用空间' if disk_free_gb < 10 else '✓'
            }
            result['checks'].append(disk_check)

        except ImportError:
            resource_check = {
                'name': '系统资源检查',
                'status': 'fail',
                'details': 'psutil未安装',
                'recommendation': '安装psutil以进行资源监控'
            }
            result['checks'].append(resource_check)

        return result

    def _compare_versions(self, version1: str, version2: str) -> int:
        """简单的版本比较"""
        def parse_version(v):
            return list(map(int, v.split('.')[:3]))

        try:
            v1_parts = parse_version(version1)
            v2_parts = parse_version(version2)

            for i in range(max(len(v1_parts), len(v2_parts))):
                v1_part = v1_parts[i] if i < len(v1_parts) else 0
                v2_part = v2_parts[i] if i < len(v2_parts) else 0

                if v1_part < v2_part:
                    return -1
                elif v1_part > v2_part:
                    return 1

            return 0
        except:
            return 0  # 无法比较时认为相等

    def run_all_checks(self) -> List[Dict[str, Any]]:
        """运行所有检查"""
        print("🚀 开始全面环境检查...\n")

        all_results = []

        # 1. Python环境检查
        all_results.append(self.check_python_environment())

        # 2. 依赖包检查
        all_results.append(self.check_required_packages())

        # 3. PyTorch和CUDA检查
        all_results.append(self.check_pytorch_cuda())

        # 4. PHM-Vibench组件检查
        all_results.append(self.check_phm_vibench_components())

        # 5. 配置文件检查
        all_results.append(self.check_configuration_files())

        # 6. 数据路径检查
        if self.detailed:
            all_results.append(self.check_data_paths())

        # 7. 系统资源检查
        all_results.append(self.check_system_resources())

        return all_results

    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """生成检查报告"""
        print("\n" + "="*60)
        print("📊 环境检查报告")
        print("="*60)

        total_checks = 0
        passed_checks = 0
        failed_checks = 0
        warning_checks = 0

        for category_result in results:
            category = category_result['category']
            checks = category_result['checks']

            print(f"\n🔸 {category}")
            print("-" * 40)

            for check in checks:
                status_icon = {
                    'pass': '✅',
                    'fail': '❌',
                    'warning': '⚠️',
                    'info': 'ℹ️'
                }.get(check['status'], '?')

                print(f"{status_icon} {check['name']}: {check['details']}")

                if check['status'] != 'pass' and check['status'] != 'info':
                    print(f"   💡 建议: {check['recommendation']}")

                total_checks += 1
                if check['status'] == 'pass':
                    passed_checks += 1
                elif check['status'] == 'fail':
                    failed_checks += 1
                elif check['status'] == 'warning':
                    warning_checks += 1

        # 总结
        print(f"\n{'='*60}")
        print("📋 检查总结")
        print(f"{'='*60}")
        print(f"总检查项: {total_checks}")
        print(f"✅ 通过: {passed_checks}")
        print(f"⚠️ 警告: {warning_checks}")
        print(f"❌ 失败: {failed_checks}")

        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        print(f"成功率: {success_rate:.1f}%")

        # 整体评估
        if failed_checks == 0:
            if warning_checks == 0:
                print("\n🎉 环境检查完美通过！ContrastiveIDTask已准备就绪。")
                overall_status = "excellent"
            else:
                print(f"\n✅ 环境检查基本通过，有{warning_checks}个警告项需要关注。")
                overall_status = "good"
        else:
            print(f"\n⚠️ 环境检查发现{failed_checks}个关键问题，需要修复后才能正常运行。")
            overall_status = "needs_fix"

        # 保存报告到文件
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'warnings': warning_checks,
                'failed': failed_checks,
                'success_rate': success_rate,
                'overall_status': overall_status
            },
            'detailed_results': results
        }

        report_file = Path(__file__).parent / f"environment_check_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        print(f"\n📄 详细报告已保存: {report_file}")

        return overall_status

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="ContrastiveIDTask环境检查工具")

    parser.add_argument('--detailed', action='store_true',
                       help='进行详细检查，包括数据文件验证')
    parser.add_argument('--fix', action='store_true',
                       help='自动修复常见问题（功能开发中）')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式，只显示总结')

    args = parser.parse_args()

    # 创建环境检查器
    checker = EnvironmentChecker(detailed=args.detailed, auto_fix=args.fix)

    try:
        # 运行检查
        results = checker.run_all_checks()

        # 生成报告
        overall_status = checker.generate_report(results)

        # 返回相应的退出码
        if overall_status == "excellent":
            return 0
        elif overall_status == "good":
            return 0  # 警告不影响基本功能
        else:
            return 1  # 有失败项

    except KeyboardInterrupt:
        print("\n⚠️ 环境检查被中断")
        return 130
    except Exception as e:
        print(f"\n❌ 环境检查出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())