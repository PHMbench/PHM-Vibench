import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from pprint import pprint

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from src.configs.config_utils import (
    load_config,
    merge_with_local_override,
    path_name,
    save_config,
    transfer_namespace,
)
from src.explain_factory.eligibility import explain_ready, write_eligibility
from src.explain_factory.metadata_reader import (
    read_meta_from_batch,
    snapshot_metadata,
    write_metadata_snapshot,
)
from src.utils.config_utils import parse_overrides, apply_overrides_to_config
from src.utils.utils import load_best_model_checkpoint, init_lab, close_lab, get_num_classes
from src.data_factory import build_data
from src.model_factory import build_model
from src.task_factory import build_task
from src.trainer_factory import build_trainer



def pipeline(args):
    """领域泛化(Domain Generalization)任务的流水线（带 explain/report 产物写出）
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        所有迭代的实验结果列表
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
    config_path = args.config_path
    print(f"[INFO] 加载配置文件: {config_path}")
    # 支持机器特定的本地覆盖 YAML（方案B）
    # 优先顺序：命令行 --local_config > configs/local/{hostname}.yaml > configs/local/local.yaml > configs/local/default.yaml
    configs = merge_with_local_override(config_path, getattr(args, 'local_config', None))

    # 应用CLI override参数（最高优先级）
    if hasattr(args, 'override') and args.override:
        print(f"[INFO] 应用CLI override参数: {args.override}")
        overrides = parse_overrides(args.override)
        configs = apply_overrides_to_config(configs, overrides)
        print(f"[INFO] 已应用 {len(overrides)} 个override参数")

    # 确保配置中包含必要的部分
    required_sections = ['data', 'model', 'task', 'trainer', 'environment']
    for section in required_sections:
        if not hasattr(configs, section):
            print(f"[ERROR] 配置文件中缺少 {section} 部分")
            return
    
    # 设置环境变量和命名空间
    args_environment = transfer_namespace(configs.environment if hasattr(configs, 'environment') else {})

    args_data = transfer_namespace(configs.data if hasattr(configs, 'data') else {})

    args_model = transfer_namespace(configs.model if hasattr(configs, 'model') else {})

    args_task = transfer_namespace(configs.task if hasattr(configs, 'task') else {})

    args_trainer = transfer_namespace(configs.trainer if hasattr(configs, 'trainer') else {})
    if args_task.name == 'Multitask':
        args_data.task_list = args_task.task_list
        args_model.task_list = args_task.task_list    
    for key, value in configs.environment.__dict__.items():
        if key.isupper():
            os.environ[key] = str(value)
            print(f"[INFO] 设置环境变量: {key}={value}")

    # 创建实验目录（依赖 environment.output_dir / path_name，不再强制依赖 VBENCH_* 变量）
    print("[INFO] 创建实验目录...")
    
    # -----------------------
    # 2. 多次迭代训练与测试
    # -----------------------
    all_results = []
    
    for it in range(args_environment.iterations):
        print(f"\n{'='*50}\n[INFO] 开始实验迭代 {it+1}/{args_environment.iterations}\n{'='*50}")
        
        # 设置路径和名称
        path, name = path_name(configs, it)
        Path(path).mkdir(parents=True, exist_ok=True)
        # 把name 加到args_trainer中
        args_trainer.logger_name = name

        # UXFD merge: always snapshot the fully-resolved config for auditability.
        try:
            save_config(configs, Path(path) / "config_snapshot.yaml")
        except Exception as e:
            print(f"[WARN] 保存 config_snapshot.yaml 失败: {e}")

        # 设置随机种子
        current_seed = args_environment.seed + it
        seed_everything(current_seed)
        print(f"[INFO] 设置随机种子: {current_seed}")
        init_lab(args_environment, args, name)


        # 构建数据工厂
        print("[INFO] 构建数据工厂...")
        data_factory = build_data(args_data, args_task)
        # 构建模型
        print("[INFO] 构建模型...")
        model = build_model(args_model,metadata=data_factory.get_metadata())
        
        # 构建任务
        print("[INFO] 构建任务...")
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=args_trainer,
            args_environment=args_environment,
            metadata=data_factory.get_metadata()
        )
        
        # 构建训练器
        print("[INFO] 构建训练器...")
        trainer = build_trainer(
            args_environment,
            args_trainer,
            args_data,
            path
        )

        # UXFD merge: always write a data metadata snapshot from the test dataloader (best-effort).
        artifacts_dir = Path(path) / "artifacts"
        meta_snapshot_path = artifacts_dir / "data_metadata_snapshot.json"
        batch_meta: dict = {}
        meta_source = "default"
        degraded = True
        try:
            test_loader = data_factory.get_dataloader("test")
            batch = next(iter(test_loader))
            x0, y0, meta0, meta_source = read_meta_from_batch(batch)
            if isinstance(meta0, dict):
                batch_meta.update(meta0)
            # attach shapes for traceability (safe even if meta is empty)
            if hasattr(x0, "shape"):
                batch_meta.setdefault("x_shape", [int(v) for v in x0.shape])
            if hasattr(y0, "shape"):
                batch_meta.setdefault("y_shape", [int(v) for v in y0.shape])

            snapshot = snapshot_metadata(meta=batch_meta, meta_source=meta_source)
            degraded = snapshot.degraded
            write_metadata_snapshot(meta_snapshot_path, snapshot)
        except Exception as e:
            print(f"[WARN] 写入 data_metadata_snapshot.json 失败: {e}")
            try:
                snapshot = snapshot_metadata(meta={}, meta_source="default")
                write_metadata_snapshot(meta_snapshot_path, snapshot)
            except Exception:
                pass

        # UXFD merge: if explain enabled, write eligibility.json (never crash).
        try:
            extensions = getattr(args_trainer, "extensions", None)
            explain_cfg = getattr(extensions, "explain", None) if extensions is not None else None
            explain_enable = bool(getattr(explain_cfg, "enable", False)) if explain_cfg is not None else False
            if explain_enable:
                explainer_id = str(getattr(explain_cfg, "explainer", "") or "unknown")
                eligibility_path = artifacts_dir / "explain" / "eligibility.json"
                required_meta_keys = []
                if explainer_id in {"timefreq", "time_freq"}:
                    required_meta_keys = ["sampling_rate"]
                ready = explain_ready(
                    explainer_id=explainer_id,
                    meta=batch_meta,
                    required_meta_keys=required_meta_keys,
                    meta_source=str(meta_source),
                    degraded=bool(degraded),
                )
                write_eligibility(eligibility_path, ready)
        except Exception as e:
            print(f"[WARN] 写入 explain eligibility 失败: {e}")
        
        # 执行训练
        print("[INFO] 开始训练...")
        trainer.fit(
            task,
            data_factory.get_dataloader('train'),
            data_factory.get_dataloader('val')
        )
        
        # 加载最佳模型并测试
        print("[INFO] 加载最佳模型并测试...")
        task = load_best_model_checkpoint(task, trainer)
        result = trainer.test(task, data_factory.get_dataloader('test'))
        all_results.append(result[0])  # Lightning返回的是包含字典的列表
        
        # 保存结果
        print("[INFO] 保存测试结果...")
        result_df = pd.DataFrame([result[0]])
        result_df.to_csv(os.path.join(path, f'test_result_{it}.csv'), index=False)

        # UXFD merge: rewrite manifest after test_result exists (callback runs earlier).
        try:
            from src.trainer_factory.extensions import ManifestWriterCallback

            # Determine main process similar to Default_trainer.py logic.
            is_main_process = True
            if "LOCAL_RANK" in os.environ:
                is_main_process = int(os.environ["LOCAL_RANK"]) == 0

            extensions = getattr(args_trainer, "extensions", None)
            report_cfg = getattr(extensions, "report", None) if extensions is not None else None
            report_enable = getattr(report_cfg, "enable", True) if report_cfg is not None else True
            manifest_enable = getattr(report_cfg, "manifest", True) if report_cfg is not None else True
            enabled = bool(report_enable) and bool(manifest_enable)

            ManifestWriterCallback(
                run_dir=path,
                paper_id=str(getattr(args_trainer, "paper_id", "") or ""),
                preset_version=str(getattr(args_trainer, "preset_version", "") or ""),
                run_id=str(getattr(args_trainer, "logger_name", "") or ""),
                enabled=enabled,
                is_main_process=is_main_process,
            ).on_test_end(trainer, task)
        except Exception as e:
            print(f"[WARN] 更新 artifacts/manifest.json 失败: {e}")

        data_factory.data.close()  # 关闭数据工厂，释放资源

        # 关闭wandb和swanlab
        close_lab()

    print(f"\n{'='*50}\n[INFO] 所有实验已完成\n{'='*50}")
    pd.DataFrame(all_results).to_csv(os.path.join(path, 'all_results.csv'), index=False)
    return all_results


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="领域泛化(DG)任务流水线")
    
    parser.add_argument('--config_path', 
                        type=str, 
                        default='/home/user/LQ/B_Signal/Signal_foundation_model/Vbench/configs/demo/Single_DG/CWRU.yaml',
                        help='配置文件路径')
    parser.add_argument('--notes', 
                        type=str, 
                        default='',
                        help='实验备注')
    parser.add_argument('--local_config',
                        type=str,
                        default=None,
                        help='本机覆盖配置路径（可选）')

    
    args = parser.parse_args()
    
    # 执行DG流水线
    results = pipeline(args)
    print(f"完成所有实验！")
