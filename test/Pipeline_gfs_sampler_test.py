import argparse
import os
import sys
from pprint import pprint

# 假设 PHM-Vibench 的根目录在 PYTHONPATH 中
# 或者根据实际项目结构调整导入路径
from src.utils.config_utils import load_config, transfer_namespace
from src.data_factory import build_data
from src.data_factory.samplers.FS_sampler import HierarchicalFewShotSampler

def pipeline(args):
    """
    专门用于测试 HierarchicalFewShotSampler 的流水线。
    """
    # -----------------------
    # 1. 加载配置文件
    # -----------------------
    config_path = args.config_path
    print(f"[INFO] [GFS Sampler Test Pipeline] 加载配置文件: {config_path}")
    configs = load_config(config_path)
    
    if not configs:
        print(f"[ERROR] [GFS Sampler Test Pipeline] 无法加载配置文件: {config_path}")
        return {"status": "error", "message": "Config load failed"}

    print(f"[INFO] [GFS Sampler Test Pipeline] 加载的配置内容:")
    pprint(configs)

    args_environment = transfer_namespace(configs.get('environment', {}))
    args_data = transfer_namespace(configs.get('data', {}))
    args_task = transfer_namespace(configs.get('task', {}))



    # -----------------------
    # 2. 构建数据工厂和数据集
    # -----------------------
    print("[INFO] [GFS Sampler Test Pipeline] 构建数据工厂...")
    try:
        data_factory = build_data(args_data, args_task)
    except Exception as e:
        print(f"[ERROR] [GFS Sampler Test Pipeline] 构建数据工厂时出错: {e}")
        return {"status": "error", "message": f"Data factory build failed: {e}"}

    print("[INFO] [GFS Sampler Test Pipeline] 从数据工厂获取用于采样器的数据集 (例如 'train' 部分)...")
    try:
        dataset_for_sampler = data_factory.get_dataset('train') 
    except Exception as e:
        print(f"[ERROR] [GFS Sampler Test Pipeline] 从数据工厂获取数据集时出错: {e}")
        if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
            data_factory.data.close()
        return {"status": "error", "message": f"Dataset retrieval failed: {e}"}
        
    if dataset_for_sampler is None:
        print("[ERROR] [GFS Sampler Test Pipeline] 未能从数据工厂获取 'train' 数据集。")
        if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
            data_factory.data.close()
        return {"status": "error", "message": "Train dataset is None"}

    print(f"[INFO] [GFS Sampler Test Pipeline] 获取到的采样器数据集: {type(dataset_for_sampler)}, 长度: {len(dataset_for_sampler)}")

    if not hasattr(dataset_for_sampler, 'get_file_windows_list') or \
       not callable(getattr(dataset_for_sampler, 'get_file_windows_list', None)) or \
       not hasattr(dataset_for_sampler, 'metadata') or \
       dataset_for_sampler.metadata is None:
        print("[ERROR] [GFS Sampler Test Pipeline] 数据集不具备 HierarchicalFewShotSampler 所需的属性/方法。")
        if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
            data_factory.data.close()
        return {"status": "error", "message": "Dataset missing required attributes for sampler"}

    # -----------------------
    # 3. 实例化采样器
    # -----------------------
    print("[INFO] [GFS Sampler Test Pipeline] 实例化 HierarchicalFewShotSampler...")
    try:
        sampler = HierarchicalFewShotSampler(
            dataset=dataset_for_sampler,
            num_episodes=args_task.num_episodes,
            num_systems_per_episode=args_task.num_systems,
            num_domains_per_system=args_task.num_domains,
            num_labels_per_domain_task=args_task.num_labels,
            num_support_per_label=args_task.num_support,
            num_query_per_label=args_task.num_query,
            system_metadata_key=getattr(args_task, 'system_metadata_key', 'Dataset_id'),
            domain_metadata_key=getattr(args_task, 'domain_metadata_key', 'Domain_id'),
            label_metadata_key=getattr(args_task, 'label_metadata_key', 'Label')
        )
        print(f"[INFO] [GFS Sampler Test Pipeline] 采样器实例化成功。预期总样本数 (len(sampler)): {len(sampler)}")
        print(f"  采样器内部 runnable_system_ids: {sampler.runnable_system_ids}")

    except ValueError as ve:
        print(f"[ERROR] [GFS Sampler Test Pipeline] 实例化采样器时发生 ValueError: {ve}")
        if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
            data_factory.data.close()
        return {"status": "error", "message": f"Sampler ValueError: {ve}"}
    except Exception as e:
        print(f"[ERROR] [GFS Sampler Test Pipeline] 实例化采样器时发生未知错误: {e}")
        if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
            data_factory.data.close()
        return {"status": "error", "message": f"Sampler instantiation error: {e}"}

    # -----------------------
    # 4. 通过迭代测试采样器
    # -----------------------
    print("[INFO] [GFS Sampler Test Pipeline] 开始迭代采样器以生成 episodes...")
    episode_count = 0
    samples_per_episode = (args_task.num_systems *
                           args_task.num_domains *
                           args_task.num_labels *
                           (args_task.num_support + args_task.num_query))

    if samples_per_episode == 0 and args_task.num_episodes > 0:
        print("[WARNING] [GFS Sampler Test Pipeline] 每个 episode 的样本数为 0。")
    
    all_generated_indices_for_episodes = []
    current_episode_indices_buffer = []
    total_yielded_indices = 0

    for global_idx in sampler:
        current_episode_indices_buffer.append(global_idx)
        total_yielded_indices += 1
        if samples_per_episode > 0 and len(current_episode_indices_buffer) == samples_per_episode:
            episode_count += 1
            print(f"  [GFS Sampler Test Pipeline] 生成 Episode {episode_count}/{args_task.num_episodes}，索引: {current_episode_indices_buffer[:5]}...")
            all_generated_indices_for_episodes.append(list(current_episode_indices_buffer))
            current_episode_indices_buffer = []
    
    if current_episode_indices_buffer: # Should be empty if logic is correct
        print(f"  [WARNING] [GFS Sampler Test Pipeline] 末尾残留索引: {len(current_episode_indices_buffer)} 个。")
        all_generated_indices_for_episodes.append(list(current_episode_indices_buffer))

    print(f"[INFO] [GFS Sampler Test Pipeline] 采样器迭代完成。")
    print(f"  共生成 {episode_count} 个完整的 episodes。") # episode_count might be less than num_episodes if sampler exhausted
    print(f"  共产生 {total_yielded_indices} 个索引。")

    if total_yielded_indices != len(sampler):
        print(f"[ERROR] [GFS Sampler Test Pipeline] 产生的总索引数 ({total_yielded_indices}) 与 len(sampler) ({len(sampler)}) 不匹配!")
    
    # Check if actual episodes match requested, only if samples_per_episode > 0
    if samples_per_episode > 0 and episode_count != args_task.num_episodes:
         print(f"[WARNING] [GFS Sampler Test Pipeline] 期望生成 {args_task.num_episodes} episodes，实际生成 {episode_count}。")

    if all_generated_indices_for_episodes and hasattr(sampler, 'samples_df') and sampler.samples_df is not None and not sampler.samples_df.empty:
        first_episode_indices = all_generated_indices_for_episodes[0]
        print(f"\n[INFO] [GFS Sampler Test Pipeline] 第一个 episode 前几个样本详情 (索引: {first_episode_indices[:3]}...):")
        for i, idx_val in enumerate(first_episode_indices[:min(3, len(first_episode_indices))]):
            try:
                sample_details_df = sampler.samples_df[sampler.samples_df['global_idx'] == idx_val]
                if not sample_details_df.empty:
                    sample_details = sample_details_df.iloc[0]
                    print(f"  样本 {i+1} (全局索引: {idx_val}): 系统={sample_details['system_id']}, 域={sample_details['domain_id']}, 标签={sample_details['label_id']}")
                else:
                    print(f"  样本 {i+1} (全局索引: {idx_val}): 在 sampler.samples_df 中未找到。")
            except Exception as e:
                 print(f"  样本 {i+1} (全局索引: {idx_val}): 获取详情时出错 - {e}")
    
    if hasattr(data_factory, 'data') and hasattr(data_factory.data, 'close'):
        data_factory.data.close()
        print("[INFO] [GFS Sampler Test Pipeline] 已关闭数据工厂资源。")

    print("[INFO] [GFS Sampler Test Pipeline] 测试流水线执行完毕。")
    return {"status": "success", "episodes_generated": episode_count, "total_indices_yielded": total_yielded_indices}

if __name__ == "__main__":
    # This allows running the test pipeline directly for debugging if needed
    parser = argparse.ArgumentParser(description="GFS Sampler Test Pipeline (Direct Run)")
    parser.add_argument('--config_path', 
                        type=str, 
                        default='../../configs/demo/GFS/test.yaml', # Adjusted default path for direct run
                        help='配置文件路径')
    parser.add_argument('--notes', type=str, default='Direct test run', help='实验备注')
    
    cli_args = parser.parse_args()
    
    # For direct execution, ensure src is in path if VBENCH_HOME is not set
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..')) # Moves up one level from 'test' to project root
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    pipeline(cli_args)
