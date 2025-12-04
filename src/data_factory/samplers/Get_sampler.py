from .Sampler import HierarchicalFewShotSampler, Same_system_Sampler

def _get_gfs_sampler(args_task, args_data, dataset, mode):
    if mode == 'train':
        sampler = HierarchicalFewShotSampler(
            dataset=dataset,
            num_episodes=args_task.num_episodes,
            num_systems_per_episode=args_task.num_systems,
            num_domains_per_system=args_task.num_domains,
            num_labels_per_domain_task=args_task.num_labels,
            num_support_per_label=args_task.num_support,
            num_query_per_label=args_task.num_query,
        )
    elif mode == 'test' or mode == 'val':
        sampler = Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=False,
            drop_last=True,
        )
    else:
        raise ValueError(f"Unknown mode for GFS sampler: {mode}")
    return sampler

def _get_cddg_sampler(args_data, dataset, mode):
    if mode == 'train':
        sampler = Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif mode == 'val' or mode == 'test':
        sampler = Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        raise ValueError(f"Unknown mode for CDDG sampler: {mode}")
    return sampler

def _get_dg_sampler(args_data, dataset, mode):
    if mode == 'train':
        sampler = Same_system_Sampler(
            dataset, 
            batch_size=args_data.batch_size,
            shuffle=True,
            drop_last=True
        )
    elif mode == 'val' or mode == 'test':
        sampler = Same_system_Sampler(
            dataset,
            batch_size=args_data.batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        raise ValueError(f"Unknown mode for DG sampler: {mode}")
    return sampler

def _get_pretrain_sampler(args_data, dataset, mode):
    if mode == 'train':
        sampler = Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=True,
            drop_last=True,
        )
    elif mode == 'val' or mode == 'test':
        sampler = Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        raise ValueError(f"Unknown mode for Pretrain sampler: {mode}")
    return sampler


def Get_sampler(args_task, args_data, dataset, mode='train'):
    """
    Initializes and returns a sampler based on the task type and mode.

    Args:
        args_task: Task-specific arguments.
        args_data: Data-specific arguments.
        dataset: The dataset for which the sampler is to be created.
        mode: 'train', 'val', or 'test'.

    Returns:
        A sampler instance or None for 'FS' type.
    """
    sampler = None # Initialize sampler to None
    if args_task.type == 'GFS': # Generalized Few-Shot Learning
        sampler = _get_gfs_sampler(args_task, args_data, dataset, mode)
    elif args_task.type == 'FS':
        # FS 视作单系统 few-shot 场景，先复用 Same_system_Sampler，保持与 DG/CDDG 一致的按系统分组 batch 行为
        if mode == 'train':
            sampler = Same_system_Sampler(
                dataset=dataset,
                batch_size=args_data.batch_size,
                shuffle=True,
                drop_last=True,
            )
        elif mode == 'val' or mode == 'test':
            sampler = Same_system_Sampler(
                dataset=dataset,
                batch_size=args_data.batch_size,
                shuffle=False,
                drop_last=True,
            )
        else:
            raise ValueError(f"Unknown mode for FS sampler: {mode}")
    elif args_task.type == 'pretrain':
        sampler = _get_pretrain_sampler(args_data, dataset, mode)
    elif args_task.type == 'CDDG':
        sampler = _get_cddg_sampler(args_data, dataset, mode)
    elif args_task.type == 'DG':
        sampler = _get_dg_sampler(args_data, dataset, mode)
    else:
        raise ValueError(f"Unknown task type for sampler: {args_task.type}")
        
    return sampler
