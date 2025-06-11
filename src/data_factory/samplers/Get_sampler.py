
def Get_sampler(args_task, args_data, dataset, mode='train'):
    """
    Initializes and returns a sampler based on the task type and mode.

    Args:
        args_task: Task-specific arguments.
        args_data: Data-specific arguments.
        dataset: The dataset for which the sampler is to be created.
        mode: 'train', 'val', or 'test'.

    Returns:
        A sampler instance.
    """
    if args_task.type == 'GFS': # Generalized Few-Shot Learning
        from .Sampler import HierarchicalFewShotSampler, Same_system_Sampler
        if mode == 'train' or mode == 'val':
            sampler = HierarchicalFewShotSampler(
                num_systems_per_episode=args_task.num_systems,    # M
                num_domains_per_system=args_task.num_domains,     # J
                num_labels_per_domain_task=args_task.num_labels, # N (N-way for each system-domain sub-task)
                # Shot and query parameters
                num_support_per_label=args_task.num_support,      # K
                num_query_per_label=args_task.num_query,        # Q
            )
        elif mode == 'test':
            sampler = Same_system_Sampler(
                dataset=dataset,
                batch_size=args_data.batch_size,
                shuffle=False,
                drop_last=True,
            )
        else:
            raise ValueError(f"Unknown mode for GFS sampler: {mode}")
    elif args_task.type == 'FS':
        pass
    elif args_task.type == 'CDDG':
        from .Sampler import Same_system_Sampler
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

    elif args_task.type == 'DG':
        from .Sampler import Same_system_Sampler
        if mode == 'train':
            sampler = Same_system_Sampler(dataset, # PyTorch Sampler typically takes dataset as first arg
                                          batch_size=args_data.batch_size,
                                          shuffle=True,
                                          drop_last=True)
        elif mode == 'val' or mode == 'test':
            sampler = Same_system_Sampler(dataset,
                                          batch_size=args_data.batch_size,
                                          shuffle=False,
                                          drop_last=True)
        else:
            raise ValueError(f"Unknown mode for DG sampler: {mode}")
    else:
        raise ValueError(f"Unknown task type for sampler: {args_task.type}")
        
    return sampler
