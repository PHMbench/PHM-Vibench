"""Task-aware batch sampler selection."""

from .Sampler import HierarchicalFewShotSampler, Same_system_Sampler


def _evaluation_sampler(args_data, dataset):
    """Keep every validation/test sample, including a final short batch."""
    return Same_system_Sampler(
        dataset=dataset,
        batch_size=args_data.batch_size,
        shuffle=False,
        drop_last=False,
    )


def _get_gfs_sampler(args_task, args_data, dataset, mode):
    if mode == "train":
        return HierarchicalFewShotSampler(
            dataset=dataset,
            num_episodes=args_task.num_episodes,
            num_systems_per_episode=args_task.num_systems,
            num_domains_per_system=args_task.num_domains,
            num_labels_per_domain_task=args_task.num_labels,
            num_support_per_label=args_task.num_support,
            num_query_per_label=args_task.num_query,
        )
    if mode in {"val", "test"}:
        return _evaluation_sampler(args_data, dataset)
    raise ValueError(f"Unknown mode for GFS sampler: {mode}")


def _get_standard_sampler(args_data, dataset, mode, task_name):
    if mode == "train":
        return Same_system_Sampler(
            dataset=dataset,
            batch_size=args_data.batch_size,
            shuffle=True,
            drop_last=True,
        )
    if mode in {"val", "test"}:
        return _evaluation_sampler(args_data, dataset)
    raise ValueError(f"Unknown mode for {task_name} sampler: {mode}")


def Get_sampler(args_task, args_data, dataset, mode="train"):
    """Return the sampler for one explicit task type and split."""
    task_type = args_task.type

    if task_type == "GFS":
        return _get_gfs_sampler(args_task, args_data, dataset, mode)
    if task_type == "FS":
        return _get_standard_sampler(args_data, dataset, mode, "FS")
    if task_type in {"pretrain", "generative"}:
        return _get_standard_sampler(args_data, dataset, mode, "Pretrain")
    if task_type == "CDDG":
        return _get_standard_sampler(args_data, dataset, mode, "CDDG")
    if task_type == "DG":
        return _get_standard_sampler(args_data, dataset, mode, "DG")
    if task_type == "multi_task":
        return _get_standard_sampler(args_data, dataset, mode, "multi_task")
    if task_type == "In_distribution":
        return _get_standard_sampler(
            args_data,
            dataset,
            mode,
            "In_distribution",
        )
    if task_type == "Default_task":
        return _get_standard_sampler(args_data, dataset, mode, "Default_task")
    raise ValueError(f"Unknown task type for sampler: {task_type}")
