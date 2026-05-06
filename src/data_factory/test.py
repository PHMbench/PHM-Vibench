"""Manual smoke helper for the hierarchical few-shot sampler.

The module must stay import-safe because factory-wide checks walk every Python
module under ``src/*_factory``.
"""

import torch

from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.data_factory.samplers.FS_sampler import HierarchicalFewShotSampler


def build_demo_dataset():
    systems = ['System1', 'System2', 'System3', 'System4']
    domains = ['DomainA', 'DomainB', 'DomainC', 'DomainD']
    labels = [0, 1, 2, 3]

    dataset_dict = {}
    metadata = {}
    current_sample_index = 0
    for system_id in systems:
        for domain_id in domains:
            for label_id in labels:
                sample_unique_id = f'sample_{current_sample_index}'
                dataset_dict[f'ref_{sample_unique_id}'] = {
                    'id': sample_unique_id,
                    'Domain_id': domain_id,
                    'Label': label_id,
                }
                metadata[sample_unique_id] = {
                    'Dataset_id': system_id,
                    'Domain_id': domain_id,
                    'Label': label_id,
                }
                current_sample_index += 1

    return IdIncludedDataset(dataset_dict=dataset_dict, metadata=metadata)


def main():
    id_included_train_dataset = build_demo_dataset()
    m_systems = 2
    j_domains_per_system = 2
    n_labels_per_domain = 2
    k_support = 2
    q_query = 2
    num_train_episodes = 200

    train_sampler = HierarchicalFewShotSampler(
        dataset=id_included_train_dataset,
        num_episodes=num_train_episodes,
        num_systems_per_episode=m_systems,
        num_domains_per_system=j_domains_per_system,
        num_labels_per_domain_task=n_labels_per_domain,
        num_support_per_label=k_support,
        num_query_per_label=q_query,
        system_metadata_key='Dataset_id',
        domain_metadata_key='Domain_id',
        label_metadata_key='Label',
    )

    batch_size_for_loader = (
        m_systems
        * j_domains_per_system
        * n_labels_per_domain
        * (k_support + q_query)
    )
    if batch_size_for_loader == 0 and len(id_included_train_dataset) > 0:
        print("Warning: calculated batch_size_for_loader is 0.")

    if batch_size_for_loader <= 0:
        print("Skipping DataLoader creation as batch_size_for_loader is 0.")
        return

    train_loader = torch.utils.data.DataLoader(
        id_included_train_dataset,
        batch_size=batch_size_for_loader,
        sampler=train_sampler,
        num_workers=4,
    )

    for episode_batch in train_loader:
        print(f"Episode batch size: {len(episode_batch)}")
    print(
        f"Created DataLoader with batch size {batch_size_for_loader} "
        f"for {num_train_episodes} episodes."
    )


if __name__ == "__main__":
    main()
