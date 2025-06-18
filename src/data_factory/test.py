

from samplers.FS_sampler import HierarchicalFewShotSampler
from balanced_data_loader import IdIncludedDataset
import torch
# Define data characteristics: 2 systems, 2 domains, 2 labels = 8 data points
systems = ['System1', 'System2', 'System3', 'System4']
domains = ['DomainA', 'DomainB', 'DomainC', 'DomainD']
labels = [0, 1, 2, 3] # Using integer labels as in the original example

dataset_dict = {}
metadata = {}

current_sample_index = 0
for system_id in systems:
    for domain_id in domains:
        for label_id in labels:
            # Create a unique ID for each of the 8 samples.
            # This ID links the entry in flat_sample_map to its metadata.
            sample_unique_id = f'sample_{current_sample_index}'

            # dataset_dict:
            # The key (e.g., f'ref_{sample_unique_id}') is for IdIncludedDataset's internal use.
            # The 'id' field in the value dict should match sample_unique_id.
            dataset_dict[f'ref_{sample_unique_id}'] = {
                'id': sample_unique_id,
                'Domain_id': domain_id, # Mirroring structure of original example
                'Label': label_id       # Mirroring structure of original example
            }

            # metadata: maps sample_unique_id to its detailed metadata
            # The keys 'Dataset_id', 'Domain_id', 'Label' must match
            # system_metadata_key, domain_metadata_key, label_metadata_key
            # used by the HierarchicalFewShotSampler.
            metadata[sample_unique_id] = {
                'Dataset_id': system_id, # This is the "system"
                'Domain_id': domain_id,
                'Label': label_id
            }
            current_sample_index += 1

id_included_train_dataset = IdIncludedDataset(
    dataset_dict=dataset_dict,
    metadata=metadata
)
M_SYSTEMS = 2
J_DOMAINS_PER_SYSTEM = 2
N_LABELS_PER_DOMAIN = 2  # This is the N-way for each sub-task
K_SUPPORT = 2            # K-shot
Q_QUERY = 2
NUM_TRAIN_EPISODES = 200


train_sampler = HierarchicalFewShotSampler(
    dataset=id_included_train_dataset,
    num_episodes=NUM_TRAIN_EPISODES,
    num_systems_per_episode=M_SYSTEMS,
    num_domains_per_system=J_DOMAINS_PER_SYSTEM,
    num_labels_per_domain_task=N_LABELS_PER_DOMAIN,
    num_support_per_label=K_SUPPORT,
    num_query_per_label=Q_QUERY,
    system_metadata_key='Dataset_id', # Adjust if your metadata uses different key names
    domain_metadata_key='Domain_id',
    label_metadata_key='Label'
)

# Create DataLoader
# The batch_size for the DataLoader should correspond to the total number of samples in one episode
batch_size_for_loader = (M_SYSTEMS *
                            J_DOMAINS_PER_SYSTEM *
                            N_LABELS_PER_DOMAIN *
                            (K_SUPPORT + Q_QUERY))

if batch_size_for_loader == 0 and len(id_included_train_dataset) > 0 : # Avoid division by zero if params are 0
    print("Warning: Calculated batch_size_for_loader is 0. Sampler or parameters might be misconfigured.")
    # Handle appropriately, maybe set a default or raise error
    # For now, if it's zero, it means no samples will be drawn per episode.

if batch_size_for_loader > 0:
    train_loader = torch.utils.data.DataLoader(
        id_included_train_dataset,
        batch_size=batch_size_for_loader, # DataLoader forms batches from yielded indices
        sampler=train_sampler,            # Use the custom sampler
        num_workers=4,
        # collate_fn=your_custom_collate_fn # If needed
    )

    # Iterating through train_loader will give one episode per batch.
    for episode_batch in train_loader:

        # episode_batch contains all samples for one episode.
        # Each sample is a dict: {'x': ..., 'y': ..., 'id': ...}
        # You'll need to parse this batch into support/query sets for each
        # (system, domain, label) task based on the M, J, N, K, Q parameters.
        # The samples are ordered:
        # all K+Q for (sys1, dom1, label1), then all K+Q for (sys1, dom1, label2) ...
        print(f"Episode batch size: {len(episode_batch)}")
    print(f"Created DataLoader with batch size {batch_size_for_loader} for {NUM_TRAIN_EPISODES} episodes.")
else:
    print("Skipping DataLoader creation as batch_size_for_loader is 0.")