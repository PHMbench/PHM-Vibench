import torch
from torch.utils.data import Sampler, Dataset
import numpy as np
import random
from collections import defaultdict
import pandas as pd # 新增导入

# Assumes IdIncludedDataset is available, e.g.:
# from src.data_factory.balanced_data_loader import IdIncludedDataset

class HierarchicalFewShotSampler(Sampler[int]):
    """
    Hierarchical Few-Shot Learning Sampler for episodic tasks.
    This sampler generates episodes by selecting a fixed number of systems, domains, and labels
    while ensuring that each episode contains enough samples for K-shot and Q-query tasks.
    It builds a hierarchy of systems, domains, and labels from the dataset metadata
    and filters them based on the specified parameters.
    Args:
        dataset (IdIncludedDataset): An instance of IdIncludedDataset containing the data and metadata.
        num_episodes (int): Total number of episodes to generate.
        num_systems_per_episode (int): Number of systems to include in each episode (M).
        num_domains_per_system (int): Number of domains per system in each episode (J).
        num_labels_per_domain_task (int): Number of labels per domain task in each episode (N).
        num_support_per_label (int): Number of support samples per label in each episode (K).
        num_query_per_label (int): Number of query samples per label in each episode (Q).
        system_metadata_key (str): Metadata key for system ID.
        domain_metadata_key (str): Metadata key for domain ID.
        label_metadata_key (str): Metadata key for label ID.
    """

    def __init__(self,
                 dataset: Dataset, # Your IdIncludedDataset instance
                 num_episodes: int,
                 # Hierarchical selection parameters
                 num_systems_per_episode: int,    # M
                 num_domains_per_system: int,     # J
                 num_labels_per_domain_task: int, # N (N-way for each system-domain sub-task)
                 # Shot and query parameters
                 num_support_per_label: int,      # K
                 num_query_per_label: int,        # Q
                 # Metadata keys
                 system_metadata_key: str = 'Dataset_id', # system
                 domain_metadata_key: str = 'Domain_id',
                 label_metadata_key: str = 'Label'):
        super().__init__(dataset)
        self.dataset = dataset
        self.num_episodes = num_episodes
        self.m_systems = num_systems_per_episode
        self.j_domains = num_domains_per_system
        self.n_labels_way = num_labels_per_domain_task # N-way
        self.k_shot = num_support_per_label
        self.q_query = num_query_per_label

        self.system_key = system_metadata_key
        self.domain_key = domain_metadata_key
        self.label_key = label_metadata_key # This is the actual class label for N-way

        if not hasattr(dataset, 'get_file_windows_list') or \
           not callable(getattr(dataset, 'get_file_windows_list', None)) or \
           not hasattr(dataset, 'dataset_dict') or \
           not hasattr(dataset, 'metadata') or dataset.metadata is None:
            raise ValueError("Dataset must be an IdIncludedDataset instance with "
                             "a 'get_file_windows_list' method, and 'dataset_dict', "
                             "and 'metadata' attributes.")

        self.metadata = dataset.metadata
        self.samples_df = None # 将用于存储样本信息的DataFrame
        self.runnable_system_ids = []
        self.system_to_valid_domains_map = defaultdict(list)
        self.domain_to_valid_labels_map = defaultdict(list)

        self._build_index_hierarchy()
        self._filter_and_prepare_for_sampling()

        self.len = self.num_episodes #* (self.m_systems *
                            #    self.j_domains *
                            #    self.n_labels_way *
                            #    (self.k_shot + self.q_query))

    def _build_index_hierarchy(self):
        """
        Builds a Pandas DataFrame mapping global_idx to system, domain, and label.
        self.samples_df will have columns: ['global_idx', 'system_id', 'domain_id', 'label_id']
        """
        file_windows_list = self.dataset.get_file_windows_list() # {'File_id': file_id, 'Window_id': window_id}

        if not file_windows_list:
            print("Warning: dataset.get_file_windows_list() returned an empty list.")
            self.samples_df = pd.DataFrame(columns=['global_idx', 'system_id', 'domain_id', 'label_id'])
            return

        data_for_df = []
        for global_idx, sample_mapping_info in enumerate(file_windows_list):
            original_dataset_key = sample_mapping_info['File_id']

            if original_dataset_key not in self.metadata:
                print(f"Warning: Key '{original_dataset_key}' from dataset.get_file_windows_list() "
                      f"not found in dataset.metadata. Skipping sample global_idx {global_idx}.")
                continue
            
            meta_entry = self.metadata[original_dataset_key]

            try:
                system_id = str(meta_entry[self.system_key])
                domain_id = str(meta_entry[self.domain_key])
                label_id = str(meta_entry[self.label_key])
                data_for_df.append({
                    'global_idx': global_idx,
                    'system_id': system_id,
                    'domain_id': domain_id,
                    'label_id': label_id
                })
            except KeyError as e:
                print(f"Warning: Metadata for key '{original_dataset_key}' is missing expected "
                      f"metadata key {e} (system:'{self.system_key}', domain:'{self.domain_key}', "
                      f"label:'{self.label_key}'). Skipping sample global_idx {global_idx}.")
                continue
        
        if not data_for_df:
            print("Warning: No valid samples found to build the DataFrame after processing metadata.")
            self.samples_df = pd.DataFrame(columns=['global_idx', 'system_id', 'domain_id', 'label_id'])
        else:
            self.samples_df = pd.DataFrame(data_for_df)

    def _filter_and_prepare_for_sampling(self):
        """
        Filters the hierarchy based on M, J, N, K, Q requirements using self.samples_df
        and prepares lists/maps for efficient sampling.
        """
        if self.samples_df.empty:
            raise ValueError("Cannot prepare for sampling as no sample data was loaded into the DataFrame.")

        # 1. Filter tasks (system-domain-label) by K+Q samples
        task_counts = self.samples_df.groupby(['system_id', 'domain_id', 'label_id']).size()
        valid_tasks_s_d_l = task_counts[task_counts >= (self.k_shot + self.q_query)].reset_index()[['system_id', 'domain_id', 'label_id']]

        if valid_tasks_s_d_l.empty:
            raise ValueError(
                f"No tasks (system-domain-label combinations) have at least "
                f"{self.k_shot + self.q_query} (K+Q) samples. "
                f"Check K, Q parameters and data distribution."
            )

        # 2. Filter domains (system-domain) by N_labels_way
        # Count unique valid labels per system-domain
        domain_label_counts = valid_tasks_s_d_l.groupby(['system_id', 'domain_id'])['label_id'].nunique()
        valid_domains_s_d = domain_label_counts[domain_label_counts >= self.n_labels_way].reset_index()[['system_id', 'domain_id']]

        if valid_domains_s_d.empty:
            raise ValueError(
                f"No domains (system-domain combinations) have at least "
                f"{self.n_labels_way} valid labels (N-way). "
                f"Check N parameter and data distribution after K+Q filtering."
            )

        # Populate self.domain_to_valid_labels_map
        # Filter valid_tasks_s_d_l to only include those in valid_domains_s_d
        # Then group by (system_id, domain_id) and collect label_ids
        merged_domain_labels = pd.merge(valid_tasks_s_d_l, valid_domains_s_d, on=['system_id', 'domain_id'], how='inner')
        for name, group in merged_domain_labels.groupby(['system_id', 'domain_id']):
            system_id, domain_id = name
            self.domain_to_valid_labels_map[(system_id, domain_id)] = group['label_id'].unique().tolist()


        # 3. Filter systems by J_domains
        # Count unique valid domains per system
        system_domain_counts = valid_domains_s_d.groupby('system_id')['domain_id'].nunique()
        runnable_systems_series = system_domain_counts[system_domain_counts >= self.j_domains]
        
        self.runnable_system_ids = runnable_systems_series.index.tolist()

        if len(self.runnable_system_ids) < self.m_systems:
            raise ValueError(
                f"Not enough systems meet the criteria. Need {self.m_systems} systems, "
                f"found {len(self.runnable_system_ids)} runnable systems. "
                f"Check M, J parameters and data distribution after N and K+Q filtering."
            )

        # Populate self.system_to_valid_domains_map
        # Filter valid_domains_s_d to only include those in self.runnable_system_ids
        # Then group by system_id and collect domain_ids
        final_valid_domains = valid_domains_s_d[valid_domains_s_d['system_id'].isin(self.runnable_system_ids)]
        for system_id, group in final_valid_domains.groupby('system_id'):
            self.system_to_valid_domains_map[system_id] = group['domain_id'].unique().tolist()


    def __iter__(self):
        if self.samples_df.empty and self.num_episodes > 0:
            print("Warning: samples_df is empty, cannot generate episodes.")
            return
        all_indices = []
        for _ in range(self.num_episodes):
            episode_indices = []
            
            if not self.runnable_system_ids:
                 print("Warning: No runnable systems available for sampling (should have been caught in init).")
                 return

            selected_system_ids = random.sample(self.runnable_system_ids, self.m_systems)
            
            for system_id in selected_system_ids:
                # available_domain_ids are pre-filtered to meet J and N requirements
                available_domain_ids = self.system_to_valid_domains_map.get(system_id, [])
                if len(available_domain_ids) < self.j_domains:
                    print(f"Warning: System {system_id} has fewer valid domains ({len(available_domain_ids)}) "
                          f"than required ({self.j_domains}) during iteration. This indicates an issue in "
                          f"_filter_and_prepare_for_sampling or data. Skipping system.")
                    continue
                
                selected_domain_ids = random.sample(available_domain_ids, self.j_domains)
                
                for domain_id in selected_domain_ids:
                    # available_label_ids are pre-filtered to meet N and K+Q requirements
                    available_label_ids = self.domain_to_valid_labels_map.get((system_id, domain_id), [])
                    if len(available_label_ids) < self.n_labels_way:
                        print(f"Warning: System-Domain ({system_id}-{domain_id}) has fewer valid labels "
                              f"({len(available_label_ids)}) than required ({self.n_labels_way}) "
                              f"during iteration. This indicates an issue. Skipping domain-task.")
                        continue

                    selected_label_ids = random.sample(available_label_ids, self.n_labels_way)
                    
                    for label_id in selected_label_ids:
                        # Query self.samples_df for global_idx for the specific task
                        # These indices are pre-filtered to ensure K+Q samples exist for this label_id
                        indices_for_task_series = self.samples_df[
                            (self.samples_df['system_id'] == system_id) &
                            (self.samples_df['domain_id'] == domain_id) &
                            (self.samples_df['label_id'] == label_id)
                        ]['global_idx']
                        
                        indices_for_task = indices_for_task_series.tolist()

                        if len(indices_for_task) < self.k_shot + self.q_query:
                             print(f"Warning: Task ({system_id}-{domain_id}-{label_id}) has fewer samples "
                                  f"({len(indices_for_task)}) than K+Q ({self.k_shot + self.q_query}) "
                                  f"during iteration. This indicates an issue. Skipping label.")
                             continue
                            
                        sampled_indices_for_task_instance = random.sample(
                            indices_for_task, self.k_shot + self.q_query
                        )
                        episode_indices.extend(sampled_indices_for_task_instance)
            
            if not episode_indices and self.m_systems > 0 :
                print(f"Warning: Generated an empty episode. Check sampling logic and data availability.")
            all_indices.append(episode_indices)
        for episode_indice in all_indices:
            yield episode_indice

    def __len__(self):
        # Total number of indices yielded by the iterator over all episodes
        return self.len 



class FewShotSampler(Sampler[int]):
    pass # TODO normal few-shot sampler for N-way K-shot tasks, not hierarchical

if __name__ == "__main__":
    # 1. Define a Mock IdIncludedDataset for testing
    class MockIdIncludedDataset(Dataset):
        def __init__(self, dataset_dict, metadata):
            self.dataset_dict = dataset_dict # Stores {file_id: list_of_dummy_samples}
            self.metadata = metadata     # Stores {file_id: {'Dataset_id': ..., 'Domain_id': ..., 'Label': ...}}
            self.file_windows_list = []
            
            current_global_idx = 0
            for file_id, original_dataset_items in self.dataset_dict.items():
                if file_id not in self.metadata:
                    print(f"Test data warning: File_id '{file_id}' not in metadata. Skipping.")
                    continue
                for _ in range(len(original_dataset_items)): # len() is important
                    # The sampler uses global_idx from enumerate(file_windows_list)
                    # So, the 'Window_id' here is more for mimicking the structure.
                    self.file_windows_list.append({'File_id': file_id, 'Window_id': current_global_idx}) 
                    current_global_idx += 1
            self._total_samples = len(self.file_windows_list)

        def get_file_windows_list(self):
            return self.file_windows_list

        def __len__(self):
            return self._total_samples

        def __getitem__(self, global_idx): # Not strictly used by sampler directly, but good for a Dataset
            if global_idx < 0 or global_idx >= self._total_samples:
                raise IndexError("Global index out of range")
            
            sample_info = self.file_windows_list[global_idx]
            file_id = sample_info['File_id']
            # window_id_in_original_dataset = sample_info['Window_id'] # if we need to get from original
            
            # For sampler, it only needs global_idx. This __getitem__ is for completeness.
            # It would typically return the actual data sample.
            return {
                "data": f"data_for_global_idx_{global_idx}", 
                "label": self.metadata[file_id]['Label'], # Example: get label from metadata
                "Id": file_id 
            }

    # 2. Setup mock data and metadata
    # K+Q = 2, N=2, J=1, M=1
    # System S1:
    #   Domain D1: L1 (3 samples), L2 (3 samples), L3 (1 sample - too few for K+Q)
    #   Domain D2: L1 (2 samples), L4 (2 samples), L0 (1 sample - too few for K+Q)
    # System S2:
    #   Domain D3: L5 (3 samples), L6 (3 samples)
    #   Domain D4: L7 (1 sample - too few for K+Q)
    # System S3 (not enough valid domains if J > 1, or not enough labels if N > 0 for its domains)
    #   Domain D5: L8 (3 samples) - only 1 label, fails N=2

    dataset_dict_mock = {
        # S1/D1
        's1d1l1_file': [None]*3, # File_id, list of dummy items (length matters)
        's1d1l2_file': [None]*3,
        's1d1l3_file': [None]*1, # Not enough for K+Q=2
        # S1/D2
        's1d2l1_file': [None]*2,
        's1d2l4_file': [None]*2,
        's1d2l0_file': [None]*1, # Not enough for K+Q=2
        # S2/D3
        's2d3l5_file': [None]*3,
        's2d3l6_file': [None]*3,
        # S2/D4
        's2d4l7_file': [None]*1, # Not enough for K+Q=2
        # S3/D5
        's3d5l8_file': [None]*3, # Domain D5 will only have 1 valid label (L8), fails N=2
    }

    metadata_mock = {
        # S1/D1
        's1d1l1_file': {'Dataset_id': 'S1', 'Domain_id': 'D1', 'Label': 'L1'},
        's1d1l2_file': {'Dataset_id': 'S1', 'Domain_id': 'D1', 'Label': 'L2'},
        's1d1l3_file': {'Dataset_id': 'S1', 'Domain_id': 'D1', 'Label': 'L3'},
        # S1/D2
        's1d2l1_file': {'Dataset_id': 'S1', 'Domain_id': 'D2', 'Label': 'L1'}, # Re-using L1 in different domain
        's1d2l4_file': {'Dataset_id': 'S1', 'Domain_id': 'D2', 'Label': 'L4'},
        's1d2l0_file': {'Dataset_id': 'S1', 'Domain_id': 'D2', 'Label': 'L0'},
        # S2/D3
        's2d3l5_file': {'Dataset_id': 'S2', 'Domain_id': 'D3', 'Label': 'L5'},
        's2d3l6_file': {'Dataset_id': 'S2', 'Domain_id': 'D3', 'Label': 'L6'},
        # S2/D4
        's2d4l7_file': {'Dataset_id': 'S2', 'Domain_id': 'D4', 'Label': 'L7'},
        # S3/D5
        's3d5l8_file': {'Dataset_id': 'S3', 'Domain_id': 'D5', 'Label': 'L8'},
    }
    
    mock_dataset = MockIdIncludedDataset(dataset_dict_mock, metadata_mock)

    print(f"Total samples in mock_dataset: {len(mock_dataset)}")
    # print("Mock dataset file_windows_list (first 5):")
    # for i, item in enumerate(mock_dataset.get_file_windows_list()[:5]):
    #     print(f"  Global Idx {i}: {item}")


    # 3. Sampler parameters
    params = {
        "num_episodes": 3,
        "num_systems_per_episode": 1,    # M
        "num_domains_per_system": 1,     # J
        "num_labels_per_domain_task": 2, # N (N-way)
        "num_support_per_label": 1,      # K
        "num_query_per_label": 1,        # Q
    }
    print(f"\nSampler Parameters: {params}")

    try:
        sampler = HierarchicalFewShotSampler(
            dataset=mock_dataset,
            num_episodes=params["num_episodes"],
            num_systems_per_episode=params["num_systems_per_episode"],
            num_domains_per_system=params["num_domains_per_system"],
            num_labels_per_domain_task=params["num_labels_per_domain_task"],
            num_support_per_label=params["num_support_per_label"],
            num_query_per_label=params["num_query_per_label"],
            system_metadata_key='Dataset_id',
            domain_metadata_key='Domain_id',
            label_metadata_key='Label'
        )

        print(f"\nRunnable systems identified: {sampler.runnable_system_ids}")
        print(f"System to valid domains map: {dict(sampler.system_to_valid_domains_map)}")
        print(f"Domain to valid labels map: {dict(sampler.domain_to_valid_labels_map)}")
        # print(f"Full samples_df head:\n{sampler.samples_df.head()}")


        samples_per_episode = (params["num_systems_per_episode"] *
                               params["num_domains_per_system"] *
                               params["num_labels_per_domain_task"] *
                               (params["num_support_per_label"] + params["num_query_per_label"]))
        
        print(f"\nExpected samples per episode: {samples_per_episode}")
        print(f"Expected total samples from sampler (len(sampler)): {len(sampler)}")

        print("\nIterating through sampler:")
        collected_episodes = []
        current_episode_indices = []
        total_yielded_indices = 0
        for global_idx in sampler:
            current_episode_indices.append(global_idx)
            total_yielded_indices += 1
            if len(current_episode_indices) == samples_per_episode:
                collected_episodes.append(list(current_episode_indices)) # Store a copy
                current_episode_indices = [] # Reset for next episode
        
        if current_episode_indices: # Should be empty if num_episodes * samples_per_episode is correct
             print(f"Warning: Trailing indices found: {current_episode_indices}")
             collected_episodes.append(list(current_episode_indices))


        for i, episode_indices in enumerate(collected_episodes):
            print(f"  Episode {i}: {episode_indices} (Length: {len(episode_indices)})")
            # You can add more detailed checks here, e.g., verify that the indices
            # correspond to the correct M, J, N, K, Q structure if you map them back
            # using sampler.samples_df.iloc[global_idx]

        print(f"\nTotal episodes generated: {len(collected_episodes)}")
        print(f"Total indices yielded: {total_yielded_indices}")

        if total_yielded_indices != len(sampler):
            print(f"Error: Total yielded indices ({total_yielded_indices}) "
                  f"does not match len(sampler) ({len(sampler)})")

        # Example of how to map global_idx back to info using samples_df
        if collected_episodes and sampler.samples_df is not None and not sampler.samples_df.empty:
            print("\nDetails for the first sample of the first episode:")
            first_idx = collected_episodes[0][0]
            sample_details = sampler.samples_df[sampler.samples_df['global_idx'] == first_idx].iloc[0]
            original_file_id = mock_dataset.get_file_windows_list()[first_idx]['File_id']
            print(f"  Global Idx: {first_idx}")
            print(f"  Mapped via sampler.samples_df: System={sample_details['system_id']}, Domain={sample_details['domain_id']}, Label={sample_details['label_id']}")
            print(f"  Original File ID from mock_dataset: {original_file_id}")
            print(f"  Metadata for this File ID: {metadata_mock[original_file_id]}")


    except ValueError as e:
        print(f"\nError during sampler instantiation or filtering: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during testing: {e}")
