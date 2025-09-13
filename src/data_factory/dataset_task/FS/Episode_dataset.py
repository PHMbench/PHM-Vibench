# src/data_factory/dataset_task/FewShot/classification_dataset.py

import torch
from ..Default_dataset import Default_dataset

class set_dataset(Default_dataset):
    """
    Dataset for Few-Shot classification tasks.
    It takes a list of indices for a whole episode from the sampler
    and returns a structured dictionary containing support and query sets.
    """
    def __init__(self, config, h5_path, data_type, transform=None):
        super().__init__(config, h5_path, data_type, transform)
        
        # Get few-shot parameters from config
        self.n_way = config['data']['n_way']
        self.k_shot_support = config['data']['k_shot_support']
        self.k_shot_query = config['data']['k_shot_query']
        
        # The total number of samples per episode
        self.samples_per_episode = self.n_way * (self.k_shot_support + self.k_shot_query)

    def __getitem__(self, episode_indices):
        """
        Args:
            episode_indices (list of int): A list of indices for one full episode,
                                           provided by the EpisodicSampler.

        Returns:
            dict: A dictionary containing structured support and query sets.
        """
        if len(episode_indices) != self.samples_per_episode:
             raise ValueError(f"Expected {self.samples_per_episode} indices for an episode, but got {len(episode_indices)}.")

        # Load all data and labels for the episode using the parent's logic
        # super().__getitem__ handles loading a single sample, so we call it in a loop.
        all_data = torch.stack([super().__getitem__(idx)[0] for idx in episode_indices])
        all_labels = torch.tensor([super().__getitem__(idx)[1] for idx in episode_indices])
        
        # Reshape and split the data into support and query sets
        # The shape will be (N, K+Q, Channels, Length)
        all_data = all_data.view(self.n_way, self.k_shot_support + self.k_shot_query, *all_data.shape[1:])
        all_labels = all_labels.view(self.n_way, self.k_shot_support + self.k_shot_query)

        # Split into support and query
        support_x = all_data[:, :self.k_shot_support]
        query_x = all_data[:, self.k_shot_support:]
        
        support_y = all_labels[:, :self.k_shot_support]
        query_y = all_labels[:, self.k_shot_support:]

        # Create a mapping from original labels to episode-local labels (0 to N-1)
        # This simplifies the loss calculation in the training step.
        unique_labels = torch.unique(support_y)
        label_map = {original_label.item(): new_label for new_label, original_label in enumerate(unique_labels)}
        
        # Apply the mapping
        support_y = torch.tensor([label_map[l.item()] for l in support_y.flatten()]).view(support_y.shape)
        query_y = torch.tensor([label_map[l.item()] for l in query_y.flatten()]).view(query_y.shape)

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
        }