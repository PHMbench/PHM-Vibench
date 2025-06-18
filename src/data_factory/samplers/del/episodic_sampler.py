# src/data_factory/samplers/episodic_sampler.py

import torch
import numpy as np
from torch.utils.data import Sampler
from collections import defaultdict

class Sampler(Sampler):
    """
    Episodic Sampler for Few-Shot Learning.
    Generates batches of indices for N-way K-shot tasks.

    Args:
        dataset_labels (list or np.ndarray): A list of labels for each sample in the dataset.
        n_episodes (int): The total number of episodes to generate in one epoch.
        n_way (int): The number of classes per episode (N-way).
        k_shot_support (int): The number of support samples per class (K-shot).
        k_shot_query (int): The number of query samples per class.
    """
    def __init__(self, dataset_labels, n_episodes, n_way, k_shot_support, k_shot_query):
        super().__init__(dataset_labels)
        self.n_episodes = n_episodes
        self.n_way = n_way
        self.k_shot_support = k_shot_support
        self.k_shot_query = k_shot_query

        # Organize indices by class
        self.label_to_indices = defaultdict(list)
        for idx, label in enumerate(dataset_labels):
            self.label_to_indices[label].append(idx)

        self.labels = list(self.label_to_indices.keys())

        # Ensure there are enough classes and samples for the task
        if len(self.labels) < self.n_way:
            raise ValueError(f"The number of classes in the dataset ({len(self.labels)}) is less than n_way ({self.n_way}).")
        
        for label, indices in self.label_to_indices.items():
            if len(indices) < self.k_shot_support + self.k_shot_query:
                raise ValueError(f"Class {label} has only {len(indices)} samples, but {self.k_shot_support + self.k_shot_query} are required for support and query sets.")


    def __len__(self):
        """
        Returns the total number of episodes per epoch.
        """
        return self.n_episodes

    def __iter__(self):
        """
        Yields a list of indices for one episode.
        """
        for _ in range(self.n_episodes):
            episode_indices = []
            
            # 1. Randomly select N classes for the episode
            selected_classes = np.random.choice(self.labels, size=self.n_way, replace=False)

            for class_label in selected_classes:
                all_indices_for_class = self.label_to_indices[class_label]
                
                # 2. Randomly select K+Q samples from each class
                num_samples_to_select = self.k_shot_support + self.k_shot_query
                selected_indices = np.random.choice(all_indices_for_class, size=num_samples_to_select, replace=False)
                
                episode_indices.extend(selected_indices)
            
            yield episode_indices