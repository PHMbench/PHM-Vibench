import random
from typing import List
from torch.utils.data import Dataset
from ..Default_dataset import Default_dataset
import torch

class set_dataset(Dataset):
    """Generate few-shot episodes from a base dataset.

    Each item is a dict with support and query sets.
    The base dataset must return dicts containing 'x' and 'y'.
    """

    def __init__(self, base_dataset: Dataset, n_way: int, k_shot: int, q_query: int, episodes_per_epoch: int = 100):
        self.base = base_dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.episodes_per_epoch = episodes_per_epoch
        self.label_to_indices = {}
        for idx in range(len(self.base)):
            label = int(self.base[idx]['y'])
            self.label_to_indices.setdefault(label, []).append(idx)
        self.labels = list(self.label_to_indices.keys())

    def __len__(self) -> int:
        return self.episodes_per_epoch

    def __getitem__(self, index):
        selected = random.sample(self.labels, self.n_way)
        support_x, support_y = [], []
        query_x, query_y = [], []
        for lab in selected:
            indices = random.sample(self.label_to_indices[lab], self.k_shot + self.q_query)
            for i in range(self.k_shot):
                item = self.base[indices[i]]
                support_x.append(item['x'])
                support_y.append(lab)
            for i in range(self.k_shot, self.k_shot + self.q_query):
                item = self.base[indices[i]]
                query_x.append(item['x'])
                query_y.append(lab)
        return {
            'support_x': torch.stack(support_x),
            'support_y': torch.tensor(support_y),
            'query_x': torch.stack(query_x),
            'query_y': torch.tensor(query_y)
        }
# if __name__ == "__main__":
