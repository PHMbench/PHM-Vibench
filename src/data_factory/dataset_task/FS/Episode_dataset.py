import torch

from ..Default_dataset import Default_dataset


class set_dataset(Default_dataset):
    """
    Episode-style dataset for few-shot tasks.

    The default data factory instantiates dataset_task classes with
    (data, metadata, args_data, args_task, mode). This class keeps that
    contract and also supports direct episode-index access for custom
    episodic samplers.
    """

    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        super().__init__(data, metadata, args_data, args_task, mode)
        self.n_way = int(getattr(args_task, "n_way", 1))
        self.k_shot_support = int(
            getattr(args_task, "k_shot_support", getattr(args_task, "k_shot", 1))
        )
        self.k_shot_query = int(
            getattr(args_task, "k_shot_query", getattr(args_task, "q_query", 1))
        )
        self.samples_per_episode = self.n_way * (self.k_shot_support + self.k_shot_query)

    def __getitem__(self, index):
        if isinstance(index, (int, torch.Tensor)):
            if isinstance(index, torch.Tensor):
                index = int(index.item())
            return super().__getitem__(index)

        episode_indices = list(index)
        if len(episode_indices) != self.samples_per_episode:
            raise ValueError(
                f"Expected {self.samples_per_episode} indices for an episode, "
                f"but got {len(episode_indices)}."
            )

        items = [Default_dataset.__getitem__(self, int(idx)) for idx in episode_indices]
        all_data = torch.stack([torch.as_tensor(item["x"]) for item in items])
        all_labels = torch.as_tensor([item["y"] for item in items], dtype=torch.long)

        per_class = self.k_shot_support + self.k_shot_query
        all_data = all_data.view(self.n_way, per_class, *all_data.shape[1:])
        all_labels = all_labels.view(self.n_way, per_class)

        support_x = all_data[:, : self.k_shot_support].reshape(
            self.n_way * self.k_shot_support, *all_data.shape[2:]
        )
        query_x = all_data[:, self.k_shot_support :].reshape(
            self.n_way * self.k_shot_query, *all_data.shape[2:]
        )
        support_y = all_labels[:, : self.k_shot_support]
        query_y = all_labels[:, self.k_shot_support :]

        unique_labels = torch.unique(support_y, sorted=True)
        if len(unique_labels) != self.n_way:
            raise ValueError(
                f"Expected {self.n_way} unique support labels, got {len(unique_labels)}."
            )
        label_map = {int(label.item()): idx for idx, label in enumerate(unique_labels)}

        support_y = torch.as_tensor(
            [label_map[int(label.item())] for label in support_y.flatten()],
            dtype=torch.long,
        )
        query_y = torch.as_tensor(
            [label_map[int(label.item())] for label in query_y.flatten()],
            dtype=torch.long,
        )

        return {
            "support_x": support_x,
            "support_y": support_y,
            "query_x": query_x,
            "query_y": query_y,
            "n_way": self.n_way,
            "n_shot": self.k_shot_support,
            "n_query": self.k_shot_query,
        }
