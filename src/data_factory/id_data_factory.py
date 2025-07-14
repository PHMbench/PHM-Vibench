"""Data factory specialized for ID datasets."""

from .data_factory import data_factory
from torch.utils.data import DataLoader


class id_data_factory(data_factory):
    """Data factory that initializes default dataloaders."""

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        """Return a DataLoader with default parameters for ``dataset``."""
        return DataLoader(
            dataset,
            batch_size=self.args_data.batch_size,
            shuffle=shuffle,
            num_workers=self.args_data.num_workers,
            pin_memory=True,
        )

    def _init_dataloader(self):
        """Initialize train/val/test dataloaders using default sampler."""
        self.train_loader = self._make_loader(self.train_dataset, shuffle=True)
        self.val_loader = self._make_loader(self.val_dataset, shuffle=False)
        self.test_loader = self._make_loader(self.test_dataset, shuffle=False)
        return self.train_loader, self.val_loader, self.test_loader
