import numpy as np
from ..Default_dataset import Default_dataset

class set_dataset(Default_dataset):
    """Dataset for time series prediction used in pretraining."""
    def __init__(self, data, metadata, args_data, args_task, mode="train"):
        self.pred_len = getattr(args_task, 'pred_len', 1)
        super().__init__(data, metadata, args_data, args_task, mode)

    def _process_single_data(self, sample_data):
        if self.args_data.dtype:
            if self.args_data.dtype == 'float32':
                sample_data = sample_data.astype(np.float32)
            elif self.args_data.dtype == 'float64':
                sample_data = sample_data.astype(np.float64)
        if sample_data.ndim == 3:
            sample_data = sample_data.reshape(sample_data.shape[0], -1)

        data_length = len(sample_data)
        num_samples = max(0, (data_length - self.window_size - self.pred_len) // self.stride + 1)
        num_samples = min(num_samples, self.args_data.num_window)

        for i in range(num_samples):
            start_idx = i * self.stride
            end_idx = start_idx + self.window_size
            pred_end = end_idx + self.pred_len
            if pred_end > data_length:
                break
            x = sample_data[start_idx:end_idx]
            y = sample_data[end_idx:pred_end]
            self.processed_data.append({'x': x, 'y': y})

    def __getitem__(self, idx):
        if idx >= self.total_samples:
            raise IndexError(f"索引 {idx} 超出范围")
        return self.processed_data[idx]
