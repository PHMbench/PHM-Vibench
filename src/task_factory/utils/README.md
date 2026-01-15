# Task Factory Utilities

This module provides core data processing utilities for the task factory, including windowing, normalization, and batch preparation for time-series signals.

## Overview

The utilities module contains fundamental functions used by various task implementations:
- **Windowing**: Split long signals into fixed-size windows
- **Normalization**: Standardize signal data for model input
- **Batch Preparation**: Convert raw data into PyTorch tensors

---

## File Structure

| File | Description |
|------|-------------|
| `data_processing.py` | Core processing functions for time-series data |

---

## API Reference

### 1. create_windows()

**Purpose**: Split a signal array into fixed-size windows with configurable sampling strategy.

**Function Signature**:
```python
def create_windows(data: np.ndarray, args_data: Any) -> List[np.ndarray]:
    """
    Split signal into windows of size window_size.

    Args:
        data: Raw signal array of shape (L, C) or (L,)
        args_data: Should define window_size, stride, num_window
                  and optionally window_sampling_strategy

    Returns:
        List of windows with length window_size
    """
```

**Parameters** (from `args_data`):
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_size` | int | Required | Length of each window |
| `stride` | int | Required | Step between windows |
| `num_window` | int | Required | Maximum number of windows to extract |
| `window_sampling_strategy` | str | `'evenly_spaced'` | Sampling strategy |

**Sampling Strategies**:

| Strategy | Description |
|----------|-------------|
| `'sequential'` | Consecutive windows from start to end |
| `'random'` | Randomly sample windows from the signal |
| `'evenly_spaced'` | Distribute windows evenly across the signal |

**Example Usage**:
```python
from src.task_factory.utils.data_processing import create_windows

# Signal of length 10000
signal = np.random.randn(10000)

# Configuration
class ArgsData:
    window_size = 1024
    stride = 512
    num_window = 10
    window_sampling_strategy = 'evenly_spaced'

windows = create_windows(signal, ArgsData())
# Returns: list of 10 windows, each of shape (1024,)
```

---

### 2. process_sample()

**Purpose**: Normalize and reshape a signal sample for model input.

**Function Signature**:
```python
def process_sample(data: np.ndarray, args_data: Any) -> np.ndarray:
    """
    Normalize and reshape one sample.

    Args:
        data: Raw array from the dataset
        args_data: Contains dtype and normalization configuration

    Returns:
        Processed array suitable for model input
    """
```

**Parameters** (from `args_data`):
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dtype` | str | `None` | Cast to `'float32'` or `'float64'` |
| `normalization` | str | `None` | `'minmax'` or `'standardization'` |

**Normalization Methods**:

| Method | Formula |
|--------|---------|
| `'minmax'` | `(x - min) / (max - min)` |
| `'standardization'` | `(x - mean) / (std + 1e-8)` |

**Example Usage**:
```python
from src.task_factory.utils.data_processing import process_sample

# Raw signal
signal = np.random.randn(5000, 2)  # 5000 samples, 2 channels

# Configuration
class ArgsData:
    dtype = 'float32'
    normalization = 'standardization'

processed = process_sample(signal, ArgsData())
# Returns: normalized array of shape (5000, 2)
```

---

### 3. prepare_batch()

**Purpose**: Transform a raw batch from ID_dataset into PyTorch tensors for training.

**Function Signature**:
```python
def prepare_batch(batch: Dict[str, Any], args_data: Any) -> Dict[str, Any]:
    """
    Transform raw batch from ID_dataset.

    Args:
        batch: Dict with keys 'data', 'metadata', 'id' containing lists
        args_data: Passed to process_sample and create_windows

    Returns:
        Dict with keys:
        - 'x': Tensor of shape (batch, window_size, channels)
        - 'y': Tensor of labels
        - 'file_id': List of file IDs
    """
```

**Input Format**:
```python
batch = {
    'data': [array1, array2, ...],      # Raw signal arrays
    'metadata': [meta1, meta2, ...],     # Metadata dicts with 'Label' key
    'id': ['file1', 'file2', ...]       # File identifiers
}
```

**Output Format**:
```python
{
    'x': Tensor,        # Shape: (batch_size, window_size, channels)
    'y': Tensor,        # Shape: (batch_size,) - class labels
    'file_id': list     # File identifiers for metadata lookup
}
```

**Example Usage**:
```python
from src.task_factory.utils.data_processing import prepare_batch

# Raw batch from ID_dataset
batch = {
    'data': [np.random.randn(5000), np.random.randn(5000)],
    'metadata': [{'Label': 0}, {'Label': 1}],
    'id': ['sample_001', 'sample_002']
}

# Process batch
processed = prepare_batch(batch, args_data)

# Use in training loop
x = processed['x']        # Input tensor
y = processed['y']        # Labels
file_ids = processed['file_id']
```

---

## Configuration Examples

### Windowing Configuration
```yaml
data:
  # Windowing parameters
  window_size: 1024
  stride: 512
  num_window: 10
  window_sampling_strategy: "evenly_spaced"  # sequential/random/evenly_spaced

  # Preprocessing
  dtype: "float32"
  normalization: "standardization"  # minmax/standardization
```

### Complete Pipeline
```python
# 1. Load raw data
raw_signal = np.random.randn(10000, 1)

# 2. Normalize
processed = process_sample(raw_signal, args_data)

# 3. Create windows
windows = create_windows(processed, args_data)

# 4. Prepare batch for training
batch_dict = prepare_batch(raw_batch, args_data)

# 5. Use in training
output = model(batch_dict['x'])
loss = criterion(output, batch_dict['y'])
```

---

## Design Notes

### Windowing Strategy Selection

- **`sequential`**: Use when temporal order is important (e.g., prediction tasks)
- **`random`**: Use for data augmentation and training diversity
- **`evenly_spaced`**: Default strategy, provides good signal coverage

### Memory Efficiency

The `prepare_batch` function processes one window per sample (the first window). For tasks requiring all windows:
- Use the ID task module which handles multiple windows per sample
- See [@../task/ID/README.md] for details

### Data Type Handling

Always specify `dtype='float32'` for:
- Reduced memory usage
- Faster computation on most GPUs
- Compatibility with mixed precision training

---

## Related Documentation

- [@../README.md] - Task Factory Overview
- [@../task/ID/README.md] - ID Task (uses these utilities)
- [@../../data_factory/README.md] - Data Factory (source of raw data)

---

## Module Structure

```
src/task_factory/utils/
├── __init__.py            # Module exports
└── data_processing.py     # Core processing functions
```

---

## Best Practices

1. **Choose appropriate window_size**: Match signal characteristics (e.g., 2-4x the period)
2. **Use 50% overlap**: Set `stride = window_size / 2` for good coverage
3. **Always normalize**: Use `standardization` for most ML models
4. **Handle edge cases**: Signals shorter than `window_size` return empty lists

---

## Troubleshooting

### Empty windows returned
- Check if `data_length >= window_size`
- Verify `args_data` has required parameters

### Shape mismatches
- Ensure input is 2D: `(length, channels)` or 1D: `(length,)`
- Check `num_window` is not too large for your signal

### Normalization issues
- For `minmax`: Check if max == min (division by zero handled)
- For `standardization`: Small std values are handled with epsilon
