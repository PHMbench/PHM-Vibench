# ID Task Module

This module provides a memory-efficient task implementation for processing time-series data with flexible windowing, batching, and preprocessing capabilities.

## Overview

The ID Task is designed for large-scale industrial fault diagnosis experiments where:
- **Memory efficiency** is critical (raw signals are processed on-demand)
- **Flexible windowing** is needed (variable-length sequences)
- **Dynamic preprocessing** is required during training

**Key Features**:
- Flexible windowing strategies (sequential, random, evenly_spaced)
- Configurable preprocessing pipelines
- Extensible batching for variable-length sequences
- Memory-efficient processing for large datasets
- Support for multi-channel time-series data

---

## Architecture

```
Input: (l, c) raw signal
         ↓
    create_windows()  → (w, window_l, c)
         ↓
    process_sample()  → Normalized/processed windows
         ↓
    prepare_batch()    → (b, w, window_l, c)
         ↓
         Model
```

---

## Core Components

### 1. BaseIDTask Class

**Base Class**: `BaseIDTask(Default_task, ABC)`

**Initialization**:
```python
BaseIDTask(
    network: torch.nn.Module,
    args_data: Any,
    args_model: Any,
    args_task: Any,
    args_trainer: Any,
    args_environment: Any,
    metadata: Any
)
```

**Key Methods**:

| Method | Purpose |
|--------|---------|
| `create_windows(signal)` | Transform (l, c) → (w, window_l, c) |
| `process_sample(signal)` | Individual sample preprocessing |
| `prepare_batch(batch)` | Extensible batching with uniform output |
| `training_step(batch, idx)` | PyTorch Lightning training step |

---

## Configuration

### Task Configuration

```yaml
task:
  name: "ID_task"
  type: "ID"

  # Windowing parameters
  window_size: 1024           # Length of each window
  stride: 512                 # Step between windows
  window_sampling: "evenly_spaced"  # Strategy: sequential/random/evenly_spaced

  # Preprocessing
  normalization: true
  normalize_method: "z_score"  # z_score, minmax, robust

  # Batching
  batch_size: 32
  max_windows_per_sample: 100  # Limit windows per sample
```

### Windowing Strategies

| Strategy | Description |
|----------|-------------|
| `sequential` | Take consecutive windows from start to end |
| `random` | Randomly sample windows from the signal |
| `evenly_spaced` | Distribute windows evenly across the signal |

---

## Usage Example

### Basic ID Task

```python
from src.task_factory.task.ID.ID_task import BaseIDTask

# The task is automatically instantiated by task_factory
# when you specify task.type = "ID" in your config

# Configuration:
task:
  type: "ID"
  name: "ID_classification"
  window_size: 1024
  stride: 512
```

### Custom ID Task

```python
from src.task_factory.task.ID.ID_task import BaseIDTask
from src.task_factory import register_task

@register_task("ID", "custom_id_task")
class CustomIDTask(BaseIDTask):
    def create_windows(self, signal):
        # Custom windowing logic
        windows = []
        # ... your implementation
        return windows

    def process_sample(self, signal):
        # Custom preprocessing
        processed = super().process_sample(signal)
        # ... additional processing
        return processed
```

---

## Data Processing Pipeline

### 1. Window Creation (`create_windows`)

**Input**: Signal with shape `(length, channels)`

**Output**: Windows with shape `(num_windows, window_length, channels)`

```python
def create_windows(self, signal):
    length, channels = signal.shape

    # Calculate number of windows
    num_windows = (length - self.window_size) // self.stride + 1

    # Extract windows
    windows = []
    for i in range(num_windows):
        start = i * self.stride
        end = start + self.window_size
        window = signal[start:end, :]
        windows.append(window)

    return np.array(windows)
```

### 2. Sample Processing (`process_sample`)

**Input**: Raw signal array

**Output**: Processed windows ready for batching

```python
def process_sample(self, signal):
    windows = self.create_windows(signal)

    # Apply normalization
    if self.normalization:
        windows = self.normalize(windows)

    return windows
```

### 3. Batch Preparation (`prepare_batch`)

**Input**: List of variable-length window sequences

**Output**: Uniform batch tensor `(batch, windows, window_length, channels)`

```python
def prepare_batch(self, batch):
    # Pad or truncate to ensure uniform size
    max_windows = max(len(x) for x in batch)

    padded = []
    for x in batch:
        if len(x) < max_windows:
            # Pad with zeros
            padded_x = np.pad(x, ...)
        else:
            # Truncate
            padded_x = x[:max_windows]
        padded.append(padded_x)

    return torch.tensor(padded)
```

---

## Processing Statistics

The task tracks processing statistics:

```python
task.processing_stats = {
    'total_samples_processed': 0,
    'total_windows_created': 0,
    'failed_samples': 0,
    'average_windows_per_sample': 0.0
}
```

Access during training:
```python
print(f"Avg windows per sample: {task.processing_stats['average_windows_per_sample']}")
```

---

## Registration

The ID task is registered in `src/task_factory/task_registry.csv`:

```csv
task_type,task_name,module_path,notes
ID,ID_task,src.task_factory.task.ID.ID_task,Memory-efficient ID-based task
```

---

## Related Documentation

- [@../README.md] - Task Factory Overview
- [@../../README.md] - Task Factory Root Documentation
- [@../DG/README.md] - Domain Generalization Tasks
- [@../FS/README.md] - Few-Shot Learning Tasks

---

## Dependencies

- `src/task_factory/utils/data_processing.py`: Core processing functions
- `src/task_factory/Default_task.py`: Base task class
- `src/task_factory/Components/`: Loss functions and metrics

---

## Best Practices

1. **Window Size**: Choose based on your signal's periodicity
   - Too small → Lose context
   - Too large → Computationally expensive

2. **Stride**: Use 50% overlap (stride = window_size / 2) for good coverage

3. **Normalization**: Always normalize signals before feeding to model

4. **Memory**: Limit `max_windows_per_sample` for very long signals

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `window_size` or `max_windows_per_sample`
   - Decrease `batch_size`

2. **Poor Performance**:
   - Check windowing strategy matches signal characteristics
   - Verify normalization is applied correctly

3. **Inconsistent Batch Shapes**:
   - Ensure `prepare_batch` handles padding/truncation
