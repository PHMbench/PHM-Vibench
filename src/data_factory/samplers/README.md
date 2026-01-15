# Data Samplers Module

This module provides specialized data samplers for few-shot learning, domain generalization, and balanced sampling strategies in PHM-Vibench.

## Overview

Samplers control how data samples are grouped into batches during training. This module implements several sampling strategies tailored for industrial fault diagnosis tasks, particularly for:
- **Few-Shot Learning**: Episode-based sampling for meta-learning
- **Domain Generalization**: System/domain-aware batch sampling
- **Balanced Sampling**: Ensuring equal representation across datasets

---

## File Structure

| File | Description |
|------|-------------|
| `FS_sampler.py` | HierarchicalFewShotSampler for episodic few-shot tasks |
| `Sampler.py` | Same_system_Sampler, BalancedIdSampler, GroupedIdBatchSampler |
| `Get_sampler.py` | Factory function to retrieve appropriate sampler |

---

## API Reference

### 1. HierarchicalFewShotSampler (`FS_sampler.py`)

**Purpose**: Implements hierarchical episode sampling for Few-Shot and Generalized Few-Shot learning tasks.

**Hierarchy Structure**:
```
Systems (M) → Domains per system (J) → Labels per domain (N)
                ↓
         Support samples (K) + Query samples (Q)
```

**Parameters**:
```python
HierarchicalFewShotSampler(
    dataset: Dataset,                    # IdIncludedDataset instance
    num_episodes: int,                   # Total episodes to generate
    num_systems_per_episode: int,        # M: systems per episode
    num_domains_per_system: int,         # J: domains per system
    num_labels_per_domain_task: int,     # N: N-way classification
    num_support_per_label: int,          # K: K-shot support samples
    num_query_per_label: int,            # Q: query samples
    system_metadata_key: str = 'Dataset_id',
    domain_metadata_key: str = 'Domain_id',
    label_metadata_key: str = 'Label'
)
```

**Usage Example**:
```python
from src.data_factory.samplers.FS_sampler import HierarchicalFewShotSampler

sampler = HierarchicalFewShotSampler(
    dataset=train_dataset,
    num_episodes=100,
    num_systems_per_episode=2,     # M=2 systems
    num_domains_per_system=2,      # J=2 domains per system
    num_labels_per_domain_task=5,  # N=5-way classification
    num_support_per_label=5,       # K=5-shot
    num_query_per_label=10         # Q=10 query samples
)

train_loader = DataLoader(dataset, batch_sampler=sampler, ...)
```

---

### 2. Same_system_Sampler (`Sampler.py`)

**Purpose**: Ensures all samples in a batch come from the same system (Dataset_id).

**Parameters**:
```python
Same_system_Sampler(
    dataset: IdIncludedDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
    system_metadata_key: str = 'Dataset_id'
)
```

**Behavior**:
- Groups samples by `system_metadata_key`
- Creates batches containing samples from only one system
- Shuffles systems and samples within systems when `shuffle=True`

---

### 3. BalancedIdSampler (`Sampler.py`)

**Purpose**: Balances sampling across different original datasets (IDs) to prevent larger datasets from dominating.

**Parameters**:
```python
BalancedIdSampler(
    data_source: IdIncludedDataset,
    batch_size: int = 32,
    common_samples_per_id: int = None,  # If None, uses max ID size
    shuffle_within_id: bool = True,
    shuffle_all: bool = True
)
```

**Behavior**:
- Oversamples smaller IDs to match the largest ID
- Ensures equal representation of all datasets in each epoch

---

### 4. GroupedIdBatchSampler (`Sampler.py`)

**Purpose**: Creates batches where all samples come from the same original file/file_id.

**Parameters**:
```python
GroupedIdBatchSampler(
    data_source: IdIncludedDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False
)
```

**Use Case**: Useful when samples from the same file should stay together (e.g., for temporal coherence).

---

### 5. Get_sampler (`Get_sampler.py`)

**Purpose**: Factory function that returns the appropriate sampler based on task configuration.

**Function Signature**:
```python
def Get_sampler(args_task, args_data, dataset, mode='train'):
    """
    Returns sampler based on task type:
    - GFS/FS: HierarchicalFewShotSampler
    - CDDG: Same_system_Sampler with CDDG-specific settings
    - Default: Standard PyTorch sampler
    """
```

---

## Task-Specific Sampler Selection

| Task Type | Sampler | Configuration |
|-----------|---------|---------------|
| **FS** (Few-Shot) | `HierarchicalFewShotSampler` | Episode-based, M×J×N×K×Q |
| **GFS** (Generalized FS) | `HierarchicalFewShotSampler` | Same as FS |
| **CDDG** (Cross-Domain DG) | `Same_system_Sampler` | System-aware batching |
| **DG** (Domain Generalization) | Default | Standard sampler |
| **Pretrain** | Default | Standard sampler |

---

## Usage in Data Factory

Samplers are automatically selected by `data_factory.py`:

```python
# In data_factory.py
from .samplers.Get_sampler import Get_sampler

sampler = Get_sampler(self.args_task, self.args_data, dataset, mode)

train_loader = DataLoader(
    dataset,
    batch_sampler=sampler if sampler else None,
    batch_size=self.args_data.batch_size if sampler is None else None,
    ...
)
```

---

## Configuration Example

```yaml
# For Few-Shot learning
task:
  name: "GFS"
  type: "FS"

  # Few-shot parameters (passed to sampler)
  num_shots: 5              # K
  num_query: 15             # Q
  num_way: 5                # N
  num_episodes: 100         # Total episodes
```

---

## Related Documentation

- [@../README.md] - Data Factory Module Overview
- [@../../task_factory/README.md] - Task Factory (Task types that use samplers)
- [../dataset_task/README.md] - Dataset task implementations

---

## Implementation Notes

1. **Dataset Requirements**: All samplers require `IdIncludedDataset` with metadata accessible via `dataset.metadata[file_id]`.

2. **Metadata Keys**: Ensure your metadata contains the required keys:
   - `Dataset_id` / `system_metadata_key`: System/source identifier
   - `Domain_id` / `domain_metadata_key`: Domain identifier
   - `Label` / `label_metadata_key`: Class label

3. **Performance**: HierarchicalFewShotSampler pre-computes episode structure for efficiency.
