# ID Query Module

This module provides functions for querying and filtering sample IDs based on metadata conditions. It is used by the data factory to select specific samples for training, validation, and testing in domain generalization and cross-domain tasks.

## Overview

The ID module enables fine-grained control over which samples are used in experiments by:
- Filtering samples by system (Dataset_id), domain (Domain_id), and label (Label)
- Dynamically splitting domains for domain generalization tasks
- Handling invalid labels and missing metadata

---

## File Structure

| File | Description |
|------|-------------|
| `Get_id.py` | Domain-based ID selection for DG/CDDG tasks |
| `Id_searcher.py` | Advanced ID search and filtering utilities |

---

## API Reference

### 1. Get_DG_ids (`Get_id.py`)

**Purpose**: Retrieves training/validation and test sample IDs for Domain Generalization (DG) tasks.

**Function Signature**:
```python
def Get_DG_ids(metadata_accessor, args_task):
    """
    Returns (train_val_ids, test_ids) for DG tasks.

    Supports two modes:
    1. Dynamic splitting: Uses target_domain_num to split domains automatically
    2. Predefined splitting: Uses source_domain_id and target_domain_id
    """
```

**Parameters**:
- `metadata_accessor`: Object with `df` attribute containing metadata DataFrame
- `args_task`: Task configuration with:
  - `target_system_id`: List of system IDs to include
  - `target_domain_num` (optional): Number of domains for test split
  - `source_domain_id` (fallback): Predefined training domains
  - `target_domain_id` (fallback): Predefined test domains

**Returns**:
- `train_val_ids` (list): Sample IDs for training/validation
- `test_ids` (list): Sample IDs for testing

**Dynamic Splitting Mode** (when `target_domain_num` is set):
```python
# Automatically allocates last N domains as test domains
args_task.target_domain_num = 2  # Last 2 domains become test domains

# Example with 5 domains [1,2,3,4,5]:
# - Train domains: [1, 2, 3]
# - Test domains: [4, 5]
```

**Predefined Splitting Mode** (fallback):
```python
args_task.source_domain_id = [1, 5, 6]  # Training domains
args_task.target_domain_id = [19]        # Test domain
```

---

### 2. Get_CDDG_ids (`Get_id.py`)

**Purpose**: Retrieves sample IDs for Cross-Dataset Domain Generalization (CDDG) tasks.

**Function Signature**:
```python
def Get_CDDG_ids(metadata_accessor, args_task):
    """
    Returns (train_val_ids, test_ids) for CDDG tasks.

    Similar to DG but optimized for cross-dataset scenarios where
    source and target may be completely different datasets.
    """
```

---

### 3. remove_invalid_labels (`Get_id.py`)

**Purpose**: Utility function to filter out samples with invalid labels (Label = -1).

**Function Signature**:
```python
def remove_invalid_labels(df, label_column='Label'):
    """
    Removes rows where label_column == -1 from DataFrame.

    Args:
        df (pd.DataFrame): Input metadata DataFrame
        label_column (str): Column name to check, default 'Label'

    Returns:
        pd.DataFrame: Filtered DataFrame with reset index
    """
```

---

## Id_searcher.py Functions

### search_ids_for_task

**Purpose**: Advanced ID search with flexible filtering conditions.

**Typical Usage**:
```python
from src.data_factory.ID.Id_searcher import search_ids_for_task

# Search by multiple criteria
ids = search_ids_for_task(
    metadata_accessor=metadata,
    dataset_ids=[1, 2, 3],
    domain_ids=[1, 2],
    labels=[0, 1, 2],
    min_samples_per_class=10
)
```

---

## Configuration Examples

### Domain Generalization (Dynamic Split)
```yaml
task:
  name: "DG"
  type: "DG"

  # System selection
  target_system_id: [1]  # CWRU dataset

  # Dynamic domain splitting
  target_domain_num: 2   # Use last 2 domains as test

  # Result: Automatically splits available domains
```

### Domain Generalization (Predefined)
```yaml
task:
  name: "DG"
  type: "DG"

  target_system_id: [1]

  # Predefined domain split
  source_domain_id: [1, 2]  # Domains for training
  target_domain_id: [3]      # Domain for testing
```

### Cross-Dataset DG
```yaml
task:
  name: "CDDG"
  type: "CDDG"

  # Multiple source systems
  target_system_id: [1, 5, 6]  # CWRU, MFPT, THU

  source_domain_id: [1, 5, 6]
  target_domain_id: [19]        # Different target system
```

---

## Metadata Requirements

The metadata DataFrame must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `Id` | int/str | Primary key, unique sample identifier |
| `Dataset_id` | int | Source dataset/system identifier |
| `Domain_id` | int | Domain/condition identifier |
| `Label` | int | Class label (-1 = invalid/unused) |

**Example Metadata Structure**:
```
Id  | Dataset_id | Domain_id | Label
----|------------|-----------|-------
1   | 1          | 1         | 0
2   | 1          | 1         | 0
3   | 1          | 2         | 1
4   | 1          | 2         | -1  (excluded by remove_invalid_labels)
```

---

## Usage in Data Factory

```python
# In data_factory.py or id_data_factory.py
from src.data_factory.ID.Get_id import Get_DG_ids, Get_CDDG_ids

if args_task.type == "DG":
    train_val_ids, test_ids = Get_DG_ids(metadata_accessor, args_task)
elif args_task.type == "CDDG":
    train_val_ids, test_ids = Get_CDDG_ids(metadata_accessor, args_task)

# Create datasets with filtered IDs
train_dataset = Dataset(train_val_ids, ...)
test_dataset = Dataset(test_ids, ...)
```

---

## Output Logging

The functions print diagnostic information:

```
DG划分 - 使用 target_domain_num=2 进行动态划分
  - 训练域: [1, 2, 3]
  - 测试域: [4, 5]
训练/验证样本数: 1500
测试样本数: 500
```

---

## Related Documentation

- [@../README.md] - Data Factory Module Overview
- [@../dataset_task/README.md] - Dataset Task Implementations
- [@../../task_factory/task/DG/README.md] - Domain Generalization Tasks

---

## Best Practices

1. **Always validate metadata**: Ensure metadata contains required columns before querying
2. **Handle empty results**: Check if returned ID lists are empty
3. **Use dynamic splitting for new datasets**: When you don't know optimal train/test split
4. **Use predefined splits for reproducibility**: When you need consistent experimental results
