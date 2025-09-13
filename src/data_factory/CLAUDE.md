# Data Factory - CLAUDE.md

This module provides guidance for working with the data factory system in PHM-Vibench, which handles dataset loading, processing, and integration.

## Architecture Overview

The data factory system uses a modular approach with these key components:
- **data_factory.py**: Main data processing factory
- **id_data_factory.py**: Memory-efficient ID-based data processing
- **H5DataDict.py**: Efficient H5 file data access
- **reader/**: Dataset-specific readers for 30+ industrial datasets
- **dataset_task/**: Task-specific dataset implementations
- **samplers/**: Custom sampling strategies

## Key Classes and Components

### Data Factories

#### `data_factory` (Standard Factory)
```python
# Main data factory for traditional processing
factory = data_factory(args_data, args_task)
train_loader, val_loader, test_loader = factory.get_dataloaders()
```

#### `id_data_factory` (Memory-Optimized)
```python
# Use when data_factory.factory_name = 'id' in config
# Defers data loading to task processing stage
factory = id_data_factory(args_data, args_task)
```

### Core Data Components

#### `H5DataDict`
Efficient data access for H5 files:
- Lazy loading for memory efficiency
- Metadata integration
- Multi-dataset support

#### Dataset Readers (reader/)
Each dataset has a dedicated reader (e.g., `RM_001_CWRU.py`):
- Inherits from `BaseReader`
- Handles dataset-specific preprocessing
- Standardizes data format and metadata

#### `IdIncludedDataset`
Wrapper class that adds file IDs to dataset samples for task processing:
- Wraps individual dataset instances in a unified interface
- **Batch Format**: When iterating through DataLoader, returns dict with keys:
  - `'x'`: Input tensor (B, L, C) - signal data
  - `'y'`: Label tensor - classification/regression target
  - `'file_id'`: Original dataset file ID for metadata lookup

**Important**: Task implementations must use dict access (not tuple unpacking):
```python
def training_step(self, batch, batch_idx):
    x = batch['x']           # Input tensor
    y = batch['y']           # Labels  
    file_id = batch['file_id'][0].item()  # File ID for metadata
```

## Common Usage Patterns

### Loading Industrial Datasets
```python
# In your configuration YAML
data:
  data_dir: "/path/to/data"
  metadata_file: "metadata_6_11.xlsx"
  batch_size: 32
  window_size: 1024
  normalization: true
```

### Working with Multiple Datasets
```python
# Cross-dataset domain generalization
task:
  type: "CDDG"
  source_domain_id: [1, 5, 6]  # CWRU, MFPT, THU
  target_domain_id: 19         # Different target dataset
```

### Custom Sampling
```python
# Few-shot learning with episode sampling
sampler = Get_sampler(
    task_type="FS",
    num_support=5,
    num_query=15,
    num_episodes=1000
)
```

## Vbench Data Components Guide

This guide provides a clear overview of the three core data files used in Vbench and how to access them.

### Core Components

The dataset is organized into three main files, linked by a common `Id`.

1.  **`metadata.xlsx` (Excel File)**
    * **Purpose**: The central index of the dataset. It contains all descriptive information, labels, and parameters for each data sample.
    * **Primary Key**: The `Id` column uniquely identifies each sample and links the three files.

2.  **`data.h5` (HDF5 File)**
    * **Purpose**: Stores the raw time-series signal data.
    * **Access**: Data is retrieved using the `Id` from the metadata file as the key.
    * **Shape**: The data for each `Id` is a 2D array of shape `(L, C)`, where `L` (Sample_lenth) and `C` (Channel) are specified in `metadata.xlsx`.

3.  **`corpus.xlsx` (Excel File)**
    * **Purpose**: Contains supplementary text descriptions and natural language annotations for each sample.
    * **Access**: Text is retrieved using the corresponding `Id`.

### `metadata.xlsx`: Column Descriptions

The first row of the metadata file contains the following headers:

* `Id`: **(Primary Key)** Unique identifier for the sample.
* `Dataset_id`: Source dataset identifier.
* `Name`: Human-readable name.
* `Description`: Brief description of the sample.
* `TYPE`: Type of data (e.g., vibration, acoustic).
* `File`: Source file name.
* `Visiable`: Visibility or usage flag.
* `Label`: The primary fault class or label.
* `Label_Description`: Textual description of the `Label`.
* `Fault_level`: Severity or stage of the fault.
* `RUL_label`: Remaining Useful Life value.
* `RUL_label_description`: Description for the RUL value.
* `Domain_id`: Identifier for the operational condition.
* `Domain_description`: Textual description of the domain.
* `Sample_rate`: Signal sampling rate (Hz).
* `Sample_lenth (L)`: Number of data points in the sample.
* `Channel (C)`: Number of channels in the sample.
* `Fault_Diagnosis`: Flag for fault diagnosis task suitability.
* `Anomaly_Detection`: Flag for anomaly detection task suitability.
* `Remaining_Life`: Flag for RUL prediction task suitability.

## Supported Industrial Datasets

### Major Datasets
- **RM_001_CWRU**: Case Western Reserve University bearing data
- **RM_002_XJTU**: Xi'an Jiaotong University bearing data
- **RM_003_FEMTO**: FEMTO bearing degradation data
- **RM_006_THU**: Tsinghua University bearing data
- **RM_026_HUST23**: HUST bearing dataset
- And 25+ more industrial datasets

### Dataset Reader Pattern
```python
class RM_XXX_DatasetName:
    def __init__(self, data_dir):
        # Initialize reader with data directory
        
    def load_data(self):
        # Load and return standardized data format
        return {
            'data': np.array,      # Signal data
            'labels': np.array,    # Class labels  
            'metadata': dict       # Additional info
        }
```

## Data Processing Pipeline

### Standard Pipeline
1. **Metadata Loading**: Read Excel metadata files
2. **Raw Data Access**: Use dataset readers to load raw signals
3. **Preprocessing**: Window segmentation, normalization
4. **Dataset Creation**: Create PyTorch datasets
5. **DataLoader Creation**: Batch processing with custom samplers

### ID-Based Pipeline (Memory Efficient)
1. **ID Collection**: Extract sample IDs and metadata only
2. **Lazy Loading**: Defer actual data loading to training time
3. **On-Demand Processing**: Apply windowing/normalization during training
4. **Memory Optimization**: Reduce memory footprint for large datasets

## Configuration Parameters

### Essential Data Config
```yaml
data:
  factory_name: "default"        # or "id" for memory-efficient
  data_dir: "/path/to/data"      # Root data directory
  metadata_file: "metadata.xlsx" # Dataset metadata
  batch_size: 32                 # Batch size for training
  num_workers: 4                 # DataLoader workers
  
  # Preprocessing
  normalization: true            # Enable normalization
  window_size: 1024              # Signal window length
  stride: 512                    # Window stride
  truncate_length: 8192          # Max signal length
```

### Advanced Options
```yaml
data:
  # Memory management
  pin_memory: true               # GPU memory optimization
  persistent_workers: true       # Keep workers alive
  
  # Data splitting
  train_ratio: 0.7               # Training data proportion
  val_ratio: 0.2                 # Validation data proportion
  
  # Task-specific
  target_domain_id: 13           # For domain generalization
  num_systems: 5                 # For multi-system tasks
```

## Adding New Datasets

### 1. Create Dataset Reader
```python
# In src/data_factory/reader/RM_XXX_NewDataset.py
class RM_XXX_NewDataset:
    def __init__(self, data_dir, **kwargs):
        self.data_dir = data_dir
        
    def load_data(self):
        # Implement dataset loading logic
        return standardized_data_dict
```

### 2. Register Dataset
Update the reader registry in `__init__.py`:
```python
from .reader.RM_XXX_NewDataset import RM_XXX_NewDataset
```

### 3. Update Metadata
Add dataset information to metadata Excel file with:
- Dataset ID and name
- File paths and structure
- Label mappings
- Technical specifications

### 4. Create Test Configuration
```yaml
# In configs/demo/Single_DG/NewDataset.yaml
data:
  target_system_id: [XXX]  # Your dataset ID
  # ... other configurations
```

## Best Practices

### Memory Management
- Use `id_data_factory` for large datasets (>10GB)
- Set appropriate `num_workers` based on CPU cores
- Enable `pin_memory` for GPU training

### Data Preprocessing
- Keep preprocessing consistent across datasets
- Use metadata files to standardize label mappings
- Implement proper normalization for signal processing

### Multi-Dataset Experiments
- Use consistent sampling strategies
- Balance dataset sizes in multi-domain tasks
- Verify label compatibility across datasets

## Troubleshooting

### Common Issues
1. **Memory Errors**: Switch to `id_data_factory` or reduce batch size
2. **Missing Data**: Check data_dir path and metadata file
3. **Inconsistent Labels**: Verify metadata label mappings
4. **Slow Loading**: Increase num_workers or use SSD storage

### Debug Commands
```python
# Test data loading
python -m src.data_factory.test
```

## Integration with Other Modules

- **Task Factory**: Provides datasets to task implementations
- **Model Factory**: Supplies data shape information for model initialization
- **Trainer Factory**: Integrates with PyTorch Lightning data modules
- **Utils**: Uses configuration utilities and registry patterns