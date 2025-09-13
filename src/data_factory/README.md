# Data Factory Module (Continue updating)

## 🎯 Purpose

The **Data Factory** module serves as the centralized and unified entry point for all data-related operations within the Vibench framework. Its core responsibilities include:

* **Dataset Creation**: Assembling datasets from various raw sources.
* **Metadata Management**: Handling sample-level metadata for precise data selection and filtering.
* **Data Loading**: Generating `DataLoader` instances for training, validation, and testing pipelines.

This module is designed to be highly extensible, allowing for the easy integration of new datasets and data processing tasks.

---

## 📂 Module Structure

The module is organized into several key components, each with a specific role:

| File / Directory | Description |
| :--- | :--- |
| `data_factory.py` | The main entry point of the module. It registers dataset readers and tasks and exposes the `data_factory(args)` function to build `DataLoaders`. |
| `ID_data_factory.py` | A specialized data factory for the `ID_dataset`, which provides raw ID data on-demand. The training loop is only for ID loop and then the task module will process the data with specific Id accordingly. |
| `datainfo.py` | Given a root directory, this script explores the directory structure to determine dataset names and file paths and generates a raw metadata with basic information. |
| `reader/` | Contains dataset-specific reader scripts (e.g., `RM_001_CWRU.py`, `RM_002_XJTU.py`). Each reader parses raw data files and converts them into standardized tensor formats. |
| `dataset_task/` | Includes task-oriented dataset wrappers that implement different data strategies, such as custom sampling, data augmentation, or specific data arrangements for tasks like Domain Generalization (`DG/`), Few-Shot Learning (`FS/`), or Pre-training (`Pretrain/`). |
| `samplers/` | Provides custom mini-batch samplers (e.g., `FS_sampler.py`) that can be plugged into the `DataLoader` to control how batches are composed, which is especially useful for advanced training schemes. |
| `ID/` | A set of utilities for querying and filtering sample IDs based on metadata criteria, enabling precise control over the data used in experiments. |
| `utils/` | Contains utility functions for data processing, such as windowing, normalization, and other common operations. |
---

## 🌊 Workflow

The data loading process follows a sequential workflow, orchestrated by the `data_factory`.



1.  **Configuration**: The process begins in the main configuration file (`.yaml`) from the main pipeline, where you define the dataset paths and specify the desired task (e.g., `DG`, `FS`).

2.  **Metadata & Reader Selection**:
    * Based on the dataset IDs specified in the configuration, the factory selects the appropriate dataset-specific `reader` (e.g., `RM_001_CWRU.py` for CWRU data).

3.  **Data Wrapping & Labeling**:
    * The selected `reader` processes the raw data into tensors.
    * A `dataset_task` then wraps the reader's output, attaching essential information like labels, domain IDs, and other task-specific metadata.

4.  **Batch Sampling**:
    * If a custom `sampler` is specified in the configuration, it is attached to the `DataLoader`.
    * The sampler controls how individual samples are grouped into mini-batches, which is critical for tasks like few-shot or contrastive learning.

5.  **Output**: The `data_factory` returns the final objects required for the training pipeline.

## metadata understanding



---

### **English Version (Optimized)**

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


---

## 🎁 Returned Objects

Upon successful execution, the `data_factory(args)` function have the following methods:

* **get_dataset(mode)**: Returns the dataset for the specified mode (`train`, `val`, or `test`).
* **get_dataloader(mode)**: Returns the dataloader for the specified mode (`train`, `val`, or `test`).
* **get_metadata()**: Returns the metadata dictionary.
* **get_data()**: Returns the data dictionary.


* **DataLoaders**: A set of `DataLoader` objects for the training, validation, and test sets.
* **Metadata Dictionary**: A comprehensive dictionary containing the loaded metadata, providing easy access to information about the datasets used in the experiment.
