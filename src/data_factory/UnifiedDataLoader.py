"""
UnifiedDataLoader for multi-dataset loading and unified metric learning.

This loader supports unified pretraining across multiple datasets and dataset-specific
fine-tuning with balanced sampling and HSE prompt injection capabilities.

Features:
- Unified data loading across all 5 datasets (CWRU, XJTU, THU, Ottawa, JNU)
- Balanced sampling to ensure equal representation across datasets
- Support for both unified pretraining and dataset-specific fine-tuning modes
- Dataset mixing strategies for robust learning
- HSE prompt injection capability for prompt-guided learning
- Zero-shot evaluation data preparation
- Comprehensive self-test with sample data validation

Author: PHM-Vibench Team
Date: 2025-09-12
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import random
from collections import defaultdict
import logging
from torch.utils.data import Dataset, DataLoader, Sampler
import copy

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.data_factory.data_factory import data_factory
    from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
    from src.data_factory.samplers.Sampler import Same_system_Sampler
    from src.data_factory.data_utils import MetadataAccessor
    from src.utils.registry import Registry
except ImportError as e:
    print(f"Warning: Could not import PHM-Vibench components: {e}")


UNIFIED_LOADER_REGISTRY = Registry()

def register_unified_loader(name: str):
    """Decorator to register a unified loader implementation."""
    return UNIFIED_LOADER_REGISTRY.register(name)


class UnifiedDataset(Dataset):
    """
    Unified dataset that combines multiple datasets for unified metric learning.
    
    Supports balanced sampling across datasets and HSE prompt injection.
    """
    
    def __init__(self,
                 datasets: Dict[str, Dataset],
                 dataset_weights: Optional[Dict[str, float]] = None,
                 mode: str = 'unified',
                 prompt_injection: bool = False,
                 system_prompts: Optional[Dict[str, str]] = None,
                 sample_prompts: Optional[Dict[str, str]] = None):
        """
        Initialize unified dataset.
        
        Args:
            datasets: Dictionary mapping dataset names to Dataset objects
            dataset_weights: Optional weights for balanced sampling
            mode: 'unified' for multi-dataset or 'single' for dataset-specific
            prompt_injection: Whether to inject HSE prompts
            system_prompts: System-level prompts for each dataset
            sample_prompts: Sample-level prompts for each dataset
        """
        self.datasets = datasets
        self.dataset_names = list(datasets.keys())
        self.mode = mode
        self.prompt_injection = prompt_injection
        self.system_prompts = system_prompts or {}
        self.sample_prompts = sample_prompts or {}
        
        # Compute dataset statistics
        self.dataset_sizes = {name: len(dataset) for name, dataset in datasets.items()}
        self.total_size = sum(self.dataset_sizes.values())
        
        # Setup balanced sampling weights
        if dataset_weights is None:
            # Equal weight for all datasets
            self.dataset_weights = {name: 1.0 / len(self.datasets) for name in self.dataset_names}
        else:
            self.dataset_weights = dataset_weights
        
        # Create unified index mapping
        self.create_unified_index()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"UnifiedDataset initialized with {len(self.datasets)} datasets")
        self.logger.info(f"Dataset sizes: {self.dataset_sizes}")
        self.logger.info(f"Total samples: {self.total_size}")
    
    def create_unified_index(self):
        """Create unified indexing across all datasets."""
        self.unified_index = []
        
        for dataset_name, dataset in self.datasets.items():
            for idx in range(len(dataset)):
                self.unified_index.append({
                    'dataset_name': dataset_name,
                    'dataset_idx': idx,
                    'global_idx': len(self.unified_index)
                })
    
    def __len__(self) -> int:
        """Return total number of samples across all datasets."""
        if self.mode == 'unified':
            return self.total_size
        else:
            # For single dataset mode, return size of the first dataset
            return self.dataset_sizes[self.dataset_names[0]]
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get sample by unified index.
        
        Args:
            idx: Unified index across all datasets
            
        Returns:
            Dictionary containing sample data with dataset information and prompts
        """
        if self.mode == 'unified':
            # Unified mode: use balanced sampling
            sample_info = self.unified_index[idx]
            dataset_name = sample_info['dataset_name']
            dataset_idx = sample_info['dataset_idx']
        else:
            # Single dataset mode: use first dataset only
            dataset_name = self.dataset_names[0]
            dataset_idx = idx
        
        # Get sample from appropriate dataset
        dataset = self.datasets[dataset_name]
        sample = dataset[dataset_idx]
        
        # Ensure sample is a dictionary
        if not isinstance(sample, dict):
            if isinstance(sample, (list, tuple)) and len(sample) >= 2:
                sample = {'data': sample[0], 'label': sample[1]}
            else:
                sample = {'data': sample, 'label': 0}
        
        # Add dataset information
        sample['dataset_name'] = dataset_name
        sample['dataset_idx'] = dataset_idx
        sample['global_idx'] = idx
        
        # Add HSE prompts if enabled
        if self.prompt_injection:
            sample['system_prompt'] = self.system_prompts.get(dataset_name, "")
            sample['sample_prompt'] = self.sample_prompts.get(dataset_name, "")
        
        return sample
    
    def get_dataset_distribution(self) -> Dict[str, float]:
        """Get the distribution of samples across datasets."""
        return {name: size / self.total_size for name, size in self.dataset_sizes.items()}
    
    def get_balanced_sampler(self, batch_size: int, shuffle: bool = True) -> 'UnifiedBalancedSampler':
        """
        Get a balanced sampler that ensures equal representation across datasets.
        
        Args:
            batch_size: Batch size for sampling
            shuffle: Whether to shuffle samples
            
        Returns:
            UnifiedBalancedSampler instance
        """
        return UnifiedBalancedSampler(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            dataset_weights=self.dataset_weights
        )


class UnifiedBalancedSampler(Sampler):
    """
    Balanced sampler that ensures equal representation across datasets.
    
    Implements stratified sampling to maintain dataset balance in each batch.
    """
    
    def __init__(self,
                 dataset: UnifiedDataset,
                 batch_size: int,
                 shuffle: bool = True,
                 dataset_weights: Optional[Dict[str, float]] = None,
                 drop_last: bool = False):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: UnifiedDataset to sample from
            batch_size: Size of each batch
            shuffle: Whether to shuffle samples
            dataset_weights: Weights for each dataset (for balanced sampling)
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dataset_weights = dataset_weights or dataset.dataset_weights
        
        # Group indices by dataset
        self.indices_by_dataset = defaultdict(list)
        for item in dataset.unified_index:
            dataset_name = item['dataset_name']
            global_idx = item['global_idx']
            self.indices_by_dataset[dataset_name].append(global_idx)
        
        # Calculate samples per batch for each dataset
        self.samples_per_dataset_per_batch = {}
        total_weight = sum(self.dataset_weights.values())
        
        for dataset_name in dataset.dataset_names:
            weight = self.dataset_weights[dataset_name]
            samples_count = int((weight / total_weight) * batch_size)
            self.samples_per_dataset_per_batch[dataset_name] = max(1, samples_count)
        
        # Adjust to ensure total equals batch_size
        current_total = sum(self.samples_per_dataset_per_batch.values())
        if current_total != batch_size:
            # Distribute remaining samples to largest datasets
            diff = batch_size - current_total
            largest_datasets = sorted(
                dataset.dataset_names,
                key=lambda x: dataset.dataset_sizes[x],
                reverse=True
            )
            
            for i, dataset_name in enumerate(largest_datasets):
                if diff == 0:
                    break
                if diff > 0:
                    self.samples_per_dataset_per_batch[dataset_name] += 1
                    diff -= 1
                else:
                    if self.samples_per_dataset_per_batch[dataset_name] > 1:
                        self.samples_per_dataset_per_batch[dataset_name] -= 1
                        diff += 1
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Balanced sampler initialized:")
        self.logger.info(f"Samples per dataset per batch: {self.samples_per_dataset_per_batch}")
    
    def __iter__(self):
        """Generate balanced batches."""
        # Shuffle indices for each dataset
        dataset_iterators = {}
        for dataset_name in self.dataset.dataset_names:
            indices = list(self.indices_by_dataset[dataset_name])
            if self.shuffle:
                random.shuffle(indices)
            dataset_iterators[dataset_name] = iter(indices)
        
        while True:
            batch = []
            
            # Try to sample from each dataset
            for dataset_name in self.dataset.dataset_names:
                samples_needed = self.samples_per_dataset_per_batch[dataset_name]
                iterator = dataset_iterators[dataset_name]
                
                for _ in range(samples_needed):
                    try:
                        idx = next(iterator)
                        batch.append(idx)
                    except StopIteration:
                        # Re-shuffle and restart iterator if needed
                        if self.shuffle:
                            indices = list(self.indices_by_dataset[dataset_name])
                            random.shuffle(indices)
                            dataset_iterators[dataset_name] = iter(indices)
                            try:
                                idx = next(dataset_iterators[dataset_name])
                                batch.append(idx)
                            except StopIteration:
                                # Dataset is empty, skip
                                pass
                        else:
                            # No more samples from this dataset
                            break
            
            if len(batch) == 0:
                # No more samples available
                break
            
            if len(batch) < self.batch_size and self.drop_last:
                # Drop incomplete batch
                break
            
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches per epoch."""
        min_dataset_size = min(self.dataset.dataset_sizes.values())
        batches_per_smallest_dataset = min_dataset_size // max(self.samples_per_dataset_per_batch.values())
        
        if self.drop_last:
            return batches_per_smallest_dataset
        else:
            return batches_per_smallest_dataset + 1


@register_unified_loader("default")
class UnifiedDataLoader:
    """
    Unified data loader for multi-dataset training and evaluation.
    
    Supports unified pretraining across multiple datasets and dataset-specific
    fine-tuning with balanced sampling and HSE prompt injection.
    """
    
    # Standard dataset mapping for unified metric learning
    STANDARD_DATASETS = {
        'CWRU': {'id': 1, 'name': 'CWRU'},
        'XJTU': {'id': 6, 'name': 'XJTU'},
        'THU': {'id': 5, 'name': 'THU'},
        'Ottawa': {'id': 19, 'name': 'Ottawa'},
        'JNU': {'id': 23, 'name': 'JNU'}
    }
    
    def __init__(self,
                 data_config: Dict[str, Any],
                 task_config: Dict[str, Any],
                 mode: str = 'unified',
                 enable_prompt_injection: bool = False,
                 target_datasets: Optional[List[str]] = None):
        """
        Initialize unified data loader.
        
        Args:
            data_config: Data configuration dictionary
            task_config: Task configuration dictionary
            mode: 'unified' for multi-dataset or 'single' for dataset-specific
            enable_prompt_injection: Whether to enable HSE prompt injection
            target_datasets: List of dataset names to load (default: all standard datasets)
        """
        self.data_config = data_config.copy()
        self.task_config = task_config.copy()
        self.mode = mode
        self.enable_prompt_injection = enable_prompt_injection
        self.target_datasets = target_datasets or list(self.STANDARD_DATASETS.keys())
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"UnifiedDataLoader initialized in '{mode}' mode")
        self.logger.info(f"Target datasets: {self.target_datasets}")
        
        # Storage for datasets and loaders
        self.datasets = {}
        self.unified_datasets = {}
        self.data_loaders = {}
        
        # HSE prompt storage
        self.system_prompts = {}
        self.sample_prompts = {}
        
        # Load datasets
        self.load_datasets()
        
        # Setup HSE prompts if enabled
        if self.enable_prompt_injection:
            self.setup_hse_prompts()
        
        # Create unified datasets
        self.create_unified_datasets()
    
    def load_datasets(self):
        """Load all target datasets using individual data factories."""
        self.logger.info("Loading individual datasets...")
        
        for dataset_name in self.target_datasets:
            if dataset_name not in self.STANDARD_DATASETS:
                self.logger.warning(f"Unknown dataset: {dataset_name}, skipping")
                continue
            
            try:
                # Create dataset-specific configuration
                dataset_config = self.create_dataset_config(dataset_name)
                
                # Create data factory for this dataset
                dataset_factory = data_factory(dataset_config, self.task_config)
                
                # Store datasets
                self.datasets[dataset_name] = {
                    'train': dataset_factory.train_dataset,
                    'val': dataset_factory.val_dataset,
                    'test': dataset_factory.test_dataset,
                    'factory': dataset_factory
                }
                
                self.logger.info(f"Successfully loaded dataset: {dataset_name}")
                self.logger.info(f"  Train: {len(dataset_factory.train_dataset)} samples")
                self.logger.info(f"  Val: {len(dataset_factory.val_dataset)} samples")
                self.logger.info(f"  Test: {len(dataset_factory.test_dataset)} samples")
                
            except Exception as e:
                self.logger.error(f"Failed to load dataset {dataset_name}: {e}")
                # Continue with other datasets
                continue
    
    def create_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """
        Create dataset-specific configuration.
        
        Args:
            dataset_name: Name of the dataset to configure
            
        Returns:
            Dataset-specific configuration dictionary
        """
        config = self.data_config.copy()
        
        # Set target dataset ID
        dataset_info = self.STANDARD_DATASETS[dataset_name]
        config['target_dataset_id'] = [dataset_info['id']]
        config['target_system_id'] = [dataset_info['id']]
        
        # Convert to namespace if needed
        if isinstance(config, dict):
            from types import SimpleNamespace
            config = SimpleNamespace(**config)
        
        return config
    
    def setup_hse_prompts(self):
        """Setup HSE prompts for each dataset."""
        self.logger.info("Setting up HSE prompts...")
        
        # Default system prompts for each dataset
        default_system_prompts = {
            'CWRU': "Bearing vibration signals from laboratory conditions with artificial defects",
            'XJTU': "Bearing degradation signals from accelerated life testing",
            'THU': "Multi-condition bearing signals with varying loads and speeds",
            'Ottawa': "Real-world bearing signals from industrial machinery",
            'JNU': "High-frequency bearing signals with natural fault progression"
        }
        
        # Default sample prompts (can be customized per sample)
        default_sample_prompts = {
            'CWRU': "Analyze fault characteristics in controlled environment",
            'XJTU': "Monitor degradation progression over time",
            'THU': "Adapt to varying operational conditions",
            'Ottawa': "Process real-world industrial signals",
            'JNU': "Detect early fault indicators"
        }
        
        self.system_prompts = default_system_prompts
        self.sample_prompts = default_sample_prompts
        
        self.logger.info("HSE prompts configured for all datasets")
    
    def create_unified_datasets(self):
        """Create unified datasets for different splits."""
        self.logger.info("Creating unified datasets...")
        
        splits = ['train', 'val', 'test']
        
        for split in splits:
            # Collect datasets for this split
            split_datasets = {}
            
            for dataset_name in self.target_datasets:
                if dataset_name in self.datasets:
                    split_datasets[dataset_name] = self.datasets[dataset_name][split]
            
            if not split_datasets:
                self.logger.warning(f"No datasets available for split: {split}")
                continue
            
            # Create unified dataset
            unified_dataset = UnifiedDataset(
                datasets=split_datasets,
                mode=self.mode,
                prompt_injection=self.enable_prompt_injection,
                system_prompts=self.system_prompts,
                sample_prompts=self.sample_prompts
            )
            
            self.unified_datasets[split] = unified_dataset
            
            self.logger.info(f"Created unified {split} dataset with {len(unified_dataset)} samples")
    
    def get_dataloaders(self,
                       batch_size: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       pin_memory: bool = True,
                       use_balanced_sampling: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get unified data loaders for train, validation, and test.
        
        Args:
            batch_size: Batch size (uses config default if None)
            num_workers: Number of workers (uses config default if None)
            pin_memory: Whether to use pinned memory
            use_balanced_sampling: Whether to use balanced sampling across datasets
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Use config defaults if not specified
        batch_size = batch_size or getattr(self.data_config, 'batch_size', 32)
        num_workers = num_workers or getattr(self.data_config, 'num_workers', 4)
        
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            if split not in self.unified_datasets:
                loaders[split] = None
                continue
            
            dataset = self.unified_datasets[split]
            
            if use_balanced_sampling and self.mode == 'unified':
                # Use balanced sampler for unified mode
                sampler = dataset.get_balanced_sampler(
                    batch_size=batch_size,
                    shuffle=(split == 'train')
                )
                
                loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    collate_fn=self.unified_collate_fn
                )
            else:
                # Use standard random sampling
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == 'train'),
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    drop_last=(split == 'train'),
                    collate_fn=self.unified_collate_fn
                )
            
            loaders[split] = loader
            self.logger.info(f"Created {split} dataloader with batch_size={batch_size}")
        
        self.data_loaders = loaders
        return loaders['train'], loaders['val'], loaders['test']
    
    def unified_collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Custom collate function for unified datasets.
        
        Args:
            batch: List of sample dictionaries
            
        Returns:
            Batched dictionary with proper tensor formatting
        """
        collated = {}
        
        # Separate data fields
        data_tensors = []
        label_tensors = []
        dataset_names = []
        dataset_indices = []
        global_indices = []
        
        # HSE prompt fields
        system_prompts = []
        sample_prompts = []
        
        for sample in batch:
            # Extract main data
            data = sample['data']
            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data).float()
            elif not isinstance(data, torch.Tensor):
                data = torch.tensor(data, dtype=torch.float32)
            
            data_tensors.append(data)
            
            # Extract labels
            label = sample['label']
            if isinstance(label, np.ndarray):
                label = torch.from_numpy(label).long()
            elif not isinstance(label, torch.Tensor):
                label = torch.tensor(label, dtype=torch.long)
            
            label_tensors.append(label)
            
            # Extract metadata
            dataset_names.append(sample['dataset_name'])
            dataset_indices.append(sample['dataset_idx'])
            global_indices.append(sample['global_idx'])
            
            # Extract prompts if available
            if 'system_prompt' in sample:
                system_prompts.append(sample['system_prompt'])
            if 'sample_prompt' in sample:
                sample_prompts.append(sample['sample_prompt'])
        
        # Stack tensors
        collated['data'] = torch.stack(data_tensors)
        collated['label'] = torch.stack(label_tensors)
        
        # Store metadata
        collated['dataset_names'] = dataset_names
        collated['dataset_indices'] = dataset_indices
        collated['global_indices'] = global_indices
        
        # Store prompts if available
        if system_prompts:
            collated['system_prompts'] = system_prompts
        if sample_prompts:
            collated['sample_prompts'] = sample_prompts
        
        # Add batch statistics
        collated['batch_size'] = len(batch)
        collated['dataset_distribution'] = self._compute_batch_distribution(dataset_names)
        
        return collated
    
    def _compute_batch_distribution(self, dataset_names: List[str]) -> Dict[str, float]:
        """Compute distribution of datasets in current batch."""
        from collections import Counter
        counts = Counter(dataset_names)
        total = len(dataset_names)
        return {name: count / total for name, count in counts.items()}
    
    def get_zero_shot_evaluation_data(self,
                                    source_datasets: List[str],
                                    target_dataset: str,
                                    batch_size: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for zero-shot evaluation.
        
        Args:
            source_datasets: List of source dataset names for training
            target_dataset: Target dataset name for evaluation
            batch_size: Batch size for evaluation
            
        Returns:
            Tuple of (source_loader, target_loader)
        """
        batch_size = batch_size or getattr(self.data_config, 'batch_size', 32)
        
        # Create source dataset (for reference/few-shot examples if needed)
        source_split_datasets = {}
        for dataset_name in source_datasets:
            if dataset_name in self.datasets:
                source_split_datasets[dataset_name] = self.datasets[dataset_name]['test']
        
        source_unified = UnifiedDataset(
            datasets=source_split_datasets,
            mode='unified',
            prompt_injection=self.enable_prompt_injection,
            system_prompts=self.system_prompts,
            sample_prompts=self.sample_prompts
        )
        
        # Create target dataset
        if target_dataset in self.datasets:
            target_split_datasets = {target_dataset: self.datasets[target_dataset]['test']}
            target_unified = UnifiedDataset(
                datasets=target_split_datasets,
                mode='single',
                prompt_injection=self.enable_prompt_injection,
                system_prompts=self.system_prompts,
                sample_prompts=self.sample_prompts
            )
        else:
            raise ValueError(f"Target dataset '{target_dataset}' not available")
        
        # Create data loaders
        source_loader = DataLoader(
            source_unified,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(self.data_config, 'num_workers', 4),
            collate_fn=self.unified_collate_fn
        )
        
        target_loader = DataLoader(
            target_unified,
            batch_size=batch_size,
            shuffle=False,
            num_workers=getattr(self.data_config, 'num_workers', 4),
            collate_fn=self.unified_collate_fn
        )
        
        self.logger.info(f"Created zero-shot evaluation loaders:")
        self.logger.info(f"  Source datasets: {source_datasets} ({len(source_unified)} samples)")
        self.logger.info(f"  Target dataset: {target_dataset} ({len(target_unified)} samples)")
        
        return source_loader, target_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive information about loaded datasets."""
        info = {
            'mode': self.mode,
            'enable_prompt_injection': self.enable_prompt_injection,
            'target_datasets': self.target_datasets,
            'loaded_datasets': list(self.datasets.keys()),
            'dataset_sizes': {},
            'unified_sizes': {},
            'total_samples': 0
        }
        
        # Individual dataset info
        for dataset_name, dataset_info in self.datasets.items():
            info['dataset_sizes'][dataset_name] = {
                'train': len(dataset_info['train']),
                'val': len(dataset_info['val']),
                'test': len(dataset_info['test'])
            }
        
        # Unified dataset info
        for split, unified_dataset in self.unified_datasets.items():
            info['unified_sizes'][split] = len(unified_dataset)
            info['total_samples'] += len(unified_dataset)
        
        return info
    
    def run_self_test(self) -> bool:
        """
        Run comprehensive self-test with sample data validation.
        
        Returns:
            True if all tests pass, False otherwise
        """
        self.logger.info("🧪 Running UnifiedDataLoader self-test...")
        
        test_results = []
        
        try:
            # Test 1: Check dataset loading
            self.logger.info("Test 1: Dataset loading")
            if not self.datasets:
                test_results.append(("Dataset loading", False, "No datasets loaded"))
            else:
                missing_datasets = set(self.target_datasets) - set(self.datasets.keys())
                if missing_datasets:
                    test_results.append(("Dataset loading", False, f"Missing datasets: {missing_datasets}"))
                else:
                    test_results.append(("Dataset loading", True, f"All {len(self.datasets)} datasets loaded"))
            
            # Test 2: Check unified dataset creation
            self.logger.info("Test 2: Unified dataset creation")
            expected_splits = ['train', 'val', 'test']
            missing_splits = set(expected_splits) - set(self.unified_datasets.keys())
            if missing_splits:
                test_results.append(("Unified datasets", False, f"Missing splits: {missing_splits}"))
            else:
                test_results.append(("Unified datasets", True, f"All {len(self.unified_datasets)} splits created"))
            
            # Test 3: Check data loader creation
            self.logger.info("Test 3: Data loader creation")
            train_loader, val_loader, test_loader = self.get_dataloaders(batch_size=8)
            if train_loader is None:
                test_results.append(("Data loaders", False, "Failed to create train loader"))
            else:
                test_results.append(("Data loaders", True, "All data loaders created"))
            
            # Test 4: Check batch sampling
            self.logger.info("Test 4: Batch sampling")
            try:
                sample_batch = next(iter(train_loader))
                required_keys = ['data', 'label', 'dataset_names', 'batch_size']
                missing_keys = set(required_keys) - set(sample_batch.keys())
                if missing_keys:
                    test_results.append(("Batch sampling", False, f"Missing keys: {missing_keys}"))
                else:
                    batch_size = sample_batch['batch_size']
                    dataset_dist = sample_batch['dataset_distribution']
                    test_results.append(("Batch sampling", True, f"Batch size: {batch_size}, distribution: {dataset_dist}"))
            except Exception as e:
                test_results.append(("Batch sampling", False, f"Sampling error: {e}"))
            
            # Test 5: Check HSE prompt injection (if enabled)
            if self.enable_prompt_injection:
                self.logger.info("Test 5: HSE prompt injection")
                if 'system_prompts' in sample_batch and 'sample_prompts' in sample_batch:
                    test_results.append(("HSE prompts", True, "Prompts successfully injected"))
                else:
                    test_results.append(("HSE prompts", False, "Prompts not found in batch"))
            
            # Test 6: Check zero-shot evaluation data preparation
            self.logger.info("Test 6: Zero-shot evaluation data")
            try:
                if len(self.target_datasets) >= 2:
                    source_datasets = self.target_datasets[:-1]
                    target_dataset = self.target_datasets[-1]
                    source_loader, target_loader = self.get_zero_shot_evaluation_data(
                        source_datasets, target_dataset, batch_size=8
                    )
                    test_results.append(("Zero-shot data", True, "Zero-shot loaders created"))
                else:
                    test_results.append(("Zero-shot data", True, "Skipped (insufficient datasets)"))
            except Exception as e:
                test_results.append(("Zero-shot data", False, f"Zero-shot error: {e}"))
            
        except Exception as e:
            test_results.append(("Self-test", False, f"Unexpected error: {e}"))
        
        # Print results
        self.logger.info("🧪 Self-test results:")
        all_passed = True
        for test_name, passed, message in test_results:
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"  {test_name}: {status} - {message}")
            if not passed:
                all_passed = False
        
        overall_status = "✅ ALL TESTS PASSED" if all_passed else "❌ SOME TESTS FAILED"
        self.logger.info(f"🧪 Overall result: {overall_status}")
        
        return all_passed


def main():
    """Main entry point for standalone testing."""
    import argparse
    from types import SimpleNamespace
    
    parser = argparse.ArgumentParser(description="UnifiedDataLoader - Multi-dataset loading")
    parser.add_argument("--mode", choices=['unified', 'single'], default='unified',
                       help="Data loading mode")
    parser.add_argument("--enable_prompts", action='store_true',
                       help="Enable HSE prompt injection")
    parser.add_argument("--datasets", nargs='+', 
                       default=['CWRU', 'XJTU', 'THU'],
                       help="Target datasets to load")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create sample configuration
    data_config = SimpleNamespace(
        data_dir="data",
        metadata_file="metadata_6_11.xlsx",
        batch_size=16,
        num_workers=2,
        window_size=1024,
        normalization=True
    )
    
    task_config = SimpleNamespace(
        task_type="classification",
        num_classes=10
    )
    
    # Create unified data loader
    print("🚀 Creating UnifiedDataLoader...")
    loader = UnifiedDataLoader(
        data_config=data_config.__dict__,
        task_config=task_config.__dict__,
        mode=args.mode,
        enable_prompt_injection=args.enable_prompts,
        target_datasets=args.datasets
    )
    
    # Run self-test
    success = loader.run_self_test()
    
    # Print dataset info
    info = loader.get_dataset_info()
    print(f"\n📊 Dataset Information:")
    print(f"Mode: {info['mode']}")
    print(f"Loaded datasets: {info['loaded_datasets']}")
    print(f"Total samples: {info['total_samples']}")
    
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())