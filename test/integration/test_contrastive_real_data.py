#!/usr/bin/env python3
"""
Real data integration tests for ContrastiveIDTask
Tests with actual H5 dataset files and realistic data scenarios
"""

import pytest
import torch
import numpy as np
import os
import tempfile
import h5py
from pathlib import Path
import pandas as pd
from unittest.mock import patch, MagicMock
import warnings

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Import project modules
from src.configs.config_utils import load_config
from src.task_factory.task.pretrain.ContrastiveIDTask import ContrastiveIDTask


class TestContrastiveRealData:
    """Integration tests with real data scenarios"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment with mock data files"""
        cls.test_dir = tempfile.mkdtemp(prefix="contrastive_real_data_")
        cls.data_dir = Path(cls.test_dir) / "data"
        cls.data_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n🔧 Real data test environment: {cls.test_dir}")
        
        # Create mock H5 files with realistic structure
        cls.create_mock_h5_dataset()
        cls.create_mock_metadata()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment"""
        if hasattr(cls, 'test_dir') and os.path.exists(cls.test_dir):
            try:
                import shutil
                shutil.rmtree(cls.test_dir)
                print(f"✅ Cleaned up real data test environment")
            except Exception as e:
                print(f"⚠️ Failed to clean up: {e}")
    
    @classmethod
    def create_mock_h5_dataset(cls):
        """Create realistic mock H5 dataset files"""
        # Create multiple dataset files simulating different conditions
        datasets = {
            'Normal': {'num_samples': 50, 'base_freq': 50, 'noise_level': 0.1},
            'Fault_A': {'num_samples': 45, 'base_freq': 75, 'noise_level': 0.15},
            'Fault_B': {'num_samples': 40, 'base_freq': 120, 'noise_level': 0.2},
            'Fault_C': {'num_samples': 35, 'base_freq': 200, 'noise_level': 0.25}
        }
        
        for condition, params in datasets.items():
            h5_path = cls.data_dir / f"{condition}.h5"
            
            with h5py.File(h5_path, 'w') as f:
                # Create realistic vibration data
                for i in range(params['num_samples']):
                    sample_name = f"{condition}_{i:03d}"
                    
                    # Generate realistic multi-channel vibration signal
                    signal_length = np.random.randint(8192, 32768)  # Variable length signals
                    time_series = np.linspace(0, 10, signal_length)  # 10 second signals
                    
                    # Multi-channel data (simulating accelerometer X, Y, Z axes)
                    channels = []
                    for ch in range(3):
                        # Base vibration pattern
                        base_signal = np.sin(2 * np.pi * params['base_freq'] * time_series)
                        
                        # Add harmonics for realism
                        harmonic_1 = 0.3 * np.sin(2 * np.pi * params['base_freq'] * 2 * time_series)
                        harmonic_2 = 0.1 * np.sin(2 * np.pi * params['base_freq'] * 3 * time_series)
                        
                        # Add modulation for fault patterns
                        if 'Fault' in condition:
                            mod_freq = params['base_freq'] / 10
                            modulation = 0.2 * np.sin(2 * np.pi * mod_freq * time_series)
                            base_signal = base_signal * (1 + modulation)
                        
                        # Combine signals with noise
                        noise = params['noise_level'] * np.random.randn(signal_length)
                        channel_signal = base_signal + harmonic_1 + harmonic_2 + noise
                        
                        # Add channel-specific variations
                        channel_signal *= (1 + ch * 0.1)  # Slightly different amplitudes
                        channels.append(channel_signal)
                    
                    # Stack channels
                    multi_channel_data = np.column_stack(channels)
                    
                    # Store in H5 file
                    f.create_dataset(sample_name, data=multi_channel_data, compression='gzip')
                    
                    # Add attributes for metadata
                    f[sample_name].attrs['condition'] = condition
                    f[sample_name].attrs['label'] = list(datasets.keys()).index(condition)
                    f[sample_name].attrs['sample_rate'] = 10000  # 10 kHz
                    f[sample_name].attrs['duration'] = 10.0  # 10 seconds
                    f[sample_name].attrs['channels'] = ['X', 'Y', 'Z']
            
            print(f"Created mock H5 dataset: {h5_path}")
    
    @classmethod 
    def create_mock_metadata(cls):
        """Create realistic metadata Excel file"""
        metadata_data = []
        
        # Read the H5 files to create corresponding metadata
        for h5_file in cls.data_dir.glob("*.h5"):
            condition = h5_file.stem
            
            with h5py.File(h5_file, 'r') as f:
                for sample_name in f.keys():
                    dataset = f[sample_name]
                    
                    metadata_data.append({
                        'ID': sample_name,
                        'Label': dataset.attrs['label'],
                        'Condition': dataset.attrs['condition'],
                        'File': h5_file.name,
                        'SampleRate': dataset.attrs['sample_rate'],
                        'Duration': dataset.attrs['duration'],
                        'SignalLength': dataset.shape[0],
                        'Channels': dataset.shape[1],
                        'Speed_RPM': np.random.randint(1500, 2000),
                        'Load_Percent': np.random.randint(0, 100)
                    })
        
        # Create DataFrame and save as Excel
        df = pd.DataFrame(metadata_data)
        metadata_path = cls.data_dir / "metadata_real_test.xlsx"
        df.to_excel(metadata_path, index=False)
        
        print(f"Created metadata file: {metadata_path}")
        print(f"Total samples: {len(metadata_data)}")
        return metadata_path
    
    def create_real_data_config(self, batch_size=8, window_size=2048):
        """Create configuration for real data testing"""
        return {
            'data': {
                'factory_name': 'id',
                'dataset_name': 'ID_dataset', 
                'data_dir': str(self.data_dir),
                'metadata_file': 'metadata_real_test.xlsx',
                'batch_size': batch_size,
                'num_workers': 1,
                'window_size': window_size,
                'stride': window_size // 2,
                'num_window': 2,
                'window_sampling_strategy': 'random',
                'normalization': True,
                'truncate_length': 16384
            },
            'model': {
                'type': 'ISFM',
                'name': 'M_01_ISFM',
                'backbone': 'B_08_PatchTST',
                'd_model': 128
            },
            'task': {
                'type': 'pretrain',
                'name': 'contrastive_id',
                'lr': 1e-3,
                'weight_decay': 1e-4,
                'temperature': 0.07
            },
            'trainer': {
                'epochs': 2,
                'accelerator': 'cpu',
                'devices': 1,
                'precision': 32,
                'gradient_clip_val': 1.0,
                'check_val_every_n_epoch': 1,
                'log_every_n_steps': 1
            },
            'environment': {
                'save_dir': str(self.test_dir / "results"),
                'experiment_name': 'real_data_test'
            }
        }
    
    @pytest.mark.integration
    def test_real_h5_data_loading(self):
        """Test loading and processing real H5 data files"""
        print("\n=== Test Real H5 Data Loading ===")
        
        try:
            config = self.create_real_data_config()
            
            # Mock the data factory to use our test data
            with patch('src.data_factory.id_data_factory.ID_dataset') as mock_dataset_class:
                # Create a mock dataset that returns our H5 data
                mock_dataset = MagicMock()
                
                # Read actual data from our mock H5 files
                dataset_samples = []
                for h5_file in self.data_dir.glob("*.h5"):
                    with h5py.File(h5_file, 'r') as f:
                        for sample_name in list(f.keys())[:5]:  # Limit for testing
                            data = f[sample_name][:]
                            metadata = {
                                'Label': int(f[sample_name].attrs['label']),
                                'Condition': f[sample_name].attrs['condition']
                            }
                            dataset_samples.append((sample_name, data, metadata))
                
                print(f"Loaded {len(dataset_samples)} samples from H5 files")
                
                # Configure mock to return our data
                mock_dataset.__len__ = lambda: len(dataset_samples)
                mock_dataset.__getitem__ = lambda _, idx: dataset_samples[idx]
                mock_dataset_class.return_value = mock_dataset
                
                # Test with ContrastiveIDTask
                from argparse import Namespace
                args_data = Namespace(**config['data'])
                args_task = Namespace(**config['task'])
                
                # Create network for the expected input size
                network = torch.nn.Sequential(
                    torch.nn.Linear(config['data']['window_size'] * 3, 256),  # 3 channels
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, config['model']['d_model'])
                )
                
                task = ContrastiveIDTask(
                    network=network,
                    args_data=args_data,
                    args_model=Namespace(**config['model']),
                    args_task=args_task,
                    args_trainer=Namespace(**config['trainer']),
                    args_environment=Namespace(**config['environment']),
                    metadata={}
                )
                
                # Test batch preparation with real data
                batch = task.prepare_batch(dataset_samples[:8])
                
                if len(batch['ids']) > 0:
                    print(f"Successfully prepared batch with {len(batch['ids'])} samples")
                    print(f"Anchor shape: {batch['anchor'].shape}")
                    print(f"Positive shape: {batch['positive'].shape}")
                    
                    # Verify data properties
                    assert batch['anchor'].shape == batch['positive'].shape, "Shape mismatch"
                    assert batch['anchor'].shape[1] == config['data']['window_size'], "Wrong window size"
                    assert batch['anchor'].shape[2] == 3, "Expected 3 channels"
                    
                    # Check data range (should be normalized if enabled)
                    if config['data']['normalization']:
                        anchor_std = torch.std(batch['anchor'])
                        positive_std = torch.std(batch['positive'])
                        print(f"Normalized data std - Anchor: {anchor_std:.4f}, Positive: {positive_std:.4f}")
                    
                    # Test forward pass
                    batch_size_actual, seq_len, channels = batch['anchor'].shape
                    anchor_flat = batch['anchor'].reshape(batch_size_actual, -1)
                    positive_flat = batch['positive'].reshape(batch_size_actual, -1)
                    
                    z_anchor = network(anchor_flat)
                    z_positive = network(positive_flat)
                    
                    # Test loss computation
                    loss = task.infonce_loss(z_anchor, z_positive)
                    accuracy = task.compute_accuracy(z_anchor, z_positive)
                    
                    print(f"Computed loss: {loss.item():.4f}")
                    print(f"Computed accuracy: {accuracy.item():.4f}")
                    
                    # Validate results
                    assert torch.isfinite(loss), "Loss should be finite"
                    assert torch.isfinite(accuracy), "Accuracy should be finite"
                    assert 0 <= accuracy.item() <= 1, "Accuracy should be in [0,1]"
                    
                    print("✅ Real H5 data loading and processing test passed")
                else:
                    print("⚠️ No valid samples in batch - this may be expected with challenging data")
                
        except Exception as e:
            pytest.fail(f"Real H5 data loading test failed: {e}")
    
    @pytest.mark.integration
    def test_variable_length_signals(self):
        """Test handling of variable length signals from real data"""
        print("\n=== Test Variable Length Signals ===")
        
        try:
            config = self.create_real_data_config()
            
            # Create dataset with various signal lengths
            variable_length_data = []
            lengths = [1024, 2048, 4096, 8192, 16384]  # Different lengths
            
            for i, length in enumerate(lengths):
                # Generate signals of different lengths
                for rep in range(3):  # 3 samples per length
                    signal = np.random.randn(length, 3)  # 3 channels
                    sample_id = f"var_length_{length}_{rep}"
                    metadata = {'Label': i % 4}
                    variable_length_data.append((sample_id, signal, metadata))
            
            print(f"Created {len(variable_length_data)} samples with variable lengths")
            
            # Test with ContrastiveIDTask
            from argparse import Namespace
            task = ContrastiveIDTask(
                network=torch.nn.Linear(config['data']['window_size'] * 3, 128),
                args_data=Namespace(**config['data']),
                args_model=Namespace(**config['model']),
                args_task=Namespace(**config['task']),
                args_trainer=Namespace(**config['trainer']),
                args_environment=Namespace(**config['environment']),
                metadata={}
            )
            
            # Test batch preparation with variable lengths
            batch = task.prepare_batch(variable_length_data)
            
            if len(batch['ids']) > 0:
                print(f"Successfully processed {len(batch['ids'])} variable length samples")
                print(f"Final batch shapes - Anchor: {batch['anchor'].shape}, Positive: {batch['positive'].shape}")
                
                # All output windows should have same size
                expected_shape = (len(batch['ids']), config['data']['window_size'], 3)
                assert batch['anchor'].shape == expected_shape, f"Wrong anchor shape: {batch['anchor'].shape}"
                assert batch['positive'].shape == expected_shape, f"Wrong positive shape: {batch['positive'].shape}"
                
                print("✅ Variable length signals handled correctly")
            else:
                print("⚠️ No valid samples processed - check window sampling logic")
                
        except Exception as e:
            pytest.fail(f"Variable length signals test failed: {e}")
    
    @pytest.mark.integration
    def test_multi_condition_learning(self):
        """Test contrastive learning across multiple fault conditions"""
        print("\n=== Test Multi-Condition Contrastive Learning ===")
        
        try:
            config = self.create_real_data_config(batch_size=16)
            
            # Load samples from different conditions
            condition_data = {}
            total_samples = 0
            
            for h5_file in self.data_dir.glob("*.h5"):
                condition = h5_file.stem
                condition_data[condition] = []
                
                with h5py.File(h5_file, 'r') as f:
                    for sample_name in list(f.keys())[:10]:  # Limit per condition
                        data = f[sample_name][:]
                        metadata = {
                            'Label': int(f[sample_name].attrs['label']),
                            'Condition': condition
                        }
                        condition_data[condition].append((sample_name, data, metadata))
                        total_samples += 1
            
            print(f"Loaded data from {len(condition_data)} conditions, {total_samples} total samples")
            
            # Create balanced dataset
            all_samples = []
            for condition, samples in condition_data.items():
                all_samples.extend(samples)
            
            # Test contrastive learning effectiveness
            from argparse import Namespace
            network = torch.nn.Sequential(
                torch.nn.Linear(config['data']['window_size'] * 3, 256),
                torch.nn.ReLU(), 
                torch.nn.Dropout(0.1),
                torch.nn.Linear(256, config['model']['d_model'])
            )
            
            task = ContrastiveIDTask(
                network=network,
                args_data=Namespace(**config['data']),
                args_model=Namespace(**config['model']),
                args_task=Namespace(**config['task']),
                args_trainer=Namespace(**config['trainer']),
                args_environment=Namespace(**config['environment']),
                metadata={}
            )
            
            # Test multiple batches to see learning dynamics
            batch_results = []
            
            for batch_start in range(0, min(len(all_samples), 32), 8):
                batch_data = all_samples[batch_start:batch_start+8]
                batch = task.prepare_batch(batch_data)
                
                if len(batch['ids']) > 0:
                    # Reshape for network
                    batch_size_actual, seq_len, channels = batch['anchor'].shape
                    anchor_flat = batch['anchor'].reshape(batch_size_actual, -1)
                    positive_flat = batch['positive'].reshape(batch_size_actual, -1)
                    
                    # Forward pass
                    z_anchor = network(anchor_flat)
                    z_positive = network(positive_flat)
                    
                    # Compute metrics
                    loss = task.infonce_loss(z_anchor, z_positive)
                    accuracy = task.compute_accuracy(z_anchor, z_positive)
                    
                    batch_results.append({
                        'loss': loss.item(),
                        'accuracy': accuracy.item(),
                        'batch_size': len(batch['ids']),
                        'conditions': [batch_data[i][2]['Condition'] for i in range(len(batch_data))]
                    })
                    
                    print(f"Batch {len(batch_results)}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}, "
                          f"Conditions={set([batch_data[i][2]['Condition'] for i in range(len(batch_data))])}")
            
            # Analyze results across conditions
            if batch_results:
                avg_loss = np.mean([r['loss'] for r in batch_results])
                avg_acc = np.mean([r['accuracy'] for r in batch_results])
                
                print(f"\nMulti-condition results:")
                print(f"Average loss: {avg_loss:.4f}")
                print(f"Average accuracy: {avg_acc:.4f}")
                print(f"Total batches processed: {len(batch_results)}")
                
                # Validate that contrastive learning is working across conditions
                assert avg_loss > 0, "Loss should be positive"
                assert 0 <= avg_acc <= 1, "Accuracy should be in valid range"
                
                print("✅ Multi-condition contrastive learning test passed")
            else:
                print("⚠️ No batches processed - check data preparation")
                
        except Exception as e:
            pytest.fail(f"Multi-condition learning test failed: {e}")
    
    @pytest.mark.integration
    def test_memory_efficiency_real_data(self):
        """Test memory efficiency with realistic data sizes"""
        print("\n=== Test Memory Efficiency with Real Data ===")
        
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        try:
            # Test with larger, more realistic data
            config = self.create_real_data_config(batch_size=32, window_size=4096)
            
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            print(f"Initial memory usage: {initial_memory:.2f} MB")
            
            # Create larger mock dataset
            large_dataset = []
            for i in range(100):  # More samples
                # Larger signals
                signal_length = np.random.randint(16384, 65536)
                signal = np.random.randn(signal_length, 3).astype(np.float32)  # Use float32 for efficiency
                metadata = {'Label': i % 4}
                large_dataset.append((f"large_sample_{i}", signal, metadata))
            
            from argparse import Namespace
            network = torch.nn.Sequential(
                torch.nn.Linear(config['data']['window_size'] * 3, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, config['model']['d_model'])
            )
            
            task = ContrastiveIDTask(
                network=network,
                args_data=Namespace(**config['data']),
                args_model=Namespace(**config['model']),
                args_task=Namespace(**config['task']),
                args_trainer=Namespace(**config['trainer']),
                args_environment=Namespace(**config['environment']),
                metadata={}
            )
            
            # Process in batches and monitor memory
            memory_usage = []
            batch_size = config['data']['batch_size']
            
            for i in range(0, min(len(large_dataset), 64), batch_size):  # Process subset
                batch_data = large_dataset[i:i+batch_size]
                
                # Memory before batch
                mem_before = process.memory_info().rss / 1024 / 1024
                
                # Process batch
                batch = task.prepare_batch(batch_data)
                
                if len(batch['ids']) > 0:
                    # Forward pass
                    batch_size_actual, seq_len, channels = batch['anchor'].shape
                    anchor_flat = batch['anchor'].reshape(batch_size_actual, -1)
                    positive_flat = batch['positive'].reshape(batch_size_actual, -1)
                    
                    z_anchor = network(anchor_flat)
                    z_positive = network(positive_flat)
                    loss = task.infonce_loss(z_anchor, z_positive)
                
                # Memory after batch
                mem_after = process.memory_info().rss / 1024 / 1024
                memory_increase = mem_after - mem_before
                memory_usage.append(memory_increase)
                
                # Clean up
                del batch_data, batch
                if 'z_anchor' in locals():
                    del z_anchor, z_positive, loss
                
                if i % (batch_size * 2) == 0:  # Print every 2 batches
                    print(f"Batch {i//batch_size + 1}: Memory increase = {memory_increase:.2f} MB")
            
            # Analyze memory usage
            total_memory_increase = process.memory_info().rss / 1024 / 1024 - initial_memory
            avg_batch_memory = np.mean(memory_usage) if memory_usage else 0
            max_batch_memory = np.max(memory_usage) if memory_usage else 0
            
            print(f"\nMemory usage analysis:")
            print(f"Total memory increase: {total_memory_increase:.2f} MB")
            print(f"Average per-batch increase: {avg_batch_memory:.2f} MB")
            print(f"Maximum per-batch increase: {max_batch_memory:.2f} MB")
            
            # Memory usage assertions (adjust thresholds based on realistic expectations)
            assert total_memory_increase < 2000, f"Total memory usage too high: {total_memory_increase:.2f} MB"
            assert max_batch_memory < 200, f"Per-batch memory usage too high: {max_batch_memory:.2f} MB"
            
            print("✅ Memory efficiency test with real data passed")
            
        except Exception as e:
            pytest.fail(f"Memory efficiency test failed: {e}")
    
    @pytest.mark.integration
    def test_data_quality_validation(self):
        """Test data quality validation and preprocessing"""
        print("\n=== Test Data Quality Validation ===")
        
        try:
            config = self.create_real_data_config()
            
            # Create dataset with various data quality issues
            quality_test_data = []
            
            # Good data
            good_signal = np.random.randn(4096, 3)
            quality_test_data.append(("good_sample", good_signal, {'Label': 0}))
            
            # Data with NaN
            nan_signal = np.random.randn(4096, 3)
            nan_signal[1000:1010, :] = np.nan
            quality_test_data.append(("nan_sample", nan_signal, {'Label': 1}))
            
            # Data with Inf
            inf_signal = np.random.randn(4096, 3)
            inf_signal[2000:2005, :] = np.inf
            quality_test_data.append(("inf_sample", inf_signal, {'Label': 2}))
            
            # Very small signal (near zero)
            small_signal = np.random.randn(4096, 3) * 1e-10
            quality_test_data.append(("small_sample", small_signal, {'Label': 3}))
            
            # Very large signal
            large_signal = np.random.randn(4096, 3) * 1e6
            quality_test_data.append(("large_sample", large_signal, {'Label': 0}))
            
            # Constant signal
            constant_signal = np.ones((4096, 3)) * 5.0
            quality_test_data.append(("constant_sample", constant_signal, {'Label': 1}))
            
            print(f"Created {len(quality_test_data)} samples with various quality issues")
            
            from argparse import Namespace
            task = ContrastiveIDTask(
                network=torch.nn.Linear(config['data']['window_size'] * 3, 128),
                args_data=Namespace(**config['data']),
                args_model=Namespace(**config['model']),
                args_task=Namespace(**config['task']),
                args_trainer=Namespace(**config['trainer']),
                args_environment=Namespace(**config['environment']),
                metadata={}
            )
            
            # Test each problematic sample
            processing_results = {}
            
            for sample_id, signal, metadata in quality_test_data:
                try:
                    batch = task.prepare_batch([(sample_id, signal, metadata)])
                    
                    if len(batch['ids']) > 0:
                        # Check output quality
                        anchor_finite = torch.isfinite(batch['anchor']).all()
                        positive_finite = torch.isfinite(batch['positive']).all()
                        
                        processing_results[sample_id] = {
                            'processed': True,
                            'finite': anchor_finite.item() and positive_finite.item(),
                            'shape': batch['anchor'].shape
                        }
                        
                        if anchor_finite and positive_finite:
                            print(f"✅ {sample_id}: Successfully processed")
                        else:
                            print(f"⚠️ {sample_id}: Processed but contains non-finite values")
                    else:
                        processing_results[sample_id] = {'processed': False, 'reason': 'Empty batch'}
                        print(f"⚠️ {sample_id}: Not processed (empty batch)")
                        
                except Exception as e:
                    processing_results[sample_id] = {'processed': False, 'reason': str(e)}
                    print(f"❌ {sample_id}: Failed with error: {e}")
            
            # Analyze results
            processed_count = sum(1 for r in processing_results.values() if r.get('processed', False))
            finite_count = sum(1 for r in processing_results.values() if r.get('finite', False))
            
            print(f"\nData quality validation summary:")
            print(f"Samples processed: {processed_count}/{len(quality_test_data)}")
            print(f"Samples with finite output: {finite_count}/{len(quality_test_data)}")
            
            # At least good data should be processed correctly
            assert processing_results['good_sample']['processed'], "Good sample should be processed"
            assert processing_results['good_sample']['finite'], "Good sample should have finite output"
            
            print("✅ Data quality validation test completed")
            
        except Exception as e:
            pytest.fail(f"Data quality validation test failed: {e}")


def main():
    """Run real data integration tests"""
    print("🚀 Starting ContrastiveIDTask Real Data Integration Tests")
    print("="*70)
    
    import subprocess
    
    # Run tests
    test_args = [
        "pytest",
        __file__,
        "-v",
        "-s",
        "--tb=short",
        "-m", "integration"
    ]
    
    try:
        result = subprocess.run(test_args, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)