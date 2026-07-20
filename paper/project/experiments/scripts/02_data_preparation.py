#!/usr/bin/env python3
"""
Data Preparation Script

Prepares datasets for LLM-enhanced fault diagnosis experiments.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add the toolkit to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../code'))

try:
    import torch
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
except ImportError as e:
    print(f"Required package not found: {e}")
    print("Please install with: pip install scikit-learn")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreparation:
    """Data preparation class for fault diagnosis experiments."""

    def __init__(self, config_path):
        """
        Initialize data preparation.

        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.setup_directories()

        # Initialize scalers
        self.scaler = None
        self.label_encoder = None

    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            # Return default config
            return self._get_default_config()

    def _get_default_config(self):
        """Get default configuration."""
        return {
            'data': {
                'dataset_name': 'THU_018',
                'sampling_rate': 1024,
                'segment_length': 4096,
                'normalization': 'z_score',
                'train_test_split': 0.2,
                'validation_split': 0.1,
                'random_seed': 42
            },
            'output': {
                'base_path': './results'
            }
        }

    def setup_directories(self):
        """Setup output directories."""
        base_path = Path(self.config.get('output', {}).get('base_path', './results'))
        self.output_dir = base_path / 'data' / 'processed'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {self.output_dir}")

    def load_thu_dataset(self, dataset_path):
        """
        Load THU dataset for fault diagnosis.

        Args:
            dataset_path: Path to THU dataset

        Returns:
            Loaded data dictionary
        """
        logger.info(f"Loading THU dataset from {dataset_path}")

        # This is a mock implementation
        # In real usage, you would load the actual THU dataset

        dataset_name = self.config['data']['dataset_name']
        segment_length = self.config['data']['segment_length']

        # Generate synthetic data for demonstration
        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset path {dataset_path} not found, generating synthetic data")
            return self.generate_synthetic_dataset()

        # Mock data loading structure
        # In reality, this would load actual .h5, .mat, or .npz files
        data = {
            'signals': self._load_signals(dataset_path),
            'labels': self._load_labels(dataset_path),
            'metadata': self._load_metadata(dataset_path)
        }

        logger.info(f"Loaded {len(data['signals'])} samples")
        return data

    def generate_synthetic_dataset(self):
        """
        Generate synthetic dataset for testing.

        Returns:
            Synthetic data dictionary
        """
        logger.info("Generating synthetic dataset for testing")

        dataset_name = self.config['data']['dataset_name']
        segment_length = self.config['data']['segment_length']
        sampling_rate = self.config['data']['sampling_rate']

        # Parameters for synthetic signals
        num_samples = 1000
        t = np.linspace(0, segment_length/sampling_rate, segment_length)

        # Fault types and their characteristics
        fault_types = {
            0: {'name': '正常', 'description': 'Normal operation'},
            1: {'name': '内圈故障', 'description': 'Inner race fault'},
            2: {'name': '外圈故障', 'description': 'Outer race fault'},
            3: {'name': '滚动体故障', 'description': 'Ball defect'},
            4: {'name': '保持架故障', 'description': 'Cage damage'},
            5: {'name': '不对中', 'description': 'Misalignment'},
            6: {'name': '不平衡', 'description': 'Imbalance'},
            7: {'name': '松动', 'description': 'Looseness'},
            8: {'name': '齿轮故障', 'description': 'Gear fault'},
            9: {'name': '其他故障', 'description': 'Other fault'}
        }

        signals = []
        labels = []
        fault_descriptions = []

        np.random.seed(self.config['data']['random_seed'])

        for i in range(num_samples):
            fault_type = np.random.randint(0, len(fault_types))

            # Generate signal based on fault type
            signal = self._generate_signal_by_fault_type(t, fault_type, dataset_name)

            signals.append(signal)
            labels.append(fault_type)
            fault_descriptions.append(fault_types[fault_type])

        # Convert to numpy arrays
        signals = np.array(signals)
        labels = np.array(labels)

        data = {
            'signals': signals,
            'labels': labels,
            'metadata': {
                'dataset_name': dataset_name,
                'num_samples': num_samples,
                'segment_length': segment_length,
                'sampling_rate': sampling_rate,
                'fault_types': fault_types,
                'generation_time': datetime.now().isoformat(),
                'synthetic': True
            }
        }

        logger.info(f"Generated synthetic dataset with {num_samples} samples")
        return data

    def _generate_signal_by_fault_type(self, t, fault_type, dataset_name):
        """Generate signal based on fault type."""

        # Base parameters
        fs = self.config['data']['sampling_rate']
        shaft_freq = 30  # Typical shaft frequency

        signal = np.zeros_like(t)

        if fault_type == 0:  # Normal
            # Normal operation: mainly shaft frequency
            signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 1:  # Inner race fault
            bpfi = 3.05 * shaft_freq
            signal = 0.2 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.3 * np.sin(2 * np.pi * bpfi * t)
            signal += 0.1 * np.sin(2 * np.pi * 2 * bpfi * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 2:  # Outer race fault
            bpfo = 2.05 * shaft_freq
            signal = 0.15 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.25 * np.sin(2 * np.pi * bpfo * t)
            signal += 0.08 * np.sin(2 * np.pi * 2 * bpfo * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 3:  # Ball defect
            bsf = 2.35 * shaft_freq
            ftf = 0.4 * shaft_freq
            signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.2 * np.sin(2 * np.pi * bsf * t)
            signal += 0.1 * np.sin(2 * np.pi * ftf * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 4:  # Misalignment
            signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.3 * np.sin(2 * np.pi * 2 * shaft_freq * t)
            signal += 0.15 * np.sin(2 * np.pi * 3 * shaft_freq * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 5:  # Imbalance
            signal = 0.4 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.05 * np.random.randn(len(t))

        elif fault_type == 6:  # Looseness
            signal = 0.2 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.3 * np.sin(2 * np.pi * (shaft_freq + 50) * t)
            signal += 0.1 * np.random.randn(len(t))

        elif fault_type == 7:  # Gear fault
            gear_mesh_freq = shaft_freq * 3.5  # Example gear ratio
            signal = 0.15 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.25 * np.sin(2 * np.pi * gear_mesh_freq * t)
            signal += 0.1 * np.sin(2 * np.pi * 2 * gear_mesh_freq * t)
            signal += 0.05 * np.random.randn(len(t))

        else:  # Other faults (mixed characteristics)
            signal = 0.2 * np.sin(2 * np.pi * shaft_freq * t)
            signal += 0.15 * np.sin(2 * np.pi * 1.7 * shaft_freq * t)
            signal += 0.1 * np.sin(2 * np.pi * 3.3 * shaft_freq * t)
            signal += 0.05 * np.random.randn(len(t))

        # Add dataset-specific modifications
        if dataset_name == "THU_018":
            # Add some THU_018 specific characteristics
            signal += 0.02 * np.sin(2 * np.pi * 0.5 * shaft_freq * t)

        return signal

    def _load_signals(self, dataset_path):
        """Load signal data from dataset."""
        # Mock implementation
        # In reality, this would load actual signal files
        return np.random.randn(100, 4096, 1)

    def _load_labels(self, dataset_path):
        """Load label data from dataset."""
        # Mock implementation
        # In reality, this would load actual label files
        return np.random.randint(0, 10, 100)

    def _load_metadata(self, dataset_path):
        """Load metadata from dataset."""
        return {
            'dataset_type': 'synthetic',
            'creation_date': datetime.now().isoformat()
        }

    def preprocess_data(self, data):
        """
        Preprocess the loaded data.

        Args:
            data: Raw data dictionary

        Returns:
            Preprocessed data dictionary
        """
        logger.info("Preprocessing data")

        signals = data['signals']
        labels = data['labels']

        # Normalization
        if self.config['data']['normalization'] == 'z_score':
            signals = self._z_score_normalize(signals)
        elif self.config['data']['normalization'] == 'min_max':
            signals = self._min_max_normalize(signals)

        # Data splitting
        train_data, test_data, train_labels, test_labels = train_test_split(
            signals, labels,
            test_size=self.config['data']['train_test_split'],
            random_state=self.config['data']['random_seed'],
            stratify=labels
        )

        # Validation split from training data
        if self.config['data']['validation_split'] > 0:
            train_data, val_data, train_labels, val_labels = train_test_split(
                train_data, train_labels,
                test_size=self.config['data']['validation_split'],
                random_state=self.config['data']['random_seed'],
                stratify=train_labels
            )
        else:
            val_data, val_labels = None, None

        # Reshape for PyTorch (batch, sequence, channels)
        train_data = train_data.reshape(train_data.shape[0], train_data.shape[1], 1)
        test_data = test_data.reshape(test_data.shape[0], test_data.shape[1], 1)
        if val_data is not None:
            val_data = val_data.reshape(val_data.shape[0], val_data.shape[1], 1)

        processed_data = {
            'train': {
                'signals': train_data,
                'labels': train_labels
            },
            'test': {
                'signals': test_data,
                'labels': test_labels
            },
            'validation': {
                'signals': val_data,
                'labels': val_labels
            } if val_data is not None else None,
            'metadata': data['metadata'],
            'preprocessing': {
                'normalization': self.config['data']['normalization'],
                'preprocessing_time': datetime.now().isoformat()
            }
        }

        # Log data statistics
        self._log_data_statistics(processed_data)

        return processed_data

    def _z_score_normalize(self, signals):
        """Apply Z-score normalization."""
        if self.scaler is None:
            self.scaler = StandardScaler()

        # Reshape for sklearn
        original_shape = signals.shape
        signals_2d = signals.reshape(signals.shape[0], -1)

        # Fit and transform
        if hasattr(self.scaler, 'fit_transform'):
            signals_normalized = self.scaler.fit_transform(signals_2d)
        else:
            signals_normalized = signals_2d  # Fallback

        # Reshape back
        return signals_normalized.reshape(original_shape)

    def _min_max_normalize(self, signals):
        """Apply Min-Max normalization."""
        if self.scaler is None:
            self.scaler = MinMaxScaler()

        # Reshape for sklearn
        original_shape = signals.shape
        signals_2d = signals.reshape(signals.shape[0], -1)

        # Fit and transform
        if hasattr(self.scaler, 'fit_transform'):
            signals_normalized = self.scaler.fit_transform(signals_2d)
        else:
            signals_normalized = signals_2d  # Fallback

        # Reshape back
        return signals_normalized.reshape(original_shape)

    def _log_data_statistics(self, data):
        """Log data statistics."""
        logger.info("Data Statistics:")

        train_signals = data['train']['signals']
        test_signals = data['test']['signals']

        logger.info(f"Training set: {train_signals.shape}")
        logger.info(f"Test set: {test_signals.shape}")

        if data['validation'] is not None:
            val_signals = data['validation']['signals']
            logger.info(f"Validation set: {val_signals.shape}")

        # Signal statistics
        logger.info(f"Signal mean: {train_signals.mean():.4f}")
        logger.info(f"Signal std: {train_signals.std():.4f}")
        logger.info(f"Signal min: {train_signals.min():.4f}")
        logger.info(f"Signal max: {train_signals.max():.4f}")

    def save_processed_data(self, data, filename=None):
        """
        Save processed data to file.

        Args:
            data: Processed data dictionary
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_name = self.config['data']['dataset_name']
            filename = f"{dataset_name}_processed_{timestamp}.npz"

        output_path = self.output_dir / filename

        try:
            # Save data
            save_dict = {
                'train_signals': data['train']['signals'],
                'train_labels': data['train']['labels'],
                'test_signals': data['test']['signals'],
                'test_labels': data['test']['labels'],
                'metadata': data['metadata'],
                'preprocessing': data['preprocessing'],
                'config': self.config
            }

            if data['validation'] is not None:
                save_dict['val_signals'] = data['validation']['signals']
                save_dict['val_labels'] = data['validation']['labels']

            np.savez_compressed(output_path, **save_dict)
            logger.info(f"Processed data saved to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save data: {e}")

        return output_path

    def generate_data_report(self, data):
        """
        Generate a data report.

        Args:
            data: Processed data dictionary
        """
        report = {
            'dataset_name': self.config['data']['dataset_name'],
            'preprocessing_time': datetime.now().isoformat(),
            'statistics': {
                'total_samples': len(data['train']['signals']) + len(data['test']['signals']),
                'train_samples': len(data['train']['signals']),
                'test_samples': len(data['test']['signals']),
                'signal_length': data['train']['signals'].shape[1],
                'num_channels': data['train']['signals'].shape[2],
                'num_classes': len(np.unique(data['train']['labels'])),
                'class_distribution': {
                    'train': dict(zip(*np.unique(data['train']['labels'], return_counts=True))),
                    'test': dict(zip(*np.unique(data['test']['labels'], return_counts=True)))
                }
            },
            'preprocessing': data['preprocessing'],
            'metadata': data['metadata']
        }

        # Save report
        report_path = self.output_dir / f"data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            import json
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Data report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

        return report


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Data preparation for LLM-enhanced fault diagnosis')
    parser.add_argument('--config', '-c',
                        default='../configs/base_config.yaml',
                        help='Configuration file path')
    parser.add_argument('--dataset', '-d',
                        default=None,
                        help='Dataset path (auto-generated if not provided)')
    parser.add_argument('--output', '-o',
                        default=None,
                        help='Output directory (overwrites config)')

    args = parser.parse_args()

    logger.info("Starting data preparation")
    logger.info(f"Config file: {args.config}")

    # Initialize data preparation
    data_prep = DataPreparation(args.config)

    # Override output directory if specified
    if args.output:
        data_prep.output_dir = Path(args.output)
        data_prep.output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate dataset
    dataset_path = args.dataset
    if dataset_path is None:
        dataset_path = f"/home/user/data/PHMbenchdata/PHM-Vibench/{data_prep.config['data']['dataset_name']}"

    try:
        # Load data
        data = data_prep.load_thu_dataset(dataset_path)

        # Preprocess data
        processed_data = data_prep.preprocess_data(data)

        # Save processed data
        saved_path = data_prep.save_processed_data(processed_data)

        # Generate report
        report = data_prep.generate_data_report(processed_data)

        logger.info("Data preparation completed successfully")
        logger.info(f"Processed data saved to: {saved_path}")

        return processed_data

    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        raise


if __name__ == "__main__":
    main()