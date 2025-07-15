import types
from argparse import Namespace
import torch
from torch.utils.data import TensorDataset

from src.data_factory.id_data_factory import id_data_factory


def test_dataloader_initialization():
    args_data = Namespace(batch_size=2, num_workers=0)
    df = id_data_factory.__new__(id_data_factory)
    df.args_data = args_data
    df.train_dataset = TensorDataset(torch.randn(4, 1))
    df.val_dataset = TensorDataset(torch.randn(4, 1))
    df.test_dataset = TensorDataset(torch.randn(4, 1))
    loaders = id_data_factory._init_dataloader(df)
    assert len(loaders) == 3
