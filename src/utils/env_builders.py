import pandas as pd
from typing import List, Union

def build_env_traditional(train_files: List[str], val_files: List[str], test_files: List[str]):
    """Stub for traditional file upload environment builder.

    Parameters
    ----------
    train_files : list of str
        Uploaded training file paths.
    val_files : list of str
        Uploaded validation file paths.
    test_files : list of str
        Uploaded test file paths.

    Returns
    -------
    dict
        A dictionary describing the dataset environment.
    """
    return {
        "train_files": [f.name if hasattr(f, "name") else f for f in train_files],
        "val_files": [f.name if hasattr(f, "name") else f for f in val_files],
        "test_files": [f.name if hasattr(f, "name") else f for f in test_files],
    }


def build_env_phmbench(
    metadata: Union[str, pd.DataFrame],
    data_dir: str,
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
):
    """Stub for PHMbench metadata environment builder.

    Parameters
    ----------
    metadata : str or pandas.DataFrame
        Metadata table or path to file.
    data_dir : str
        Directory that stores h5 files.
    train_ids : list[str]
        Selected training ids.
    val_ids : list[str]
        Selected validation ids.
    test_ids : list[str]
        Selected testing ids.

    Returns
    -------
    dict
        A dictionary describing the dataset environment.
    """
    if isinstance(metadata, str):
        if metadata.endswith(".csv"):
            meta_df = pd.read_csv(metadata)
        else:
            meta_df = pd.read_excel(metadata)
    else:
        meta_df = metadata
    return {
        "metadata": meta_df,
        "data_dir": data_dir,
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
    }

