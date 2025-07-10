import numpy as np
import torch
from typing import Any, Dict, List


def create_windows(data: np.ndarray, args_data: Any) -> List[np.ndarray]:
    """Split an array into windows.

    Parameters
    ----------
    data : np.ndarray
        Raw sample array of shape ``(L, C)`` or ``(L,)``.
    args_data : Namespace
        Should define ``window_size``, ``stride`` and ``num_window``. Optionally
        ``window_sampling_strategy``.

    Returns
    -------
    List[np.ndarray]
        Sequence of windows with length ``window_size``. Empty list if input is
        shorter than the window size.
    """
    win_size = args_data.window_size
    stride = args_data.stride
    num_window = args_data.num_window
    strategy = getattr(args_data, 'window_sampling_strategy', 'evenly_spaced')

    data_length = len(data)
    windows: List[np.ndarray] = []

    if data_length < win_size:
        return windows

    if strategy == 'sequential':
        num = max(0, (data_length - win_size) // stride + 1)
        num = min(num, num_window)
        for i in range(num):
            s = i * stride
            windows.append(data[s:s + win_size])
    elif strategy == 'random':
        possible = np.arange(data_length - win_size + 1)
        starts = possible if len(possible) < num_window else np.random.choice(possible, size=num_window, replace=False)
        for s in starts:
            windows.append(data[s:s + win_size])
    else:  # evenly_spaced
        if num_window <= 1:
            start = max(0, (data_length - win_size) // 2)
            windows.append(data[start:start + win_size])
        else:
            effective = data_length - win_size
            step = 0 if num_window == 1 else effective / (num_window - 1)
            for i in range(num_window):
                s = int(round(i * step))
                s = min(s, data_length - win_size)
                windows.append(data[s:s + win_size])

    return windows


def process_sample(data: np.ndarray, args_data: Any) -> np.ndarray:
    """Normalize and reshape one sample.

    Parameters
    ----------
    data : np.ndarray
        Raw array from the dataset.
    args_data : Namespace
        Contains ``dtype`` and ``normalization`` configuration.

    Returns
    -------
    np.ndarray
        Processed array suitable for model input.
    """
    if getattr(args_data, 'dtype', None):
        if args_data.dtype == 'float32':
            data = data.astype(np.float32)
        elif args_data.dtype == 'float64':
            data = data.astype(np.float64)

    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)

    norm = getattr(args_data, 'normalization', None)
    if norm == 'minmax':
        min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        denom = max_vals - min_vals
        denom[denom == 0] = 1
        data = (data - min_vals) / denom
    elif norm == 'standardization':
        mean_vals = np.mean(data, axis=0)
        std_vals = np.std(data, axis=0)
        data = (data - mean_vals) / (std_vals + 1e-8)
    return data


def prepare_batch(batch: Dict[str, Any], args_data: Any) -> Dict[str, Any]:
    """Transform a raw batch from :class:`ID_dataset`.

    Parameters
    ----------
    batch : dict
        Keys ``'data'``, ``'metadata'`` and ``'id'`` contain lists.
    args_data : Namespace
        Passed to :func:`process_sample` and :func:`create_windows`.

    Returns
    -------
    dict
        ``{'x': Tensor, 'y': Tensor, 'file_id': List[str]}``
    """
    xs, ys, fids = [], [], []
    for data_arr, meta, fid in zip(batch['data'], batch['metadata'], batch['id']):
        arr = process_sample(np.array(data_arr), args_data)
        windows = create_windows(arr, args_data)
        if not windows:
            continue
        xs.append(torch.tensor(windows[0], dtype=torch.float32))
        ys.append(meta['Label'])
        fids.append(fid)
    return {
        'x': torch.stack(xs),
        'y': torch.tensor(ys, dtype=torch.long),
        'file_id': fids,
    }
