"""One invertible per-channel transform for a fixed set of training windows."""
import numpy as np


def fit_window_affine(datasets):
    count = 0
    windows = 0
    mean = None
    squared = None
    for dataset in datasets:
        for window in dataset.processed_data:
            x = np.asarray(window, dtype=np.float64)
            x = x.reshape(len(x), -1)
            local = x.mean(axis=0)
            within = np.sum((x-local)**2, axis=0)
            if mean is None:
                mean, squared = local, within
            else:
                if mean.shape != local.shape:
                    raise ValueError("Training windows disagree about channel count")
                delta = local - mean
                squared += within + delta**2 * (count * len(x) / (count+len(x)))
                mean += delta * (len(x)/(count+len(x)))
            count += len(x)
            windows += 1
    if not count:
        raise ValueError("Empty global-affine training set")
    scale = np.sqrt(squared/count)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Training channels require finite mean and positive scale")
    return {"schema_version":"0.2.1", "method":"train_global_standardization",
            "scope":"per_channel", "source_split":"train", "source":"selected_raw_training_windows",
            "num_windows":windows, "num_values_per_channel":count,
            "channels_count":len(mean), "channels":{str(i):{"mean":float(m),"std":float(s)}
                for i,(m,s) in enumerate(zip(mean,scale))}}


def affine_parameters(evidence):
    if (evidence.get("source_split"), evidence.get("method")) != ("train","train_global_standardization"):
        raise ValueError("Expected a saved training global affine transform")
    channels = evidence["channels"]
    mean, scale = (np.array([channels[str(i)][k] for i in range(len(channels))])
                   for k in ("mean","std"))
    if not len(mean) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Invalid saved affine channel parameters")
    return mean, scale
