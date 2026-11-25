from types import SimpleNamespace

import numpy as np
import torch

from src.data_factory.global_affine import affine_parameters, fit_window_affine
from src.task_factory.Components.generative.normalization import build_normalization_evidence


def test_saved_global_affine_inverts_unseen_amplitudes_and_none_is_identity():
    windows = [np.arange(16.).reshape(8,2), 20+np.arange(24.).reshape(12,2)]
    evidence = fit_window_affine([SimpleNamespace(processed_data=[w]) for w in windows])
    mean,scale = affine_parameters(evidence)
    expected = np.concatenate(windows)
    np.testing.assert_allclose(mean, expected.mean(axis=0))
    np.testing.assert_allclose(scale, expected.std(axis=0))
    unseen = 100+windows[0]
    np.testing.assert_allclose(((unseen-mean)/scale)*scale+mean, unseen)
    identity = build_normalization_evidence(torch.tensor(unseen.T[None]),method="none")
    assert identity["channels"]["0"] == {"mean":0.,"std":1.}


def test_sliced_vibration_normalized_with_frozen_train_params():
    from src.data_factory.dataset_task.Default_dataset import Default_dataset
    n = 16
    raw = np.stack((np.arange(n)*1000, -np.arange(n)*1000, np.arange(n)*2),axis=1)[:,:,None]
    args = SimpleNamespace(window_size=8,num_window=2,window_sampling_strategy="sequential",
                           stride=8,split={"strategy":"grouped_metadata"},normalization="train_global_standardization",
                           channel_indices=[2],dtype="float32")
    meta = {1:{"Label":0},2:{"Label":0}}
    train=Default_dataset({1:raw},meta,args,SimpleNamespace(type="generative"),"train")
    val=Default_dataset({2:raw+20},meta,args,SimpleNamespace(type="generative"),"val")
    train.global_affine = val.global_affine = affine_parameters(fit_window_affine([train]))
    m,s = train.global_affine
    np.testing.assert_allclose(val[0]["x"]*s+m, raw[:8,2,:]+20, atol=1.e-5)
    assert val[0]["x"].shape == (8,1)
