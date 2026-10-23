"""Generate small labeled tensor fixtures and a native TII M0 configuration.

These are constructed sine signals, never industrial samples or J1-C evidence.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    metadata, records = [], []
    for source, rate in ((1, 64), (2, 16)):
        name = f'TII_tensor_fixture_{source}'
        with h5py.File(root / f'{name}.h5', 'w') as h5:
            h5.attrs['evidence_kind'] = 'tensor_fixture_not_industrial_data'
            for i in range(8):
                fid = (source-1)*8+i
                label = i % 2
                t = np.arange(2*rate)/rate
                signal = np.sin(2*np.pi*(3+2*label)*t)
                if source == 1:
                    signal += .3*np.cos(2*np.pi*(12+4*label)*t)
                signal *= 1+.03*i
                h5.create_dataset(str(fid), data=signal.astype('float32')[:, None, None])
                metadata.append(dict(Id=fid, Name=name, Dataset_id=source, Label=label,
                                     Sample_rate=rate, Channel=1, File=f'tensor_fixture_{fid}', Domain_id=i//4))
                records.append(dict(Id=fid, dataset_id=source, recording_id=i, group=i,
                    role='source_train' if i < 4 else 'source_val', native_rate_hz=rate,
                    effective_rate_hz=rate, channel=0, input_unit='m/s^2', unit_scale=1.,
                    common_support='fully_usable', increment_support='fully_usable' if source == 1 else 'unavailable',
                    support_basis='tensor_fixture', support_evidence='Analytic sine fixture, not hardware qualification'))
    pd.DataFrame(metadata).to_csv(root/'metadata.csv', index=False)
    pd.DataFrame(records).to_csv(root/'records.csv', index=False)
    config = dict(
        pipeline='Pipeline_01_Fault_Diagnosis',
        environment=dict(project='tii_M0_tensor_fixture', seed=0, iterations=1,
            output_dir=str(root/'runs'), notes='M0 tensor fixture; not industrial J1-C', wandb=False, swanlab=False),
        data=dict(factory_name='default', data_dir=str(root), metadata_file='metadata.csv',
            record_inventory=str(root/'records.csv'), cache_dir=str(root/'scratch'), use_cache=True,
            evidence_kind='tensor_fixture', normalization='source_rms', batch_size=32, source_batch_size=32,
            num_workers=0, seed=0, rounds=20, duration_s=1., grid_rate_hz=64., num_patches=4,
            patch_size=16, output_unit='m/s^2', common_bands_hz=[[1., 8.]], increment_bands_hz=[[8., 24.]]),
        model=dict(type='ISFM', name='M_01_ISFM', embedding='SupportConditionedTokenizer',
            token_organization='ordinary', backbone='B_04_Dlinear', task_head='H_01_Linear_cla',
            patch_size_L=16, num_patches=4, output_dim=8, d_model=8, source_rms='source_train'),
        task=dict(type='DG', name='tii_joint', target_system_id=[1, 2], source_system_ids=[1, 2],
            sampling_seed=0, loss='CE', metrics=['acc'], optimizer='adamw', lr=.001, weight_decay=.0001,
            lambda_common=0, lambda_private=0, scheduler=None),
        trainer=dict(name='Default_trainer', num_epochs=1, test_after_fit=False, early_stopping=False,
            device='cpu', devices=1, monitor='val_group_nll', monitor_mode='min', checkpoint_min_delta=1e-8,
            val_check_interval=10, num_sanity_val_steps=0, log_every_n_steps=1, deterministic=True))
    (root/'native.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(root/'native.yaml')


if __name__ == '__main__': main()
