# High-Performance Computing (HPC) Usage

PHM-Vibench does not currently maintain a cluster-specific High-Performance
Computing (HPC) deployment guide.

The former content at this path was a YCRC/Slurm note containing changing
site-specific partition names, hardware counts, queue advice, generic
`src/train.py` commands, and local environment assumptions. It was not validated
against the maintained PHM-Vibench entrypoint and should not be used as current
cluster policy.

## Stable PHM-Vibench contract

A cluster job should invoke the same public command used locally:

```bash
python main.py --config <yaml> [--override key=value ...]
```

Keep site-specific values outside maintained demo configurations. Typical local
or scheduler-provided overrides include:

```text
data.data_dir
trainer.device
trainer.devices
data.num_workers
environment.output_dir
```

Start by validating the configuration and running a small CPU or single-device
job:

```bash
python -m scripts.config_inspect \
  --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.num_epochs=1

python main.py --config configs/demo/00_smoke/dummy_dg.yaml \
  --override trainer.device=cpu \
  --override trainer.devices=1 \
  --override trainer.num_epochs=1 \
  --override data.num_workers=0
```

## Site documentation is authoritative

For modules, partitions, GPU types, wall-time limits, storage, job arrays,
containers, and queue policy, follow the current documentation of the cluster
operator. Do not copy another institution's Slurm directives or filesystem paths
without review.

A future PHM-Vibench HPC guide should be site-neutral and tested. Cluster-specific
examples should identify their site, verification date, external source, expected
environment, and maintenance owner.

The previous detailed YCRC note remains recoverable from Git history; it is not
part of the maintained documentation surface.
