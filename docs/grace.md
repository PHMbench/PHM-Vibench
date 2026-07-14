# Grace Cluster Notes — Retired

This compatibility path previously contained a contributor-specific filesystem
path, local Conda environment name, and editor launch commands. Those values are
not portable PHM-Vibench documentation and have been removed.

Use:

- [Installation](installation.md) for the supported environment baseline;
- [Quickstart](quickstart.md) for the maintained runtime command;
- [HPC usage](HPC.md) for the site-neutral scheduler boundary;
- `configs/local/local.yaml` or CLI overrides for machine-specific data paths.

Never commit a personal cluster path into a maintained configuration or user
guide. The prior note remains recoverable from Git history if needed for private
provenance.
