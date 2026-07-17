# Foundation-model adapters

This directory contains narrow adapters around optional upstream packages. The
upstream implementation and checkpoint are not copied into PHM-Vibench.

## MantisV1

Install the separately pinned dependency:

```bash
python -m pip install -r requirements-optional-mantis.txt
```

Download or otherwise provision the official checkpoint outside this repository,
then point the experiment at its local directory:

```yaml
model:
  type: FoundationModel
  name: MantisV1
  checkpoint_path: artifacts/checkpoints/mantis-8m
  seq_len: 512
  input_channels: 1
  freeze_backbone: true
```

The adapter never accepts a remote model ID. It hashes the local checkpoint
directory, freezes the upstream backbone, encodes channels independently, and
trains only a LayerNorm plus linear classification head. Input shape is
`[batch, length, channels]`; length must match `seq_len` exactly and be divisible
by 32. Any resize is an explicit data-protocol decision.

This is an experimental adapter, not a maintained release-supported model. A
real-checkpoint smoke test and leakage-safe dataset benchmark are required before
promotion beyond `experimental_candidate`.

Upstream references:

- Paper: <https://arxiv.org/abs/2502.15637>
- Code: <https://github.com/vfeofanov/mantis>
