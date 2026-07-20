# Paper 07 Baseline And Ablation Matrix

Status date: 2026-05-11

This matrix binds the Paper 07 baseline and ablation surface to concrete
PHM-Vibench commands. It is not accepted submission evidence yet. The commands
must still be run on the accepted industrial datasets with the same split,
seed protocol, metrics, preprocessing, and local GPU metadata.

Machine-readable matrix: `submission_prep/baseline_ablation_matrix.yaml`
Rejection-recovery evidence contract:
`submission_prep/rejection_recovery_contract.md`

## Validation Performed

The following models were checked with `python -m scripts.config_inspect` from
the parent repository and resolved to registered model/task targets:

| ID | Target |
|---|---|
| B01/A01 | `X_model.NSN` |
| B02 | `X_model.Resnet` |
| B03 | `X_model.Sincnet` |
| B04 | `X_model.TFN` |
| B05 | `X_model.WKN` |
| B06 | `Transformer.ConvTransformer` |
| B07 | `CNN.AttentionCNN` |
| A02/A03/A04/A05/A06 | `X_model.NSN` with operator/temperature overrides |

The base Python environment still reports `ModuleNotFoundError("No module named
'pytorch_lightning'")` for trainer import during config inspection. Use
`LQ_signal` for executable runs. Config target resolution is not a substitute
for final training evidence.

The local baseline commands were also executed in `LQ_signal` against the
dummy-data entrypoint:

| ID | Result | Metric |
|---|---|---|
| B01/A01 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7205665111541748`, `test_acc_Dummy_Data=0.0` |
| B02 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=1.1218299865722656`, `test_acc_Dummy_Data=0.0` |
| B03 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=4.801812171936035`, `test_acc_Dummy_Data=0.0` |
| B04 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.8415476679801941`, `test_acc_Dummy_Data=0.0` |
| B05 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.6259610652923584`, `test_acc_Dummy_Data=0.625` |
| B06 | pass after restoring legacy `register_model` compatibility and adding `model.input_dim=2`, CPU fallback because GPU/NVML was unavailable | `test_loss=7.736400604248047`, `test_acc_Dummy_Data=0.0` |
| B07 | pass after adding `model.input_dim=2`, CPU fallback because GPU/NVML was unavailable | `test_loss=0.677749752998352`, `test_acc_Dummy_Data=1.0` |

These smoke runs verify command executability only. They do not satisfy the
industrial-data, GPU-feasibility, baseline-table, or SOTA gates.

All six local ablation commands were also executed in `LQ_signal` against the
dummy-data entrypoint:

| ID | Result | Metric |
|---|---|---|
| A01 | pass, same run as B01, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7205665111541748`, `test_acc_Dummy_Data=0.0` |
| A02 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7205665111541748`, `test_acc_Dummy_Data=0.0` |
| A03 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.721332311630249`, `test_acc_Dummy_Data=0.0` |
| A04 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7206393480300903`, `test_acc_Dummy_Data=0.0` |
| A05 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7220381498336792`, `test_acc_Dummy_Data=0.0` |
| A06 | pass, CPU fallback because GPU/NVML was unavailable | `test_loss=0.7221447229385376`, `test_acc_Dummy_Data=0.0` |

These ablation smokes verify variant executability only. They do not satisfy
the industrial-data, GPU-feasibility, statistical, or SOTA gates.

## Required Next Evidence

1. Run every command in the YAML matrix with `CUDA_VISIBLE_DEVICES=0` or
   `CUDA_VISIBLE_DEVICES=1`.
2. Capture `run_meta.yaml`, `metrics.json`, stdout/stderr, seed, batch size,
   precision, runtime, GPU model, and OOM/failure reason.
3. Repeat under the accepted CWRU/XJTU or industrial protocol with enough
   seeds for mean, standard deviation, and confidence intervals.
4. Bind TimeMixer, SARAD, CATCH, DADA, PGRFNet, GTM, CSLSTM, and TSPulse to
   exact or representative local commands, or mark exact reproduction as
   `resource-blocked` under the 2x4090 budget.
5. Allow SOTA wording only after the proposed method beats all accepted
   baselines under the same protocol.
