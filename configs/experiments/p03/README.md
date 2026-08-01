# P03 — Evidence-Grounded LLM XFD: experiment configs

Paper: `P03-Evidence-Grounded-LLM-XFD`
Engine: PHM-Vibench_fix (`python main.py --config <yaml> [--override key=value ...]`)

This directory now has two deliberately separated surfaces:

- the retained **diagnostic-backbone** configs, which can provide frozen
  upstream evidence but do not support a new backbone-accuracy claim; and
- `verifiable_report_smoke.yaml`, the CPU-only typed claim--evidence verifier
  contract used to test report verification without invoking a renderer.

The backbone configs are composed from maintained base blocks
(`configs/base/**`) and registered component IDs only. Each derives from a
maintained demo and retains a smoke-friendly one-epoch default.

## Why a separate group of configs

P03 has two layers of evidence:

1. **Diagnostic backbone** — train/eval fault-diagnosis models on real PHM
   datasets and feed their (prediction, structured explanation) output into the
   LLM explanation protocol. This layer IS reproducible on PHM-Vibench_fix and
   is what these configs address.
2. **Verifiable-report layer** — the deterministic verifier is implemented and
   smoke-tested in `src/utils/claim_evidence_verifier.py`. The matched renderer
   comparison remains `review_only`: generic, RAG, structured, PCN-style,
   RefChecker-style, and proposed-condition adapters may not generate
   evidence-bearing results until the experiment protocol is explicitly
   approved. The old human-study, deployment, and backbone-superiority claims
   remain outside the current paper boundary.

## Files

| File | Derives from | Pipeline | Model | Task | Dataset IDs |
|---|---|---|---|---|---|
| `verifiable_report_smoke.yaml` | `paper/method/method_contract.md` | standalone deterministic verifier | none | report verification | embedded synthetic contract only |
| `e1_tspn_uxfd_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `X_model / TSPN_UXFD` | `CDDG / classification` | 1,2,6,14,16 (CWRU,XJTU,THU,THU24,DIRG) |
| `e2_isfm_dlinear_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `ISFM / M_01_ISFM` (E_01_HSE/B_04_Dlinear/H_01_Linear_cla) | `CDDG / classification` | 1,2,6,14,16 |
| `e3_resnet1d_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `CNN / ResNet1D` | `CDDG / classification` | 1,2,6,14,16 |
| `e4_convtransformer_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `Transformer / ConvTransformer` | `CDDG / classification` | 1,2,6,14,16 |
| `e5_base_explainable_cnn_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `X_model / BASE_ExplainableCNN` | `CDDG / classification` | 1,2,6,14,16 |
| `e6_mwa_cnn_cddg.yaml` | `configs/demo/02_cross_system/multi_system_cddg.yaml` | `Pipeline_01_default` | `X_model / MWA_CNN` | `CDDG / classification` | 1,2,6,14,16 |
| `e1_tspn_uxfd_cwru_dg.yaml` | `configs/demo/01_cross_domain/cwru_dg.yaml` | `Pipeline_01_default` | `X_model / TSPN_UXFD` | `DG / classification` | 1 (CWRU, intra-dataset domain split) |

## Verifier smoke

The smoke is wiring evidence only and cannot support the paper claim:

```bash
conda run -n LQ_signal python -m src.utils.claim_evidence_verifier \
  --config configs/experiments/p03/verifiable_report_smoke.yaml \
  --output /tmp/p03-verifiable-report-smoke.json

conda run -n LQ_signal python -m pytest -q \
  test/test_claim_evidence_verifier.py
```

## Dataset ID map (from `data/metadata.xlsx`)

- 1 = RM_001_CWRU (CWRU)
- 2 = RM_002_XJTU (XJTU)
- 6 = RM_006_THU (≈ THU_006 in paper text)
- 14 = RM_018_THU24 (≈ THU_018 in paper text)
- 16 = RM_020_DIRG (DIRG)

## Leakage policy

All CDDG arms use the maintained `base_cross_system.yaml` data block with
**leakage-safe grouped splits** (`data.split.strategy: grouped_metadata`,
`test_policy: task_defined`). The group key is `Domain_id`:

- `metadata.xlsx` **has no `Bearing_id` column** (despite `configs/README.md`
  documenting it as the canonical key — that is a known repo gap, also recorded
  by P06). `Domain_id` is the populated, per-operating-condition column that
  keeps each domain whole across the train/val boundary.
- `File` is per-row unique (1 group per row) and therefore degenerates to a
  random row split — it is **not** a leakage-safe group and is not used here.
- `stratify_key` is intentionally omitted: `Domain_id` groups carry multiple
  `Label` values, so single-label stratification is undefined
  (`src/data_factory/splitting.py:_group_labels` raises if set).
- `fractions` is `{train: 0.9, val: 0.1}` — `test_policy: task_defined`
  requires exactly `{train, val}` (the test set IS the held-out target
  domain); including a `test` fraction raises in `_validate_spec`.

If the installed engine version rejects `data.split.*`, record the run as T2
and re-run once the policy is wired.

## Engine compatibility notes (validated by smoke, 2026-07-18)

Each arm was smoke-validated on real metadata (CWRU→XJTU hold-out) in the
`LQ_signal` env (pytorch-lightning + torch + CUDA). Results:

| Arm | Model factory | Smoke `test_acc` | Notes |
|---|---|---|---|
| E1 | X_model / TSPN_UXFD | 0.950 | operators `[I,HT]`, `out_channels: 6` (must divide module_num=2; FFT excluded — returns complex `L//2+1`) |
| E2 | ISFM / M_01_ISFM (E_01_HSE/B_04_Dlinear/H_01_Linear_cla) | 0.950 | finite loss |
| E3 | CNN / ResNet1D | 0.898 | — |
| E4 | Transformer / ConvTransformer | 0.947 | — |
| E5 | X_model / BASE_ExplainableCNN | 0.912 | uses `in_channels` |
| E6 | X_model / MWA_CNN | n/a (GPU-only) | reads `in_channels` (not `input_dim`); DWT layers hardcode `.cuda()` → accepted-run GPU-only |
| E1-DG | X_model / TSPN_UXFD (CWRU DG) | runs end-to-end | same operator/out_channels rules as E1 |

The 1-epoch CPU smokes can produce `test_loss=nan` for the softmax-weighted
TSPN arms, which trips the post-hoc `write_run_summary` finite-metric guard;
this is a smoke-only artifact and does not affect the accepted GPU run.

## Smoke vs accepted

The committed `trainer.num_epochs: 1` is a **smoke** default. The accepted
protocol is launched by override:

```bash
python main.py --config configs/experiments/p03/e1_tspn_uxfd_cddg.yaml \
  --override trainer.num_epochs=100 \
  --override environment.seed=42
```

Never edit these files to bake in a long schedule; use `--override`.
