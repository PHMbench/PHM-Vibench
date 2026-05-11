# UXFD TOP Recent Work Citation And Reproduction README

This README is the accepted 2024-2026 related-work source of truth for the
seven UXFD papers. It intentionally uses TOP journals and computer-science
top-conference methods rather than low-tier bearing-fault-diagnosis papers.

## Venue Gate

Accepted core sources:

- Conferences: NeurIPS, ICML, ICLR, CVPR, ICCV, ECCV, KDD, AAAI, IJCAI, ACL,
  EMNLP, NAACL, SIGIR, and WWW.
- Journals: IEEE TPAMI, IEEE TNNLS, IEEE TKDE, IEEE TCYB, IEEE TFS, IEEE TII,
  IEEE TIE, IEEE TSP, Information Fusion, Mechanical Systems and Signal
  Processing, and Pattern Recognition.

Rejected for UXFD core related work, baselines, and SOTA positioning:

- Scientific Reports, publisher-level MDPI journals, IEEE Transactions on
  Instrumentation and Measurement, IEEE Access, Applied Sciences, Electronics,
  Sensors, Mathematics, and similar low-tier or application-only venues.
- These sources must not appear in accepted baseline tables or SOTA comparison
  text. If a lower-tier source is useful for background, record it outside the
  accepted method pool and do not use it to establish novelty.

## Reproduction Status Policy

- `exact-runnable`: the paper's own implementation or faithful reimplementation runs locally with command, config, log, and artifact path.
- `representative-runnable`: PHM-Vibench has a registered model family that represents the method class, but it is not an exact paper reproduction.
- `literature-only`: the paper is cited for positioning, motivation, or reviewer context and must not be counted as a reproduced baseline.
- `resource-blocked`: exact reproduction is blocked by the local 2x4090 budget even if the method is otherwise relevant and high quality.
- `blocked`: exact reproduction is blocked by missing code, license, dependency, data, or unclear protocol.

SOTA comparisons may count only `exact-runnable` baselines and clearly labelled
`representative-runnable` baselines under the same dataset split, seed protocol,
preprocessing, and metrics. Literature-only and blocked works may support
related-work text but not performance claims.

The local compute budget is fixed at `CUDA_VISIBLE_DEVICES=0,1` on two RTX
4090-class GPUs. Exact reproduction that needs larger models, more GPUs,
multi-node execution, or cloud-only hardware must be labelled
`resource-blocked`. A `resource-blocked` work may still have a
`representative-runnable` PHM-Vibench proxy, but the proxy must not be described
as exact reproduction.

## Accepted TOP Method Pool

| ID | Year | Venue tier | Work | Venue | UXFD relevance | Initial status | PHM-Vibench representative run |
|---|---:|---|---|---|---|---|---|
| RWTOP2024-TIMEXPP | 2024 | `top-conference` | Liu et al., [TimeX++: Learning Time-Series Explanations with Information Bottleneck](https://www.microsoft.com/en-us/research/publication/timex-learning-time-series-explanations-with-information-bottleneck/) | ICML 2024 | Time-series explanation baseline for Papers 1, 3, 5, 6. | `representative-runnable` | Toolkit explanation gate plus information-bottleneck attribution proxy. |
| RWTOP2024-TIMELLM | 2024 | `top-conference` | Jin et al., [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://research.ibm.com/publications/time-llm-time-series-forecasting-by-reprogramming-large-language-models) | ICLR 2024 | LLM/time-series adaptation baseline for Paper 3. | `representative-runnable` | Evidence-grounded prompt/report generation with frozen LLM or local proxy. |
| RWTOP2024-TIMEMIXER | 2024 | `top-conference` | Wang et al., [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://openreview.net/forum?id=7oLshfEIC2) | ICLR 2024 | Multiscale temporal modeling baseline for Papers 2 and 7. | `representative-runnable` | `CNN.TCN`, multiscale CNN, or transformer representative under the same FD split. |
| RWTOP2024-MOMENT | 2024 | `top-conference` | Goswami et al., [MOMENT: A Family of Open Time-series Foundation Models](https://openreview.net/forum?id=FVvf69a5rx) | ICML 2024 | Foundation-model representation baseline for Papers 1, 2, 3, 7. | `representative-runnable` | `Transformer.PatchTST`/foundation-style frozen encoder proxy until exact code is integrated. |
| RWTOP2024-SARAD | 2024 | `top-conference` | Dai et al., [SARAD: Spatial Association-Aware Anomaly Detection and Diagnosis for Multivariate Time Series](https://proceedings.neurips.cc/paper_files/paper/2024/hash/56ad264ac7448239145606cf4106042f-Abstract-Conference.html) | NeurIPS 2024 | Multivariate diagnosis/anomaly baseline for Papers 2, 6, 7. | `representative-runnable` | Graph/channel association proxy with `X_model.NSN` or channel-attention transformer. |
| RWTOP2025-TIMEMOE | 2025 | `top-conference` | Shi et al., [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://proceedings.iclr.cc/paper_files/paper/2025/hash/558d48c1f08675daa636e09bfe94a89e-Abstract-Conference.html) | ICLR 2025 Spotlight | Sparse MoE/foundation baseline for Paper 4 and Paper 3. | `representative-runnable` | MoE routing proxy with route entropy and expert activation artifacts. |
| RWTOP2025-MOIRAIMOE | 2025 | `top-conference` | Liu et al., [Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts](https://icml.cc/virtual/2025/poster/45201) | ICML 2025 | Token-level sparse expert specialization baseline for Paper 4. | `representative-runnable` | MoE routing proxy with expert-count and sparsity ablations. |
| RWTOP2025-CATCH | 2025 | `top-conference` | Wu et al., [CATCH: Channel-Aware Multivariate Time Series Anomaly Detection via Frequency Patching](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2b25c39788e5cf11d3541de433ebf4c0-Abstract-Conference.html) | ICLR 2025 | Channel/frequency modeling baseline for Papers 2 and 7. | `representative-runnable` | `X_model.TFN`, `X_model.WKN`, and frequency/channel-attention representative runs. |
| RWTOP2025-DADA | 2025 | `top-conference` | Shentu et al., [Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ca7998666c2e53cc1e882b7268414d8a-Abstract-Conference.html) | ICLR 2025 | General anomaly detector and bottleneck baseline for Papers 1, 2, 7. | `representative-runnable` | Autoencoder/bottleneck representative run under the same FD protocol. |
| RWTOP2025-CFCBM | 2025 | `top-conference` | Dominici et al., [Counterfactual Concept Bottleneck Models](https://research.ibm.com/publications/counterfactual-concept-bottleneck-models) | ICLR 2025 | Concept/counterfactual explanation baseline for Papers 1, 5, 6. | `literature-only` until concept labels are defined for FD datasets. | blocked; do not count as reproduced baseline. |
| RWTOP2025-CBAE | 2025 | `top-conference` | Kulkarni et al., [Interpretable Generative Models through Post-hoc Concept Bottlenecks](https://cvpr.thecvf.com/virtual/2025/poster/32807) | CVPR 2025 | Post-hoc concept bottleneck baseline for Papers 1, 3, 5. | `literature-only` until image/concept protocol is adapted. | blocked; do not count as reproduced baseline. |
| RWTOP2025-IFCBM | 2025 | `top-journal` | Interpretable prognostics with concept bottleneck models | Information Fusion 2025 | Concept-bottleneck prognostics/explainability comparator for Papers 1, 5, 6. | `literature-only` until dataset/task mapping is defined. | blocked; do not count as reproduced baseline. |

## Per-Paper TOP-Source Minimums

| Paper | Required TOP recent methods before submission | Runnable minimum |
|---|---|---|
| 1 Toolkit | RWTOP2024-TIMEXPP, RWTOP2024-MOMENT, RWTOP2025-DADA, RWTOP2025-CFCBM | At least one Toolkit explanation representative run. |
| 2 1D-2D Fusion | RWTOP2024-TIMEMIXER, RWTOP2024-MOMENT, RWTOP2025-CATCH, RWTOP2025-DADA | At least one multiscale/frequency representative run. |
| 3 LLM Toolkit | RWTOP2024-TIMELLM, RWTOP2024-MOMENT, RWTOP2025-TIMEMOE, RWTOP2025-CBAE | At least one evidence-grounded LLM or local proxy run. |
| 4 MoE | RWTOP2025-TIMEMOE, RWTOP2025-MOIRAIMOE, RWTOP2024-MOMENT | At least one sparse-router representative run with route artifacts. |
| 5 Fuzzy-XFD | RWTOP2024-TIMEXPP, RWTOP2025-CFCBM, RWTOP2025-CBAE, RWTOP2025-IFCBM | At least one concept/rule explanation representative run. |
| 6 Neuralsymbolic | RWTOP2024-TIMEXPP, RWTOP2024-SARAD, RWTOP2025-CFCBM, RWTOP2025-IFCBM | At least one concept/constraint representative run. |
| 7 Operator Attention | RWTOP2024-TIMEMIXER, RWTOP2024-SARAD, RWTOP2025-CATCH, RWTOP2025-DADA | At least one frequency/channel/operator representative run. |

## 2x4090 Exact-Reproduction Feasibility

| ID | Exact reproduction under `CUDA_VISIBLE_DEVICES=0,1` | Local policy |
|---|---|---|
| RWTOP2024-TIMEXPP | feasible only after exact code/config is integrated | use `representative-runnable` until exact artifacts exist. |
| RWTOP2024-TIMELLM | `resource-blocked` for large LLM variants beyond 2x4090 | use local LLM/proxy evidence-chain evaluation. |
| RWTOP2024-TIMEMIXER | feasible only if exact implementation fits one RTX 4090 | use multiscale representative runs until exact artifacts exist. |
| RWTOP2024-MOMENT | `resource-blocked` for large foundation-model settings beyond 2x4090 | use frozen/compact foundation-style proxy. |
| RWTOP2024-SARAD | feasible only after exact code/config is integrated | use association/channel representative runs until exact artifacts exist. |
| RWTOP2025-TIMEMOE | `resource-blocked` for billion-scale exact reproduction | use local sparse-MoE proxy and route artifacts. |
| RWTOP2025-MOIRAIMOE | `resource-blocked` for large foundation-model exact reproduction | use local sparse-MoE proxy and expert-count ablations. |
| RWTOP2025-CATCH | feasible only if exact frequency-patching run fits one RTX 4090 | use frequency/channel representative runs until exact artifacts exist. |
| RWTOP2025-DADA | feasible only if exact pretraining/evaluation fits 2x4090 | use bottleneck/anomaly representative runs until exact artifacts exist. |
| RWTOP2025-CFCBM | `resource-blocked` until FD concepts and compute-feasible protocol exist | literature-only; do not count as reproduced baseline. |
| RWTOP2025-CBAE | `resource-blocked` until concept/image protocol fits 2x4090 | literature-only; do not count as reproduced baseline. |
| RWTOP2025-IFCBM | `resource-blocked` until task mapping and local concept protocol exist | literature-only; do not count as reproduced baseline. |

## Initial Local Reproduction Commands

These commands validate repository-side runnable representatives. They do not
claim exact reproduction of the external TOP papers above.

```bash
python -m scripts.baseline_mapping
python -m pytest -q test/test_baseline_mapping_contract.py
python -m pytest -q test/test_model_registry_contract.py test/test_x_model_smoke.py
```

For paper-specific baselines, the next milestone must bind each selected TOP
work to an exact command/log/artifact inside the owning submodule before it is
counted in a SOTA comparison.
