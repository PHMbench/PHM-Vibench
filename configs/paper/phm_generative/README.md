# PHM Generative Paper Configs

These configs are paper-matrix entries, not lightweight demos. They keep the
same five-block contract and `python main.py --config <yaml>` entrypoint while
separating train, sample, eval, seed, condition-policy, and ablation variants.

The dummy paths used for checkpoints and generated samples are explicit
placeholders for paper runs and are validated only as configuration contracts.

Benchmark-effect evaluation is driven by `benchmark_effect_matrix.yaml`; see
[`docs/phm_generative/BENCHMARK_EFFECT_EVALUATION.md`](../../../docs/phm_generative/BENCHMARK_EFFECT_EVALUATION.md).
