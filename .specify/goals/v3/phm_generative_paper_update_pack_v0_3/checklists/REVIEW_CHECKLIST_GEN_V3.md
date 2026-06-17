# PHM-GenBench v0.3 Review Checklist

## Entry path

- [ ] `python main.py --config <yaml>` still works.
- [ ] `--preflight-only` does not start training.
- [ ] unsupported pipeline fails fast.
- [ ] malformed YAML fails fast.

## Pipeline

- [ ] train/sample/eval remain separate.
- [ ] no paperpack logic inside pipeline.
- [ ] sample requires checkpoint unless untrained smoke is explicitly enabled.
- [ ] untrained smoke forces exploratory.

## Manifest

- [ ] synthetic manifest records config/protocol/dependency hashes.
- [ ] normalization params are recorded.
- [ ] forbidden source splits are rejected.
- [ ] condition counts are recorded.
- [ ] leakage checks are recorded.
- [ ] missing evidence prevents benchmark-valid.

## Metrics

- [ ] every metric has status/reason.
- [ ] missing labels produce not_computable, not silent drop.
- [ ] nearest-centroid utility is named as a probe.
- [ ] leakage threshold is documented.

## Paperpack

- [ ] paperpack can locate sample manifest.
- [ ] manifest_completeness.csv is meaningful.
- [ ] missing metrics appear in appendix.
- [ ] paper draft refuses submission-ready when gaps exist.

## Research-only methods

- [ ] MeanFlow/Drifting/TFM/OT-NFM require experimental=true.
- [ ] They require num_steps=1.
- [ ] They cannot be benchmark-valid.
- [ ] The paper does not claim faithful implementation before promotion.
