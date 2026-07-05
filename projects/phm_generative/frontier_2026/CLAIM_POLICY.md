# Claim Policy

## Allowed

- “Implemented an exploratory PHM adaptation.”
- “Completed dummy/toy train/sample/eval smoke validation.”
- “The method is a minimal pilot, not a faithful reproduction.”
- “Promotion requires method-specific evidence.”

## Forbidden before promotion

- “State-of-the-art on PHM.”
- “Benchmark-valid frontier baseline.”
- “Faithful MeanFlow/Drifting/Mamba reproduction” when method-specific implementation is absent.
- “Improves downstream diagnosis” based only on nearest-centroid probes.

## Promotion minimum

- method card + loss card
- method-specific objective
- unit tests
- dummy E2E run
- condition budget evidence
- sample manifest
- eval evidence manifest
- stage ledger
- statistical adequacy
- reviewer PASS
