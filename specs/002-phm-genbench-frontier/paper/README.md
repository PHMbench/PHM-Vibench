# PHM-GenBench Paper Artifacts

This directory stores feature-scoped paper drafts, evidence gaps, and
submission-readiness notes for the six-dataset benchmark queue.

Generate the Markdown draft only from completed benchmark evidence:

```bash
python -m scripts.generative_submission_draft \
  --summary results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_summary.csv \
  --manifest results/paper/phm_generative/six_dataset_submission_v1/effect/benchmark_effect_manifest.json \
  --output specs/002-phm-genbench-frontier/paper/PAPER_DRAFT.md \
  --require-submission-ready
```

The generator writes `SUBMISSION_READY` only when the evidence covers at least
six benchmark-valid datasets with computable quality and utility rows, no
manifest dataset coverage gaps, and traceable metric/manifest source paths.
Otherwise it writes `NOT_SUBMISSION_READY` and reports the evidence gaps. It
also writes `evidence_gaps.md` and `submission_readiness.md` sidecars next to
the draft output.

Working paper artifacts belong here. Module-specific runtime guidance belongs
in the corresponding module README next to the code.
