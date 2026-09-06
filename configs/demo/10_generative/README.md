# Conditional Flow Matching candidate smoke

`dummy_generative_cfm.yaml` is a CPU-only candidate for the maintained
Pipeline 06 train → sample → eval smoke. It uses the repository Dummy_Data,
the canonical `Pipeline_06_Generative_Modeling` entrypoint, CFM velocity loss,
Euler sampling, and direct `fault_label`/`domain_id` conditions.

The registry status remains `needs_smoke`. Historical branch results do not
prove that this candidate works on the locked current topology, and no output
from this smoke is paper or benchmark evidence.

Run the train stage with:

```bash
CUDA_VISIBLE_DEVICES="" python main.py \
  --config configs/demo/10_generative/dummy_generative_cfm.yaml \
  --override trainer.num_epochs=1 \
  --override trainer.device=cpu \
  --override data.num_workers=0
```

Sampling and evaluation are separate invocations. Supply the exact checkpoint,
normalization path/hash, generated sample, and synthetic manifest recorded in
`stage_ledger.json`. Promotion to `sanity_ok` requires the focused CPU E-chain
test to pass and the resulting artifacts to be hashed.

Known limits:

- functional smoke only; no performance or scientific-validity claim;
- no support claim for arbitrary datasets, models, samplers, or accelerators;
- GPU 2 is forbidden by the portfolio execution contract.
