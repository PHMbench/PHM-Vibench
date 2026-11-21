# Complete declared stages and distinguish training from evaluation

Pipeline 02 now executes every configured stage through its existing orchestrator,
instead of truncating the request to the two-stage compatibility view. Evaluation
stages still require one complete nonempty finite test metric mapping. An explicitly
training-only stage (`trainer.test_after_fit: false`) fits, restores its selected
checkpoint, and returns `trained_not_evaluated` without test metrics.

The two existing fit implementations share their actual lifecycle; no new runner,
state manager or result system is added. Checkpoint handoff cannot silently reuse an
older state after a missing new checkpoint. Stage names and completion order are
checked before success. Each fit has a separate output directory, preventing rapid
stages from overwriting the same second-resolution checkpoint location. The old
public two-stage seed behavior is retained: stage index is not a seed offset.

Verification on Python 3.13.5 / PyTorch 2.10.0+cpu / Lightning 2.6.5:

- Focused Pipeline 02, metric contract, classification runtime and run-summary tests:
  **109 passed**.
- Actual bundled-Dummy software execution: three one-epoch fits, first stage train-only,
  two evaluations, three separate output directories and explicit checkpoint handoff.
- Each selected checkpoint has its own two-step optimizer rather than resumed step
  counts. Independent NumPy computation from selected weights agrees with complete
  test-population Accuracy and cross-entropy for both evaluated stages (8/8 windows).
  Float32-versus-float64 loss tolerance was `rtol=2e-6, atol=2e-7`.
- The initial independent-check script expected 32 instead of the configured eight test
  windows. Corrected that check and recomputed retained outputs without retraining;
  the original failed check is retained in the validation artifact.

These are software/lifecycle checks, not real-bearing performance or new scientific
results. The bundled fixture's original split remains unchanged and is not promoted.
No MFPT changes, optional-LLM relocation, dependency migration, full-experiment resume,
asset cleanup, or release-policy change is included.
