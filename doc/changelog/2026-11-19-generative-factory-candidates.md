# Conditional GAN, VAE and DDPM Factory candidates

Pipeline 06 previously had an implemented CFM Task, while GAN, VAE and DDPM had
no executable Model/Task bindings. This change supplies ordinary conditional
versions through the existing Model import mechanism, Task discovery and explicit
Default_dataset adapter mappings.

GAN uses separate generator/discriminator optimizers with one update each per
batch. VAE uses a declared Gaussian likelihood and analytic KL; sampling includes
the same observation noise. DDPM predicts epsilon and uses its full configured
ancestral schedule with fixed posterior variance. The shared sampling result is
finite float32 `[N,C,L]` on the checked CPU path, with method identity and NFE in
the existing output metadata.

CFM's existing batch extraction, condition lookup and Lightning lifecycle move
unchanged into the shared GenerativeTask. Its loss and Euler computation remain
unchanged. No Reader, Trainer, Pipeline, CLI, schema or registry layer is added.

The four experiment YAMLs use copied Dummy inputs and caller-owned output paths.
The existing public-stage test now covers all four and the original CFM demo,
including strict checkpoint restoration and both observed condition pairs. The
implementation contract, equations, scope and BibTeX are recorded in the existing
`configs/experiments/generative/` guide.

## Validation

Local environment: Python 3.12.14, PyTorch 2.6.0+cpu, Lightning 2.6.6.

- Generative suite: 105 tests passed on the first run; four stage tests reached
  completed train/sample/eval but failed a single-class-only assertion about the
  legacy metric-computability flag. After correcting that fixture assumption, the
  four affected tests passed. All 109 collected cases are covered; the suite was
  not repeated wholesale locally.
- All four new/aligned experiment configurations pass `analyze_config()`.
- Maintained config validation: 13/13 pass.
- Documentation checks: pass, 96 files.
- Existing Agent boundary and case-collision checks: pass.
- No H5 data cache was added to the checkout; integration artifacts use temporary
  caller-owned directories.

The legacy `paper_smoke_metric_eligible` field means all smoke metrics were
computable. Its value becomes true when both classes are sampled, while
`paper_smoke_ready` and `benchmark_valid` remain false. Runtime eligibility does
not promote a PHM result.

No GPU or real-PHM comparison was run. The eight-step DDPM fixture does not reach
a Gaussian forward endpoint, GAN validation BCE is not distribution quality, and
the existing evaluator still uses training-reference smoke metrics. Method
quality, physical fidelity and independent real-test augmentation remain untested.
