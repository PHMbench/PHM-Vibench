# Conditional GAN, VAE and DDPM Factory candidates

At `dev@f105bdb`, Pipeline 06 had an implemented CFM Task but no integrated GAN,
VAE or DDPM Task. Historical branches already contained DDPM training and sampling
code. This change adds the currently missing conditional GAN/VAE candidates,
adapts the historical DDPM capability to the current runtime, and reuses CFM
through the existing Model imports, Task discovery and Default_dataset mappings.

The 2026-11-20 history review corrected the initial assumption that absence on
current `dev` meant absence from the repository. `Feature_factory-update@ed1bc7d`
and `cleanup/repo-slim-2026-07-05@7b97deb` contain `ddpm_epsilon`, epsilon loss,
scheduler, sampler and YAMLs. The old sparse-time sampler used adjacent-step
updates even when sampling fewer steps than the training schedule; the current
Task requires the full schedule. Its condition interface also follows current
CFM. The existing migration contract and source links are recorded in the
generative guide. No second DDPM path is registered. GAN/VAE were not found in
the checked major snapshots; this is not a repository-wide historical absence
claim.

The merge review also reproduced a DDPM restoration defect: a checkpoint with
the same number of steps but different beta endpoints silently replaced the
configured schedule. The Task now compares its fixed schedule buffers before
loading weights and rejects a mismatch. Three focused checkpoint regressions
cover changed beta endpoints, unchanged destination state on rejection, and
successful restoration with the original schedule. Cross-configuration GAN/VAE
checkpoint reuse remains outside the verified path; `strict` key/shape matching
alone does not validate window or observation-likelihood semantics.

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
- Merge-review follow-up: the focused DDPM contract file passed all 9 tests;
  after the final buffer-type guard, the 3 affected checkpoint tests passed.
  Documentation validation still passes (96 files). No neural training was
  repeated for the follow-up.

The legacy `paper_smoke_metric_eligible` field means all smoke metrics were
computable. Its value becomes true when both classes are sampled, while
`paper_smoke_ready` and `benchmark_valid` remain false. Runtime eligibility does
not promote a PHM result.

No GPU or real-PHM comparison was run. The eight-step DDPM fixture does not reach
a Gaussian forward endpoint, GAN validation BCE is not distribution quality, and
the existing evaluator still uses training-reference smoke metrics. Method
quality, physical fidelity and independent real-test augmentation remain untested.
