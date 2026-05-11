# UXFD Codex XHigh Subagent Risk Register

Date: 2026-05-11

## Critical Risks

- Accepted evidence root is missing: no reviewer-grade `run_meta.yaml`,
  `metrics.json`, logs, config snapshots, or GPU metadata exist under
  `paper/UXFD_paper/results/accepted_runs`.
- GPU preflight fails in the current session; all final baseline, ablation,
  TOP representative, and SOTA claims remain blocked.
- All seven papers still have `submission_ready: false`.
- TOP representative evidence is pending for every paper.
- Several manuscripts still overclaim relative to the strict gate, especially
  Paper02 and Paper07 performance/GPU/SOTA language.

## Paper-Specific Risks

- Paper01: existing `notes.accepted: true` schema packs may be mistaken for
  reviewer-grade accepted evidence; they lack strict 2x4090 metadata.
- Paper02: `NatureMi.cls` and placeholder/noncanonical TeX can cause immediate
  desk rejection; true Fusion1D2D ablations are absent.
- Paper03: no accepted LLM evidence package exists; template demos and smoke
  runners are explicitly non-accepted.
- Paper04: MoE route claims can be rejected without real route entropy,
  expert activation, and same-protocol ablation evidence.
- Paper05: fuzzy explanation claims need rule metrics, membership values,
  safety cases, and reviewer ablations beyond dummy sensitivities.
- Paper06: P2 currently fails and must be treated as boundary evidence unless
  real-data validation proves otherwise.
- Paper07: synthetic operator-selection success does not establish industrial
  SOTA or robustness.

## Process Risks

- External Claude Team launch remains policy-blocked; local Codex subagent
  evidence is a replacement review artifact, not external validation.
- Dirty submodules mean current evidence is working-tree state, not clean
  accepted paper-local milestones.
- Low-tier exclusion is enforced for the accepted TOP pool, but final manuscript
  citation hygiene still requires broader checks.
