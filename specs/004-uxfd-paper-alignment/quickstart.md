# Quickstart: UXFD Paper Alignment

Use these commands from the repository root while implementing Slice 4.

## Confirm Active Feature

```bash
cat .specify/feature.json
```

Expected feature directory:

```text
specs/004-uxfd-paper-alignment
```

## List UXFD Contracts

```bash
find paper/UXFD_paper -maxdepth 2 -name VIBENCH.md | sort
find paper/UXFD_paper -path '*/configs/vibench/min.yaml' | sort
```

## Inspect Submodule State

```bash
git submodule status --recursive
git status --short
```

## Run One Minimal UXFD Root Gate

```bash
python main.py --config paper/UXFD_paper/1D-2D_fusion_explainable/configs/vibench/min.yaml --override trainer.num_epochs=1
```

Run additional submodule minimal configs only after recording the command and
expected artifact contract.

## Discover LaTeX Entrypoints

```bash
find paper/UXFD_paper -path '*/manuscript/*.tex' -o -path '*/paper_draft/*.tex' -o -name main.tex | sort
```

## Check TeX Tooling

```bash
which latexmk
which xelatex
which pdflatex
```

Treat missing tools as skipped or blocked compile gates with impact recorded.

## Parent Validation

```bash
python -m scripts.validate_docs
python -m pytest -q test/test_collect_uxfd_runs.py
```

## Evidence Log

Recorded on 2026-05-11 from the repository root.

### Setup And Inventory

- `.specify/feature.json`: points to `specs/004-uxfd-paper-alignment`.
- `find paper/UXFD_paper -maxdepth 2 -name VIBENCH.md | sort`: found 7/7 UXFD contracts.
- `find paper/UXFD_paper -path '*/configs/vibench/min.yaml' | sort`: found 7/7 minimal configs.
- `git submodule status --recursive`: all seven UXFD gitlinks resolved to a commit.
- `git status --short`: parent worktree is dirty before/after this slice; UXFD submodule entries include dirty/untracked markers and `paper/UXFD_paper/results/`. This slice did not intentionally update submodule pointers.

### Parent Validation

- Initial `python -m scripts.validate_docs`: failed on two stale image links in `src/model_factory/X_model/legacy_collection/TFN/README.md`.
- Fixed by replacing the missing TFN README image links with text references.
- Final `python -m scripts.validate_docs`: `[OK] Documentation checks passed (127 files scanned).`
- `python -m pytest -q test/test_collect_uxfd_runs.py`: `1 passed`.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py`: `8 passed`.
- `python -m pytest -q test/test_uxfd_paper_alignment_contract.py test/test_collect_uxfd_runs.py`: `9 passed`.

### Contract Audit

Command:

```bash
python -m scripts.uxfd_paper_alignment
```

Result: exit code 0. All seven indexed UXFD submodules have `VIBENCH.md` and `configs/vibench/min.yaml`.

Contract status from the audit:

- `paper/UXFD_paper/1D-2D_fusion_explainable`: `unverified`; has the fullest Slice 1 artifact expectation coverage.
- `paper/UXFD_paper/Explainable_FD_Toolkit`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.
- `paper/UXFD_paper/LLM_Explainable_FD_Toolkit`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.
- `paper/UXFD_paper/MOE_explainable`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.
- `paper/UXFD_paper/Neuralsymbolic_theory`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.
- `paper/UXFD_paper/Paper_fuzzy_XFD`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.
- `paper/UXFD_paper/TII_operator_attention`: `unverified`; `VIBENCH.md` lacks full Slice 1 artifact expectations.

### UXFD Root CLI Gates

Command:

```bash
for config in paper/UXFD_paper/*/configs/vibench/min.yaml; do
  python main.py --config "$config" --override trainer.num_epochs=1
done
```

Result: all seven minimal configs completed with `trainer.num_epochs=1` in the `LQ_signal` conda environment.

Latest manifest evidence:

- `1D-2D_fusion_explainable`: `results/uxfd/pilot/1D-2D_fusion_explainable/metadata_dummy.csv/M_NSN/T_DGclassification_11_005637/iter_0/artifacts/manifest.json`
- `Explainable_FD_Toolkit`: `results/uxfd/pilot/Explainable_FD_Toolkit/metadata_dummy.csv/M_NSN/T_DGclassification_11_005647/iter_0/artifacts/manifest.json`
- `LLM_Explainable_FD_Toolkit`: `results/uxfd/pilot/LLM_Explainable_FD_Toolkit/metadata_dummy.csv/M_NSN/T_DGclassification_11_005656/iter_0/artifacts/manifest.json`
- `MOE_explainable`: `results/uxfd/pilot/MOE_explainable/metadata_dummy.csv/M_NSN/T_DGclassification_11_005706/iter_0/artifacts/manifest.json`
- `Neuralsymbolic_theory`: `results/uxfd/pilot/Neuralsymbolic_theory/metadata_dummy.csv/M_NSN/T_DGclassification_11_005715/iter_0/artifacts/manifest.json`
- `Paper_fuzzy_XFD`: `results/uxfd/pilot/Paper_fuzzy_XFD/metadata_dummy.csv/M_NSN/T_DGclassification_11_005723/iter_0/artifacts/manifest.json`
- `TII_operator_attention`: `results/uxfd/pilot/TII_operator_attention/metadata_dummy.csv/M_NSN/T_DGclassification_11_005732/iter_0/artifacts/manifest.json`

### LaTeX Entrypoints And Compile Gates

Toolchain:

- `/usr/bin/latexmk`
- `/usr/bin/xelatex`
- `/usr/bin/pdflatex`

Entrypoint discovery:

- Selected final entrypoints: `1D-2D_fusion_explainable`, `Explainable_FD_Toolkit`, `MOE_explainable`, `Neuralsymbolic_theory`, `Paper_fuzzy_XFD`.
- `LLM_Explainable_FD_Toolkit`: non-final `manuscript/tables/table_4_quality_metrics.tex` only.
- `TII_operator_attention`: missing `manuscript/final_tex/main.tex` and no alternate TeX entrypoint discovered.

Compile command shape used for selected final entrypoints:

```bash
cd <entrypoint-dir>
latexmk -xelatex -interaction=nonstopmode -halt-on-error -outdir=/tmp/uxfd_latex_xe/<submodule> main.tex
```

Compile results:

- `MOE_explainable`: pass; PDF `/tmp/uxfd_latex_xe/MOE_explainable/main.pdf`, log `/tmp/uxfd_latex_xe/MOE_explainable/main.log`.
- `1D-2D_fusion_explainable`: fail; log `/tmp/uxfd_latex_xe/1D-2D_fusion_explainable/main.log`; first actionable error `! Missing $ inserted.` at the path text containing underscores.
- `Explainable_FD_Toolkit`: fail; log `/tmp/uxfd_latex_xe/Explainable_FD_Toolkit/main.log`; first actionable error `! Unable to load picture or PDF file '../../figures/example.pdf'.`
- `Neuralsymbolic_theory`: fail; log `/tmp/uxfd_latex_xe/Neuralsymbolic_theory/main.log`; first actionable error `! Unable to load picture or PDF file '../../figures/example.pdf'.`
- `Paper_fuzzy_XFD`: fail; log `/tmp/uxfd_latex_xe/Paper_fuzzy_XFD/main.log`; first actionable error `! Unable to load picture or PDF file '../../figures/example.pdf'.`

No LaTeX source patches were made in UXFD submodules because the blockers are paper-local source issues and submodule edit ownership was not established for this slice.

### Claim Evidence Status

- Selected final entrypoints contain claim surfaces that require artifact-level audit before claim verification.
- `LLM_Explainable_FD_Toolkit` and `TII_operator_attention` remain blocked for final-paper claim alignment because they do not expose a selected final entrypoint.
- Slice 2 and Slice 3 evidence is available in parent docs, but UXFD paper claims were not rewritten without a proven stale claim and clear submodule ownership.
