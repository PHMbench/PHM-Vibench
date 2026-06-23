# Data Model: UXFD Paper Alignment

## UXFD Submodule Contract

Represents one paper submodule's parent-facing reproduction contract.

**Fields**:

- `submodule_path`: path under `paper/UXFD_paper/`.
- `vibench_path`: `VIBENCH.md` path.
- `min_config_path`: `configs/vibench/min.yaml` path.
- `maintained_command`: root CLI command or paper-local command.
- `expected_artifacts`: runtime and paper-specific artifacts.
- `status`: smoke-passed, blocked, skipped, paper-local-only, or unverified.
- `blocker_reason`: required unless status is smoke-passed.

**Validation rules**:

- All seven indexed submodules must have one contract status.
- Missing `VIBENCH.md` or minimal config is a blocker.
- Paper-local-only commands must not be counted as root CLI evidence.

## Minimal Evidence Gate

Represents an attempted UXFD minimal run or explicit skip.

**Fields**:

- `submodule_path`: owning submodule.
- `command`: exact command.
- `result`: pass, fail, skipped, or blocked.
- `artifact_paths`: produced artifacts when available.
- `log_path`: command log when captured.
- `reason`: required for fail, skipped, or blocked.

**Validation rules**:

- Passing root CLI gates must satisfy Slice 1 artifact expectations.
- Skipped gates must state the missing prerequisite and impact on paper claims.

## LaTeX Entry Point

Represents a TeX file selected for claim alignment or compilation.

**Fields**:

- `submodule_path`: owning submodule.
- `tex_path`: selected TeX path.
- `entrypoint_status`: selected, missing, non-final, or blocked.
- `bibliography_paths`: bibliography files used by the entrypoint.
- `compile_command`: selected compile command when available.

**Validation rules**:

- Entry points must be discovered from actual files.
- Missing final entrypoints must be blockers, not invented files.

## Claim Evidence Link

Represents one figure, table, metric, baseline, or text claim.

**Fields**:

- `claim_id`: stable local identifier.
- `submodule_path`: owning submodule.
- `tex_path`: TeX source containing the claim.
- `claim_type`: figure, table, metric, baseline, or text.
- `artifact_path`: generated artifact or external source when available.
- `status`: verified, blocked, unresolved, or external-source.
- `reason`: required when not verified.

**Validation rules**:

- Verified claims require an artifact path or documented source.
- Blocked Slice 2/3 task/model/baseline evidence propagates to paper claims.

## Compile Gate

Represents a compile attempt for a selected LaTeX entrypoint.

**Fields**:

- `tex_path`: selected entrypoint.
- `command`: exact compile command.
- `result`: pass, fail, skipped, or blocked.
- `pdf_path`: produced PDF path when available.
- `log_path`: compile log path.
- `first_error`: first actionable error when failed.

**Validation rules**:

- Passing compile status requires a PDF and no fatal compile error.
- Missing toolchain is a skipped or blocked gate with impact.

## Submodule Pointer State

Represents parent/submodule git safety state.

**Fields**:

- `submodule_path`: submodule path.
- `submodule_status`: clean, dirty, committed, or unknown.
- `parent_gitlink_status`: unchanged, changed-intentional, changed-unintended, or unknown.
- `commit_sha`: submodule commit when relevant.
- `reason`: required for changed or dirty states.

**Validation rules**:

- Parent gitlink changes are intentional only when backed by a submodule-local
  commit and recorded reason.
