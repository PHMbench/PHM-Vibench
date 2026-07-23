# PHMFactory v0.3 Runtime, Reader, and Repository Baseline

## Immutable basis

- Repository: `PHMbench/PHM-Vibench`
- Frozen main/runtime commit: `a331769d4005018bc833534ecf4efeb5e8a5a78d`
- Repository-contract commit: `d044d2031165cd4186d1da462fb154f101d6d493`
- v0.2.x tags visible to the generator: `none found`

This PR records evidence only. It does not modify a reader, runtime callable,
submodule, paper workspace, result, package name, Pipeline, config, or test.

## Protected runtime

- Protected Python files fingerprinted: **256**
- Python parse errors recorded: **0**
- Every protected file was byte-compared with the frozen runtime commit before
  the inventories were emitted.
- Callable fingerprints use `ast.dump(..., include_attributes=False)` and SHA-256.

Artifacts:

- `phmfactory-v0.3-protected-runtime-fingerprints.json`
- `phmfactory-v0.3-reader-inventory.csv`

## Reader classification

| Status | Count | Meaning |
| --- | ---: | --- |
| `maintained` | 20 | Top-level `read` callable and an active metadata/config reference, or the offline Dummy reader. |
| `compatibility` | 0 | Callable exists under a legacy/non-RM module name; retained without a new support claim. |
| `experimental` | 2 | Non-empty reader-area module without the standard top-level `read` callable. |
| `unverified` | 2 | Top-level `read` callable exists, but no active metadata/config reference was found. |
| `placeholder` | 1 | Empty or effectively non-executable placeholder. |

Classification is an audit result, not a promise to delete non-maintained
files. `THU.py`, `THU24.py`, and similar compatibility/placeholder files remain
protected until a separate implementation-aware decision.

## Submodule baseline

- Configured submodules: **10**
- Allowlisted baseline entries: **0**
- The frozen baseline does not contain the proposed `phm-data-factory` backend.
- The deny-by-default allowlist records that backend as the sole candidate and
  records every existing baseline entry as legacy/non-allowlisted.

Artifacts:

- `phmfactory-v0.3-submodule-baseline.csv`
- `.github/phmfactory-v0.3-submodules.allowlist.yml`

## Personal and ownership-boundary inventory

The scanner records path, line, and category without copying line contents into
the public report. This avoids turning an inventory into a second source of
personal configuration values.

| Category | Matches |
| --- | ---: |
| `linux_home` | 400 |
| `macos_home` | 8 |
| `personal_account` | 7 |
| `personal_environment` | 3 |
| `personal_github_ssh` | 1 |
| `personal_prefix` | 41 |
| `windows_home` | 0 |

Artifacts:

- `phmfactory-v0.3-personal-path-inventory.csv`
- `phmfactory-v0.3-boundary-inventory.csv`

## Interpretation and next actions

1. PR-03 may remove generated and Agent/personal-only paths only after private-fork
   preservation and reference checks.
2. Reader cleanup PRs must compare against the protected callable fingerprints.
3. Paper/result/submodule deletion requires destination, immutable source SHA,
   content/hash verification, and reviewer confirmation.
4. The proposed backend remains optional and must not make the core wheel, CLI,
   Dummy smoke, or CWRU quickstart depend on an initialized submodule.
5. This baseline does not authorize algorithm changes or broad formatting.

## Regeneration

The generated artifacts are deterministic for the frozen snapshot:

```bash
mkdir -p /tmp/phmfactory-v030-baseline
git archive a331769d4005018bc833534ecf4efeb5e8a5a78d | tar -x -C /tmp/phmfactory-v030-baseline
python tools/repo/v030_generate_baseline.py \
  --snapshot-root /tmp/phmfactory-v030-baseline \
  --output-root .
```
