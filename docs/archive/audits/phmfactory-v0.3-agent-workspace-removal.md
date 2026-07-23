# PHMFactory v0.3 Agent workspace removal audit

## Decision

The public PHMFactory upstream does not maintain vendor-specific or personal Agent
workspaces. This bounded change removes:

```text
.claude/**
.codex/**
AGENTS.md
AGENTS_CN.md
CLAUDE.md
CLAUDE_CN.md
GEMINI.md
Codex_agent.md
```

Neutral project guidance remains under `README.md`, `docs/`, `CONTRIBUTING.md`, and
the factory-level README and contribution documents.

## Archive-first preservation

All removed content was preserved before public deletion in the approved personal
fork:

```text
repository: liq22/PHM-Vibench
branch:     archive/phmfactory-v0.3.0-removals

upstream-archive/phmfactory-v0.3.0/agent-root/
upstream-archive/phmfactory-v0.3.0/agent-hidden/
```

The immutable preservation baseline is:

```text
a331769d4005018bc833534ecf4efeb5e8a5a78d
```

Verification evidence records:

```text
root Agent documents:          6/6 source/archive Git blobs match
.claude and .codex files:     65/65 source/archive Git blobs match
```

Root file identities:

| Removed path | Source and archive Git blob |
| --- | --- |
| `AGENTS.md` | `7f4773f09dd214de48438137a5c43f2c02cecf4b` |
| `AGENTS_CN.md` | `7a3b99c4b26a5600997e5992779289bd4d70290f` |
| `CLAUDE.md` | `1e1ed47829ec201aa04738a63055fceb5cc743bd` |
| `CLAUDE_CN.md` | `c9546501912881f0ea6a8784796c27a0561473d8` |
| `GEMINI.md` | `b17c5bb41f09610efbe1738a38e4b7aeb97bad42` |
| `Codex_agent.md` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |

The public deletion is based on canonical migration parent:

```text
6ab67111c9c1609f3cdd2339016e4cad237466ef
```

Public PHMFactory has no runtime, build, test, data, CI, or release dependency on the
private archive.

## Neutral link repair

The removed root `CLAUDE.md` was previously used as an architecture link from utils
documentation. Those links now point to:

```text
docs/developer_guide.md
```

No private-fork URL replaces a public documentation authority.

## Module Agent documents deliberately retained

Twenty-one module-level `CLAUDE.md` files remain temporarily. They are not treated as
safe to delete merely because their filenames are vendor-specific. Their disposition
is recorded in:

```text
docs/archive/audits/phmfactory-v0.3-module-agent-document-inventory.md
```

Each must follow one of these implementation-aware paths:

```text
merge-neutral-first
protected-review
```

Reader, model, task, trainer, config, and utility module knowledge will be compared
against actual code and canonical README files in separate module-scoped PRs. This
change does not rewrite or delete those protected documents.

## Regression prevention

`tools/repo/check_agent_boundaries.py` rejects:

- top-level `.claude/`, `.codex/`, `.agents/`, and `.gemini/` workspaces;
- root Agent documents listed by the public boundary contract.

It intentionally permits module-level `CLAUDE.md` files until their neutral migration
is individually reviewed. `.github/workflows/repository-layout.yml` runs the boundary
guard together with the case-collision guard and focused tests.

## Protected runtime boundary

No dataset reader, Data Factory behavior, model, task, trainer, Pipeline algorithm,
configuration behavior, CWRU data contract, dependency set, or Streamlit runtime is
changed.

## Rollback

A normal revert restores the public paths. Exact content remains available in the
private archive and immutable public Git history.
