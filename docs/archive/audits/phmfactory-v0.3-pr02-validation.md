# PHMFactory v0.3 PR-02 Validation Record

## Scope

This record covers the inventory-only PR that freezes runtime, reader, personal-path,
repository-boundary, and submodule evidence. It does not authorize deletion or
runtime changes.

## Immutable inputs

```text
repository:              PHMbench/PHM-Vibench
runtime baseline commit: a331769d4005018bc833534ecf4efeb5e8a5a78d
repository contract:     d044d2031165cd4186d1da462fb154f101d6d493
```

## Generator evidence

The repository-native generator completed the following checks:

```text
export immutable baseline snapshot                 PASS
compile baseline generator                          PASS
byte-compare all protected runtime files           PASS
generate reader/runtime/submodule inventories      PASS
run the generator twice                             PASS
compare generated SHA-256 sets                      PASS
commit only the bounded PR-02 artifact set          PASS
```

The initial generation exposed the repository's global `*.json` ignore rule. Rather
than retain a forced exception and a large pretty-printed JSON diff, the fingerprint
artifact was redesigned as a compact CSV and the complete deterministic generation
sequence was rerun successfully.

## Classification review

A second deterministic run corrected two audit classifications before review:

- `THU.py` is `unverified`, not `maintained`, because its legacy non-RM module name
  and nonstandard callable signature require implementation-aware review;
- `paper/LQ_vibench_fix` is `personal`, not a general paper dependency.

Post-correction assertions passed:

```text
THU.py       -> unverified
THU24.py     -> placeholder
LQ fix       -> personal / non-allowlisted
```

## Protected-runtime result

```text
protected Python files fingerprinted: 256
Python parse errors:                  0
protected runtime files changed:      0
```

The compact CSV records one row per protected file: full-file SHA-256, byte count,
top-level callable count, and a deterministic aggregate callable-AST SHA-256.
Reader implementations, factories, tasks, trainers, samplers, and Pipeline files are
unchanged.

## Occam compaction

The initially generated pretty-printed JSON was replaced before review by a compact
257-line CSV: one header plus 256 protected files. It preserves the required change
detection fields while avoiding thousands of low-value diff lines and avoids a forced
exception to the repository's current global `*.json` ignore rule.

Compaction checks:

```text
marker-based generator update                       PASS
compact generator compile                           PASS
compact artifact generation                         PASS
second compact generation                           PASS
compact artifact SHA-256 parity                     PASS
fingerprint CSV line count = 257                     PASS
verbose JSON removed                                PASS
temporary compaction workflow removed               PASS
```

## Repository-native quality gates

This human-authored evidence commit triggers the normal pull-request quality workflow
after the compacting bot-authored commit. The PR remains Draft until all of the
following report success on the current head:

```text
Docs and config contracts
Offline config-first smoke
Pipeline 06 shell contract
UXFD focused contract
```

## Rollback

A normal revert removes the inventories, allowlist, generator, and this evidence
record. No runtime or dataset state is modified by PR-02.
