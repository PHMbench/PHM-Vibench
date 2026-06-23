# Research: UXFD Paper Alignment

## Decision: Treat VIBENCH.md and min.yaml as the parent-facing contract

**Rationale**: `paper/UXFD_paper/README.md` states that each UXFD submodule is
expected to provide `VIBENCH.md` and `configs/vibench/min.yaml`, and warns that
older READMEs may contain historical paths or old CLI flags. These files are the
stable parent-facing reproduction contract.

**Alternatives considered**:

- Use every submodule README as the source of truth. Rejected because several
  README files may still contain historical paths or roadmap material.
- Copy paper mapping docs into the parent `docs/` directory. Rejected because the
  parent guide explicitly keeps mapping docs inside submodules.

## Decision: Discover LaTeX entrypoints from actual files

**Rationale**: The UXFD family contains different paper structures. Current discovery
shows several `manuscript/final_tex/main.tex` files, one TII sample TeX entry, and
no obvious final main entrypoint for every submodule. The plan must record real
entrypoints and blockers rather than assuming one uniform `main.tex`.

**Alternatives considered**:

- Require every submodule to have `main.tex`. Rejected because it would create false
  blockers and encourage invented files.

## Decision: Evidence links must allow blocker status

**Rationale**: Some submodules may depend on paper-local scripts, private data,
optional dependencies, or missing artifacts. A blocker is valid evidence of current
state; claiming verification without artifacts is not.

**Alternatives considered**:

- Skip blocked claims silently. Rejected because unsupported paper claims must be
  visible before submission.

## Decision: Compile gates are selected during implementation

**Rationale**: TeX commands depend on the actual entrypoint, bibliography layout, and
available toolchain. The plan should require recording commands/logs rather than
hard-coding one compile command for all submodules.

**Alternatives considered**:

- Use one global compile command for every paper. Rejected because the submodule
  layouts are not uniform.

## Decision: Respect submodule ownership boundaries

**Rationale**: Parent docs state that paper-specific edits must be committed inside
the submodule first, and the parent should only record the gitlink pointer when that
pointer change is intentional.

**Alternatives considered**:

- Edit submodule files and leave parent dirty. Rejected because it breaks review and
  reproducibility of paper changes.
