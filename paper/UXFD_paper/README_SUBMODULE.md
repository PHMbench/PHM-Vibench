# Submodule Setup (UXFD_paper)

This is the UXFD-local submodule quick guide. The parent-level guide is
`paper/README_SUBMODULE.md`.

## Initialize

Initialize all paper submodules:

```bash
git submodule update --init --recursive paper/UXFD_paper
```

Initialize one UXFD paper submodule:

```bash
git submodule update --init --recursive paper/UXFD_paper/<paper_repo>
```

## Sync To Recorded Commits

Update submodules to the commits recorded by the parent repo:

```bash
git submodule update --recursive paper/UXFD_paper
```

## Move A Submodule Forward

If you need a newer upstream commit:

```bash
cd paper/UXFD_paper/<paper_repo>
git fetch
git checkout <branch>
git pull --ff-only
cd -
git status --short
git add paper/UXFD_paper/<paper_repo>
git commit -m "Update <paper_repo> submodule"
```

## Rules

- Commit content changes inside the target submodule before committing the
  parent gitlink update.
- Keep `VIBENCH.md` and `configs/vibench/min.yaml` inside each submodule.
- Do not add paper-specific mapping docs to the parent `docs/` directory.
