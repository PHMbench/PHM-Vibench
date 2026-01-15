# Submodule Setup (UXFD_paper)

Initialize and update all UXFD paper submodules:

```bash
git submodule update --init --recursive
```

Update to the latest recorded commits:

```bash
git submodule update --recursive
```

If you need to update a submodule to a newer upstream commit:
1) `cd paper/UXFD_paper/<paper_repo>`
2) `git fetch && git checkout <branch> && git pull`
3) `cd -` and commit the updated gitlink in the main repo

