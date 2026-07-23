# Documentation Archive

This directory stores historical documentation that remains useful for provenance,
release reconstruction, design review, or research context but is not current user
guidance.

Archived documents may contain obsolete commands, paths, component names, status
claims, or environment assumptions. Do not follow them unless a maintained page
explicitly links to a specific archived procedure.

## Archive rules

Archive a document instead of deleting it when it records at least one of:

- a released migration or compatibility decision;
- an architecture decision that still explains current constraints;
- an experiment protocol or failure with reproducibility value;
- a branch, release, or governance audit needed to reconstruct history;
- an externally cited path that should remain recoverable.

Delete a document only when it is wholly duplicated, empty, unrelated to the
repository, or a template with no historical or operational value, and only after
checking references and preserving required provenance.

## Current historical locations

Public repository history is organized as follows:

- `docs/archive/` — maintained public audits, migration records, and historical
  evidence that still supports repository governance;
- `configs/v0.0.9/` — retained compatibility configurations, not the maintained
  quickstart surface.

The former `docs/v0.1.0/`, `docs/past/`, `dev/`, and root `.archive/` workspaces are
preserved in immutable Git history and in the approved personal-fork archive. They
are not public framework inputs and are not current guidance.

`configs/v0.0.9/` must remain while protected compatibility code references its
presets. New historical material belongs under `docs/archive/`, not in new versioned
or `past/` directory trees.

## Maintained documentation

Return to the [documentation index](../index.md) for current installation, usage,
configuration, development, support, and release guidance.
