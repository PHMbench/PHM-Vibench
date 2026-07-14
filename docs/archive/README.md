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
checking references.

## Current historical locations

The repository already contains historical material outside this directory:

- `docs/v0.1.0/`
- `docs/past/`
- `configs/v0.0.9/`
- selected material under `dev/`

Those paths are preserved to avoid unnecessary link breakage. New historical audit
snapshots should be placed under `docs/archive/` rather than added to the maintained
documentation root.

## Maintained documentation

Return to the [documentation index](../index.md) for current installation, usage,
configuration, development, support, and release guidance.
