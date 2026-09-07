# Version and Change Documentation

This directory keeps version-specific records out of the repository root while preserving
one clear location for each kind of document.

## Structure

```text
doc/
├── changelog/   # dated records of changes that were actually completed
├── migration/   # instructions for moving user configurations and commands between versions
└── release/     # version-specific release notes and claim boundaries
```

## Migration guides

- [v0.1.x to v0.2.0](migration/MIGRATION_v0.1_to_v0.2.md)
- [v0.2 to v0.3](migration/MIGRATION_v0.2_to_v0.3.md)

## Release notes

- [v0.2.0 release-candidate notes](release/RELEASE_NOTES_v0.2.0.md)
- [v0.3.0 release notes](release/RELEASE_NOTES_v0.3.0.md)

## Change records

Dated implementation and maintenance records live under [`changelog/`](changelog/).
The root [`CHANGELOG.md`](../CHANGELOG.md) remains the concise chronological entry point.

## Related current authorities

- [Current limitations](../KNOWN_LIMITATIONS.md)
- [Supported combinations](../SUPPORTED_COMBINATIONS.md)
- [v0.3 release readiness](../docs/PHMFACTORY_V0_3_RELEASE_READINESS.md)
- [Documentation index](../docs/index.md)

Historical audits under `docs/archive/` preserve the paths and statements that were true
at the time of each audit. They are not rewritten when maintained documents move.
