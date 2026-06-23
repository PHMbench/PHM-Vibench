# Research: PHM 2025+ Literature Integration

## Decision: Use A Validated Offline Inventory

Create a CSV inventory and a local Python validation/report module rather than
adding 50 model stubs.

**Rationale**: The constitution requires fail-fast behavior and prohibits making
unverified capabilities appear supported. A structured inventory lets PHM-Vibench
track current research while preserving the existing factory/registry contract.

**Alternatives considered**:

- Add model registry rows for every paper: rejected because most methods are not
  implemented and would become false support claims.
- Keep only prose references: rejected because the user asked to run modules and
  because prose is not enough for validation.

## Decision: Interpret "25年之后" As Year >= 2025

The inventory accepts papers with publication year 2025 or 2026.

**Rationale**: The current date is 2026-05-11 and the user asked for latest work
after "25"; including 2025 and later is the most useful and verifiable boundary.

**Alternatives considered**:

- Strictly year > 2025: rejected because it would discard much of the latest 2025
  PHM literature and conflict with the user's "25年之后" shorthand.

## Decision: Map Papers To Existing PHM-Vibench Surfaces

Each paper receives task family, method family, repository surface, and support
status labels.

**Rationale**: This connects literature to the repository system without runtime
overclaiming. Current supported surfaces remain derived from existing registries
and tests.

**Alternatives considered**:

- Use a free-form notes field only: rejected because it cannot support tests or
  future filtering.

## Decision: Keep Network Search Outside Runtime Validation

The web search is performed during curation. The committed inventory validates
offline.

**Rationale**: CI/local validation should not depend on search engine results or
publisher availability.

**Alternatives considered**:

- Live validation of URLs: rejected because it introduces network flakiness and
  would fail in the repository's restricted/offline environments.
