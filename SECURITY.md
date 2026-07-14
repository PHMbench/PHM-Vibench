# Security Policy

## Supported code

PHM-Vibench is an alpha research-software project. Security fixes are considered
on a best-effort basis for:

| Code line | Security support |
| --- | --- |
| Current `main` | Active development; reports are accepted |
| Latest tagged release, when one exists | Reports are accepted; fixes may require upgrading |
| Older tags, historical configs, archived branches, and research snapshots | Not normally patched |

A registry entry, paper directory, historical branch, or experimental application
does not create a long-term security-support commitment.

## Report a vulnerability privately

Do **not** disclose a suspected vulnerability in a public issue, pull request,
discussion, log, or dataset attachment.

Use GitHub's private vulnerability reporting interface for this repository when
the **Report a vulnerability** option is available on the Security page.

If private vulnerability reporting is unavailable, contact a repository maintainer
through their GitHub profile and request a private reporting channel before
sharing exploit details, credentials, private paths, or affected data. A public
issue may ask for a private contact method only when it contains no sensitive
technical information.

## Include in the report

Provide as much of the following as is safe:

- affected commit, branch, tag, or release;
- affected file, component, workflow, or optional application;
- vulnerability class and potential impact;
- prerequisites and attack surface;
- minimal reproduction or proof of concept;
- operating system, Python and dependency versions;
- whether credentials, private data, model artifacts, or external services are
  involved;
- suggested mitigation, when known;
- disclosure timeline constraints.

Remove real secrets and personal data from examples. Use synthetic values whenever
possible.

## Project response

Maintainers will review the report, reproduce it when possible, determine the
affected supported surface, and coordinate a fix or documented mitigation. The
project does not promise a fixed acknowledgement or remediation deadline;
maintainer availability and the need to validate scientific workflows can vary.

A coordinated response may include:

- a private patch branch;
- dependency or workflow changes;
- a release or upgrade recommendation;
- revocation of exposed credentials or artifacts;
- a public advisory after users have a reasonable mitigation path.

Credit is provided when requested and appropriate, subject to safe disclosure and
consent.

## Out of scope for private security reporting

Use normal issues for non-sensitive bugs such as incorrect metrics, shape errors,
missing optional dependencies, documentation mistakes, or unsupported platform
behavior. Follow [CONTRIBUTING.md](CONTRIBUTING.md) and include a minimal
reproduction.

Licensing or provenance concerns involving a dataset, model, paper artifact, or
copied implementation may be sensitive. When public discussion could expose
private data or an unaddressed legal risk, use the private path above first.
