# Security Policy

## Supported versions

PHM-Vibench is currently an alpha project and does not publish a long-term
security-support schedule.

| Version | Security-fix policy |
|---|---|
| Current `main` | Best-effort fixes for confirmed issues |
| Latest tagged release, when one exists | Best-effort fixes or documented mitigation |
| Older commits/releases | No guaranteed security updates |

A release-support statement in `SUPPORTED_COMPONENTS.md` or
`SUPPORTED_COMBINATIONS.md` describes functional scope; it is not a security
maintenance guarantee.

## Report a vulnerability privately

Do **not** disclose a suspected vulnerability in a public issue, pull request,
discussion, log, or example config.

Preferred reporting path:

- use GitHub's private vulnerability reporting form for this repository:
  <https://github.com/PHMbench/PHM-Vibench/security/advisories/new>

If GitHub does not present a private form to you, contact a PHMbench organization
owner or repository maintainer through an existing private channel. When no
private channel is available, open a non-sensitive issue asking for private
contact instructions; do not include exploit details, affected users, secrets,
or personally identifiable information.

## Include in the report

Provide as much of the following as can be shared safely:

- affected commit, branch, or release;
- affected file, component, config, or dependency;
- vulnerability description and realistic impact;
- reproduction steps or proof of concept;
- required data, privileges, network access, and environment;
- Python, PyTorch, CUDA, operating-system, and package versions;
- whether the issue affects the CLI, Streamlit workspace, artifacts,
  checkpoints, data handling, or third-party integration;
- proposed mitigation, if known;
- disclosure constraints or embargo requests.

Remove real credentials, private datasets, access tokens, and personal data from
attachments whenever possible.

## Project response

Maintainers will validate the report, identify the affected support boundary,
and decide whether to fix, mitigate, document, or reject it. The project does not
promise a fixed acknowledgement or remediation SLA.

For a confirmed issue, maintainers should:

1. minimize public disclosure until a mitigation is available;
2. add a regression test when technically appropriate;
3. update affected documentation and known limitations;
4. publish an advisory or release note when users need to take action;
5. credit the reporter when requested and safe.

## Scope notes

Security reports may include, but are not limited to:

- command or argument injection;
- unsafe path traversal, symlink handling, or artifact discovery;
- arbitrary code execution through configs, checkpoints, or serialized data;
- secret or private-data exposure;
- unsafe subprocess management in optional applications;
- dependency vulnerabilities with a concrete impact on PHM-Vibench;
- malicious or untrusted dataset/model handling.

General bugs, unsupported combinations, and performance disagreements should use
the normal issue templates instead of the private security process.
