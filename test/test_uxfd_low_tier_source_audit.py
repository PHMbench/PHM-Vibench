import json
from pathlib import Path

from scripts.uxfd_low_tier_source_audit import (
    build_payload,
    evaluate_low_tier_source_audit,
    main,
    render_markdown,
)


PERSISTED_LOW_TIER_AUDIT_MD = Path("paper/UXFD_paper/results/low_tier_source_audit.md")
PERSISTED_LOW_TIER_AUDIT_JSON = Path("paper/UXFD_paper/results/low_tier_source_audit.json")


def test_low_tier_source_audit_has_no_active_manuscript_or_bib_blockers() -> None:
    report = evaluate_low_tier_source_audit()

    assert report.ready is True
    assert report.blocker_count == 0
    assert report.triage_count > 0
    assert all("/goal/" not in finding.path for finding in report.findings)
    assert all(finding.severity != "blocker" for finding in report.findings)


def test_low_tier_source_audit_blocks_user_named_low_quality_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "UXFD_paper"
    manuscript = root / "PaperA" / "manuscript"
    manuscript.mkdir(parents=True)
    (manuscript / "draft.md").write_text(
        "\n".join(
            [
                "Scientific Reports baseline is not allowed in an active draft.",
                "MDPI source is not allowed in an active draft.",
                "IEEE TIM target venue is not allowed in an active draft.",
                (
                    "IEEE Transactions on Instrumentation and Measurement is not "
                    "allowed in an active draft."
                ),
            ]
        ),
        encoding="utf-8",
    )
    (root / "PaperA" / "ref.bib").write_text(
        "\n".join(
            [
                "@article{bad_applied,",
                "  journal = {Applied Sciences},",
                "}",
                "@article{bad_sensors,",
                "  journal = {Sensors},",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    report = evaluate_low_tier_source_audit(root)
    markers = {finding.marker for finding in report.findings}

    assert report.ready is False
    assert report.blocker_count == len(report.findings) == 6
    assert markers == {
        "Scientific Reports",
        "MDPI",
        "IEEE TIM",
        "IEEE Transactions on Instrumentation and Measurement",
        "Applied Sciences",
        "Sensors",
    }
    assert all(finding.severity == "blocker" for finding in report.findings)


def test_low_tier_source_audit_cli_writes_reports(tmp_path: Path) -> None:
    output = tmp_path / "low_tier" / "audit.json"

    assert main(["--format", "json", "--output", str(output)]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["ready"] is True
    assert payload["blocker_count"] == 0
    assert payload["triage_count"] > 0

    markdown = tmp_path / "low_tier" / "audit.md"
    assert (
        main(
            [
                "--format",
                "markdown",
                "--output",
                str(markdown),
                "--allow-not-ready",
            ]
        )
        == 0
    )
    text = markdown.read_text(encoding="utf-8")
    assert "UXFD Low-Tier Source Audit" in text
    assert "Disallowed active-source markers" in text
    assert "`Scientific Reports`" in text
    assert "`MDPI`" in text
    assert "`IEEE TIM`" in text
    assert "Disallowed exact BibTeX journal fields" in text
    assert "## Blockers" in text


def test_persisted_low_tier_source_audit_reports_match_current_audit() -> None:
    report = evaluate_low_tier_source_audit()

    expected_json = json.dumps(build_payload(report), indent=2) + "\n"
    assert PERSISTED_LOW_TIER_AUDIT_JSON.read_text(encoding="utf-8") == expected_json
    assert PERSISTED_LOW_TIER_AUDIT_MD.read_text(encoding="utf-8") == render_markdown(report)
