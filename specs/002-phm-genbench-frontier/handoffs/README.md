# Handoffs

This directory stores feature-scoped handoffs for PHM-GenBench goals.

Each handoff must record:

- Goal ID
- Objective
- Files changed
- Runtime behavior changed: yes/no
- Contracts touched
- Validation commands run
- Validation results
- Known risks
- Required reviewers
- Required context files
- Review output format
- Next steps

Handoffs should distinguish verified results from blocked work. A blocked
Claude review or unavailable GPU 6/7 run is evidence of non-execution, not
independent approval.

Claude review output, when available, must end with:

```xml
<REVIEW_DECISION>APPROVE | REQUEST_CHANGES | BLOCKING</REVIEW_DECISION>
<BLOCKING_ISSUES>
...
</BLOCKING_ISSUES>
<NON_BLOCKING_ISSUES>
...
</NON_BLOCKING_ISSUES>
<FIX_INSTRUCTION>
Codex-ready patch instruction.
</FIX_INSTRUCTION>
```
