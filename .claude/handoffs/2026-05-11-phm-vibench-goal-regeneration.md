# Session Handoff: PHM-Vibench Goal Regeneration

**Date:** 2026-05-11
**Project:** `/home/user/LQ/B_Signal/vibench_fix/PHM-Vibench_fix`
**Session Duration:** Short goal-document update

## Current State

**Task:** Regenerate the PHM-Vibench controlling goal under `.specify/goals` using handoff, Claude Code Teams, and Speckit/specify requirements.
**Phase:** planning/documentation.
**Progress:** Goal document has been updated. The Speckit pipeline has not been re-run in this handoff step.

## What We Did

Updated `.specify/goals/phm-vibench-full-phm-experiment-platform.md` so it again requires the full mandatory Speckit chain, including `speckit-taskstoissues`, matching the latest user instruction. Strengthened the goal with explicit capability-inventory gates for PHM tasks, models, algorithms, losses, metrics, and baselines.

## Decisions Made

- **Latest instruction wins** - A prior session had waived `speckit-taskstoissues`; the latest user request explicitly requires Tasks & Issues, so the goal now includes `speckit-taskstoissues` again.
- **Goal-level only** - This update defines the mandatory execution protocol; it does not claim the Speckit pipeline was re-executed.
- **Codex remains lead-of-record** - Claude Code Teams are allowed only for partitioned, large, repetitive, or review-heavy work, and Codex must verify outputs before accepting them.

## Code Changes

**Files modified:**

- `.specify/goals/phm-vibench-full-phm-experiment-platform.md` - restored full Speckit mandatory chain, added capability inventory gates, clarified Claude Code Teams policy, and required handoff at blocked steps/team runs/phase completions.

**Key code context:** No runtime code was changed.

## Open Questions

- [ ] Should the already generated specs be regenerated to reflect the restored `speckit-taskstoissues` requirement, or should only future slices follow the restored chain?

## Blockers / Issues

- The repository remains dirty from broader ongoing work; do not revert unrelated changes.
- The current update only changes the controlling goal document.

## Context to Remember

User prefers first-principles and Occam-style minimal changes. The goal should force explicit unsupported/dependency-blocked statuses rather than silent fallback or inflated support claims.

## Next Steps

1. [ ] If implementation resumes, start from `.agents/skills/speckit-constitution` or the earliest blocked/invalidated Speckit step.
2. [ ] Run `speckit-taskstoissues` for applicable slices unless the user gives a newer explicit waiver.
3. [ ] Keep capability inventory docs derived from registries and validated scripts.

## Files to Review on Resume

- `.specify/goals/phm-vibench-full-phm-experiment-platform.md` - controlling goal and mandatory Speckit chain.
- `.claude/handoffs/2026-05-11-phm-vibench-final-goal-audit.md` - prior completed-audit context before this goal requirement was restored.
