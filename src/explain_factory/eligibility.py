from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class ExplainEligibility:
    code: str
    message: str
    suggestion: str = ""
    missing_keys: Optional[List[str]] = None


@dataclass(frozen=True)
class ExplainReadyResult:
    ok: bool
    explainer_id: str
    reasons: List[ExplainEligibility]
    meta_source: str = ""
    degraded: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "explainer_id": self.explainer_id,
            "meta_source": self.meta_source,
            "degraded": self.degraded,
            "reasons": [asdict(r) for r in self.reasons],
        }


def explain_ready(
    explainer_id: str,
    meta: Optional[Dict[str, Any]],
    required_meta_keys: Optional[List[str]] = None,
    meta_source: str = "",
    degraded: bool = False,
) -> ExplainReadyResult:
    required = required_meta_keys or []
    meta = meta or {}

    missing = [k for k in required if k not in meta or meta.get(k) in (None, "")]
    reasons: List[ExplainEligibility] = []
    if missing:
        reasons.append(
            ExplainEligibility(
                code="MISSING_META",
                message="Missing required data metadata for explainer.",
                suggestion="Provide the required metadata keys via dataset/batch metadata.",
                missing_keys=missing,
            )
        )

    ok = not reasons
    return ExplainReadyResult(
        ok=ok,
        explainer_id=explainer_id,
        reasons=reasons,
        meta_source=meta_source,
        degraded=degraded,
    )


def write_eligibility(path: Path, result: ExplainReadyResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, ensure_ascii=False), encoding="utf-8")
