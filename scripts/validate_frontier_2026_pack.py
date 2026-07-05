from __future__ import annotations
import json
from pathlib import Path
import yaml


def main() -> int:
    registry = Path("configs/registry/generative_frontier_methods.yaml")
    if not registry.exists():
        raise FileNotFoundError(registry)
    data = yaml.safe_load(registry.read_text(encoding="utf-8")) or {}
    methods = data.get("methods", {})
    cards = Path("projects/phm_generative/frontier_2026/method_cards")
    losses = Path("projects/phm_generative/frontier_2026/loss_cards")
    issues = []
    for method_id, spec in methods.items():
        if not (cards / f"{method_id}.md").exists():
            issues.append(f"missing method card: {method_id}")
        if not (losses / f"{method_id}.md").exists():
            issues.append(f"missing loss card: {method_id}")
        if spec.get("claim_status") != "exploratory":
            issues.append(f"frontier method must default exploratory: {method_id}")
    print(json.dumps({"method_count": len(methods), "issues": issues}, indent=2, ensure_ascii=False))
    return 0 if len(methods) == 10 and not issues else 1

if __name__ == "__main__":
    raise SystemExit(main())
