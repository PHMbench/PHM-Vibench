from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class CheckResult:
    ok: bool
    required_missing: List[str]
    optional_missing: List[str]
    required_found: List[str]
    optional_found: List[str]


def _read_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _glob_any(run_dir: Path, pattern: str) -> List[Path]:
    return sorted(run_dir.glob(pattern))


def _exists_by_pattern(run_dir: Path, pattern: str) -> bool:
    if any(ch in pattern for ch in ["*", "?", "["]):
        return any(p.exists() for p in run_dir.glob(pattern))
    return (run_dir / pattern).exists()


def _first_match(run_dir: Path, pattern: str) -> Optional[Path]:
    for p in _glob_any(run_dir, pattern):
        if p.exists():
            return p
    if (run_dir / pattern).exists():
        return run_dir / pattern
    return None


def _discover_run_dirs_from_manifests(root_dir: Path, manifests_glob: str) -> List[Path]:
    manifests = sorted(root_dir.glob(manifests_glob))
    run_dirs: List[Path] = []
    for mp in manifests:
        # <run_dir>/artifacts/manifest.json
        try:
            run_dirs.append(mp.parents[1])
        except Exception:
            continue
    # de-dup while stable
    seen = set()
    unique: List[Path] = []
    for rd in run_dirs:
        rp = str(rd.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(rd)
    return unique


def _run_checks(run_dir: Path, required: Sequence[str], optional: Sequence[str]) -> CheckResult:
    required_missing: List[str] = []
    required_found: List[str] = []
    optional_missing: List[str] = []
    optional_found: List[str] = []

    for pat in required:
        if _exists_by_pattern(run_dir, pat):
            required_found.append(pat)
        else:
            required_missing.append(pat)

    for pat in optional:
        if _exists_by_pattern(run_dir, pat):
            optional_found.append(pat)
        else:
            optional_missing.append(pat)

    ok = not required_missing
    return CheckResult(
        ok=ok,
        required_missing=required_missing,
        optional_missing=optional_missing,
        required_found=required_found,
        optional_found=optional_found,
    )


def _configure_matplotlib() -> None:
    # Headless-safe default; no seaborn/scienceplots dependency.
    import matplotlib

    matplotlib.use("Agg", force=True)


def _plot_learning_curve(
    run_dir: Path,
    metrics_glob: str,
    out_dir: Path,
    save_formats: Sequence[str],
    series: Sequence[Dict[str, Any]],
) -> Tuple[bool, str, List[str]]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import pandas as pd

    metrics_path = _first_match(run_dir, metrics_glob)
    if metrics_path is None:
        return False, f"metrics not found: {metrics_glob}", []

    df = pd.read_csv(metrics_path)
    if df.empty:
        return False, f"empty metrics csv: {metrics_path}", []

    x_col = "step" if "step" in df.columns else ("epoch" if "epoch" in df.columns else None)
    if x_col is None:
        x = list(range(len(df)))
        x_label = "index"
    else:
        x = df[x_col].tolist()
        x_label = x_col

    # Auto-pick common scalar columns if not specified.
    picked: List[Tuple[str, str]] = []
    if series:
        for item in series:
            key = str(item.get("key", "")).strip()
            label = str(item.get("label", "")).strip() or key
            if key and key in df.columns:
                picked.append((key, label))
    else:
        # KISS heuristic: pick up to 8 columns containing loss/acc/f1/precision/recall.
        preferred_tokens = ("loss", "acc", "f1", "precision", "recall")
        numeric_cols = [c for c in df.columns if c != x_col and str(df[c].dtype) != "object"]
        for token in preferred_tokens:
            for c in numeric_cols:
                if token in c.lower():
                    picked.append((c, c))
        # de-dup preserve order
        seen = set()
        picked2: List[Tuple[str, str]] = []
        for k, lbl in picked:
            if k in seen:
                continue
            seen.add(k)
            picked2.append((k, lbl))
        picked = picked2[:8]

    if not picked:
        return False, f"no plottable columns in {metrics_path}", []

    plt.figure(figsize=(10, 6))
    for key, label in picked:
        try:
            plt.plot(x, df[key].tolist(), label=label)
        except Exception:
            continue

    plt.xlabel(x_label)
    plt.ylabel("value")
    plt.title("Learning Curve")
    plt.grid(True, linewidth=0.4, alpha=0.6)
    plt.legend(loc="best")
    plt.tight_layout()

    written: List[str] = []
    for fmt in save_formats:
        fmt_norm = fmt.lstrip(".").lower()
        out_path = out_dir / f"learning_curve.{fmt_norm}"
        plt.savefig(out_path, dpi=200)
        written.append(str(out_path))
    plt.close()

    return True, "ok", written


def _plot_confusion_matrix(
    run_dir: Path,
    predictions_glob: str,
    out_dir: Path,
    save_formats: Sequence[str],
    normalize: bool,
) -> Tuple[bool, str, List[str]]:
    _configure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    pred_path = _first_match(run_dir, predictions_glob)
    if pred_path is None:
        return False, f"predictions not found: {predictions_glob}", []

    data = np.load(pred_path, allow_pickle=False)
    if "y_true" not in data or ("y_pred" not in data and "logits" not in data):
        return False, f"predictions missing y_true and (y_pred|logits): {pred_path}", []

    y_true = data["y_true"]
    if "y_pred" in data:
        y_pred = data["y_pred"]
    else:
        logits = data["logits"]
        y_pred = logits.argmax(axis=1)

    y_true = y_true.astype(int).reshape(-1)
    y_pred = y_pred.astype(int).reshape(-1)

    num_classes = int(max(y_true.max(initial=0), y_pred.max(initial=0)) + 1)
    mat = np.zeros((num_classes, num_classes), dtype=float)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            mat[t, p] += 1.0

    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        mat = mat / row_sum

    plt.figure(figsize=(7, 6))
    plt.imshow(mat, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix" + (" (normalized)" if normalize else ""))
    plt.xlabel("pred")
    plt.ylabel("true")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()

    written: List[str] = []
    for fmt in save_formats:
        fmt_norm = fmt.lstrip(".").lower()
        out_path = out_dir / f"confusion_matrix.{fmt_norm}"
        plt.savefig(out_path, dpi=200)
        written.append(str(out_path))
    plt.close()
    return True, "ok", written


def _write_plot_eligibility(run_dir: Path, payload: Dict[str, Any]) -> None:
    try:
        out_dir = run_dir / "artifacts" / "plots"
        _safe_mkdir(out_dir)
        (out_dir / "plot_eligibility.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception:
        return


def _collect_run_report(
    run_dir: Path, check: CheckResult, plot_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "checks": {
            "ok": check.ok,
            "required_found": check.required_found,
            "required_missing": check.required_missing,
            "optional_found": check.optional_found,
            "optional_missing": check.optional_missing,
        },
        "plots": plot_results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="UXFD post-run checker and offline plotting (standalone; not integrated into main.py)."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to post-run YAML config.")
    args = parser.parse_args(argv)

    cfg = _read_yaml(Path(args.config))

    input_cfg = cfg.get("input", {}) or {}
    mode = str(input_cfg.get("mode", "manifest_glob"))
    root_dir = Path(str(input_cfg.get("root_dir", "save")))
    manifests_glob = str(input_cfg.get("manifests_glob", "**/artifacts/manifest.json"))
    run_dirs_cfg = input_cfg.get("run_dirs", []) or []

    if mode == "run_dirs":
        run_dirs = [Path(str(p)) for p in run_dirs_cfg]
    else:
        run_dirs = _discover_run_dirs_from_manifests(root_dir, manifests_glob)

    checks_cfg = cfg.get("checks", {}) or {}
    check_enable = bool(checks_cfg.get("enable", True))
    fail_on_missing_required = bool(checks_cfg.get("fail_on_missing_required", True))
    required = [str(x) for x in (checks_cfg.get("required", []) or [])]
    optional = [str(x) for x in (checks_cfg.get("optional", []) or [])]

    plots_cfg = cfg.get("plots", {}) or {}
    plot_enable = bool(plots_cfg.get("enable", True))
    out_dir_rel = str(plots_cfg.get("out_dir", "figures"))
    save_formats = [str(x) for x in (plots_cfg.get("save_formats", ["png"]) or ["png"])]
    items = plots_cfg.get("items", []) or []

    if not run_dirs:
        print("[postrun] no run dirs discovered")
        return 0

    any_fail = False
    for run_dir in run_dirs:
        run_dir = run_dir.resolve()
        plot_out_dir = run_dir / out_dir_rel
        plot_results: List[Dict[str, Any]] = []

        if check_enable:
            check = _run_checks(run_dir, required=required, optional=optional)
        else:
            check = CheckResult(
                ok=True,
                required_missing=[],
                optional_missing=[],
                required_found=[],
                optional_found=[],
            )

        if fail_on_missing_required and not check.ok:
            any_fail = True

        if plot_enable:
            _safe_mkdir(plot_out_dir)
            for item in items:
                kind = str(item.get("type", "")).strip()
                if kind == "learning_curve":
                    ok, msg, written = _plot_learning_curve(
                        run_dir=run_dir,
                        metrics_glob=str(item.get("metrics_glob", "logs/**/metrics.csv")),
                        out_dir=plot_out_dir,
                        save_formats=save_formats,
                        series=item.get("series", []) or [],
                    )
                elif kind == "confusion_matrix":
                    ok, msg, written = _plot_confusion_matrix(
                        run_dir=run_dir,
                        predictions_glob=str(item.get("predictions_glob", "artifacts/predictions.npz")),
                        out_dir=plot_out_dir,
                        save_formats=save_formats,
                        normalize=bool(item.get("normalize", True)),
                    )
                else:
                    ok, msg, written = False, f"unknown plot type: {kind}", []

                plot_results.append({"type": kind, "ok": ok, "message": msg, "written": written})

        report = _collect_run_report(run_dir, check=check, plot_results=plot_results)
        _write_plot_eligibility(run_dir, report)

        print(f"[postrun] run_dir={run_dir} checks_ok={report['checks']['ok']} plots={len(plot_results)}")
        if report["checks"]["required_missing"]:
            print(f"  - missing required: {report['checks']['required_missing']}")
        for pr in plot_results:
            status = "ok" if pr["ok"] else "skip"
            print(f"  - plot {pr['type']}: {status} ({pr['message']})")

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())

