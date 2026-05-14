from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


DEFAULT_QUEUE = Path("paper/UXFD_paper/goal/09_gpu_execution_queue.yaml")
DISALLOWED_LAUNCH_COMMAND_MARKERS = (
    "smoke",
    "demo",
    "dummy",
    "template",
    "pending",
)


@dataclass(frozen=True)
class QueueCommand:
    queue_id: str
    paper_id: str
    phase: str
    entry_id: str
    label: str
    command: str
    status: str
    matrix_path: str


@dataclass(frozen=True)
class QueueLaunchCommand:
    queue_id: str
    paper_id: str
    phase: str
    entry_id: str
    label: str
    device: str
    workdir: str
    command: str
    original_command: str
    status: str


@dataclass(frozen=True)
class QueueValidation:
    can_execute: bool
    structural_issues: Tuple[str, ...]
    resource_reason: str


@dataclass(frozen=True)
class LivePreflight:
    accepted: bool
    nvidia_smi_ok: bool
    torch_cuda_available: bool
    torch_cuda_device_count: int
    gpu_names: Tuple[str, ...]
    reason: str


def _load_yaml(path: Path) -> Mapping[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _entry_map(matrix: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    entries: Dict[str, Mapping[str, Any]] = {}
    proposed = matrix.get("proposed")
    if isinstance(proposed, Mapping):
        entries[str(proposed["id"])] = proposed
    for phase in ("baselines", "ablations"):
        for item in matrix.get(phase, []):
            entries[str(item["id"])] = item
    return entries


def _launch_command_disallowed_marker(command: str) -> str:
    lowered = command.lower()
    return next(
        (marker for marker in DISALLOWED_LAUNCH_COMMAND_MARKERS if marker in lowered),
        "",
    )


def _iter_matrix_commands(
    queue_item: Mapping[str, Any],
    matrix: Mapping[str, Any],
) -> Sequence[QueueCommand]:
    rows: List[QueueCommand] = []
    matrix_path = str(queue_item["matrix_path"])
    paper_id = str(queue_item["paper_id"])
    queue_id = str(queue_item["queue_id"])

    proposed = matrix.get("proposed")
    if isinstance(proposed, Mapping):
        rows.append(
            QueueCommand(
                queue_id=queue_id,
                paper_id=paper_id,
                phase="proposed",
                entry_id=str(proposed.get("id", "")),
                label=str(proposed.get("label", "")),
                command=str(proposed.get("command", "")),
                status=str(proposed.get("accepted_evidence_status", "")),
                matrix_path=matrix_path,
            )
        )

    for phase in ("baselines", "ablations"):
        for item in matrix.get(phase, []):
            rows.append(
                QueueCommand(
                    queue_id=queue_id,
                    paper_id=paper_id,
                    phase=phase,
                    entry_id=str(item.get("id", "")),
                    label=str(item.get("label", "")),
                    command=str(item.get("command", "")),
                    status=str(item.get("accepted_evidence_status", "")),
                    matrix_path=matrix_path,
                )
            )
    return tuple(rows)


def load_queue(path: Path = DEFAULT_QUEUE) -> Mapping[str, Any]:
    return _load_yaml(path)


def expand_queue(path: Path = DEFAULT_QUEUE) -> Tuple[QueueCommand, ...]:
    queue = load_queue(path)
    rows: List[QueueCommand] = []
    for queue_item in queue["paper_queue"]:
        matrix_path = Path(queue_item["matrix_path"])
        matrix = _load_yaml(matrix_path)
        rows.extend(_iter_matrix_commands(queue_item, matrix))

    for binding in queue.get("top_representative_bindings", []):
        rows.append(
            QueueCommand(
                queue_id=str(binding["binding_id"]),
                paper_id=str(binding["paper_id"]),
                phase="top_representatives",
                entry_id=",".join(binding["local_proxy_matrix_entries"]),
                label=str(binding["external_work_id"]),
                command=str(binding["command_source"]),
                status=str(binding["status"]),
                matrix_path="",
            )
        )
    return tuple(rows)


def validate_queue(path: Path = DEFAULT_QUEUE) -> QueueValidation:
    queue = load_queue(path)
    issues: List[str] = []
    matrix_entries: Dict[str, Dict[str, Mapping[str, Any]]] = {}

    resource_preflight = queue.get("resource_preflight", {})
    required_devices = [str(device) for device in resource_preflight.get("required_devices", [])]
    scheduler_devices = [str(device) for device in queue.get("scheduler", {}).get("default_devices", [])]
    if required_devices != ["0", "1"]:
        issues.append("resource_preflight: required_devices must be ['0', '1']")
    if scheduler_devices and scheduler_devices != required_devices:
        issues.append("scheduler: default_devices must match resource_preflight.required_devices")

    for queue_item in queue.get("paper_queue", []):
        paper_id = str(queue_item.get("paper_id", ""))
        matrix_path = Path(str(queue_item.get("matrix_path", "")))
        if not matrix_path.exists():
            issues.append(f"{paper_id}: missing matrix {matrix_path}")
            continue
        matrix = _load_yaml(matrix_path)
        if matrix.get("paper_id") != paper_id:
            issues.append(f"{paper_id}: matrix paper_id mismatch")
        matrix_entries[paper_id] = _entry_map(matrix)
        rejected_commands = [
            f"{row.phase}:{row.entry_id}:{_launch_command_disallowed_marker(row.command)}"
            for row in _iter_matrix_commands(queue_item, matrix)
            if _launch_command_disallowed_marker(row.command)
        ]
        if rejected_commands:
            issues.append(
                f"{paper_id}: {len(rejected_commands)} launch commands reference "
                "smoke/demo/dummy/template/pending evidence: "
                + ", ".join(rejected_commands[:6])
            )

    for binding in queue.get("top_representative_bindings", []):
        paper_id = str(binding.get("paper_id", ""))
        entries = matrix_entries.get(paper_id, {})
        for entry_id in binding.get("local_proxy_matrix_entries", []):
            if entry_id not in entries:
                issues.append(f"{paper_id}: missing TOP proxy entry {entry_id}")
        if binding.get("status") != "pending_gpu_and_artifacts":
            issues.append(f"{paper_id}: TOP binding is not pending")

    current = resource_preflight.get("current_session_result", {})
    required_count = len(required_devices)
    required_gpu_class = str(resource_preflight.get("required_gpu_class", ""))
    device_count = int(current.get("torch_cuda_device_count", 0))
    gpu_names = tuple(str(name) for name in current.get("gpu_names", []) or [])
    device_count_ok = required_count > 0 and device_count == required_count
    gpu_class_ok = (
        required_gpu_class != ""
        and len(gpu_names) == required_count
        and all(required_gpu_class in name for name in gpu_names)
    )
    can_execute = (
        queue.get("status") != "blocked_resource_preflight"
        and bool(current.get("torch_cuda_available"))
        and device_count_ok
        and gpu_class_ok
        and not issues
    )
    resource_reason = str(current.get("verdict", "resource preflight not recorded"))
    return QueueValidation(
        can_execute=can_execute,
        structural_issues=tuple(issues),
        resource_reason=resource_reason,
    )


def run_live_preflight() -> LivePreflight:
    try:
        nvidia = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        nvidia_ok = nvidia.returncode == 0
        nvidia_reason = nvidia.stdout.strip() or nvidia.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        nvidia_ok = False
        nvidia_reason = f"{type(exc).__name__}: {exc}"

    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        cuda_available = False
        device_count = 0
        gpu_names: Tuple[str, ...] = ()
        torch_reason = f"torch import failed: {type(exc).__name__}: {exc}"
    else:
        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count())
        gpu_names = tuple(
            str(torch.cuda.get_device_name(index)) for index in range(device_count)
        )
        torch_reason = (
            f"torch cuda_available={cuda_available}, device_count={device_count}"
        )

    gpu_class_ok = device_count == 2 and all("4090" in name for name in gpu_names)
    accepted = nvidia_ok and cuda_available and gpu_class_ok
    if accepted:
        reason = (
            "accepted: nvidia-smi and PyTorch expose exactly two "
            "RTX 4090-class CUDA devices"
        )
    else:
        class_reason = (
            ""
            if gpu_class_ok
            else f"; required_gpu_class=RTX 4090 not satisfied by gpu_names={gpu_names}"
        )
        reason = f"blocked: {nvidia_reason}; {torch_reason}{class_reason}"
    return LivePreflight(
        accepted=accepted,
        nvidia_smi_ok=nvidia_ok,
        torch_cuda_available=cuda_available,
        torch_cuda_device_count=device_count,
        gpu_names=gpu_names,
        reason=reason,
    )


CUDA_VISIBLE_DEVICES_PREFIX = re.compile(r"^CUDA_VISIBLE_DEVICES=[^\s]+\s+")
LAUNCHABLE_PHASES = {"proposed", "baselines", "ablations"}


def _bind_command_to_device(command: str, device: str) -> str:
    if CUDA_VISIBLE_DEVICES_PREFIX.match(command):
        return CUDA_VISIBLE_DEVICES_PREFIX.sub(
            f"CUDA_VISIBLE_DEVICES={device} ", command, count=1
        )
    if command.startswith("python "):
        return f"CUDA_VISIBLE_DEVICES={device} {command}"
    return command


def _paper_root_from_matrix(matrix_path: str) -> str:
    path = Path(matrix_path)
    if path.name == "baseline_ablation_matrix.yaml" and path.parent.name == "submission_prep":
        return str(path.parent.parent)
    return "."


def _command_body(command: str) -> str:
    return CUDA_VISIBLE_DEVICES_PREFIX.sub("", command, count=1)


def _is_paper_local_command(command: str) -> bool:
    body = _command_body(command)
    return body.startswith(
        (
            "python scripts/",
            "python experiments/",
            "python code/",
            "python simple_validation_demo.py",
            "python -m pytest -q code/",
        )
    )


def _shell_command(command: str, workdir: str) -> str:
    if workdir == ".":
        return command
    return f"(cd {shlex.quote(workdir)} && {command})"


def _static_validation_guard(validation: QueueValidation) -> Tuple[str, ...]:
    if validation.can_execute:
        return ()
    lines = [
        "# Static queue validation failed at generation time.",
        "# Regenerate this launch plan only after the queue and resource gates pass.",
        "printf '%s\\n' 'Blocked: static queue validation can_execute=False'",
        f"printf '%s\\n' {shlex.quote('Resource reason: ' + validation.resource_reason)}",
        f"printf '%s\\n' {shlex.quote('Structural issues: ' + str(len(validation.structural_issues)))}",
    ]
    if validation.structural_issues:
        joined = "; ".join(validation.structural_issues)
        lines.append(f"printf '%s\\n' {shlex.quote('Structural issue detail: ' + joined)}")
    lines.append("exit 2")
    return tuple(lines)


def build_launch_plan(
    rows: Sequence[QueueCommand],
    devices: Sequence[str] = ("0", "1"),
) -> Tuple[QueueLaunchCommand, ...]:
    if not devices:
        raise ValueError("devices must not be empty")

    launch_rows: List[QueueLaunchCommand] = []
    for row in rows:
        if row.phase not in LAUNCHABLE_PHASES:
            continue
        if not row.command or row.command.startswith("blocked:"):
            continue

        device = str(devices[len(launch_rows) % len(devices)])
        workdir = _paper_root_from_matrix(row.matrix_path) if _is_paper_local_command(row.command) else "."
        launch_rows.append(
            QueueLaunchCommand(
                queue_id=row.queue_id,
                paper_id=row.paper_id,
                phase=row.phase,
                entry_id=row.entry_id,
                label=row.label,
                device=device,
                workdir=workdir,
                command=_bind_command_to_device(row.command, device),
                original_command=row.command,
                status=row.status,
            )
        )
    return tuple(launch_rows)


def render_shell_plan(
    rows: Sequence[QueueCommand],
    validation: QueueValidation,
    devices: Sequence[str] = ("0", "1"),
    device_filter: Optional[str] = None,
) -> str:
    launch_rows = build_launch_plan(rows, devices=devices)
    if device_filter is not None:
        launch_rows = tuple(row for row in launch_rows if row.device == device_filter)
        title = f"# Launch shard for CUDA device {device_filter}"
    else:
        title = "# Generated by scripts.uxfd_gpu_queue."
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        title,
        "# Run only on the local 2x4090 machine after this preflight passes.",
        f"# Queue validation can_execute at generation time: {validation.can_execute}",
        f"# Queue validation resource reason: {validation.resource_reason}",
        f"# Launchable commands: {len(launch_rows)}",
        "",
    ]
    guard_lines = _static_validation_guard(validation)
    if guard_lines:
        lines.extend(guard_lines)
        lines.append("")
    lines.extend(
        [
            "python -m scripts.uxfd_experiment_launch_gate --format markdown",
            "",
            "nvidia-smi -L",
            (
                "python -c \"import torch; assert torch.cuda.is_available(); "
                "assert torch.cuda.device_count() == 2; "
                "names=[torch.cuda.get_device_name(i) for i in range(2)]; "
                "assert all('RTX 4090' in name for name in names), names; "
                "print(names[0]); print(names[1])\""
            ),
            "",
        ]
    )
    for row in launch_rows:
        lines.extend(
            [
                (
                    f"# {row.queue_id} {row.paper_id} {row.phase} "
                    f"{row.entry_id} device={row.device} workdir={row.workdir}: {row.label}"
                ),
                _shell_command(row.command, row.workdir),
                "",
            ]
        )
    return "\n".join(lines)


def write_shell_shards(
    rows: Sequence[QueueCommand],
    validation: QueueValidation,
    shard_dir: Path,
    devices: Sequence[str] = ("0", "1"),
) -> Tuple[Path, ...]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for device in devices:
        path = shard_dir / f"gpu{device}.sh"
        path.write_text(
            render_shell_plan(
                rows,
                validation,
                devices=devices,
                device_filter=str(device),
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        written.append(path)

    readme = shard_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# UXFD 2x4090 Launch Shards",
                "",
                "Generated from `paper/UXFD_paper/goal/09_gpu_execution_queue.yaml`.",
                "These scripts are launch plans, not accepted evidence.",
                "Run them only after the embedded preflight passes.",
                "",
                "| Device | Script |",
                "|---|---|",
                *[f"| `{device}` | `gpu{device}.sh` |" for device in devices],
                "",
            ]
        ),
        encoding="utf-8",
    )
    written.append(readme)
    return tuple(written)


def render_markdown(
    rows: Sequence[QueueCommand],
    validation: QueueValidation,
    live_preflight: Optional[LivePreflight] = None,
) -> str:
    summary = summarize_rows(rows)
    lines = [
        "# UXFD GPU Queue Dry Run",
        "",
        f"- Can execute now: `{validation.can_execute}`",
        f"- Resource status: {validation.resource_reason}",
        f"- Structural issues: `{len(validation.structural_issues)}`",
        f"- Total dry-run entries: `{summary['total']}`",
        f"- Blocked entries: `{summary['blocked']}`",
        f"- TOP representative entries: `{summary['top_representatives']}`",
    ]
    if live_preflight is not None:
        lines.extend(
            [
                f"- Live preflight accepted: `{live_preflight.accepted}`",
                f"- Live preflight reason: {live_preflight.reason}",
            ]
        )
    lines.extend(
        [
            "",
            "| Queue | Paper | Phase | Entry | Status | Command source |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in rows:
        command = row.command.replace("|", "\\|")
        lines.append(
            f"| `{row.queue_id}` | `{row.paper_id}` | `{row.phase}` | "
            f"`{row.entry_id}` | `{row.status}` | `{command}` |"
        )
    return "\n".join(lines) + "\n"


def summarize_rows(rows: Sequence[QueueCommand]) -> Mapping[str, Any]:
    per_phase: Dict[str, int] = {}
    per_paper: Dict[str, Dict[str, int]] = {}
    blocked = 0
    main_py_commands = 0
    for row in rows:
        per_phase[row.phase] = per_phase.get(row.phase, 0) + 1
        paper_counts = per_paper.setdefault(row.paper_id, {})
        paper_counts[row.phase] = paper_counts.get(row.phase, 0) + 1
        is_blocked = row.command.startswith("blocked:") or "blocked" in row.status.lower()
        if is_blocked:
            blocked += 1
            paper_counts["blocked"] = paper_counts.get("blocked", 0) + 1
        if "python main.py --config" in row.command:
            main_py_commands += 1
    return {
        "total": len(rows),
        "blocked": blocked,
        "main_py_commands": main_py_commands,
        "top_representatives": per_phase.get("top_representatives", 0),
        "per_phase": per_phase,
        "per_paper": per_paper,
    }


def build_payload(
    rows: Sequence[QueueCommand],
    validation: QueueValidation,
    live_preflight: Optional[LivePreflight] = None,
) -> Mapping[str, Any]:
    payload: Dict[str, Any] = {
        "validation": asdict(validation),
        "summary": summarize_rows(rows),
        "commands": [asdict(row) for row in rows],
    }
    if live_preflight is not None:
        payload["live_preflight"] = asdict(live_preflight)
    return payload


def render_output(
    rows: Sequence[QueueCommand],
    validation: QueueValidation,
    output_format: str,
    live_preflight: Optional[LivePreflight] = None,
) -> str:
    if output_format == "json":
        return json.dumps(build_payload(rows, validation, live_preflight), indent=2) + "\n"
    if output_format == "shell":
        return render_shell_plan(rows, validation)
    return render_markdown(rows, validation, live_preflight)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Expand the UXFD 2x4090 dry-run queue")
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--format", choices=("markdown", "json", "shell"), default="markdown")
    parser.add_argument("--output", type=Path, help="Optional dry-run manifest output path")
    parser.add_argument(
        "--shard-dir",
        type=Path,
        help="Optional directory for per-GPU shell launch shards",
    )
    parser.add_argument("--live-preflight", action="store_true")
    parser.add_argument("--require-preflight", action="store_true")
    args = parser.parse_args(argv)

    rows = expand_queue(args.queue)
    validation = validate_queue(args.queue)
    live_preflight = run_live_preflight() if args.live_preflight else None
    output = render_output(rows, validation, args.format, live_preflight)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    else:
        print(output, end="")
    if args.shard_dir:
        write_shell_shards(rows, validation, args.shard_dir)

    if validation.structural_issues:
        return 1
    preflight_accepted = (
        live_preflight.accepted if live_preflight is not None else validation.can_execute
    )
    if args.require_preflight and not preflight_accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
