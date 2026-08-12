#!/usr/bin/env python3
"""Smoke runner for LLM grounding, context, and latency evidence surfaces."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CODE_DIR = PROJECT_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from llm_explainable_toolkit.core.intermediate_representation import (  # noqa: E402
    LLMIntermediateRepresentation,
)

from run_minimal_llm_demo import MinimalLLMDemo  # noqa: E402


FAULT_TERMS = ("内圈故障", "外圈故障", "齿轮故障", "不对中", "复合故障")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cuda_metadata() -> Mapping[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [
        torch.cuda.get_device_name(index)
        for index in range(device_count)
    ]
    return {
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "gpu_names": gpu_names,
    }


def _clone_ir(ir: LLMIntermediateRepresentation) -> LLMIntermediateRepresentation:
    return LLMIntermediateRepresentation.from_dict(ir.to_dict())


def _strip_domain_context(ir: LLMIntermediateRepresentation) -> LLMIntermediateRepresentation:
    stripped = _clone_ir(ir)
    stripped.device_context.device_type = "Unknown"
    stripped.device_context.load_condition = "Unknown"
    stripped.device_context.operating_speed = None
    stripped.signal_analysis.key_findings = []
    stripped.technical_explanation.important_features = []
    stripped.technical_explanation.frequency_components = []
    return stripped


def _expected_tokens(ir: LLMIntermediateRepresentation) -> List[str]:
    tokens = [
        ir.fault_info.fault_type,
        ir.device_context.device_type,
        f"{ir.fault_info.confidence:.1%}",
    ]
    dominant = ir.signal_analysis.frequency_analysis.get("dominant_frequency")
    if dominant is not None:
        tokens.append(f"{float(dominant):.1f}")
    for component in ir.technical_explanation.frequency_components:
        frequency = component.get("frequency")
        if frequency is not None:
            tokens.append(f"{float(frequency):.1f}")
    return [token for token in tokens if token and token != "Unknown"]


def _unsupported_claims(
    response: str,
    ir: LLMIntermediateRepresentation,
    checker_enabled: bool,
) -> List[str]:
    if not checker_enabled:
        return []

    unsupported: List[str] = []
    for fault_term in FAULT_TERMS:
        if fault_term != ir.fault_info.fault_type and fault_term in response:
            unsupported.append(f"unexpected_fault_term:{fault_term}")

    expected = set(_expected_tokens(ir))
    response_numbers = set(re.findall(r"(\d+(?:\.\d+)?)\s*Hz", response))
    expected_numbers = {
        token.rstrip("%")
        for token in expected
        if re.fullmatch(r"\d+(?:\.\d+)?%?", token)
    }
    unexpected_numbers = response_numbers - expected_numbers
    if unexpected_numbers:
        unsupported.append("unexpected_numeric_token:" + ",".join(sorted(unexpected_numbers)))

    if ir.device_context.device_type != "Unknown":
        other_devices = ("滚动轴承", "电机驱动系统", "齿轮箱", "高速离心机")
        for device in other_devices:
            if device != ir.device_context.device_type and device in response:
                unsupported.append(f"unexpected_device:{device}")

    return unsupported


def _condition_queries(condition: str) -> Sequence[str]:
    if condition == "latency_short":
        return ("解释故障。",)
    if condition == "latency_long":
        return (
            "请给出包含故障机理、信号证据、维修建议、风险等级和监测方案的详细解释。",
        )
    return (
        "请解释这个故障的原因",
        "应该如何维修这个故障？",
        "故障的严重程度如何？",
        "请提供详细的技术分析",
    )


def _style_for_condition(condition: str) -> str:
    if condition == "latency_short":
        return "simple"
    if condition == "latency_long":
        return "detailed"
    return "standard"


def _case_irs(condition: str, demo: MinimalLLMDemo) -> Iterable[LLMIntermediateRepresentation]:
    for case in demo.demo_cases:
        ir = case["ir"]
        if condition == "no_domain_context":
            yield _strip_domain_context(ir)
        else:
            yield ir


def _run_condition(
    condition: str,
    output_root: Path,
    seed: int,
    repeats: int,
) -> None:
    started_at = datetime.now().isoformat()
    started_perf = time.perf_counter()
    demo = MinimalLLMDemo()
    checker_enabled = condition != "no_checker"
    if condition == "no_domain_context":
        demo.llm.fault_knowledge = {}
    demo.llm.set_style(_style_for_condition(condition))

    condition_root = output_root / condition / f"seed_{seed}"
    inputs_root = condition_root / "inputs"
    outputs_root = condition_root / "outputs"
    logs_root = condition_root / "logs"
    artifacts_root = condition_root / "artifacts"
    for directory in (inputs_root, outputs_root, logs_root, artifacts_root):
        directory.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    prompt_set: List[Dict[str, Any]] = []
    responses_path = outputs_root / "responses.jsonl"
    with responses_path.open("w", encoding="utf-8") as response_file:
        for repeat in range(repeats):
            for case_index, ir in enumerate(_case_irs(condition, demo), start=1):
                for query_index, query in enumerate(_condition_queries(condition), start=1):
                    context: Optional[Dict[str, Any]]
                    context = {"intermediate_representation": ir}
                    prompt_set.append(
                        {
                            "repeat": repeat,
                            "case_id": case_index,
                            "query_id": query_index,
                            "query": query,
                            "fault_type": ir.fault_info.fault_type,
                        }
                    )
                    start = time.perf_counter()
                    error = ""
                    try:
                        response = demo.llm.generate(query, context)
                    except Exception as exc:  # pragma: no cover - safety record
                        response = ""
                        error = repr(exc)
                    latency = time.perf_counter() - start
                    unsupported = _unsupported_claims(response, ir, checker_enabled)
                    record = {
                        "condition": condition,
                        "repeat": repeat,
                        "case_id": case_index,
                        "query_id": query_index,
                        "query": query,
                        "response": response,
                        "latency_seconds": latency,
                        "checker_enabled": checker_enabled,
                        "unsupported_claims": unsupported,
                        "error": error,
                    }
                    records.append(record)
                    response_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    (inputs_root / "prompt_set.json").write_text(
        json.dumps(prompt_set, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    latencies = [record["latency_seconds"] for record in records]
    prompt_count = len(records)
    failures = [record for record in records if record["error"] or not record["response"]]
    unsupported = [record for record in records if record["unsupported_claims"]]
    metrics = {
        "paper_id": "LLM_Explainable_FD_Toolkit",
        "protocol_id": "llm_evidence_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "acceptance_blocker": "smoke runner only; no same-protocol GPU reviewer evidence",
        "seed": seed,
        "sample_count": len(demo.demo_cases),
        "prompt_count": prompt_count,
        "checker_enabled": checker_enabled,
        "latency_p50_seconds": float(np.percentile(latencies, 50)) if latencies else 0.0,
        "latency_p95_seconds": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "unsupported_claim_rate_proxy": len(unsupported) / prompt_count if prompt_count else 0.0,
        "failure_rate": len(failures) / prompt_count if prompt_count else 0.0,
        "response_records_path": str(responses_path),
    }
    ended_at = datetime.now().isoformat()
    run_meta = {
        "paper_id": "LLM_Explainable_FD_Toolkit",
        "protocol_id": "llm_evidence_smoke",
        "condition_id": condition,
        "accepted_evidence": False,
        "seed": seed,
        "command": "python " + " ".join(sys.argv),
        "working_directory": str(Path.cwd()),
        "submodule_commit": _git_commit(),
        "input_artifact_paths": [str(inputs_root / "prompt_set.json")],
        "output_artifact_paths": [str(responses_path)],
        "log_path": str(logs_root),
        "metrics_path": str(condition_root / "metrics.json"),
        "started_at": started_at,
        "ended_at": ended_at,
        "runtime_seconds": time.perf_counter() - started_perf,
        "batch_size_or_prompt_batch_size": 1,
        "precision_or_quantization": "local-template-fp32-smoke",
        "dataset_split_or_prompt_set": "demo_cases",
        "oom_or_failure_reason": "",
        "cuda": _cuda_metadata(),
    }

    (condition_root / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (condition_root / "run_meta.yaml").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"{condition}: prompts={prompt_count}, "
        f"unsupported_proxy={metrics['unsupported_claim_rate_proxy']:.3f}, "
        f"p95={metrics['latency_p95_seconds']:.6f}s"
    )


def _conditions(selection: str) -> Sequence[str]:
    if selection == "all":
        return (
            "grounded",
            "no_checker",
            "no_domain_context",
            "latency_short",
            "latency_long",
        )
    return (selection,)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate non-accepted LLM evidence smoke artifacts."
    )
    parser.add_argument(
        "--condition",
        choices=[
            "all",
            "grounded",
            "no_checker",
            "no_domain_context",
            "latency_short",
            "latency_long",
        ],
        default="all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/llm_evidence/demo_smoke"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    for condition in _conditions(args.condition):
        _run_condition(condition, args.output, args.seed, args.repeats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
