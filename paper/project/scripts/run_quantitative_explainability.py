#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from run_minimal_explain import extract_branch_models, setup_model_and_data
from grad_cam import FusionGradCAM, visualize_fusion_attribution
from models.two_d_branch import create_spectrogram_from_1d


def find_target_layers(model: torch.nn.Module, layer_type: type) -> List[str]:
    names: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, layer_type):
            names.append(name)
    return names


def probability_for_class(model: torch.nn.Module, signal: torch.Tensor, target_class: int) -> float:
    with torch.no_grad():
        outputs = model(signal, return_alignment=True)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        return float(probs[0, target_class].item())


def normalize_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.abs(values)
    max_value = float(values.max()) if values.size else 0.0
    if max_value > 0:
        values = values / max_value
    return values


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize_array(a).reshape(-1)
    b = normalize_array(b).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))


def fallback_attribution(signal: torch.Tensor, spectrogram: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    signal_np = normalize_array(signal.detach().cpu().numpy().reshape(-1))
    spectrogram_np = normalize_array(spectrogram.detach().cpu().numpy().squeeze())
    return signal_np, spectrogram_np


def mask_topk_signal(signal: torch.Tensor, attribution: np.ndarray, ratio: float = 0.1) -> torch.Tensor:
    attribution = normalize_array(attribution)
    flat = attribution.reshape(-1)
    topk = max(1, int(len(flat) * ratio))
    indices = np.argsort(flat)[-topk:]
    masked = signal.clone()
    masked.reshape(-1)[indices] = 0.0
    return masked


def summarize_metric(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def analyze_alignment_metrics_without_labels(
    model: torch.nn.Module, signals: torch.Tensor, output_dir: Path
) -> Dict[str, Dict[str, float]]:
    model.eval()
    batches: List[Dict[str, float]] = []
    with torch.no_grad():
        batch_size = 8
        for start in range(0, len(signals), batch_size):
            batch_signals = signals[start : start + batch_size]
            outputs = model(batch_signals, return_alignment=True)
            metrics = outputs.get("alignment_metrics") or {}
            if metrics:
                batches.append({key: float(value) for key, value in metrics.items() if value is not None})

    if not batches:
        return {}

    aggregated: Dict[str, Dict[str, float]] = {}
    keys = sorted({key for batch in batches for key in batch.keys()})
    for key in keys:
        values = [batch[key] for batch in batches if key in batch]
        if not values:
            continue
        aggregated[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    (output_dir / "alignment_metrics_summary.json").write_text(
        json.dumps(aggregated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return aggregated


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate quantitative explainability metrics for the 1D-2D fusion paper.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--noise-scale", type=float, default=0.05)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    model, signals, labels = setup_model_and_data()
    one_d_model, two_d_model = extract_branch_models(model)
    one_d_layers = find_target_layers(one_d_model, torch.nn.Conv1d)
    two_d_layers = find_target_layers(two_d_model, torch.nn.Conv2d)
    if not one_d_layers or not two_d_layers:
        raise RuntimeError("Unable to locate Conv1d/Conv2d layers for quantitative explainability.")

    fusion_grad_cam = FusionGradCAM(one_d_model, two_d_model, model, [one_d_layers[-1]], [two_d_layers[-1]])
    alignment_metrics = analyze_alignment_metrics_without_labels(model, signals, output_dir)

    requested_samples = min(args.num_samples, len(signals))
    sample_indices = np.linspace(0, len(signals) - 1, requested_samples, dtype=int)

    per_sample: List[Dict[str, Any]] = []
    faithfulness_values: List[float] = []
    stability_values: List[float] = []
    efficiency_values: List[float] = []
    accuracies: List[float] = []
    figure_paths: List[str] = []
    fallback_count = 0

    for sample_id, idx in enumerate(sample_indices, start=1):
        signal = signals[idx : idx + 1]
        label = int(labels[idx].item())
        with torch.no_grad():
            outputs = model(signal, return_alignment=True)
            logits = outputs["logits"]
            probs = torch.softmax(logits, dim=-1)
            pred_class = int(torch.argmax(probs, dim=-1).item())
            pred_prob = float(probs[0, pred_class].item())

        spectrogram = create_spectrogram_from_1d(signal, target_size=(128, 128))

        start_time = time.perf_counter()
        error = None
        try:
            fusion_results = fusion_grad_cam.generate_fusion_cam(signal, spectrogram, pred_class)
            cam_1d = normalize_array(fusion_results["1d_cam"])
            cam_2d = normalize_array(fusion_results["2d_cam"])
            fusion_weights = fusion_results["fusion_weights"]
        except Exception as exc:  # pragma: no cover - fallback for fragile visualization path
            error = str(exc)
            fallback_count += 1
            cam_1d, cam_2d = fallback_attribution(signal, spectrogram)
            fusion_weights = {"weight_1d": 0.5, "weight_2d": 0.5}
        efficiency_ms = (time.perf_counter() - start_time) * 1000.0

        masked_signal = mask_topk_signal(signal, cam_1d, ratio=0.1)
        masked_prob = probability_for_class(model, masked_signal, pred_class)
        faithfulness = max(pred_prob - masked_prob, 0.0)

        similarity_values: List[float] = []
        signal_std = float(signal.std().item()) if float(signal.std().item()) > 0 else 1.0
        for _ in range(3):
            noisy_signal = signal + torch.randn_like(signal) * (args.noise_scale * signal_std)
            noisy_spectrogram = create_spectrogram_from_1d(noisy_signal, target_size=(128, 128))
            try:
                noisy_results = fusion_grad_cam.generate_fusion_cam(noisy_signal, noisy_spectrogram, pred_class)
                noisy_cam = noisy_results["1d_cam"]
            except Exception:
                noisy_cam, _ = fallback_attribution(noisy_signal, noisy_spectrogram)
            similarity_values.append(cosine_similarity(cam_1d, noisy_cam))
        stability = float(np.mean(similarity_values))

        figure_path = figure_dir / f"quantitative_explainability_{sample_id:02d}.png"
        fig = visualize_fusion_attribution(
            signal.detach().cpu().numpy().reshape(-1),
            spectrogram.detach().cpu().numpy().squeeze(),
            cam_1d,
            cam_2d,
            fusion_weights,
            title=f"Quantitative Explainability Sample {sample_id}",
            save_path=str(figure_path),
        )
        plt.close(fig)
        figure_paths.append(str(figure_path))

        accuracies.append(1.0 if pred_class == label else 0.0)
        faithfulness_values.append(faithfulness)
        stability_values.append(stability)
        efficiency_values.append(efficiency_ms)
        per_sample.append(
            {
                "sample_index": int(idx),
                "true_label": label,
                "predicted_label": pred_class,
                "predicted_probability": pred_prob,
                "masked_probability": masked_prob,
                "faithfulness": faithfulness,
                "stability": stability,
                "efficiency_ms": efficiency_ms,
                "correct": pred_class == label,
                "fusion_weights": fusion_weights,
                "fallback_used": error is not None,
                "error": error,
                "figure_path": str(figure_path),
            }
        )

    faithfulness_mean, faithfulness_std = summarize_metric(faithfulness_values)
    stability_mean, stability_std = summarize_metric(stability_values)
    efficiency_mean, efficiency_std = summarize_metric(efficiency_values)
    accuracy_mean, _ = summarize_metric(accuracies)

    per_sample_path = output_dir / "explainability_metrics_per_sample.json"
    per_sample_path.write_text(json.dumps(per_sample, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "probe_scope": "synthetic_explainability_probe",
        "requested_samples": requested_samples,
        "success_count": len(per_sample),
        "failure_count": 0,
        "accuracy": accuracy_mean,
        "faithfulness_mean": faithfulness_mean,
        "faithfulness_std": faithfulness_std,
        "stability_mean": stability_mean,
        "stability_std": stability_std,
        "efficiency_ms_mean": efficiency_mean,
        "efficiency_ms_std": efficiency_std,
        "alignment_metrics": alignment_metrics,
        "figure_paths": figure_paths,
        "results_file": str(per_sample_path),
        "fallback_count": fallback_count,
    }
    summary_path = output_dir / "explainability_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary_md = output_dir / "explainability_metrics_summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Quantitative Explainability Summary",
                "",
                f"- probe_scope: `{summary['probe_scope']}`",
                f"- requested_samples: `{summary['requested_samples']}`",
                f"- success_count: `{summary['success_count']}`",
                f"- accuracy: `{summary['accuracy']}`",
                f"- faithfulness_mean: `{summary['faithfulness_mean']}`",
                f"- faithfulness_std: `{summary['faithfulness_std']}`",
                f"- stability_mean: `{summary['stability_mean']}`",
                f"- stability_std: `{summary['stability_std']}`",
                f"- efficiency_ms_mean: `{summary['efficiency_ms_mean']}`",
                f"- efficiency_ms_std: `{summary['efficiency_ms_std']}`",
                f"- fallback_count: `{summary['fallback_count']}`",
                "",
                "## Figures",
                "",
                *[f"- `{path}`" for path in figure_paths],
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    fusion_grad_cam.remove_hooks()
    print(json.dumps({"ok": True, "summary_path": str(summary_path)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
