"""Config binding for the existing frozen-source and one-step Tent runtimes.

No training population, new adaptation algorithm, optimizer, Trainer or result schema.
The target archive is prepared upstream; this task never fits target preprocessing.
"""
from collections.abc import Mapping
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from phmfactory.source_stream import run_source_only_stream
from phmfactory.tent_stream import run_tent_stream
from src.config_schema import AdaptationProtocolConfig
from src.data_factory.target_stream import build_target_stream
from src.model_factory import build_model
from src.task_factory.Components.metrics import get_metrics, prepare_metric_inputs
from src.utils.run_summary import normalize_metric_result, write_run_summary

PHASES = ("validate", "evaluate")


def validate_config(config):
    """Reject unsupported learning semantics before reading data or source weights."""
    allowed = {
        "environment": {"project", "output_dir", "seed", "iterations", "notes"},
        "data": {"path", "batch_size", "order", "preprocessing"},
        "task": {"type", "name", "execution", "protocol", "adaptation", "metrics"},
        "trainer": {"device", "devices"},
    }
    if set(config) != {"pipeline", "environment", "data", "model", "task", "trainer"}:
        raise ValueError("stream config requires only pipeline and the five Factory blocks")
    for block, keys in allowed.items():
        values = config[block]
        if not isinstance(values, Mapping) or set(values) - keys:
            raise ValueError(f"unsupported {block} settings for the bounded target-stream task")
    env, data, model, task, trainer = (config[k] for k in
                                      ("environment", "data", "model", "task", "trainer"))
    if not isinstance(model, Mapping) or any(
            not isinstance(model.get(key), str) or not model[key].isidentifier()
            for key in ("type", "name")):
        raise ValueError("model.type/name must select an explicit Model Factory module")
    if config["pipeline"] != "Pipeline_01_Fault_Diagnosis":
        raise ValueError("TTA/stream requires the classification pipeline declaration")
    if (type(env.get("seed")) is not int or not 0 <= env["seed"] < 2**32
            or type(env.get("iterations")) is not int or env["iterations"] != 1):
        raise ValueError("declare one run and an integer seed in [0, 2**32)")
    if trainer != {"device": "cpu", "devices": 1} or type(trainer.get("devices")) is not int:
        raise ValueError("this config entry supports only explicit single-device CPU execution")
    if (type(data.get("batch_size")) is not int or data["batch_size"] < 1
            or data.get("order") != "archive_rows"
            or data.get("preprocessing") != "precomputed"):
        raise ValueError("declare a positive batch_size, archive_rows order and precomputed inputs")
    for value, name in ((data.get("path"), "data.path"),
                        (model.get("weights_path"), "model.weights_path")):
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"{name} must name an explicit file without surrounding whitespace")
    if (type(model.get("num_classes")) is not int or model["num_classes"] < 2
            or model.get("weights_strict", True) is not True):
        raise ValueError("explicit source num_classes>=2 and strict source weights are required")
    if (task.get("type"), task.get("name"), task.get("execution")) != ("TTA", "stream", "research"):
        raise ValueError("requires task.type=TTA, name=stream, execution=research")
    protocol = AdaptationProtocolConfig.model_validate(task.get("protocol"), strict=True)
    for key, expected in dict(source_access="checkpoint_only", target_label_access="none",
                              timing="predict_then_update", state_persistence="persistent",
                              label_space="closed_set").items():
        if getattr(protocol, key) != expected:
            raise ValueError(f"stream requires protocol.{key}={expected!r}")
    adaptation = task.get("adaptation")
    if not isinstance(adaptation, Mapping) or adaptation.get("method") not in {"none", "tent"}:
        raise ValueError("supported adaptation methods are only none and tent")
    method = adaptation["method"]
    if method == "none":
        if set(adaptation) != {"method"} or protocol.regime != "source_only":
            raise ValueError("method=none requires source_only and no adaptation hyperparameters")
    else:
        if set(adaptation) != {"method", "learning_rate"} or protocol.regime not in {"online_tta", "continual_tta"}:
            raise ValueError("tent requires online/continual_tta and one explicit learning_rate")
        lr = adaptation["learning_rate"]
        if type(lr) not in (int, float) or not np.isfinite(lr) or lr <= 0:
            raise ValueError("Tent learning_rate must be finite, positive and selected without target labels")
    names = task.get("metrics")
    if (not isinstance(names, list) or not names or len(set(names)) != len(names)
            or any(name not in {"acc", "f1", "precision", "recall"} for name in names)):
        raise ValueError("metrics must be distinct supported classification metrics")
    return protocol


def execute(config, phase, output):
    """Validate or evaluate one target-only stream through the existing public command.

    Validation checks settings, not data provenance, architecture compatibility or E3.
    Output appears only after complete evaluation; existing directories are never reused.
    Algorithm checkpoint payload/resume remains owned by the existing Tent interface;
    this entry does not introduce a second checkpoint lifecycle.
    """
    if phase not in PHASES:
        raise ValueError(f"TTA/stream supports phases {PHASES}, not {phase!r}")
    protocol = validate_config(config)
    if phase == "validate":
        return {"status": "configuration_validated", "data_qualified": False,
                "execution_verified": False}
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"use a new output directory: {output}")
    source = Path(config["model"]["weights_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    seed = config["environment"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # ModelFactory owns the architecture; its strict loader is called once by B01/B02.
    model_args = {k: v for k, v in config["model"].items()
                  if k not in {"weights_path", "weights_strict"}}
    model = build_model(SimpleNamespace(**model_args), None).cpu().eval()
    loader = build_target_stream(config["data"]["path"], batch_size=config["data"]["batch_size"])
    k = config["model"]["num_classes"]
    # These are source-ontology declarations, not acquired or target-derived records.
    ontology = pd.DataFrame({"Name": ["target"] * k, "Dataset_id": [0] * k,
                             "Label": list(range(k))})
    metrics = get_metrics(config["task"]["metrics"], ontology, loss_name="CE")["target"]
    saved_logits = []

    def evaluate(logits, view):
        target = view["y"]
        if logits.shape[1] != k or torch.any((target < 0) | (target >= k)):
            raise ValueError("prediction/target must use the fixed source label ontology")
        for name in config["task"]["metrics"]:
            pred, y = prepare_metric_inputs(name, logits, target, loss_name="CE")
            metrics[f"test_{name}"].update(pred, y)
        saved_logits.append(logits.detach().cpu().numpy().copy())

    adaptation = config["task"]["adaptation"]
    if adaptation["method"] == "none":
        count = run_source_only_stream(model, loader, protocol,
                                       checkpoint_path=source, evaluate=evaluate)
    else:
        adapter = run_tent_stream(model, loader, protocol, checkpoint_path=source,
                                  learning_rate=adaptation["learning_rate"], evaluate=evaluate)
        count = adapter.num_samples
    result = normalize_metric_result({f"test_{name}": metrics[f"test_{name}"].compute()
                                      for name in config["task"]["metrics"]})
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame([result]).to_csv(output / "all_results.csv", index=False)
    write_run_summary(output / "run_summary.json", [result], [seed])
    (output / "config.yaml").write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    np.savez(output / "predictions.npz", logits=np.concatenate(saved_logits),
             sample_id=np.arange(count, dtype=np.int64))
    return {"status": "completed", "result_dir": str(output),
            "test_metrics": str(output / "all_results.csv"),
            "run_summary": str(output / "run_summary.json"), "primary_metrics": result,
            "source_checkpoint": str(source), "num_samples": count,
            "benchmark_ready": False}
