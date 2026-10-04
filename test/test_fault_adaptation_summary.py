"""Native-array summary checks using synthetic fixtures, not PHM results."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.task_factory.task.GFS.fault_adaptation import summary


class SummaryChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p09-summary-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "episode-0000"
        self.folder.mkdir()
        self.episode = {"dataset": "synthetic", "fold": "heldout", "shots": 1, "seed": 0, "draw": 0,
                        "novel_classes": [2, 3]}
        self.cfg = {"query_evaluated": True, "actual_arms": list(summary.ARMS["E1"]), "episodes": [self.episode]}
        self.status = {"status": "completed", "experiment": "E1", "query_evaluated": True}
        self.settings = {"steps": {arm: 0 if arm == "A4" else 1 for arm in summary.ARMS["E1"]},
                         "trajectory_steps": {arm: [0, 1] for arm in ("A2", "A3")}}
        self.write_json("status.json", self.status)
        self.write_json("config.resolved.json", self.cfg)
        self.write_json("episode-0000/source.settings.json", {"base_classes": [0, 1]})
        self.write_json("episode-0000/adaptation.settings.json", self.settings)
        labels = np.array([0, 0, 0, 1, 2, 3])
        groups = np.array([10, 10, 11, 20, 30, 40])
        source = np.array([[-1., 0.], [-1., 0.], [-1., 0.], [0., -1.], [1., 0.], [0., 1.]])
        self.metrics, self.predictions = [], []
        for arm in summary.ARMS["E1"]:
            step = self.settings["steps"][arm]
            logits = np.full((len(labels), 4), -1.)
            logits[np.arange(len(labels)), labels] = 2.
            logits[4:, 0] = 0.
            if arm != "A7":
                logits[2, 2] = 3.
            np.savez(self.folder / f"{arm}-{step}.evaluation.npz", label=labels, group=groups,
                     class_order=np.arange(4), logits=logits, representation=source, source_representation=source,
                     source_base_logits=logits[:, :2].copy())
            identity = {key: self.episode[key] for key in summary.IDENTITY}
            self.metrics.append({**identity, "arm": arm, "step": step,
                                 "base_acc": 1. if arm == "A7" else .75, "novel_acc": 1.,
                                 "joint_acc": 1. if arm == "A7" else .875,
                                 "signed_base_accuracy_loss": 0. if arm == "A7" else .25,
                                 "harmonic": 1. if arm == "A7" else 6 / 7,
                                 "base_only_acc": 1., "a0_base_acc": 1., "intrusion": 0. if arm == "A7" else .25})
            for index, label in enumerate(labels):
                self.predictions.append({**identity, "arm": arm, "step": step, "observation": index,
                                         "label": int(label), "group": int(groups[index]),
                                         "joint_prediction": int(logits[index].argmax()),
                                         "base_prediction": int(logits[index, :2].argmax()),
                                         "source_base_prediction": 1 if index == 3 else 0})
        self.write_rows("metrics.csv", self.metrics)
        self.write_rows("predictions.csv", self.predictions)

    def write_json(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def write_rows(self, name, rows):
        with (self.root / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)

    def test_independent_group_balance_and_paired_effects(self):
        destination = summary.summarize(self.root)
        rows = summary._read_csv(destination / "metrics.recomputed.csv")
        self.assertEqual(len(rows), 5)
        a4 = next(row for row in rows if row["arm"] == "A4")
        self.assertEqual(float(a4["base_acc"]), .75)  # Unequal windows, equal acquisition weights.
        self.assertEqual(float(a4["intrusion"]), .25)
        effects = summary._read_csv(destination / "paired_effects.csv")
        self.assertEqual(len(effects), 4)
        self.assertTrue(all(float(row["base_acc_difference"]) == .25 for row in effects))
        with self.assertRaises(FileExistsError):
            summary.summarize(self.root)

    def test_improved_old_class_accuracy_retains_negative_signed_loss(self):
        for path in self.folder.glob("*.evaluation.npz"):
            with np.load(path) as source:
                arrays = {key: source[key] for key in source.files}
            arrays["source_base_logits"] = np.tile([1., 0.], (len(arrays["label"]), 1))
            np.savez(path, **arrays)
        for row in self.predictions:
            row["source_base_prediction"] = 0
        for row in self.metrics:
            row["a0_base_acc"] = .5
            row["signed_base_accuracy_loss"] = .5 - row["base_acc"]
        self.write_rows("predictions.csv", self.predictions)
        self.write_rows("metrics.csv", self.metrics)
        destination = summary.summarize(self.root)
        rows = summary._read_csv(destination / "metrics.recomputed.csv")
        method = next(row for row in rows if row["arm"] == "A7")
        self.assertEqual(float(method["signed_base_accuracy_loss"]), -.5)

    def test_failed_and_query_blind_runs_do_not_emit_summaries(self):
        for change in ({"status": "failed"}, {"query_evaluated": False}):
            with self.subTest(change=change):
                self.write_json("status.json", {**self.status, **change})
                with self.assertRaisesRegex(ValueError, "completed query-evaluated"):
                    summary.summarize(self.root)
                self.assertFalse((self.root / "summary").exists())

    def test_missing_endpoint_cannot_be_hidden_by_removing_both_csv_groups(self):
        self.write_rows("metrics.csv", [row for row in self.metrics if row["arm"] != "A2"])
        self.write_rows("predictions.csv", [row for row in self.predictions if row["arm"] != "A2"])
        with self.assertRaisesRegex(ValueError, "complete grid"):
            summary.summarize(self.root)

    def test_tuning_subset_cannot_masquerade_as_complete_comparison(self):
        self.write_json("config.resolved.json", {**self.cfg, "actual_arms": ["A7"]})
        with self.assertRaisesRegex(ValueError, "complete formal"):
            summary.summarize(self.root)

    def test_duplicate_and_missing_observations_are_rejected(self):
        for rows in (self.predictions + [self.predictions[0]], self.predictions[1:]):
            with self.subTest(length=len(rows)):
                self.write_rows("predictions.csv", rows)
                with self.assertRaisesRegex(ValueError, "prediction observations"):
                    summary.summarize(self.root)

    def test_source_base_prediction_is_checked_against_native_unscaled_logits(self):
        self.predictions[0]["source_base_prediction"] = 1
        self.write_rows("predictions.csv", self.predictions)
        with self.assertRaisesRegex(ValueError, "source-base mapping"):
            summary.summarize(self.root)

    def test_native_prediction_and_reported_metric_tampering_are_rejected(self):
        self.predictions[0]["joint_prediction"] = 3
        self.write_rows("predictions.csv", self.predictions)
        with self.assertRaisesRegex(ValueError, "native scores"):
            summary.summarize(self.root)
        self.predictions[0]["joint_prediction"] = 0
        self.write_rows("predictions.csv", self.predictions)
        self.metrics[0]["base_acc"] = 1.
        self.write_rows("metrics.csv", self.metrics)
        with self.assertRaisesRegex(ValueError, "primary metrics"):
            summary.summarize(self.root)

    def test_saved_arrays_must_share_query_population(self):
        path = self.folder / "A2-1.evaluation.npz"
        with np.load(path) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["group"][0] = 999
        np.savez(path, **arrays)
        with self.assertRaisesRegex(ValueError, "share native query"):
            summary.summarize(self.root)

    def test_a0_uses_unscaled_native_scores_when_scaled_scores_underflow(self):
        for path in self.folder.glob("*.evaluation.npz"):
            with np.load(path) as source:
                arrays = {key: source[key] for key in source.files}
            arrays["logits"] = np.zeros_like(arrays["logits"])
            np.savez(path, **arrays)
        for row in self.predictions:
            row["joint_prediction"] = row["base_prediction"] = 0
        for row in self.metrics:
            row.update(base_acc=.5, novel_acc=0., joint_acc=.25, signed_base_accuracy_loss=.5, harmonic=0., base_only_acc=.5, intrusion=0.)
        self.write_rows("predictions.csv", self.predictions)
        self.write_rows("metrics.csv", self.metrics)
        destination = summary.summarize(self.root)
        rows = summary._read_csv(destination / "metrics.recomputed.csv")
        self.assertTrue(all(float(row["a0_base_acc"]) == 1. for row in rows))

    def test_e2_checks_every_declared_checkpoint_without_selecting_one(self):
        self.status["experiment"] = "E2"
        self.cfg["actual_arms"] = list(summary.ARMS["E2"])
        self.write_json("status.json", self.status)
        self.write_json("config.resolved.json", self.cfg)
        self.metrics = [row for row in self.metrics if row["arm"] != "A6"]
        self.predictions = [row for row in self.predictions if row["arm"] != "A6"]
        for arm in ("A2", "A3"):
            (self.folder / f"{arm}-0.evaluation.npz").write_bytes((self.folder / f"{arm}-1.evaluation.npz").read_bytes())
            self.metrics += [{**row, "step": 0} for row in list(self.metrics) if row["arm"] == arm]
            self.predictions += [{**row, "step": 0} for row in list(self.predictions) if row["arm"] == arm]
        self.write_rows("metrics.csv", self.metrics)
        self.write_rows("predictions.csv", self.predictions)
        destination = summary.summarize(self.root)
        self.assertEqual(len(summary._read_csv(destination / "metrics.recomputed.csv")), 6)
        self.assertEqual(summary._read_csv(destination / "paired_effects.csv"), [])


if __name__ == "__main__":
    unittest.main()
