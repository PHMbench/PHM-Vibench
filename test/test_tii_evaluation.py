"""Synthetic CSV fixtures for fail-fast query isolation, not industrial results."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.task_factory.Components.tii_evaluation import evaluate_query_predictions


class QueryEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.records = pd.DataFrame([
            dict(dataset="industrial", recording_id=r, group=g, true_label=y)
            for r, g, y in [
                ("s0", "gs0", 0), ("s1", "gs1", 1), ("same_support", "gs0", 0),
                ("q0", "g0", 0), ("q1", "g1", 1), ("q2", "g0", 0),
            ]
        ])
        context = dict(target="industrial", fold="held-out", episode=1729,
                       support_split_id="episode-1729-support", query_split_id="episode-1729-query")
        self.support = pd.DataFrame([
            dict(row, **context, role="support")
            for row in self.records.iloc[:2].to_dict("records")
        ])
        self.query = pd.DataFrame([
            dict(row, **context, role="query")
            for row in self.records.iloc[3:].to_dict("records")
        ])
        windows = []
        for method in ("S", "U"):
            for seed in (0, 1):
                for row in self.query.to_dict("records"):
                    for start in ([0, 10] if row["recording_id"] == "q0" else [0]):
                        windows.append(dict(row, method=method, seed=seed, channel="0",
                                            window_start=start, window_end=start+10,
                                            checkpoint=f"{method}/seed-{seed}/frozen.ckpt"))
        self.expected = pd.DataFrame(windows)
        self.predictions = self.expected.copy()
        logits = [[float(row.seed), -float(row.window_start)/10]
                  for row in self.predictions.itertuples()]
        self.predictions["logits"] = [json.dumps(x) for x in logits]
        self.predictions["probabilities"] = [json.dumps((np.exp(x)/np.exp(x).sum()).tolist())
                                             for x in logits]
        self.write_files()

    def write_files(self):
        for name in ("records", "support", "query", "expected", "predictions"):
            getattr(self, name).to_csv(self.root / f"{name}.csv", index=False)

    def evaluate(self, **changes):
        arguments = dict(records_file=self.root/"records.csv", support_file=self.root/"support.csv",
                         query_file=self.root/"query.csv", expected_windows_file=self.root/"expected.csv",
                         local_class_map={"industrial": [0, 1]})
        arguments.update(changes)
        return evaluate_query_predictions(self.root/"predictions.csv", **arguments)

    def assert_rejected_before_nll(self, pattern):
        self.write_files()
        with patch("src.task_factory.Components.tii_evaluation.logsumexp",
                   side_effect=AssertionError("NLL must not run on contaminated input")) as nll:
            with self.assertRaisesRegex(ValueError, pattern):
                self.evaluate()
            nll.assert_not_called()

    def test_preserves_multiple_windows_and_averages_windows_then_seeds(self):
        windows, groups = self.evaluate()
        self.assertEqual(len(windows), 16)
        self.assertEqual(len(groups), 4)
        for row in windows.itertuples():
            logits = json.loads(row.logits)
            expected_loss = np.logaddexp(*logits) - logits[row.true_label]
            self.assertAlmostEqual(row.nll, expected_loss, places=14)
        for row in groups.itertuples():
            selected = windows[(windows.method == row.method) & (windows.group == row.group)]
            independent = np.mean([part.nll.to_numpy().mean() for _, part in selected.groupby("seed")])
            self.assertAlmostEqual(row.nll, independent, places=14)
            self.assertEqual(row.seed_count, 2)
        self.assertEqual(groups.loc[groups.group == "g0", "prediction_count"].tolist(), [6, 6])

    def test_zero_probability_keeps_large_logit_nll_without_floor(self):
        row = self.predictions.index[self.predictions.recording_id == "q1"][-1]
        self.predictions.loc[row, "logits"] = "[0, -1000]"
        self.predictions.loc[row, "probabilities"] = "[1, 0]"
        self.write_files()
        windows, _ = self.evaluate()
        self.assertEqual(windows.loc[row, "nll"], 1000)

    def test_large_common_logit_offset_keeps_nonzero_loss(self):
        self.predictions.loc[0, "logits"] = "[1e100, 1e100]"
        self.predictions.loc[0, "probabilities"] = "[0.5, 0.5]"
        self.write_files()
        windows, _ = self.evaluate()
        self.assertAlmostEqual(windows.loc[0, "nll"], np.log(2), places=14)

    def test_missing_group_or_record_id_fails_before_nll(self):
        for field in ("group", "recording_id"):
            for bad in (None, np.nan, np.inf, -np.inf, "", "   "):
                with self.subTest(field=field, bad=bad):
                    original = self.predictions.loc[0, field]
                    self.predictions.loc[0, field] = bad
                    self.assert_rejected_before_nll(field)
                    self.predictions.loc[0, field] = original

    def test_support_row_in_predictions_fails_before_nll(self):
        self.predictions.loc[0, ["recording_id", "group", "role"]] = ["s0", "gs0", "support"]
        self.assert_rejected_before_nll("query role")

    def test_another_record_from_support_group_fails_even_with_forged_query_split(self):
        for table in (self.query, self.expected, self.predictions):
            select = table.recording_id == "q0"
            table.loc[select, ["recording_id", "group"]] = ["same_support", "gs0"]
        self.assert_rejected_before_nll("support group contamination")

    def test_missing_or_unknown_role_fails_before_nll(self):
        self.predictions = self.predictions.drop(columns="role")
        self.assert_rejected_before_nll("missing columns.*role")
        self.predictions["role"] = "validation"
        self.assert_rejected_before_nll("query role")

    def test_missing_split_file_fails_before_nll(self):
        for name in ("support", "query"):
            with self.subTest(split=name):
                self.write_files()
                (self.root/f"{name}.csv").unlink()
                with patch("src.task_factory.Components.tii_evaluation.logsumexp") as nll:
                    with self.assertRaisesRegex(ValueError, "missing inventory/split file"):
                        self.evaluate()
                    nll.assert_not_called()

    def test_both_arms_missing_query_group_fail_before_nll(self):
        self.predictions = self.predictions[self.predictions.group != "g1"]
        self.assert_rejected_before_nll("complete predeclared query window inventory")

    def test_both_arms_missing_query_window_fail_before_nll(self):
        self.predictions = self.predictions[self.predictions.window_start != 10]
        self.assert_rejected_before_nll("complete predeclared query window inventory")

    def test_duplicate_prediction_composite_key_fails_before_nll(self):
        self.predictions = pd.concat([self.predictions, self.predictions.iloc[:1]], ignore_index=True)
        self.assert_rejected_before_nll("predictions: duplicate composite key")

    def test_original_records_must_be_unique_but_prediction_windows_need_not_be(self):
        self.records = pd.concat([self.records, self.records.iloc[:1]], ignore_index=True)
        self.assert_rejected_before_nll("records: duplicate composite key")

    def test_unknown_dataset_group_label_checkpoint_and_split_fail(self):
        for field, value in (("dataset", "other"), ("group", "other"), ("true_label", 1),
                             ("checkpoint", "other.ckpt"), ("support_split_id", "other"),
                             ("query_split_id", "other"), ("channel", "other"),
                             ("target", "other"), ("fold", "other"), ("seed", 99),
                             ("episode", 1749), ("method", "H"), ("window_end", 11)):
            with self.subTest(field=field):
                original = self.predictions.loc[0, field]
                self.predictions.loc[0, field] = value
                self.assert_rejected_before_nll("provenance|independent query split|window inventory")
                self.predictions.loc[0, field] = original

    def test_invalid_last_prediction_prevents_all_nll(self):
        invalid = [
            ("logits", "[NaN, 0]", "finite vectors"),
            ("logits", "[Infinity, 0]", "finite vectors"),
            ("logits", "[0, 0, 0]", "local class map"),
            ("logits", "null", "numeric JSON"),
            ("probabilities", "[0.3, 0.3]", "summing to one"),
            ("probabilities", "[-0.1, 1.1]", "nonnegative"),
            ("probabilities", "[NaN, 1]", "finite vectors"),
            ("probabilities", "[0.5, 0.5]", "disagree with logits"),
        ]
        row = self.predictions.index[-1]
        for field, value, message in invalid:
            with self.subTest(field=field, value=value):
                original = self.predictions.loc[row, field]
                self.predictions.loc[row, field] = value
                try:
                    self.assert_rejected_before_nll(message)
                finally:
                    self.predictions.loc[row, field] = original

    def test_class_map_order_controls_true_label_column(self):
        # Reversing both column arrays and the declared map preserves NLL.
        baseline, _ = self.evaluate()
        for column in ("logits", "probabilities"):
            self.predictions[column] = [json.dumps(json.loads(value)[::-1])
                                        for value in self.predictions[column]]
        self.write_files()
        reversed_columns, _ = self.evaluate(local_class_map={"industrial": [1, 0]})
        np.testing.assert_array_equal(reversed_columns.nll, baseline.nll)

    def test_predeclared_runs_need_matched_complete_query_windows(self):
        self.expected = self.expected[~((self.expected.method == "S") &
                                       (self.expected.window_start == 10))]
        self.assert_rejected_before_nll("share the same query window inventory")

    def test_numeric_zero_and_leading_zero_identities_survive_csv(self):
        for table in (self.records, self.support, self.query, self.expected, self.predictions):
            table.loc[table.group == "g0", "group"] = "0"
            table.loc[table.recording_id == "q0", "recording_id"] = "001"
        self.write_files()
        windows, groups = self.evaluate()
        self.assertIn("001", windows.recording_id.tolist())
        self.assertIn("0", groups.group.tolist())


if __name__ == "__main__":
    unittest.main()
