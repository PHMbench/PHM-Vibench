"""CPU tests for the frozen E1 four-arm model binding."""

from __future__ import annotations

import unittest

import torch

from src.p08_evidence.e1_data import EVALUATION_RATES_HZ, expected_rate_points
from src.p08_evidence.e1_model import arm_spec, build_model, pretraining_loss


class E1ModelTests(unittest.TestCase):
    def test_arm_switches_are_distinct_and_locked(self) -> None:
        dn = arm_spec("P08-DN", duration_ms=15.0)
        method = arm_spec("P08-M", duration_ms=15.0)
        bg = arm_spec(
            "P08-BG",
            global_resample_numerator_hz=51_200,
            global_resample_denominator=3,
        )
        nc = arm_spec("P08-NC")
        self.assertFalse(dn.use_prompt)
        self.assertTrue(method.use_prompt)
        self.assertEqual(bg.fixed_raw_token_points, 256)
        self.assertEqual(nc.fixed_raw_token_points, 128)
        self.assertFalse(bg.use_band_projection)
        with self.assertRaises(ValueError):
            arm_spec(
                "P08-BG",
                global_resample_numerator_hz=25_600,
                global_resample_denominator=True,
            )
        with self.assertRaises(ValueError):
            arm_spec("P08-DN", duration_ms=7.0)

    def test_all_arms_forward_and_pretraining_loss(self) -> None:
        device = torch.device("cpu")
        cases = (
            (arm_spec("P08-DN", duration_ms=10.0), 512, 25_600.0),
            (arm_spec("P08-M", duration_ms=10.0), 512, 25_600.0),
            (
                arm_spec(
                    "P08-BG",
                    global_resample_numerator_hz=25_600,
                    global_resample_denominator=1,
                ),
                512,
                25_600.0,
            ),
            (arm_spec("P08-NC"), 512, 25_600.0),
        )
        for spec, points, rate in cases:
            model = build_model(spec, seed=42, device=device).train()
            signals = torch.randn(4, points, 1)
            rates = torch.full((4,), rate)
            labels = torch.arange(4)
            logits, features = model(
                signals,
                task_id="classification",
                return_feature=True,
                sampling_rate_hz=rates,
            )
            self.assertEqual(tuple(logits.shape), (4, 4))
            self.assertEqual(tuple(features.shape), (4, 128))
            loss, parts = pretraining_loss(logits, features, labels)
            self.assertTrue(torch.isfinite(loss))
            self.assertEqual(set(parts), {"classification_loss", "contrastive_loss", "total_loss"})

    def test_shared_components_have_exact_paired_initialization(self) -> None:
        seed = 42
        device = torch.device("cpu")
        models = {
            "P08-DN": build_model(
                arm_spec("P08-DN", duration_ms=10.0), seed=seed, device=device
            ),
            "P08-M": build_model(
                arm_spec("P08-M", duration_ms=10.0), seed=seed, device=device
            ),
            "P08-BG": build_model(
                arm_spec(
                    "P08-BG",
                    global_resample_numerator_hz=25_600,
                    global_resample_denominator=1,
                ),
                seed=seed,
                device=device,
            ),
            "P08-NC": build_model(arm_spec("P08-NC"), seed=seed, device=device),
        }
        reference = models["P08-DN"].state_dict()
        shared_keys = [
            key
            for key in reference
            if key.startswith("embedding.patch_encoder.")
            or key.startswith("embedding.band_encoder.")
            or key.startswith("backbone.")
            or key.startswith("task_head.")
        ]
        self.assertTrue(shared_keys)
        for arm_id, model in models.items():
            state = model.state_dict()
            for key in shared_keys:
                self.assertTrue(
                    torch.equal(reference[key], state[key]),
                    msg=f"{arm_id} does not share paired initialization for {key}",
                )

    def test_all_frozen_candidate_and_rate_shape_boundaries(self) -> None:
        device = torch.device("cpu")
        seed = 123
        native_cases = []
        for arm_id in ("P08-DN", "P08-M"):
            for duration_ms in (5.0, 10.0, 15.0):
                native_cases.extend(
                    (arm_spec(arm_id, duration_ms=duration_ms), rate)
                    for rate in EVALUATION_RATES_HZ
                )
        native_cases.extend(
            (arm_spec("P08-NC"), rate) for rate in EVALUATION_RATES_HZ
        )
        for spec, rate in native_cases:
            model = build_model(spec, seed=seed, device=device).eval()
            points = expected_rate_points(rate)
            with torch.inference_mode():
                logits = model(
                    torch.zeros(2, points, 1),
                    task_id="classification",
                    sampling_rate_hz=torch.full((2,), float(rate)),
                )
            self.assertEqual(tuple(logits.shape), (2, 4))

        bg_cases = ((51_200, 3, 341), (25_600, 1, 512), (51_200, 1, 1024))
        for numerator, denominator, points in bg_cases:
            spec = arm_spec(
                "P08-BG",
                global_resample_numerator_hz=numerator,
                global_resample_denominator=denominator,
            )
            model = build_model(spec, seed=seed, device=device).eval()
            self.assertFalse(
                any(
                    parameter.requires_grad
                    for parameter in model.embedding.band_encoder.parameters()
                )
            )
            with torch.inference_mode():
                logits = model(
                    torch.zeros(2, points, 1),
                    task_id="classification",
                    sampling_rate_hz=torch.full((2,), numerator / denominator),
                )
            self.assertEqual(tuple(logits.shape), (2, 4))


if __name__ == "__main__":
    unittest.main()
