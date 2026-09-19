"""Explicit periodic tensor fixtures; these do not qualify any real corpus."""
import math
import unittest

import torch

from src.data_factory.tii_physical import project_window


def settings(**changes):
    values = dict(
        channel=0, native_rate_hz=128., effective_rate_hz=128., grid_rate_hz=256.,
        duration_s=1., num_patches=8, patch_size=32,
        common_bands_hz=[(0., 16.)], increment_bands_hz=[(16., 32.)],
        common_support='fully_usable', increment_support='fully_usable',
        input_unit='g', output_unit='m/s^2', unit_scale=9.80665,
        support_basis='tensor_fixture', support_evidence='analytic Fourier tensor fixture',
    )
    values.update(changes)
    return values


def waves(rate=128., duration=1.):
    time = torch.arange(round(rate * duration), dtype=torch.float64) / rate
    return tuple(torch.sin(2 * math.pi * hz * time) for hz in (4., 20., 48.))


class PhysicalTests(unittest.TestCase):
    def test_disjoint_projection_unit_conversion_and_union_reconstruction(self):
        c, p, excluded = waves()
        raw = torch.stack((c + 2*p + 3*excluded, torch.full_like(c, 99)), dim=-1)
        original = raw.clone()
        out = project_window(raw, **settings())
        expected_c, expected_p, _ = waves(rate=256.)
        self.assertEqual(out['common'].shape, (8, 32))
        torch.testing.assert_close(out['common'].flatten(), expected_c * 9.80665, rtol=0, atol=1e-12)
        torch.testing.assert_close(out['incremental'].flatten(), 2*expected_p * 9.80665, rtol=0, atol=1e-12)
        union = out['common'] + out['incremental']
        expected_union = (expected_c + 2*expected_p) * 9.80665
        torch.testing.assert_close(union.flatten(), expected_union, rtol=0, atol=1e-12)
        self.assertLess(abs((out['common'] * out['incremental']).sum().item()), 1e-10)
        torch.testing.assert_close(raw, original, rtol=0, atol=0)
        # Reprojecting either component preserves it and removes the other one.
        again = project_window(union.flatten()[:, None], **settings(
            effective_rate_hz=256., input_unit='m/s^2', unit_scale=1.))
        torch.testing.assert_close(again['common'], out['common'], rtol=0, atol=1e-12)
        torch.testing.assert_close(again['incremental'], out['incremental'], rtol=0, atol=1e-12)

    def test_rate_changes_preserve_duration_frequency_and_amplitude(self):
        outputs = []
        for rate in (128., 256., 512.):
            c, p, excluded = waves(rate, duration=.5)
            outputs.append(project_window((c + p + excluded)[:, None], **settings(
                native_rate_hz=rate, effective_rate_hz=rate, duration_s=.5,
                num_patches=4)))
        for out in outputs[1:]:
            torch.testing.assert_close(out['common'], outputs[0]['common'], rtol=0, atol=1e-12)
            torch.testing.assert_close(out['incremental'], outputs[0]['incremental'], rtol=0, atol=1e-12)
        # A cached upsampling does not increase the native hardware support.
        with self.assertRaisesRegex(ValueError, 'native/effective Nyquist'):
            project_window(torch.zeros(128, 1), **settings(native_rate_hz=32.))
        with self.assertRaisesRegex(ValueError, 'physical duration'):
            project_window(torch.zeros(64, 1), **settings())
        with self.assertRaisesRegex(ValueError, 'num_patches'):
            project_window(torch.zeros(128, 1), **settings(patch_size=16))

    def test_unavailable_increment_isolated_in_value_and_gradient(self):
        common, increment, _ = waves()
        amplitude = torch.tensor(3., dtype=torch.float64, requires_grad=True)
        raw = (common + amplitude * increment)[:, None]
        out = project_window(raw, **settings(increment_support='unavailable'))
        changed = project_window((common - 100*increment)[:, None], **settings(increment_support='unavailable'))
        torch.testing.assert_close(out['common'], changed['common'], rtol=0, atol=5e-12)
        self.assertFalse(out['increment_available'].item())
        self.assertEqual(torch.count_nonzero(out['incremental']).item(), 0)
        out['common'].square().sum().backward()
        self.assertLess(abs(amplitude.grad.item()), 1e-10)
        # The missing set can lie beyond actual sampling support; it is excluded.
        low = project_window(torch.zeros(32, 1), **settings(
            native_rate_hz=32., effective_rate_hz=32., increment_support='unavailable'))
        self.assertEqual(torch.count_nonzero(low['incremental']).item(), 0)

    def test_qualification_and_geometry_fail_before_projection(self):
        raw = torch.zeros(128, 1)
        cases = (
            ({'common_support': 'unavailable'}, 'missing common'),
            ({'common_support': 'partial'}, 'common_support'),
            ({'increment_support': 'unknown'}, 'partial/unknown'),
            ({'increment_support': 'partial'}, 'partial/unknown'),
            ({'support_basis': 'nyquist'}, 'Nyquist is insufficient'),
            ({'support_evidence': ''}, 'support_evidence'),
            ({'input_unit': 'unknown'}, 'known physical unit'),
            ({'unit_scale': float('nan')}, 'unit_scale'),
            ({'common_bands_hz': [(0., 20.)]}, 'mutually disjoint'),
            ({'increment_bands_hz': []}, 'nonempty'),
            ({'increment_bands_hz': [(16.1, 16.2)]}, 'Fourier mode'),
            ({'channel': 1}, 'existing raw channel'),
        )
        for change, error in cases:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, error):
                project_window(raw, **settings(**change))


if __name__ == '__main__':
    unittest.main()
