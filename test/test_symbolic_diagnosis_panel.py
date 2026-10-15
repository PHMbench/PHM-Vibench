"""Acquisition-boundary regression fixtures; not measured data or paper evidence."""
import unittest
import numpy as np
from phmfactory.p06 import calibrate, features
from src.task_factory.task.classification.symbolic_diagnosis.panel import inspect_panel


def fixture():
    """Small deterministic waveforms exercise grouping, not method efficacy."""
    rows = {key: [] for key in ('x', 'y', 'unit', 'split', 'condition',
                                'acquisition', 'speed_hz', 'nominal_speed_hz')}
    t = np.arange(2048) / 4096
    for split in ('train', 'val'):
        for label in range(3):
            unit = f'{split}-{label}'
            for condition, nominal in (('source', 30.), ('target', 21.)):
                for repeat in range(3):
                    speed = nominal + (repeat - 1) * .01
                    a = np.array([.12, .10])
                    if label:
                        a[label - 1] = .45
                    a *= 1 + .08 * (repeat - 1)
                    if condition == 'target':
                        a *= .8
                    x = np.sin(2*np.pi*speed*t)
                    x += a[0]*np.sin(2*np.pi*3.2*speed*t + .13*repeat)
                    x += a[1]*np.sin(2*np.pi*4.8*speed*t + .19*repeat)
                    values = (x, label, unit, split, condition,
                              f'{unit}/{condition}/{repeat}', speed, nominal)
                    for key, value in zip(rows, values):
                        rows[key].append(value)
    return {**{key: np.asarray(value) for key, value in rows.items()},
            'fs': np.asarray(4096.)}


def inspect(data):
    return inspect_panel(data, [3.2, 4.8], 'source', 'target')


class MeasurementPanelTests(unittest.TestCase):

    def setUp(self):
        self.data = fixture()

    def test_three_class_panel_with_variable_measured_speed(self):
        report, rows = inspect(self.data)
        self.assertEqual(report['physical_unit_count'], 6)
        self.assertEqual(report['acquisition_count'], 36)
        self.assertEqual(len(rows), 36)
        self.assertEqual(len(report['gamma']), 2)
        self.assertTrue(all((width > 0 for width in report['gamma'])))
        self.assertFalse(report['physical_pairing_established'])
        self.assertNotIn('certified', report)
        self.assertNotIn('macro_f1', report)
        selected = (self.data['split'] == 'train') & (self.data['condition'] == 'source')
        old_input = {key: self.data[key][selected] for key in ('y', 'unit', 'split', 'speed_hz')}
        e = features(self.data['x'][selected], self.data['speed_hz'][selected], 4096.0, [3.2, 4.8])
        with self.assertRaisesRegex(ValueError, 'repeats'):
            calibrate(e, old_input)

    def test_only_training_source_can_change_calibration(self):
        before, old_rows = inspect(self.data)
        at = (self.data['split'] == 'val') | (self.data['condition'] == 'target')
        t = np.arange(2048) / 4096
        self.data['x'][at] += 2 * np.sin(2 * np.pi * 3.2 * self.data['speed_hz'][at, None] * t)
        after, new_rows = inspect(self.data)
        self.assertEqual(before['theta'], after['theta'])
        self.assertEqual(before['gamma'], after['gamma'])
        self.assertNotEqual(old_rows, new_rows)

    def test_record_order_does_not_create_cross_condition_pairs(self):
        before, _ = inspect(self.data)
        order = np.arange(len(self.data['x']))[::-1]
        shuffled = {key: value if key == 'fs' else value[order] for key, value in self.data.items()}
        after, _ = inspect(shuffled)
        self.assertEqual(before, after)

    def test_duplicate_acquisition_rejected(self):
        self.data['acquisition'][1] = self.data['acquisition'][0]
        with self.assertRaisesRegex(ValueError, 'Duplicate acquisition'):
            inspect(self.data)

    def test_physical_unit_leakage_rejected(self):
        self.data['unit'][self.data['unit'] == 'val-0'] = 'train-0'
        with self.assertRaisesRegex(ValueError, 'leakage'):
            inspect(self.data)

    def test_test_set_is_rejected(self):
        self.data['split'][self.data['split'] == 'val'] = 'test'
        with self.assertRaisesRegex(ValueError, 'train/val only'):
            inspect(self.data)

    def test_missing_repeats_rejected(self):
        keep = np.ones(len(self.data['x']), bool)
        keep[:2] = False
        data = {key: value if key == 'fs' else value[keep] for key, value in self.data.items()}
        with self.assertRaisesRegex(ValueError, 'Two or more'):
            inspect(data)

    def test_class_order_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, 'declared fault classes'):
            inspect_panel(self.data, [3.2, 4.8, 6.4], 'source', 'target')

    def test_nonfinite_signal_rejected(self):
        self.data['x'][0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            inspect(self.data)

    def test_inconsistent_nominal_condition_rejected(self):
        self.data['nominal_speed_hz'][0] = 31.0
        with self.assertRaisesRegex(ValueError, 'documented nominal speed'):
            inspect(self.data)

    def test_nonpositive_width_has_no_substitute(self):
        for unit in np.unique(self.data['unit']):
            for condition in ('source', 'target'):
                at = (self.data['unit'] == unit) & (self.data['condition'] == condition)
                i = np.flatnonzero(at)[0]
                self.data['x'][at] = self.data['x'][i]
                self.data['speed_hz'][at] = self.data['speed_hz'][i]
        with self.assertRaisesRegex(ValueError, 'Nonpositive calibrated width'):
            inspect(self.data)
