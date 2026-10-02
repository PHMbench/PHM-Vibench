"""CPU fixtures for P06 scientific semantics; no real-data evidence is produced."""
import json
import unittest

import numpy as np

from phmfactory.p06 import (Head, SymbolicCertificateTask, calibrate, features,
                            fit_head, representations)


def repeat_fixture(*, measured_speed_fluctuates=False):
    t = np.arange(2048) / 4096
    amplitudes = np.array([.03, .06, .04, .07, .5, .6, .55, .65])
    speed = np.full(8, 30.)
    if measured_speed_fluctuates:
        speed += np.linspace(-.005, .005, 8)
    x = (np.sin(2*np.pi*speed[:, None]*t) +
         amplitudes[:, None]*np.sin(2*np.pi*3.2*speed[:, None]*t))
    return dict(x=x, speed_hz=speed, nominal_speed_hz=np.full(8, 30.), fs=4096.,
                y=np.repeat([0, 1], 4),
                unit=np.repeat(['healthy-a', 'healthy-b', 'fault-a', 'fault-b'], 2),
                split=np.full(8, 'train'), condition=np.full(8, 'source'),
                acquisition=np.asarray([f'record-{i}' for i in range(8)]))


class ExactCertificateTests(unittest.TestCase):
    def test_stored_float_cancellation_keeps_exact_winner(self):
        head = Head([[1., 1.], [1., 0.]], [0., 0.])
        self.assertEqual(head.decision([1e16, 1.]), 0)
        self.assertEqual(head.check([1e16, 1.], [1e16, 1.])['certified'], 1)

    def test_strict_certificate_boundary_and_adjacent_binary_float(self):
        head = Head([[0.], [1.]], [1., 0.])
        boundary = head.check([0.], [1.])
        self.assertEqual((boundary['certified'], boundary['prediction'], boundary['changed']),
                         (0, -1, 1))
        inside = head.check([0.], [np.nextafter(1., 0.)])
        self.assertEqual((inside['certified'], inside['prediction'], inside['changed']),
                         (1, 0, 0))

    def test_expected_tie_and_constant_head(self):
        tied = Head([[1.], [1.]], [0., 0.]).check([0.], [0.])
        self.assertEqual((tied['expected'], tied['certified'], tied['changed']), (-1, 0, None))
        self.assertEqual(Head([[0.], [0.]], [1., 0.]).check([0.], [100.])['certified'], 1)


class SymbolicCertificateTaskTests(unittest.TestCase):
    def setUp(self):
        self.data = repeat_fixture()

    def test_fit_predict_certificate_and_existing_primitive_parity(self):
        for arm in ('hard_order', 'uncertainty_order'):
            with self.subTest(arm=arm):
                task = SymbolicCertificateTask(orders=[3.2], representation=arm).fit(self.data)
                e = features(self.data['x'], self.data['speed_hz'], self.data['fs'], [3.2])
                theta, gamma = calibrate(e, self.data)
                z = representations(e, theta, gamma)[arm]
                head = fit_head(z, self.data['y'])
                np.testing.assert_array_equal(task.theta, theta)
                np.testing.assert_array_equal(task.gamma, gamma)
                np.testing.assert_array_equal(task.symbols(self.data), z)
                np.testing.assert_array_equal(task.predict(self.data), self.data['y'])
                self.assertEqual((task.head.w, task.head.b), (head.w, head.b))
                checked = task.check_pairs(z, expected_symbols=z)
                self.assertEqual(checked, [head.check(row, row) for row in z])
                self.assertTrue(all(row['certified'] and row['changed'] == 0 for row in checked))

    def test_fit_requires_explicit_train_population(self):
        missing = dict(self.data)
        missing.pop('split')
        with self.assertRaisesRegex(ValueError, 'explicit training'):
            SymbolicCertificateTask(orders=[3.2]).fit(missing)
        for held_out in ('val', 'test'):
            mixed = {**self.data, 'split': self.data['split'].copy()}
            mixed['split'][-1] = held_out
            with self.assertRaisesRegex(ValueError, 'train rows only'):
                SymbolicCertificateTask(orders=[3.2]).fit(mixed)

    def test_calibration_rejects_invalid_labels_repeats_and_conditions(self):
        invalid = [
            {**self.data, 'unit': np.arange(8).astype(str)},
            {**self.data, 'y': self.data['y'] + 2},
            {**self.data, 'condition': np.array(['load-a', 'load-b']*4)},
            {**self.data, 'unit': np.full(8, 'contradictory')},
            {**self.data, 'x': np.repeat(self.data['x'][::2], 2, axis=0)},
        ]
        for data in invalid:
            with self.subTest(fields=list(data)), self.assertRaises(ValueError):
                SymbolicCertificateTask(orders=[3.2]).fit(data)

    def test_real_condition_uses_actual_features_and_nominal_repeat_groups(self):
        data = repeat_fixture(measured_speed_fluctuates=True)
        task = SymbolicCertificateTask(orders=[3.2]).fit(data)
        actual_energy = features(data['x'], data['speed_hz'], data['fs'], [3.2])
        with self.assertRaisesRegex(ValueError, 'same-unit, same-speed repeats'):
            calibrate(actual_energy, data)
        theta, gamma = calibrate(actual_energy, {**data, 'speed_hz': data['nominal_speed_hz']})
        np.testing.assert_array_equal(task.theta, theta)
        np.testing.assert_array_equal(task.gamma, gamma)
        np.testing.assert_array_equal(task.symbols(data),
            representations(actual_energy, theta, gamma)['uncertainty_order'])
        nominal_energy = features(data['x'], data['nominal_speed_hz'], data['fs'], [3.2])
        self.assertGreater(float(np.max(np.abs(actual_energy - nominal_energy))), 0.)
        state = task.state()
        self.assertEqual(state['source_nominal_speed_hz'], 30.)
        self.assertEqual(state['calibration_groups'], 'unit_and_nominal_speed_within_source_condition')
        np.testing.assert_array_equal(task.predict(data), data['y'])

    def test_real_condition_rejects_missing_or_invalid_nominal_and_acquisitions(self):
        for required in ('nominal_speed_hz', 'acquisition'):
            missing = dict(self.data)
            missing.pop(required)
            with self.subTest(missing=required), self.assertRaises(ValueError):
                SymbolicCertificateTask(orders=[3.2]).fit(missing)
        invalid = [
            {**self.data, 'nominal_speed_hz': np.array([30.]*7 + [31.])},
            {**self.data, 'nominal_speed_hz': np.full(8, np.nan)},
            {**self.data, 'nominal_speed_hz': np.zeros(8)},
            {**self.data, 'nominal_speed_hz': np.array([30.])},
            {**self.data, 'acquisition': np.full(8, 'one-windowed-record')},
            {**self.data, 'acquisition': np.full(8, '')},
            {**self.data, 'unit': np.full(8, '')},
            {**self.data, 'y': self.data['y'].astype(float)},
        ]
        for index, data in enumerate(invalid):
            with self.subTest(index=index), self.assertRaises(ValueError):
                SymbolicCertificateTask(orders=[3.2]).fit(data)

    def test_synthetic_repeat_grouping_retains_supplied_speed_semantics(self):
        data = dict(self.data)
        for field in ('condition', 'nominal_speed_hz', 'acquisition'):
            data.pop(field)
        task = SymbolicCertificateTask(orders=[3.2]).fit(data)
        np.testing.assert_array_equal(task.predict(data), data['y'])
        self.assertEqual(task.state()['calibration_groups'], 'unit_and_supplied_speed')

    def test_predict_and_failed_refit_do_not_use_held_out_rows_for_fit(self):
        task = SymbolicCertificateTask(orders=[3.2]).fit(self.data)
        original = task.state()
        held_out = {**self.data, 'x': self.data['x'] * 1.25,
                    'split': np.full(8, 'test')}
        task.predict(held_out)
        self.assertEqual(task.state(), original)
        with self.assertRaises(ValueError):
            task.fit(held_out)
        self.assertEqual(task.state(), original)

    def test_pairing_is_explicit_and_expected_symbols_are_consumed(self):
        task = SymbolicCertificateTask(orders=[3.2]).fit(self.data)
        z = task.symbols(self.data)
        with self.assertRaises(TypeError):
            task.check_pairs(z)
        original = task.check_pairs(z, expected_symbols=z)
        reversed_expected = task.check_pairs(z, expected_symbols=1-z)
        self.assertNotEqual(original[0]['expected'], reversed_expected[0]['expected'])
        self.assertGreater(reversed_expected[0]['delta'], original[0]['delta'])
        for invalid in (z[:-1], np.zeros((8, 2)), np.full(z.shape, np.nan)):
            with self.assertRaises(ValueError):
                task.check_pairs(z, expected_symbols=invalid)

    def test_state_roundtrip_is_exact_and_returns_independent_values(self):
        task = SymbolicCertificateTask(orders=[3.2]).fit(self.data)
        saved = json.loads(json.dumps(task.state(), allow_nan=False))
        rebuilt = Head(saved['weights'], saved['bias'])
        symbols = task.symbols(self.data)
        self.assertEqual([rebuilt.decision(z) for z in symbols], task.predict(self.data).tolist())
        self.assertEqual([rebuilt.check(z, z) for z in symbols],
                         task.check_pairs(symbols, expected_symbols=symbols))
        saved['theta'][0] = 1000
        saved['weights'][0][0] = 1000
        self.assertNotEqual(saved['theta'], task.state()['theta'])
        self.assertNotEqual(saved['weights'], task.state()['weights'])

    def test_inference_requires_fit_and_matching_sampling_rate(self):
        task = SymbolicCertificateTask(orders=[3.2])
        with self.assertRaises(RuntimeError):
            task.predict(self.data)
        task.fit(self.data)
        with self.assertRaisesRegex(ValueError, 'Sampling rate'):
            task.predict({**self.data, 'fs': 2048.})


if __name__ == '__main__':
    unittest.main()
