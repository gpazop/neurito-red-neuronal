"""Pruebas del núcleo matemático de Neurito (sin Pygame)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from neurito.brain import NeuralNet, softmax
from neurito.data import make_dataset, teacher_actions
from neurito.metrics import confusion_matrix, expected_calibration_error


class TestSoftmax(unittest.TestCase):
    def test_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
        probabilities = softmax(logits)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_shift_invariance(self):
        logits = np.array([[0.2, -1.1, 3.4]])
        np.testing.assert_allclose(softmax(logits), softmax(logits + 100.0))

    def test_large_logits_do_not_overflow(self):
        probabilities = softmax(np.array([[1000.0, 1001.0, 999.0]]))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


class TestGradients(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(7)
        model = NeuralNet(n_inputs=3, hidden=5, n_outputs=3, seed=11)
        X = rng.random((17, 3))
        y = rng.integers(0, 3, size=17)
        l2 = 1e-3

        analytic = model.gradients(X, y, l2=l2)
        epsilon = 1e-6
        for name in ("W1", "b1", "W2", "b2"):
            parameter = getattr(model, name)
            numeric = np.zeros_like(parameter)
            iterator = np.nditer(parameter, flags=["multi_index"])
            for _ in iterator:
                index = iterator.multi_index
                original = parameter[index]
                parameter[index] = original + epsilon
                plus = model.loss(X, y, l2=l2)
                parameter[index] = original - epsilon
                minus = model.loss(X, y, l2=l2)
                parameter[index] = original
                numeric[index] = (plus - minus) / (2 * epsilon)
            np.testing.assert_allclose(
                analytic[name],
                numeric,
                rtol=2e-5,
                atol=2e-7,
                err_msg=f"El gradiente de {name} no coincide.",
            )


class TestTeacher(unittest.TestCase):
    def test_known_cases(self):
        X = np.array(
            [
                [0.10, 0.20, 0.30],  # sueño 0.90
                [0.90, 0.80, 0.10],  # hambre 0.80
                [0.90, 0.10, 0.90],  # aburrimiento 0.90
                [0.50, 0.50, 0.50],  # empate triple -> dormir
            ]
        )
        np.testing.assert_array_equal(teacher_actions(X), [0, 1, 2, 0])

    def test_dataset_is_reproducible_and_disjoint(self):
        first = make_dataset(n=300, seed=99)
        second = make_dataset(n=300, seed=99)
        np.testing.assert_array_equal(first.X_train, second.X_train)
        np.testing.assert_array_equal(first.y_test, second.y_test)
        self.assertEqual(len(first.X_train) + len(first.X_val) + len(first.X_test), 300)


class TestPersistence(unittest.TestCase):
    def test_save_and_load(self):
        model = NeuralNet(n_inputs=3, hidden=4, n_outputs=3, seed=5)
        X = np.array([[0.2, 0.3, 0.4]])
        expected = model.predict_proba(X)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path, {"actions": ["dormir", "comer", "jugar"]})
            loaded, metadata = NeuralNet.load(path)
        np.testing.assert_allclose(loaded.predict_proba(X), expected)
        self.assertEqual(metadata["hidden"], 4)
        self.assertEqual(metadata["actions"][0], "dormir")


class TestMetrics(unittest.TestCase):
    def test_confusion_matrix(self):
        matrix = confusion_matrix(np.array([0, 0, 1, 2]), np.array([0, 1, 1, 1]), 3)
        np.testing.assert_array_equal(matrix, [[1, 1, 0], [0, 1, 0], [0, 1, 0]])

    def test_ece_is_zero_for_perfect_calibration(self):
        probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(expected_calibration_error(probabilities, [0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
