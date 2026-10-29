"""One-step/rollout consistency and independent horizon checks."""

from unittest.mock import patch
import unittest

import numpy as np

from test_pagerank_zero_rows import load_functions


EVAL = load_functions("04_03_Ramachandran_evaluate.py", {"_solve", "err1", "errM"})


class ReadoutTimingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(29)
        self.psi = np.empty((151, 5))
        self.psi[0] = rng.normal(size=5)
        for t in range(150):
            self.psi[t + 1] = .75 * self.psi[t] + rng.normal(scale=.2, size=5)
        self.ntr = 100
        self.order = np.array([2, 0, 4, 3, 1])
        self.si = np.array([0, 3])

    def one_step(self, nt=None):
        x = self.psi[:self.ntr].T
        y = self.psi[1:self.ntr+1].T
        nt = len(self.psi) - self.ntr - 1 if nt is None else nt
        xt = self.psi[self.ntr:self.ntr+nt].T
        yt = self.psi[self.ntr+1:self.ntr+nt+1].T
        return EVAL["err1"](self.order, 3, x, y, y[self.si], xt, yt[self.si], self.ntr)

    def test_horizon_one_matches_all_one_step_test_pairs(self):
        multi = EVAL["errM"](self.order, 3, self.psi, self.si, 1, self.ntr)
        np.testing.assert_allclose(multi[0], self.one_step(), rtol=1e-13, atol=1e-14)

    def test_longer_rollout_first_horizon_matches_same_starting_points(self):
        steps = 7
        multi = EVAL["errM"](self.order, 3, self.psi, self.si, steps, self.ntr)
        nt = len(self.psi) - self.ntr - steps
        np.testing.assert_allclose(multi[0], self.one_step(nt), rtol=1e-13, atol=1e-14)

    def test_horizons_match_independent_matrix_power_predictions(self):
        steps = 4
        multi = EVAL["errM"](self.order, 3, self.psi, self.si, steps, self.ntr)
        idx = self.order[:3]
        x = self.psi[:self.ntr].T
        y = self.psi[1:self.ntr+1].T
        ksub, kread = EVAL["_solve"](idx, x, y, y[self.si], self.ntr)
        nt = len(self.psi) - self.ntr - steps
        starts = self.psi[self.ntr:self.ntr+nt, idx].T
        expected = []
        for h in range(1, steps + 1):
            pred = kread @ np.linalg.matrix_power(ksub, h - 1) @ starts
            target = self.psi[self.ntr+h:self.ntr+h+nt][:, self.si].T
            expected.append(np.mean(np.sum((target - pred)**2, axis=0)))
        np.testing.assert_allclose(multi, expected, rtol=1e-13, atol=1e-14)

    def test_exact_scalar_model_has_no_one_step_offset(self):
        psi = (2. ** np.arange(10))[:, None]
        exact = lambda *args: (np.array([[2.]]), np.array([[2.]]))
        with patch.dict(EVAL, {"_solve": exact}):
            actual = EVAL["errM"](np.array([0]), 1, psi, np.array([0]), 3, 4)
        np.testing.assert_array_equal(actual, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
