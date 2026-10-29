"""Check accelerated scoring and paired uncertainty against direct definitions."""

import unittest

import numpy as np

from test_pagerank_zero_rows import load_functions


EVAL = load_functions("04_03_Ramachandran_evaluate.py", {
    "_solve", "err1", "one_step_statistics", "one_step_from_statistics",
    "paired_ratio_summary"})


class FullEvaluationTests(unittest.TestCase):
    def test_accelerated_curve_matches_direct_predictions(self):
        rng = np.random.default_rng(103)
        psi = rng.normal(size=(301, 12))
        psi[1:] += .7 * psi[:-1]
        ntr = 220
        si = np.array([0, 2, 4, 6])
        stats = EVAL["one_step_statistics"](psi, ntr, si)
        for order in [np.arange(12), rng.permutation(12)]:
            for n in [4, 7, 12]:
                direct = EVAL["err1"](order, n, psi[:ntr].T,
                    psi[1:ntr+1].T, psi[1:ntr+1][:, si].T,
                    psi[ntr:-1].T, psi[ntr+1:][:, si].T, ntr)
                actual = EVAL["one_step_from_statistics"](order, n, stats)
                self.assertAlmostEqual(actual, direct, places=12)

    def test_paired_ratio_spread_preserves_trajectory_dependence(self):
        random = np.array([[1., 10.], [2., 20.], [8., 100.]])
        # Exactly half the error on every trajectory has zero ratio spread.
        mean, std, rom = EVAL["paired_ratio_summary"](.5 * random, random)
        np.testing.assert_array_equal(mean, [.5, .5])
        np.testing.assert_array_equal(std, [0., 0.])
        np.testing.assert_array_equal(rom, [.5, .5])
        method = np.array([[.5, 10.], [2., 10.], [1., 50.]])
        mean, std, rom = EVAL["paired_ratio_summary"](method, random)
        ratios = method / random
        np.testing.assert_allclose(mean, ratios.mean(axis=0))
        np.testing.assert_allclose(std, ratios.std(axis=0, ddof=0))
        np.testing.assert_allclose(rom, method.mean(axis=0) / random.mean(axis=0))
        self.assertFalse(np.allclose(mean, rom))


if __name__ == "__main__":
    unittest.main()
