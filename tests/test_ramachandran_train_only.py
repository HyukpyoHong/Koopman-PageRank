"""Held-out-tail perturbation checks without running the experiment scripts."""

from contextlib import redirect_stdout
import io
import unittest
from types import SimpleNamespace

import numpy as np

from test_pagerank_zero_rows import load_functions


FIT = load_functions("04_02_Ramachandran_edmd_pagerank.py",
                     {"build_dictionary", "pagerank", "ppr", "fit_trajectory"})
EVAL = load_functions("04_03_Ramachandran_evaluate.py",
                      {"validate_training_result", "_solve"})


class TrainingOnlyTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.trajectory = rng.uniform(-np.pi, np.pi, size=(1000, 2))
        self.ntr = int(.8 * (len(self.trajectory) - 1))
        self.stop = self.ntr + 1
        self.changed = self.trajectory.copy()
        self.changed[self.stop:] = rng.uniform(-1, 1, size=self.changed[self.stop:].shape)

    def fit(self, trajectory):
        with redirect_stdout(io.StringIO()):
            return FIT["fit_trajectory"](trajectory, random_seed=123)

    def assert_training_unchanged(self, first, second):
        for key in ("dictionary_mean", "dictionary_std", "K_edmd", "K_hat",
                    "pi_pr", "pi_ppr", "pr_order", "ppr_order", "tica_score",
                    "tica_order", "fdr", "pcca_order"):
            np.testing.assert_allclose(first[key], second[key], atol=1e-12, rtol=1e-12,
                                       err_msg=key)
        np.testing.assert_allclose(first["Psi"][:self.stop], second["Psi"][:self.stop])
        self.assertFalse(np.allclose(first["Psi"][self.stop:], second["Psi"][self.stop:]))
        self.assertEqual(first["preprocessing_fit_stop"], self.stop)
        self.assertEqual(first["n_train_pairs"], self.ntr)

    def test_fitted_scaling_ignores_heldout_tail(self):
        build = FIT["build_dictionary"]
        a = build(*self.trajectory.T, fit_stop=self.stop, return_scaling=True)
        b = build(*self.changed.T, fit_stop=self.stop, return_scaling=True)
        np.testing.assert_allclose(a[3], b[3])
        np.testing.assert_allclose(a[4], b[4])
        np.testing.assert_allclose(a[0][:self.stop], b[0][:self.stop])
        np.testing.assert_allclose(a[0][:self.stop].mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(a[0][:self.stop].std(axis=0), 1, atol=1e-14)

    def test_every_estimator_fits_only_training_prefix(self):
        calls = []

        class TICA:
            def __init__(self, **kwargs):
                pass

            def fit_fetch(self, data):
                calls.append(("tica", data.copy()))
                return SimpleNamespace(singular_vectors_left=np.eye(data.shape[1], 10))

        class KMeans:
            def __init__(self, **kwargs):
                pass

            def fit_fetch(self, data):
                calls.append(("kmeans", data.copy()))
                return self

            def transform(self, data):
                return np.digitize(data[:, 0], [-1., 1.])

        class Counts:
            states = np.arange(3)

            def __init__(self, **kwargs):
                pass

            def fit_fetch(self, labels):
                calls.append(("counts", labels.copy()))
                return self

            def submodel_largest(self):
                return self

        class MSM:
            def __init__(self, **kwargs):
                pass

            def fit_fetch(self, counts):
                return self

            def pcca(self, n_macro):
                return SimpleNamespace(memberships=np.eye(n_macro))

        FIT.update(TICA=TICA, dtKMeans=KMeans,
                   TransitionCountEstimator=Counts, MaximumLikelihoodMSM=MSM)
        first = self.fit(self.trajectory)
        second = self.fit(self.changed)
        self.assert_training_unchanged(first, second)
        self.assertEqual([name for name, _ in calls],
                         ["tica", "kmeans", "counts"] * 2)
        for (name, a), (_, b) in zip(calls[:3], calls[3:]):
            self.assertEqual(len(a), self.stop, name)
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(calls[1][1], self.trajectory[:self.stop])
        # The readout is fitted from the same saved training pairs in Step 3.
        fits = []
        for result in (first, second):
            ntr = EVAL["validate_training_result"](result, len(result["Psi"]))
            x = result["Psi"][:ntr].T
            y = result["Psi"][1:ntr+1].T
            fits.append(EVAL["_solve"](np.arange(5), x, y, y[:4], ntr))
        for a, b in zip(*fits):
            np.testing.assert_array_equal(a, b)

    def test_cache_requires_protocol_and_saved_training_boundary(self):
        record = dict(evaluation_protocol="train_only_v1", n_train_pairs=79999,
                      preprocessing_fit_stop=80000,
                      dictionary_mean=np.zeros(196), dictionary_std=np.ones(196))
        self.assertEqual(EVAL["validate_training_result"](record, 100000), 79999)
        for changed in ({}, {**record, "evaluation_protocol": "full_trajectory"},
                        {**record, "preprocessing_fit_stop": 100000},
                        {**record, "dictionary_std": np.zeros(196)}):
            with self.assertRaises(ValueError):
                EVAL["validate_training_result"](changed, 100000)

    def test_actual_deeptime_rankings_ignore_changed_heldout_tail(self):
        try:
            from deeptime.decomposition import TICA
            from deeptime.clustering import KMeans
            from deeptime.markov import TransitionCountEstimator
            from deeptime.markov.msm import MaximumLikelihoodMSM
        except ImportError:
            self.skipTest("deeptime is not installed")
        FIT.update(TICA=TICA, dtKMeans=KMeans,
                   TransitionCountEstimator=TransitionCountEstimator,
                   MaximumLikelihoodMSM=MaximumLikelihoodMSM)
        first = self.fit(self.trajectory)
        second = self.fit(self.changed)
        self.assert_training_unchanged(first, second)


if __name__ == "__main__":
    unittest.main()
