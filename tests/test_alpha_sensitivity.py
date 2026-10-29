"""Check sweep scoring against direct prediction and stationary iteration."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_pagerank_zero_rows import ROOT, load_functions


spec = importlib.util.spec_from_file_location(
    "alpha_sensitivity", ROOT / "experiments/04_04_Ramachandran_alpha_sensitivity.py")
sweep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sweep)
EVAL = load_functions("04_03_Ramachandran_evaluate.py", {"_solve", "err1"})


class AlphaSensitivityTests(unittest.TestCase):
    def test_sufficient_statistics_match_direct_test_predictions(self):
        rng = np.random.default_rng(101)
        psi = rng.normal(size=(251, 8))
        ntr = 200
        stats = sweep.readout_statistics(psi, ntr, 4)
        for selected in ([0, 3, 1], [7, 4, 2, 0], list(range(8))):
            direct = EVAL["err1"](
                np.array(selected), len(selected), psi[:ntr].T, psi[1:ntr+1].T,
                psi[1:ntr+1, :4].T, psi[ntr:-1].T, psi[ntr+1:, :4].T, ntr)
            fast = sweep.readout_mse(stats, selected)
            np.testing.assert_allclose(fast, direct, atol=1e-12, rtol=1e-12)

    def test_resolvent_matches_independent_row_iteration(self):
        p = np.array([[.9, .1, 0.], [.5, .1, .4], [.1, .2, .7]])
        preferences = np.column_stack([np.ones(3) / 3, [1., 0., 0.]])
        for alpha in (.05, .85, .95):
            scores = sweep.resolvent_scores(p, preferences, alpha)
            for i in range(2):
                v = preferences[:, i].copy()
                for _ in range(10000):
                    new = alpha * v @ p + (1 - alpha) * preferences[:, i]
                    if np.max(np.abs(v - new)) < 1e-14:
                        break
                    v = new
                np.testing.assert_allclose(scores[:, i], new, atol=1e-12)

    def cache(self, path, protocol="train_only_v1"):
        rng = np.random.default_rng(102)
        psi = rng.normal(size=(301, 196))
        ntr = 240
        mean = psi[:ntr+1].mean(axis=0)
        std = psi[:ntr+1].std(axis=0)
        psi = (psi - mean) / std
        p = rng.uniform(size=(196, 196))
        p /= p.sum(axis=1, keepdims=True)
        pref = np.column_stack([np.ones(196) / 196, np.r_[np.ones(4) / 4, np.zeros(192)]])
        scores = sweep.resolvent_scores(p, pref, .85)
        np.savez(path, Psi=psi, K_hat=p, n_train_pairs=ntr, n_state=4,
                 preprocessing_fit_stop=241, evaluation_protocol=protocol,
                 ppr_retained_indices=np.arange(196), pi_pr=scores[:, 0], pi_ppr=scores[:, 1],
                 pr_order=np.argsort(-scores[:, 0]), ppr_order=np.argsort(-scores[:, 1]),
                 tica_order=np.arange(196), pcca_order=np.arange(196))
        return psi, ntr

    def test_full_sweep_cache_scoring_matches_direct_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            psi, ntr = self.cache(path)
            rows, baseline, check = sweep.analyze_trajectory(path, 0, alphas=[.5, .85], sizes=[5, 10])
            self.assertEqual(len(rows), 8)
            self.assertEqual(len(baseline), 10)
            for row in rows:
                selected = np.array(row["selected_indices"])
                direct = EVAL["err1"](
                    selected, row["N"], psi[:ntr].T, psi[1:ntr+1].T,
                    psi[1:ntr+1, :4].T, psi[ntr:-1].T, psi[ntr+1:, :4].T, ntr)
                np.testing.assert_allclose(row["mse"], direct, atol=1e-11, rtol=1e-11)
                if row["alpha"] == .85:
                    self.assertEqual(row["jaccard"], 1.)
            for method in ("PR", "PPR"):
                self.assertLess(check["baseline_checks"][method]["max_abs_score_difference"], 1e-12)

    def test_sweep_rejects_old_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            self.cache(path, protocol="full_trajectory")
            with self.assertRaisesRegex(ValueError, "wrong evaluation protocol"):
                sweep.analyze_trajectory(path, 0)


if __name__ == "__main__":
    unittest.main()
