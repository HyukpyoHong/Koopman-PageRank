"""Check the appendix example and compare cached scoring with direct predictions."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from test_pagerank_zero_rows import ROOT, load_functions


spec = importlib.util.spec_from_file_location(
    "normalization_comparison", ROOT / "experiments/04_05_Ramachandran_normalization_comparison.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)
EVAL = load_functions("04_03_Ramachandran_evaluate.py", {"_solve", "err1"})


class NormalizationComparisonTests(unittest.TestCase):
    def test_toy_scores_and_selected_sets_match_the_appendix(self):
        toy = comparison.toy_example()
        row, glob = toy["row_normalized"], toy["globally_normalized"]
        self.assertEqual([f"{x:.4f}" for x in row["scores"][1:]], ["0.0551", "0.0897"])
        self.assertEqual([f"{x:.4f}" for x in glob["scores"][1:]], ["0.0481", "0.0337"])
        self.assertEqual(set(row["selected_at_N2"]), {1, 3})
        self.assertEqual(set(glob["selected_at_N2"]), {1, 2})
        self.assertAlmostEqual(row["score_sum"], 1.)
        self.assertLess(glob["score_sum"], 1.)

    def test_global_score_matches_independent_substochastic_iteration(self):
        w = np.array([[0., 2., 1.], [1., 1., 0.], [0., 1., .2]])
        scores = comparison.normalization_scores(w, [0], .85)
        transition = w / w.sum(axis=1).max()
        preference = np.array([1., 0., 0.])
        value = preference.copy()
        for _ in range(10000):
            new = .85 * value @ transition + .15 * preference
            if np.max(np.abs(new - value)) < 1e-14:
                break
            value = new
        np.testing.assert_allclose(scores["globally_normalized"], new, atol=1e-12)

    def test_old_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.npz"
            np.savez(path, evaluation_protocol="full_trajectory")
            with self.assertRaisesRegex(ValueError, "train_only_v1"):
                comparison.analyze_trajectory(path, 0)

    def test_both_normalizations_use_the_existing_direct_readout(self):
        rng = np.random.default_rng(142)
        psi = rng.normal(size=(101, 196))
        w = rng.uniform(size=(196, 196))
        scores = comparison.normalization_scores(w, list(range(4)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.npz"
            np.savez(path, evaluation_protocol="train_only_v1", dictionary_size=196,
                     Psi=psi, n_train_pairs=80, preprocessing_fit_stop=81, n_state=4,
                     K_edmd=w, K_hat=scores["transition"],
                     ppr_retained_indices=scores["retained"])
            rows = comparison.analyze_trajectory(path, 0, sizes=[5, 10])
        self.assertEqual(len(rows), 4)
        for row in rows:
            selected = np.array(row["selected_indices"])
            expected = EVAL["err1"](selected, row["N"], psi[:80].T, psi[1:81].T,
                                     psi[1:81, :4].T, psi[80:-1].T,
                                     psi[81:, :4].T, 80)
            np.testing.assert_allclose(row["mse"], expected, atol=1e-11, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
