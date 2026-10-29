"""Direction checks for the review sweep, without loading cached experiments."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from test_pagerank_zero_rows import ROOT, ROW_HELPERS, kp


SWEEP = (ROOT.parents[2] / "Writing/Koopman-Neurips-review/rebuttal"
         / "seed_set_sensitivity_sweep.py")


class OrientationTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[.9, .1, 0.], [.5, .1, .4], [.1, .2, .7]])
        self.alpha = .85

    def reference(self, preference):
        return np.linalg.solve(np.eye(3) - self.alpha * self.p.T,
                               (1 - self.alpha) * preference)

    def test_pr_matches_row_iteration_and_direct_resolvent(self):
        expected = self.reference(np.ones(3) / 3)
        column = kp.pagerank(self.p.T, num_iterations=100_000, d=self.alpha)
        row = ROW_HELPERS["pagerank"](self.p, alpha=self.alpha)
        np.testing.assert_allclose(column, expected, rtol=1e-5, atol=3e-6)
        np.testing.assert_allclose(row, expected, atol=1e-11)
        np.testing.assert_allclose(column, row, rtol=1e-5, atol=3e-6)

    def test_ppr_single_and_multiple_seeds_match_both_references(self):
        for seeds in ([0], [1], [0, 2]):
            preference = np.zeros(3)
            preference[seeds] = 1 / len(seeds)
            expected = self.reference(preference)
            column = kp.personalized_pagerank(self.p.T, 100_000, self.alpha, seeds)
            row = ROW_HELPERS["ppr"](self.p, seed_nodes=seeds, alpha=self.alpha)
            np.testing.assert_allclose(column, expected, rtol=1e-5, atol=3e-6)
            np.testing.assert_allclose(row, expected, atol=1e-11)
        wrong = kp.personalized_pagerank(self.p, 100_000, self.alpha, [0])
        self.assertGreater(np.max(np.abs(wrong - self.reference(np.array([1., 0., 0.])))), .1)

    @unittest.skipUnless(SWEEP.exists(), "Review sweep is outside this standalone release")
    def test_actual_sweep_sends_transpose_at_all_three_call_sites(self):
        calls = []

        def pr(matrix, *args, **kwargs):
            np.testing.assert_array_equal(matrix, self.p.T)
            calls.append("PR")
            return kp.pagerank(matrix, *args, **kwargs)

        def ppr(matrix, *args, **kwargs):
            np.testing.assert_array_equal(matrix, self.p.T)
            calls.append("PPR")
            return kp.personalized_pagerank(matrix, *args, **kwargs)

        names = {"run_ramachandran", "jaccard", "summarize"}
        tree = ast.parse(SWEEP.read_text())
        module = ast.Module(body=[n for n in tree.body
                                  if isinstance(n, ast.FunctionDef) and n.name in names],
                            type_ignores=[])
        namespace = dict(
            np=np, Path=Path, ALPHA=self.alpha,
            kp=SimpleNamespace(pagerank=pr, personalized_pagerank=ppr),
            load_ramachandran_items=lambda *args: [dict(K_hat=self.p, n_state=1, Psi=None)],
            ram_split=lambda *args: (None,) * 6,
            seed_policies=lambda *args: {"default": np.array([0]), "alternative": np.array([1])},
            ram_error=lambda *args: 0.,
        )
        exec(compile(module, str(SWEEP), "exec"), namespace)
        rows = namespace["run_ramachandran"](Path("unused"), Path("unused"), 1, [1])
        # One PR call, one baseline PPR call, then two policy PPR calls.
        self.assertEqual(calls, ["PR", "PPR", "PPR", "PPR"])
        self.assertEqual({row["seed_policy"] for row in rows}, {"default", "alternative"})


if __name__ == "__main__":
    unittest.main()
