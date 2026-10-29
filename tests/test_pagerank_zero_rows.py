"""Small graph regressions; no trajectory data or experiment scripts are run."""

import ast
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import koopman_source as kp
from pagerank_utils import prune_zero_rows, retained_preference, expand_scores


def load_functions(filename, names):
    # Experiment modules load data at top level. Exercise only their helpers.
    tree = ast.parse((ROOT / "experiments" / filename).read_text())
    module = ast.Module(
        body=[node for node in tree.body
              if isinstance(node, ast.FunctionDef) and node.name in names],
        type_ignores=[],
    )
    namespace = dict(np=np, prune_zero_rows=prune_zero_rows,
                     retained_preference=retained_preference,
                     expand_scores=expand_scores)
    exec(compile(module, filename, "exec"), namespace)
    return namespace


ROW_HELPERS = load_functions("04_02_Ramachandran_edmd_pagerank.py", {"pagerank", "ppr"})
EVAL_HELPERS = load_functions("04_03_Ramachandran_evaluate.py", {"err1", "errM"})
BUILD_DICTIONARY = load_functions("04_02_Ramachandran_edmd_pagerank.py",
                                  {"build_dictionary"})["build_dictionary"]


def reference_scores(weights, seeds=None, alpha=0.85):
    transition, retained = prune_zero_rows(weights)
    preference = retained_preference(retained, len(weights), seeds)
    scores = np.linalg.solve(np.eye(len(retained)) - alpha * transition.T,
                             (1 - alpha) * preference)
    return expand_scores(scores, retained, len(weights))


class ZeroRowPageRankTests(unittest.TestCase):
    def setUp(self):
        self.weights = np.array([[0., 0., 0.], [3., 2., 1.], [0., 1., 1.]])

    def test_pruning_renormalizes_remaining_edges_without_mutating_input(self):
        original = self.weights.copy()
        transition, retained = prune_zero_rows(self.weights)
        np.testing.assert_array_equal(retained, [1, 2])
        np.testing.assert_allclose(transition, [[2/3, 1/3], [1/2, 1/2]])
        np.testing.assert_allclose(transition.sum(axis=1), 1)
        np.testing.assert_array_equal(original, self.weights)

    def test_pruning_repeats_when_removal_creates_another_zero_row(self):
        weights = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 2.]])
        transition, retained = prune_zero_rows(weights)
        np.testing.assert_array_equal(retained, [2])
        np.testing.assert_array_equal(transition, [[1.]])
        np.testing.assert_allclose(kp.pagerank(weights.T), [0, 0, 1])

    def test_row_and_column_pr_and_ppr_match_linear_system(self):
        for seeds in (None, [1], [1, 2]):
            expected = reference_scores(self.weights, seeds)
            column = kp.personalized_pagerank(self.weights.T, d=0.85, seed_nodes=seeds)
            row = ROW_HELPERS["ppr"](self.weights, seed_nodes=seeds)
            np.testing.assert_allclose(column, expected, rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(row, expected, atol=1e-11)
            self.assertAlmostEqual(column.sum(), 1)
            self.assertEqual(column[0], 0)
        np.testing.assert_allclose(kp.pagerank(self.weights.T, d=0.85),
                                   reference_scores(self.weights), rtol=2e-5)
        np.testing.assert_allclose(ROW_HELPERS["pagerank"](self.weights),
                                   reference_scores(self.weights), atol=1e-11)

    def test_removed_seeds_raise_instead_of_changing_objective(self):
        for seeds in ([0], [0, 1]):
            with self.assertRaisesRegex(ValueError, "Seed observables"):
                kp.personalized_pagerank(self.weights.T, seed_nodes=seeds)
            with self.assertRaisesRegex(ValueError, "Seed observables"):
                ROW_HELPERS["ppr"](self.weights, seed_nodes=seeds)
        cascade = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 1.]])
        with self.assertRaisesRegex(ValueError, "Seed observables"):
            kp.personalized_pagerank(cascade.T, seed_nodes=[1])

    def test_no_retained_graph_raises(self):
        for weights in (np.zeros((2, 2)), np.array([[0., 1.], [0., 0.]])):
            with self.assertRaisesRegex(ValueError, "every observable"):
                kp.pagerank(weights.T)
            with self.assertRaisesRegex(ValueError, "every observable"):
                ROW_HELPERS["pagerank"](weights)

    def test_nonzero_graph_preserves_legacy_column_results(self):
        matrix = np.array([[1., 2., 3.], [2., 1., 0.], [1., 4., 2.]])
        transition = matrix / matrix.sum(axis=0)
        for seeds in (None, [0, 2]):
            preference = np.ones(3) / 3 if seeds is None else np.array([.5, 0., .5])
            scores = np.ones(3) / 3
            for _ in range(100_000):
                next_scores = .15 * preference + .85 * transition.dot(scores)
                if np.allclose(scores, next_scores, atol=1e-10):
                    break
                scores = next_scores
            actual = kp.personalized_pagerank(matrix, d=.85, seed_nodes=seeds)
            np.testing.assert_allclose(actual, next_scores, atol=1e-14, rtol=1e-14)

    def test_orderings_exclude_removed_nodes_and_error_sizes_stay_nan(self):
        data = np.eye(3)
        orders = kp.compute_orderings(self.weights.T, 3, 1, np.array([1]), data, data)
        for order in orders.values():
            self.assertEqual(set(order), {1, 2})
        errors = kp.compute_errors(orders, data, data, data, data, 1, 3, [1])
        for error in errors.values():
            self.assertTrue(np.isnan(error[2]))

    def test_experiment_evaluation_rejects_unavailable_dictionary_sizes(self):
        args = [np.array([1, 2]), 3] + [None] * 6
        self.assertTrue(np.isnan(EVAL_HELPERS["err1"](*args)))
        result = EVAL_HELPERS["errM"](np.array([1, 2]), 3, None, None, 4, None)
        np.testing.assert_array_equal(np.isnan(result), np.ones(4, dtype=bool))

    def test_tiny_positive_rows_are_normalized_without_threshold_pruning(self):
        transition, retained = prune_zero_rows(np.diag([1e-20, 1.]))
        np.testing.assert_array_equal(retained, [0, 1])
        np.testing.assert_array_equal(transition, np.eye(2))

    def test_subsampled_variant_handles_pruning(self):
        weights = np.array([[0., 0., 0.], [1., 0., 0.], [0., 0., 2.]])
        np.testing.assert_allclose(kp.subsampled_pagerank(weights.T, subsample_rate=1),
                                   [0, 0, 1])

    def test_delay_selection_respects_original_retained_indices(self):
        scores = np.array([0., .4, .6])
        selected = kp.select_top_indices(scores, np.array([1]), 1, 2,
                                         group_by_delay=False, retained_indices=[1, 2])
        np.testing.assert_array_equal(selected, [1, 2])
        with self.assertRaisesRegex(ValueError, "retained observable count"):
            kp.select_top_indices(scores, np.array([1]), 1, 3,
                                  group_by_delay=False, retained_indices=[1, 2])
        with self.assertRaisesRegex(ValueError, "coordinate observables"):
            kp.select_top_indices(scores, np.array([0]), 1, 2,
                                  group_by_delay=False, retained_indices=[1, 2])

    def test_grouped_delay_selection_excludes_incomplete_groups(self):
        retained = [0, 1, 2, 3, 5, 6, 7, 8]
        scores = np.array([1., 1., 1., 10., 0., 10., 1., 1., 1.])
        selected = kp.select_top_indices(scores, np.arange(3), 3, 6,
                                         retained_indices=retained)
        np.testing.assert_array_equal(selected, [0, 1, 2, 6, 7, 8])
        with self.assertRaisesRegex(ValueError, "complete retained delay groups"):
            kp.select_top_indices(scores, np.arange(3), 3, 8,
                                  retained_indices=retained)


class PrunedDictionaryTests(unittest.TestCase):
    def test_196_columns_keep_state_seeds_and_rbf_index_mapping(self):
        rng = np.random.default_rng(17)
        angles = rng.uniform(-np.pi, np.pi, size=(1024, 2))
        dictionary, names, n_state = BUILD_DICTIONARY(angles[:, 0], angles[:, 1])
        self.assertEqual(dictionary.shape, (1024, 196))
        self.assertEqual(n_state, 4)
        self.assertEqual(names[:4], ["sin(φ)", "cos(φ)", "sin(ψ)", "cos(ψ)"])
        self.assertEqual(sum(name.startswith("G(") for name in names), 100)
        self.assertTrue(names[96].startswith("G("))
        self.assertFalse(any("φ+" in name for name in names))
        np.testing.assert_allclose(dictionary.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(dictionary.std(axis=0), 1, atol=1e-14)

    def test_diagonal_modes_are_in_the_retained_fourier_span(self):
        rng = np.random.default_rng(18)
        angles = rng.uniform(-np.pi, np.pi, size=(1024, 2))
        dictionary, _, _ = BUILD_DICTIONARY(angles[:, 0], angles[:, 1])
        # Include the constant because centering subtracts a mode's sample mean.
        span = np.column_stack([np.ones(len(angles)), dictionary[:, :96]])
        modes = []
        done = set()
        for k in range(-4, 5):
            for l in range(-4, 5):
                if 0 < abs(k) + abs(l) <= 4 and (k, l) not in done and (-k, -l) not in done:
                    modes.extend([np.sin(k * angles[:, 0] + l * angles[:, 1]),
                                  np.cos(k * angles[:, 0] + l * angles[:, 1])])
                    done.add((k, l))
        target = np.column_stack(modes)
        self.assertEqual(target.shape[1], 40)
        coefficients = np.linalg.lstsq(span, target, rcond=None)[0]
        np.testing.assert_allclose(span @ coefficients, target, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
