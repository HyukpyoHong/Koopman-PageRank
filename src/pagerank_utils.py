"""Zero-row pruning and index bookkeeping for EDMD PageRank.

Matrices in this module use rows as sources. Removed observables receive zero
in a score vector of the original size; rankings must use the retained indices.
"""

import numpy as np


def prune_zero_rows(weights):
    """Remove zero-outgoing nodes until stable, then normalize remaining rows.

    Returns ``(transition, retained_indices)`` in the original index convention.
    Removal is iterative because deleting a column can create another zero row.
    An empty retained graph has no PageRank distribution and raises ValueError.
    """
    matrix = np.asarray(weights, dtype=float)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
            or matrix.shape[0] == 0):
        raise ValueError("PageRank weights must be a nonempty square matrix")
    if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
        raise ValueError("PageRank weights must be finite and nonnegative")

    retained = np.arange(matrix.shape[0])
    while True:
        row_sums = matrix.sum(axis=1)
        keep = row_sums > 0
        if np.all(keep):
            return matrix / row_sums[:, None], retained
        if not np.any(keep):
            raise ValueError("Zero-row pruning removed every observable")
        retained = retained[keep]
        matrix = matrix[np.ix_(keep, keep)]


def retained_preference(retained, original_size, seed_nodes=None):
    """Uniform preference on retained nodes, or on explicitly supplied seeds.

    Seeds use original observable indices. A removed seed is an error, rather
    than a silent change of the prediction objective.
    """
    if seed_nodes is None or len(seed_nodes) == 0:
        return np.full(len(retained), 1.0 / len(retained))
    seeds = np.asarray(seed_nodes)
    if (seeds.ndim != 1 or not np.issubdtype(seeds.dtype, np.integer)
            or np.any(seeds < 0) or np.any(seeds >= original_size)):
        raise ValueError("Seed nodes must be valid original integer indices")
    if len(np.unique(seeds)) != len(seeds):
        raise ValueError("Seed nodes must be distinct")
    removed = seeds[~np.isin(seeds, retained)]
    if len(removed):
        raise ValueError(f"Seed observables {removed.tolist()} were removed by zero-row pruning")
    return np.isin(retained, seeds).astype(float) / len(seeds)


def expand_scores(scores, retained, original_size):
    """Restore the original observable indexing, assigning removed nodes zero."""
    result = np.zeros(original_size)
    result[retained] = scores
    return result
