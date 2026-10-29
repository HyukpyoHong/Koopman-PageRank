"""Reproduce the appendix toy example and normalization comparison on saved fits."""

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from pagerank_utils import prune_zero_rows, retained_preference

HELPER_PATH = HERE / "04_04_Ramachandran_alpha_sensitivity.py"
_spec = importlib.util.spec_from_file_location("alpha_readout_helpers", HELPER_PATH)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)

ALPHA = .85
SIZES = [5, 10, 20, 40]


def normalization_scores(weights, seeds, alpha=ALPHA):
    """Compare row and global normalization on the same retained row graph.

    Both scores use (1-alpha) * inverse(I-alpha*T.T) * preference. The
    globally scaled graph is substochastic; its scores need not sum to one.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one")
    weights = np.asarray(weights, dtype=float)
    transition, retained = prune_zero_rows(weights)
    preference = retained_preference(retained, len(weights), seeds)
    kept_weights = weights[np.ix_(retained, retained)]
    rmax = float(kept_weights.sum(axis=1).max())
    return dict(
        retained=retained, transition=transition, rmax=rmax,
        row_normalized=_helpers.resolvent_scores(transition, preference, alpha),
        globally_normalized=_helpers.resolvent_scores(
            kept_weights / rmax, preference, alpha))


def toy_example():
    weights = np.array([[.90, .05, .05], [.60, .40, 0.], [.03, 0., .07]])
    scores = normalization_scores(weights, [0])
    result = dict(W=weights.tolist(), seed=[1., 0., 0.], alpha=ALPHA,
                  rmax=scores["rmax"])
    for method in ["row_normalized", "globally_normalized"]:
        values = scores[method]
        result[method] = dict(scores=values.tolist(), score_sum=float(values.sum()),
                              selected_at_N2=(np.argsort(-values)[:2] + 1).tolist())
    return result


def analyze_trajectory(path, trajectory_index, sizes=SIZES):
    """Rank with the saved training fit, then score the saved held-out frames."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["evaluation_protocol"]) != "train_only_v1":
            raise ValueError(f"{path}: expected train_only_v1")
        psi = data["Psi"]
        ntr, n_state = int(data["n_train_pairs"]), int(data["n_state"])
        if (int(data["dictionary_size"]) != 196 or psi.shape[1] != 196
                or n_state != 4 or ntr != int(.8 * (len(psi) - 1))
                or int(data["preprocessing_fit_stop"]) != ntr + 1):
            raise ValueError(f"{path}: invalid dictionary or training split")
        # The saved column-prediction matrix corresponds to K^T in the paper.
        weights = np.abs(data["K_edmd"])
        if weights.shape != (196, 196):
            raise ValueError(f"{path}: expected a 196 by 196 EDMD matrix")
        scores = normalization_scores(weights, list(range(n_state)))
        retained = scores["retained"]
        cached_p = data["K_hat"]
        if (len(retained) < max(sizes)
                or not np.array_equal(retained, data["ppr_retained_indices"])
                or cached_p.shape != scores["transition"].shape
                or not np.allclose(cached_p, scores["transition"], rtol=1e-12, atol=1e-12)):
            raise ValueError(f"{path}: cached transition does not match the EDMD weights")
        statistics = _helpers.readout_statistics(psi, ntr, n_state)
        rows = []
        for method in ["row_normalized", "globally_normalized"]:
            order = retained[np.argsort(-scores[method])]
            for n in sizes:
                selected = order[:n]
                rows.append(dict(trajectory=trajectory_index, method=method, N=n,
                                 mse=_helpers.readout_mse(statistics, selected),
                                 selected_indices=selected.tolist()))
        return rows


def summarize(rows, sizes=SIZES):
    summary = []
    for method in ["row_normalized", "globally_normalized"]:
        for n in sizes:
            errors = np.array([r["mse"] for r in rows
                               if r["method"] == method and r["N"] == n])
            summary.append(dict(method=method, N=n, n_trajectories=len(errors),
                                mean_mse=float(errors.mean()),
                                std_mse=float(errors.std(ddof=0))))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy-only", action="store_true",
                        help="Print the 3x3 example without loading caches or writing files")
    parser.add_argument("--results-dir", type=Path, default=HERE / "results_196_train_only")
    parser.add_argument("--output-dir", type=Path,
                        default=HERE.parent / "validation/normalization_comparison_196_train_only")
    parser.add_argument("--n-trajectories", type=int, default=20)
    args = parser.parse_args()
    toy = toy_example()
    print(json.dumps(toy, indent=2), flush=True)
    if args.toy_only:
        return
    if args.n_trajectories < 1:
        parser.error("--n-trajectories must be positive")
    rows = []
    for i in range(args.n_trajectories):
        path = args.results_dir / f"edmd_results_{i:02d}.npz"
        if not path.is_file():
            parser.error(f"Missing {path}; run Step 2 to generate train-only fits first")
        rows.extend(analyze_trajectory(path, i))
        print(f"Trajectory {i+1}/{args.n_trajectories}: normalization comparison complete",
              flush=True)
    summary = summarize(rows)
    report = dict(dictionary_size=196, evaluation_protocol="train_only_v1", alpha=ALPHA,
                  N=SIZES, n_trajectories=args.n_trajectories, std_ddof=0,
                  seed_policy="four state-coordinate seeds; no fallback or forced inclusion",
                  globally_normalized_score="(1-alpha) * inverse(I-alpha*(W/rmax).T) * seed",
                  toy=toy, summary=summary, per_trajectory=rows,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  readout_helper_sha256=hashlib.sha256(HELPER_PATH.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "normalization_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "normalization_comparison_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print("Saved comparison JSON and summary CSV to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
