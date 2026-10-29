"""Alpha sensitivity on saved 196-observable, train-only trajectory fits.

Only the damping factor varies. Test statistics are used exclusively to score
the predefined candidates, never to fit scaling, rankings, or readout weights.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ALPHAS = [.05, .15, .25, .35, .5, .65, .75, .85, .9, .95]
SIZES = [5, 10, 20, 40]
BASE_ALPHA = .85


def resolvent_scores(transition, preference, alpha):
    """Return column score vectors for one or several preferences on row P."""
    scores = np.linalg.solve(np.eye(len(transition)) - alpha * transition.T,
                             (1 - alpha) * preference)
    if np.any(scores < -1e-10) or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid PageRank scores")
    residual = scores - alpha * transition.T @ scores - (1 - alpha) * preference
    if np.max(np.abs(residual)) > 1e-10:
        raise ValueError("PageRank solve failed its stationary equation")
    return scores


def readout_statistics(psi, ntr, n_state):
    """Sufficient statistics for the existing ridge readout and held-out MSE."""
    x = psi[:ntr]
    y = psi[1:ntr + 1, :n_state]
    xt = psi[ntr:-1]
    yt = psi[ntr + 1:, :n_state]
    if len(xt) == 0:
        raise ValueError("No held-out pairs")
    return dict(train_gram=x.T @ x / len(x), train_cross=y.T @ x / len(x),
                test_gram=xt.T @ xt / len(xt), test_cross=yt.T @ xt / len(xt),
                test_target_sq=float(np.sum(yt * yt) / len(yt)))


def readout_mse(statistics, selected):
    """Same ridge rule and MSE as Step 3, evaluated without repeated predictions."""
    selected = np.asarray(selected, dtype=int)
    g = statistics["train_gram"][np.ix_(selected, selected)]
    lam = max(1e-5, 1e-3 * np.linalg.eigvalsh(g).max())
    readout = statistics["train_cross"][:, selected] @ np.linalg.inv(g + lam * np.eye(len(g)))
    gt = statistics["test_gram"][np.ix_(selected, selected)]
    error = (statistics["test_target_sq"]
             - 2 * np.sum(readout * statistics["test_cross"][:, selected])
             + np.sum((readout @ gt) * readout))
    if error < -1e-10 * max(1., statistics["test_target_sq"]):
        raise ValueError("Negative MSE from numerical cancellation")
    return float(max(error, 0.))


def jaccard(a, b):
    a, b = set(map(int, a)), set(map(int, b))
    return len(a & b) / len(a | b)


def analyze_trajectory(path, trajectory_index, alphas=ALPHAS, sizes=SIZES):
    with np.load(path, allow_pickle=False) as result:
        if str(result["evaluation_protocol"]) != "train_only_v1":
            raise ValueError(f"{path}: wrong evaluation protocol")
        psi = result["Psi"]
        ntr = int(result["n_train_pairs"])
        n_state = int(result["n_state"])
        if (psi.shape[1] != 196 or ntr != int(.8 * (len(psi) - 1))
                or int(result["preprocessing_fit_stop"]) != ntr + 1):
            raise ValueError(f"{path}: invalid dictionary or train/test split")
        p = result["K_hat"]
        retained = result["ppr_retained_indices"]
        if (len(retained) < max(sizes) or p.shape != (len(retained), len(retained))
                or not np.allclose(p.sum(axis=1), 1, atol=1e-12) or np.any(p < 0)):
            raise ValueError(f"{path}: invalid retained transition matrix")
        if not np.all(np.isin(np.arange(n_state), retained)):
            raise ValueError(f"{path}: a state-coordinate seed was removed")
        preferences = np.column_stack([
            np.ones(len(p)) / len(p), (retained < n_state).astype(float) / n_state])
        baseline_scores = resolvent_scores(p, preferences, BASE_ALPHA)
        baseline_orders = [retained[np.argsort(-baseline_scores[:, i])] for i in range(2)]
        statistics = readout_statistics(psi, ntr, n_state)
        error_cache = {}

        def error(order, n):
            # Ridge fitting is invariant under permutations of the selected columns.
            key = tuple(sorted(map(int, order[:n])))
            if len(key) != n or len(set(key)) != n:
                raise ValueError("Invalid selected dictionary")
            if key not in error_cache:
                error_cache[key] = readout_mse(statistics, key)
            return error_cache[key]

        rows = []
        for alpha in alphas:
            scores = resolvent_scores(p, preferences, alpha)
            for col, method in enumerate(["PR", "PPR"]):
                order = retained[np.argsort(-scores[:, col])]
                for n in sizes:
                    rows.append(dict(trajectory=trajectory_index, method=method, alpha=alpha, N=n,
                                     mse=error(order, n),
                                     jaccard=jaccard(order[:n], baseline_orders[col][:n]),
                                     state_coordinates_selected=int(np.count_nonzero(order[:n] < n_state)),
                                     selected_indices=order[:n].tolist()))

        comparisons = []
        baseline_methods = dict(PR=baseline_orders[0], PPR=baseline_orders[1],
                                TICA=result["tica_order"], PCCA=result["pcca_order"],
                                Random=np.random.RandomState(trajectory_index).permutation(196))
        for method, order in baseline_methods.items():
            for n in sizes:
                comparisons.append(dict(trajectory=trajectory_index, method=method, N=n,
                                        mse=error(order, n)))
        checks = {}
        for i, method in enumerate(["PR", "PPR"]):
            cached = result["pi_pr" if method == "PR" else "pi_ppr"][retained]
            checks[method] = dict(max_abs_score_difference=float(np.max(np.abs(cached - baseline_scores[:, i]))),
                                  max_stationary_residual=float(np.max(np.abs(
                                      cached - BASE_ALPHA * p.T @ cached
                                      - (1 - BASE_ALPHA) * preferences[:, i]))),
                                  cached_order_jaccard={str(n): jaccard(
                                      result["pr_order" if method == "PR" else "ppr_order"][:n],
                                      baseline_orders[i][:n]) for n in sizes})
        return rows, comparisons, dict(trajectory=trajectory_index, baseline_checks=checks,
                                       retained_observables=len(retained), unique_readout_fits=len(error_cache))


def summarize(rows, comparisons, alphas=ALPHAS, sizes=SIZES):
    summary = []
    trajectories = sorted({r["trajectory"] for r in rows})
    for method in ["PR", "PPR"]:
        for n in sizes:
            lookup = {(r["trajectory"], r["alpha"]): r for r in rows
                      if r["method"] == method and r["N"] == n}
            errors = np.array([[lookup[t, a]["mse"] for a in alphas] for t in trajectories])
            overlaps = np.array([[lookup[t, a]["jaccard"] for a in alphas] for t in trajectories])
            mean_error = errors.mean(axis=0)
            cvs = errors.std(axis=1) / errors.mean(axis=1)
            summary.append(dict(method=method, N=n, n_trajectories=len(trajectories),
                                min_mean_mse=float(mean_error.min()), max_mean_mse=float(mean_error.max()),
                                cv_of_mean_mse=float(mean_error.std() / mean_error.mean()),
                                mean_trajectory_cv=float(cvs.mean()), max_trajectory_cv=float(cvs.max()),
                                min_mean_jaccard=float(overlaps.mean(axis=0).min()),
                                min_trajectory_jaccard=float(overlaps.min()),
                                by_alpha=[dict(alpha=a, mean_mse=float(errors[:, i].mean()),
                                               std_mse=float(errors[:, i].std()),
                                               mean_jaccard=float(overlaps[:, i].mean()),
                                               std_jaccard=float(overlaps[:, i].std()))
                                          for i, a in enumerate(alphas)]))
    baseline = []
    values = {(r["trajectory"], r["method"], r["N"]): r["mse"] for r in comparisons}
    for method in ["PR", "PPR", "TICA", "PCCA", "Random"]:
        for n in sizes:
            errors = np.array([values[t, method, n] for t in trajectories])
            random_errors = np.array([values[t, "Random", n] for t in trajectories])
            ratios = errors / random_errors
            baseline.append(dict(method=method, N=n, mean_mse=float(errors.mean()),
                                 std_mse=float(errors.std()), ratio_of_means=float(errors.mean() / random_errors.mean()),
                                 mean_paired_ratio=float(ratios.mean()), std_paired_ratio=float(ratios.std())))
    return summary, baseline


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), sharex=True)
    for i, n in enumerate(SIZES):
        for method, color in [("PR", "#2a6fb8"), ("PPR", "#dd3d26")]:
            row = next(r for r in summary if r["method"] == method and r["N"] == n)
            x = [v["alpha"] for v in row["by_alpha"]]
            axes[0, i].plot(x, [v["mean_mse"] for v in row["by_alpha"]], "o-", color=color, label=method)
            axes[1, i].plot(x, [v["mean_jaccard"] for v in row["by_alpha"]], "o-", color=color)
        axes[0, i].set_title(f"N = {n}")
        axes[1, i].set_xlabel("Damping factor α")
        axes[1, i].set_ylim(0, 1.05)
        for ax in axes[:, i]:
            ax.grid(alpha=.2)
    axes[0, 0].set_ylabel("Mean held-out MSE")
    axes[1, 0].set_ylabel("Mean Jaccard vs α = 0.85")
    axes[0, 0].legend()
    fig.suptitle("Ramachandran: 196 observables, train-only fitting, 20 trajectories")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parent / "results_196_train_only")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).parents[1] / "validation/alpha_sensitivity_196_train_only")
    parser.add_argument("--n-trajectories", type=int, default=20)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, comparisons, checks = [], [], []
    for i in range(args.n_trajectories):
        path = args.results_dir / f"edmd_results_{i:02d}.npz"
        trajectory_rows, baseline, check = analyze_trajectory(path, i)
        rows.extend(trajectory_rows)
        comparisons.extend(baseline)
        checks.append(check)
        print(f"Trajectory {i+1}/{args.n_trajectories}: alpha sweep complete", flush=True)
    summary, baseline = summarize(rows, comparisons)
    report = dict(dictionary_size=196, evaluation_protocol="train_only_v1", alphas=ALPHAS,
                  N_values=SIZES, baseline_alpha=BASE_ALPHA, std_ddof=0,
                  metric="Mean summed squared residual across four training-standardized circular observables",
                  seed_policy="fixed state-coordinate seeds; no fallback or forced dictionary inclusion",
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  per_trajectory=rows, summary=summary, baseline_comparison=baseline,
                  baseline_per_trajectory=comparisons, validation=checks)
    (args.output_dir / "alpha_sensitivity_results.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "alpha_sensitivity_summary.csv").open("w", newline="") as file:
        columns = [k for k in summary[0] if k != "by_alpha"]
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{k: row[k] for k in columns} for row in summary])
    plot_summary(summary, args.output_dir / "alpha_sensitivity.png")
    print("Saved results, summary CSV and plot to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
