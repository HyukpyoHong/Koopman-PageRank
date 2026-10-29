"""
Step 2: Build the pruned 196-observable dictionary, compute EDMD and rankings.
Runs over 20 independent trajectories produced by 01_simulate.py.
Each trajectory produces one edmd_results_XX.npz file in OUTPUT_DIR.
"""
import os
import sys
from pathlib import Path
import numpy as np
from deeptime.decomposition import TICA
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.clustering import KMeans as dtKMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pagerank_utils import prune_zero_rows, retained_preference, expand_scores

DATA_DIR = os.environ.get("KOOPMAN_RAMACHANDRAN_DATA_DIR", "data/")
OUTPUT_DIR = os.environ.get("KOOPMAN_RAMACHANDRAN_RESULTS_DIR", "results_196_train_only/")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load all trajectories ───
all_traj = np.load(os.path.join(DATA_DIR, "trajectories_20.npy"))  # (20, T, 2)
n_traj   = all_traj.shape[0]
print(f"Loaded {n_traj} trajectories, each {all_traj.shape[1]} frames")

# ═══════════════════════════════════════════════════════════════
# Helper functions (same for every trajectory)
# ═══════════════════════════════════════════════════════════════
def pagerank(P, alpha=0.85, tol=1e-12, max_iter=2000):
    """PR from row weights, pruning zero-outgoing nodes before normalization."""
    original_size = P.shape[0]
    P, retained = prune_zero_rows(P)
    n = len(retained)
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = alpha * pi @ P + (1 - alpha) / n
        if np.abs(pi_new - pi).sum() < tol:
            return expand_scores(pi_new, retained, original_size)
        pi = pi_new
    return expand_scores(pi, retained, original_size)

def ppr(P, seed_nodes=None, alpha=0.85, tol=1e-12, max_iter=2000):
    """PPR from row weights; seed nodes retain their original indices."""
    original_size = P.shape[0]
    P, retained = prune_zero_rows(P)
    e = retained_preference(retained, original_size, seed_nodes)
    pi = e.copy()
    for _ in range(max_iter):
        pi_new = alpha * pi @ P + (1 - alpha) * e
        if np.abs(pi_new - pi).sum() < tol:
            return expand_scores(pi_new, retained, original_size)
        pi = pi_new
    return expand_scores(pi, retained, original_size)

def build_dictionary(phi, psi, fit_stop=None, return_scaling=False):
    """Build the 196-observable dictionary without redundant diagonal Fourier modes.

    Scaling is fitted on frames [:fit_stop] and applied to all frames.
    Returns (Psi, names, n_state); return_scaling also returns the fitted mean/std.
    State coordinates remain the first four columns.
    """
    obs, names = [], []
    n_rbf = 10
    sig   = 0.45

    # Group 0: State coordinates
    for func, fn, var, v in [(np.sin,"sin","φ",phi),(np.cos,"cos","φ",phi),
                              (np.sin,"sin","ψ",psi),(np.cos,"cos","ψ",psi)]:
        obs.append(func(v)); names.append(f"{fn}({var})")
    n_state = 4

    # Group 1: Higher Fourier harmonics k=2..8
    for k in range(2, 9):
        for func, fn, var, v in [(np.sin,"sin","φ",phi),(np.cos,"cos","φ",phi),
                                  (np.sin,"sin","ψ",psi),(np.cos,"cos","ψ",psi)]:
            obs.append(func(k*v)); names.append(f"{fn}({k}{var})")

    # Group 2: Fourier cross terms k,l=1..4
    for k in range(1, 5):
        for l in range(1, 5):
            for f1, n1 in [(np.sin,"s"),(np.cos,"c")]:
                for f2, n2 in [(np.sin,"s"),(np.cos,"c")]:
                    obs.append(f1(k*phi)*f2(l*psi))
                    names.append(f"{n1}{k}φ·{n2}{l}ψ")

    # Group 3: Gaussian RBFs. Diagonal Fourier modes are already in Groups 0--2.
    for cp in np.linspace(-np.pi, np.pi, n_rbf, endpoint=False):
        for cq in np.linspace(-np.pi, np.pi, n_rbf, endpoint=False):
            dp = np.arctan2(np.sin(phi-cp), np.cos(phi-cp))
            dq = np.arctan2(np.sin(psi-cq), np.cos(psi-cq))
            obs.append(np.exp(-0.5*(dp**2+dq**2)/sig**2))
            names.append(f"G({cp:.1f},{cq:.1f})")

    Psi = np.array(obs).T           # (N_frames, tilde_N)
    fit_stop = len(Psi) if fit_stop is None else fit_stop
    if not isinstance(fit_stop, (int, np.integer)) or not 1 <= fit_stop <= len(Psi):
        raise ValueError("fit_stop must select a nonempty training prefix")
    mu  = Psi[:fit_stop].mean(axis=0)
    sd  = Psi[:fit_stop].std(axis=0); sd[sd < 1e-10] = 1.0
    Psi = (Psi - mu) / sd
    if return_scaling:
        return Psi, names, n_state, mu, sd
    return Psi, names, n_state

# ═══════════════════════════════════════════════════════════════
# Main loop over trajectories
# ═══════════════════════════════════════════════════════════════
def fit_trajectory(traj, random_seed=0):
    """Fit all rankings on the training prefix, then transform the held-out tail.

    The last training frame is also the known initial state of the first test
    pair. No test target is used to fit scaling, EDMD, TICA, clustering, or MSM.
    """
    traj = np.asarray(traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] != 2:
        raise ValueError("Expected a trajectory with shape (n_frames, 2)")
    phi, psi = traj[:, 0], traj[:, 1]
    n_train_pairs = int(0.8 * (len(traj) - 1))
    train_stop = n_train_pairs + 1
    if n_train_pairs < 1 or train_stop >= len(traj):
        raise ValueError("Trajectory must contain both training pairs and test targets")

    # ── Dictionary ──
    Psi, names, n_state, dictionary_mean, dictionary_std = build_dictionary(
        phi, psi, fit_stop=train_stop, return_scaling=True)
    Psi_train = Psi[:train_stop]
    traj_train = traj[:train_stop]
    tilde_N = Psi.shape[1]
    if tilde_N != 196:
        raise ValueError(f"Expected the pruned 196-observable dictionary, got {tilde_N}")
    print(f"  Dictionary: {tilde_N} observables")

    # ── EDMD ──
    X = Psi_train[:-1].T          # Only pairs whose targets are in training.
    Y = Psi_train[1:].T
    M = X.shape[1]

    G      = X @ X.T / M
    A_cov  = Y @ X.T / M
    eigs_G = np.linalg.eigvalsh(G)
    lam    = max(1e-5, 1e-3 * eigs_G.max())
    K_edmd = A_cov @ np.linalg.inv(G + lam * np.eye(tilde_N))

    K_abs  = np.abs(K_edmd)
    K_hat, ppr_retained_indices = prune_zero_rows(K_abs)
    print(f"  EDMD: M={M}, reg={lam:.2e}, ||K||_max={np.abs(K_edmd).max():.4f}")

    # ── PageRank ──
    pi_pr    = pagerank(K_abs, alpha=0.85)
    pr_order = ppr_retained_indices[np.argsort(-pi_pr[ppr_retained_indices])]

    # ── PPR (multi-seed at state coords) ──
    pi_ppr    = ppr(K_abs, seed_nodes=list(range(n_state)), alpha=0.85)
    ppr_order = ppr_retained_indices[np.argsort(-pi_ppr[ppr_retained_indices])]
    seed      = -1   # multi-seed flag

    # ── TICA ──
    tica       = TICA(lagtime=1, dim=min(20, tilde_N))
    tica_model = tica.fit_fetch(Psi_train)
    loadings   = np.abs(tica_model.singular_vectors_left[:, :10])
    tica_score = loadings.max(axis=1)
    tica_order = np.argsort(-tica_score)

    # ── PCCA+ via MSM ──
    n_clusters = 50
    km         = dtKMeans(n_clusters=n_clusters, n_jobs=1, fixed_seed=random_seed)
    km_model   = km.fit_fetch(traj_train)
    dtraj_train = km_model.transform(traj_train)
    dtraj_km   = km_model.transform(traj)

    counts = TransitionCountEstimator(lagtime=5, count_mode='effective').fit_fetch(dtraj_train)
    counts = counts.submodel_largest()
    msm    = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)

    n_macro     = 3
    pcca        = msm.pcca(n_macro)
    memberships = pcca.memberships
    active_set  = counts.states

    frame_macro = np.full(len(traj), -1, dtype=int)
    for i, d in enumerate(dtraj_km):
        loc = np.where(active_set == d)[0]
        if len(loc) > 0:
            frame_macro[i] = np.argmax(memberships[loc[0]])

    # Test labels may be transformed for plots, but cannot affect the ranking.
    valid = (np.arange(len(traj)) < train_stop) & (frame_macro >= 0)
    fdr   = np.zeros(tilde_N)
    for j in range(tilde_N):
        vals_j     = Psi[valid, j]
        labels_j   = frame_macro[valid]
        grand_mean = vals_j.mean()
        ss_between = ss_within = 0.0
        for c in range(n_macro):
            mask = labels_j == c
            if mask.sum() < 2:
                continue
            c_mean      = vals_j[mask].mean()
            ss_between += mask.sum() * (c_mean - grand_mean)**2
            ss_within  += mask.sum() * vals_j[mask].var()
        fdr[j] = ss_between / max(ss_within, 1e-12)
    pcca_order = np.argsort(-fdr)

    return dict(
             K_edmd=K_edmd, K_hat=K_hat,
             dictionary_size=tilde_N,
             evaluation_protocol="train_only_v1",
             n_train_pairs=n_train_pairs, preprocessing_fit_stop=train_stop,
             dictionary_mean=dictionary_mean, dictionary_std=dictionary_std,
             fit_seed=random_seed,
             ppr_retained_indices=ppr_retained_indices,
             pi_pr=pi_pr, pi_ppr=pi_ppr,
             pr_order=pr_order, ppr_order=ppr_order,
             tica_order=tica_order, pcca_order=pcca_order,
             tica_score=tica_score, fdr=fdr,
             obs_names=np.array(names, dtype=object),
             Psi=Psi, seed=seed, traj=traj,
             n_state=n_state, frame_macro=frame_macro)


# Main pipeline; helper-only tests do not execute this loop.
for traj_i in range(n_traj):
    print(f"\n{'='*55}\nTrajectory {traj_i+1}/{n_traj}")
    result = fit_trajectory(all_traj[traj_i], random_seed=traj_i)
    out_path = os.path.join(OUTPUT_DIR, f"edmd_results_{traj_i:02d}.npz")
    np.savez(out_path, **result)
    print(f"  Saved -> {out_path}")

print("\nAll done!")
