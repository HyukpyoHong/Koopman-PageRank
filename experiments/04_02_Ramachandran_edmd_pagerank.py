"""
Step 2: Build dictionary, compute EDMD, rank observables by PR, PPR, TICA, PCCA+.
Runs over 20 independent trajectories produced by 01_simulate.py.
Each trajectory produces one edmd_results_XX.npz file in OUTPUT_DIR.
"""
import os
import numpy as np
from deeptime.decomposition import TICA
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.clustering import KMeans as dtKMeans

DATA_DIR   = "data/"
OUTPUT_DIR = "results/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Load all trajectories ───
all_traj = np.load(os.path.join(DATA_DIR, "trajectories_20.npy"))  # (20, T, 2)
n_traj   = all_traj.shape[0]
print(f"Loaded {n_traj} trajectories, each {all_traj.shape[1]} frames")

# ═══════════════════════════════════════════════════════════════
# Helper functions (same for every trajectory)
# ═══════════════════════════════════════════════════════════════
def pagerank(P, alpha=0.85, tol=1e-12, max_iter=2000):
    n = P.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        pi_new = alpha * pi @ P + (1 - alpha) / n
        if np.abs(pi_new - pi).sum() < tol:
            break
        pi = pi_new
    return pi

def ppr(P, seed_nodes=None, alpha=0.85, tol=1e-12, max_iter=2000):
    n = P.shape[0]
    e = np.zeros(n)
    if seed_nodes is not None and len(seed_nodes) > 0:
        for s in seed_nodes:
            e[s] = 1.0 / len(seed_nodes)
    pi = e.copy()
    for _ in range(max_iter):
        pi_new = alpha * pi @ P + (1 - alpha) * e
        if np.abs(pi_new - pi).sum() < tol:
            break
        pi = pi_new
    return pi

def build_dictionary(phi, psi):
    """Build and standardize the observable dictionary. Returns (Psi, names, n_state)."""
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

    # Group 3: Diagonal Fourier |k|+|l|≤4
    done = set()
    for k in range(-4, 5):
        for l in range(-4, 5):
            if 0 < abs(k)+abs(l) <= 4 and (k,l) not in done and (-k,-l) not in done:
                obs.append(np.sin(k*phi + l*psi)); names.append(f"sin({k}φ+{l}ψ)")
                obs.append(np.cos(k*phi + l*psi)); names.append(f"cos({k}φ+{l}ψ)")
                done.add((k,l))

    # Group 4: Gaussian RBFs
    for cp in np.linspace(-np.pi, np.pi, n_rbf, endpoint=False):
        for cq in np.linspace(-np.pi, np.pi, n_rbf, endpoint=False):
            dp = np.arctan2(np.sin(phi-cp), np.cos(phi-cp))
            dq = np.arctan2(np.sin(psi-cq), np.cos(psi-cq))
            obs.append(np.exp(-0.5*(dp**2+dq**2)/sig**2))
            names.append(f"G({cp:.1f},{cq:.1f})")

    Psi = np.array(obs).T           # (N_frames, tilde_N)
    mu  = Psi.mean(axis=0)
    sd  = Psi.std(axis=0); sd[sd < 1e-10] = 1.0
    Psi = (Psi - mu) / sd
    return Psi, names, n_state

# ═══════════════════════════════════════════════════════════════
# Main loop over trajectories
# ═══════════════════════════════════════════════════════════════
for traj_i in range(n_traj):
    print(f"\n{'='*55}\nTrajectory {traj_i+1}/{n_traj}")

    traj     = all_traj[traj_i]          # (T, 2)
    phi, psi = traj[:, 0], traj[:, 1]

    # ── Dictionary ──
    Psi, names, n_state = build_dictionary(phi, psi)
    tilde_N = Psi.shape[1]
    print(f"  Dictionary: {tilde_N} observables")

    # ── EDMD ──
    X = Psi[:-1].T          # (tilde_N, M)
    Y = Psi[1:].T
    M = X.shape[1]

    G      = X @ X.T / M
    A_cov  = Y @ X.T / M
    eigs_G = np.linalg.eigvalsh(G)
    lam    = max(1e-5, 1e-3 * eigs_G.max())
    K_edmd = A_cov @ np.linalg.inv(G + lam * np.eye(tilde_N))

    K_abs  = np.abs(K_edmd)
    K_hat  = K_abs / np.maximum(K_abs.sum(axis=1, keepdims=True), 1e-12)
    print(f"  EDMD: M={M}, reg={lam:.2e}, ||K||_max={np.abs(K_edmd).max():.4f}")

    # ── PageRank ──
    pi_pr    = pagerank(K_hat, alpha=0.85)
    pr_order = np.argsort(-pi_pr)

    # ── PPR (multi-seed at state coords) ──
    pi_ppr    = ppr(K_hat, seed_nodes=list(range(n_state)), alpha=0.85)
    ppr_order = np.argsort(-pi_ppr)
    seed      = -1   # multi-seed flag

    # ── TICA ──
    tica       = TICA(lagtime=1, dim=min(20, tilde_N))
    tica_model = tica.fit_fetch(Psi)
    loadings   = np.abs(tica_model.singular_vectors_left[:, :10])
    tica_score = loadings.max(axis=1)
    tica_order = np.argsort(-tica_score)

    # ── PCCA+ via MSM ──
    n_clusters = 50
    km         = dtKMeans(n_clusters=n_clusters, n_jobs=1)
    km_model   = km.fit_fetch(traj)
    dtraj_km   = km_model.transform(traj)

    counts = TransitionCountEstimator(lagtime=5, count_mode='effective').fit_fetch(dtraj_km)
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

    valid = frame_macro >= 0
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

    # ── Save ──
    out_path = os.path.join(OUTPUT_DIR, f"edmd_results_{traj_i:02d}.npz")
    np.savez(out_path,
             K_edmd=K_edmd, K_hat=K_hat,
             pi_pr=pi_pr, pi_ppr=pi_ppr,
             pr_order=pr_order, ppr_order=ppr_order,
             tica_order=tica_order, pcca_order=pcca_order,
             tica_score=tica_score, fdr=fdr,
             obs_names=np.array(names, dtype=object),
             Psi=Psi, seed=seed, traj=traj,
             n_state=n_state, frame_macro=frame_macro)
    print(f"  Saved -> {out_path}")

print("\nAll done!")
