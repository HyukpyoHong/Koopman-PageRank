"""
Step 3: Comprehensive evaluation and publication-quality figures.
Aggregates results over 20 independent trajectories to produce error bars.
"""
import os, json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = os.environ.get("KOOPMAN_RAMACHANDRAN_RESULTS_DIR", "results_196_train_only/")
PLOT_N_MAX = int(os.environ.get("KOOPMAN_RAMACHANDRAN_PLOT_N_MAX", "75"))
if not 4 < PLOT_N_MAX <= 80:
    raise ValueError("Plot upper bound must be greater than 4 and at most 80")
PLOT_ONLY = os.environ.get("KOOPMAN_RAMACHANDRAN_PLOT_ONLY", "0") == "1"
FIG = os.environ.get("KOOPMAN_RAMACHANDRAN_FIGURE_DIR",
                     f"figures_196_train_only_N{PLOT_N_MAX}/")
EVALUATION = Path(os.environ.get("KOOPMAN_RAMACHANDRAN_EVALUATION_DIR",
                                  "../validation/full_evaluation_196_train_only"))
os.makedirs(FIG, exist_ok=True)
EVALUATION.mkdir(parents=True, exist_ok=True)


plt.rcParams.update({
    "font.family": "sans-serif",
    "font.serif": ['Arial'],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

C  = {'PR':'#2a6fb8','PPR':'#dd3d26','TICA':'#2A9D8F','PCCA+':'#E9C46A','Random':'#AAAAAA'}
LS = {'PR':'-','PPR':'-','TICA':'--','PCCA+':'-.','Random':':'}
LW = {'PR':2.2,'PPR':2.2,'TICA':1.5,'PCCA+':1.5,'Random':1.2}
pn = ['Random','PCCA+','TICA','PR','PPR']

Nr     = np.arange(4, 81)
n_traj = 20

# ═══════════════════════════════════════════════════════════════
# Error functions (operate on arrays passed in, no globals)
# ═══════════════════════════════════════════════════════════════
def validate_training_result(result, n_frames):
    """Reject caches fitted on test data or using a different split."""
    required = {"evaluation_protocol", "n_train_pairs", "preprocessing_fit_stop",
                "dictionary_mean", "dictionary_std"}
    if not required.issubset(result.keys()):
        raise ValueError("Cache lacks train-only metadata; regenerate Step 2")
    if str(result["evaluation_protocol"]) != "train_only_v1":
        raise ValueError("Cache uses a different evaluation protocol; regenerate Step 2")
    ntr = int(result["n_train_pairs"])
    if (ntr != int(0.8 * (n_frames - 1)) or not 0 < ntr < n_frames - 1
            or int(result["preprocessing_fit_stop"]) != ntr + 1):
        raise ValueError("Cache training split is inconsistent with this trajectory")
    mean = np.asarray(result["dictionary_mean"])
    scale = np.asarray(result["dictionary_std"])
    if (mean.shape != (196,) or scale.shape != (196,)
            or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale))
            or np.any(scale <= 0)):
        raise ValueError("Cache dictionary scaling metadata is invalid")
    return ntr


def _solve(idx, Xtr, Ytr, Ytr_si, ntr):
    Xs   = Xtr[idx]
    G    = Xs @ Xs.T / ntr
    eigs = np.linalg.eigvalsh(G)
    lam  = max(1e-5, 1e-3 * eigs.max())
    Ginv  = np.linalg.inv(G + lam * np.eye(len(idx)))
    Ksub  = (Ytr[idx]  @ Xs.T / ntr) @ Ginv
    Kread = (Ytr_si    @ Xs.T / ntr) @ Ginv
    return Ksub, Kread

def err1(order, N, Xtr, Ytr, Ytr_si, Xt, Yt_si, ntr):
    if N > len(order):
        return np.nan  # This dictionary size is unavailable after pruning.
    idx = order[:N]
    _, Kread = _solve(idx, Xtr, Ytr, Ytr_si, ntr)
    return np.mean(np.sum((Yt_si - Kread @ Xt[idx])**2, axis=0))

def errM(order, N, Psi, si, steps, ntr):
    """Score Kread @ Ksub**(h-1) on starts with all requested horizons available."""
    if N > len(order):
        return np.full(steps, np.nan)
    idx = order[:N]
    Xtr   = Psi[:ntr].T
    Ytr   = Psi[1:ntr+1].T
    Ytr_si = Ytr[si]
    Ksub, Kread = _solve(idx, Xtr, Ytr, Ytr_si, ntr)
    st  = ntr
    nt  = Psi.shape[0] - st - steps
    if nt <= 0:
        raise ValueError("No test starts have all requested horizons available")
    # Each column is an independent test start; preserve readout-before-propagation.
    z = Psi[st:st+nt, idx].T.copy()
    errs = np.empty(steps)
    for h in range(1, steps + 1):
        prediction = Kread @ z
        target = Psi[st+h:st+h+nt][:, si].T
        errs[h-1] = np.mean(np.sum((target - prediction)**2, axis=0))
        z = Ksub @ z
    return errs


def one_step_statistics(psi, ntr, si):
    """Training fit and held-out scoring statistics; no fit uses test data."""
    x, y = psi[:ntr], psi[1:ntr+1][:, si]
    xt, yt = psi[ntr:-1], psi[ntr+1:][:, si]
    return dict(gram=x.T @ x / len(x), cross=y.T @ x / len(x),
                test_gram=xt.T @ xt / len(xt), test_cross=yt.T @ xt / len(xt),
                target_sq=float(np.sum(yt * yt) / len(yt)))


def one_step_from_statistics(order, n, statistics):
    """Identical ridge readout/MSE to err1, avoiding repeated sample products."""
    idx = order[:n]
    if len(idx) != n:
        return np.nan
    g = statistics["gram"][np.ix_(idx, idx)]
    lam = max(1e-5, 1e-3 * np.linalg.eigvalsh(g).max())
    readout = statistics["cross"][:, idx] @ np.linalg.inv(g + lam * np.eye(n))
    gt = statistics["test_gram"][np.ix_(idx, idx)]
    error = (statistics["target_sq"]
             - 2 * np.sum(readout * statistics["test_cross"][:, idx])
             + np.sum((readout @ gt) * readout))
    if not np.isfinite(error) or error < -1e-10:
        raise ValueError("Invalid one-step MSE")
    return float(max(error, 0.))


def paired_ratio_summary(method_errors, random_errors):
    """Empirical mean/std of paired trajectory ratios, plus ratio of means."""
    ratios = np.asarray(method_errors) / np.asarray(random_errors)
    return ratios.mean(axis=0), ratios.std(axis=0), (np.mean(method_errors, axis=0)
                                                     / np.mean(random_errors, axis=0))


figure_axes_limits = {}

def save_figure(fig, name):
    figure_axes_limits[name] = [list(ax.get_xlim()) for ax in fig.axes]
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════
# Aggregate over all trajectories
# ═══════════════════════════════════════════════════════════════
# E1_all[method][traj_i] = array of shape (len(Nr),)
E1_all = {nm: [] for nm in ['PR','PPR','TICA','PCCA+','Random']}
EM_all = {nm: [] for nm in ['PR','PPR','TICA','PCCA+','Random']}

# We store one representative result (traj 0) for the non-error-bar figures
rep = None

Nms   = 15
steps = 30

if PLOT_ONLY:
    with np.load(EVALUATION / "trajectory_errors.npz", allow_pickle=False) as cached:
        if (not np.array_equal(cached["N"], Nr) or int(cached["N_multistep"]) != Nms
                or not np.array_equal(cached["horizons"], np.arange(1, steps+1))):
            raise ValueError("Cached evaluation grid differs from the expected full grid")
        E1_all = {nm: cached["E1_"+nm].copy() for nm in E1_all}
        EM_all = {nm: cached["EM_"+nm].copy() for nm in EM_all}
        if any(v.shape != (n_traj, len(Nr)) or not np.all(np.isfinite(v)) for v in E1_all.values()):
            raise ValueError("Incomplete one-step cache")
        if any(v.shape != (n_traj, steps) or not np.all(np.isfinite(v)) for v in EM_all.values()):
            raise ValueError("Incomplete multi-step cache")
    print("Plot-only: reusing full N=4..80 evaluation; no error arrays will be rewritten", flush=True)

for traj_i in range(1 if PLOT_ONLY else n_traj):
    path = os.path.join(OUT, f"edmd_results_{traj_i:02d}.npz")
    d    = np.load(path, allow_pickle=True)

    K        = d['K_edmd']
    Psi      = d['Psi']
    pr_o     = d['pr_order']
    ppr_o    = d['ppr_order']
    tica_o   = d['tica_order']
    pcca_o   = d['pcca_order']
    ns       = int(d['n_state'])
    tN       = K.shape[0]
    if K.shape != (196, 196) or Psi.shape[1] != 196:
        raise ValueError(f"{path}: expected 196-observable results; regenerate Step 2")
    si       = np.arange(ns)

    M    = Psi.shape[0] - 1
    ntr  = validate_training_result(d, len(Psi))
    Xtr  = Psi[:ntr].T
    Ytr  = Psi[1:ntr+1].T
    Xt   = Psi[ntr:-1].T
    Yt   = Psi[ntr+1:].T
    Ytr_si = Ytr[si]
    Yt_si  = Yt[si]

    rng    = np.random.RandomState(traj_i)   # different random order per trajectory
    ords_i = {
        'PR':     pr_o,
        'PPR':    ppr_o,
        'TICA':   tica_o,
        'PCCA+':  pcca_o,
        'Random': rng.permutation(tN),
    }

    if not PLOT_ONLY:
        print(f"Trajectory {traj_i+1}/{n_traj}: computing errors...")
        statistics = one_step_statistics(Psi, ntr, si)
        for nm, o in ords_i.items():
            e1 = np.array([one_step_from_statistics(o, N, statistics) for N in Nr])
            E1_all[nm].append(e1)
            em = errM(o, Nms, Psi, si, steps, ntr)
            EM_all[nm].append(em)
        np.savez_compressed(EVALUATION / "trajectory_errors.npz", N=Nr, N_multistep=Nms,
                            horizons=np.arange(1, steps+1),
                            **{"E1_"+nm: np.asarray(v) for nm, v in E1_all.items()},
                            **{"EM_"+nm: np.asarray(v) for nm, v in EM_all.items()})
        print(f"Trajectory {traj_i+1}/{n_traj}: saved one-step and multi-step errors", flush=True)

    # Save representative data from trajectory 0 for heatmap / score figures
    if traj_i == 0:
        rep = dict(
            K=K, Psi=Psi,
            pi_pr=d['pi_pr'], pi_ppr=d['pi_ppr'],
            pr_o=pr_o, ppr_o=ppr_o, tica_o=tica_o, pcca_o=pcca_o,
            names=d['obs_names'], traj=d['traj'],
            ns=ns, tN=tN, si=si,
            fm=d['frame_macro'],
            ords=ords_i,
        )

# Convert to arrays: shape (n_traj, len(Nr)) and (n_traj, steps)
E1_arr = {nm: np.array(E1_all[nm]) for nm in E1_all}   # (20, 77)
EM_arr = {nm: np.array(EM_all[nm]) for nm in EM_all}   # (20, 30)

E1_mean = {nm: E1_arr[nm].mean(axis=0) for nm in E1_arr}
E1_std  = {nm: E1_arr[nm].std(axis=0)  for nm in E1_arr}
EM_mean = {nm: EM_arr[nm].mean(axis=0) for nm in EM_arr}
EM_std  = {nm: EM_arr[nm].std(axis=0)  for nm in EM_arr}

# Pair methods with Random within each trajectory; retain ratio-of-means separately.
ratio_mean, ratio_std, ratio_of_means = {}, {}, {}
for nm in ['PR','PPR','TICA','PCCA+']:
    ratio_mean[nm], ratio_std[nm], ratio_of_means[nm] = paired_ratio_summary(
        E1_arr[nm], E1_arr['Random'])
summary = dict(dictionary_size=196, evaluation_protocol="train_only_v1", n_trajectories=n_traj,
               N=Nr.tolist(), N_multistep=Nms, horizons=list(range(1, steps+1)), std_ddof=0,
               metric="Mean summed squared residual of four training-standardized circular observables",
               one_step_test_pairs=20000, multistep_common_starts=len(Psi)-ntr-steps,
               ratio_definition="mean/std of per-trajectory method MSE divided by same-trajectory Random MSE",
               methods={nm: dict(one_step_mean=E1_mean[nm].tolist(), one_step_std=E1_std[nm].tolist(),
                                 multistep_mean=EM_mean[nm].tolist(), multistep_std=EM_std[nm].tolist(),
                                 **({} if nm == 'Random' else dict(
                                     paired_ratio_mean=ratio_mean[nm].tolist(),
                                     paired_ratio_std=ratio_std[nm].tolist(),
                                     ratio_of_means=ratio_of_means[nm].tolist()))) for nm in pn})
if not PLOT_ONLY:
    (EVALUATION / "evaluation_results.json").write_text(json.dumps(summary, indent=2) + "\n")

print("\nAggregation complete.")

# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Prediction errors with error bands (1x2)
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

ax = axes[0]
for nm in pn:
    ax.plot(Nr, E1_mean[nm], label=nm, color=C[nm], ls=LS[nm], lw=LW[nm])
    ax.fill_between(Nr,
                    E1_mean[nm] - E1_std[nm],
                    E1_mean[nm] + E1_std[nm],
                    color=C[nm], alpha=0.15)
ax.set_xlabel('Sub-dictionary size $N$')
ax.set_ylabel('One-step state prediction MSE')
ax.set_title('(a) One-step error (mean ± 1 std, 20 trajectories)')
ax.set_xlim(4, PLOT_N_MAX)
ax.set_xticks([n for n in [5,15,30,45,60,75] if n < PLOT_N_MAX] + [PLOT_N_MAX])
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[1]
for nm in pn:
    ax.plot(range(1, steps+1), EM_mean[nm], label=nm, color=C[nm], ls=LS[nm], lw=LW[nm])
    ax.fill_between(range(1, steps+1),
                    EM_mean[nm] - EM_std[nm],
                    EM_mean[nm] + EM_std[nm],
                    color=C[nm], alpha=0.15)
ax.set_xlabel('Prediction horizon (steps)')
ax.set_ylabel('State prediction MSE')
ax.set_title(f'(b) Multi-step error ($N={Nms}$, mean ± 1 std)')
ax.legend(); ax.grid(True, alpha=0.2)

fig.tight_layout()
save_figure(fig, "ala2_errors")
print("Saved ala2_errors.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2: EDMD heatmaps (representative trajectory 0)
# ═══════════════════════════════════════════════════════════════
K     = rep['K']
ords0 = rep['ords']

fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))
for ax, title, o in zip(axes,
    ['(a) Random', '(b) TICA', '(c) PCCA+', '(d) PageRank', '(e) PPR'],
    [ords0['Random'], rep['tica_o'], rep['pcca_o'], rep['pr_o'], rep['ppr_o']]):
    K_perm  = np.abs(K[np.ix_(o, o)])
    log_K   = np.log10(K_perm + 1e-8)
    vmin_log = np.percentile(log_K, 5)
    vmax_log = np.percentile(log_K, 99)
    im = ax.imshow(log_K, cmap='coolwarm', vmin=vmin_log, vmax=vmax_log, aspect='auto')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)
fig.colorbar(im, ax=axes[-1], shrink=0.75, pad=0.02, label=r'$\log_{10}|K_{ij}|$')
fig.tight_layout()
save_figure(fig, "ala2_heatmaps")
print("Saved ala2_heatmaps.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Score distributions + improvement ratio with error bands
# ═══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

ax = axes[0]
ax.bar(range(rep['tN']), rep['pi_pr'][rep['pr_o']], color=C['PR'], alpha=0.7, width=1)
ax.set_xlabel('Observable (PR-sorted)'); ax.set_ylabel('PR score')
ax.set_title('(a) PageRank scores (traj 0)')

ax = axes[1]
ax.bar(range(rep['tN']), rep['pi_ppr'][rep['ppr_o']], color=C['PPR'], alpha=0.7, width=1)
ax.set_xlabel('Observable (PPR-sorted)'); ax.set_ylabel('PPR score')
ax.set_title('(b) PPR scores (traj 0)')

ax = axes[2]
for nm in ['PR','PPR','TICA','PCCA+']:
    ax.plot(Nr, ratio_mean[nm], label=nm, color=C[nm], ls=LS[nm], lw=LW[nm])
    ax.fill_between(Nr,
                    ratio_mean[nm] - ratio_std[nm],
                    ratio_mean[nm] + ratio_std[nm],
                    color=C[nm], alpha=0.15)
ax.axhline(1.0, color='gray', ls=':', lw=0.8)
ax.set_xlabel('Sub-dictionary size $N$')
ax.set_ylabel('Paired error / Random error')
ax.set_title('(c) Improvement over Random (mean ± 1 std)')
ax.legend(); ax.grid(True, alpha=0.2)
ax.set_ylim(bottom=0)
ax.set_xlim(4, PLOT_N_MAX)
ax.set_xticks([n for n in [5,15,30,45,60,75] if n < PLOT_N_MAX] + [PLOT_N_MAX])

fig.tight_layout()
save_figure(fig, "ala2_scores")
print("Saved ala2_scores.pdf")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Landscape + PCCA+ + observable types  (traj 0)
# ═══════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(15, 4.2))
traj0  = rep['traj']
names0 = rep['names']
ppr_o0 = rep['ppr_o']
pr_o0  = rep['pr_o']
fm0    = rep['fm']

def potential(phi, psi):
    cx = np.array([-1.0,-1.0,1.1]); cy = np.array([-1.0,1.2,-0.3])
    A  = np.array([-6.0,-6.0,-4.0]); s  = np.array([0.55,0.55,0.65])
    V  = np.zeros_like(phi)
    for k in range(3):
        dp = np.arctan2(np.sin(phi-cx[k]), np.cos(phi-cx[k]))
        dq = np.arctan2(np.sin(psi-cy[k]), np.cos(psi-cy[k]))
        V += A[k]*np.exp(-0.5*(dp**2+dq**2)/s[k]**2)
    V += 0.3*(np.cos(2*phi)+np.cos(2*psi))
    return V

ax = fig.add_subplot(1,4,1)
pgrid = np.linspace(-np.pi, np.pi, 100)
PP, QQ = np.meshgrid(pgrid, pgrid)
VV = potential(PP, QQ)
ax.contourf(PP, QQ, VV, levels=30, cmap='viridis_r', alpha=0.8)
ax.contour(PP, QQ, VV, levels=10, colors='k', linewidths=0.3, alpha=0.5)
ax.scatter(traj0[::50,0], traj0[::50,1], s=0.1, c='white', alpha=0.15, rasterized=True)
ax.set_xlabel(r'$\phi$'); ax.set_ylabel(r'$\psi$')
ax.set_title('(a) Potential landscape')
ax.set_xlim(-np.pi,np.pi); ax.set_ylim(-np.pi,np.pi); ax.set_aspect('equal')

rbf_centers = []
for idx in ppr_o0[:19]:
    nm = str(names0[idx])
    if nm.startswith('G('):
        inner = nm[2:-1].split(',')
        rbf_centers.append((float(inner[0]), float(inner[1])))
rbf_centers = np.array(rbf_centers) if rbf_centers else np.empty((0,2))
if len(rbf_centers) > 0:
    ax.scatter(rbf_centers[:,0], rbf_centers[:,1],
               s=10, marker='o', c='white', edgecolors='k',
               linewidths=0.8, zorder=5, label='PPR top RBFs')

ax = fig.add_subplot(1,4,2)
valid0 = fm0 >= 0
mc     = ['#E63946','#457B9D','#2A9D8F']
for c in range(3):
    m = valid0 & (fm0==c)
    ax.scatter(traj0[m,0][::5], traj0[m,1][::5], s=0.2, alpha=0.25,
               c=mc[c], label=f'Macro {c}', rasterized=True)
ax.set_xlabel(r'$\phi$'); ax.set_ylabel(r'$\psi$')
ax.set_title('(b) PCCA+ macrostates')
ax.set_xlim(-np.pi,np.pi); ax.set_ylim(-np.pi,np.pi); ax.set_aspect('equal')
ax.legend(markerscale=15, fontsize=7, loc='upper right')

def classify(nm):
    nm = str(nm)
    if nm.startswith('G('): return 'RBF'
    if '·' in nm:           return 'Cross'
    if '+' in nm and not nm.startswith('sin(') and not nm.startswith('cos('): return 'Diag'
    return 'Fourier'

bc    = {'Fourier':'#E63946','Cross':'#F4A261','Diag':'#264653','RBF':'#457B9D'}
n_top = 20

ax = fig.add_subplot(1,4,3)
types_pr = [classify(names0[i]) for i in pr_o0[:n_top]]
tc = {t: types_pr.count(t) for t in ['Fourier','Cross','Diag','RBF'] if types_pr.count(t)>0}
ax.bar(tc.keys(), tc.values(), color=[bc[k] for k in tc])
ax.set_ylabel('Count'); ax.set_title(f'(c) PR top-{n_top} by type')

ax = fig.add_subplot(1,4,4)
types_ppr = [classify(names0[i]) for i in ppr_o0[:n_top]]
tc2 = {t: types_ppr.count(t) for t in ['Fourier','Cross','Diag','RBF'] if types_ppr.count(t)>0}
ax.bar(tc2.keys(), tc2.values(), color=[bc[k] for k in tc2])
ax.set_ylabel('Count'); ax.set_title(f'(d) PPR top-{n_top} by type')

fig.tight_layout()
save_figure(fig, "ala2_landscape")
print("Saved ala2_landscape.pdf")

# Combined appendix figure matching the manuscript's five-panel structure.
fig = plt.figure(figsize=(13, 7))
grid = GridSpec(2, 3, figure=fig)
ax = fig.add_subplot(grid[0, 0])
ax.contourf(PP, QQ, VV, levels=30, cmap='viridis_r')
if len(rbf_centers):
    ax.scatter(rbf_centers[:, 0], rbf_centers[:, 1], s=18, c='white',
               edgecolors='black', linewidths=.6)
ax.set(xlabel=r'$\phi$', ylabel=r'$\psi$', title=f'(a) Top-{len(rbf_centers)} selected RBF centers',
       xlim=(-np.pi, np.pi), ylim=(-np.pi, np.pi))
ax.set_aspect('equal')
ax = fig.add_subplot(grid[0, 1])
types = ['Fourier', 'Cross', 'RBF']
x = np.arange(len(types))
ax.bar(x-.18, [types_pr.count(t) for t in types], .36, color=C['PR'], label='PR')
ax.bar(x+.18, [types_ppr.count(t) for t in types], .36, color=C['PPR'], label='PPR')
ax.set_xticks(x, types)
ax.set(ylabel='Count', title='(b) Top-20 observable types')
ax.legend()
for pos, method, scores, order, label in [
        (grid[0, 2], 'PR', rep['pi_pr'], rep['pr_o'], '(c) PR scores'),
        (grid[1, 0], 'PPR', rep['pi_ppr'], rep['ppr_o'], '(d) PPR scores')]:
    ax = fig.add_subplot(pos)
    ax.bar(np.arange(rep['tN']), scores[order], width=1, color=C[method])
    ax.set(xlabel='Observable (rank order)', ylabel='Score', title=label)
ax = fig.add_subplot(grid[1, 1:])
for nm in ['PR','PPR','TICA','PCCA+']:
    ax.plot(Nr, ratio_mean[nm], label=nm, color=C[nm], ls=LS[nm], lw=LW[nm])
    ax.fill_between(Nr, ratio_mean[nm]-ratio_std[nm], ratio_mean[nm]+ratio_std[nm],
                    color=C[nm], alpha=.15)
ax.axhline(1, color='gray', ls=':')
ax.set(xlabel='Sub-dictionary size $N$', ylabel='Paired error / Random error',
       title='(e) Mean ± SD of paired trajectory ratios', ylim=(0, None))
ax.set_xlim(4, PLOT_N_MAX)
ax.set_xticks([n for n in [5,15,30,45,60,75] if n < PLOT_N_MAX] + [PLOT_N_MAX])
ax.legend(ncol=4, fontsize=8)
ax.grid(alpha=.2)
fig.tight_layout()
save_figure(fig, 'ala2_macro-scores')

# ═══════════════════════════════════════════════════════════════
# SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY  (mean ± std over 20 trajectories)")
print("="*70)

print(f"\n{'Method':7s} | {'N=5':>14s} {'N=10':>14s} {'N=15':>14s} {'N=20':>14s} {'N=40':>14s}")
print("-"*80)
for nm in pn:
    vals = []
    for n in [5,10,15,20,40]:
        if n in Nr:
            i = np.where(Nr==n)[0][0]
            vals.append(f"{E1_mean[nm][i]:6.3f}±{E1_std[nm][i]:5.3f}")
        else:
            vals.append(f"{'N/A':>14s}")
    print(f"  {nm:5s} | {' '.join(f'{v:>14s}' for v in vals)}")

print(f"\nImprovement ratio (mean ± std over 20 trajectories):")
print(f"{'Method':7s} | {'N=5':>14s} {'N=10':>14s} {'N=15':>14s} {'N=20':>14s}")
print("-"*65)
for nm in ['PR','PPR','TICA','PCCA+']:
    vals = []
    for n in [5,10,15,20]:
        if n in Nr:
            i = np.where(Nr==n)[0][0]
            vals.append(f"{ratio_mean[nm][i]:5.3f}±{ratio_std[nm][i]:5.3f}")
        else:
            vals.append(f"{'N/A':>14s}")
    print(f"  {nm:5s} | {' '.join(f'{v:>14s}' for v in vals)}")

print(f"\nFigures saved to {FIG}/")

plot_manifest = dict(plot_only=PLOT_ONLY, evaluation_N_range=[int(Nr[0]), int(Nr[-1])],
                     displayed_N_range=[4, PLOT_N_MAX], figure_axes_limits=figure_axes_limits,
                     source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     source_npz_sha256=hashlib.sha256((EVALUATION / "trajectory_errors.npz").read_bytes()).hexdigest(),
                     source_summary_sha256=hashlib.sha256((EVALUATION / "evaluation_results.json").read_bytes()).hexdigest(),
                     latex_compiled=False)
(Path(FIG) / "plot_manifest.json").write_text(json.dumps(plot_manifest, indent=2) + "\n")
