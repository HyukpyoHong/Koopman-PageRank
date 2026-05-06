"""
Step 3: Comprehensive evaluation and publication-quality figures.
Aggregates results over 20 independent trajectories to produce error bars.
"""
import os, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = "results/"
FIG = "figures/"
os.makedirs(FIG, exist_ok=True)


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
    idx = order[:N]
    _, Kread = _solve(idx, Xtr, Ytr, Ytr_si, ntr)
    return np.mean(np.sum((Yt_si - Kread @ Xt[idx])**2, axis=0))

def errM(order, N, Psi, si, steps, ntr):
    idx = order[:N]
    Xtr   = Psi[:ntr].T
    Ytr   = Psi[1:ntr+1].T
    Ytr_si = Ytr[si]
    Ksub, Kread = _solve(idx, Xtr, Ytr, Ytr_si, ntr)
    st  = ntr
    nt  = Psi.shape[0] - st - steps
    errs = np.zeros(steps)
    for t in range(nt):
        z = Psi[st+t, idx].copy()
        for s in range(steps):
            z  = Ksub @ z
            sp = Kread @ z
            errs[s] += np.sum((Psi[st+t+s+1, si] - sp)**2)
    return errs / nt

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

for traj_i in range(n_traj):
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
    si       = np.arange(ns)

    M    = Psi.shape[0] - 1
    ntr  = int(0.8 * M)
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

    print(f"Trajectory {traj_i+1}/{n_traj}: computing errors...")
    for nm, o in ords_i.items():
        e1 = np.array([err1(o, N, Xtr, Ytr, Ytr_si, Xt, Yt_si, ntr) for N in Nr])
        E1_all[nm].append(e1)
        em = errM(o, Nms, Psi, si, steps, ntr)
        EM_all[nm].append(em)

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

# Convenience: ratio mean/std
ratio_mean = {nm: E1_mean[nm] / E1_mean['Random'] for nm in ['PR','PPR','TICA','PCCA+']}
ratio_std  = {nm: ratio_mean[nm] * np.sqrt(
                  (E1_std[nm]         / E1_mean[nm]        )**2 +
                  (E1_std['Random']   / E1_mean['Random']  )**2)
              for nm in ['PR','PPR','TICA','PCCA+']}

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
ax.set_ylabel('One-step state prediction error')
ax.set_title('(a) One-step error (mean ± 1 std, 20 trajectories)')
ax.legend(); ax.grid(True, alpha=0.2)

ax = axes[1]
for nm in pn:
    ax.plot(range(1, steps+1), EM_mean[nm], label=nm, color=C[nm], ls=LS[nm], lw=LW[nm])
    ax.fill_between(range(1, steps+1),
                    EM_mean[nm] - EM_std[nm],
                    EM_mean[nm] + EM_std[nm],
                    color=C[nm], alpha=0.15)
ax.set_xlabel('Prediction horizon (steps)')
ax.set_ylabel('State prediction error')
ax.set_title(f'(b) Multi-step error ($N={Nms}$, mean ± 1 std)')
ax.legend(); ax.grid(True, alpha=0.2)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "ala2_errors.pdf"), bbox_inches='tight')
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
fig.savefig(os.path.join(FIG, "ala2_heatmaps.pdf"), bbox_inches='tight')
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
ax.set_ylabel('Error / Random error')
ax.set_title('(c) Improvement over Random (mean ± 1 std)')
ax.legend(); ax.grid(True, alpha=0.2)
ax.set_ylim(0, 1.5)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "ala2_scores.pdf"), bbox_inches='tight')
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
fig.savefig(os.path.join(FIG, "ala2_landscape.pdf"), bbox_inches='tight')
print("Saved ala2_landscape.pdf")

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
