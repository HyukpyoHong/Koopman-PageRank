"""
koopman_source.py
=================
Core library for Koopman operator / EDMD experiments.

Sections
--------
1. Dynamical systems        -- Euler-step and RK4 integrators
2. Observable dictionaries  -- Laguerre, monomial, radial bases
3. PageRank / PPR           -- on the EDMD weight matrix
4. EDMD                     -- find_best_lin
5. T-SSD                    -- Tunable Symmetric Subspace Decomposition
6. Graph utilities          -- SBM, random walk (retained for completeness)
"""

from math import comb
import numpy as np
from numpy import linalg as LA
import networkx as nx
from scipy.linalg import subspace_angles
from scipy.integrate import solve_ivp

try:
    from .pagerank_utils import prune_zero_rows, retained_preference, expand_scores
except ImportError:  # Direct imports from the notebooks' src path.
    from pagerank_utils import prune_zero_rows, retained_preference, expand_scores


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Dynamical systems
# ═════════════════════════════════════════════════════════════════════════════

# ── 2-D toy model ────────────────────────────────────────────────────────────

def toy_d1(t, x0, mu0, lambda0):
    """Exact linear map (no dt): toy model 1."""
    x_mat = np.zeros((len(x0), t))
    x = x0.copy()
    for i in range(t):
        x[0] = (1 + mu0) * x[0]
        x[1] = (1 + lambda0) * x[1]
        x_mat[:, i] = x
    return x_mat


def toy_d2(x_old, dt, mu0, lambda0):
    """One Euler step of the 2-D toy model.

    x1(k+1) = (1 + dt*mu0) * x1(k)
    x2(k+1) = (1 + dt*lambda0) * x2(k) - dt*lambda0 * x1(k)^2
    """
    x = x_old.copy()
    x[0] = (1 + dt * mu0) * x_old[0]
    x[1] = ((1 + dt * lambda0) * x_old[1]
            - dt * lambda0 * x_old[0] ** 2)
    return x


# ── Duffing oscillator ────────────────────────────────────────────────────────

def Duffing(x_old, alpha0, beta0, delta0, dt):
    """One Euler step of the Duffing oscillator.

    dx/dt = y
    dy/dt = -delta*y - alpha*x - beta*x^3
    """
    x = x_old.copy()
    x[0] = x_old[0] + x_old[1] * dt
    x[1] = x_old[1] + (-delta0 * x_old[1]
                        - alpha0 * x_old[0]
                        - beta0 * x_old[0] ** 3) * dt
    return x


# ── Van der Pol oscillator ────────────────────────────────────────────────────

def VanDerPol(x_old, mu0, dt):
    """One Euler step of the Van der Pol oscillator.

    dx/dt = y
    dy/dt = mu*(1 - x^2)*y - x
    """
    x = x_old.copy()
    x[0] = x_old[0] + x_old[1] * dt
    x[1] = x_old[1] + (mu0 * (1 - x_old[0] ** 2) * x_old[1] - x_old[0]) * dt
    return x


# ── Lotka-Volterra ────────────────────────────────────────────────────────────

def Lotka_Voltera(x_old, alpha0, beta0, gamma0, delta0, dt):
    """One Euler step of the Lotka-Volterra predator-prey system."""
    x = x_old.copy()
    x[0] = x_old[0] + (alpha0 * x_old[0] - beta0 * x_old[0] * x_old[1]) * dt
    x[1] = x_old[1] + (-gamma0 * x_old[1] + delta0 * x_old[0] * x_old[1]) * dt
    return x


# ── Lorenz system ─────────────────────────────────────────────────────────────

def Lorenz(x_old, sigma0, rho0, beta0, dt):
    """One Euler step of the Lorenz system (for legacy use; prefer lorenz_rk4)."""
    x = x_old.copy()
    x[0] = x_old[0] + sigma0 * (x_old[1] - x_old[0]) * dt
    x[1] = x_old[1] + (x_old[0] * (rho0 - x_old[2]) - x_old[2]) * dt
    x[2] = x_old[2] + (x_old[0] * x_old[2] - beta0 * x_old[2]) * dt
    return x


def lorenz_rhs(s, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """Right-hand side of the Lorenz system."""
    return np.array([
        sigma * (s[1] - s[0]),
        s[0] * (rho - s[2]) - s[1],
        s[0] * s[1] - beta * s[2],
    ])


def lorenz_rk4(x, dt, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    """One RK4 step of the Lorenz system."""
    k1 = lorenz_rhs(x, sigma, rho, beta)
    k2 = lorenz_rhs(x + 0.5 * dt * k1, sigma, rho, beta)
    k3 = lorenz_rhs(x + 0.5 * dt * k2, sigma, rho, beta)
    k4 = lorenz_rhs(x + dt * k3, sigma, rho, beta)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ── SIAM DS example ───────────────────────────────────────────────────────────

def SIAMDS1(x_old, mu0, dt):
    """One Euler step of SIAM DS Example 1."""
    x = x_old.copy()
    x[0] = x_old[0] + x_old[1] * dt
    x[1] = (x_old[1]
            + (mu0 * x_old[1] - x_old[0]
               + mu0 * x_old[1] * x_old[0] ** 2) * dt)
    return x


# ── Matrix evolution helper ───────────────────────────────────────────────────

def matrix_evolution(x_mat, func):
    """Apply func to each column of x_mat.

    Parameters
    ----------
    x_mat : (d, M) array
    func  : callable  (d,) -> (d,)

    Returns
    -------
    y_mat : (d, M)
    """
    y = x_mat.copy()
    for i in range(x_mat.shape[1]):
        y[:, i] = func(x_mat[:, i])
    return y


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Observable / dictionary functions
# ═════════════════════════════════════════════════════════════════════════════

def psi_id(x):
    return x

def psi_d1(x):
    return x


def psi_d2(x, max_order, subset=-1):
    """Monomial basis for a 2-D system, total degree <= max_order.

    Returns
    -------
    obs[subset], ind_mat[subset]
    """
    d = 2
    num_obs = comb(max_order + d, d)
    if isinstance(subset, int):
        subset = np.arange(num_obs)
    obs = np.zeros(num_obs)
    ind_mat = []
    idx = 0
    for i in range(max_order + 1):
        for j in range(i + 1):
            obs[idx] = x[0] ** (i - j) * x[1] ** j
            ind_mat.append([i - j, j])
            idx += 1
    ind_mat = np.array(ind_mat)
    return obs[subset], ind_mat[subset, :]


def psi_d3(x, max_order, subset=-1):
    """Monomial basis for a 3-D system, total degree <= max_order."""
    d = 3
    num_obs = comb(max_order + d, d)
    if isinstance(subset, int):
        subset = np.arange(num_obs)
    obs = np.zeros(num_obs)
    ind_mat = []
    idx = 0
    for i in range(max_order + 1):
        for j in range(i + 1):
            for k in range(j + 1):
                obs[idx] = x[0] ** (i - j) * x[1] ** (j - k) * x[2] ** k
                ind_mat.append([i - j, j - k, k])
                idx += 1
    ind_mat = np.array(ind_mat)
    return obs[subset], ind_mat[subset, :]


# ── Laguerre basis ────────────────────────────────────────────────────────────

def Laguerre_poly(x, n):
    """Evaluate the n-th Laguerre polynomial L_n(x) via three-term recurrence."""
    if n == 0:
        return 1
    if n == 1:
        return -x + 1
    return ((2 * n - 1 - x) * Laguerre_poly(x, n - 1)
            - (n - 1) * Laguerre_poly(x, n - 2)) / n


def psi_Laguerre(x, d, max_order, subset=-1):
    """Tensor-product Laguerre dictionary, graded lex order, d in {2, 3}.

    Parameters
    ----------
    x         : (d,) state vector
    d         : state dimension
    max_order : maximum total polynomial degree
    subset    : int (-1 = all) or array of indices

    Returns
    -------
    obs[subset], ind_mat[subset]
    """
    num_obs = comb(max_order + d, d)
    if isinstance(subset, int):
        subset = np.arange(num_obs)
    obs = []
    ind_mat = []
    if d == 3:
        for i in range(max_order + 1):
            for j in range(i + 1):
                for k in range(j + 1):
                    obs.append(
                        Laguerre_poly(x[0], i - j)
                        * Laguerre_poly(x[1], j - k)
                        * Laguerre_poly(x[2], k)
                    )
                    ind_mat.append([i - j, j - k, k])
    elif d == 2:
        for i in range(max_order + 1):
            for j in range(i + 1):
                obs.append(
                    Laguerre_poly(x[0], i - j) * Laguerre_poly(x[1], j)
                )
                ind_mat.append([i - j, j])
    else:
        raise ValueError(f"psi_Laguerre: d must be 2 or 3, got {d}")
    obs = np.array(obs)
    ind_mat = np.array(ind_mat)
    return obs[subset], ind_mat[subset, :]


def psi_d2_Laguerre(x):
    """Fixed-order (max_order=2) Laguerre dictionary for 2-D systems."""
    d, max_order = 2, 2
    num_obs = comb(max_order + d, d)
    obs = np.zeros(num_obs)
    idx = 0
    for i in range(max_order + 1):
        for j in range(i + 1):
            obs[idx] = Laguerre_poly(x[0], i - j) * Laguerre_poly(x[1], j)
            idx += 1
    return obs


def psi_radial(x, cj, gamma):
    """Radial basis function: exp(-gamma * ||x - cj||_2)."""
    return np.exp(-gamma * LA.norm(x - cj, 2))


# ── Dictionary evaluation on data matrices ────────────────────────────────────

def psi_evaluation_laguerre(x_mat, d, max_order):
    """Evaluate Laguerre dictionary on a data matrix.

    Parameters
    ----------
    x_mat     : (d, M) array of M data points
    d         : state dimension
    max_order : polynomial order

    Returns
    -------
    Psi : (M, N) observable matrix   where N = C(d + max_order, d)
    """
    M = x_mat.shape[1]
    N = int(comb(d + max_order, d))
    Psi = np.zeros((M, N))
    for i in range(M):
        Psi[i, :] = psi_Laguerre(x_mat[:, i], d, max_order)[0]
    return Psi


# ═════════════════════════════════════════════════════════════════════════════
# 3.  PageRank and Personalized PageRank
# ═════════════════════════════════════════════════════════════════════════════

def pagerank(M, num_iterations=100_000, d=0.8):
    """Standard PageRank on the column-stochastic matrix derived from M.

    Parameters
    ----------
    M               : (N, N) non-negative weight matrix
    num_iterations  : maximum power iterations
    d               : damping factor (teleportation = 1-d)

    Returns
    -------
    v : (N,) PageRank scores in original indexing (removed nodes score zero).
        Zero-outgoing nodes and their incident edges are pruned iteratively.
        Use prune_zero_rows(M.T)[1] to obtain the retained original indices.
    """
    original_size = M.shape[1]
    P, retained = prune_zero_rows(np.asarray(M).T)
    M_hat = P.T
    N = len(retained)

    v = np.ones(N) / N
    p = np.ones(N) / N
    for _ in range(num_iterations):
        v_next = (1 - d) * p + d * M_hat.dot(v)
        if np.allclose(v, v_next, atol=1e-10):
            return expand_scores(v_next, retained, original_size)
        v = v_next
    return expand_scores(v, retained, original_size)


def personalized_pagerank(M, num_iterations=100_000, d=0.8, seed_nodes=None):
    """Personalized PageRank (PPR) with teleportation toward seed_nodes.

    Parameters
    ----------
    M               : (N, N) non-negative weight matrix
    num_iterations  : maximum power iterations
    d               : damping factor
    seed_nodes      : list[int].  Teleportation is uniform over these nodes.
                      None or [] falls back to uniform (standard PageRank).

    Returns
    -------
    v : (N,) PPR scores in original indexing; removed nodes score zero.
        Seeds removed by iterative zero-outgoing pruning raise ValueError.
    """
    original_size = M.shape[1]
    P, retained = prune_zero_rows(np.asarray(M).T)
    N = len(retained)
    p = retained_preference(retained, original_size, seed_nodes)
    M_hat = P.T

    v = np.ones(N) / N
    for _ in range(num_iterations):
        v_next = (1 - d) * p + d * M_hat.dot(v)
        if np.allclose(v, v_next, atol=1e-10):
            return expand_scores(v_next, retained, original_size)
        v = v_next
    return expand_scores(v, retained, original_size)


def subsampled_pagerank(M, num_iterations=100, d=0.85,
                        subsample_rate=0.1, epsilon=1e-8):
    """Stochastic PageRank with column sub-sampling (large-N approximation)."""
    original_size = M.shape[0]
    P, retained = prune_zero_rows(np.asarray(M).T)
    N = len(retained)
    M_hat = P.T
    v = np.ones(N) / N
    for _ in range(num_iterations):
        v_prev = v.copy()
        mask = np.random.random(N) < subsample_rate
        v = ((1 - d) / N
             + d * M_hat[:, mask] @ (v_prev[mask] / subsample_rate))
        if np.abs(v - v_prev).sum() < epsilon:
            break
    return expand_scores(v / v.sum(), retained, original_size)


# ═════════════════════════════════════════════════════════════════════════════
# 4.  EDMD
# ═════════════════════════════════════════════════════════════════════════════

def find_best_lin(Psi_X, Psi_Y):
    """Least-squares EDMD: K = pinv(Psi_X) @ Psi_Y.

    Parameters
    ----------
    Psi_X : (M, N) observable matrix at current time
    Psi_Y : (M, N) observable matrix at next time

    Returns
    -------
    K : (N, N) EDMD matrix  (row = target, col = source)
    """
    return np.linalg.pinv(Psi_X) @ Psi_Y


# ── Ordering helpers ──────────────────────────────────────────────────────────

def data_dist(d, M, low, high, rndseed=1):
    """Sample M points i.i.d. uniformly from [low, high]^d."""
    return np.random.default_rng(rndseed).uniform(low, high, size=(d, M))


def compute_orderings(K_full, N, d, default_basis, Psi_X, Psi_Y):
    """Return Unordered / Incremental / PageRank / PPR orderings.

    Parameters
    ----------
    K_full        : (N, N) full EDMD matrix
    N             : total dictionary size
    d             : state dimension
    default_basis : indices of coordinate observables (always ranked first)
    Psi_X, Psi_Y  : (M, N) data matrices (unused here but kept for API compat.)

    Returns
    -------
    dict with keys 'Unordered', 'Incremental', 'PageRank', 'PPR'
    """
    _, retained = prune_zero_rows(np.asarray(np.abs(K_full)).T)
    retained_preference(retained, N, default_basis)  # Reject removed targets.
    other_basis = np.setdiff1d(retained, default_basis)

    # Unordered: random permutation of non-default observables
    random_idx = np.append(
        default_basis, np.random.permutation(other_basis)
    ).astype(int)

    # Incremental: lexicographic / degree order
    inc_idx = np.append(default_basis, other_basis).astype(int)

    # PageRank
    eig_v = pagerank(np.abs(K_full), 100_000, 0.85)
    ranked_idx = np.append(
        default_basis,
        other_basis[np.argsort(eig_v[other_basis])[::-1]]
    ).astype(int)

    # PPR (seeds = all coordinate functions)
    ppr_v = personalized_pagerank(
        np.abs(K_full), 100_000, 0.85, seed_nodes=list(default_basis)
    )
    ppr_idx = np.append(
        default_basis,
        other_basis[np.argsort(ppr_v[other_basis])[::-1]]
    ).astype(int)

    return {
        'Unordered':   random_idx,
        'Incremental': inc_idx,
        'PageRank':    ranked_idx,
        'PPR':         ppr_idx,
    }


def compute_errors(methods, Psi_X, Psi_Y, Psi_X_test, Psi_Y_test,
                   d, N, default_basis):
    """One-step L2 prediction error for each method at every sub-dict size.

    Parameters
    ----------
    methods       : dict[str -> ordering array]
    Psi_X, Psi_Y  : (M, N) training data matrices
    Psi_X_test, Psi_Y_test : test data matrices
    d             : state dimension
    N             : total dictionary size
    default_basis : indices of coordinate observables

    Returns
    -------
    errors : dict[str -> (N,) array]  — NaN where prediction fails
    """
    errors = {}
    for name, ordering in methods.items():
        err = np.full(N, np.nan)
        for nr in range(d, min(N, len(ordering)) + 1):
            sel = ordering[:nr].astype(int)
            K_sub = find_best_lin(Psi_X[:, sel], Psi_Y[:, sel])
            pred = Psi_X_test[:, sel] @ K_sub
            total_sq = 0.0
            valid = True
            for dd in range(d):
                obs_id = default_basis[dd]
                local = np.where(ordering[:nr] == obs_id)[0]
                if len(local) > 0:
                    total_sq += np.linalg.norm(
                        pred[:, local[0]] - Psi_Y_test[:, obs_id]
                    ) ** 2
                else:
                    valid = False
            if valid:
                err[nr - 1] = np.sqrt(total_sq)
        errors[name] = err
    return errors


def compute_errors_multiseed(d, max_order, M, low, high,
                              evolve_func, default_basis,
                              num_rnd=20, test_rndseed_offset=1000):
    """Run the full EDMD pipeline over multiple random seeds.

    Each seed controls:
      - training data sampling   (rndseed = seed)
      - test data sampling       (rndseed = seed + test_rndseed_offset)
      - the Unordered permutation

    Parameters
    ----------
    d, max_order   : dictionary parameters
    M              : number of training samples
    low, high      : data domain bounds (uniform on [low, high]^d)
    evolve_func    : callable  x -> x_next
    default_basis  : coordinate observable indices
    num_rnd        : number of random seeds  (change to 2-3 for quick checks)
    test_rndseed_offset : offset added to seed for test data

    Returns
    -------
    all_errors : dict[str -> (num_rnd, N) array]
    """
    N = int(comb(d + max_order, d))
    method_names = ['Unordered', 'Incremental', 'PageRank', 'PPR']
    all_errors = {name: [] for name in method_names}

    for seed in range(num_rnd):
        np.random.seed(seed)

        x_tr = data_dist(d, M, low, high, rndseed=seed)
        y_tr = matrix_evolution(x_tr, evolve_func)
        Psi_X = psi_evaluation_laguerre(x_tr, d, max_order)
        Psi_Y = psi_evaluation_laguerre(y_tr, d, max_order)
        K = find_best_lin(Psi_X, Psi_Y)

        x_te = data_dist(d, M, low, high, rndseed=seed + test_rndseed_offset)
        y_te = matrix_evolution(x_te, evolve_func)
        Psi_X_te = psi_evaluation_laguerre(x_te, d, max_order)
        Psi_Y_te = psi_evaluation_laguerre(y_te, d, max_order)

        methods = compute_orderings(K, N, d, default_basis, Psi_X, Psi_Y)
        errs = compute_errors(methods, Psi_X, Psi_Y, Psi_X_te, Psi_Y_te,
                              d, N, default_basis)
        for name in method_names:
            all_errors[name].append(errs[name])

    return {name: np.array(all_errors[name]) for name in method_names}


def compute_state_predictions(methods, Psi_X, Psi_Y, pred_pts, d,
                               max_order, default_basis, Nr_pred,
                               evolve_func, n_steps, psi_eval_func,
                               orth_basis=True):
    """Multi-step state predictions for each method plus ground truth.

    Parameters
    ----------
    methods        : dict[str -> ordering]
    Psi_X, Psi_Y   : (M, N) training data
    pred_pts       : (d, n_ic) initial conditions to predict from
    d              : state dimension
    max_order      : dictionary order
    default_basis  : coordinate observable indices
    Nr_pred        : sub-dictionary size to use for predictions
    evolve_func    : true one-step map
    n_steps        : prediction horizon
    psi_eval_func  : callable  (d, n_pts) -> (n_pts, N) Psi matrix
    orth_basis     : if True, apply  x_d = 1 - Psi_next[obs_d]
                     (Laguerre L_0 = 1, so L_1(x) = 1 - x recovers x)

    Returns
    -------
    pred_data : dict[str -> (n_steps+1, d, n_ic) array]
    """
    pred_data = {}
    for name, ordering in methods.items():
        sel = ordering[:Nr_pred].astype(int)
        K_sub = find_best_lin(Psi_X[:, sel], Psi_Y[:, sel])
        cur = pred_pts.copy()
        all_pts = [cur.copy()]
        for _ in range(n_steps):
            Psi_cur = psi_eval_func(cur)
            Psi_next = Psi_cur[:, sel] @ K_sub
            cur = np.zeros_like(cur)
            for dd in range(d):
                obs_id = default_basis[dd]
                li = np.where(ordering[:Nr_pred] == obs_id)[0][0]
                cur[dd, :] = (1.0 - Psi_next[:, li] if orth_basis
                              else Psi_next[:, li])
            all_pts.append(cur.copy())
        pred_data[name] = np.array(all_pts)

    # True trajectory
    true_cur = pred_pts.copy()
    true_all = [true_cur.copy()]
    for _ in range(n_steps):
        true_cur = matrix_evolution(true_cur, evolve_func)
        true_all.append(true_cur.copy())
    pred_data['True'] = np.array(true_all)
    return pred_data


# ═════════════════════════════════════════════════════════════════════════════
# 5.  T-SSD (Tunable Symmetric Subspace Decomposition)
# ═════════════════════════════════════════════════════════════════════════════

def tssd_algorithm(Psi_X, Psi_Y, epsilon=0.1, max_iter=100):
    """Tunable SSD (Haseli & Cortés 2023).

    Finds the maximal subspace that is epsilon-approximately invariant
    under both forward and backward EDMD.  epsilon=0 recovers exact SSD.

    Parameters
    ----------
    Psi_X, Psi_Y : (M, N) data matrices
    epsilon      : invariance tolerance (larger => less pruning)
    max_iter     : iteration cap

    Returns
    -------
    C : (N, N_tssd) projection matrix onto the retained subspace
    """
    N = Psi_X.shape[1]
    C = np.eye(N)

    for _ in range(max_iter):
        DX = Psi_X @ C
        DY = Psi_Y @ C
        Kf = np.linalg.pinv(DX) @ DY
        Kb = np.linalg.pinv(DY) @ DX

        n_cur = C.shape[1]
        if n_cur <= 1:
            break

        Uf, Sf, _ = np.linalg.svd(Kf, full_matrices=False)
        if Sf[0] < 1e-14:
            break
        rank_f = max(1, int(np.sum(Sf / Sf[0] > 1e-12)))
        Af = Uf[:, :rank_f]

        Ub, Sb, _ = np.linalg.svd(Kb.T, full_matrices=False)
        if Sb[0] < 1e-14:
            break
        rank_b = max(1, int(np.sum(Sb / Sb[0] > 1e-12)))
        Ab = Ub[:, :rank_b]

        Pf = Af @ Af.T
        Pb = Ab @ Ab.T

        eigvals, eigvecs = np.linalg.eigh(Pf + Pb)
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        threshold = 2.0 - epsilon ** 2
        n_new = int(np.sum(eigvals > threshold))
        if n_new == 0:
            n_new = 1
        if n_new >= n_cur:
            break

        C = C @ eigvecs[:, :n_new]

    return C


def compute_tssd_errors(Psi_X, Psi_Y, Psi_X_test, Psi_Y_test,
                         default_basis,
                         epsilons=None):
    """Run T-SSD at multiple epsilon values and return (eps, N_sel, error).

    default_basis observables are forced into the subspace via QR augmentation.

    Parameters
    ----------
    epsilons : 1-D array of epsilon values to sweep

    Returns
    -------
    results : (n_valid, 3) array  — columns: (epsilon, N_selected, error)
    """
    if epsilons is None:
        epsilons = np.linspace(0.0, 1.0, 40)

    results = []
    for eps in epsilons:
        C = tssd_algorithm(Psi_X, Psi_Y, epsilon=eps)

        # Force default_basis into col(C) via QR augmentation
        C_aug = C.copy()
        for obs_id in default_basis:
            e = np.zeros(Psi_X.shape[1])
            e[obs_id] = 1.0
            proj = C_aug @ (np.linalg.pinv(C_aug) @ e)
            if np.linalg.norm(e - proj) > 1e-8:
                C_aug = np.hstack([C_aug, e[:, None]])
                Q, _ = np.linalg.qr(C_aug, mode='reduced')
                C_aug = Q

        DX = Psi_X @ C_aug
        DY = Psi_Y @ C_aug
        K_aug = np.linalg.pinv(DX) @ DY
        pred = (Psi_X_test @ C_aug) @ K_aug

        total_sq = 0.0
        for obs_id in default_basis:
            e = np.zeros(Psi_X.shape[1])
            e[obs_id] = 1.0
            coeff = C_aug.T @ e
            total_sq += np.linalg.norm(pred @ coeff - Psi_Y_test[:, obs_id]) ** 2

        results.append((eps, C_aug.shape[1], np.sqrt(total_sq)))

    return np.array(results)


# ═════════════════════════════════════════════════════════════════════════════
# 6.  Lorenz time-delay embedding helpers
# ═════════════════════════════════════════════════════════════════════════════

def build_delay_embedding(DATA, M, N_delay, h):
    """Build time-delay embedding with interleaved coordinate ordering.

    Column layout: [x(t), y(t), z(t),  x(t-h), y(t-h), z(t-h), ...]

    Parameters
    ----------
    DATA    : (3, T_total) full trajectory
    M       : number of snapshot pairs to return
    N_delay : number of delay levels
    h       : stride between delay levels (fine-step units)

    Returns
    -------
    PsiX, PsiY : each (M, 3*N_delay)
    """
    d = DATA.shape[0]
    PsiX = np.zeros((M, d * N_delay))
    PsiY = np.zeros((M, d * N_delay))

    for j in range(N_delay):
        offset = h * j
        for dim in range(d):
            col = d * j + dim
            PsiX[:, col] = DATA[dim, offset:offset + M]
            PsiY[:, col] = DATA[dim, offset + 1:offset + 1 + M]

    return PsiX, PsiY


def select_top_indices(rank_vector, coord_indices, N_delay, N_select,
                       group_by_delay=True, retained_indices=None):
    """Select top-N_select observable indices from a rank vector.

    Parameters
    ----------
    rank_vector   : (N_total,) PageRank / PPR scores
    coord_indices : indices of current-time observables (always kept first)
    N_delay       : total number of delay levels
    N_select      : total number of observables to return
    group_by_delay: if True, x/y/z at each delay level are selected together
    retained_indices: optional original indices surviving zero-row pruning

    Returns
    -------
    selected : (N_select,) int array, coord_indices first
    """
    coord_set  = set(coord_indices.tolist())
    retained = (set(range(len(rank_vector))) if retained_indices is None
                else set(np.asarray(retained_indices).tolist()))
    if not coord_set.issubset(retained):
        raise ValueError("Required coordinate observables were removed by zero-row pruning")
    if N_select > len(retained):
        raise ValueError("Requested dictionary size exceeds the retained observable count")
    other_mask = np.array([i for i in range(len(rank_vector))
                           if i not in coord_set and i in retained], dtype=int)
    n_extra = max(0, N_select - len(coord_indices))

    if not group_by_delay:
        scores_other = rank_vector[other_mask]
        top_other    = other_mask[np.argsort(scores_other)[::-1][:n_extra]]
    else:
        d = 3
        other_levels  = sorted(set(col // d for col in other_mask))
        level_scores  = {}
        for lv in other_levels:
            cols = [d * lv + dim for dim in range(d) if d * lv + dim not in coord_set]
            if cols and all(col in retained for col in cols):
                level_scores[lv] = rank_vector[cols].sum()
        eligible_count = sum(len([d * lv + dim for dim in range(d)
                                  if d * lv + dim not in coord_set])
                             for lv in level_scores)
        if n_extra > eligible_count:
            raise ValueError("Requested dictionary size exceeds the complete retained delay groups")
        sorted_levels = sorted(level_scores, key=level_scores.get, reverse=True)
        top_other = []
        for lv in sorted_levels:
            cols = [d * lv + dim for dim in range(d) if d * lv + dim not in coord_set]
            top_other.extend(cols)
            if len(top_other) >= n_extra:
                break
        top_other = np.array(top_other[:n_extra], dtype=int)

    return np.concatenate([coord_indices, top_other]).astype(int)


# ═════════════════════════════════════════════════════════════════════════════
# 7.  Graph utilities  (retained for completeness)
# ═════════════════════════════════════════════════════════════════════════════

def list2onehot(y, list_classes=None):
    if list_classes is None:
        list_classes = list(np.sort(y))
    Y = np.zeros((len(y), len(list_classes)), dtype=int)
    for i, yi in enumerate(y):
        for j, c in enumerate(list_classes):
            if yi == c:
                Y[i, j] = 1
    return Y


def SBM(W, c):
    k = W.shape[0]
    C = list2onehot(c, list_classes=list(range(k)))
    P = C @ W @ C.T
    return weight2graph(P)


def weight2digraph(P):
    n = P.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    nodes = list(G.nodes())
    for i in range(n):
        for j in range(n):
            if np.random.rand() < P[i, j]:
                G.add_edge(nodes[i], nodes[j])
    return G


def weight2graph(P):
    n = P.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    nodes = list(G.nodes())
    for i in range(n):
        for j in range(i + 1, n):
            if np.random.rand() < P[i, j]:
                G.add_edge(nodes[i], nodes[j])
    return G


def RW(G, x0=None, steps=1, return_history=False):
    x = np.random.choice(list(G.nodes())) if x0 is None else x0
    history = []
    for _ in range(steps):
        nbrs = list(G.neighbors(x))
        if not nbrs:
            x = np.random.choice(list(G.nodes()))
        else:
            x = np.random.choice(nbrs)
        if return_history:
            history.append(x)
    return history if return_history else x
