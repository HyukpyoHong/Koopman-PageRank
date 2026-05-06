"""
Step 1: 3-well Ramachandran potential, overdamped Langevin, numba-accelerated.
"""
import os, numpy as np, math
from numba import njit

OUTPUT_DIR = "data/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@njit
def simulate(n_steps, save_every, dt, beta, phi0, psi0, rng_seed = 123):
    np.random.seed(rng_seed)
    sig = math.sqrt(2*dt/beta)
    phi, psi = phi0, psi0
    n_save = n_steps // save_every
    out = np.empty((n_save, 2))
    eps = 1e-5
    idx = 0
    cx = np.array([-1.0, -1.0,  1.1])
    cy = np.array([-1.0,  1.2, -0.3])
    A  = np.array([-6.0, -6.0, -4.0])
    s  = np.array([ 0.55, 0.55, 0.65])
    for step in range(n_steps):
        Vpp = 0.0; Vpm = 0.0; Vpq = 0.0; Vmq = 0.0
        for k in range(3):
            dp1 = math.atan2(math.sin(phi+eps-cx[k]), math.cos(phi+eps-cx[k]))
            dp2 = math.atan2(math.sin(phi-eps-cx[k]), math.cos(phi-eps-cx[k]))
            dp0 = math.atan2(math.sin(phi-cx[k]),     math.cos(phi-cx[k]))
            dq0 = math.atan2(math.sin(psi-cy[k]),     math.cos(psi-cy[k]))
            dq1 = math.atan2(math.sin(psi+eps-cy[k]), math.cos(psi+eps-cy[k]))
            dq2 = math.atan2(math.sin(psi-eps-cy[k]), math.cos(psi-eps-cy[k]))
            s2 = s[k]*s[k]
            Vpp += A[k]*math.exp(-0.5*(dp1*dp1+dq0*dq0)/s2)
            Vpm += A[k]*math.exp(-0.5*(dp2*dp2+dq0*dq0)/s2)
            Vpq += A[k]*math.exp(-0.5*(dp0*dp0+dq1*dq1)/s2)
            Vmq += A[k]*math.exp(-0.5*(dp0*dp0+dq2*dq2)/s2)
        Vpp += 0.3*(math.cos(2*(phi+eps))+math.cos(2*psi))
        Vpm += 0.3*(math.cos(2*(phi-eps))+math.cos(2*psi))
        Vpq += 0.3*(math.cos(2*phi)+math.cos(2*(psi+eps)))
        Vmq += 0.3*(math.cos(2*phi)+math.cos(2*(psi-eps)))
        gp = (Vpp-Vpm)/(2*eps)
        gq = (Vpq-Vmq)/(2*eps)
        phi += -gp*dt + np.random.randn()*sig
        psi += -gq*dt + np.random.randn()*sig
        phi = math.atan2(math.sin(phi), math.cos(phi))
        psi = math.atan2(math.sin(psi), math.cos(psi))
        if step % save_every == 0:
            out[idx,0] = phi; out[idx,1] = psi; idx += 1
    return out

# JIT warmup
print("JIT compiling...")
_ = simulate(100, 10, 0.005, 1.0, -1.0, -1.0)

# Equilibrate
print("Equilibrating...")
eq = simulate(100000, 100000, 0.005, 1.0, -1.0, -1.0)
phi0, psi0 = eq[-1]

# Production
print("Production run...")
trajectory = simulate(1000000, 10, 0.005, 1.0, phi0, psi0)
print(f"Generated {len(trajectory)} frames")

# Basin stats
cx = np.array([-1.0, -1.0, 1.1])
cy = np.array([-1.0, 1.2, -0.3])
def assign(traj):
    basins = np.zeros(len(traj), dtype=int)
    for i in range(len(traj)):
        dists = [(np.arctan2(np.sin(traj[i,0]-cx[k]),np.cos(traj[i,0]-cx[k]))**2 +
                  np.arctan2(np.sin(traj[i,1]-cy[k]),np.cos(traj[i,1]-cy[k]))**2)
                 for k in range(3)]
        basins[i] = np.argmin(dists)
    return basins

basins = assign(trajectory)
trans = np.sum(np.diff(basins) != 0)
print(f"Transitions: {trans} ({trans/len(trajectory)*100:.1f}%)")
for b in range(3):
    print(f"  Basin {b}: {np.mean(basins==b)*100:.1f}%")

np.save(os.path.join(OUTPUT_DIR, "trajectory.npy"), trajectory)
np.save(os.path.join(OUTPUT_DIR, "basins.npy"), basins)

n_traj = 20

print(f"Generating {n_traj} trajectories...")
trajectories = []
for i in range(n_traj):
    # Cycle through well centers + small perturbation for diversity
    # wc = eq[-1]well_centers[i % 3]
    rng = np.random.RandomState(i)
    # phi0 = wc[0]# + rng.uniform(-0.3, 0.3)
    # psi0 = wc[1]# + rng.uniform(-0.3, 0.3)
    phi0, psi0 = eq[-1]

    # Short equilibration
    # eq = simulate(100_000, 100_000, 0.005, 1.0, phi0, psi0, rng_seed=1000+i)
    # phi0, psi0 = eq[-1]

    # Production
    traj = simulate(1_000_000, 10, 0.005, 1.0, phi0, psi0, rng_seed=i)
    trajectories.append(traj)
    print(f"  [{i+1:2d}/20] {len(traj)} frames, "
          f"phi0={phi0:.2f}, psi0={psi0:.2f}")

# Save
np.save(os.path.join(OUTPUT_DIR, "trajectories_20.npy"),
        np.array(trajectories))   # shape: (20, 100000, 2)
print("Saved trajectories_20.npy  shape:", np.array(trajectories).shape)

print("Done!")
