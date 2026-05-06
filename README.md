# Finding Koopman Invariant Subspaces via Personalized PageRank

Code for the paper:

> **Finding Koopman Invariant Subspaces via Personalized PageRank**  
> Anonymous Authors

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   └── koopman_source.py
└── experiments/
    ├── 01_toy_model.ipynb
    ├── 02_detection_windows.ipynb
    ├── 03_duffing_vdp.ipynb
    ├── 04_01_Ramachandran_generate_trajectories.py
    ├── 04_02_Ramachandran_edmd_pagerank.py
    ├── 04_03_Ramachandran_evaluate.py
    ├── 05_lorenz.ipynb
    ├── data/
    ├── results/
    └── figures/
```

---

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10.

---

## Reproducing the results

### Step 1 — Notebooks (self-contained)

The following notebooks require no pre-computed data and can be run directly in JupyterLab:

```bash
cd experiments
jupyter lab
```

Open and run all cells in order:

| Notebook | Output |
|---|---|
| `01_toy_model.ipynb` | Appendix B (toy model, zero block verification, Theorem 2) |
| `02_detection_windows.ipynb` | Figure 1 (detection window curves) |
| `03_duffing_vdp.ipynb` | Figures 2, 5, 6, 7 (Duffing and Van der Pol) |
| `05_lorenz.ipynb` | Figure 4 (Lorenz eigenfunctions) |

> **Tip.** `03_duffing_vdp.ipynb` has a `num_rnd` variable in the first setup cell that controls the number of random seeds used for error-band plots. Set it to `2`–`3` for a quick check; the default `20` is used for the paper figures. When `num_rnd = 20`, it takes approximately 10 minutes. 
> ```python
> num_rnd = 20   # publication quality
> # num_rnd = 3  # quick sanity check
> ```

### Step 2 — Ramachandran pipeline (Table 1, Figures 3, 8, 9)

The Ramachandran experiment is split into three scripts that must be run sequentially from the `experiments/` directory:

```bash
cd experiments

python 04_01_Ramachandran_generate_trajectories.py
python 04_02_Ramachandran_edmd_pagerank.py
python 04_03_Ramachandran_evaluate.py
```

Each script writes its output to a subdirectory of `experiments/`:

| Script | Output location | Contents |
|---|---|---|
| `04_01_...` | `data/` | Simulated trajectories |
| `04_02_...` | `results/` | Per-trajectory EDMD results |
| `04_03_...` | `figures/` | Error table (Table 1) and figure panels (Figures 3, 8, 9) |

> **Tip.** The first code `04_01_...`  contains `n_traj` representing the number of trajectories used for analysis. With the default value of `n_traj = 20`, the first code `04_01_...` takes 10 seconds, the second code `04_02_...` takes 2 minutes, and the third code (there is the same variable `n_traj` separately in the third code) takes approximately 8 minutes.

---

## Source library (`src/koopman_source.py`)

All shared algorithms and dynamical system definitions live in a single source file. Key functions are listed below.

**Dynamical systems**

| Function | Description |
|---|---|
| `Duffing`, `VanDerPol`, `Lorenz`, `Lotka_Voltera` | Euler-step integrators |
| `lorenz_rk4` | RK4 integrator for the Lorenz system |

**Observable dictionaries**

| Function | Description |
|---|---|
| `psi_Laguerre`, `psi_evaluation_laguerre` | Laguerre polynomial dictionary |

**PageRank variants**

| Function | Description |
|---|---|
| `pagerank` | Standard PageRank on the EDMD weight matrix |
| `personalized_pagerank` | PPR with seed-node teleportation |

**EDMD and ordering**

| Function | Description |
|---|---|
| `find_best_lin` | Least-squares EDMD: `K = pinv(Psi_X) @ Psi_Y` |
| `compute_orderings` | Returns Unordered / Incremental / PR / PPR orderings |
| `compute_errors` | One-step L2 error at every sub-dictionary size |
| `compute_errors_multiseed` | Repeats over `num_rnd` seeds; returns `(num_rnd, N)` arrays |
| `compute_state_predictions` | Multi-step prediction trajectories |

**Baselines**

| Function | Description |
|---|---|
| `tssd_algorithm`, `compute_tssd_errors` | T-SSD (Haseli & Cortés 2023) |

**Lorenz helpers**

| Function | Description |
|---|---|
| `build_delay_embedding`, `select_top_indices` | Time-delay embedding construction and index selection |

---

## License

MIT
