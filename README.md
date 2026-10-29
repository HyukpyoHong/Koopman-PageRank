# Finding Koopman Invariant Subspaces via Personalized PageRank

Code for the paper:

> **Finding Koopman Invariant Subspaces via Personalized PageRank**  
> Hyukpyo Hong, Matthew J. Colbrook, Qin Li, and Hanbaek Lyu

---

## Repository layout

```
.
├── README.md
├── requirements.txt
├── src/
│   ├── koopman_source.py
│   └── pagerank_utils.py
├── tests/
└── experiments/
    ├── 01_toy_model.ipynb
    ├── 02_detection_windows.ipynb
    ├── 03_duffing_vdp.ipynb
    ├── 04_01_Ramachandran_generate_trajectories.py
    ├── 04_02_Ramachandran_edmd_pagerank.py
    ├── 04_03_Ramachandran_evaluate.py
    ├── 04_04_Ramachandran_alpha_sensitivity.py
    ├── 05_lorenz.ipynb
    ├── data/
    ├── results_196_train_only/
    └── figures_196_train_only_N75/
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

After Step 2, run the damping-factor study with:

```bash
python 04_04_Ramachandran_alpha_sensitivity.py
```

It evaluates 10 predefined alpha values from 0.05 to 0.95 at N=5,10,20,40,
using the four fixed state-coordinate seeds and the saved train-only models.
It writes per-trajectory results, a summary CSV, and a PNG under
`validation/alpha_sensitivity_196_train_only/`. Held-out MSE is computed with
sufficient statistics using the same ridge readout as Step 3, with regression
checks against direct prediction. The reported alpha CV is computed within
each trajectory before aggregation; it is distinct from variation across
trajectories. Baseline comparison output distinguishes ratios of mean errors
from means/standard deviations of paired per-trajectory ratios.

Output paths below are relative to `experiments/`:

| Script | Output location | Contents |
|---|---|---|
| `04_01_...` | `data/` | Simulated trajectories |
| `04_02_...` | `results_196_train_only/` | Per-trajectory EDMD results fitted on the training prefix |
| `04_03_...` | `figures_196_train_only_N75/` and `../validation/full_evaluation_196_train_only/` | Five PDF/PNG figure sets, raw trajectory errors and JSON summary |

The Ramachandran dictionary now contains 196 observables: 4 state coordinates,
28 higher Fourier terms, 64 cross terms, and 100 Gaussian RBFs. The 40 diagonal
Fourier modes are omitted because angle-addition identities express them using
the preceding Fourier/cross terms. State-coordinate seeds retain indices 0--3.
Older 236-observable results are preserved; Step 3 rejects them rather than
presenting them as results from the 196-observable dictionary.

To reuse data in another directory, set `KOOPMAN_RAMACHANDRAN_DATA_DIR`.
The new result and figure directories can be overridden with
`KOOPMAN_RAMACHANDRAN_RESULTS_DIR` and `KOOPMAN_RAMACHANDRAN_FIGURE_DIR`.
For each trajectory of `T` frames, `n_train_pairs = floor(0.8 * (T - 1))`.
Scaling is fitted on frames `0` through `n_train_pairs` inclusive. EDMD uses
only the pairs between those frames. TICA, k-means, MSM/PCCA+, and the Fisher
ranking also use only that prefix. The trained scaling and cluster mapping may
be applied to test frames for prediction and plots without refitting. The last
training frame supplies the initial state for the first test prediction; its
target is held out. This split does not assert statistical independence of the
two trajectory segments.

Saved results include the fitted dictionary mean/std, the split boundary, fit
seed, and `evaluation_protocol="train_only_v1"`. Evaluation rejects older caches
without that protocol metadata. The state readout predicts the next state, so
the horizon-h prediction is `Kread @ Ksub**(h-1) @ z_t`: read out first, score
the prediction, then propagate. Multi-step evaluation uses starts with every
requested horizon available. With `steps=1`, it uses all one-step test pairs;
for longer rollouts, compare its first horizon with one-step error on the same
restricted starting points. New publication results require rerunning the
experiments; no results are regenerated by changing the code.

Step 3 evaluates N=4--80 and a 30-step rollout at N=15 on all 20 trajectories.
One-step MSE uses all 20,000 held-out pairs; the 30-step curve uses the same
19,971 starts at every horizon. The accelerated one-step scoring is checked
against direct predictions, and batched rollouts preserve the readout timing.
`trajectory_errors.npz` contains each trajectory's errors, and
`evaluation_results.json` contains means and empirical standard deviations
(ddof=0). Ratio plots and tables use the mean/std of method-to-Random ratios
paired within each trajectory; ratio-of-means values are stored separately.
The validation output location can be set with
`KOOPMAN_RAMACHANDRAN_EVALUATION_DIR`. Generated figures are saved separately
from the manuscript figures until the result update is approved. The full
N=4--80 evaluation is retained; one-step and ratio plots display N=4--75.
To redraw figures from the saved error arrays without rerunning evaluation:

```bash
KOOPMAN_RAMACHANDRAN_PLOT_ONLY=1 python 04_03_Ramachandran_evaluate.py
```

`KOOPMAN_RAMACHANDRAN_PLOT_N_MAX` changes only the displayed upper bound
(default 75, at most 80). The default output folder includes that bound, so
older N=80 plots remain preserved. `plot_manifest.json` records both the full
source range and displayed range, input hashes and actual axis limits.

> **Tip.** The first code `04_01_...`  contains `n_traj` representing the number of trajectories used for analysis. With the default value of `n_traj = 20`, the first code `04_01_...` takes 10 seconds, the second code `04_02_...` takes 2 minutes, and the third code uses accelerated scoring and batched rollouts. Timings depend on hardware.

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

PR/PPR removes zero-outgoing observables and their corresponding rows/columns
repeatedly, then normalizes the retained graph. `koopman_source` accepts column
weights; the Ramachandran helpers accept row weights. Score vectors keep the
original observable indexing and assign removed nodes zero. Obtain the retained
indices with `pagerank_utils.prune_zero_rows(weights_in_row_convention)[1]`;
only those indices participate in PR/PPR rankings. A seed removed by pruning,
or a graph with no retained nodes, raises `ValueError`.
`select_top_indices` accepts these indices as `retained_indices`; the Lorenz
notebook supplies them for PR/PPR and raises if a requested size cannot be met.

The Ramachandran results save `ppr_retained_indices` alongside `K_hat`, whose
rows and columns use that retained order. PR/PPR ordering arrays use original
observable indices. Evaluation returns `NaN` for requested dictionary sizes
larger than an available ordering. Existing saved results are not regenerated
by this code change.

Run the focused regression checks from the repository root with
`python -m unittest discover -s tests`.

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

The code in this repository is released under the MIT License; see `LICENSE`.
Third-party dependencies are distributed under their respective licenses,
including LGPL v3 for `deeptime`.
