# HAMCOR
### Hamiltonian-based AGN Multi-constraint CORonal inference framework

[![arXiv](https://img.shields.io/badge/arXiv-2607.11805-b31b1b.svg)](https://arxiv.org/abs/2607.11805)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

---

## What is HAMCOR?

HAMCOR is a physics-driven Python framework for inferring the geometry of the X-ray corona in active galactic nuclei (AGN) from reverberation lag measurements — **without assuming a lamppost geometry or any other a priori morphology**.

The corona is represented as a discrete emissivity distribution over a cylindrical grid. Its geometry emerges from minimising a physical Hamiltonian encoding five competing constraints simultaneously:

| Term | Physical meaning |
|------|-----------------|
| H_mag | Magnetic coherence |
| H_geo | Lag consistency |
| H_rad | Illumination consistency |
| H_pair | Pair-production stability |
| H_ener | Energy budget feasibility |

Minimisation uses projected gradient descent with Armijo backtracking on the probability simplex.

---

## Citation

**If you use HAMCOR in your research, you must cite the following paper:**

```
Buffoli, F. (2026). HAMCOR: A physics-driven Hamiltonian framework for
inferring AGN coronal geometry from X-ray reverberation lags.
arXiv:2607.11805 [astro-ph.HE]
```

BibTeX:
```bibtex
@article{Buffoli2026_HAMCOR,
  author  = {Buffoli, F.},
  title   = {{HAMCOR}: A physics-driven {Hamiltonian} framework for
             inferring {AGN} coronal geometry from {X}-ray
             reverberation lags},
  journal = {arXiv e-prints},
  year    = {2026},
  eprint  = {2607.11805},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.HE},
  doi     = {10.48550/arXiv.2607.11805}
}
```

> **Note:** Use of this code without citing the above paper is a violation of the licence terms (CC BY 4.0).

---

## Installation

```bash
git clone https://github.com/fbuffoli95/HAMCOR.git
cd HAMCOR
pip install -e .
```

Requirements: Python ≥ 3.9, NumPy, Matplotlib, Astropy, PyYAML.

---

## Quick start

```bash
# Run full pipeline (all sources)
python run_all.py

# Run single source fit
python run_mrk335.py

# Synthetic validation
python run_recovery.py
python make_figure_v3.py

# Cyg X-1 (cross-mass-scale)
python run_cygx1.py

# Mrk 335 multi-epoch
python run_mrk335_multiepoch.py
```

---

## Repository structure

```
HAMCOR/
├── qcorona/                    # Core library
│   ├── geometry/               # Grid construction
│   ├── hamiltonian/            # H terms + assembly
│   ├── physics/                # Lags, illumination, compactness
│   ├── optimization/           # Projected gradient descent
│   └── data/                   # Synthetic geometry tools
├── data/
│   └── sources/                # Observed light curves (npy)
│       ├── mrk335/
│       ├── 1h0707/
│       ├── iras13224/
│       ├── mcg6/
│       ├── mrk335_2009/        # Multi-epoch
│       ├── mrk335_2015/
│       ├── mrk335_2018/
│       └── mrk335_2019/
├── results/                    # Output figures and npz files
├── run_all.py                  # Master pipeline script
├── run_mrk335.py               # Mrk 335 2006 fit
├── run_mrk335_multiepoch.py    # Mrk 335 5-epoch analysis
├── run_1h0707.py               # 1H 0707-495 fit
├── run_iras13224.py            # IRAS 13224-3809 fit
├── run_mcg6.py                 # MCG-6-30-15 fit
├── run_cygx1.py                # Cyg X-1 fit
├── run_recovery.py             # Synthetic validation
├── make_figure_v3.py           # Publication figure (recovery)
├── run_sensitivity.py          # Hyperparameter sensitivity
├── run_gr_validation.py        # Schwarzschild corrections
├── schwarzschild_lags.py       # GR lag module
├── lag_energy_psi.py           # Multi-energy psi_obs
├── run_lag_mrk335.py           # Lag measurement Mrk 335 2006
├── run_lag_mrk335_epoch.py     # Lag measurement any epoch
├── run_lag_1h0707.py           # Lag measurement 1H 0707
├── compute_lag_iras13224.py    # Lag measurement IRAS 13224
├── convert_lc_to_numpy.py      # SAS FITS → numpy
├── save_best_emissivities.py   # Patch recovery npz
├── config.yaml                 # Grid and hyperparameter defaults
├── pyproject.toml              # Package metadata
├── LICENSE                     # CC BY 4.0
└── README.md
```

---

## Results from Buffoli (2026)

| Source | M_bh (M_sun) | lag_obs (r_g/c) | lag_pred (r_g/c) | R_c (r_g) | z_c (r_g) |
|--------|-------------|-----------------|-----------------|-----------|-----------|
| Mrk 335 | 1.3×10⁷ | 1.46 ± 0.69 | 2.24 | 7.5 | 0.4 |
| 1H 0707-495 | 2.0×10⁶ | 2.83 ± 0.50 | 3.65 | ~10 | ~1.5 |
| IRAS 13224 | 1.5×10⁶ | 2.28 ± 1.42 | 3.14 | ~8 | ~2 |
| MCG-6-30-15 | 3.0×10⁶ | 0.70 ± 0.68 | 1.64 | ~5 | ~0.5 |
| Cyg X-1 | 14.8 | 28.8 ± 5.5 | 23.8 | ~5 | ~6 |

Mrk 335 multi-epoch: coronal centroid stable at (R_c, z_c) ≈ (6.3, 0.5) r_g across 2006–2019 despite factor ~15 variation in lag amplitude.

---

## Contact

Fabio Buffoli — f.buffoli008@unibs.it  
Università degli Studi di Brescia, Italy

---

## Licence

This software is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** licence.  
You are free to use, share, and adapt this code **provided you cite Buffoli (2026), arXiv:2607.11805**.  
See [LICENSE](LICENSE) for full terms.
