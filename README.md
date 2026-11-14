# HAMCOR

**H**amiltonian-based **A**GN **M**ulti-constraint **COR**onal inference — a geometry-agnostic
method to infer the emissivity distribution of the X-ray corona of accreting black holes from
reverberation lags.

[![arXiv](https://img.shields.io/badge/arXiv-2607.11805-b31b1b.svg)](https://arxiv.org/abs/2607.11805)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-%E2%89%A53.10-blue)

Instead of fitting a parametric geometry (lamppost, slab, …), HAMCOR represents the corona as a
non-negative emissivity field on a cylindrical grid around the black hole and selects, among all
fields compatible with the data, the one that minimises a Hamiltonian

```
H(w) = H_mag(w)  +  ½ χ²_lag(w)  +  ½ χ²_I(w)  [ +  ½ χ²_lag-energy(w) ]
       smoothness    lag-frequency   illumination    energy-resolved lags
       (GMRF prior)  spectrum        fraction R_f/(1+R_f)
```

on the probability simplex, with projected gradient descent and multiple starts.

**Main features**

* **Exact photon transfer** — null geodesics in **Schwarzschild** (`raytrace.py`) and **Kerr**
  (`kerr.py`, Kerr–Schild coordinates, prograde disc from the ISCO) spacetimes: light bending,
  capture, Shapiro delay, frame dragging. Weak-field and flat-space transfer are available as
  comparisons.
* **Estimator-matched likelihood** — the model lags pass through the same cross-spectral
  estimator as the data (power-weighted Fourier frequencies, band averaging), with
  **dilution** by the continuum in each band and a profiled **intrinsic (continuum) hard lag**.
* **Measured illumination fraction** — `I = R_f/(1+R_f)` from relativistic reflection fits
  (relxill), with statistical, spin and emissivity-model systematics.
* **Energy-resolved transfer** — gravitational and Doppler shifts of the reflected spectrum
  (`energy.py`, rest-frame xillver templates), giving channel-dependent transfer functions and a
  lag-energy likelihood with an analytic gradient.
* **Honest uncertainty** — bootstrap refits, Fisher-mode analysis of how many emissivity
  combinations the data constrain (`N_eff`), parity test, lamppost comparison by AIC.
* **Full pipeline** from XMM-Newton ODFs to the tables and PGFPlots figures of the paper.

> **Scientific summary (v2).** Lag-frequency spectra constrain only ~4 combinations of the
> emissivity; centroids are recovered within 2–6 r_g. In eight XMM-Newton observations of
> Mrk 335, 1H 0707−495, IRAS 13224−3809 and MCG−6-30-15 the data are consistent with, but do not
> require, an extended corona: a lamppost is preferred by the AIC. See the paper for details.

---

## Contents

| Path | What |
|---|---|
| `hamcor2/` | the Python package (grids, ray tracing, forward model, Hamiltonian, optimiser, timing, I/O) |
| `scripts/` | numbered pipeline steps `s00`–`s13` |
| `config/sources.yaml` | sources, ObsIDs, masses, inclinations, frequency bands, spectral models |
| `examples/quickstart.py` | five-minute synthetic example, no X-ray software needed |
| `docs/TUTORIAL.md` | step-by-step tutorial (Python API and full pipeline) |
| `run_all.sh` | runs the whole pipeline |
| `CITATION.cff` | how to cite |

### Package modules

| Module | Content |
|---|---|
| `grid.py` | coronal grid (log-spaced R, z; uniform φ) and disc grid |
| `raytrace.py` | Schwarzschild ray tracing: cell→disc solid angles Ω_ik, path delays, escape, observer delays |
| `kerr.py` | Kerr null geodesics (RK4 Hamiltonian flow), ZAMO/static frames, shooting for observer delays, image-plane disc tracing (delays and redshifts g) |
| `forward.py` | `ForwardModel` (extended corona) and `PointSourceModel` (lamppost); metrics `flat`, `weakfield`, `schwarzschild`, `kerr` |
| `hamiltonian.py` | the Hamiltonian, its analytic gradient, diagnostics, continuum-lag profiling |
| `optimize.py` | projected gradient descent on the simplex with Armijo backtracking, multi-start |
| `fitting.py` | `fit`, `bootstrap`, map export |
| `energy.py` | energy-resolved transfer and the lag-energy likelihood term |
| `timing.py` | segment selection, cross spectra, lag-frequency, band lags, lag-energy, coherence |
| `metrics.py` | recovery metrics, parity split, Fisher spectrum / `N_eff` |
| `synthetic.py` | synthetic coronae and synthetic observables |
| `io.py`, `config.py`, `pgf.py` | configuration, observables, PGFPlots output |

---

## Installation

```bash
git clone https://github.com/fbuffoli95/HAMCOR.git
cd HAMCOR
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -e .
python scripts/s00_tests.py              # must end with "ALL PASS"
python examples/quickstart.py            # synthetic example, ~1 min
```

The Python part (steps 0, 2, 4–9, 11) runs anywhere with Python ≥ 3.10.
The real-data steps need, in addition (tested on WSL2 Ubuntu 22.04):

* **XMM-Newton SAS** ≥ 22.1 (step 1: light curves and spectra),
* **HEASoft** ≥ 6.33 with command-line **XSPEC** (steps 3, 10, 12),
* **relxill** ≥ 2.3 with the `xillverCp` tables (`export RELXILL_DIR=/path/to/relxill`).

```bash
source ~/heainit.sh && source $SAS_DIR/setsas.sh && source .venv/bin/activate
```

Data products and results are written under `$HAMCOR_WORK` (default: the repository root):
`data/<source>/…` and `results/…`. Ray-tracing tables are cached in `cache/` and reused.

---

## Quick start (no X-ray software)

```bash
python examples/quickstart.py                          # weak-field transfer
python examples/quickstart.py --metric schwarzschild   # exact ray tracing (builds a cache once)
python examples/quickstart.py --geometry ring
```

The example builds the grids, simulates the lag-frequency spectrum of a known corona through
the real estimator, recovers it, reports centroid offset, map correlation and `N_eff`, and
compares with a lamppost by AIC. A full walk-through of the API is in
[`docs/TUTORIAL.md`](docs/TUTORIAL.md).

A dry run of the whole pipeline on simulated light curves (no SAS/XSPEC):

```bash
python scripts/mock_pipeline.py
```

---

## Reproducing the paper

1. Download the XMM-Newton ODFs of the ObsIDs listed in `config/sources.yaml` from the
   [XMM-Newton Science Archive](https://nxsa.esac.esa.int) into `data/raw/<obsid>.tar`
   (or extracted into `data/raw/<obsid>/`).
2. Run

```bash
bash run_all.sh                 # or STEP=n bash run_all.sh to restart from step n
```

| Step | Script | Needs | Output |
|---:|---|---|---|
| 0 | `s00_tests.py` | — | unit tests (ray tracing, gradients, estimator) |
| 1 | `s01_sas_extract.py` | SAS | event filtering, light curves (soft, hard, 16 channels), spectra, regions |
| 2 | `s02_lagfreq.py` | 1 | lag-frequency, band lags, lag-energy, coherence, per-channel rates |
| 3 | `s03_fit_spectra.py` | XSPEC + relxill | R_f, I, dilution R_s, R_h, per-channel R_E, errors and systematics |
| 4 | `s05_synthetic.py` | — | recovery tests, flat vs exact transfer, J scan, Fisher N_eff, parity |
| 5 | `s06_fit_sources.py` | 2, 3 | HAMCOR fits (Schwarzschild), bootstrap, variants, Fisher modes |
| 6 | `s07_lamppost.py` | 5 | lamppost fits, ΔAIC |
| 7 | `s08_lag_energy.py` | 5, 6 | lag-energy posterior-predictive check |
| 8 | `s09_tables.py` | all | LaTeX tables |
| 9 | `s10_kerr.py` | 2, 3 | Kerr transfer at a = 0.998 and 0: fits, lamppost, parity |
| 10 | `s11_xillver_templates.py` | XSPEC + relxill | rest-frame reflection templates |
| 11 | `s12_energy_resolved.py` | 5, 9, 10 | energy-resolved predictions and joint lag-frequency + lag-energy fits |
| 12 | `s13_reflection_robustness.py` | XSPEC + relxill | model dependence of the reflection fraction |

Run times on two cores: steps 5–8 a few minutes per source; the Kerr transfer (step 9) about
30 min per source and spin the first time (cached afterwards).

Before trusting the results check, for each source, `data/<key>/header.json`
(OBJECT, DATE-OBS, SUBMODE), the background region in `data/<key>/regions.reg` and the
pile-up plot in `data/<key>/sas/`.

### Configuration

`config/sources.yaml` holds global defaults (spin, systematics, grid, disc) and one block per
observation: ObsID, mass, inclination, energy bands, reverberation frequency band
(`fit_band`), spectral model and fit options, and the emissivity-model systematic on the
dilution (`D_emiss_sys`). To add a source, copy a block and change these fields.

---

## Citation

If you use HAMCOR, please cite the paper and the software (see [`CITATION.cff`](CITATION.cff);
GitHub's *Cite this repository* button gives BibTeX):

```bibtex
@article{Buffoli2026HAMCOR,
  author  = {Buffoli, Fabio},
  title   = {{HAMCOR}: inferring X-ray coronal geometry from reverberation lags with a
             Hamiltonian formulation and exact photon transfer in Schwarzschild and Kerr spacetimes},
  journal = {arXiv e-prints},
  year    = {2026},
  eprint  = {2607.11805},
  archivePrefix = {arXiv},
  url     = {https://arxiv.org/abs/2607.11805}
}

@software{Buffoli2026HAMCORcode,
  author  = {Buffoli, Fabio},
  title   = {{HAMCOR}: Hamiltonian inference of X-ray coronal geometry (v2.0.0)},
  year    = {2026},
  url     = {https://github.com/fbuffoli95/HAMCOR}
}
```

## Licence

MIT — see [`LICENSE`](LICENSE).

## Contact

Fabio Buffoli — Università degli Studi di Brescia — fabio.buffoli@unibs.it —
ORCID [0009-0002-6524-2388](https://orcid.org/0009-0002-6524-2388)
