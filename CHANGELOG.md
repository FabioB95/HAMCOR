# Changelog

## v2.0.0 (2026-11)

Complete rewrite of the method and of the data analysis with respect to v1
(arXiv:2607.11805v1).

### Added
* Exact Kerr photon transfer (`hamcor2/kerr.py`): Kerr–Schild null geodesics, disc from the
  prograde ISCO, multi-start shooting for observer delays (earliest arrival = primary image),
  image-plane tracing for disc delays and redshifts; validated against the Schwarzschild module
  at a = 0.
* Energy-resolved transfer (`hamcor2/energy.py`): g-factors and rest-frame xillver templates,
  channel-dependent transfer functions, lag-energy likelihood with analytic gradient, joint
  lag-frequency + lag-energy fits (`s12`).
* Lamppost scan down to h = 1.12 r_g (just outside the horizon at a = 0.998).
* Spectral robustness of the reflection fraction (`s13`).
* Coherence quality cut for lag-energy spectra (median raw coherence ≥ 0.05).

### Changed with respect to v1
| Issue | v1 | v2 |
|---|---|---|
| Spacetime | flat in all fits; GR only in an appendix | exact Schwarzschild null geodesics (`raytrace.py`: light bending, capture, Shapiro delay) in every fit; weak-field and flat as comparisons |
| Illumination | I = sum Omega / 4 pi with Omega = cos dA/d^2 (over-counts near the disc: I up to 0.75) | ray-traced, I = N_disc/(N_disc+N_inf) = R_f/(1+R_f), the relxill definition; Schwarzschild maximum I = 0.56 |
| Spectral spin | a = 0.998 (inconsistent with a Schwarzschild forward model) | a = 0, R_in = ISCO = 6 r_g (same disc as HAMCOR); a = 0.998 refit as systematic on I and D |
| Continuum lag | not modelled | profiled A (nu/nu0)^-a (A >= 0) |
| Shapiro delay | v1 module: wrong sign of observer projection; monotonic legs returned the 20 r_g/c cap (gave "~79 %") | exact weak-field integral `2 ln[(r_A+r_B+l)/(r_A+r_B-l)]`, valid for V-shaped and monotonic legs; Ω-weighted correction ≈ 12–13 % |
| Data term | Gaussian ψ_obs "prior" centred on the scalar lag | χ² on the measured lag-frequency points; model passed through the same cross-spectral estimator (power-weighted, end-matched segments) |
| Dilution | ignored | τ_sh = [arg(1+R_s ψ̃) − arg(1+R_h ψ̃)]/(2πν), R_s, R_h from the spectral model; low-f limit −𝒟⟨t⟩ |
| R_obs | hard-coded (0.3, 0.5, 3.0, invented per epoch) | relxill R_f with covariance draws, pexrav cross-check as systematic; I = R_f/(1+R_f) is the definition of R_f (Dauser+2016) |
| Light curves | whole pn field, no source region, no background subtraction | 40″ source / 50″ background, `epiclccorr` |
| Lag errors | 50 % overlapping segments, error over-reduced by √N | independent segments, Bendat & Piersol errors, Poisson-corrected coherence |
| H_mag | −J Σ w_i w_j n_i·n_j: unbounded towards two-cell clumps | + on-site self-energy ⇒ (J/2)κ Σ (w_i−w_j)² (GMRF prior), positive semi-definite |
| "Column" synthetic | radius 3 r_g on a grid with R ≥ 3.25 r_g: it was a uniform corona | redefined on the grid (R < 5, 1 < z < 5) |
| GR validation | monkey-patched an attribute of shape (N_c, N_d) — may have replaced Ω, not τ | explicit `ForwardModel(metric=...)` |
| Masses | MCG-6 3e6 (Bentz+16: 1.6e6), Cyg X-1 14.8 | literature values in `config/sources.yaml` |
