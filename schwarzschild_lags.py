"""
schwarzschild_lags.py
─────────────────────
Drop-in replacement for flat-spacetime lag matrix computation,
adding first-order Schwarzschild (non-rotating) corrections via
the Shapiro time delay.

Physics:
  For a photon path with periapsis between A and B (geometry a V):
      Δt = 2 [ln(r_A+√(r_A²−b²)) + ln(r_B+√(r_B²−b²)) − 2 ln b]

  For a monotonic path (no periapsis between A and B):
      Δt = 2 |ln(r_B+√(r_B²−b²)) − ln(r_A+√(r_A²−b²))|

  b is clipped at B_MIN_RG = 1.5 (near photon sphere).

Usage:
    from schwarzschild_lags import compute_lag_matrix_schwarzschild
    LM = compute_lag_matrix_schwarzschild(GRID, DISK,
                                           observer_inclination=45.0)

Reference:
  Misner, Thorne & Wheeler (1973), §40.4
  Shapiro (1964), Phys. Rev. Lett.
  Dauser et al. (2013), MNRAS 430, 1694
"""

import numpy as np
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT     = 1.0
B_MIN_RG    = 1.5
SHAPIRO_CAP = 20.0
R_FAR       = 1.0e6   # proxy per "infinito" nei leg verso osservatore
# ─────────────────────────────────────────────────────────────────────────────


def _cartesian(R: float, z: float, phi: float) -> np.ndarray:
    return np.array([R * np.cos(phi), R * np.sin(phi), z])


def _schwarzschild_r(R: float, z: float) -> float:
    return float(np.sqrt(R**2 + z**2))


def _impact_parameter(x_A: np.ndarray, x_B: np.ndarray) -> Tuple[float, bool]:
    """
    Ritorna (b, has_periapsis_between).

    b = distanza minima dall'origine al segmento A–B.
    has_periapsis_between = True se il punto di massimo avvicinamento
    cade strettamente tra A e B (geometria a V).
    """
    AB = x_B - x_A
    len_AB = float(np.linalg.norm(AB))
    if len_AB < 1e-12:
        return float(np.linalg.norm(x_A)), False

    t_star_raw = -float(np.dot(x_A, AB)) / (len_AB**2)
    has_periapsis = (0.0 <= t_star_raw <= 1.0)
    t_star = float(np.clip(t_star_raw, 0.0, 1.0))
    closest = x_A + t_star * AB
    return float(np.linalg.norm(closest)), has_periapsis


def _shapiro_delay(
    r_A: float,
    r_B: float,
    b: float,
    has_periapsis_between: bool,
) -> float:
    """
    Shapiro delay in r_g/c.

    Con periapsi tra A e B:  2 (log_A + log_B - 2 log b)
    Senza (path monotono):   2 |log_B - log_A|
    """
    b_eff = max(float(b), B_MIN_RG)
    term_A2 = max(0.0, r_A**2 - b_eff**2)
    term_B2 = max(0.0, r_B**2 - b_eff**2)
    log_A = np.log(r_A + np.sqrt(term_A2))
    log_B = np.log(r_B + np.sqrt(term_B2))

    if has_periapsis_between:
        dt = 2.0 * (log_A + log_B - 2.0 * np.log(b_eff))
    else:
        dt = 2.0 * abs(log_B - log_A)

    return float(np.clip(dt, 0.0, SHAPIRO_CAP))


def _observer_hat(incl_deg: float) -> np.ndarray:
    th = np.radians(incl_deg)
    return np.array([np.sin(th), 0.0, np.cos(th)])


def compute_lag_matrix_schwarzschild(
    grid,
    disk,
    observer_inclination: float = 30.0,
    spin: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    """
    Lag matrix (N_corona, N_disc) con correzioni Schwarzschild.

    lag = (d_ik + d_k_obs - d_i_obs)_flat + Δt_ik + Δt_k_obs - Δt_i_obs
    """
    if spin != 0.0:
        import warnings
        warnings.warn(
            "Non-zero spin not yet implemented; using Schwarzschild (a=0).",
            UserWarning,
            stacklevel=2,
        )

    obs_hat = _observer_hat(observer_inclination)
    N = grid.n_cells
    M = disk.n_elements
    lag_matrix = np.zeros((N, M))

    for i in range(N):
        if verbose and i % 100 == 0:
            print(f"  cell {i}/{N}", end='\r', flush=True)

        Ri, zi, phi_i = float(grid.R[i]), float(grid.z[i]), float(grid.phi[i])
        x_i = _cartesian(Ri, zi, phi_i)
        r_i = _schwarzschild_r(Ri, zi)
        d_i_obs_flat = float(np.dot(x_i, obs_hat))

        for k in range(M):
            Rk = float(disk.R[k])
            phi_k = float(disk.phi[k])
            x_k = _cartesian(Rk, 0.0, phi_k)
            r_k = Rk

            # ── Leg 1: corona i → disco k ─────────────────────────────────
            d_ik_flat = float(np.linalg.norm(x_i - x_k))
            b_ik, has_peri_ik = _impact_parameter(x_i, x_k)
            dt_ik_GR = _shapiro_delay(r_i, r_k, b_ik, has_peri_ik)

            # ── Leg 2: disco k → osservatore (verso infinito, no periapsi) ─
            d_k_obs_flat = float(np.dot(x_k, obs_hat))
            cos_k = float(np.dot(x_k / (r_k + 1e-15), obs_hat))
            sin_k = float(np.sqrt(max(0.0, 1.0 - cos_k**2)))
            b_k_obs = max(r_k * sin_k, B_MIN_RG)
            dt_k_obs_GR = _shapiro_delay(r_k, R_FAR, b_k_obs, False)

            # ── Leg 3: corona i → osservatore (verso infinito, no periapsi) ─
            cos_i = float(np.dot(x_i / (r_i + 1e-15), obs_hat))
            sin_i = float(np.sqrt(max(0.0, 1.0 - cos_i**2)))
            b_i_obs = max(r_i * sin_i, B_MIN_RG)
            dt_i_obs_GR = _shapiro_delay(r_i, R_FAR, b_i_obs, False)

            # ── Totale ────────────────────────────────────────────────────
            lag_flat = (d_ik_flat + d_k_obs_flat - d_i_obs_flat) / C_LIGHT
            lag_GR = lag_flat + dt_ik_GR + dt_k_obs_GR - dt_i_obs_GR
            lag_matrix[i, k] = max(0.0, lag_GR)

    if verbose:
        print()

    return lag_matrix


def compare_flat_vs_schwarzschild(grid, disk, incl_deg: float = 30.0) -> dict:
    from qcorona.physics.lags import compute_lag_matrix as flat_lm

    LM_flat = flat_lm(grid, disk, observer_inclination=incl_deg)
    LM_schw = compute_lag_matrix_schwarzschild(
        grid, disk, observer_inclination=incl_deg,
    )
    correction = LM_schw - LM_flat
    frac = correction / (LM_flat + 1e-10)

    return dict(
        LM_flat=LM_flat,
        LM_schwarzschild=LM_schw,
        mean_correction_rgc=float(correction.mean()),
        median_frac_correction=float(np.median(frac)),
        max_frac_correction=float(frac.max()),
        min_lag_flat=float(LM_flat.min()),
        min_lag_schwarzschild=float(LM_schw.min()),
    )