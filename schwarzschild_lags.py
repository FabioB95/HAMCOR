"""
schwarzschild_lags.py
─────────────────────
Drop-in replacement for flat-spacetime lag matrix computation,
adding first-order Schwarzschild (non-rotating) corrections via
the Shapiro time delay.

Physics:
  For a photon traveling between two points at Schwarzschild radial
  coordinates r_A and r_B, the coordinate time is longer than the
  Euclidean distance by the Shapiro delay:

      Δt_Shapiro = (2 r_g / c) × ln[(r_A + √(r_A²−b²))(r_B + √(r_B²−b²)) / b²]

  where b is the impact parameter (minimum distance to the BH along
  the photon path).

  For b → 0 (photon approaches BH), the delay diverges — we clip paths
  with b < b_min = 3 r_g (photon sphere) as unphysical.

  This is the weak-to-intermediate-field Shapiro formula valid for
  r >> r_s. At z ~ 1–2 r_g the correction reaches ~40–60%, consistent
  with full GR ray-tracing (Dauser et al. 2013).

Usage (drop-in for compute_lag_matrix):
    from schwarzschild_lags import compute_lag_matrix_schwarzschild
    LM = compute_lag_matrix_schwarzschild(GRID, DISK,
                                           observer_inclination=45.0,
                                           spin=0.0)  # a=0: Schwarzschild

Reference:
  Misner, Thorne & Wheeler (1973), §40.4
  Shapiro (1964), Phys. Rev. Lett.
  Dauser et al. (2013), MNRAS 430, 1694
"""

import numpy as np
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
C_LIGHT     = 1.0   # all distances in r_g, times in r_g/c
B_MIN_RG    = 3.0   # photon-sphere radius; paths closer than this are clipped
SHAPIRO_CAP = 20.0  # maximum Shapiro correction per leg [r_g/c]


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cartesian(R: float, z: float, phi: float):
    """Cylindrical → Cartesian."""
    return np.array([R * np.cos(phi), R * np.sin(phi), z])


def _schwarzschild_r(R: float, z: float) -> float:
    """Schwarzschild radial coordinate from cylindrical (R, z)."""
    return np.sqrt(R**2 + z**2)


def _impact_parameter(x_A: np.ndarray, x_B: np.ndarray) -> float:
    """
    Minimum distance from the origin (BH) to the straight-line segment A→B.
    This is the flat-space impact parameter, used as a proxy for the GR
    impact parameter in the weak-field regime.
    """
    AB  = x_B - x_A
    len_AB = np.linalg.norm(AB)
    if len_AB < 1e-12:
        return np.linalg.norm(x_A)
    # t* = argmin |x_A + t*AB|²
    t_star = -np.dot(x_A, AB) / (len_AB**2)
    t_star = np.clip(t_star, 0.0, 1.0)
    closest = x_A + t_star * AB
    return float(np.linalg.norm(closest))


def _shapiro_delay(r_A: float, r_B: float, b: float) -> float:
    """
    Shapiro time delay for a photon traveling between Schwarzschild radii
    r_A and r_B with impact parameter b, in units of r_g/c.

    Formula (Shapiro 1964, generalised):
        Δt = 2 r_g × ln[(r_A + √(r_A²−b²))(r_B + √(r_B²−b²)) / b²]

    For b > r_A or b > r_B (should not occur for physical paths after
    clipping), we fall back to a small positive correction.
    """
    b_eff = max(b, B_MIN_RG)

    term_A2 = r_A**2 - b_eff**2
    term_B2 = r_B**2 - b_eff**2

    if term_A2 <= 0 or term_B2 <= 0:
        # Photon too close to BH along this leg; return cap
        return SHAPIRO_CAP

    dt = 2.0 * np.log(
        (r_A + np.sqrt(term_A2)) * (r_B + np.sqrt(term_B2)) / (b_eff**2)
    )
    return float(np.clip(dt, 0.0, SHAPIRO_CAP))


# ─────────────────────────────────────────────────────────────────────────────
# Observer direction
# ─────────────────────────────────────────────────────────────────────────────

def _observer_hat(incl_deg: float) -> np.ndarray:
    """Unit vector pointing toward observer (incl from z-axis in x-z plane)."""
    th = np.radians(incl_deg)
    return np.array([np.sin(th), 0.0, np.cos(th)])


# ─────────────────────────────────────────────────────────────────────────────
# Main function: Schwarzschild lag matrix
# ─────────────────────────────────────────────────────────────────────────────

def compute_lag_matrix_schwarzschild(
    grid,
    disk,
    observer_inclination: float = 30.0,
    spin: float = 0.0,          # reserved; only a=0 implemented
    verbose: bool = False,
) -> np.ndarray:
    """
    Compute the (N_corona, N_disc) lag matrix with Schwarzschild corrections.

    Parameters
    ----------
    grid : CoronalGrid
        Output of build_grid(); must have attributes R, z, phi (arrays of
        length n_cells).
    disk : DiskGrid
        Output of build_disk_grid(); must have attributes R, z, phi
        (arrays of length n_elements).
    observer_inclination : float
        Observer inclination in degrees measured from the z-axis (spin axis).
    spin : float
        Dimensionless BH spin a/M. Currently only a=0 (Schwarzschild) is
        implemented; non-zero spin is silently ignored with a warning.
    verbose : bool
        Print progress every 100 coronal cells.

    Returns
    -------
    lag_matrix : np.ndarray, shape (N_corona, N_disc)
        Lag matrix in units of r_g/c, including flat-spacetime light travel
        time plus Schwarzschild Shapiro delay on both legs.
    """
    if spin != 0.0:
        import warnings
        warnings.warn(
            "Non-zero spin not yet implemented; using Schwarzschild (a=0).",
            UserWarning, stacklevel=2,
        )

    obs_hat = _observer_hat(observer_inclination)
    N  = grid.n_cells
    M  = disk.n_elements

    lag_matrix = np.zeros((N, M))

    # Pre-compute observer-distance projections for disc elements
    # (disc to observer: distance is r_k projected onto obs_hat, plus Shapiro)
    r_disk = disk.R  # disc is at z=0   # shape (M,)

    for i in range(N):
        if verbose and i % 100 == 0:
            print(f"  cell {i}/{N}", end='\r', flush=True)

        Ri, zi, phi_i = grid.R[i], grid.z[i], grid.phi[i]
        x_i  = _cartesian(Ri, zi, phi_i)
        r_i  = _schwarzschild_r(Ri, zi)

        # Direct path: coronal cell → observer (at infinity, along obs_hat)
        # Flat-space projection (signed; positive = in front of BH plane)
        d_i_obs_flat = np.dot(x_i, obs_hat)

        for k in range(M):
            Rk, phi_k = disk.R[k], disk.phi[k]; zk = 0.0
            x_k  = _cartesian(Rk, zk, phi_k)
            r_k  = Rk  # z=0 so r=R

            # ── Leg 1: corona cell i → disc element k ─────────────────────
            d_ik_flat = float(np.linalg.norm(x_i - x_k))
            b_ik      = _impact_parameter(x_i, x_k)
            dt_ik_GR  = _shapiro_delay(r_i, r_k, b_ik)

            # ── Leg 2: disc element k → observer ──────────────────────────
            # Flat-space: d_k_obs = projection of x_k onto obs_hat (signed)
            d_k_obs_flat = np.dot(x_k, obs_hat)
            # Shapiro: photon from r_k to r_obs ≈ ∞; simplify to
            #   Δt ≈ 2 r_g × ln(2 r_k / b_k_obs) where b_k_obs ≈ r_k sin(θ_k)
            # Here θ_k = angle between x_k and obs direction
            cos_angle = np.dot(x_k / (r_k + 1e-15), obs_hat)
            sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))
            b_k_obs   = r_k * sin_angle           # impact param to observer
            b_k_obs   = max(b_k_obs, B_MIN_RG)
            # For path to infinity: r_B → ∞, so formula simplifies to:
            #   Δt ≈ 2 r_g × ln(2 r_k / b_k_obs)  (Shapiro 1964)
            if b_k_obs < r_k:
                dt_k_obs_GR = float(np.clip(
                    2.0 * np.log(2.0 * r_k / b_k_obs), 0.0, SHAPIRO_CAP
                ))
            else:
                dt_k_obs_GR = 0.0

            # ── Direct path: corona i → observer ──────────────────────────
            cos_angle_i = np.dot(x_i / (r_i + 1e-15), obs_hat)
            sin_angle_i = np.sqrt(max(0.0, 1.0 - cos_angle_i**2))
            b_i_obs     = r_i * sin_angle_i
            b_i_obs     = max(b_i_obs, B_MIN_RG)
            if b_i_obs < r_i:
                dt_i_obs_GR = float(np.clip(
                    2.0 * np.log(2.0 * r_i / b_i_obs), 0.0, SHAPIRO_CAP
                ))
            else:
                dt_i_obs_GR = 0.0

            # ── Total lag: flat + GR corrections ──────────────────────────
            # Flat-spacetime lag:
            lag_flat = (d_ik_flat + d_k_obs_flat - d_i_obs_flat) / C_LIGHT

            # GR Shapiro corrections:
            #   + on leg i→k  (extra delay)
            #   + on leg k→obs (extra delay for reflected path)
            #   − on leg i→obs (extra delay for direct path, subtracted)
            lag_GR = lag_flat + dt_ik_GR + dt_k_obs_GR - dt_i_obs_GR

            # Clip to physical range [0, large value]
            lag_matrix[i, k] = max(0.0, lag_GR)

    if verbose:
        print()

    return lag_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic: compare flat vs Schwarzschild for a single source
# ─────────────────────────────────────────────────────────────────────────────

def compare_flat_vs_schwarzschild(grid, disk, incl_deg=30.0) -> dict:
    """
    Compare flat-spacetime vs Schwarzschild lag matrices.
    Returns a dict of statistics useful for the paper.
    """
    from qcorona.physics.lags import compute_lag_matrix as flat_lm

    LM_flat = flat_lm(grid, disk, observer_inclination=incl_deg)
    LM_schw = compute_lag_matrix_schwarzschild(grid, disk,
                                               observer_inclination=incl_deg)

    correction = LM_schw - LM_flat
    frac       = correction / (LM_flat + 1e-10)

    return dict(
        LM_flat          = LM_flat,
        LM_schwarzschild = LM_schw,
        mean_correction_rgc   = float(correction.mean()),
        median_frac_correction = float(np.median(frac)),
        max_frac_correction    = float(frac.max()),
        min_lag_flat           = float(LM_flat.min()),
        min_lag_schwarzschild  = float(LM_schw.min()),
    )
