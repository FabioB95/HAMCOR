"""
polarization.py
───────────────
Diagnostico di polarizzazione post-hoc (Opzione C).

Calcola:
  - momenti del secondo ordine della distribuzione di emissività proiettata
  - eccentricità e Position Angle (PA) geometrici
  - fattore di calibrazione k  (PD = k * eccentricity)
  - termine di costo H_pol per confronto con dati IXPE
"""

from __future__ import annotations
import numpy as np
from typing import Dict


def compute_projected_moments(
    grid,
    emissivity: np.ndarray,
    observer_inclination: float = 30.0,
) -> Dict[str, float]:
    """
    Proietta la corona sul piano del cielo e calcola i momenti del 2° ordine.

    Parameters
    ----------
    grid : oggetto griglia (deve avere .x, .y, .z  oppure .R, .phi, .z)
    emissivity : array 1-D di pesi (somma = 1)
    observer_inclination : inclinazione in gradi

    Returns
    -------
    dict con chiavi:
        'eccentricity' : float   (0 = circolare, 1 = molto allungata)
        'PA_pred_deg'  : float   (Position Angle in gradi, range [-90, +90])
        'Q', 'U'       : momenti normalizzati (Stokes-like)
        'Ixx', 'Iyy', 'Ixy' : momenti grezzi
    """
    incl = np.deg2rad(observer_inclination)

    # Coordinate cartesiane della griglia
    if hasattr(grid, "x") and hasattr(grid, "y") and hasattr(grid, "z"):
        x = np.asarray(grid.x)
        y = np.asarray(grid.y)
        z = np.asarray(grid.z)
    else:
        # fallback: ricostruisci da R, phi, z
        R = np.asarray(grid.R)
        phi = np.asarray(grid.phi)
        z = np.asarray(grid.z)
        x = R * np.cos(phi)
        y = R * np.sin(phi)

    w = np.asarray(emissivity, dtype=float)
    w = w / w.sum()

    # Proiezione sul piano del cielo (asse y' lungo la linea dei nodi)
    # x_sky  ≈ x
    # y_sky  ≈ y * cos(i) + z * sin(i)
    x_sky = x
    y_sky = y * np.cos(incl) + z * np.sin(incl)

    # Momenti del secondo ordine centrati
    x0 = np.sum(w * x_sky)
    y0 = np.sum(w * y_sky)

    dx = x_sky - x0
    dy = y_sky - y0

    Ixx = np.sum(w * dx * dx)
    Iyy = np.sum(w * dy * dy)
    Ixy = np.sum(w * dx * dy)

    # Autovalori del tensore di inerzia 2-D
    # λ± = (Ixx + Iyy ± sqrt((Ixx-Iyy)² + 4 Ixy²)) / 2
    delta = np.sqrt((Ixx - Iyy) ** 2 + 4.0 * Ixy ** 2)
    lambda_max = 0.5 * (Ixx + Iyy + delta)
    lambda_min = 0.5 * (Ixx + Iyy - delta)

    # Eccentricità (ellitticità)
    if lambda_max > 1e-30:
        eccentricity = np.sqrt(1.0 - lambda_min / lambda_max)
    else:
        eccentricity = 0.0

    # Position Angle dell’asse maggiore
    # PA = ½ arctan(2 Ixy / (Ixx - Iyy))   (in radianti → gradi)
    if abs(Ixx - Iyy) < 1e-30 and abs(Ixy) < 1e-30:
        PA_rad = 0.0
    else:
        PA_rad = 0.5 * np.arctan2(2.0 * Ixy, Ixx - Iyy)

    PA_deg = np.rad2deg(PA_rad)          # già in [-90, +90]

    # Stokes-like (normalizzati)
    Q = (Ixx - Iyy) / (Ixx + Iyy + 1e-30)
    U = (2.0 * Ixy) / (Ixx + Iyy + 1e-30)

    return {
        "eccentricity": float(eccentricity),
        "PA_pred_deg": float(PA_deg),
        "Q": float(Q),
        "U": float(U),
        "Ixx": float(Ixx),
        "Iyy": float(Iyy),
        "Ixy": float(Ixy),
        "lambda_max": float(lambda_max),
        "lambda_min": float(lambda_min),
    }


def calibrate_k(eccentricity: float, PD_obs_pct: float) -> float:
    """
    Calibra il fattore k tale che  PD = k * eccentricity  riproduca PD_obs.

    Se eccentricity ≈ 0 restituisce 0 (non calibrabile).
    """
    if eccentricity < 1e-10:
        return 0.0
    return float(PD_obs_pct / eccentricity)


def H_pol(
    PA_pred_deg: float,
    PD_pred_pct: float,
    PA_obs_deg: float,
    PD_obs_pct: float,
    w_PA: float = 1.0,
    w_PD: float = 1.0,
) -> float:
    """
    Termine di costo polarimetrico (post-hoc).

    ΔPA è calcolato modulo 180° e ridotto nell’intervallo [-90, +90].
    """
    # differenza angolare minima modulo 180°
    dPA = ((PA_pred_deg - PA_obs_deg + 90.0) % 180.0) - 90.0
    dPD = PD_pred_pct - PD_obs_pct

    return float(w_PA * (dPA / 90.0) ** 2 + w_PD * (dPD / max(PD_obs_pct, 1.0)) ** 2)