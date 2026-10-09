"""
polarization_term.py
──────────────────────
H_pol come vincolo attivo, con gradiente analitico completo rispetto a w_i.

Versione autosufficiente: non importa nulla da polarization.py, per evitare
dipendenze fragili tra file. La proiezione sul piano del cielo e' reimplementata
qui identica (stessa convenzione: e1=(0,1,0), e2=obs_hat x e1).
"""

import numpy as np

_EPS_DISC = 1e-10


def _sky_plane_axes(observer_inclination_deg):
    th = np.radians(observer_inclination_deg)
    obs_hat = np.array([np.sin(th), 0.0, np.cos(th)])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(obs_hat, e1)
    e2 = e2 / np.linalg.norm(e2)
    return e1, e2


def compute_H_pol(grid, w, observer_inclination, PA_obs_deg, PD_obs_percent,
                   k_calib, delta_pol=1.0, w_PA=1.0):
    """
    H_pol = delta * [(PD_pred - PD_obs)^2 + w_PA * sin^2(PA_pred - PA_obs)]
    PD_pred = k_calib * eccentricity  (k_calib fisso, calibrato una volta sola)

    Ritorna (H_pol, PA_pred_deg, PD_pred, grad_w).
    """
    e1, e2 = _sky_plane_axes(observer_inclination)
    coords = grid.get_cartesian()
    u = coords @ e1
    v = coords @ e2

    W = np.sum(w)
    ubar = np.sum(w * u) / W
    vbar = np.sum(w * v) / W
    du = u - ubar
    dv = v - vbar

    Suu = np.sum(w * du**2) / W
    Svv = np.sum(w * dv**2) / W
    Suv = np.sum(w * du * dv) / W

    trace = Suu + Svv
    diff  = Suu - Svv
    disc  = np.sqrt(diff**2 + 4*Suv**2 + 1e-30)
    disc_safe = max(float(disc), _EPS_DISC)

    eccentricity = disc / (trace + 1e-30)
    PA_pred_rad  = 0.5*np.arctan2(2*Suv, diff)
    PA_pred_deg  = np.degrees(PA_pred_rad)
    PD_pred      = k_calib * eccentricity

    dSuu_dw = (du**2 - Suu) / W
    dSvv_dw = (dv**2 - Svv) / W
    dSuv_dw = (du*dv - Suv) / W

    dtrace_dw = dSuu_dw + dSvv_dw
    ddiff_dw  = dSuu_dw - dSvv_dw
    ddisc_dw  = (diff*ddiff_dw + 4*Suv*dSuv_dw) / disc_safe

    de_dw  = (ddisc_dw*trace - disc*dtrace_dw) / (trace**2 + 1e-30)
    dPA_dw = (diff*dSuv_dw - Suv*ddiff_dw) / (disc_safe**2)

    dPD_dw = k_calib * de_dw

    dPA_rad = np.radians(PA_pred_deg - PA_obs_deg)
    dH_dPD  = 2.0 * (PD_pred - PD_obs_percent)
    dH_dPA  = w_PA * np.sin(2.0 * dPA_rad)

    grad_w = delta_pol * (dH_dPD * dPD_dw + dH_dPA * dPA_dw)

    H_pol = delta_pol * ((PD_pred - PD_obs_percent)**2 + w_PA*np.sin(dPA_rad)**2)

    return H_pol, PA_pred_deg, PD_pred, grad_w