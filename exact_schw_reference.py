"""
exact_schw_reference.py — integratore Schwarzschild esatto indipendente,
usato come verita' di riferimento (non weak-field, non formula periapsi
a mano). Riduce il moto al piano della geodetica (simmetria sferica),
integra r(lambda) con flip di segno automatico agli zeri di R(r).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

def R_pot(r, b):
    return 1.0 - b**2 * (1.0 - 2.0/r) / r**2

def _rhs(lam, y, b, sr):
    r = y[0]
    val = max(R_pot(r, b), 0.0)
    dr = sr * np.sqrt(val)
    dt = 1.0 / (1.0 - 2.0/r)
    dphi = b / r**2
    return [dr, dt, dphi]

def _ev_turn(lam, y, b, sr):
    return R_pot(y[0], b)
_ev_turn.terminal = True
_ev_turn.direction = -1

def exact_time_given_b(r_A, r_B, b, sr0=-1, max_step=0.01, r_horizon_margin=2.05):
    """Tempo coordinato esatto da r_A a r_B con impact parameter b, gestendo
    automaticamente un eventuale turning point (nessuna formula a mano)."""
    y = [r_A, 0.0, 0.0]
    sr = sr0
    for _ in range(5):
        sol = solve_ivp(_rhs, [0, 1e4], y, args=(b, sr), events=_ev_turn,
                         max_step=max_step, rtol=1e-10, atol=1e-12,
                         dense_output=True)
        # cerca l'attraversamento di r=r_B lungo la soluzione
        r_arr = sol.y[0]
        hit = np.where((r_arr - r_B)[:-1] * (r_arr - r_B)[1:] <= 0)[0]
        if len(hit) > 0:
            j = hit[0]
            f = lambda s: sol.sol(s)[0] - r_B
            s_hit = brentq(f, sol.t[j], sol.t[j+1])
            return float(sol.sol(s_hit)[1])   # tempo coordinato
        if len(sol.t_events[0]) == 0:
            return None   # non raggiunto, orizzonte o r_max
        y = sol.y[:, -1].tolist()
        sr = -sr
    return None

def find_b_from_geometry(r_A, r_B, b_geom):
    """b geometrico piatto e' un buon guess iniziale; qui lo usiamo diretto
    (il vero b della geodetica va trovato per shooting sul target r_B, ma
    per la VALIDAZIONE vogliamo il tempo AL b geometrico dato, non shooting)."""
    return b_geom