"""
check_weak_field_shapiro_accuracy_v2.py
Confronto corretto: stessa geometria "a V" per exact e weak-field.
Decompone l'integrale esatto in due tratti dal periapsi (raggio di minimo
avvicinamento) verso r_A e verso r_B, coerente con flat_euclidean e con
la formula weak-field standard.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import sys
sys.path.insert(0, '.')
from schwarzschild_lags import _shapiro_delay

def potential_term(r, b):
    f = 1.0 - 2.0/r
    return 1.0 - b**2 * f / r**2

def dt_dr_exact(r, b):
    f = 1.0 - 2.0/r
    inside = potential_term(r, b)
    if inside <= 1e-14:
        return np.inf
    return 1.0 / (f * np.sqrt(inside))

def find_periapsis(b, r_search_max=50.0):
    """Trova r_p tale che potential_term(r_p, b) = 0 (turning point)."""
    # cerca la radice piu' vicina a r=b (approssimazione flat) verso l'alto
    r_lo, r_hi = 2.0001, r_search_max
    f_lo, f_hi = potential_term(r_lo, b), potential_term(r_hi, b)
    if f_lo > 0:
        return r_lo  # nessun turning point sopra l'orizzonte (b troppo grande o piccolo)
    return brentq(potential_term, r_lo, r_hi, args=(b,))

def exact_V_time(r_A, r_B, b):
    """Tempo coordinato per il percorso A -> periapsi -> B (geometria a V)."""
    r_p = find_periapsis(b)
    # integrale con singolarita' integrabile al periapsi (1/sqrt vicino allo zero)
    t_A = quad(dt_dr_exact, r_p, r_A, args=(b,), limit=200, points=[r_p])[0]
    t_B = quad(dt_dr_exact, r_p, r_B, args=(b,), limit=200, points=[r_p])[0]
    return t_A + t_B, r_p

print(f"{'r_A':>6} {'r_B':>6} {'b':>6} {'r_p':>6} {'exact_shapiro':>13} {'weak_shapiro':>13} {'ratio':>7}")
print("-" * 70)

test_cases = [
    (5.0, 6.0, 3.0), (5.0, 6.0, 4.5), (5.0, 6.0, 1.0),
    (8.0, 10.0, 5.0), (8.0, 10.0, 7.0), (8.0, 10.0, 2.0),
    (3.0, 4.0, 2.0), (3.0, 4.0, 2.5), (10.0, 15.0, 8.0),
]

for r_A, r_B, b in test_cases:
    exact_total, r_p = exact_V_time(r_A, r_B, b)
    flat_euclidean = np.sqrt(r_A**2 - b**2) + np.sqrt(r_B**2 - b**2)
    exact_shapiro = exact_total - flat_euclidean
    weak_shapiro = _shapiro_delay(r_A, r_B, b)
    ratio = weak_shapiro / exact_shapiro if exact_shapiro != 0 else float('nan')
    print(f"{r_A:6.1f} {r_B:6.1f} {b:6.1f} {r_p:6.3f} {exact_shapiro:13.4f} "
          f"{weak_shapiro:13.4f} {ratio:7.3f}")