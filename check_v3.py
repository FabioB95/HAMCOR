"""check_v4.py — flat coerente con la geometria effettiva (monotona=differenza,
periapsi=somma), non piu' assunta a priori."""
import sys; sys.path.insert(0, '.')
import numpy as np
from exact_schw_reference import exact_time_given_b
from schwarzschild_lags import _shapiro_delay

cases = [(5.0,6.0,3.0), (5.0,6.0,4.5), (5.0,6.0,1.0),
         (8.0,10.0,5.0), (8.0,10.0,2.0), (3.0,4.0,2.0), (10.0,15.0,8.0)]

print(f"{'r_A':>5}{'r_B':>5}{'b':>5}{'exact_t':>10}{'flat_diff':>10}{'exact_shap':>11}{'weak_shap':>10}{'ratio':>7}")
for r_A,r_B,b in cases:
    t_exact = exact_time_given_b(r_A, r_B, b, sr0=+1) or exact_time_given_b(r_A, r_B, b, sr0=-1)
    flat_diff = abs(np.sqrt(max(r_A**2-b**2,0)) - np.sqrt(max(r_B**2-b**2,0)))
    if t_exact is None:
        print(f"{r_A:5.1f}{r_B:5.1f}{b:5.1f}  non raggiunto"); continue
    exact_shap = t_exact - flat_diff
    weak = _shapiro_delay(r_A, r_B, b)
    print(f"{r_A:5.1f}{r_B:5.1f}{b:5.1f}{t_exact:10.4f}{flat_diff:10.4f}{exact_shap:11.4f}{weak:10.4f}{weak/exact_shap:7.3f}")