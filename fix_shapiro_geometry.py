"""fix_shapiro_geometry.py — sostituisce _shapiro_delay in schwarzschild_lags.py
con versione che sceglie sum (V, periapsi tra A e B) o diff (monotona) in base
alla geometria reale, usando il t_star NON clippato gia' calcolato altrove."""
import numpy as np

def _shapiro_delay_v2(r_A, r_B, b, has_periapsis_between):
    """
    has_periapsis_between: True se il punto di minimo avvicinamento
    (t_star in [0,1] non clippato) cade tra A e B -> formula sum (V).
    False -> formula diff (monotona), il caso tipico corona->disco.
    """
    from schwarzschild_lags import B_MIN_RG, SHAPIRO_CAP
    b_eff = max(b, B_MIN_RG)
    term_A2 = max(0.0, r_A**2 - b_eff**2)
    term_B2 = max(0.0, r_B**2 - b_eff**2)
    log_A = np.log(r_A + np.sqrt(term_A2))
    log_B = np.log(r_B + np.sqrt(term_B2))

    if has_periapsis_between:
        dt = 2.0 * (log_A + log_B - 2.0*np.log(b_eff))   # formula originale (V)
    else:
        dt = 2.0 * abs(log_B - log_A)                     # NUOVA: monotona

    return float(np.clip(dt, 0.0, SHAPIRO_CAP))


def impact_parameter_with_geometry(x_A, x_B):
    """Come _impact_parameter ma ritorna anche se il periapsi cade tra A e B
    (t_star grezzo, NON clippato a [0,1])."""
    AB = x_B - x_A
    len_AB = np.linalg.norm(AB)
    if len_AB < 1e-12:
        return np.linalg.norm(x_A), False
    t_star_raw = -np.dot(x_A, AB) / (len_AB**2)
    has_periapsis = 0.0 <= t_star_raw <= 1.0
    t_star_clipped = np.clip(t_star_raw, 0.0, 1.0)
    closest = x_A + t_star_clipped * AB
    b = float(np.linalg.norm(closest))
    return b, has_periapsis