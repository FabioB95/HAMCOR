"""
lag_energy_psi.py
─────────────────
Replaces the scalar Gaussian psi_obs with a multi-energy
lag-energy constraint, eliminating the single weakest assumption
in the HAMCOR framework.

PHYSICS:
  Different X-ray energy bands trace different emission components:
    - Soft (0.3–0.8 keV): Fe L, O edge, soft excess — inner disc
    - Medium (0.8–2.0 keV): transition region
    - Hard (2.0–4.0 keV): direct continuum reference

  The lag-energy spectrum τ(E) encodes how different energies lag
  behind the reference band, tracing the reverberation transfer
  function at multiple scales simultaneously.

  HAMCOR uses this as a set of simultaneous constraints:
    H_geo = α Σ_b [ψ_pred(t_b) − ψ_obs(t_b)]²

  where ψ_obs(t) is now a multi-peaked function derived from the
  full lag-energy spectrum rather than a single Gaussian.

USAGE:
    from lag_energy_psi import (
        compute_lag_energy_spectrum,
        build_psi_obs_from_lag_energy,
        plot_lag_energy_spectrum,
    )

    # Compute lags at multiple energies from light curves
    lag_energy = compute_lag_energy_spectrum(
        lc_dict,      # dict: {energy_label: (time, rate)}
        lc_ref,       # reference band (hard band) light curve
        freq_band=(fmin, fmax),
        seg_len=256, dt=100.0,
    )

    # Build psi_obs(t) encoding the full lag-energy structure
    psi_obs, t_bins = build_psi_obs_from_lag_energy(lag_energy, n_bins=80)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnergyBandLag:
    """Result for one energy band."""
    label:       str
    E_lo:        float   # keV
    E_hi:        float   # keV
    lag_s:       float   # s  (coherence-weighted mean in freq_band)
    lag_err_s:   float   # s
    lag_rg:      float   # r_g/c
    lag_err_rg:  float   # r_g/c
    coherence:   float   # mean coherence in freq_band
    n_seg:       int

@dataclass
class LagEnergySpectrum:
    """Full lag-energy spectrum."""
    bands:       list    # list of EnergyBandLag
    freq_band:   Tuple   # (fmin, fmax) Hz used for averaging
    r_g_c:       float   # s per r_g/c for this source
    source_name: str


# ─────────────────────────────────────────────────────────────────────────────
# Cross-spectrum computation (per energy band)
# ─────────────────────────────────────────────────────────────────────────────

def _cross_spectrum_segments(s_arr, h_arr, seg_len, dt):
    """Compute mean cross-spectrum via Vaughan+2003 method."""
    n_bins = len(s_arr)
    freq   = np.fft.rfftfreq(seg_len, d=dt)
    cross_sum = np.zeros(len(freq), dtype=complex)
    pow_s_sum = np.zeros(len(freq))
    pow_h_sum = np.zeros(len(freq))
    n_valid   = 0

    for i in range((n_bins - seg_len) // (seg_len // 2) + 1):
        i0 = i * (seg_len // 2)
        i1 = i0 + seg_len
        if i1 > n_bins:
            break
        s_seg = s_arr[i0:i1]
        h_seg = h_arr[i0:i1]
        if np.sum(s_seg > 0) < seg_len * 0.7:
            continue
        S = np.fft.rfft(s_seg - s_seg.mean())
        H = np.fft.rfft(h_seg - h_seg.mean())
        cross_sum += np.conj(H) * S
        pow_s_sum += np.abs(S)**2
        pow_h_sum += np.abs(H)**2
        n_valid   += 1

    if n_valid == 0:
        return None

    cross_mean = cross_sum / n_valid
    pow_s_mean = pow_s_sum / n_valid
    pow_h_mean = pow_h_sum / n_valid
    coherence  = np.abs(cross_mean)**2 / (pow_s_mean * pow_h_mean + 1e-30)

    freq = freq[1:]
    cross_mean = cross_mean[1:]
    coherence  = coherence[1:]

    phase   = np.angle(cross_mean)
    with np.errstate(divide='ignore', invalid='ignore'):
        lag     = phase / (2 * np.pi * freq)
        lag_err = (np.sqrt((1 - coherence) / (2 * coherence * n_valid + 1e-30))
                   / (2 * np.pi * freq))

    return freq, lag, lag_err, coherence, n_valid


# ─────────────────────────────────────────────────────────────────────────────
# Main: compute lag-energy spectrum from light curve dict
# ─────────────────────────────────────────────────────────────────────────────

def compute_lag_energy_spectrum(
    lc_dict:    Dict[str, Tuple],    # {'0.3-0.5': (rate_arr), '0.5-1.0': ...}
    lc_ref:     np.ndarray,          # reference band (hard) rate array
    energy_lo:  Dict[str, float],    # {'0.3-0.5': 0.3, ...}  keV
    energy_hi:  Dict[str, float],    # {'0.3-0.5': 0.5, ...}  keV
    freq_band:  Tuple[float, float], # (fmin, fmax) Hz
    r_g_c:      float,               # s per r_g/c
    seg_len:    int   = 256,
    dt:         float = 100.0,
    source_name:str   = "source",
) -> LagEnergySpectrum:
    """
    Compute lag at multiple energy bands relative to reference band.

    Parameters
    ----------
    lc_dict : dict
        Keys are band labels, values are rate arrays (same length as lc_ref).
    lc_ref : array
        Reference band light curve (hard band, 1.5–4 keV recommended).
    energy_lo/hi : dict
        Lower/upper energy bounds for each band in keV.
    freq_band : (fmin, fmax)
        Frequency range for coherence-weighted lag averaging [Hz].
    r_g_c : float
        Conversion from seconds to r_g/c for this source.
    """
    fmin, fmax = freq_band
    bands = []

    for label, rate in sorted(lc_dict.items(),
                               key=lambda x: energy_lo[x[0]]):
        result = _cross_spectrum_segments(rate, lc_ref, seg_len, dt)
        if result is None:
            print(f"  {label}: insufficient segments, skipping")
            continue

        freq, lag, lag_err, coh, n_valid = result

        # Coherence-weighted mean in freq_band
        mask = (freq >= fmin) & (freq <= fmax)
        if mask.sum() < 2 or coh[mask].sum() <= 0:
            print(f"  {label}: no valid frequencies in [{fmin:.3g}, {fmax:.3g}] Hz")
            continue

        w         = coh[mask]
        lag_mean  = np.average(lag[mask],     weights=w)
        lag_e     = (np.sqrt(np.average(lag_err[mask]**2, weights=w))
                     / np.sqrt(mask.sum()))
        coh_mean  = float(np.mean(w))

        band = EnergyBandLag(
            label       = label,
            E_lo        = energy_lo[label],
            E_hi        = energy_hi[label],
            lag_s       = lag_mean,
            lag_err_s   = lag_e,
            lag_rg      = lag_mean  / r_g_c,
            lag_err_rg  = lag_e     / r_g_c,
            coherence   = coh_mean,
            n_seg       = n_valid,
        )
        bands.append(band)
        print(f"  {label}: lag = {lag_mean:.1f} +/- {lag_e:.1f} s "
              f"= {band.lag_rg:.2f} +/- {band.lag_err_rg:.2f} r_g/c  "
              f"(coh={coh_mean:.3f})")

    return LagEnergySpectrum(
        bands=bands, freq_band=freq_band,
        r_g_c=r_g_c, source_name=source_name,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build psi_obs(t) from lag-energy spectrum
# ─────────────────────────────────────────────────────────────────────────────

def build_psi_obs_from_lag_energy(
    spectrum: LagEnergySpectrum,
    n_bins:   int   = 80,
    t_min:    Optional[float] = None,
    t_max:    Optional[float] = None,
    weights:  Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build ψ_obs(t) as a weighted sum of per-band Gaussians.

    Each energy band contributes a Gaussian peak at its measured lag,
    weighted by its coherence. This encodes the full lag-energy structure
    as a single transfer function constraint.

    Returns
    -------
    psi_obs : (n_bins,) array, normalised to sum=1
    t_bins  : (n_bins+1,) array of bin edges in r_g/c
    """
    bands = spectrum.bands
    if not bands:
        raise ValueError("No valid bands in spectrum")

    # Time axis: cover 3σ beyond the most extreme band
    lags = [abs(b.lag_rg) for b in bands]
    sigs = [b.lag_err_rg  for b in bands]

    if t_min is None:
        t_min = max(0.0, min(l - 3*s for l, s in zip(lags, sigs)))
    if t_max is None:
        t_max = max(l + 4*s for l, s in zip(lags, sigs)) * 1.2

    t_bins   = np.linspace(t_min, t_max, n_bins + 1)
    t_c      = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs  = np.zeros(n_bins)

    for band in bands:
        # Weight by coherence (or user-provided weights)
        w = (weights.get(band.label, 1.0) if weights else 1.0) * band.coherence
        # Gaussian centred on |lag| (absolute value: lag is negative for reverb)
        lag_abs = abs(band.lag_rg)
        sig     = max(band.lag_err_rg, (t_max - t_min) / n_bins)
        gauss   = np.exp(-0.5 * ((t_c - lag_abs) / sig)**2)
        psi_obs += w * gauss

    # Normalise
    norm = psi_obs.sum()
    if norm > 0:
        psi_obs /= norm
    else:
        raise ValueError("psi_obs is all zeros — check band lags")

    return psi_obs, t_bins


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: build from scalar lag measurements (already computed)
# ─────────────────────────────────────────────────────────────────────────────

def build_psi_obs_from_scalar_lags(
    lag_rg_list:     list,   # [(lag_rg, lag_err_rg, coherence), ...]
    label_list:      list,   # ['0.3-0.5 keV', ...]
    n_bins:          int   = 80,
    t_min:           float = None,
    t_max:           float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build multi-peak psi_obs directly from a list of (lag, err, coherence).
    Useful when lags are already computed and stored as scalars.
    """
    lags  = [abs(l[0]) for l in lag_rg_list]
    sigs  = [l[1]      for l in lag_rg_list]
    cohs  = [l[2]      for l in lag_rg_list]

    if t_min is None:
        t_min = max(0.0, min(l - 3*s for l, s in zip(lags, sigs)))
    if t_max is None:
        t_max = max(l + 5*s for l, s in zip(lags, sigs)) * 1.2

    t_bins  = np.linspace(t_min, t_max, n_bins + 1)
    t_c     = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs = np.zeros(n_bins)

    for lag, sig, coh, lbl in zip(lags, sigs, cohs, label_list):
        sig_eff = max(sig, (t_max - t_min) / n_bins)
        gauss   = np.exp(-0.5 * ((t_c - lag) / sig_eff)**2)
        psi_obs += coh * gauss

    norm = psi_obs.sum()
    if norm > 0:
        psi_obs /= norm

    return psi_obs, t_bins


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_lag_energy_spectrum(
    spectrum: LagEnergySpectrum,
    psi_obs:  Optional[np.ndarray] = None,
    t_bins:   Optional[np.ndarray] = None,
    save_path: str = None,
):
    """Plot lag-energy spectrum and the resulting psi_obs."""
    bands = spectrum.bands
    E_c   = [0.5 * (b.E_lo + b.E_hi) for b in bands]
    lags  = [b.lag_rg  for b in bands]
    errs  = [b.lag_err_rg for b in bands]

    n_panels = 2 if psi_obs is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 4.0))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: lag vs energy
    ax = axes[0]
    ax.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
    ax.errorbar(E_c, lags, yerr=errs,
                fmt='o', color='#2166ac', capsize=3, markersize=5)
    for band in bands:
        ax.axvspan(band.E_lo, band.E_hi, alpha=0.06, color='#2166ac')
    ax.set_xlabel('Energy [keV]')
    ax.set_ylabel(r'Lag $[r_g/c]$')
    ax.set_title(f'{spectrum.source_name} lag-energy spectrum')
    ax.set_xscale('log')

    # Panel 2: psi_obs
    if psi_obs is not None and t_bins is not None:
        ax2 = axes[1]
        t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
        ax2.fill_between(t_c, psi_obs, alpha=0.3, color='#d6604d')
        ax2.plot(t_c, psi_obs, color='#d6604d', lw=1.5)
        # Mark individual band contributions
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(bands)))
        for band, col in zip(bands, colors):
            lag_abs = abs(band.lag_rg)
            sig     = max(band.lag_err_rg, (t_bins[-1]-t_bins[0])/80)
            g = band.coherence * np.exp(-0.5*((t_c-lag_abs)/sig)**2)
            g /= (g.sum() + 1e-15)
            ax2.plot(t_c, g * band.coherence * 0.5,
                     color=col, lw=0.8, ls=':', alpha=0.7,
                     label=f'{band.E_lo:.1f}–{band.E_hi:.1f} keV')
        ax2.set_xlabel(r'$\Delta t\;[r_g/c]$')
        ax2.set_ylabel(r'$\psi_{\rm obs}(t)$')
        ax2.set_title('Multi-energy transfer function')
        ax2.legend(fontsize=6.5, loc='upper right', framealpha=0.8)
        ax2.set_ylim(bottom=0)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"Saved: {save_path}")
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Demo: apply to Mrk 335 using already-measured lags
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import os
    os.makedirs('results', exist_ok=True)

    # Mrk 335 lag-energy (these need to be measured per-band)
    # For now, use a self-consistent set based on the 1.46 r_g/c mean lag
    # and the expected energy dependence (soft bands lag more than hard)
    # These should be replaced with actual measured values once per-band
    # light curves are extracted from XMM SAS.
    print("Demo: Mrk 335 multi-energy psi_obs")
    print("(Using illustrative lag-energy values —")
    print(" replace with measured values from SAS extraction)")

    # Illustrative lag-energy spectrum for Mrk 335
    # Soft energies lag more (reflection dominated)
    # Hard energies lag less (continuum dominated)
    lag_energy_demo = [
        # (lag_rg, lag_err_rg, coherence, label)
        (2.8, 1.2, 0.65, '0.3-0.5 keV'),
        (2.2, 0.9, 0.72, '0.5-0.8 keV'),
        (1.5, 0.8, 0.70, '0.8-1.2 keV'),
        (0.9, 0.6, 0.55, '1.2-2.0 keV'),
    ]

    lags_list  = [(l[0], l[1], l[2]) for l in lag_energy_demo]
    label_list = [l[3] for l in lag_energy_demo]

    psi_obs, t_bins = build_psi_obs_from_scalar_lags(
        lags_list, label_list, n_bins=80,
    )

    t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
    print(f"\nMulti-energy psi_obs peak at: {t_c[psi_obs.argmax()]:.2f} r_g/c")
    print(f"Covers range: {t_bins[0]:.2f} -- {t_bins[-1]:.2f} r_g/c")
    print(f"\nTo use in HAMCOR fits, replace:")
    print("  psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)")
    print("with:")
    print("  from lag_energy_psi import build_psi_obs_from_scalar_lags")
    print("  psi_obs, t_bins = build_psi_obs_from_scalar_lags(lags_list, labels)")
