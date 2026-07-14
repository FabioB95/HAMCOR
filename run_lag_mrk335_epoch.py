"""
run_lag_mrk335_epoch.py
───────────────────────
Measures the reverberation lag for any Mrk 335 XMM epoch.

Usage:
    python run_lag_mrk335_epoch.py --obsid 0600540601 --year 2009
    python run_lag_mrk335_epoch.py --obsid 0741280201 --year 2015
    python run_lag_mrk335_epoch.py --obsid 0780500301 --year 2018
    python run_lag_mrk335_epoch.py --obsid 0831790601 --year 2019

Expects in data/sources/mrk335_YEAR/:
    soft_lc.npy   (0.3-1.0 keV, 100s bins)
    hard_lc.npy   (1.5-4.0 keV, 100s bins)
    time.npy      (time axis in seconds)

Output:
    results/lag_frequency_mrk335_YEAR.pdf
    data/sources/mrk335_YEAR/lag_rev.npy
    (prints the lag value to paste into run_mrk335_multiepoch.py)
"""

import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans',
})

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--obsid', required=True, help='XMM ObsID')
parser.add_argument('--year',  required=True, type=int, help='Observation year')
args = parser.parse_args()

OBSID = args.obsid
YEAR  = args.year

# Reverberation band (same as 2006 Mrk 335 analysis)
REV_FMIN = 2e-4   # Hz  (0.2 mHz)
REV_FMAX = 7e-4   # Hz  (0.7 mHz)

M_BH  = 1.3e7     # M_sun
R_G_C = 63.0      # s per r_g/c
dt    = 100.0      # s

# ── load ──────────────────────────────────────────────────────────────────────
if YEAR == 2006:
    DATA_DIR = 'data/sources/mrk335'
else:
    DATA_DIR = f'data/sources/mrk335_{YEAR}'

try:
    s = np.load(f'{DATA_DIR}/soft_lc.npy')
    h = np.load(f'{DATA_DIR}/hard_lc.npy')
    t = np.load(f'{DATA_DIR}/time.npy')
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print(f"Make sure SAS reduction is complete and .npy files exist in {DATA_DIR}/")
    sys.exit(1)

print(f"Mrk 335 — ObsID {OBSID} ({YEAR})")
print(f"  Bins: {len(s)}  |  Duration: {(t[-1]-t[0])/1000:.1f} ks")
print(f"  Soft: {s.mean():.2f} ct/s  |  Hard: {h.mean():.2f} ct/s")

# ── cross-spectrum (Vaughan+2003) ─────────────────────────────────────────────
seg_len = 256
n_bins  = len(s)

# For short observations, use smaller segments
if n_bins < 512:
    seg_len = 128
    print(f"  Short observation — using {seg_len}-bin segments")

freq      = np.fft.rfftfreq(seg_len, d=dt)
cross_sum = np.zeros(len(freq), dtype=complex)
pow_s_sum = np.zeros(len(freq))
pow_h_sum = np.zeros(len(freq))
n_valid   = 0

for i in range((n_bins - seg_len) // (seg_len // 2) + 1):
    i0 = i * (seg_len // 2); i1 = i0 + seg_len
    if i1 > n_bins: break
    s_seg = s[i0:i1]; h_seg = h[i0:i1]
    if np.sum(s_seg > 0) < seg_len * 0.8: continue
    S = np.fft.rfft(s_seg - s_seg.mean())
    H = np.fft.rfft(h_seg - h_seg.mean())
    cross_sum += np.conj(H) * S
    pow_s_sum += np.abs(S)**2
    pow_h_sum += np.abs(H)**2
    n_valid   += 1

print(f"  Segments: {n_valid} valid")

if n_valid < 3:
    print("  WARNING: too few segments for reliable cross-spectrum")

cross_mean = cross_sum / n_valid
pow_s_mean = pow_s_sum / n_valid
pow_h_mean = pow_h_sum / n_valid
coherence  = np.abs(cross_mean)**2 / (pow_s_mean * pow_h_mean + 1e-20)

freq       = freq[1:]; cross_mean = cross_mean[1:]; coherence = coherence[1:]
phase      = np.angle(cross_mean)
with np.errstate(divide='ignore', invalid='ignore'):
    lag     = phase / (2 * np.pi * freq)
    lag_err = (np.sqrt((1 - coherence) / (2 * coherence * n_valid + 1e-20))
               / (2 * np.pi * freq))

# ── log-frequency binning ─────────────────────────────────────────────────────
f_edges = np.logspace(np.log10(7e-5), np.log10(5e-3), 13)
f_c     = np.sqrt(f_edges[:-1] * f_edges[1:])
n_fb    = len(f_c)

lag_b = np.full(n_fb, np.nan)
err_b = np.full(n_fb, np.nan)
coh_b = np.full(n_fb, np.nan)

for i in range(n_fb):
    mask = (freq >= f_edges[i]) & (freq < f_edges[i+1])
    if mask.sum() < 2: continue
    w = coherence[mask]
    if w.sum() <= 0: continue
    lag_b[i] = np.average(lag[mask], weights=w)
    err_b[i] = np.sqrt(np.average(lag_err[mask]**2, weights=w)) / np.sqrt(mask.sum())
    coh_b[i] = np.mean(w)

# ── reverberation lag ─────────────────────────────────────────────────────────
rev = (freq >= REV_FMIN) & (freq <= REV_FMAX)
w_r = coherence[rev]

if w_r.sum() > 0:
    lag_rev = np.average(lag[rev], weights=w_r)
    err_rev = np.sqrt(np.average(lag_err[rev]**2, weights=w_r)) / np.sqrt(rev.sum())
else:
    lag_rev = np.nan; err_rev = np.nan
    print("  WARNING: no valid frequencies in reverberation band")

lag_rg = abs(lag_rev) / R_G_C if not np.isnan(lag_rev) else np.nan
err_rg = err_rev      / R_G_C if not np.isnan(err_rev) else np.nan

print(f"\n  Reverberation band (0.2-0.7 mHz):")
print(f"    lag = {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"    lag = {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0),
                                gridspec_kw={'height_ratios': [3, 1.2]},
                                sharex=True)
valid = ~np.isnan(lag_b)

ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax1.axvspan(REV_FMIN*1e3, REV_FMAX*1e3, alpha=0.12, color='#d6604d',
            label='Reverberation band')
ax1.errorbar(f_c[valid]*1e3, lag_b[valid], yerr=err_b[valid],
             fmt='o', color='#2166ac', capsize=3, markersize=4.5,
             label='Lag (soft$-$hard)')
ax1.set_ylabel(r'Time lag [s]  (soft $-$ hard)')
ax1.set_title(f'Mrk~335 — ObsID~{OBSID} ({YEAR})')
ax1.legend(fontsize=7.5, framealpha=0.8)
ax1.set_xscale('log'); ax1.yaxis.grid(True, lw=0.4, alpha=0.4)

if not np.isnan(lag_rev):
    f_ann = np.sqrt(REV_FMIN * REV_FMAX) * 1e3
    ax1.annotate(
        fr'$\Delta t = {lag_rev:.0f} \pm {err_rev:.0f}$ s',
        xy=(f_ann, lag_rev),
        xytext=(f_ann * 3, lag_rev - abs(lag_rev) * 0.5),
        fontsize=7.5, color='#d6604d',
        arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8),
    )

ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax2.axvspan(REV_FMIN*1e3, REV_FMAX*1e3, alpha=0.12, color='#d6604d')
ax2.plot(f_c[valid]*1e3, coh_b[valid], 'o-', color='#4dac26', ms=4, lw=0.8)
ax2.set_ylabel('Coherence'); ax2.set_xlabel('Frequency [mHz]')
ax2.set_ylim(0, 1); ax2.yaxis.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
pdf = f'results/lag_frequency_mrk335_{YEAR}.pdf'
fig.savefig(pdf, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n  Saved: {pdf}")

# ── save and print copy-paste line ────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
if not np.isnan(lag_rev):
    np.save(f'{DATA_DIR}/lag_rev.npy',
            np.array([lag_rev, err_rev, lag_rg, err_rg]))

print()
print("=" * 60)
print(f"SUMMARY — Mrk 335  ObsID {OBSID}  ({YEAR})")
print("=" * 60)
print(f"  Lag: {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"  Lag: {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")
print()
print("Paste into run_mrk335_multiepoch.py EPOCHS dict:")
print(f"  'lag_s': {lag_rev:.1f},  'lag_err_s': {err_rev:.1f},")
print("=" * 60)
