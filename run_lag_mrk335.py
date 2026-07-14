"""
run_lag_mrk335.py  —  Lag-frequency spectrum of Mrk 335.

Observation: XMM-Newton ObsID 0306870101 (2006 Jan 03)
Bands:       soft 0.3-1.0 keV vs hard 1.5-4.0 keV
Method:      Vaughan+2003 cross-spectrum, 256-bin segments, 50% overlap
Reference:   Kara et al. (2013b), MNRAS 428, 2795

Output:
    results/lag_frequency_v2_mrk335.pdf
    results/lag_frequency_v2_mrk335.tex
    data/sources/mrk335/lag_rev.npy
"""

import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family':       'serif',
    'font.size':         9,
    'axes.labelsize':    9,
    'axes.titlesize':    9,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.linewidth':    0.6,
    'lines.linewidth':   1.0,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'mathtext.fontset':  'dejavusans',
})

# ── source parameters ─────────────────────────────────────────────────────────
M_BH   = 1.3e7   # M_sun (Peterson+2004 reverberation mapping)
R_G_C  = 63.0    # s per r_g/c  (= GM/c^3 for M=1.3e7 Msun)
dt     = 100.0   # s  (light curve bin size)

# Reverberation band from Kara+2013b
REV_FMIN = 2e-4  # Hz  (0.2 mHz)
REV_FMAX = 7e-4  # Hz  (0.7 mHz)

# ── load light curves ─────────────────────────────────────────────────────────
DATA_DIR = 'data/sources/mrk335'

try:
    s = np.load(f'{DATA_DIR}/soft_lc.npy')
    h = np.load(f'{DATA_DIR}/hard_lc.npy')
    t = np.load(f'{DATA_DIR}/time.npy')
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Expected files in data/sources/mrk335/:")
    print("  soft_lc.npy   — 0.3–1.0 keV count rate (100 s bins)")
    print("  hard_lc.npy   — 1.5–4.0 keV count rate (100 s bins)")
    print("  time.npy      — time axis (s)")
    sys.exit(1)

print(f"Mrk 335 — ObsID 0306870101 (2006)")
print(f"  Bins:      {len(s)}")
print(f"  Duration:  {(t[-1]-t[0])/1000:.1f} ks")
print(f"  Soft mean: {s.mean():.2f} ct/s")
print(f"  Hard mean: {h.mean():.2f} ct/s")

# ── Vaughan+2003 cross-spectrum ───────────────────────────────────────────────
seg_len = 256       # bins = 25.6 ks
n_bins  = len(s)
n_seg   = (n_bins - seg_len) // (seg_len // 2) + 1

freq      = np.fft.rfftfreq(seg_len, d=dt)
cross_sum = np.zeros(len(freq), dtype=complex)
pow_s_sum = np.zeros(len(freq))
pow_h_sum = np.zeros(len(freq))
n_valid   = 0

for i in range(n_seg):
    i0 = i * (seg_len // 2)
    i1 = i0 + seg_len
    if i1 > n_bins:
        break
    s_seg = s[i0:i1]
    h_seg = h[i0:i1]
    # Require at least 80% non-zero bins
    if np.sum(s_seg > 0) < seg_len * 0.8:
        continue
    S = np.fft.rfft(s_seg - s_seg.mean())
    H = np.fft.rfft(h_seg - h_seg.mean())
    cross_sum += np.conj(H) * S
    pow_s_sum += np.abs(S)**2
    pow_h_sum += np.abs(H)**2
    n_valid   += 1

print(f"  Segments:  {n_valid} valid (of {n_seg} total)")

cross_mean = cross_sum / n_valid
pow_s_mean = pow_s_sum / n_valid
pow_h_mean = pow_h_sum / n_valid
coherence  = np.abs(cross_mean)**2 / (pow_s_mean * pow_h_mean + 1e-20)

# Remove DC term
freq       = freq[1:]
cross_mean = cross_mean[1:]
coherence  = coherence[1:]

phase   = np.angle(cross_mean)
with np.errstate(divide='ignore', invalid='ignore'):
    lag     = phase / (2 * np.pi * freq)
    lag_err = (np.sqrt((1 - coherence) / (2 * coherence * n_valid + 1e-20))
               / (2 * np.pi * freq))

# ── log-frequency binning ─────────────────────────────────────────────────────
f_min, f_max = 7e-5, 5e-3
n_fbins  = 12
f_edges  = np.logspace(np.log10(f_min), np.log10(f_max), n_fbins + 1)
f_c      = np.sqrt(f_edges[:-1] * f_edges[1:])

lag_b = np.full(n_fbins, np.nan)
err_b = np.full(n_fbins, np.nan)
coh_b = np.full(n_fbins, np.nan)

for i in range(n_fbins):
    mask = (freq >= f_edges[i]) & (freq < f_edges[i+1])
    if mask.sum() < 2:
        continue
    w = coherence[mask]
    if w.sum() <= 0:
        continue
    lag_b[i] = np.average(lag[mask],     weights=w)
    err_b[i] = (np.sqrt(np.average(lag_err[mask]**2, weights=w))
                / np.sqrt(mask.sum()))
    coh_b[i] = np.mean(w)

print(f"\n  Lag-frequency spectrum:")
print(f"  {'f [mHz]':>10} {'lag [s]':>10} {'err [s]':>10} {'coh':>8}")
print("  " + "─"*42)
for i in range(n_fbins):
    if not np.isnan(lag_b[i]):
        print(f"  {f_c[i]*1e3:10.3f} {lag_b[i]:10.1f} "
              f"{err_b[i]:10.1f} {coh_b[i]:8.3f}")

# ── reverberation lag (coherence-weighted mean in 0.2–0.7 mHz) ───────────────
rev_mask = (freq >= REV_FMIN) & (freq <= REV_FMAX)
w_rev    = coherence[rev_mask]

if w_rev.sum() > 0:
    lag_rev = np.average(lag[rev_mask],     weights=w_rev)
    err_rev = (np.sqrt(np.average(lag_err[rev_mask]**2, weights=w_rev))
               / np.sqrt(rev_mask.sum()))
else:
    lag_rev = np.nan
    err_rev = np.nan

lag_rg  = abs(lag_rev) / R_G_C if not np.isnan(lag_rev) else np.nan
err_rg  = err_rev      / R_G_C if not np.isnan(err_rev) else np.nan

print(f"\n  Reverberation band (0.2–0.7 mHz):")
print(f"    lag = {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"    lag = {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")

# ── save numerical results ────────────────────────────────────────────────────
os.makedirs('results',   exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

if not np.isnan(lag_rev):
    np.save(f'{DATA_DIR}/lag_rev.npy',
            np.array([lag_rev, err_rev, lag_rg, err_rg]))
    np.save(f'{DATA_DIR}/lag_freq_centers.npy', f_c)
    np.save(f'{DATA_DIR}/lag_freq_values.npy',  lag_b)
    np.save(f'{DATA_DIR}/lag_freq_errors.npy',  err_b)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(5.5, 5.0),
    gridspec_kw={'height_ratios': [3, 1.2]},
    sharex=True,
)

valid = ~np.isnan(lag_b)

# Top panel: lag vs frequency
ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax1.axvspan(REV_FMIN * 1e3, REV_FMAX * 1e3,
            alpha=0.12, color='#d6604d', label='Reverberation band')
ax1.errorbar(
    f_c[valid] * 1e3, lag_b[valid], yerr=err_b[valid],
    fmt='o', color='#2166ac', capsize=3, markersize=4.5,
    label='Lag (soft$-$hard)',
)
ax1.set_ylabel(r'Time lag [s]  (soft $-$ hard)')
ax1.set_title('Mrk~335 — EPIC-pn, ObsID~0306870101 (2006)')
ax1.legend(fontsize=7.5, framealpha=0.8)
ax1.set_xscale('log')
ax1.yaxis.grid(True, lw=0.4, alpha=0.4)

# Annotate the measured lag
if not np.isnan(lag_rev):
    # Find the x position near the reverberation band centre
    f_ann = np.sqrt(REV_FMIN * REV_FMAX) * 1e3  # mHz
    ax1.annotate(
        fr'$\Delta t = {lag_rev:.0f} \pm {err_rev:.0f}$ s',
        xy=(f_ann, lag_rev),
        xytext=(f_ann * 3, lag_rev - abs(lag_rev) * 0.5),
        fontsize=7.5, color='#d6604d',
        arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8),
    )

# Bottom panel: coherence
ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax2.axvspan(REV_FMIN * 1e3, REV_FMAX * 1e3, alpha=0.12, color='#d6604d')
ax2.plot(f_c[valid] * 1e3, coh_b[valid],
         'o-', color='#4dac26', markersize=4, lw=0.8)
ax2.set_ylabel('Coherence')
ax2.set_xlabel('Frequency [mHz]')
ax2.set_ylim(0, 1)
ax2.yaxis.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()

pdf_path = 'results/lag_frequency_v2_mrk335.pdf'
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\n  Saved: {pdf_path}")

# ── pgfplots LaTeX ────────────────────────────────────────────────────────────
def cerr(xs, ys, es):
    pts = []
    for x, y, e in zip(xs, ys, es):
        if not np.isnan(y):
            pts.append(f'({x:.4f},{y:.2f}) +- (0,{abs(e):.2f})')
    return ' '.join(pts)

tex = [
    r'% Mrk 335 lag-frequency spectrum — pgfplots source',
    r'% Generated by run_lag_mrk335.py',
    r'\begin{tikzpicture}',
    r'\begin{axis}[',
    r'  width=8cm, height=6cm,',
    r'  xmode=log,',
    r'  xlabel={Frequency [mHz]},',
    r'  ylabel={Time lag [s]},',
    r'  title={Mrk~335 lag-frequency spectrum (EPIC-pn, 2006)},',
    r'  tick label style={font=\scriptsize},',
    r'  label style={font=\scriptsize},',
    r'  title style={font=\small},',
    r'  axis line style={thin},',
    r'  ymajorgrids=true, grid style={dotted,thin},',
    r']',
    r'\addplot[dashed,black,thin,domain=0.07:5] {0};',
    r'\addplot[fill=red!15,draw=none] coordinates {',
    f'  ({REV_FMIN*1e3:.3f},-1000)({REV_FMAX*1e3:.3f},-1000)'
    f'({REV_FMAX*1e3:.3f},1000)({REV_FMIN*1e3:.3f},1000)',
    r'} \closedcycle;',
    r'\addplot[',
    r'  only marks, mark=*, mark size=1.5pt,',
    r'  blue!70!black,',
    r'  error bars/.cd, y dir=both, y explicit,',
    r'] coordinates {',
    '  ' + cerr(f_c[valid]*1e3, lag_b[valid], err_b[valid]),
    r'};',
]
if not np.isnan(lag_rev):
    tex.append(
        r'\node[red!70!black,font=\scriptsize] at '
        f'(axis cs:{np.sqrt(REV_FMIN*REV_FMAX)*1e3*3:.3f},{lag_rev*1.5:.0f}) '
        r'{$\Delta t=' + f'{lag_rev:.0f}' + r'\,\mathrm{s}$};'
    )
tex += [r'\end{axis}', r'\end{tikzpicture}']

tex_path = 'results/lag_frequency_v2_mrk335.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex))
print(f"  Saved: {tex_path}")

print(f"\n  SUMMARY — Mrk 335")
print(f"  Lag (0.2–0.7 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"  In r_g/c:          {lag_rg:.2f} +/- {err_rg:.2f}")
