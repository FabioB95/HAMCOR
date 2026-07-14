"""
run_lag_1h0707.py  —  Lag-frequency spectrum di 1H 0707-495.

Osservazione: XMM-Newton ObsID 0511580101 (2008, gennaio)
Bande: soft 0.3-1.0 keV vs hard 1.5-4.0 keV
Metodo: Vaughan+2003 con segmenti sovrapposti

Output:
    results/lag_frequency_1h0707.png
    results/lag_frequency_1h0707.tex
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

# ── carica dati ───────────────────────────────────────────────────────────────
DATA_DIR = 'data/sources/1h0707'
s = np.load(f'{DATA_DIR}/soft_lc.npy')
h = np.load(f'{DATA_DIR}/hard_lc.npy')
t = np.load(f'{DATA_DIR}/time.npy')
dt = 100.0  # s

print(f"1H 0707-495 — ObsID 0511580101 (2008)")
print(f"Bin validi: {len(s)}")
print(f"Durata: {(t[-1]-t[0])/1000:.1f} ks")
print(f"Soft mean: {s.mean():.2f} cts/s")
print(f"Hard mean: {h.mean():.2f} cts/s")

# ── Vaughan+2003 cross-spectrum ───────────────────────────────────────────────
seg_len = 256   # bin (~25.6 ks)
n_seg   = (len(s) - seg_len) // (seg_len // 2) + 1

freq      = np.fft.rfftfreq(seg_len, d=dt)
cross_sum = np.zeros(len(freq), dtype=complex)
pow_s_sum = np.zeros(len(freq))
pow_h_sum = np.zeros(len(freq))
n_valid   = 0

for i in range(n_seg):
    i0 = i * (seg_len // 2)
    i1 = i0 + seg_len
    if i1 > len(s):
        break
    s_seg = s[i0:i1]
    h_seg = h[i0:i1]
    if np.sum(s_seg > 0) < seg_len * 0.8:
        continue
    S = np.fft.rfft(s_seg - s_seg.mean())
    H = np.fft.rfft(h_seg - h_seg.mean())
    cross_sum += np.conj(H) * S
    pow_s_sum += np.abs(S)**2
    pow_h_sum += np.abs(H)**2
    n_valid   += 1

print(f"Segmenti validi: {n_valid}")

cross_mean = cross_sum / n_valid
pow_s_mean = pow_s_sum / n_valid
pow_h_mean = pow_h_sum / n_valid
coherence  = np.abs(cross_mean)**2 / (pow_s_mean * pow_h_mean)

freq       = freq[1:]
cross_mean = cross_mean[1:]
coherence  = coherence[1:]

phase   = np.angle(cross_mean)
with np.errstate(divide='ignore', invalid='ignore'):
    lag     = phase / (2 * np.pi * freq)
    lag_err = np.sqrt((1 - coherence) / (2 * coherence * n_valid)) / (2 * np.pi * freq)

# ── binning logaritmico ───────────────────────────────────────────────────────
f_min, f_max = 7e-5, 5e-3
n_bins  = 12
f_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins+1)
f_c     = np.sqrt(f_edges[:-1] * f_edges[1:])

lag_b = np.full(n_bins, np.nan)
err_b = np.full(n_bins, np.nan)
coh_b = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = (freq >= f_edges[i]) & (freq < f_edges[i+1])
    if mask.sum() < 2:
        continue
    w = coherence[mask]
    if w.sum() <= 0:
        continue
    lag_b[i] = np.average(lag[mask],     weights=w)
    err_b[i] = np.sqrt(np.average(lag_err[mask]**2, weights=w)) / np.sqrt(mask.sum())
    coh_b[i] = np.mean(w)

print(f"\nLag-frequency spectrum (1H 0707-495):")
print(f"{'f [mHz]':>10} {'lag [s]':>10} {'err [s]':>10} {'coh':>8}")
print("-"*42)
for i in range(n_bins):
    if not np.isnan(lag_b[i]):
        print(f"{f_c[i]*1e3:10.3f} {lag_b[i]:10.1f} {err_b[i]:10.1f} {coh_b[i]:8.3f}")

# ── banda reverberation (0.6-3 mHz, Kara+2013 per 1H0707) ───────────────────
rev = (freq >= 6e-4) & (freq <= 3e-3)
w_rev    = coherence[rev]
lag_rev  = np.average(lag[rev],     weights=w_rev)
err_rev  = np.sqrt(np.average(lag_err[rev]**2, weights=w_rev)) / np.sqrt(rev.sum())

# Converti in r_g/c (M_bh ~ 2e6 M_sun, r_g/c ~ 10 s)
R_G_C   = 10.0
lag_rg  = abs(lag_rev) / R_G_C
err_rg  = err_rev / R_G_C

print(f"\n>>> Reverberation lag (0.6-3 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f">>> In r_g/c: {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")

# Salva per QCORONA
os.makedirs('results', exist_ok=True)
np.save(f'{DATA_DIR}/lag_rev.npy',
        np.array([lag_rev, err_rev, lag_rg, err_rg]))
np.save(f'{DATA_DIR}/lag_freq_centers.npy', f_c)
np.save(f'{DATA_DIR}/lag_freq_values.npy',  lag_b)
np.save(f'{DATA_DIR}/lag_freq_errors.npy',  err_b)

# ── figura ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0),
                                gridspec_kw={'height_ratios': [3, 1.2]},
                                sharex=True)

valid = ~np.isnan(lag_b)
ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax1.axvspan(0.6, 3.0, alpha=0.12, color='#d6604d', label='Reverberation band')
ax1.errorbar(f_c[valid]*1e3, lag_b[valid], yerr=err_b[valid],
             fmt='o', color='#2166ac', capsize=3, markersize=4.5,
             label='Lag (soft$-$hard)')
ax1.set_ylabel(r'Time lag [s]  (soft $-$ hard)')
ax1.set_title('1H 0707-495 — EPIC-pn, ObsID 0511580101 (2008)')
ax1.legend(fontsize=7.5, framealpha=0.8)
ax1.set_xscale('log')
ax1.yaxis.grid(True, lw=0.4, alpha=0.4)

ax1.annotate(fr'$\Delta t = {lag_rev:.0f} \pm {err_rev:.0f}$ s',
             xy=(1.2, lag_rev),
             xytext=(0.2, lag_rev - abs(lag_rev)*0.4),
             fontsize=7.5, color='#d6604d',
             arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))

ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax2.axvspan(0.6, 3.0, alpha=0.12, color='#d6604d')
ax2.plot(f_c[valid]*1e3, coh_b[valid], 'o-',
         color='#4dac26', markersize=4, lw=0.8)
ax2.set_ylabel('Coherence')
ax2.set_xlabel('Frequency [mHz]')
ax2.set_ylim(0, 1)
ax2.yaxis.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()
png_path = 'results/lag_frequency_1h0707.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {png_path}")

# ── pgfplots ──────────────────────────────────────────────────────────────────
def cerr(xs, ys, es):
    pts = []
    for x, y, e in zip(xs, ys, es):
        if not np.isnan(y):
            pts.append(f'({x:.4f},{y:.2f}) +- (0,{abs(e):.2f})')
    return ' '.join(pts)

tex = [
    r'% 1H 0707-495 lag-frequency spectrum — pgfplots source',
    r'% \usepackage{pgfplots}',
    r'% \pgfplotsset{compat=1.18}',
    r'',
    r'\begin{tikzpicture}',
    r'\begin{axis}[',
    r'  width=8cm, height=6cm,',
    r'  xmode=log,',
    r'  xlabel={Frequency [mHz]},',
    r'  ylabel={Time lag [s]},',
    r'  title={1H~0707$-$495 lag-frequency spectrum (EPIC-pn, 2008)},',
    r'  tick label style={font=\scriptsize},',
    r'  label style={font=\scriptsize},',
    r'  title style={font=\small},',
    r'  axis line style={thin},',
    r'  ymajorgrids=true, grid style={dotted,thin},',
    r']',
    r'\addplot[dashed,black,thin,domain=0.07:5] {0};',
    r'\addplot[fill=red!15,draw=none] coordinates {',
    r'  (0.6,-500)(3.0,-500)(3.0,500)(0.6,500)',
    r'} \closedcycle;',
    r'\addplot[',
    r'  only marks, mark=*, mark size=1.5pt,',
    r'  blue!70!black,',
    r'  error bars/.cd, y dir=both, y explicit,',
    r'] coordinates {',
    '  ' + cerr(f_c[valid]*1e3, lag_b[valid], err_b[valid]),
    r'};',
    r'\node[red!70!black,font=\scriptsize] at (axis cs:1.5,' + f'{lag_rev*1.3:.0f}' + r') ',
    r'  {$\Delta t=' + f'{lag_rev:.0f}' + r'\,\mathrm{s}$};',
    r'\end{axis}',
    r'\end{tikzpicture}',
]

tex_path = 'results/lag_frequency_1h0707.tex'
with open(tex_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex))
print(f"Saved: {tex_path}")
print(f"\nSUMMARY — 1H 0707-495")
print(f"  Lag (0.6-3 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"  In r_g/c:        {lag_rg:.2f} +/- {err_rg:.2f}")
