"""
compute_lag_iras13224.py  —  Lag-frequency spectrum di IRAS 13224.

Adattato per dati brevi (segmenti di lunghezza automatica).
Osservazione: XMM-Newton ObsID 0780560101 (2016)
Bande: soft 0.3-1.0 keV vs hard 1.5-4.0 keV
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
DATA_DIR = 'data/sources/iras13224'
s = np.load(f'{DATA_DIR}/soft_lc.npy')
h = np.load(f'{DATA_DIR}/hard_lc.npy')
t = np.load(f'{DATA_DIR}/time.npy')
dt = 100.0  # s

print(f"IRAS 13224 — ObsID 0780560101 (2016)")
print(f"Bin validi: {len(s)}")
print(f"Durata: {(t[-1]-t[0])/1000:.1f} ks")
print(f"Soft mean: {s.mean():.2f} cts/s")
print(f"Hard mean: {h.mean():.2f} cts/s")

# ── adattamento segmenti ──────────────────────────────────────────────────────
n_bins = len(s)
seg_len = 64
if n_bins > 128:
    seg_len = 128
if n_bins > 256:
    seg_len = 256
if n_bins < seg_len:
    seg_len = n_bins // 2
    if seg_len < 16:
        print("ERRORE: dati troppo brevi")
        sys.exit(1)

print(f"Lunghezza segmenti: {seg_len} bin ({seg_len*dt/1000:.1f} ks)")

n_seg = (n_bins - seg_len) // (seg_len // 2) + 1
if n_seg < 1:
    print("ERRORE: nessun segmento")
    sys.exit(1)

print(f"Segmenti totali (con sovrapposizione): {n_seg}")

# ── Vaughan+2003 cross-spectrum ───────────────────────────────────────────────
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
    if np.sum(s_seg > 0) < seg_len * 0.8:
        continue
    S = np.fft.rfft(s_seg - s_seg.mean())
    H = np.fft.rfft(h_seg - h_seg.mean())
    cross_sum += np.conj(H) * S
    pow_s_sum += np.abs(S)**2
    pow_h_sum += np.abs(H)**2
    n_valid   += 1

print(f"Segmenti validi: {n_valid}")

if n_valid == 0:
    print("ERRORE: nessun segmento valido")
    sys.exit(1)

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
n_bins_f = 12
f_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins_f+1)
f_c     = np.sqrt(f_edges[:-1] * f_edges[1:])

lag_b = np.full(n_bins_f, np.nan)
err_b = np.full(n_bins_f, np.nan)
coh_b = np.full(n_bins_f, np.nan)

for i in range(n_bins_f):
    mask = (freq >= f_edges[i]) & (freq < f_edges[i+1])
    if mask.sum() < 2:
        continue
    w = coherence[mask]
    if w.sum() <= 0:
        continue
    lag_b[i] = np.average(lag[mask],     weights=w)
    err_b[i] = np.sqrt(np.average(lag_err[mask]**2, weights=w)) / np.sqrt(mask.sum())
    coh_b[i] = np.mean(w)

print(f"\nLag-frequency spectrum (IRAS 13224):")
print(f"{'f [mHz]':>10} {'lag [s]':>10} {'err [s]':>10} {'coh':>8}")
print("-"*42)
for i in range(n_bins_f):
    if not np.isnan(lag_b[i]):
        print(f"{f_c[i]*1e3:10.3f} {lag_b[i]:10.1f} {err_b[i]:10.1f} {coh_b[i]:8.3f}")

# ── banda reverberation (0.6-3 mHz) ──────────────────────────────────────────
REV_FMIN = 6e-4   # Hz
REV_FMAX = 3e-3   # Hz
rev   = (freq >= REV_FMIN) & (freq <= REV_FMAX)
w_rev = coherence[rev]

if w_rev.sum() > 0:
    lag_rev = np.average(lag[rev],     weights=w_rev)
    err_rev = np.sqrt(np.average(lag_err[rev]**2, weights=w_rev)) / np.sqrt(rev.sum())
else:
    lag_rev, err_rev = np.nan, np.nan

R_G_C = 7.4   # s per r_g/c  (M_bh ~ 1.5e6 M_sun)
if not np.isnan(lag_rev):
    lag_rg = abs(lag_rev) / R_G_C
    err_rg = err_rev / R_G_C
else:
    lag_rg, err_rg = np.nan, np.nan

print(f"\n>>> Reverberation lag (0.6-3 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f">>> In r_g/c: {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")

os.makedirs('results', exist_ok=True)
if not np.isnan(lag_rev):
    np.save(f'{DATA_DIR}/lag_rev.npy',
            np.array([lag_rev, err_rev, lag_rg, err_rg]))
    np.save(f'{DATA_DIR}/lag_freq_centers.npy', f_c)
    np.save(f'{DATA_DIR}/lag_freq_values.npy',  lag_b)
    np.save(f'{DATA_DIR}/lag_freq_errors.npy',  err_b)

# ── figura ────────────────────────────────────────────────────────────────────
valid = ~np.isnan(lag_b)
if not np.any(valid):
    print("Nessun punto valido -> figura non creata.")
    sys.exit(0)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0),
                                gridspec_kw={'height_ratios': [3, 1.2]},
                                sharex=True)

ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax1.axvspan(REV_FMIN*1e3, REV_FMAX*1e3, alpha=0.12, color='#d6604d',
            label='Reverberation band')
ax1.errorbar(f_c[valid]*1e3, lag_b[valid], yerr=err_b[valid],
             fmt='o', color='#2166ac', capsize=3, markersize=4.5,
             label='Lag (soft$-$hard)')
ax1.set_ylabel(r'Time lag [s]  (soft $-$ hard)')
ax1.set_title('IRAS 13224 — EPIC-pn, ObsID 0780560101 (2016)')
ax1.legend(fontsize=7.5, framealpha=0.8, loc='upper right')
ax1.set_xscale('log')
ax1.yaxis.grid(True, lw=0.4, alpha=0.4)

# Annotazione lag — testo a destra della banda per non sovrapporsi all'asse y
if not np.isnan(lag_rev):
    f_ann = np.sqrt(REV_FMIN * REV_FMAX) * 1e3   # mHz, centro banda
    # testo posizionato a destra del centro banda
    ax1.annotate(
        fr'$\Delta t = {lag_rev:.0f} \pm {err_rev:.0f}$ s',
        xy=(f_ann, lag_rev),
        xytext=(f_ann * 4, lag_rev + abs(lag_rev) * 0.4),
        fontsize=7.5, color='#d6604d',
        arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8),
    )

ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax2.axvspan(REV_FMIN*1e3, REV_FMAX*1e3, alpha=0.12, color='#d6604d')
ax2.plot(f_c[valid]*1e3, coh_b[valid], 'o-',
         color='#4dac26', markersize=4, lw=0.8)
ax2.set_ylabel('Coherence')
ax2.set_xlabel('Frequency [mHz]')
ax2.set_ylim(0, 1)
ax2.yaxis.grid(True, lw=0.4, alpha=0.4)

plt.tight_layout()

# Salva come PDF
pdf_path = 'results/lag_frequency_iras13224.pdf'
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(f"\nSaved: {pdf_path}")

print(f"\nSUMMARY — IRAS 13224")
print(f"  Lag (0.6-3 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f"  In r_g/c:        {lag_rg:.2f} +/- {err_rg:.2f}")