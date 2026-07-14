"""
run_mcg6.py  —  Lag-frequency spectrum + QCORONA fit di MCG-6-30-15.

Osservazione: XMM-Newton ObsID 0029740701 (2001, agosto)
Bande: soft 0.3-1.0 keV vs hard 1.5-4.0 keV
Lag atteso: ~100-200 s = ~7-13 r_g/c (Kara+2014)

Output:
    results/lag_frequency_mcg6.pdf / .tex
    results/fig_mcg6.png / .tex
"""

import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig,
    ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_lag_matrix, compute_transfer_function_full

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

# ═══════════════════════════════════════════════════════════
# PARTE 1 — LAG-FREQUENCY SPECTRUM
# ═══════════════════════════════════════════════════════════

DATA_DIR = 'data/sources/mcg6'
s = np.load(f'{DATA_DIR}/soft_lc.npy')
h = np.load(f'{DATA_DIR}/hard_lc.npy')
t = np.load(f'{DATA_DIR}/time.npy')
dt = 100.0  # s

print("MCG-6-30-15 — ObsID 0029740701 (2001)")
print(f"Bin validi: {len(s)}")
print(f"Durata: {(t[-1]-t[0])/1000:.1f} ks")
print(f"Soft mean: {s.mean():.2f} cts/s")
print(f"Hard mean: {h.mean():.2f} cts/s")

# ── Vaughan+2003 cross-spectrum ───────────────────────────
seg_len = 256
freq      = np.fft.rfftfreq(seg_len, d=dt)
cross_sum = np.zeros(len(freq), dtype=complex)
pow_s_sum = np.zeros(len(freq))
pow_h_sum = np.zeros(len(freq))
n_valid   = 0

for i in range((len(s) - seg_len) // (seg_len // 2) + 1):
    i0, i1 = i * (seg_len // 2), i * (seg_len // 2) + seg_len
    if i1 > len(s): break
    s_seg, h_seg = s[i0:i1], h[i0:i1]
    if np.sum(s_seg > 0) < seg_len * 0.8: continue
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

# ── binning logaritmico ───────────────────────────────────
f_min, f_max = 7e-5, 5e-3
n_bins  = 12
f_edges = np.logspace(np.log10(f_min), np.log10(f_max), n_bins+1)
f_c     = np.sqrt(f_edges[:-1] * f_edges[1:])

lag_b = np.full(n_bins, np.nan)
err_b = np.full(n_bins, np.nan)
coh_b = np.full(n_bins, np.nan)

for i in range(n_bins):
    mask = (freq >= f_edges[i]) & (freq < f_edges[i+1])
    if mask.sum() < 2: continue
    w = coherence[mask]
    if w.sum() <= 0: continue
    lag_b[i] = np.average(lag[mask],     weights=w)
    err_b[i] = np.sqrt(np.average(lag_err[mask]**2, weights=w)) / np.sqrt(mask.sum())
    coh_b[i] = np.mean(w)

print(f"\nLag-frequency spectrum (MCG-6-30-15):")
print(f"{'f [mHz]':>10} {'lag [s]':>10} {'err [s]':>10} {'coh':>8}")
print("-"*42)
for i in range(n_bins):
    if not np.isnan(lag_b[i]):
        print(f"{f_c[i]*1e3:10.3f} {lag_b[i]:10.1f} {err_b[i]:10.1f} {coh_b[i]:8.3f}")

# ── banda reverberation MCG-6-30-15: 0.3-1.0 mHz (Kara+2014) ────────────────
rev = (freq >= 3e-4) & (freq <= 1e-3)
w_rev   = coherence[rev]
lag_rev = np.average(lag[rev], weights=w_rev)
err_rev = np.sqrt(np.average(lag_err[rev]**2, weights=w_rev)) / np.sqrt(rev.sum())

# r_g/c per M=3e6 Msun: r_g/c = 1.48e3 * 3 = 14.8 s ~ 15 s
R_G_C   = 15.0
lag_rg  = abs(lag_rev) / R_G_C
err_rg  = err_rev / R_G_C

print(f"\n>>> Reverberation lag (0.3-1.0 mHz): {lag_rev:.1f} +/- {err_rev:.1f} s")
print(f">>> In r_g/c: {lag_rg:.2f} +/- {err_rg:.2f} r_g/c")

os.makedirs('results', exist_ok=True)
np.save(f'{DATA_DIR}/lag_rev.npy',           np.array([lag_rev, err_rev, lag_rg, err_rg]))
np.save(f'{DATA_DIR}/lag_freq_centers.npy',  f_c)
np.save(f'{DATA_DIR}/lag_freq_values.npy',   lag_b)
np.save(f'{DATA_DIR}/lag_freq_errors.npy',   err_b)

# ── figura lag-frequency ──────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 5.0),
                                gridspec_kw={'height_ratios': [3, 1.2]},
                                sharex=True)
valid = ~np.isnan(lag_b)
ax1.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax1.axvspan(0.3, 1.0, alpha=0.12, color='#d6604d', label='Reverberation band')
ax1.errorbar(f_c[valid]*1e3, lag_b[valid], yerr=err_b[valid],
             fmt='o', color='#2166ac', capsize=3, markersize=4.5,
             label='Lag (soft$-$hard)')
ax1.set_ylabel(r'Time lag [s]  (soft $-$ hard)')
ax1.set_title('MCG-6-30-15 — EPIC-pn, ObsID 0029740701 (2001)')
ax1.legend(fontsize=7.5, framealpha=0.8)
ax1.set_xscale('log')
ax1.yaxis.grid(True, lw=0.4, alpha=0.4)
ax1.annotate(fr'$\Delta t = {lag_rev:.0f} \pm {err_rev:.0f}$ s',
             xy=(0.55, lag_rev), xytext=(2.0, lag_rev - abs(lag_rev)*0.4),
             fontsize=7.5, color='#d6604d',
             arrowprops=dict(arrowstyle='->', color='#d6604d', lw=0.8))
ax2.axhline(0, color='k', lw=0.7, ls='--', alpha=0.4)
ax2.axvspan(0.3, 1.0, alpha=0.12, color='#d6604d')
ax2.plot(f_c[valid]*1e3, coh_b[valid], 'o-', color='#4dac26', markersize=4, lw=0.8)
ax2.set_ylabel('Coherence')
ax2.set_xlabel('Frequency [mHz]')
ax2.set_ylim(0, 1)
ax2.yaxis.grid(True, lw=0.4, alpha=0.4)
plt.tight_layout()
fig.savefig('results/lag_frequency_mcg6.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: results/lag_frequency_mcg6.pdf")

# ═══════════════════════════════════════════════════════════
# PARTE 2 — QCORONA FIT
# ═══════════════════════════════════════════════════════════

# Parametri MCG-6-30-15
# M_bh ~ 3e6 Msun (Ponti+2004, dalla relazione M-sigma)
# r_g/c ~ 15 s
# Lag misurato nella banda 0.3-1.0 mHz
# Inclinazione ~ 30 deg (Reynolds+2004)
# Reflection fraction ~ 0.5 (Kara+2014)

M_BH       = 3.0e6
LAG_OBS_S  = abs(lag_rev)
LAG_ERR_S  = max(err_rev, 10.0)   # minimo 10s per evitare prior troppo stretta
LAG_OBS_RG = LAG_OBS_S / R_G_C
LAG_ERR_RG = LAG_ERR_S / R_G_C
R_OBS      = 0.5
L_X        = 2.0e43
INCL       = 30.0
N_STARTS   = 5
MAX_ITER   = 2000
N_BINS_TF  = 80

print(f"\n{'='*60}")
print("HAMCOR — MCG-6-30-15 (XMM ObsID 0029740701, 2001)")
print(f"{'='*60}")
print(f"  M_bh    = {M_BH:.1e} M_sun")
print(f"  r_g/c   = {R_G_C:.0f} s")
print(f"  Lag obs = {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s "
      f"= {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f} r_g/c")

print("\nBuilding grids...")
GRID = build_grid(n_r=10, n_z=10, n_phi=8,
                  r_in=3.0, r_out=15.0,
                  z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
SA   = compute_solid_angles(GRID, DISK)
LM   = compute_lag_matrix(GRID, DISK, observer_inclination=INCL)
print(f"  Lag matrix range: {LM.min():.2f} -- {LM.max():.2f} r_g/c")
print(f"  Lag obs {LAG_OBS_RG:.2f} in range: {LM.min() < LAG_OBS_RG < LM.max()}")

T_MIN  = max(0.0, float(LM.min()))
T_MAX  = 15.0
t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
t_c    = 0.5 * (t_bins[:-1] + t_bins[1:])

psi_obs = np.exp(-0.5 * ((t_c - LAG_OBS_RG) / LAG_ERR_RG)**2)
psi_obs = psi_obs / psi_obs.sum()

constraints = ObservationalConstraints(
    delta_t_obs          = LAG_OBS_RG,
    R_obs                = R_OBS,
    L_X                  = L_X,
    M_bh                 = M_BH,
    eddington_ratio      = 0.1,
    observer_inclination = INCL,
    source_name          = "MCG6_XMM2001",
    psi_obs              = psi_obs,
    t_bins_obs           = t_bins,
)

h_config = HamiltonianConfig(
    J=0.3, alpha=50.0, beta=10.0,
    gamma=1.0, l_crit=1000.0,
    delta=1.0, f_max=1.0,
    use_pair_barrier=False,
    use_energy_barrier=False,
)

H_obj = QCORONAHamiltonian(GRID, DISK, constraints, h_config)

print(f"\nOptimizing ({N_STARTS} cold starts, {MAX_ITER} iter each)...")
print(f"{'seed':>5} {'H_final':>10} {'lag_pred':>10} {'I_pred':>8} {'time':>7}")
print("-" * 45)

opt_cfg = OptimizationConfig(
    max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
    use_adaptive_lr=True, use_continuation=True,
    continuation_steps=200, convergence_tol=1e-7, verbose=False,
)

all_results = []
for seed in range(N_STARTS):
    rng = np.random.default_rng(seed)
    w0  = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
    s0  = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
    opt = QCORONAOptimizer(H_obj, opt_cfg)
    t0  = time.time()
    res = opt.optimize(initial_state=s0)
    elapsed = time.time() - t0
    d = res.final_H.diagnostics
    print(f"{seed:>5} {res.final_H.H_total:>10.4f} "
          f"{d['delta_t_pred']:>10.2f} {d['I_pred']:>8.3f} {elapsed:>6.0f}s")
    all_results.append(res)

best_idx = int(np.argmin([r.final_H.H_total for r in all_results]))
best     = all_results[best_idx]
w_best   = best.final_state.emissivity
d_best   = best.final_H.diagnostics

print(f"\nBest result (seed {best_idx}):")
print(f"  lag_pred = {d_best['delta_t_pred']:.2f} r_g/c  "
      f"(obs: {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f})")
print(f"  lag_pred = {d_best['delta_t_pred']*R_G_C:.1f} s  "
      f"(obs: {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s)")

tf_pred = compute_transfer_function_full(
    GRID, DISK, w_best, SA,
    observer_inclination=INCL, n_bins=N_BINS_TF,
    t_min=float(t_bins[0]), t_max=float(t_bins[-1]), lag_matrix=LM,
)

# ── figura QCORONA ────────────────────────────────────────
def rz_map(w):
    return w.reshape(GRID.n_r, GRID.n_z, GRID.n_phi).sum(axis=2)

R_c = np.unique(np.round(GRID.R, 8))[:GRID.n_r]
z_c = np.unique(np.round(GRID.z, 8))[:GRID.n_z]

fig = plt.figure(figsize=(7.2, 4.0))
gs  = gridspec.GridSpec(1, 3, figure=fig,
                         left=0.09, right=0.97, top=0.88, bottom=0.14, wspace=0.48)

ax0 = fig.add_subplot(gs[0, 0])
for i, res in enumerate(all_results):
    H_arr = np.array([h['H_total'] for h in res.history])
    ax0.semilogy(np.arange(len(H_arr)), np.abs(H_arr),
                 color='#2166ac', alpha=1.0 if i==best_idx else 0.25,
                 lw=1.4 if i==best_idx else 0.7)
ax0.set_xlabel('Iteration'); ax0.set_ylabel(r'$|\mathcal{H}|$')
ax0.set_title('Convergence'); ax0.tick_params(labelsize=7)

ax1 = fig.add_subplot(gs[0, 1])
im = ax1.pcolormesh(R_c, z_c, rz_map(w_best).T,
                     cmap='hot_r', shading='auto', rasterized=True)
ax1.set_xlabel(r'$R\;[r_g]$'); ax1.set_ylabel(r'$z\;[r_g]$')
ax1.set_title('Recovered corona')
cb = fig.colorbar(im, ax=ax1, fraction=0.038, pad=0.02, shrink=0.85)
cb.ax.tick_params(labelsize=6); cb.set_label(r'$w_i$', fontsize=7)
cb.ax.yaxis.set_major_locator(MaxNLocator(3)); ax1.tick_params(labelsize=7)
ax1.text(0.05, 0.95,
         f'lag = {d_best["delta_t_pred"]:.2f} $r_g/c$\nobs: {LAG_OBS_RG:.2f} $r_g/c$',
         transform=ax1.transAxes, va='top', fontsize=7,
         bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.85))

ax2 = fig.add_subplot(gs[0, 2])
ax2.fill_between(t_c, psi_obs, alpha=0.18, color='#d6604d')
ax2.plot(t_c, psi_obs,     color='#d6604d', lw=1.3, label=r'$\psi_\mathrm{obs}$')
ax2.plot(t_c, tf_pred.psi, color='#2166ac', lw=1.1, ls='--', label=r'$\psi_\mathrm{pred}$')
ax2.set_xlabel(r'$\Delta t\;[r_g/c]$'); ax2.set_ylabel(r'$\psi(t)$')
ax2.set_title('Transfer function')
ax2.legend(fontsize=7, framealpha=0.85); ax2.set_ylim(bottom=0)
sig = np.where(psi_obs > 0.001 * psi_obs.max())[0]
if len(sig) > 1:
    ax2.set_xlim(max(0.0, t_c[sig[0]]-0.5), t_c[sig[-1]]+2.0)
ax2.tick_params(labelsize=7)
chi2 = float(np.sum((tf_pred.psi - psi_obs)**2))
ax2.text(0.97, 0.93, f'$\\chi^2={chi2:.2e}$',
         transform=ax2.transAxes, ha='right', va='top', fontsize=6.5, color='gray')

fig.suptitle('MCG-6-30-15 — HAMCOR fit (XMM ObsID 0029740701, 2001)',
             fontsize=9.5, y=0.98)
fig.savefig('results/fig_mcg6.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig)
print("Saved: results/fig_mcg6.pdf")

# ── pgfplots ──────────────────────────────────────────────
def coords(xs, ys):
    return ' '.join(f'({float(x):.4f},{float(y):.6g})' for x, y in zip(xs, ys))

tex = [
    r'% HAMCOR MCG-6-30-15 — pgfplots',
    r'\begin{tikzpicture}',
    r'\begin{axis}[width=6cm,height=4.5cm,',
    r'  xlabel={$\Delta t\;[r_g/c]$},ylabel={$\psi(t)$},ymin=0,',
    r'  title={MCG$-$6$-$30$-$15 transfer function},',
    r'  tick label style={font=\tiny},label style={font=\scriptsize}]',
    r'\addplot[thick,red!60!black,fill=red!15,fill opacity=0.3]',
    r'  coordinates {' + coords(t_c, psi_obs) + r'} \closedcycle;',
    r'\addplot[dashed,thick,blue!70!black] coordinates {',
    '  ' + coords(t_c, tf_pred.psi), r'};',
    r'\legend{$\psi_\mathrm{obs}$,$\psi_\mathrm{pred}$}',
    r'\end{axis}',
    r'\end{tikzpicture}',
]
with open('results/fig_mcg6.tex', 'w', encoding='utf-8') as f:
    f.write('\n'.join(tex))
print("Saved: results/fig_mcg6.tex")

print(f"\n{'='*60}")
print("SUMMARY — MCG-6-30-15 HAMCOR fit")
print(f"{'='*60}")
print(f"  Lag obs:    {LAG_OBS_S:.1f} +/- {LAG_ERR_S:.1f} s "
      f"= {LAG_OBS_RG:.2f} +/- {LAG_ERR_RG:.2f} r_g/c")
print(f"  Lag pred:   {d_best['delta_t_pred']*R_G_C:.1f} s "
      f"= {d_best['delta_t_pred']:.2f} r_g/c")
print(f"  Illum obs:  {constraints.I_obs:.3f}")
print(f"  Illum pred: {d_best['I_pred']:.3f}")
print(f"  H_final:    {best.final_H.H_total:.4f}")
print(f"  chi2(psi):  {chi2:.4e}")
print(f"{'='*60}")