#!/usr/bin/env python3
"""
multi_single.py
───────────────
Mrk 335 multi-epoch, UN'EPOCA ALLA VOLTA.

Per ogni epoca:
  - fit HAMCOR (Schwarzschild)
  - salva summary CSV/JSON parziale
  - salva figura PDF + PNG in results/mrk335_multiepoch/

Se interrompi e rilanci, le epoche già fatte vengono saltate
(se esiste results/mrk335_multiepoch/epoch_YYYY.npz).
"""

import sys
import os
import json
import time
import csv
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (
    QCORONAHamiltonian, HamiltonianConfig,
    ObservationalConstraints, CoronalState,
)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.hamiltonian.terms import vertical_orientations
from qcorona.physics.illumination import compute_solid_angles
from qcorona.physics.lags import compute_transfer_function_full
from schwarzschild_lags import compute_lag_matrix_schwarzschild

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 9,
    'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.6, 'lines.linewidth': 1.0,
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.fontset': 'dejavusans',
})

# ── paths ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path('results/mrk335_multiepoch')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Mrk 335 ───────────────────────────────────────────────────────────────────
M_BH  = 1.3e7
R_G_C = 63.0

EPOCHS = [
    {
        'obsid': '0306870101', 'year': 2006, 'state': 'High',
        'lag_s': 91.9, 'lag_err_s': 43.4,
        'R_obs': 0.3, 'L_X': 1.0e43, 'incl': 45.0,
        'done': True,
    },
    {
        'obsid': '0600540601', 'year': 2009, 'state': 'Low',
        'lag_s': 6.0, 'lag_err_s': 6.0,
        'R_obs': 2.0, 'L_X': 5.0e41, 'incl': 45.0,
        'done': True,
    },
    {
        'obsid': '0741280201', 'year': 2015, 'state': 'Intermediate',
        'lag_s': 19.5, 'lag_err_s': 8.7,
        'R_obs': 1.0, 'L_X': 3.0e42, 'incl': 45.0,
        'done': True,
    },
    {
        'obsid': '0780500301', 'year': 2018, 'state': 'Intermediate',
        'lag_s': 7.7, 'lag_err_s': 6.5,
        'R_obs': 0.8, 'L_X': 5.0e42, 'incl': 45.0,
        'done': True,
    },
    {
        'obsid': '0831790601', 'year': 2019, 'state': 'Recovering',
        'lag_s': 39.0, 'lag_err_s': 7.7,
        'R_obs': 0.5, 'L_X': 8.0e42, 'incl': 45.0,
        'done': True,
    },
]

N_STARTS = 5
MAX_ITER = 2000
N_BINS_TF = 80

COLOR_MAP = {
    'High': '#d6604d',
    'Low': '#2166ac',
    'Intermediate': '#4dac26',
    'Recovering': '#9c4dac',
}


def epoch_npz_path(year: int) -> Path:
    return OUT_DIR / f'epoch_{year}.npz'


def run_one_epoch(epoch: dict) -> dict | None:
    if not epoch.get('done', False) or epoch.get('lag_s') is None:
        print(f"  Skip ObsID {epoch['obsid']}: not ready.")
        return None

    year = epoch['year']
    lag_rg = epoch['lag_s'] / R_G_C
    lag_err = epoch['lag_err_s'] / R_G_C

    print("\n" + "=" * 60)
    print(f"EPOCH {year}  ObsID {epoch['obsid']}  ({epoch['state']})")
    print(f"  lag = {epoch['lag_s']:.1f} +/- {epoch['lag_err_s']:.1f} s"
          f" = {lag_rg:.2f} +/- {lag_err:.2f} r_g/c")
    print("=" * 60)

    GRID = build_grid(
        n_r=10, n_z=10, n_phi=8,
        r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0,
    )
    DISK = build_disk_grid(n_r=30, n_phi=18)
    SA = compute_solid_angles(GRID, DISK)
    LM = compute_lag_matrix_schwarzschild(
        GRID, DISK, observer_inclination=epoch['incl'],
    )

    T_MIN = max(0.0, float(LM.min()))
    T_MAX = 15.0
    t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
    t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs = np.exp(-0.5 * ((t_c - lag_rg) / lag_err) ** 2)
    psi_obs = psi_obs / psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=lag_rg,
        R_obs=epoch['R_obs'],
        L_X=epoch['L_X'],
        M_bh=M_BH,
        eddington_ratio=epoch['L_X'] / (1.3e38 * M_BH),
        observer_inclination=epoch['incl'],
        source_name=f"Mrk335_{year}",
        psi_obs=psi_obs,
        t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(
        GRID, DISK, constraints, h_config, lag_matrix=LM,
    )
    opt_cfg = OptimizationConfig(
        max_iter=MAX_ITER,
        lr_emissivity=0.01,
        lr_orientation=0.01,
        use_adaptive_lr=True,
        use_continuation=True,
        continuation_steps=200,
        convergence_tol=1e-7,
        verbose=False,
    )

    all_H_histories = []
    all_res_meta = []
    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0 = rng.exponential(size=GRID.n_cells)
        w0 /= w0.sum()
        s0 = CoronalState(
            emissivity=w0,
            orientations=vertical_orientations(GRID.n_cells),
        )
        t0 = time.time()
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        elapsed = time.time() - t0
        d = res.final_H.diagnostics
        print(
            f"  seed {seed}: H={res.final_H.H_total:.4f}  "
            f"lag={d['delta_t_pred']:.2f}  ({elapsed:.0f}s)"
        )
        H_hist = np.array([h['H_total'] for h in res.history], dtype=float)
        all_H_histories.append(H_hist)
        all_res_meta.append({
            'seed': seed,
            'H_final': float(res.final_H.H_total),
            'lag_pred': float(d['delta_t_pred']),
            'I_pred': float(d['I_pred']),
        })
        if seed == 0 or res.final_H.H_total < min(m['H_final'] for m in all_res_meta[:-1] or [{'H_final': 1e99}]):
            pass  # best tracked below

    best_idx = int(np.argmin([m['H_final'] for m in all_res_meta]))
    # re-run not needed: keep best from loop
    # We need w_best from the best seed — re-optimize only best is expensive;
    # store during loop:
    # Fix: keep best state in loop properly
    return None  # placeholder — see full loop below


def run_one_epoch_full(epoch: dict) -> dict | None:
    """Versione completa con best state salvato nel loop."""
    if not epoch.get('done', False) or epoch.get('lag_s') is None:
        print(f"  Skip ObsID {epoch['obsid']}: not ready.")
        return None

    year = int(epoch['year'])
    lag_rg = float(epoch['lag_s']) / R_G_C
    lag_err = float(epoch['lag_err_s']) / R_G_C

    print("\n" + "=" * 60)
    print(f"EPOCH {year}  ObsID {epoch['obsid']}  ({epoch['state']})")
    print(
        f"  lag = {epoch['lag_s']:.1f} +/- {epoch['lag_err_s']:.1f} s"
        f" = {lag_rg:.2f} +/- {lag_err:.2f} r_g/c"
    )
    print("=" * 60)

    GRID = build_grid(
        n_r=10, n_z=10, n_phi=8,
        r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0,
    )
    DISK = build_disk_grid(n_r=30, n_phi=18)
    SA = compute_solid_angles(GRID, DISK)
    LM = compute_lag_matrix_schwarzschild(
        GRID, DISK, observer_inclination=epoch['incl'],
    )

    T_MIN = max(0.0, float(LM.min()))
    T_MAX = 15.0
    t_bins = np.linspace(T_MIN, T_MAX, N_BINS_TF + 1)
    t_c = 0.5 * (t_bins[:-1] + t_bins[1:])
    psi_obs = np.exp(-0.5 * ((t_c - lag_rg) / max(lag_err, 1e-6)) ** 2)
    psi_obs = psi_obs / psi_obs.sum()

    constraints = ObservationalConstraints(
        delta_t_obs=lag_rg,
        R_obs=epoch['R_obs'],
        L_X=epoch['L_X'],
        M_bh=M_BH,
        eddington_ratio=epoch['L_X'] / (1.3e38 * M_BH),
        observer_inclination=epoch['incl'],
        source_name=f"Mrk335_{year}",
        psi_obs=psi_obs,
        t_bins_obs=t_bins,
    )
    h_config = HamiltonianConfig(
        J=0.3, alpha=50.0, beta=10.0,
        gamma=1.0, l_crit=1000.0, delta=1.0, f_max=1.0,
        use_pair_barrier=False, use_energy_barrier=False,
    )
    H_obj = QCORONAHamiltonian(
        GRID, DISK, constraints, h_config, lag_matrix=LM,
    )
    opt_cfg = OptimizationConfig(
        max_iter=MAX_ITER,
        lr_emissivity=0.01,
        lr_orientation=0.01,
        use_adaptive_lr=True,
        use_continuation=True,
        continuation_steps=200,
        convergence_tol=1e-7,
        verbose=False,
    )

    best_H = np.inf
    best_idx = 0
    best_w = None
    best_d = None
    all_H_histories = []

    for seed in range(N_STARTS):
        rng = np.random.default_rng(seed)
        w0 = rng.exponential(size=GRID.n_cells)
        w0 /= w0.sum()
        s0 = CoronalState(
            emissivity=w0,
            orientations=vertical_orientations(GRID.n_cells),
        )
        t0 = time.time()
        res = QCORONAOptimizer(H_obj, opt_cfg).optimize(initial_state=s0)
        elapsed = time.time() - t0
        d = res.final_H.diagnostics
        H_tot = float(res.final_H.H_total)
        print(
            f"  seed {seed}: H={H_tot:.4f}  "
            f"lag={d['delta_t_pred']:.2f}  ({elapsed:.0f}s)"
        )
        H_hist = np.array([h['H_total'] for h in res.history], dtype=float)
        all_H_histories.append(H_hist)
        if H_tot < best_H:
            best_H = H_tot
            best_idx = seed
            best_w = res.final_state.emissivity.copy()
            best_d = dict(d)

    R_centroid = float(np.dot(best_w, GRID.R)) / float(np.sum(best_w))
    z_centroid = float(np.dot(best_w, GRID.z)) / float(np.sum(best_w))
    lag_pred = float(best_d['delta_t_pred'])

    print(f"  BEST seed {best_idx}: H={best_H:.4f}  lag_pred={lag_pred:.2f}")
    print(f"  centroid: R={R_centroid:.2f} rg, z={z_centroid:.2f} rg")

    tf_pred = compute_transfer_function_full(
        GRID, DISK, best_w, SA,
        observer_inclination=epoch['incl'],
        n_bins=N_BINS_TF,
        t_min=float(t_bins[0]),
        t_max=float(t_bins[-1]),
        lag_matrix=LM,
    )

    # pad histories to same length for saving
    max_len = max(len(h) for h in all_H_histories)
    H_mat = np.full((N_STARTS, max_len), np.nan)
    for i, h in enumerate(all_H_histories):
        H_mat[i, :len(h)] = h

    result = {
        'year': year,
        'obsid': epoch['obsid'],
        'state': epoch['state'],
        'L_X': float(epoch['L_X']),
        'R_obs': float(epoch['R_obs']),
        'incl': float(epoch['incl']),
        'lag_rg': lag_rg,
        'lag_err': lag_err,
        'lag_pred': lag_pred,
        'R_centroid': R_centroid,
        'z_centroid': z_centroid,
        'H_final': best_H,
        'I_pred': float(best_d['I_pred']),
        'best_idx': best_idx,
        't_c': t_c,
        'psi_obs': psi_obs,
        'psi_pred': np.asarray(tf_pred.psi, dtype=float),
        'w_best': best_w,
        'H_histories': H_mat,
        'GRID_R': np.asarray(GRID.R, dtype=float),
        'GRID_z': np.asarray(GRID.z, dtype=float),
        'n_r': GRID.n_r,
        'n_z': GRID.n_z,
        'n_phi': GRID.n_phi,
    }
    return result


def save_epoch(result: dict) -> None:
    year = result['year']
    npz_path = epoch_npz_path(year)
    np.savez_compressed(
        npz_path,
        year=result['year'],
        lag_rg=result['lag_rg'],
        lag_err=result['lag_err'],
        lag_pred=result['lag_pred'],
        R_centroid=result['R_centroid'],
        z_centroid=result['z_centroid'],
        H_final=result['H_final'],
        I_pred=result['I_pred'],
        best_idx=result['best_idx'],
        L_X=result['L_X'],
        t_c=result['t_c'],
        psi_obs=result['psi_obs'],
        psi_pred=result['psi_pred'],
        w_best=result['w_best'],
        H_histories=result['H_histories'],
        GRID_R=result['GRID_R'],
        GRID_z=result['GRID_z'],
        n_r=result['n_r'],
        n_z=result['n_z'],
        n_phi=result['n_phi'],
        obsid=result['obsid'],
        state=result['state'],
    )
    print(f"  Saved data: {npz_path}")

    # JSON one-liner summary
    summary = {
        'year': result['year'],
        'obsid': result['obsid'],
        'state': result['state'],
        'L_X': result['L_X'],
        'lag_rg': result['lag_rg'],
        'lag_err': result['lag_err'],
        'lag_pred': result['lag_pred'],
        'R_centroid': result['R_centroid'],
        'z_centroid': result['z_centroid'],
        'H_final': result['H_final'],
        'I_pred': result['I_pred'],
        'best_idx': result['best_idx'],
    }
    json_path = OUT_DIR / f'epoch_{year}_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary: {json_path}")


def plot_epoch(result: dict) -> None:
    year = result['year']
    state = result['state']
    color = COLOR_MAP.get(state, 'k')
    n_r, n_z, n_phi = result['n_r'], result['n_z'], result['n_phi']
    w = result['w_best']
    mmap = w.reshape(n_r, n_z, n_phi).sum(axis=2)

    R_cells = result['GRID_R']
    z_cells = result['GRID_z']
    R_c = np.unique(np.round(R_cells, 8))[:n_r]
    z_c = np.unique(np.round(z_cells, 8))[:n_z]

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.09, right=0.97, top=0.86, bottom=0.14, wspace=0.45,
    )

    ax0 = fig.add_subplot(gs[0, 0])
    H_mat = result['H_histories']
    for i in range(H_mat.shape[0]):
        h = H_mat[i]
        h = h[~np.isnan(h)]
        ax0.semilogy(
            np.abs(h), color=color,
            alpha=1.0 if i == result['best_idx'] else 0.25,
            lw=1.3 if i == result['best_idx'] else 0.6,
        )
    ax0.set_xlabel('Iteration')
    ax0.set_ylabel(r'$|\mathcal{H}|$')
    ax0.set_title(f'{year} ({state})')
    ax0.tick_params(labelsize=7)

    ax1 = fig.add_subplot(gs[0, 1])
    im = ax1.pcolormesh(
        R_c, z_c, mmap.T, cmap='hot_r', shading='auto', rasterized=True,
    )
    ax1.plot(
        result['R_centroid'], result['z_centroid'],
        'c+', ms=12, mew=1.5,
    )
    ax1.set_xlabel(r'$R\;[r_g]$')
    ax1.set_ylabel(r'$z\;[r_g]$')
    ax1.set_title('Recovered corona')
    fig.colorbar(im, ax=ax1, fraction=0.04, pad=0.02).ax.tick_params(labelsize=6)
    ax1.text(
        0.05, 0.95,
        f"$R_c={result['R_centroid']:.1f}$\n$z_c={result['z_centroid']:.1f}$",
        transform=ax1.transAxes, va='top', fontsize=7,
        bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.85),
    )
    ax1.tick_params(labelsize=7)

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.fill_between(result['t_c'], result['psi_obs'], alpha=0.18, color='#d6604d')
    ax2.plot(result['t_c'], result['psi_obs'], color='#d6604d', lw=1.3, label=r'$\psi_{\rm obs}$')
    ax2.plot(
        result['t_c'], result['psi_pred'], color='#2166ac',
        lw=1.1, ls='--', label=r'$\psi_{\rm pred}$',
    )
    ax2.set_xlabel(r'$\Delta t\;[r_g/c]$')
    ax2.set_ylabel(r'$\psi(t)$')
    ax2.set_title('Transfer function')
    ax2.legend(fontsize=7)
    ax2.set_ylim(bottom=0)
    ax2.tick_params(labelsize=7)
    ax2.text(
        0.97, 0.93,
        f"{result['lag_pred']:.2f} vs {result['lag_rg']:.2f}",
        transform=ax2.transAxes, ha='right', va='top', fontsize=7,
    )

    fig.suptitle(
        f"Mrk 335 — {year} ({state})  lag_pred={result['lag_pred']:.2f} $r_g/c$",
        fontsize=10,
    )

    base = OUT_DIR / f'fig_epoch_{year}'
    fig.savefig(f'{base}.pdf', bbox_inches='tight', facecolor='white')
    fig.savefig(f'{base}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved figures: {base}.pdf  and  {base}.png")


def update_summary_csv() -> None:
    """Rilegge tutti gli epoch_*.npz e riscrive summary.csv."""
    rows = []
    for ep in EPOCHS:
        p = epoch_npz_path(ep['year'])
        if not p.exists():
            continue
        d = np.load(p, allow_pickle=True)
        rows.append({
            'year': int(d['year']),
            'obsid': str(d['obsid']),
            'state': str(d['state']),
            'L_X': float(d['L_X']),
            'lag_rg': float(d['lag_rg']),
            'lag_err': float(d['lag_err']),
            'lag_pred': float(d['lag_pred']),
            'R_centroid': float(d['R_centroid']),
            'z_centroid': float(d['z_centroid']),
            'H_final': float(d['H_final']),
            'I_pred': float(d['I_pred']),
        })
    if not rows:
        return
    csv_path = OUT_DIR / 'summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nUpdated {csv_path}")
    print(f"{'Year':<6} {'State':<14} {'R_c':>8} {'z_c':>8} {'lag_pred':>10}")
    print("-" * 50)
    for r in rows:
        print(
            f"{r['year']:<6} {r['state']:<14} "
            f"{r['R_centroid']:>8.2f} {r['z_centroid']:>8.2f} "
            f"{r['lag_pred']:>10.2f}"
        )


def plot_evolution_if_ready() -> None:
    """Figura evoluzione se ≥2 epoche completate."""
    loaded = []
    for ep in EPOCHS:
        p = epoch_npz_path(ep['year'])
        if p.exists():
            d = np.load(p, allow_pickle=True)
            loaded.append({
                'year': int(d['year']),
                'state': str(d['state']),
                'L_X': float(d['L_X']),
                'R_centroid': float(d['R_centroid']),
                'z_centroid': float(d['z_centroid']),
            })
    if len(loaded) < 2:
        return

    fig2, axes = plt.subplots(1, 3, figsize=(9, 3.5))
    L_X = [r['L_X'] for r in loaded]
    R_c = [r['R_centroid'] for r in loaded]
    z_c = [r['z_centroid'] for r in loaded]
    colors = [COLOR_MAP.get(r['state'], '#888') for r in loaded]

    axes[0].scatter(np.log10(L_X), R_c, c=colors, s=60, zorder=3)
    axes[0].set_xlabel(r'$\log_{10}(L_X)$')
    axes[0].set_ylabel(r'$R_{\rm centroid}\;[r_g]$')
    axes[0].set_title('Radial extent vs luminosity')

    axes[1].scatter(np.log10(L_X), z_c, c=colors, s=60, zorder=3)
    axes[1].set_xlabel(r'$\log_{10}(L_X)$')
    axes[1].set_ylabel(r'$z_{\rm centroid}\;[r_g]$')
    axes[1].set_title('Height vs luminosity')

    axes[2].scatter(R_c, z_c, c=colors, s=60, zorder=3)
    for r in loaded:
        axes[2].annotate(
            str(r['year']), (r['R_centroid'], r['z_centroid']),
            fontsize=7, ha='left',
        )
    axes[2].set_xlabel(r'$R_{\rm centroid}\;[r_g]$')
    axes[2].set_ylabel(r'$z_{\rm centroid}\;[r_g]$')
    axes[2].set_title('Coronal geometry track')

    for ax in axes:
        ax.tick_params(labelsize=8)
    fig2.tight_layout()
    base = OUT_DIR / 'fig_evolution'
    fig2.savefig(f'{base}.pdf', bbox_inches='tight', facecolor='white')
    fig2.savefig(f'{base}.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    print(f"Saved evolution: {base}.pdf / .png")


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("Mrk 335 multi-epoch — ONE EPOCH AT A TIME")
    print(f"Output dir: {OUT_DIR.resolve()}")
    print("=" * 60)

    # opzionale: forza rifare un anno  →  python multi_single.py 2009
    force_year = None
    if len(sys.argv) > 1:
        force_year = int(sys.argv[1])
        print(f"Force re-run year: {force_year}")

    for epoch in EPOCHS:
        year = epoch['year']
        npz = epoch_npz_path(year)

        if npz.exists() and force_year != year:
            print(f"\n[skip] {year} already done → {npz.name}")
            continue

        result = run_one_epoch_full(epoch)
        if result is None:
            continue

        save_epoch(result)
        plot_epoch(result)
        update_summary_csv()
        plot_evolution_if_ready()

        print(f"\n>>> Epoch {year} COMPLETE. Puoi interrompere e rilanciare: "
              f"le altre epoche continueranno da qui.")

    update_summary_csv()
    plot_evolution_if_ready()
    print("\nAll requested epochs processed.")