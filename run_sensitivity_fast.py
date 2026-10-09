"""
run_sensitivity_fast.py -- reduced HAMCOR hyperparameter scan (same setup as run_recovery.py).

Run ONE parameter per terminal (3 terminals in parallel):
    python run_sensitivity_fast.py alpha
    python run_sensitivity_fast.py beta
    python run_sensitivity_fast.py J
Then, when all three are done:
    python run_sensitivity_fast.py plot

Each point is saved immediately to results/sens_fast_<param>.npz (a restart skips finished points).
Optional env vars:  SENS_SEEDS (default 1)   SENS_ITER (default 400)
"""
import sys, os, time
import numpy as np
sys.path.insert(0, '.')

MODE = sys.argv[1] if len(sys.argv) > 1 else 'alpha'
N_SEEDS  = int(os.environ.get('SENS_SEEDS', 1))
MAX_ITER = int(os.environ.get('SENS_ITER', 400))

REF = dict(alpha=10.0, beta=10.0, J=0.05)
VALS = dict(alpha=[1.0, 10.0, 100.0],
            beta =[1.0, 10.0, 100.0],
            J    =[0.005, 0.05, 1.0])
GEOMS = ['lamppost', 'column', 'ring']
METRICS = ('corr', 'overlap', 'lag_err', 'illum_err', 'H_final')


def plot_all():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    COL = {'lamppost': '#2166ac', 'column': '#d6604d', 'ring': '#4dac26'}
    LAB = {'lamppost': 'Lamp-post', 'column': 'Column', 'ring': 'Ring'}
    MK = {'lamppost': 'o', 'column': 's', 'ring': '^'}
    XL = {'alpha': r'$\alpha$', 'beta': r'$\beta$', 'J': r'$J$'}
    YL = [('corr', r'Correlation $\rho$'), ('overlap', 'Overlap'), ('lag_err', 'Lag error')]
    fig, axs = plt.subplots(3, 3, figsize=(7.2, 6.2))
    for c, p in enumerate(['alpha', 'beta', 'J']):
        d = np.load(f'results/sens_fast_{p}.npz')
        for r, (m, yl) in enumerate(YL):
            ax = axs[r, c]
            for g in GEOMS:
                ax.plot(d['vals'], d[f'{g}_{m}'], color=COL[g], marker=MK[g], ms=4,
                        label=LAB[g] if (r == 0 and c == 0) else None)
            ax.axvline(REF[p], color='k', ls='--', lw=0.8, alpha=0.5)
            ax.set_xscale('log'); ax.set_xlabel(XL[p])
            if c == 0: ax.set_ylabel(yl)
            if m != 'lag_err': ax.set_ylim(0, 1.05)
            ax.grid(axis='y', lw=0.4, alpha=0.4)
    axs[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig('results/fig_sensitivity.pdf', bbox_inches='tight', facecolor='white')
    print('Saved results/fig_sensitivity.pdf')
    for p in ['alpha', 'beta', 'J']:
        d = np.load(f'results/sens_fast_{p}.npz')
        print(p, list(d['vals']))
        for g in GEOMS:
            print('  ', g, 'rho=', np.round(d[f'{g}_corr'], 3), 'O=', np.round(d[f'{g}_overlap'], 3))


if MODE == 'plot':
    plot_all(); sys.exit(0)
assert MODE in VALS, "usage: python run_sensitivity_fast.py alpha|beta|J|plot"

from qcorona.geometry.grid import build_grid, build_disk_grid
from qcorona.hamiltonian.total import (QCORONAHamiltonian, HamiltonianConfig,
                                       ObservationalConstraints, CoronalState)
from qcorona.optimization.classical import QCORONAOptimizer, OptimizationConfig
from qcorona.data.synthetic import (create_synthetic_geometry, emissivity_correlation,
                                    emissivity_overlap, centroid_distance)
from qcorona.hamiltonian.terms import vertical_orientations

# production grid + recovery geometries (identical to run_recovery.py)
GRID = build_grid(n_r=10, n_z=10, n_phi=8, r_in=3.0, r_out=15.0, z_min=0.1, z_max=5.0)
DISK = build_disk_grid(n_r=30, n_phi=18)
GEOM_KW = {
    'lamppost': dict(height=2.5, width=0.8),
    'column':   dict(z_min=0.5, z_max=4.5, radius=5.0),
    'ring':     dict(R_center=8.0, z_center=2.0, R_width=2.0, z_width=1.0),
}
SYN = {g: create_synthetic_geometry(GRID, DISK, g, **kw) for g, kw in GEOM_KW.items()}

OPT_CFG = OptimizationConfig(max_iter=MAX_ITER, lr_emissivity=0.01, lr_orientation=0.01,
                             use_adaptive_lr=True, use_continuation=True,
                             continuation_steps=200, convergence_tol=1e-7, verbose=False)


def make_H(g, alpha, beta, J):
    syn = SYN[g]
    cons = ObservationalConstraints(
        delta_t_obs=syn.delta_t_true, R_obs=syn.I_true / (1 - syn.I_true + 1e-10),
        L_X=1e43, M_bh=1e8, eddington_ratio=0.1, observer_inclination=30.0,
        source_name=f"synthetic_{g}")
    cfg = HamiltonianConfig(J=J, alpha=alpha, beta=beta, gamma=1.0, l_crit=1000.0,
                            delta=1.0, f_max=1.0, use_pair_barrier=False,
                            use_energy_barrier=False)
    return QCORONAHamiltonian(GRID, DISK, cons, cfg)


def run_one(g, alpha, beta, J):
    Hobj, syn = make_H(g, alpha, beta, J), SYN[g]
    best_H, best_w, best_d = np.inf, None, None
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        w0 = rng.exponential(size=GRID.n_cells); w0 /= w0.sum()
        s0 = CoronalState(emissivity=w0, orientations=vertical_orientations(GRID.n_cells))
        res = QCORONAOptimizer(Hobj, OPT_CFG).optimize(initial_state=s0)
        if res.final_H.H_total < best_H:
            best_H, best_w, best_d = res.final_H.H_total, res.final_state.emissivity, res.final_H.diagnostics
    return dict(
        corr=float(emissivity_correlation(syn.emissivity, best_w)),
        overlap=float(emissivity_overlap(syn.emissivity, best_w)),
        lag_err=abs(best_d['delta_t_pred'] - syn.delta_t_true) / (syn.delta_t_true + 1e-10),
        illum_err=abs(best_d['I_pred'] - syn.I_true) / (syn.I_true + 1e-10),
        H_final=best_H)


os.makedirs('results', exist_ok=True)
path = f'results/sens_fast_{MODE}.npz'
vals = VALS[MODE]
out = {f'{g}_{m}': np.full(len(vals), np.nan) for g in GEOMS for m in METRICS}
if os.path.exists(path):                      # resume
    old = np.load(path)
    for k in out:
        out[k] = old[k].copy()
print(f"Scan {MODE}: {vals}  (others at ref {REF}); seeds={N_SEEDS}, iter={MAX_ITER}", flush=True)

for i, v in enumerate(vals):
    if all(np.isfinite(out[f'{g}_corr'][i]) for g in GEOMS):
        print(f"  {MODE}={v} already done, skipping", flush=True); continue
    pars = dict(REF); pars[MODE] = v
    for g in GEOMS:
        t0 = time.time()
        r = run_one(g, pars['alpha'], pars['beta'], pars['J'])
        for m in METRICS:
            out[f'{g}_{m}'][i] = r[m]
        print(f"  {MODE}={v:<7g} {g:<9} rho={r['corr']:+.3f} O={r['overlap']:.3f} "
              f"lag_err={r['lag_err']*100:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
        np.savez(path, vals=np.array(vals), **out)   # save after every geometry
print("Done.", path)
