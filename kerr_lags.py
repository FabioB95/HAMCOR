"""
kerr_lags.py
─────────────
Lag matrix Kerr:
  LM[i,k] = t_ck + t_kobs - t_iobs

t_iobs da reciprocita': punti lungo i raggi observer→BH,
interpolazione SOLO sullo stesso ray_id (niente mix tra raggi).
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

from kerr_geodesics import integrate_geodesic


def build_observer_ray_table_with_trajectories(
    a, incl_deg, n_alpha=100, n_beta=100, screen_size=25.0, r_obs=200.0,
):
    """
    Ritorna:
      disk_hits: (N_hit, 5)  alpha, beta, r_disk, phi_disk, t_hit
      all_points: (N_pts, 5) r, theta, phi, t, ray_id
    """
    incl = np.radians(incl_deg)
    alphas = np.linspace(-screen_size, screen_size, n_alpha)
    betas = np.linspace(-screen_size, screen_size, n_beta)

    all_points = []
    disk_hits = []
    status_count = {}
    ray_id = 0

    for alpha in alphas:
        for beta in betas:
            Lz = -alpha * np.sin(incl)
            Q = beta**2 + (alpha**2 - a**2) * np.cos(incl)**2
            sr0 = -1
            sth0 = 1 if beta >= 0 else -1

            res = integrate_geodesic(
                r_obs, incl, 0.0, a, 1.0, Lz, Q, sr0, sth0,
                lam_max=5.0 * r_obs,
                r_max=2.0 * r_obs,
                max_bounces=80,
                terminal_at_disk=True,
                save_trajectory=True,
            )
            status_count[res['status']] = status_count.get(res['status'], 0) + 1

            if res.get('trajectory'):
                for (r, th, ph, t) in res['trajectory']:
                    all_points.append((r, th, ph, t, ray_id))

            if res['status'] == 'disk_hit' and 1.5 < res['r'] < 100.0:
                disk_hits.append([
                    alpha, beta,
                    float(res['r']),
                    float(res['phi']) % (2 * np.pi),
                    abs(float(res['t'])),
                ])
            ray_id += 1

    print(f"  ray status counts: {status_count}")
    print(f"  hits disco: {len(disk_hits)}")
    print(f"  punti traiettoria: {len(all_points)}  (rays={ray_id})")

    hits = np.array(disk_hits) if disk_hits else np.zeros((0, 5))
    pts = np.array(all_points) if all_points else np.zeros((0, 5))
    return hits, pts


def build_observer_ray_table(a, incl_deg, **kwargs):
    hits, _ = build_observer_ray_table_with_trajectories(a, incl_deg, **kwargs)
    return hits


def build_ti_obs_lookup(all_trajectory_points):
    if len(all_trajectory_points) == 0:
        return None, None, None, None

    r = all_trajectory_points[:, 0]
    th = all_trajectory_points[:, 1]
    phi = all_trajectory_points[:, 2]
    t = all_trajectory_points[:, 3]
    rid = all_trajectory_points[:, 4].astype(int)

    x = r * np.sin(th) * np.cos(phi)
    y = r * np.sin(th) * np.sin(phi)
    z = r * np.cos(th)
    xyz = np.column_stack([x, y, z])
    tree = cKDTree(xyz)
    return tree, t, rid, xyz


def lookup_ti_obs(
    tree, t_values, ray_ids, xyz_all,
    r_c, theta_c, phi_c,
    max_dist=1.0, k_neighbors=8,
):
    if tree is None:
        return None

    x = r_c * np.sin(theta_c) * np.cos(phi_c)
    y = r_c * np.sin(theta_c) * np.sin(phi_c)
    z = r_c * np.cos(theta_c)
    query = np.array([x, y, z])

    dists, idxs = tree.query(query, k=min(k_neighbors, len(t_values)))
    dists = np.atleast_1d(dists)
    idxs = np.atleast_1d(idxs)

    if dists[0] > max_dist:
        return None

    best_ray = ray_ids[idxs[0]]
    same_ray_mask = ray_ids[idxs] == best_ray
    idxs_ray = idxs[same_ray_mask]
    dists_ray = dists[same_ray_mask]

    if len(idxs_ray) < 2:
        return float(abs(t_values[idxs[0]]))

    order = np.argsort(dists_ray)[:2]
    i1, i2 = idxs_ray[order[0]], idxs_ray[order[1]]
    d1, d2 = dists_ray[order[0]], dists_ray[order[1]]
    w1 = d2 / (d1 + d2 + 1e-12)
    w2 = d1 / (d1 + d2 + 1e-12)
    return float(abs(w1 * t_values[i1] + w2 * t_values[i2]))



def corona_to_disk_time(r_c, theta_c, phi_c, r_d, phi_d, a, n_guesses=8, return_score=False):
    def try_params(Lz, Q, sr0, sth0):
        res = integrate_geodesic(
            r_c, theta_c, phi_c, a, 1.0, Lz, Q, sr0, sth0,
            lam_max=max(3000.0, 20.0*r_c), r_max=max(300.0, 5.0*r_c),
            max_bounces=40, terminal_at_disk=True,
        )
        if res['status'] != 'disk_hit':
            return None
        r_hit, phi_hit = res['r'], res['phi']
        if r_hit < 1.2:
            return None
        dphi = (phi_hit - phi_d + np.pi) % (2*np.pi) - np.pi
        pos_score = abs(r_hit - r_d) + 0.5*abs(dphi)
        n_bounce = res.get('n_bounce', 0)
        return (r_hit, phi_hit, res['t'], pos_score, n_bounce)

    def residual(params, sr0, sth0):
        hit = try_params(params[0], params[1], sr0, sth0)
        if hit is None:
            return [1e3, 1e3]
        r_hit, phi_hit, _, _, _ = hit
        dphi = (phi_hit - phi_d + np.pi) % (2*np.pi) - np.pi
        return [r_hit - r_d, dphi]

    Lz_scale = max(1.0, r_c)
    Q_scale = max(1.0, r_d**2)
    Lz_grid = np.linspace(-2*Lz_scale, 2*Lz_scale, 9)
    Q_grid = np.linspace(0.05*Q_scale, 2*Q_scale, 9)

    candidates = []
    for sr0, sth0 in [(-1,1), (-1,-1), (1,1), (1,-1)]:
        for Lz in Lz_grid:
            for Q in Q_grid:
                hit = try_params(Lz, Q, sr0, sth0)
                if hit is not None and hit[3] < 0.3:
                    candidates.append((Lz, Q, sr0, sth0, hit))

    if not candidates:
        return None

    for Lz0, Q0, sr0, sth0, hit0 in sorted(candidates, key=lambda c: c[4][4]):
        try:
            sol1 = least_squares(residual, [Lz0, Q0], args=(sr0, sth0), method='lm',
                                  max_nfev=100, ftol=1e-6, xtol=1e-6)
            sol2 = least_squares(residual, sol1.x, args=(sr0, sth0), method='lm',
                                  max_nfev=300, ftol=1e-11, xtol=1e-11)
            hit = try_params(sol2.x[0], sol2.x[1], sr0, sth0)
        except Exception:
            hit = None
        if hit is not None and hit[3] < 0.05:
            if return_score:
                return (float(hit[2]), float(hit[3]))
            return float(hit[2])

    return None



def compute_lag_matrix_kerr(
    grid, disk, a, observer_inclination, obs_table=None, verbose=True,
):
    if verbose:
        print("  Costruzione tabella + traiettorie (reciprocita', same-ray interp)...")
    obs_hits, all_traj = build_observer_ray_table_with_trajectories(
        a, observer_inclination,
        n_alpha=100, n_beta=100,
    )
    if obs_table is not None:
        obs_hits = obs_table

    N, M = grid.n_cells, disk.n_elements
    LM = np.full((N, M), np.nan)

    if len(obs_hits) == 0:
        if verbose:
            print("  ATTENZIONE: obs_table vuota.")
        return LM, obs_hits

    tree_disk = cKDTree(obs_hits[:, 2:4])
    tree_ti, t_values, ray_ids, xyz_all = build_ti_obs_lookup(all_traj)

    t_i_obs_cache = {}
    n_direct_ok = 0
    for i in range(N):
        Ri = float(grid.R[i])
        zi = float(grid.z[i])
        phii = float(grid.phi[i])
        r_c = np.sqrt(Ri**2 + zi**2)
        theta_c = np.arctan2(Ri, zi) if zi > 0 else np.pi / 2
        t_i = lookup_ti_obs(
            tree_ti, t_values, ray_ids, xyz_all,
            r_c, theta_c, phii,
            max_dist=1.0, k_neighbors=8,
        )
        t_i_obs_cache[i] = t_i
        if t_i is not None:
            n_direct_ok += 1

    if verbose:
        print(f"  t_i_obs da reciprocita' (same-ray): {n_direct_ok}/{N}")

    for i in range(N):
        if verbose and i % 10 == 0:
            print(f"  corona cell {i}/{N}", end='\r', flush=True)
        t_i_obs = t_i_obs_cache[i]
        if t_i_obs is None:
            continue

        Ri = float(grid.R[i])
        zi = float(grid.z[i])
        phii = float(grid.phi[i])
        r_c = np.sqrt(Ri**2 + zi**2)
        theta_c = np.arctan2(Ri, zi) if zi > 0 else np.pi / 2

        for k in range(M):
            Rk = float(disk.R[k])
            phik = float(disk.phi[k])
            t_ck = corona_to_disk_time(r_c, theta_c, phii, Rk, phik, a)
            if t_ck is None:
                continue
            _, idx = tree_disk.query([Rk, phik % (2 * np.pi)])
            t_k_obs = float(obs_hits[idx, 4])
            lag = t_ck + t_k_obs - t_i_obs
            LM[i, k] = max(0.0, lag)

    if verbose:
        print()
    return LM, obs_hits