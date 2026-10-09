"""
kerr_geodesics.py
───────────────────
Geodetiche nulle in Kerr (Boyer-Lindquist), formalismo di Carter.
"""

import numpy as np
from scipy.integrate import solve_ivp


def sigma(r, theta, a):
    return r**2 + a**2 * np.cos(theta)**2


def delta(r, a):
    return r**2 - 2.0 * r + a**2


def carter_R(r, E, Lz, Q, a):
    P = E * (r**2 + a**2) - Lz * a
    return P**2 - delta(r, a) * ((Lz - a * E)**2 + Q)


def carter_Theta(theta, E, Lz, Q, a):
    cos2 = np.cos(theta)**2
    sin2 = max(np.sin(theta)**2, 1e-12)
    return Q + cos2 * (a**2 * E**2 - Lz**2 / sin2)


def geodesic_rhs(lam, y, a, E, Lz, Q, sr, sth):
    r, theta, phi, t = y
    Sig = max(sigma(r, theta, a), 1e-12)
    Del = max(delta(r, a), 1e-12)
    sin2 = max(np.sin(theta)**2, 1e-12)

    R_val = max(carter_R(r, E, Lz, Q, a), 0.0)
    Th_val = max(carter_Theta(theta, E, Lz, Q, a), 0.0)

    dr_dlam = sr * np.sqrt(R_val) / Sig
    dth_dlam = sth * np.sqrt(Th_val) / Sig

    P = E * (r**2 + a**2) - Lz * a
    dphi_dlam = (-(a * E - Lz / sin2) + a * P / Del) / Sig
    dt_dlam = (-a * (a * E * sin2 - Lz) + (r**2 + a**2) * P / Del) / Sig

    return [dr_dlam, dth_dlam, dphi_dlam, dt_dlam]


def _event_R_zero(lam, y, a, E, Lz, Q, sr, sth):
    return carter_R(y[0], E, Lz, Q, a)


_event_R_zero.terminal = True
_event_R_zero.direction = -1


def _event_Theta_zero(lam, y, a, E, Lz, Q, sr, sth):
    return carter_Theta(y[1], E, Lz, Q, a)


_event_Theta_zero.terminal = True
_event_Theta_zero.direction = -1


def _event_horizon(lam, y, a, E, Lz, Q, sr, sth):
    r_horizon = 1.0 + np.sqrt(max(1.0 - a**2, 0.0)) + 0.05
    return y[0] - r_horizon


_event_horizon.terminal = True
_event_horizon.direction = -1


def _make_escape_event(r_max):
    def _event_escape_bound(lam, y, a, E, Lz, Q, sr, sth):
        return y[0] - r_max

    _event_escape_bound.terminal = True
    _event_escape_bound.direction = 1
    return _event_escape_bound


def _make_equatorial_event(terminal):
    def _event_equatorial_crossing(lam, y, a, E, Lz, Q, sr, sth):
        return y[1] - np.pi / 2

    _event_equatorial_crossing.terminal = terminal
    _event_equatorial_crossing.direction = 0
    return _event_equatorial_crossing


def integrate_geodesic(
    r0, theta0, phi0, a, E, Lz, Q, sr0, sth0,
    lam_max=None, r_max=None, max_bounces=80,
    rtol=1e-8, atol=1e-10,
    terminal_at_disk=True,
    save_trajectory=False,
):
    r0 = float(r0)
    if lam_max is None:
        lam_max = max(5000.0, 5.0 * max(r0, 50.0))
    if r_max is None:
        r_max = max(1e4, 2.0 * r0)

    y = [r0, float(theta0), float(phi0), 0.0]
    lam = 0.0
    sr = int(np.sign(sr0) or 1)
    sth = int(np.sign(sth0) or 1)
    equatorial_crossings = []
    status = 'max_bounces'
    traj = [] if save_trajectory else None
    n_bounce = 0

    escape_event = _make_escape_event(r_max)
    equatorial_event = _make_equatorial_event(terminal_at_disk)

    for _bounce in range(max_bounces):
        events = [
            _event_R_zero,
            _event_Theta_zero,
            equatorial_event,
            _event_horizon,
            escape_event,
        ]

        sol = solve_ivp(
            geodesic_rhs,
            [lam, lam + lam_max],
            y,
            args=(a, E, Lz, Q, sr, sth),
            events=events,
            rtol=rtol,
            atol=atol,
            max_step=lam_max / 200,
            dense_output=False,
        )

        if save_trajectory:
            for j in range(len(sol.t)):
                traj.append((
                    float(sol.y[0][j]),
                    float(sol.y[1][j]),
                    float(sol.y[2][j]),
                    float(sol.y[3][j]),
                ))

        if len(sol.t_events[2]) > 0:
            r_x, th_x, phi_x, t_x = sol.y_events[2][0]
            equatorial_crossings.append(
                (float(r_x), float(phi_x), float(t_x))
            )
            if terminal_at_disk:
                return dict(
                    r=float(r_x),
                    theta=float(th_x),
                    phi=float(phi_x),
                    t=float(t_x),
                    equatorial_crossings=equatorial_crossings,
                    status='disk_hit',
                    trajectory=traj,
                    n_bounce=n_bounce,
                )

        y = sol.y[:, -1].tolist()
        lam = float(sol.t[-1])

        if len(sol.t_events[3]) > 0:
            status = 'horizon'
            break
        if len(sol.t_events[4]) > 0:
            status = 'escaped'
            break
        if len(sol.t_events[0]) > 0:
            sr = -sr
            n_bounce += 1
            continue
        if len(sol.t_events[1]) > 0:
            sth = -sth
            n_bounce += 1
            continue
        status = 'lam_max_reached'
        break

    return dict(
        r=y[0],
        theta=y[1],
        phi=y[2],
        t=y[3],
        equatorial_crossings=equatorial_crossings,
        status=status,
        trajectory=traj,
        n_bounce=n_bounce,
    )