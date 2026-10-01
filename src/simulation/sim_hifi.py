#!/usr/bin/env python3
"""sim_hifi.py -- high-fidelity plant study (root revision).

Explicitly OUTSIDE the certified model: this module runs the same
two-stage cascade, feedforward, filter, RULE, and latch logic on a
plant with sampled control, actuator dynamics, resistance, grade,
adhesion derating, and sensing errors, to measure whether the
ideal-model certificate remains empirically conservative.  Nothing
here feeds the certificate; every parameter below is a plausible
sensitivity value, not an identified vehicle model, and is labeled
as such in the manuscript.

Realism layers (each individually toggleable):
  T_s        local control/QP sample-and-hold period [s] (ZOH)
  tau_act    first-order traction/brake actuator lag [s]
  jerk_max   command slew limit [m/s^3]
  davis      resistance a0 + a1 v + a2 v^2 [m/s^2 equivalent]
  grade      per-unit constant grade acceleration [m/s^2]
  derate     multiplicative braking-capability factor (adhesion)
  noise_gap / noise_vel   zero-mean Gaussian sensing noise (std)
  bias_gap   constant ranging bias [m]
  quant_vel  odometry quantization step [m/s]
  v_stop     standstill threshold with hysteresis [m/s]
Plant integration stays at 1 ms; control updates every T_s.
Contact terminates the run.
"""
import numpy as np


DEFAULT = dict(T_s=0.010, tau_act=0.30, jerk_max=0.60,
               davis=(0.005, 0.0004, 0.00012), grade=0.003,
               derate=0.90, noise_gap=0.05, noise_vel=0.02,
               bias_gap=0.02, quant_vel=0.01, v_stop=0.05,
               # implementation-level compensations (review E5,
               # 2026-09-07; all default OFF, none certified):
               #   grade_ff  fraction of the local grade fed forward
               #             (track-database knowledge; 1 = exact)
               #   res_ff    fraction of the Davis resistance fed
               #             forward from the own measured speed
               #   ramp_lead lead [s] applied to the head's stored
               #             profile for command purposes only
               #             (compensates actuator lag and jerk
               #             build-up; markers unchanged)
               # with compensation on, the transmitted predecessor
               # command is the intended realized acceleration
               # (held command minus the unit's own compensation)
               grade_ff=0.0, res_ff=0.0, ramp_lead=0.0)


def run(P, d_s, v0, e0, hifi=None, seed=0, T_pad=40.0):
    H = dict(DEFAULT, **(hifi or {}))
    rng = np.random.default_rng(seed)
    n = P['n']; dt = P['dt']
    N = int((P['T_END'] + T_pad) / dt)
    a_b, v_xi, T_xi = P['a_b'], P['v_xi'], P['T_xi']
    alpha, beta, gamma = P['alpha'], P['beta'], P['gamma']
    phi, eps_det = P['phi'], P['eps_det']
    ell, s_m, kappa, U = P['ell'], P['s_m'], P['kappa'], P['U']
    A = np.array(P['amax'], float) * H['derate']
    b = np.array(P['amax'], float) / P['v_c']
    Umin = np.minimum(U, A)
    # per-unit actuator lag (scalar broadcast; heterogeneous build-up
    # is passed as an n-vector); a zero lag is an ideal actuator
    tau_act = np.broadcast_to(np.asarray(H['tau_act'], float),
                              (n,)).astype(float)
    filt_events = 0

    d_s = np.array(d_s, float); vv = np.array(v0, float).copy()
    s = -np.cumsum(ell + d_s + np.array(e0, float)); sref = 0.0
    mark_ref = v_xi * T_xi + v_xi ** 2 / (2 * a_b)
    marks = mark_ref - np.cumsum(ell + d_s)

    delay = int(round(P['dbar'] / dt))
    qu = [np.zeros(N + delay + 2) for _ in range(n)]
    qf = [np.zeros(N + delay + 2, bool) for _ in range(n)]
    mode = np.zeros(n, int); mode[0] = 1
    stopped = np.zeros(n, bool)
    tstop = np.full(n, np.nan)
    u_cmd = np.zeros(n)          # held (ZOH) commanded acceleration
    u_act = np.zeros(n)          # actuator state (realized accel.)
    gmin = np.full(n - 1, 1e9)
    upk = np.zeros(n); jpk = np.zeros(n)
    hold = int(round(H['T_s'] / dt))
    contact_time = None
    grade_acc = H['grade'] * np.array([1, 1, -1, -1, 1][:n])
    comp = np.zeros(n)           # own compensation added to the command
    lead = float(H['ramp_lead'])

    for k in range(N):
        t = k * dt
        vref = v_xi if t < T_xi else max(0.0, v_xi - a_b * (t - T_xi))
        uref = 0.0 if t < T_xi else (-a_b if vref > 1e-12 else 0.0)
        # led profile for the head's command only (markers unchanged)
        tl = t + lead
        vref_l = v_xi if tl < T_xi else max(0.0, v_xi - a_b * (tl - T_xi))
        uref_l = 0.0 if tl < T_xi else (-a_b if vref_l > 1e-12 else 0.0)
        if k % hold == 0:                      # sampled control + QP
            for i in range(n):
                vp = vref_l if i == 0 else vv[i - 1]
                sp = sref if i == 0 else s[i - 1]
                # sensing errors
                gap_m = (sp - s[i] - ell) + H['bias_gap'] \
                    + rng.normal(0.0, H['noise_gap'])
                vi_m = np.round((vv[i] + rng.normal(0.0, H['noise_vel']))
                                / H['quant_vel']) * H['quant_vel']
                vp_m = vp if i == 0 else np.round(
                    (vp + rng.normal(0.0, H['noise_vel']))
                    / H['quant_vel']) * H['quant_vel']
                eps = vp_m - vi_m; e = gap_m - d_s[i]
                uh = uref_l if i == 0 else qu[i][k]
                fl = True if i == 0 else qf[i][k]
                if mode[i] == 0 and fl:
                    mode[i] = 1
                m = mode[i]
                if m == 1 and abs(eps) <= eps_det:
                    mode[i] = 2; m = 2
                if stopped[i]:
                    un = 0.0
                elif m == 3:
                    un = -a_b
                elif m == 2:
                    un = uh + beta * e + gamma * eps
                elif m == 1:
                    un = uh + alpha * np.clip(eps / phi, -1, 1)
                else:
                    un = uh
                # standstill threshold with hysteresis
                if m in (2, 3) and not stopped[i] \
                        and vi_m <= H['v_stop'] and t > T_xi:
                    stopped[i] = True; tstop[i] = t; un = 0.0
                elif m == 2 and (vp_m <= eps_det if i > 0
                                 else vref <= 1e-12) and t > T_xi:
                    mode[i] = 3; un = -a_b
                # sequential delayed-surrogate filter (sampled)
                if i >= 1 and not stopped[i]:
                    g = gap_m - s_m
                    h = g + vp_m / b[i - 1] - vi_m / b[i]
                    ucbf = b[i] * (eps + uh / b[i - 1] + kappa * h)
                    un_f = max(-Umin[i], min(un, ucbf, U))
                    if un_f < un - 1e-12:
                        filt_events += 1
                    un = un_f
                # implementation-level feedforward compensation of the
                # known grade and resistance (review E5; default 0)
                a0, a1, a2 = H['davis']
                comp[i] = H['grade_ff'] * grade_acc[i] + H['res_ff'] * (
                    a0 + a1 * abs(vi_m) + a2 * vi_m ** 2)
                if stopped[i]:
                    comp[i] = 0.0
                un = un + comp[i]
                # jerk limit on the held command
                dumax = H['jerk_max'] * H['T_s']
                un = float(np.clip(un, u_cmd[i] - dumax,
                                   u_cmd[i] + dumax))
                jpk[i] = max(jpk[i], abs(un - u_cmd[i]) / H['T_s'])
                u_cmd[i] = float(np.clip(un, -A[i], A[i]))
        # actuator lag + plant with resistance and grade (1 ms)
        for i in range(n):
            if tau_act[i] > 0.0:
                u_act[i] += dt / tau_act[i] * (u_cmd[i] - u_act[i])
            else:
                u_act[i] = u_cmd[i]
            if stopped[i]:
                continue
            a0, a1, a2 = H['davis']
            acc = u_act[i] - np.sign(vv[i]) * (
                a0 + a1 * vv[i] + a2 * vv[i] ** 2) - grade_acc[i]
            vv[i] = max(0.0, vv[i] + acc * dt)
            s[i] += vv[i] * dt
            upk[i] = max(upk[i], abs(u_act[i]))
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            gmin[i - 1] = min(gmin[i - 1], g)
            if g + s_m <= 0.0 and contact_time is None:
                contact_time = t
        if contact_time is not None:
            break
        for i in range(n - 1):
            qu[i + 1][k + delay] = u_cmd[i] - comp[i]
            qf[i + 1][k + delay] = qf[i + 1][max(0, k + delay - 1)] \
                or (mode[i] >= 2)
        sref += vref * dt

    egap = (np.concatenate(([sref], s[:-1])) - s - ell) - d_s
    return dict(
        tstop=tstop, all_stopped=bool(np.all(stopped)),
        disp=float(np.nanmax(tstop) - np.nanmin(tstop))
        if np.all(np.isfinite(tstop)) else None,
        align=float(np.max(np.abs(s - marks))),
        gaperr=float(np.max(np.abs(egap))),
        gmin=[float(x) for x in gmin], upk=[float(x) for x in upk],
        jerk_pk=[float(x) for x in jpk],
        filt_events=int(filt_events),
        contact_time=contact_time)
