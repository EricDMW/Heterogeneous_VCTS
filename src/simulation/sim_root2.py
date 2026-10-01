#!/usr/bin/env python3
"""sim_root2.py -- parameterized event-level simulator (root revision,
benchmark B2 program).

The closed-loop update sequence is copied operation-for-operation from
the frozen sim_v56.py, generalized to
  * arbitrary design constants (speeds, brake rate, gains, geometry),
  * time-varying FIFO per-hop delay schedules (constant-delay input
    reproduces the frozen queue semantics exactly),
  * controller variants for the baseline study:
      'proposed'  two-stage law with predecessor-command feedforward
      'single'    linear S2 law from activation (no S1 stage)
      'noff'      two-stage law WITHOUT the command feedforward
      'lqr'       clamped pair-state feedback u = K_e e + K_v eps with
                  the documented LQR gains (Q = I, R = 1 on the double
                  integrator: K = [1, sqrt(3)]), no feedforward
  * RULE implementations: 'gapclose' (frozen sim: constant-deceleration
    closure to the design gap, capability-clipped) and 'ab' (the
    theorem's -a_b rule),
  * an explicitly applied sequential delayed-surrogate filter
    (apply_filter=True): u* = clip(min(u_nom, u_cbf), -U^-, U) with
    u_cbf = b_i (eps + uhat/b_{i-1} + kappa h); activation events and
    infeasibility episodes are counted,
  * the emergency / channel-loss scenario with optional watchdog.

validate_featured() reruns the featured case (takeover_v42, constant
0.06 s delays, 'gapclose' RULE, filter monitored not applied) and
asserts bit-identical agreement with the archived sim_v56_traj.npz.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

FEATURED = dict(n=5, dt=1e-3, dbar=0.06, a_b=0.70, v_xi=10.0,
                T_xi=180.0, T_END=200.0, alpha=0.30, beta=0.005625,
                gamma=0.15, phi=0.005, eps_det=0.015, ell=25.0,
                s_m=3.0, kappa=1.0, U=1.6,
                amax=(1.60, 1.48, 1.38, 1.29, 1.21), v_c=1.2)

K_LQR = (1.0, np.sqrt(3.0))     # Q = I, R = 1 double-integrator LQR
# tuned feedforward-PID baseline (documented tuning): PID on the pair
# gap error with the same predecessor-command feedforward; PD part
# tuned like a near-critically-damped closure of the 40 m error, plus
# a small integral with anti-windup, sized so the integral authority
# never exceeds 0.05 m/s^2 during the closure.
K_PID = dict(Kp=0.003, Kd=0.115, Ki=5e-5, Imax=0.05)


K_CACC = dict(h_w=0.5)          # constant-time-headway spacing [s]


def run(P, d_s, v0, e0, delays=None, variant='proposed',
        rule='gapclose', apply_filter=False, scenario=None,
        t_emergency=25.0, emergency_unit=2, T_wd=None,
        filter_slopes='heterogeneous', sample_every=50, record=False,
        terminate_on_contact=False, v_thr=None, emergency_brake=False,
        extra=False):
    """One closed-loop run.  d_s, v0, e0 are per-unit arrays
    extra=True additionally returns diagnostic accumulators that do
    not alter the closed loop (audit revision 2026-09-08): the filter
    inactivity margin m_i = u_cbf - u_nom (minimum per pair and, with
    record=True, its sampled series), the handoff pair errors at
    T_xi, the realized braking-window mismatch integral int|w_i|
    (from T_xi to the pair's first standstill), the braking-window
    velocity-error peak, and the realized ramp-fed command envelope
    max|u_{i-1}+a_b| after the last possible ramp receipt.
    (d_s[0] is the head slot; e0[0] the head pair error).
    delays: None -> constant P['dbar']; else an (n, N) int array of
    per-send delay steps (hop i uses row i, i >= 1); FIFO enforced.
    v_thr: predecessor-standstill detection threshold [m/s] for RULE
    (None -> the detector band eps_det, the frozen behavior).
    variant 'cacc': delay-aware constant-time-headway law (the same
    predecessor-command feedforward, linear S2-type gains from
    activation, spacing target d_s + h_w v_i) -- the same-information
    cooperative-adaptive-cruise-control baseline.
    Returns metrics (and sampled trajectories when record=True)."""
    n = P['n']; dt = P['dt']; N = int(P['T_END'] / dt)
    a_b, v_xi, T_xi = P['a_b'], P['v_xi'], P['T_xi']
    alpha, beta, gamma = P['alpha'], P['beta'], P['gamma']
    phi, eps_det = P['phi'], P['eps_det']
    v_rule = eps_det if v_thr is None else float(v_thr)
    h_w = K_CACC['h_w']
    ell, s_m, kappa, U = P['ell'], P['s_m'], P['kappa'], P['U']
    A = np.array(P['amax'], float); b = A / P['v_c']
    Umin = np.minimum(U, A)
    bf = b.copy()
    if filter_slopes == 'common':
        bf = np.full(n, float(np.round(np.mean(b), 2)))

    d_s = np.array(d_s, float); vv = np.array(v0, float).copy()
    e0 = np.array(e0, float)
    s = -np.cumsum(ell + d_s + e0); sref = 0.0
    mark_ref = v_xi * T_xi + v_xi ** 2 / (2 * a_b)
    marks = mark_ref - np.cumsum(ell + d_s)

    const_delay = delays is None
    if const_delay:
        delay = int(round(P['dbar'] / dt))
        qu = [np.zeros(N + delay + 2) for _ in range(n)]
        qf = [np.zeros(N + delay + 2, bool) for _ in range(n)]
    else:
        delays = np.asarray(delays, int)
        # FIFO arrival schedule per hop: arr[k] = max over j<=k (j+d[j])
        arr = [None] * n
        for i in range(1, n):
            a = np.arange(N) + delays[i][:N]
            arr[i] = np.maximum.accumulate(a)
        rcv_u = np.zeros(n); rcv_f = np.zeros(n, bool)
        ptr = np.zeros(n, int)
        hist_u = np.zeros((n, N)); hist_f = np.zeros((n, N), bool)

    mode = np.zeros(n, int); mode[0] = 1
    aR = np.zeros(n)
    tstop = np.full(n, np.nan); hits = 0
    infeas = 0; act_time = np.zeros(n)
    # --- diagnostics (extra=True only; never feed back into the loop)
    mmin = np.full(n - 1, np.inf); MG = [[] for _ in range(n - 1)]
    wint = np.zeros(n - 1); epsmax_b = np.zeros(n - 1)
    fb_real = np.zeros(n - 1)
    e_hnd = np.full(n, np.nan); eps_hnd = np.full(n, np.nan)
    k_xi = int(round(T_xi / dt))
    dsteps = int(round(P['dbar'] / dt))
    gmin = np.full(n - 1, 1e9); hmin = np.full(n - 1, 1e9)
    upk = np.zeros(n)
    H = [[] for _ in range(n - 1)]; G = [[] for _ in range(n - 1)]
    VT = [[] for _ in range(n)]; ET = [[] for _ in range(n)]
    UT = [[] for _ in range(n)]
    TS = []; tact = np.full(n, np.nan); tact[0] = 0.0
    tset = np.full(n, np.nan)
    uprev = np.zeros(n)
    lost = False
    wd_latch = np.zeros(n, bool)
    t_loss = np.nan
    pid_I = np.zeros(n)
    contact_time = None

    for k in range(N):
        t = k * dt
        if scenario == 'emergency' and not lost and t >= t_emergency:
            lost = True; t_loss = t
        vref = v_xi if t < T_xi else max(0.0, v_xi - a_b * (t - T_xi))
        uref = 0.0 if t < T_xi else (-a_b if vref > 1e-12 else 0.0)
        u = np.zeros(n)
        for i in range(n):
            vp = vref if i == 0 else vv[i - 1]
            sp = sref if i == 0 else s[i - 1]
            eps = vp - vv[i]; gap = sp - s[i] - ell; e = gap - d_s[i]
            if i == 0:
                uh = uref; fl = True
            elif const_delay:
                uh = qu[i][k]; fl = qf[i][k]
            else:
                while ptr[i] < N and arr[i][ptr[i]] <= k:
                    rcv_u[i] = hist_u[i][ptr[i]]
                    rcv_f[i] = rcv_f[i] or hist_f[i][ptr[i]]
                    ptr[i] += 1
                uh = rcv_u[i]; fl = rcv_f[i]
            if lost and i >= 1:
                uh = 0.0                      # stale/zero channel
            if mode[i] == 0 and fl:
                mode[i] = 1 if variant == 'proposed' else 2
                tact[i] = t
                if variant != 'proposed':
                    tset[i] = t
            m = mode[i]
            if m == 1 and abs(eps) <= eps_det:
                mode[i] = 2; tset[i] = t; m = 2
            ff = 0.0 if variant in ('noff', 'lqr') else uh
            if variant == 'pid' and m == 2:
                pid_I[i] = np.clip(pid_I[i] + e * dt,
                                   -K_PID['Imax'] / K_PID['Ki'],
                                   K_PID['Imax'] / K_PID['Ki'])
            if m == 4:
                un = 0.0
            elif m == 3:
                if rule == 'gapclose':
                    un = -aR[i] if aR[i] > 0 else ff + beta * e + gamma * eps
                else:
                    un = -a_b
                if vv[i] <= 1e-9:
                    mode[i] = 4; tstop[i] = t; un = 0.0
            elif m == 2:
                if variant == 'lqr':
                    un = K_LQR[0] * e + K_LQR[1] * eps
                elif variant == 'pid':
                    un = ff + K_PID['Kp'] * e + K_PID['Kd'] * eps \
                        + K_PID['Ki'] * pid_I[i]
                elif variant == 'cacc' and i >= 1:
                    un = ff + beta * (e - h_w * vv[i]) + gamma * eps
                else:
                    un = ff + beta * e + gamma * eps
            elif m == 1:
                un = ff + alpha * np.clip(eps / phi, -1, 1)
            else:
                un = ff
            # LATCH: first standstill during braking
            if m in (2, 3) and vv[i] <= 1e-4 and t > T_xi:
                mode[i] = 4; tstop[i] = t; un = 0.0
            # RULE: predecessor standstill sensed
            elif m == 2 and vp <= (v_rule if i > 0 else 1e-12) \
                    and t > T_xi:
                mode[i] = 3
                if rule == 'gapclose':
                    r = gap - d_s[i]
                    aR[i] = vv[i] ** 2 / (2 * r) if r > 1e-9 else 0.0
                    un = -aR[i] if aR[i] > 0 else un
                else:
                    un = -a_b if vv[i] > 1e-12 else un
            # emergency-braking unit (archived Case-A ablation semantics,
            # sim_record_v56: the unit applies its full capability
            # through the override until arrest and bypasses the
            # service filter); default off keeps the pure channel-loss
            # study of the benchmark program unchanged
            ebrk = bool(lost and emergency_brake and i == emergency_unit)
            if ebrk:
                un = -A[i] if vv[i] > 1e-9 else 0.0
                if vv[i] <= 1e-9 and np.isnan(tstop[i]):
                    tstop[i] = t
            # watchdog (emergency scenario)
            if lost and T_wd is not None and i >= 1 and not ebrk \
                    and t >= t_loss + T_wd:
                wd_latch[i] = True
            if wd_latch[i] and mode[i] != 4:
                un = -A[i]
            # sequential delayed-surrogate filter
            if apply_filter and i >= 1 and mode[i] != 4 and not ebrk:
                g = gap - s_m
                h = g + vp / bf[i - 1] - vv[i] / bf[i]
                ucbf = bf[i] * (eps + uh / bf[i - 1] + kappa * h)
                if ucbf < -Umin[i] - 1e-12:
                    infeas += 1; un_f = -Umin[i]
                else:
                    un_f = min(un, ucbf)
                if un_f < un - 1e-12:
                    act_time[i] += dt
                if extra:
                    mg = ucbf - un
                    mmin[i - 1] = min(mmin[i - 1], mg)
                    if record and k % sample_every == 0:
                        MG[i - 1].append((t, mg))
                un = un_f
            if extra and i >= 1:
                if k == k_xi:
                    e_hnd[i] = e; eps_hnd[i] = eps
                if k >= k_xi and np.isnan(tstop[i - 1]) \
                        and np.isnan(tstop[i]):
                    wint[i - 1] += abs(u[i - 1] - uh) * dt
                    epsmax_b[i - 1] = max(epsmax_b[i - 1], abs(eps))
                if k >= k_xi + i * dsteps and np.isnan(tstop[i - 1]):
                    fb_real[i - 1] = max(fb_real[i - 1],
                                         abs(u[i - 1] + a_b))
            if extra and i == 0 and k == k_xi:
                e_hnd[0] = e; eps_hnd[0] = eps
            un = float(np.clip(un, -A[i], A[i]))
            u[i] = un; upk[i] = max(upk[i], abs(un))
            if i >= 1:
                g = gap - s_m
                h = g + vp / b[i - 1] - vv[i] / b[i]
                gmin[i - 1] = min(gmin[i - 1], g)
                hmin[i - 1] = min(hmin[i - 1], h)
                hdot = uprev[i - 1] / b[i - 1] - un / b[i] + eps
                if hdot + kappa * h < -1e-12:
                    hits += 1
                if record and k % sample_every == 0:
                    H[i - 1].append(h); G[i - 1].append(g)
            if record and k % sample_every == 0:
                VT[i].append(vv[i]); ET[i].append(eps); UT[i].append(un)
        if record and k % sample_every == 0:
            TS.append(t)
        uprev = u.copy()
        if terminate_on_contact and contact_time is None:
            for i in range(1, n):
                if (s[i - 1] - s[i] - ell) <= 0.0:
                    contact_time = t
                    break
            if contact_time is not None:
                break
        for i in range(n):
            if mode[i] != 4:
                vv[i] = max(0.0, vv[i] + u[i] * dt)
                s[i] += vv[i] * dt
            if i + 1 < n:
                if const_delay:
                    qu[i + 1][k + delay] = u[i]
                    qf[i + 1][k + delay] = \
                        qf[i + 1][max(0, k + delay - 1)] or (mode[i] >= 2)
                else:
                    hist_u[i + 1][k] = u[i]
                    hist_f[i + 1][k] = (mode[i] >= 2)
        sref += vref * dt

    egap = (np.concatenate(([sref], s[:-1])) - s - ell) - d_s
    out = dict(tstop=tstop,
               disp=float(np.nanmax(tstop) - np.nanmin(tstop)),
               align=float(np.max(np.abs(s - marks))),
               gaperr=float(np.max(np.abs(egap))),
               egap=egap, hits=hits, infeas=infeas,
               act_time=act_time.tolist(),
               gmin=gmin, hmin=hmin, upk=upk, s=s, marks=marks,
               tact=tact, tset=tset, contact_time=contact_time,
               all_stopped=bool(np.all(np.isfinite(tstop))
                                and contact_time is None))
    if record:
        out.update(t=np.array(TS), v=np.array(VT), e=np.array(ET),
                   u=np.array(UT), h=np.array(H), g=np.array(G))
    if extra:
        out.update(mmin=mmin, wint=wint, epsmax_b=epsmax_b,
                   fb_real=fb_real, e_hnd=e_hnd, eps_hnd=eps_hnd)
        if record:
            out['m'] = [np.array(x) for x in MG]     # (t, m) pairs
    return out


def validate_featured():
    TK = json.load(open(os.path.join(HERE, '..', '..', 'data',
                                     'takeover_v42.json')))
    R = run(FEATURED, TK['d_s'], TK['v'], TK['e0'], record=True,
            rule='gapclose', apply_filter=False)
    D = np.load(os.path.join(HERE, '..', '..', 'data',
                             'sim_v56_traj.npz'), allow_pickle=True)
    ok = True
    for key in ('t', 'v', 'e', 'u', 'h', 'g', 'tstop', 'tact', 'tset',
                'marks', 's', 'egap'):
        a = np.asarray(R[key], float); bb = np.asarray(D[key], float)
        same = a.shape == bb.shape and np.array_equal(
            np.nan_to_num(a, nan=-1e18), np.nan_to_num(bb, nan=-1e18))
        if not same:
            ok = False
            print('MISMATCH', key,
                  float(np.nanmax(np.abs(a - bb))) if a.shape == bb.shape
                  else (a.shape, bb.shape))
    # constant-delay path vs explicit FIFO schedule must agree too
    delays = np.full((5, int(200.0 / 1e-3)), 60, int)
    R2 = run(FEATURED, TK['d_s'], TK['v'], TK['e0'], delays=delays,
             rule='gapclose')
    for key in ('disp', 'align', 'gaperr'):
        if abs(R2[key] - R[key]) > 0:
            ok = False; print('FIFO-path mismatch', key, R[key], R2[key])
    # NOTE (root-revision audit): the frozen sim's inline "oracle"
    # labels (0.0057187 s / 0.8244702 m) are stale relative to its own
    # archived output (0.0050 s / 0.8205878 m); the archive is the
    # authority here, and the manuscript's realized dispersion digit
    # is corrected accordingly.
    if abs(R['disp'] - 0.005) > 1e-6:
        ok = False; print('disp archive mismatch', R['disp'])
    print('sim_root2: featured reproduction ' +
          ('BIT-IDENTICAL (all arrays match the archive; FIFO path '
           'agrees)' if ok else 'FAILED'))
    return ok


if __name__ == '__main__':
    validate_featured()
