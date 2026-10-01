#!/usr/bin/env python3
"""sim_mpc.py -- cooperative arrival MPC baseline (native-architecture
track of the pre-submission review, item E7; 2026-09-07).

This is NOT a same-information baseline.  It is a transparent,
clearly labeled arrival-specific predictive controller in the spirit
of the cooperative/hierarchical MPC line of work, keeping the richer
information those architectures assume:

  * every unit stores the arrival reference profile (shared reference)
    and knows its own absolute position and its marker;
  * each follower receives, once per control period, its predecessor's
    predicted position/velocity trajectory over the horizon (the
    predecessor's last plan), used with an age of one control period
    (0.1 s >= the 0.02 s age bound of the proposed method);
  * the same actuation limits and the same sequential delayed-surrogate
    safety filter are applied downstream of the controller.

Controller: at every control period T_mpc each unit solves a
box-constrained least-squares MPC over N steps,

  min  sum_k  w_e (s_tgt_k - s_k)^2/sig_e^2 + w_v (v_tgt_k - v_k)^2/sig_v^2
             + w_u u_k^2/sig_u^2 + w_du (u_k - u_{k-1})^2/sig_du^2
  s.t. u_k in [-U_i^-, U],

with the double-integrator prediction model; targets are the reference
(head) or the predecessor's plan shifted by ell + d_s (followers).  The
QP is solved by scipy.optimize.lsq_linear (bounded variable least
squares).  Standstill: the first zero-speed instant during braking
latches the unit (same interlock as every other baseline).

Plant: 'ideal' (the certified double-integrator model at 1 ms) or
'hifi' (the perturbed plant of sim_hifi.py: actuator lag, jerk limit,
resistance, grade, derate, sensing noise, standstill threshold).
Nothing here feeds the certificate; the module realizes the review's
comparison track only.
"""
import time

import numpy as np
from scipy.optimize import lsq_linear

DEFAULT_W = dict(w_e=3.0, w_v=1.0, w_u=1.0, w_du=0.3)
# unit normalization of the objective (review E7): 1 m position error,
# 1 m/s speed error, 1 m/s^2 input, 0.5 m/s^2 input step per period
SIG = dict(e=1.0, v=1.0, u=1.0, du=0.5)
T_MPC = 0.10                                     # control period [s]
N_H = 20                                         # horizon steps (2 s)
V_STOP_IDEAL = 0.02                              # standstill detector [m/s]
# planner layer (hierarchical arrival MPC): the tracking MPC follows a
# closing profile whose speed differs from the predecessor's plan by at
# most DV_MAX, closing the spacing error with time constant TAU_C
TAU_C = 5.0                                      # [s]
DV_MAX = 0.5                                     # [m/s]


def _planner(s_now, s_tgt, v_tgt, T, tau_c=TAU_C, dv_max=DV_MAX):
    """Feasible closing profile toward (s_tgt_k, v_tgt_k): speed equals
    the target speed plus a saturated proportional closing term on the
    remaining spacing error, integrated from the current position."""
    N = len(s_tgt)
    s_des = np.zeros(N); v_des = np.zeros(N); sp = s_now
    for k in range(N):
        # spacing error at step k under the target speed itself
        err = s_tgt[k] - (sp + v_tgt[k] * T)
        v_des[k] = max(0.0, v_tgt[k] + float(np.clip(err / tau_c,
                                                     -dv_max, dv_max)))
        sp = sp + v_des[k] * T; s_des[k] = sp
    return s_des, v_des


def _pred_mats(N, T):
    Su = np.zeros((N, N)); Vu = np.zeros((N, N))
    for k in range(1, N + 1):
        for j in range(k):
            Su[k - 1, j] = T * T * (0.5 + (k - 1 - j))
            Vu[k - 1, j] = T
    return Su, Vu


class _MPC:
    def __init__(self, W, N=N_H, T=T_MPC):
        self.W = dict(DEFAULT_W, **(W or {})); self.N = N; self.T = T
        self.Su, self.Vu = _pred_mats(N, T)
        D = np.eye(N) - np.eye(N, k=-1)
        self.A = np.vstack([np.sqrt(self.W['w_e']) / SIG['e'] * self.Su,
                            np.sqrt(self.W['w_v']) / SIG['v'] * self.Vu,
                            np.sqrt(self.W['w_u']) / SIG['u'] * np.eye(N),
                            np.sqrt(self.W['w_du']) / SIG['du'] * D])
        self.ks = np.arange(1, N + 1)

    def solve(self, s0, v0, s_tgt, v_tgt, u_tgt, u_prev, lo, hi):
        """u_tgt: target acceleration sequence (the reference or the
        predecessor's planned acceleration), so the input penalty is
        feedforward-aware and introduces no steady tracking bias."""
        N = self.N; W = self.W
        b = np.concatenate([
            np.sqrt(W['w_e']) / SIG['e'] * (s_tgt - s0 - v0 * self.ks * self.T),
            np.sqrt(W['w_v']) / SIG['v'] * (v_tgt - v0),
            np.sqrt(W['w_u']) / SIG['u'] * u_tgt,
            np.sqrt(W['w_du']) / SIG['du'] * np.r_[u_prev, np.zeros(N - 1)]])
        r = lsq_linear(self.A, b, bounds=(lo, hi), method='bvls',
                       lsq_solver='exact', max_iter=60)
        u = r.x
        s_plan = s0 + v0 * self.ks * self.T + self.Su @ u
        v_plan = np.maximum(0.0, v0 + self.Vu @ u)
        return u, s_plan, v_plan


def _ref_positions(t, ks, T, v_xi, T_xi, a_b):
    """Reference position increments over the horizon (closed form of
    the piecewise profile), and reference speeds."""
    ts = t + ks * T
    vr = np.where(ts < T_xi, v_xi, np.maximum(0.0, v_xi - a_b * (ts - T_xi)))

    def S(tt):
        if tt < T_xi:
            return v_xi * tt
        tau = min(tt - T_xi, v_xi / a_b)
        return v_xi * T_xi + v_xi * tau - 0.5 * a_b * tau * tau
    return np.array([S(tt) for tt in ts]), vr, S(t)


def run(P, d_s, v0, e0, W=None, plant='ideal', hifi=None, seed=0,
        apply_filter=True, terminate_on_contact=True, T_pad=25.0,
        trace=None):
    import sim_hifi as sh
    H = dict(sh.DEFAULT, **(hifi or {}))
    rng = np.random.default_rng(seed)
    n = P['n']; dt = P['dt']
    N = int((P['T_END'] + (T_pad if plant == 'hifi' else 0.0)) / dt)
    a_b, v_xi, T_xi = P['a_b'], P['v_xi'], P['T_xi']
    ell, s_m, kappa, U = P['ell'], P['s_m'], P['kappa'], P['U']
    amax = np.array(P['amax'], float)
    A_cap = amax * (H['derate'] if plant == 'hifi' else 1.0)
    b = amax / P['v_c']
    Umin = np.minimum(U, amax)
    mpc = _MPC(W)
    d_s = np.array(d_s, float); vv = np.array(v0, float).copy()
    s = -np.cumsum(ell + d_s + np.array(e0, float)); sref = 0.0
    mark_ref = v_xi * T_xi + v_xi ** 2 / (2 * a_b)
    marks = mark_ref - np.cumsum(ell + d_s)
    delay = int(round(P['dbar'] / dt))
    qu = [np.zeros(N + delay + 2) for _ in range(n)]
    hold = int(round(T_MPC / dt))
    stopped = np.zeros(n, bool); tstop = np.full(n, np.nan)
    u_cmd = np.zeros(n); u_prev = np.zeros(n); u_act = np.zeros(n)
    comp = np.zeros(n)
    plans = [None] * n           # (s_plan, v_plan) of the last solve
    plans_prev = [None] * n      # one control period old (delivered)
    gmin = np.full(n - 1, 1e9); hmin = np.full(n - 1, 1e9)
    upk = np.zeros(n); hits = 0; infeas = 0; act_time = np.zeros(n)
    contact_time = None
    solve_ms = []
    tau_act = np.broadcast_to(np.asarray(H['tau_act'], float), (n,)) \
        .astype(float)
    grade_acc = H['grade'] * np.array([1, 1, -1, -1, 1][:n]) \
        if plant == 'hifi' else np.zeros(n)
    for k in range(N):
        t = k * dt
        vref = v_xi if t < T_xi else max(0.0, v_xi - a_b * (t - T_xi))
        if k % hold == 0:
            plans_prev = list(plans)
            t0 = time.perf_counter()
            s_ref_h, v_ref_h, s_ref_now = _ref_positions(
                t, mpc.ks, mpc.T, v_xi, T_xi, a_b)
            for i in range(n):
                # measurements
                if plant == 'hifi':
                    si_m = s[i] + rng.normal(0.0, H['noise_gap'])
                    vi_m = np.round((vv[i] + rng.normal(0.0, H['noise_vel']))
                                    / H['quant_vel']) * H['quant_vel']
                    if i > 0:
                        gap_m = (s[i - 1] - s[i] - ell) + H['bias_gap'] \
                            + rng.normal(0.0, H['noise_gap'])
                        vp_m = np.round((vv[i - 1] + rng.normal(
                            0.0, H['noise_vel'])) / H['quant_vel']) \
                            * H['quant_vel']
                else:
                    si_m = s[i]; vi_m = vv[i]
                    if i > 0:
                        gap_m = s[i - 1] - s[i] - ell; vp_m = vv[i - 1]
                if stopped[i]:
                    un = 0.0
                    plans[i] = (np.full(N_H, s[i]), np.zeros(N_H))
                else:
                    if i == 0:
                        s_tgt = sref + (s_ref_h - s_ref_now) \
                            - (ell + d_s[0])
                        v_tgt = v_ref_h
                        u_tgt = np.diff(np.r_[vref, v_ref_h]) / mpc.T
                    else:
                        pp = plans_prev[i - 1]
                        if pp is None:           # no plan received yet
                            sp = si_m + gap_m + ell
                            s_tgt = sp + vp_m * mpc.ks * mpc.T - (ell + d_s[i])
                            v_tgt = np.full(N_H, vp_m)
                            u_tgt = np.zeros(N_H)
                        else:
                            ps, pv = pp
                            u_tgt = np.diff(np.r_[pv, pv[-1]]) / mpc.T
                            ps = np.r_[ps[1:], ps[-1] + pv[-1] * mpc.T]
                            pv = np.r_[pv[1:], pv[-1]]
                            s_tgt = ps - (ell + d_s[i]); v_tgt = pv
                    # planner layer: feasible closing profile, then the
                    # tracking MPC follows it (feedforward-aware)
                    s_tgt, v_tgt = _planner(si_m, s_tgt, v_tgt, mpc.T)
                    u_tgt = np.diff(np.r_[vi_m, v_tgt]) / mpc.T
                    u_seq, s_plan, v_plan = mpc.solve(
                        si_m, vi_m, s_tgt, v_tgt, u_tgt, u_prev[i],
                        -Umin[i], U)
                    un = float(u_seq[0])
                    plans[i] = (s_plan, v_plan)
                # matched safety layer: sequential delayed-surrogate filter
                if apply_filter and i >= 1 and not stopped[i]:
                    uh = qu[i][k]
                    g = gap_m - s_m
                    h = g + vp_m / b[i - 1] - vi_m / b[i]
                    eps = vp_m - vi_m
                    ucbf = b[i] * (eps + uh / b[i - 1] + kappa * h)
                    if ucbf < -Umin[i] - 1e-12:
                        infeas += 1; un_f = -Umin[i]
                    else:
                        un_f = min(un, ucbf)
                    if un_f < un - 1e-12:
                        hits += 1; act_time[i] += T_MPC
                    un = un_f
                un = float(np.clip(un, -Umin[i], U))
                if plant == 'hifi':
                    dumax = H['jerk_max'] * T_MPC
                    un = float(np.clip(un, u_cmd[i] - dumax, u_cmd[i] + dumax))
                u_prev[i] = un
                u_cmd[i] = float(np.clip(un, -A_cap[i], A_cap[i]))
                upk[i] = max(upk[i], abs(u_cmd[i]))
            solve_ms.append((time.perf_counter() - t0) * 1e3)
            if trace is not None and k % (50 * hold) == 0:
                gaps = np.concatenate(([sref], s[:-1])) - s - ell - d_s
                trace.append((round(t, 2), [round(float(x), 3) for x in gaps],
                              [round(float(x), 3) for x in vv],
                              [round(float(x), 3) for x in u_cmd]))
        # plant
        for i in range(n):
            if plant == 'hifi':
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
                # standstill detector: below the threshold while the
                # predecessor (or the reference) is at rest
                pred_rest = (vref <= 1e-12) if i == 0 else stopped[i - 1]
                if t > T_xi and vv[i] <= H['v_stop'] and pred_rest:
                    stopped[i] = True; tstop[i] = t; vv[i] = 0.0
            else:
                if stopped[i]:
                    continue
                vv[i] = max(0.0, vv[i] + u_cmd[i] * dt)
                s[i] += vv[i] * dt
                # standstill detector of the MPC baseline (ideal plant):
                # a 0.02 m/s threshold once the predecessor (or the
                # reference) is at rest, since the predictive law has no
                # sensed-standstill rule of its own
                pred_rest = (vref <= 1e-12) if i == 0 else stopped[i - 1]
                if t > T_xi and vv[i] <= V_STOP_IDEAL and pred_rest:
                    stopped[i] = True; tstop[i] = t; vv[i] = 0.0
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            h = g + vv[i - 1] / b[i - 1] - vv[i] / b[i]
            gmin[i - 1] = min(gmin[i - 1], g); hmin[i - 1] = min(hmin[i - 1], h)
            if g + s_m <= 0.0 and contact_time is None:
                contact_time = t
        if contact_time is not None and terminate_on_contact:
            break
        for i in range(n - 1):
            qu[i + 1][k + delay] = u_cmd[i] - comp[i]
        sref += vref * dt
    egap = (np.concatenate(([sref], s[:-1])) - s - ell) - d_s
    fin = np.all(np.isfinite(tstop))
    return dict(
        tstop=[None if not np.isfinite(x) else float(x) for x in tstop],
        disp=float(np.nanmax(tstop) - np.nanmin(tstop)) if fin else None,
        align=float(np.max(np.abs(s - marks))),
        gaperr=float(np.max(np.abs(egap))),
        egap=[float(x) for x in egap],
        gmin=[float(x) for x in gmin], hmin=[float(x) for x in hmin],
        upk=[float(x) for x in upk], hits=int(hits), infeas=int(infeas),
        act_time=[float(x) for x in act_time],
        contact_time=contact_time,
        all_stopped=bool(fin and contact_time is None),
        solve_ms_mean=float(np.mean(solve_ms)) if solve_ms else None,
        solve_ms_max=float(np.max(solve_ms)) if solve_ms else None,
        W=dict(mpc.W), plant=plant)
