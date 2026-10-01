#!/usr/bin/env python3
r"""sim_bench_hr.py -- study simulator of the head-only reference arrival
(HR design, data/new_d19/headref/sim_spec.md), 2026-09-28.

The module is the counterpart of sim_bench.py (studies of the design with a
plan at every unit) for the HR design, in which only the head stores the
arrival reference and every follower stores the constants of a gap-closing
schedule that it starts when it processes the relayed closure flag.

Reuse (read-only imports, no file is modified):
  sim_hr.hr_setup   reference of the head, schedules of the followers (exact
                    rational checks of the pieces), input sets;
  sim_cont.deliveries  FIFO delivery tables (constant transport or per-send
                    schedules, batch delivery);
  sim_bench         the comparators 'plans' (sim_bench 'proposed': a plan at
                    every unit, common time base, deviation messages) and
                    'mpc' (cooperative arrival MPC), run through its API with
                    the input set [-U_i^-, U]; the constants of the perturbed
                    plant (HIFI_DEFAULT, GRADE_SIGNS, U_ACT) and of the PID
                    law (K_PID).

Model and time stepping (plant conventions of sim_bench)
  ds_i/dt = v_i, dv_i/dt = u_i, u_i in [-U_i^-, U]; five units of length
  ell; unit 1 (index 0) is the head.  Step dt = 1 ms, message period
  T_m = detector period T_c = dt, transport of every hop constant (dbar -
  T_m, 19 ms) or a per-send FIFO schedule (sim_cont.deliveries).
  plant 'ideal': the command of every unit is computed at every grid
    instant k dt after the events of that instant and held (T_u = dt); the
    double integrator is integrated exactly.  Zero-speed instants inside a
    step are located exactly (tau = -v/u) one at a time, earliest first, as
    in sim_hr (scheme 'zoh'): the unit latches there, and a follower in S2
    whose predecessor stops enters RULE at that instant with an immediate
    command update (exact standstill sensing, spec Sec. 3); the rest of the
    step is then integrated.  Between these instants every pair clearance
    is quadratic; its minimum, the first instant with g_i < 0 (buffer lost)
    and the first instant with c_i <= 0 (contact) are computed in closed
    form (sim_bench).  A run ends at contact, when every unit is latched, or
    at T_end (default T_xi + 60 s).
  plant 'hifi': the perturbed plant of sim_bench (sim_hifi.DEFAULT): control
    every T_s = 10 ms on measured signals (ranging noise and bias, speed
    noise and quantization), first-order actuator lag, jerk limit, running
    resistance, per-unit grade, braking derated, actuator limits
    [-derate amax_i, U_ACT]; optional compensation of grade and resistance
    (grade_ff, res_ff: added to the net command after the input set, as in
    sim_bench).  Standstill detection from measured speeds: LATCH at a
    measured speed <= v_stop (modes S2, RULE, BRAKE, FF), RULE at a measured
    predecessor speed <= eps_det (mode S2 or FF).  sim_bench gates both by
    t > T_xi; the followers of the HR design do not know T_xi, so no time
    gate is used here (before the braking every measured speed is near 1 m/s
    or above, about 45 noise standard deviations above the thresholds).
    A planned jump or the head switch to BRAKE off the 10 ms grid is
    processed at the next control instant.  ramp_lead is not supported.

Event order at a control instant, unit by unit (head first; sim_hr): LATCH,
RULE, head BRAKE (k >= k_xi) and head CLOSE raise (k >= k_t0), packet
batch (held command, flags MATCHED and CLOSE, activation, closure onset),
detector sample, loss watchdog, command update; then every unit sends its
applied command (post-event value; on the perturbed plant the actuator
command without the compensation, as sim_bench) and its flags.  Because the
transport is at least one step, this order equals 'all events, all
commands, all sends'.  The loss onset (receivers hold zero, no delivery, no
flag from then on) precedes the events of its instant.

Controller variants (run(variant=...))
  'proposed'     the HR laws of sim_spec.md: head S1 a_r + alpha sat(eps_1/phi),
                 S2 a_r + beta e_1 + gamma eps_1, BRAKE -a_b from T_xi;
                 follower COAST uhat + rho_i, S1 uhat + rho_i + alpha sat,
                 S2 uhat + rho_i + beta e_i + gamma eps_i (schedule-relative
                 errors), RULE -a_b; flags MATCHED and CLOSE relayed; filter
                 u_cbf = b_i((v_{i-1}-v_i) + uhat/b_{i-1} + kappa h_i),
                 sat_[-U_i^-, U](min{u_nom, u_cbf}), fallback -U_i^-;
                 LATCH.  Reproduces sim_hr.run(scheme 'zoh', T_u = 1 ms).
  'nofilter'     'proposed' without the filter (the input set stays).
  'single'       'proposed' without the S1 stage of the followers: COAST to
                 S2 at activation (MATCHED raised there); the head keeps S1.
  'nofeedback'   no feedback: every follower applies uhat + rho_i from
                 takeover (mode FF), the head applies a_r (mode FF) and brakes
                 at T_xi; RULE (from FF) and LATCH kept; CLOSE relayed; filter
                 kept.
  'feedback'     frozen laws of the paper before the redesign: no schedules;
                 follower errors relative to the parking gap,
                 e_i = c_i - d_s,i, eps_i = v_{i-1} - v_i (regulated from the
                 S2 switch), feedforward uhat; two stages; head identical.
  'feedback_step' sim_hr nosched='step': d_i = c_i(0) until the closure
                 onset, d_s,i from it; rho_i = dd_i = 0; two stages.
  'cacc'         as 'feedback', single stage, spacing target
                 d_s,i + h_w v_i (h_w default 0.5 s).
  'pid'          single stage; S2 feedback Kp e_i + Kd eps_i + Ki I_i on the
                 schedule-relative errors, I_i <- clip(I_i + e_i T_u,
                 +-Imax/Ki) at every regular command update in S2 (sim_bench
                 convention); feedforward uhat + rho_i; the head runs S1 and
                 then PID on (e_1, eps_1) with a_r.
  'plans'        comparator with richer information: sim_bench 'proposed'
                 (plan at every unit, common time base, deviation messages,
                 synchronized braking at T_xi) through sim_bench.run with the
                 input set [-U_i^-, U] and the same gain; constant transport.
  'mpc'          sim_bench cooperative arrival MPC, unchanged (bounds
                 [-U_i^-, U]).  Information: absolute positions and speeds of
                 the unit, the planned position and speed of unit 1 (the HR
                 reference) at the head, the predicted trajectory of the
                 predecessor (received one control period of 0.1 s late),
                 the received applied command in the filter.  It uses neither
                 the follower plans of 'plans' nor the schedules of the HR
                 design.  sim_bench's MPC has no compensation of grade and
                 resistance (grade_ff and res_ff are ignored by run_mpc).

Scenarios: communication loss at t_loss (every link fails at the first grid
instant at or after t_loss; every receiver sets its held command to zero and
keeps its flags and mode; no message or flag is delivered afterwards),
optional emergency braking of one unit at its full capability -amax_i from
the loss instant until it rests (bypassing laws, filter and input set), and
a per-unit watchdog: a follower that has received no packet for T_wd applies
-amax_i until it rests.  The watchdog acts only in the loss scenario.  Clock
offsets are not modelled: in the HR design no unit uses a common clock (the
head evaluates the reference from its takeover, followers their schedules
from their own closure onsets); the only absolute times are the head's own
events t_0 and T_xi.

Plant conventions shared with sim_bench: a unit latches at its first
zero-speed instant in every mode on the ideal plant (a unit in COAST or S1
that reaches zero speed is counted in n_latch_before_S2; sim_hr lets such a
unit continue to negative speeds instead; the count is zero in every run of
the regression check).  Numba is required; the cache goes to the temporary
directory of the system.
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR',
                      os.path.join(tempfile.gettempdir(),
                                   'numba_cache_sim_bench_hr'))
from numba import njit                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
for _p in (HERE, os.path.join(CODE, 'src', 'certification')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import sim_hr as hr                                      # noqa: E402
import sim_cont as sc                                    # noqa: E402
import sim_bench as sb                                   # noqa: E402

# ------------------------------------------------------------------ modes
COAST, S1, S2, RULE, LATCH, BRAKE, FF = 0, 1, 2, 3, 4, 5, 6
MODE_NAMES = ('COAST', 'S1', 'S2', 'RULE', 'LATCH', 'BRAKE', 'FF')

# kernel laws
K_PROP, K_SINGLE, K_NOFB, K_PID, K_FB, K_CACC = range(6)

# run() variants of the kernel: (law, schedule mode, filter default)
#   schedule mode 'plan': rho_i, d_i, dd_i of the design (spec Sec. 2);
#   'step': d_i = c_i(0) before the closure onset, d_s,i from it;
#   'none': no schedule (the law uses d_s,i directly)
VARIANTS = {
    'proposed': (K_PROP, 'plan', True),
    'nofilter': (K_PROP, 'plan', False),
    'single': (K_SINGLE, 'plan', True),
    'nofeedback': (K_NOFB, 'plan', True),
    'feedback': (K_FB, 'none', True),
    'feedback_step': (K_PROP, 'step', True),
    'cacc': (K_CACC, 'none', True),
    'pid': (K_PID, 'plan', True),
}
EXTERNAL = ('plans', 'mpc')
ALL_VARIANTS = tuple(VARIANTS) + EXTERNAL

# time accumulators (held command of a unit before its latch)
(Q_ATU, Q_NONNEG, Q_NOMHI, Q_CLIPU, Q_CLIPK, Q_BAR, Q_BARRAW, Q_INFEAS,
 Q_OVR, Q_NOMPOS, Q_N) = range(11)
Q_NAMES = ('t_atU', 't_nonneg', 't_nom_above_U', 't_clipU', 't_clipk',
           't_barrier', 't_barrier_raw', 't_infeas', 't_override',
           't_nom_pos')

# parameter slots
(I_N, I_NSTEP, I_NC, I_HOLD, I_VAR, I_FILT, I_PLANT, I_LOSS, I_KLOSS, I_EM,
 I_KWD, I_KXI, I_KT0, I_NH, I_REC, I_NREC, I_NSEND, I_NI) = range(18)
(F_DT, F_AB, F_ALPHA, F_BETA, F_GAMMA, F_PHI, F_EPSDET, F_KAPPA, F_ELL, F_SM,
 F_U, F_KP, F_KD, F_KI, F_IMAX, F_HW, F_UACT, F_DERATE, F_JERK, F_A0, F_A1,
 F_A2, F_NGAP, F_NVEL, F_BIAS, F_QUANT, F_VSTOP, F_GFF, F_RFF, F_TXI,
 F_AHOLD, F_NF) = range(32)
(O_CONTACT, O_TEND, O_NO) = range(3)
(J_CPAIR, J_KEND, J_NREC, J_NLATPRE, J_NI) = range(5)

REQ = dict(disp=1.0, marker=3.0, gap=1.0)
# final design parameters (2026-09-28): lam = 0.15; vhnd enters only the
# certificate (recorded in the configuration, no effect on the simulation)
DEFAULTS = dict(lam=0.15, U=0.02, Uminus=1.2, eps_e=0.05, eps_v=0.005,
                dbar=0.020, vhnd=0.01)


# ================================================================ kernels
@njit(cache=True)
def _href(t, j, HT, HA, HS, HV):
    """Head reference (s_0, v_r, a_r) at t on the frozen piece j (sim_hr)."""
    tau = t - HT[j]
    a = HA[j]
    return HS[j] + HV[j] * tau + 0.5 * a * tau * tau, HV[j] + a * tau, a


@njit(cache=True)
def _sched(i, t, dt, kcl, js, SK, SR, SD, SV, DPRE):
    """Schedule (d_i, dd_i, rho_i) of follower i at t on piece js[i] (before
    the closure onset: DPRE[i], 0, 0), float operations of sim_hr."""
    if kcl[i] < 0:
        return DPRE[i], 0.0, 0.0
    j = js[i]
    tau = t - (kcl[i] + SK[i, j]) * dt
    r = SR[i, j]
    return SD[i, j] + SV[i, j] * tau - 0.5 * r * tau * tau, \
        SV[i, j] - r * tau, r


@njit(cache=True)
def _err(i, var, t, dt, s, sg, vim, vpm, pe0, s0r, vr, ar, uh, kcl, js, SK,
         SR, SD, SV, DPRE, d_s, F):
    """Measured quantities of unit i: (clearance, predecessor speed, own
    speed, e_i, eps_i, feedforward)."""
    ell = F[F_ELL]
    if i == 0:
        gap = s0r - s[0] - ell
        e = (gap - d_s[0]) + pe0
        vi = vim[0]
        return gap, vr, vi, e, vr - vi, ar
    gap = sg[i] - ell
    vp = vpm[i]
    vi = vim[i]
    epsr = vp - vi
    if var == K_FB:
        return gap, vp, vi, sg[i] - (d_s[i] + ell), epsr, uh[i]
    if var == K_CACC:
        return gap, vp, vi, sg[i] - (d_s[i] + ell) - F[F_HW] * vi, epsr, \
            uh[i]
    d, dd, rho = _sched(i, t, dt, kcl, js, SK, SR, SD, SV, DPRE)
    return gap, vp, vi, gap - d, epsr - dd, uh[i] + rho


@njit(cache=True)
def _cmd(i, var, md, ovr, gap, vp, vi, e, eps, ff, uhi, pidI, upd_int, T_u,
         F, bf, Umin, amax, filt):
    """Nominal law, filter and input set of unit i at one instant; returns
    (applied command, u_nom, filter output before the input set, u_cbf,
    bits of Q_*)."""
    if md == LATCH:
        return 0.0, 0.0, 0.0, np.nan, 0
    if ovr:
        u = -amax[i]
        return u, u, u, np.nan, 1 << Q_OVR
    if md == RULE or md == BRAKE:
        un = -F[F_AB]
    elif md == S2:
        if var == K_PID:
            if upd_int:
                lim = F[F_IMAX] / F[F_KI]
                y = pidI[i] + e * T_u
                if y > lim:
                    y = lim
                elif y < -lim:
                    y = -lim
                pidI[i] = y
            un = ff + F[F_KP] * e + F[F_KD] * eps + F[F_KI] * pidI[i]
        else:
            un = ff + F[F_BETA] * e + F[F_GAMMA] * eps
    elif md == S1:
        x = eps / F[F_PHI]
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        un = ff + F[F_ALPHA] * x
    else:                                   # COAST, FF
        un = ff
    uq = un
    ucbf = np.nan
    bits = 0
    lo = -Umin[i]
    hi = F[F_U]
    if i >= 1:
        g = gap - F[F_SM]
        hb = g + vp / bf[i - 1] - vi / bf[i]
        ucbf = bf[i] * ((vp - vi) + uhi / bf[i - 1] + F[F_KAPPA] * hb)
        if filt:
            if ucbf < lo - 1e-12:
                uq = lo
                bits |= 1 << Q_INFEAS
            elif ucbf < un:
                uq = ucbf
    ua = uq
    if ua < lo:
        ua = lo
    elif ua > hi:
        ua = hi
    unf = un
    if unf < lo:
        unf = lo
    elif unf > hi:
        unf = hi
    if ua == hi:
        bits |= 1 << Q_ATU
    if ua >= 0.0:
        bits |= 1 << Q_NONNEG
    if un > hi:
        bits |= 1 << Q_NOMHI
    if uq > hi:
        bits |= 1 << Q_CLIPU
    if ua == lo:
        bits |= 1 << Q_CLIPK
    if ua < unf - 1e-12:
        bits |= 1 << Q_BAR
    if uq < un - 1e-12:
        bits |= 1 << Q_BARRAW
    if un > 0.0:
        bits |= 1 << Q_NOMPOS
    return ua, un, uq, ucbf, bits


@njit(cache=True)
def _first_below(q0, d0, r, L):
    """First tau in [0, L] with q0 + d0 tau + r tau^2/2 <= 0 for q0 > 0;
    -1.0 if the quadratic stays positive on [0, L] (sim_bench)."""
    qL = q0 + d0 * L + 0.5 * r * L * L
    qm = qL
    if r > 0.0 and d0 < 0.0:
        tv = -d0 / r
        if tv < L:
            qm = q0 + 0.5 * d0 * tv
    if qm > 0.0 and qL > 0.0:
        return -1.0
    D = d0 * d0 - 2.0 * r * q0
    if D < 0.0:
        D = 0.0
    den = -d0 + math.sqrt(D)
    if den <= 0.0:
        return L
    tau = 2.0 * q0 / den
    if tau > L:
        tau = L
    if tau < 0.0:
        tau = 0.0
    return tau


@njit(cache=True)
def _pairs_sub(t0, L, s, v, u, latb, ell, s_m, gmin, tgmin, tbuf):
    """Clearance statistics over [t0, t0 + L] of the ideal plant, in which
    every unit moves with its held command and no unit stops inside the
    interval (each pair clearance is one quadratic): smallest margin g_i and
    its instant, first instant with g_i < 0; returns the earliest contact
    offset and its pair index ((-1, -1): none)."""
    n = s.shape[0]
    tc_best = -1.0
    ip_best = -1
    for i in range(1, n):
        g0 = s[i - 1] - s[i] - ell - s_m
        vp = 0.0 if latb[i - 1] else v[i - 1]
        vf = 0.0 if latb[i] else v[i]
        ap = 0.0 if latb[i - 1] else u[i - 1]
        af = 0.0 if latb[i] else u[i]
        d0 = vp - vf
        r = ap - af
        if g0 < gmin[i - 1]:
            gmin[i - 1] = g0
            tgmin[i - 1] = t0
        g1 = g0 + d0 * L + 0.5 * r * L * L
        gm = g1
        tm = L
        if r > 0.0 and d0 < 0.0:
            tv = -d0 / r
            if tv < L:
                gm = g0 + 0.5 * d0 * tv
                tm = tv
        if gm < gmin[i - 1]:
            gmin[i - 1] = gm
            tgmin[i - 1] = t0 + tm
        if tbuf[i - 1] < 0.0:
            if g0 < 0.0:
                tbuf[i - 1] = t0
            else:
                tb = _first_below(g0, d0, r, L) if g0 > 0.0 else 0.0
                if tb >= 0.0:
                    tbuf[i - 1] = t0 + tb
        c0 = g0 + s_m
        if c0 <= 0.0:
            tcn = 0.0
        else:
            tcn = _first_below(c0, d0, r, L)
        if tcn >= 0.0:
            if tc_best < 0.0 or tcn < tc_best:
                tc_best = tcn
                ip_best = i
    return tc_best, ip_best


@njit(cache=True)
def _step_hifi(t, dt, s, v, ua, uc, latb, tau_act, grade_acc, a0, a1, a2,
               ell, s_m, gmin, tgmin, tbuf, uamx, a_hold, tstop, sstop):
    """One step of the perturbed plant (copy of sim_bench._step_hifi):
    actuator lag (explicit Euler), resistance and grade,
    v <- max(0, v + a dt), s += v dt; clearance statistics at the end of
    the step.  A latched unit that still moves (the latch acts on the
    measured speed) decelerates at the holding-brake rate a_hold > 0, with
    resistance and grade, until its true speed is zero; its stop instant
    and position are recorded then (a_hold = 0: the unit rests at once, the
    convention before 2026-09-29).  Returns the index of the first pair in
    contact (-1: none)."""
    n = s.shape[0]
    for i in range(n):
        if tau_act[i] > 0.0:
            ua[i] += dt / tau_act[i] * (uc[i] - ua[i])
        else:
            ua[i] = uc[i]
        if latb[i]:
            if a_hold > 0.0 and v[i] > 0.0:
                ah = -a_hold - (a0 + a1 * v[i] + a2 * v[i] * v[i]) \
                    - grade_acc[i]
                vh = v[i] + ah * dt
                if vh <= 0.0:
                    th = -v[i] / ah
                    s[i] += v[i] * th + 0.5 * ah * th * th
                    v[i] = 0.0
                    tstop[i] = t + th
                    sstop[i] = s[i]
                else:
                    v[i] = vh
                    s[i] += vh * dt
            continue
        sgn = 0.0
        if v[i] > 0.0:
            sgn = 1.0
        elif v[i] < 0.0:
            sgn = -1.0
        a = ua[i] - sgn * (a0 + a1 * v[i] + a2 * v[i] * v[i]) - grade_acc[i]
        vn = v[i] + a * dt
        if vn < 0.0:
            vn = 0.0
        v[i] = vn
        s[i] += vn * dt
        if abs(ua[i]) > uamx[i]:
            uamx[i] = abs(ua[i])
    ip = -1
    for i in range(1, n):
        g = s[i - 1] - s[i] - ell - s_m
        if g < gmin[i - 1]:
            gmin[i - 1] = g
            tgmin[i - 1] = t + dt
        if g < 0.0 and tbuf[i - 1] < 0.0:
            tbuf[i - 1] = t + dt
        if g + s_m <= 0.0 and ip < 0:
            ip = i
    return ip


@njit(cache=True)
def _stats(i, ua, un, ucbf, bits, md, ovr, t, T_xi, umn, umx, umn_pre,
           umx_pre, mmin, tmmin, cnt):
    """Statistics of one command update of unit i (before its latch)."""
    if md == LATCH:
        return
    if ua < umn[i]:
        umn[i] = ua
    if ua > umx[i]:
        umx[i] = ua
    if t < T_xi:
        if ua < umn_pre[i]:
            umn_pre[i] = ua
        if ua > umx_pre[i]:
            umx_pre[i] = ua
    cnt[Q_N, i] += 1
    for q in range(Q_N):
        if bits & (1 << q):
            cnt[q, i] += 1
    if i >= 1 and not ovr:
        mg = ucbf - un
        if mg < mmin[i - 1]:
            mmin[i - 1] = mg
            tmmin[i - 1] = t


@njit(cache=True)
def _core(I, F, amax, Umin, bf, d_s, s, v, deliv, HT, HK, HA, HS, HV, NS,
          SK, SR, SD, SV, DPRE, Z, tau_act, grade_acc, tstop, sstop, tact,
          tset, tcl, trule, twd, gmin, tgmin, hmin, thmin, mmin, tmmin, tbuf,
          umn, umx, umn_pre, umx_pre, uamx, ucmx, jpk, tim, cnt, e_hnd,
          eps_hnd, fout, iout, rec, mode_out):
    """Closed loop of the HR variants on the ideal or the perturbed plant
    (module docstring)."""
    n = I[I_N]
    N = I[I_NSTEP]
    Nc = I[I_NC]
    hold = I[I_HOLD]
    var = I[I_VAR]
    filt = I[I_FILT] == 1
    hifi = I[I_PLANT] == 1
    loss = I[I_LOSS] == 1
    k_loss = I[I_KLOSS]
    em = I[I_EM]
    k_wd = I[I_KWD]
    k_xi = I[I_KXI]
    kt0 = I[I_KT0]
    NH = I[I_NH]
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    Nsend = I[I_NSEND]
    dt = F[F_DT]
    T_u = hold * dt
    ell = F[F_ELL]
    s_m = F[F_SM]
    eps_det = F[F_EPSDET]
    T_xi = F[F_TXI]
    single = var == K_SINGLE or var == K_PID or var == K_CACC
    nofb = var == K_NOFB

    mode = np.zeros(n, np.int64)
    if nofb:
        for i in range(n):
            mode[i] = FF
    else:
        mode[0] = S1
    kact = np.full(n, -1, np.int64)
    kact[0] = 0
    uh = np.zeros(n)
    flagm = np.zeros(n, np.bool_)
    flagc = np.zeros(n, np.bool_)
    closed = np.zeros(n, np.bool_)
    kcl = np.full(n, -1, np.int64)
    js = np.zeros(n, np.int64)
    jh = 0
    ptr = np.zeros(n, np.int64)
    k_rx = np.zeros(n, np.int64)
    ucmd = np.zeros(n)
    uc = np.zeros(n)
    ua = np.zeros(n)
    comp = np.zeros(n)
    ccur = np.zeros(n, np.int64)
    pidI = np.zeros(n)
    wd = np.zeros(n, np.bool_)
    usent = np.zeros((n, N + 2))
    msent = np.zeros((n, N + 2), np.bool_)
    csent = np.zeros((n, N + 2), np.bool_)
    sg = np.zeros(n)
    vim = np.zeros(n)
    vpm = np.zeros(n)
    latb = np.zeros(n, np.bool_)
    pe0 = 0.0
    lost = False
    contact = np.nan
    cpair = -1
    irec = 0
    k_end = N
    nlat_pre = 0
    hnd_done = False
    stop_all = False

    for k in range(N + 1):
        t = k * dt
        ctrl = (k % hold) == 0
        # ------------------------------------ planned pieces (right limits)
        while jh + 1 < NH and HK[jh + 1] >= 0 and HK[jh + 1] <= k:
            jh += 1
        for i in range(1, n):
            if kcl[i] >= 0:
                while js[i] + 1 < NS[i] and kcl[i] + SK[i, js[i] + 1] <= k:
                    js[i] += 1
        # ------------------------------------ barrier and clearance at t_k
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            hb = g + v[i - 1] / bf[i - 1] - v[i] / bf[i]
            if hb < hmin[i - 1]:
                hmin[i - 1] = hb
                thmin[i - 1] = t
            if hifi and k == 0:
                gmin[i - 1] = g
                tgmin[i - 1] = 0.0
        # ------------------------------------ pair errors at T_xi (true state)
        if (not hnd_done) and k >= k_xi:
            hnd_done = True
            s0x, vrx, arx = _href(t, jh, HT, HA, HS, HV)
            e_hnd[0] = s0x - s[0] - ell - d_s[0]
            eps_hnd[0] = vrx - v[0]
            for i in range(1, n):
                d, dd, rho = _sched(i, t, dt, kcl, js, SK, SR, SD, SV, DPRE)
                e_hnd[i] = s[i - 1] - s[i] - ell - d
                eps_hnd[i] = v[i - 1] - v[i] - dd
        # ------------------------------------ loss onset
        if loss and (not lost) and k >= k_loss:
            lost = True
            for i in range(1, n):
                uh[i] = 0.0
        if ctrl:
            # -------------------------------- measurements
            if hifi:
                jz = k // hold
                pe0 = F[F_BIAS] + F[F_NGAP] * Z[jz, 0, 0]
                qv = F[F_QUANT]
                for i in range(n):
                    vim[i] = np.round((v[i] + F[F_NVEL] * Z[jz, i, 1])
                                      / qv) * qv
                    if i >= 1:
                        sg[i] = (s[i - 1] - s[i]) + F[F_BIAS] \
                            + F[F_NGAP] * Z[jz, i, 0]
                        vpm[i] = np.round((v[i - 1] + F[F_NVEL]
                                           * Z[jz, i, 2]) / qv) * qv
            else:
                pe0 = 0.0
                for i in range(n):
                    vim[i] = v[i]
                    if i >= 1:
                        sg[i] = s[i - 1] - s[i]
                        vpm[i] = v[i - 1]
            s0r, vr, ar = _href(t, jh, HT, HA, HS, HV)
            for i in range(n):
                md0 = mode[i]
                ovr0 = (lost and i == em) or wd[i]
                # ---------------------------- LATCH, RULE
                lat = False
                if md0 != LATCH and (md0 == S2 or md0 == RULE or md0 == BRAKE
                                     or md0 == FF or ovr0):
                    if hifi:
                        lat = vim[i] <= F[F_VSTOP]
                    else:
                        lat = v[i] <= 0.0
                rul = False
                if (not lat) and i >= 1 and (md0 == S2 or md0 == FF):
                    if hifi:
                        rul = vpm[i] <= eps_det
                    else:
                        rul = v[i - 1] <= 0.0
                if lat:
                    mode[i] = LATCH
                    if F[F_AHOLD] > 0.0 and v[i] > 0.0:
                        pass            # holding brake: stop recorded by the plant
                    else:
                        tstop[i] = t
                        sstop[i] = s[i]
                        v[i] = 0.0
                elif rul:
                    mode[i] = RULE
                    trule[i] = t
                # ---------------------------- head clock events
                if i == 0:
                    if k >= k_xi and (mode[0] == S1 or mode[0] == S2
                                      or mode[0] == FF):
                        mode[0] = BRAKE
                    if k >= kt0 and not closed[0]:
                        closed[0] = True
                        tcl[0] = t
                else:
                    # ------------------------ packet batch
                    if not lost:
                        got = False
                        while ptr[i] < Nsend and deliv[i, ptr[i]] <= k:
                            j = ptr[i]
                            uh[i] = usent[i - 1, j]
                            if msent[i - 1, j]:
                                flagm[i] = True
                            if csent[i - 1, j]:
                                flagc[i] = True
                            ptr[i] += 1
                            got = True
                        if got:
                            k_rx[i] = k
                    if mode[i] == COAST and flagm[i]:
                        kact[i] = k
                        tact[i] = t
                        if single:
                            mode[i] = S2
                            tset[i] = t
                        else:
                            mode[i] = S1
                    if flagc[i] and not closed[i]:
                        closed[i] = True
                        tcl[i] = t
                        if NS[i] > 0:
                            kcl[i] = k
                            js[i] = 0
                            while js[i] + 1 < NS[i] and SK[i, js[i] + 1] <= 0:
                                js[i] += 1
                # ---------------------------- errors, detector
                gap, vp, vi, e, eps, ff = _err(i, var, t, dt, s, sg, vim, vpm,
                                               pe0, s0r, vr, ar, uh, kcl, js,
                                               SK, SR, SD, SV, DPRE, d_s, F)
                if mode[i] == S1 and (k - kact[i]) % Nc == 0:
                    if abs(eps) <= eps_det:
                        mode[i] = S2
                        tset[i] = t
                # ---------------------------- watchdog
                if mode[i] != LATCH and k_wd >= 0 and i >= 1 and \
                        (not wd[i]) and k - k_rx[i] >= k_wd:
                    wd[i] = True
                    twd[i] = t
                ovr = (lost and i == em) or wd[i]
                # ---------------------------- command update
                u, un, uq, ucbf, bits = _cmd(i, var, mode[i], ovr, gap, vp, vi,
                                             e, eps, ff, uh[i], pidI, True,
                                             T_u, F, bf, Umin, amax, filt)
                ucmd[i] = u
                ccur[i] = bits
                _stats(i, u, un, ucbf, bits, mode[i], ovr, t, T_xi, umn, umx,
                       umn_pre, umx_pre, mmin, tmmin, cnt)
                if hifi:
                    if mode[i] == LATCH:
                        c = 0.0
                    elif ovr:
                        c = 0.0
                    else:
                        c = F[F_GFF] * grade_acc[i] + F[F_RFF] * (
                            F[F_A0] + F[F_A1] * abs(vim[i])
                            + F[F_A2] * vim[i] * vim[i])
                    comp[i] = c
                    x = u + c
                    dumax = F[F_JERK] * T_u
                    if x < uc[i] - dumax:
                        x = uc[i] - dumax
                    elif x > uc[i] + dumax:
                        x = uc[i] + dumax
                    if mode[i] != LATCH:
                        jk = abs(x - uc[i]) / T_u
                        if jk > jpk[i]:
                            jpk[i] = jk
                    lo = -F[F_DERATE] * amax[i]
                    if x < lo:
                        x = lo
                    elif x > F[F_UACT]:
                        x = F[F_UACT]
                    uc[i] = x
                else:
                    uc[i] = u
                if mode[i] != LATCH and abs(uc[i]) > ucmx[i]:
                    ucmx[i] = abs(uc[i])
        # ------------------------------------ record
        if rec_every > 0 and k % rec_every == 0 and irec < nrec:
            rec[irec, 0] = t
            for i in range(n):
                rec[irec, 1 + i] = s[i]
                rec[irec, 1 + n + i] = v[i]
                rec[irec, 1 + 2 * n + i] = uc[i]
                rec[irec, 1 + 3 * n + i] = mode[i]
            irec += 1
        # ------------------------------------ send (post-event values)
        for i in range(n):
            usent[i, k] = uc[i] - comp[i]
            msent[i, k] = mode[i] >= S2
            csent[i, k] = closed[i]
        # ------------------------------------ end of the run
        done = True
        for i in range(n):
            if mode[i] != LATCH or v[i] > 0.0:   # latched and at true rest
                done = False
                break
        if done or k == N:
            k_end = k
            break
        # ------------------------------------ integrate to t_{k+1}
        for i in range(n):
            latb[i] = mode[i] == LATCH
        if hifi:
            ip = _step_hifi(t, dt, s, v, ua, uc, latb, tau_act, grade_acc,
                            F[F_A0], F[F_A1], F[F_A2], ell, s_m, gmin, tgmin,
                            tbuf, uamx, F[F_AHOLD], tstop, sstop)
            for i in range(n):
                if not latb[i]:
                    c = ccur[i]
                    for q in range(Q_N):
                        if c & (1 << q):
                            tim[q, i] += dt
            if ip >= 0:
                contact = t + dt
                cpair = ip
                k_end = k + 1
                break
        else:
            tcur = t
            t_end = (k + 1) * dt
            while True:
                rem = t_end - tcur
                if rem <= 0.0:
                    break
                bi = -1
                tau = rem
                for i in range(n):
                    if mode[i] != LATCH and v[i] > 0.0 and ucmd[i] < 0.0:
                        tz = -v[i] / ucmd[i]
                        if tz <= tau:
                            tau = tz
                            bi = i
                for i in range(n):
                    latb[i] = mode[i] == LATCH
                tc, ip = _pairs_sub(tcur, tau, s, v, ucmd, latb, ell, s_m,
                                    gmin, tgmin, tbuf)
                if ip >= 0:
                    for i in range(n):
                        if not latb[i]:
                            s[i] = s[i] + v[i] * tc + 0.5 * ucmd[i] * tc * tc
                            v[i] = v[i] + ucmd[i] * tc
                            cq = ccur[i]
                            for q in range(Q_N):
                                if cq & (1 << q):
                                    tim[q, i] += tc
                    contact = tcur + tc
                    cpair = ip
                    stop_all = True
                    break
                for i in range(n):
                    if not latb[i]:
                        s[i] = s[i] + v[i] * tau + 0.5 * ucmd[i] * tau * tau
                        v[i] = v[i] + ucmd[i] * tau
                        cq = ccur[i]
                        for q in range(Q_N):
                            if cq & (1 << q):
                                tim[q, i] += tau
                for i in range(1, n):
                    hb = s[i - 1] - s[i] - ell - s_m + v[i - 1] / bf[i - 1] \
                        - v[i] / bf[i]
                    if hb < hmin[i - 1]:
                        hmin[i - 1] = hb
                        thmin[i - 1] = tcur + tau
                if bi < 0:
                    break
                # ---- zero-speed instant of unit bi (inside the step)
                tcur = tcur + tau
                if mode[bi] == COAST or mode[bi] == S1:
                    nlat_pre += 1
                v[bi] = 0.0
                mode[bi] = LATCH
                tstop[bi] = tcur
                sstop[bi] = s[bi]
                ucmd[bi] = 0.0
                uc[bi] = 0.0
                ccur[bi] = 0
                q = bi + 1
                if q < n and (mode[q] == S2 or mode[q] == FF):
                    mode[q] = RULE
                    trule[q] = tcur
                    # event-triggered command update (right limit), exact
                    # sensing of the current state
                    sg[q] = s[q - 1] - s[q]
                    vpm[q] = v[q - 1]
                    vim[q] = v[q]
                    gap, vp, vi, e, eps, ff = _err(q, var, tcur, dt, s, sg,
                                                   vim, vpm, 0.0, 0.0, 0.0,
                                                   0.0, uh, kcl, js, SK, SR,
                                                   SD, SV, DPRE, d_s, F)
                    ovr = (lost and q == em) or wd[q]
                    u, un, uq, ucbf, bits = _cmd(q, var, mode[q], ovr, gap, vp,
                                                 vi, e, eps, ff, uh[q], pidI,
                                                 False, T_u, F, bf, Umin, amax,
                                                 filt)
                    ucmd[q] = u
                    uc[q] = u
                    ccur[q] = bits
                    _stats(q, u, un, ucbf, bits, mode[q], ovr, tcur, T_xi,
                           umn, umx, umn_pre, umx_pre, mmin, tmmin, cnt)
                    if abs(u) > ucmx[q]:
                        ucmx[q] = abs(u)
            if stop_all:
                k_end = k + 1
                break
    for i in range(n):
        mode_out[i] = mode[i]
    fout[O_CONTACT] = contact
    fout[O_TEND] = k_end * dt if np.isnan(contact) else contact
    iout[J_CPAIR] = cpair
    iout[J_KEND] = k_end
    iout[J_NREC] = irec
    iout[J_NLATPRE] = nlat_pre


# ================================================================== setup
def load_design(path=None):
    """design.json of the benchmark (read-only)."""
    import json
    if path is None:
        path = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')
    return json.load(open(path))


def design_cfg(cfg, lam=None, eps_e=None, eps_v=None, dbar=None,
               vhnd=None):
    """Copy of cfg with the design parameters set."""
    import json
    c = json.loads(json.dumps(cfg))
    if lam is not None:
        c['lam'] = float(lam)
    if eps_e is not None:
        c['eps_e'] = float(eps_e)
    if eps_v is not None:
        c['eps_v'] = float(eps_v)
    if dbar is not None:
        c['dbar'] = float(dbar)
    if vhnd is not None:
        c['vhnd'] = float(vhnd)
    return c


def _on_grid(x, h, name):
    q = float(x) / h
    k = int(round(q))
    if abs(q - k) > 1e-6 or k < 0:
        raise ValueError('%s = %.12g is not a nonnegative multiple of %g'
                         % (name, float(x), h))
    return k


def setup(cfg, lam=DEFAULTS['lam'], U=DEFAULTS['U'],
          Uminus=DEFAULTS['Uminus'], eps_e=DEFAULTS['eps_e'],
          eps_v=DEFAULTS['eps_v'], dbar=DEFAULTS['dbar'],
          vhnd=DEFAULTS['vhnd'], dt=1e-3):
    """Simulator inputs of the HR design of the Case D configuration cfg
    (design.json['cfg']) with the design parameters lam (beta = lam^2,
    gamma = 2 lam for every unit), U (traction limit), Uminus (braking
    limit), eps_e, eps_v (handoff box), dbar (age bound; constant
    transport dbar - T_m) and vhnd (handoff pad of the certificate; recorded
    only).  The reference and the schedules come from sim_hr.hr_setup
    (exact rational checks); the comparators 'plans' and 'mpc' use
    sim_bench.setup on the same cfg."""
    c = design_cfg(cfg, lam=lam, eps_e=eps_e, eps_v=eps_v, dbar=dbar,
                   vhnd=vhnd)
    T_m = float(c.get('T_c', dt))
    if abs(T_m - dt) > 1e-15:
        raise ValueError('sim_bench_hr: T_m = T_c = dt required')
    S = hr.hr_setup(c, U=U, Uminus=Uminus)
    n = int(S.n)
    # ---- head reference pieces (frozen convention of sim_hr)
    NH = len(S.head)
    HT = np.zeros(NH)
    HK = np.full(NH, -1, np.int64)
    HA = np.zeros(NH)
    HS = np.zeros(NH)
    HV = np.zeros(NH)
    for j, (t0, a, s0, v0) in enumerate(S.head):
        HA[j] = a
        HS[j] = s0
        HV[j] = v0
        HT[j] = t0
        if t0 <= S.T_xi + 1e-12:
            kj = _on_grid(t0, dt, 'head jump %d' % j)
            HK[j] = kj
            HT[j] = float(kj) * dt
    if HK[0] != 0:
        raise ValueError('the head reference must start at t = 0')
    # ---- follower schedules (design); a shaped FB setup has none (its
    # proposed controller is the variant 'feedback', 2026-09-29)
    no_sched = S.sched is None
    MS = 1 if no_sched else max(len(S.sched[i]['sigma_f']) for i in range(1, n))
    SK = np.zeros((n, MS), np.int64)
    SR = np.zeros((n, MS))
    SD = np.zeros((n, MS))
    SV = np.zeros((n, MS))
    NS = np.zeros(n, np.int64)
    for i in range(1, n):
        if no_sched:
            continue
        Sd = S.sched[i]
        NS[i] = len(Sd['sigma_f'])
        for j in range(NS[i]):
            SK[i, j] = _on_grid(Sd['sigma_f'][j], dt,
                                'schedule jump %d of unit %d' % (j, i))
            SR[i, j] = Sd['rho_f'][j]
            SD[i, j] = Sd['d_f'][j]
            SV[i, j] = Sd['dd_f'][j]
    DPRE = np.array(S.d_pre, float)
    DPRE[0] = 0.0
    k_xi = _on_grid(S.T_xi, dt, 'T_xi')
    kt0 = _on_grid(S.t_0, dt, 't_0')
    Ntr = int(round((float(c['dbar']) - T_m) / dt))
    if Ntr < 1:
        raise ValueError('transport must be at least one step')
    Bsb = sb.setup(c, dt=dt)
    B = dict(
        n=n, dt=dt, T_m=T_m, T_c=T_m, dbar=float(c['dbar']), Ntr=Ntr,
        transport=Ntr * dt, U=float(U), Uminus=np.array(S.Uminus, float),
        lam=float(c['lam']), beta=float(S.beta), gamma=float(S.gamma),
        alpha=float(S.alpha), phi=float(S.phi), eps_det=float(S.eps_det),
        eps_e=float(S.eps_e), eps_v=float(S.eps_v), a_b=float(S.a_b),
        kappa=float(S.kappa), s_m=float(S.s_m), ell=float(S.ell),
        v_c=float(S.v_c), amax=np.array(S.amax, float),
        bsl=np.array(S.b, float), d_s=np.array(S.d_s, float),
        c0=np.array(S.c0, float), s0=np.array(S.s0, float),
        v0=np.array(S.v0, float), t_0=float(S.t_0), T_xi=float(S.T_xi),
        k_xi=k_xi, kt0=kt0, tau_r=float(S.tau_r), v_c1=float(S.v_c1),
        s_ref_stop=float(S.s_ref_stop), markers=np.array(S.markers, float),
        HT=HT, HK=HK, HA=HA, HS=HS, HV=HV, NH=NH, SK=SK, SR=SR, SD=SD, SV=SV,
        NS=NS, DPRE=DPRE, S=S, cfg=c, sb=Bsb,
        t_join=[None] * n if no_sched else
        [None] + [float(S.sched[i]['t_join']) for i in range(1, n)],
        no_sched=no_sched,
        req=dict(disp=float(c.get('sync_tol', REQ['disp'])),
                 marker=float(c.get('align_tol', REQ['marker'])),
                 gap=float(c.get('gap_tol', REQ['gap']))),
        params=dict(lam=float(c['lam']), beta=float(S.beta),
                    gamma=float(S.gamma), U=float(U), Uminus=float(Uminus)
                    if np.ndim(Uminus) == 0 else [float(x) for x in Uminus],
                    eps_e=float(S.eps_e), eps_v=float(S.eps_v),
                    dbar=float(c['dbar']), T_m=T_m, T_c=T_m,
                    transport=Ntr * dt,
                    vhnd=float(c['vhnd']) if 'vhnd' in c else None))
    return B


def doubled_mismatch(B):
    """Takeover speeds with the increments v_i(0) - v_{i-1}(0) doubled and
    the head speed unchanged (sim_bench.doubled_mismatch)."""
    return sb.doubled_mismatch(dict(v0=B['v0']))


def with_gain(B, lam):
    """B with the common gain lam of the critically damped family (beta =
    lam^2, gamma = 2 lam) for every unit, the head included (the design
    gain returns B itself)."""
    lam = float(lam)
    if lam == B['lam']:
        return dict(B)
    Bs = dict(B['sb'], lam=lam, beta=lam * lam, gamma=2.0 * lam)
    # B['params'] stays the design; the run records its own lam, beta, gamma
    return dict(B, lam=lam, beta=lam * lam, gamma=2.0 * lam, sb=Bs)


# ==================================================================== run
def _none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _lst(a):
    return [_none(x) for x in np.asarray(a, float)]


def _fit(fifo, Nsend):
    """FIFO rows cover every send of the run (padded by the last value, as
    sim_hr._fit)."""
    f = np.asarray(fifo)
    if f.shape[1] >= Nsend:
        return f[:, :Nsend]
    pad = np.repeat(f[:, -1:], Nsend - f.shape[1], axis=1)
    return np.concatenate([f, pad], axis=1)


def _finish(B, out, R, s, v, t0):
    """Terminal indices and requirements (common to every variant)."""
    n = B['n']
    tstop = np.array([np.nan if x is None else x for x in out['tstop']],
                     float)
    fin = np.isfinite(tstop)
    contact = out.get('contact_time')
    all_stopped = bool(np.all(fin) and contact is None)
    marks = B['markers']
    gap = (s[:-1] - s[1:] - B['ell']) - B['d_s'][1:]
    mark = s - marks
    out.update(
        stop_order=[int(x) + 1 for x in np.argsort(tstop, kind='stable')]
        if all_stopped else None,
        all_stopped=all_stopped, n_stopped=int(np.sum(fin)),
        disp=float(np.max(tstop) - np.min(tstop)) if all_stopped else None,
        T_f=float(np.max(tstop)) if all_stopped else None,
        first_stop=float(np.min(tstop[fin])) if np.any(fin) else None,
        marker_signed=[float(x) for x in mark],
        marker_abs_max=float(np.max(np.abs(mark))),
        gap_signed=[float(x) for x in gap],
        gap_abs_max=float(np.max(np.abs(gap))),
        terminal_defined=all_stopped,
        s_final=[float(x) for x in s], v_final=[float(x) for x in v])
    req = B['req']
    out['pass_disp'] = bool(all_stopped and out['disp'] <= req['disp'])
    out['pass_marker'] = bool(all_stopped
                              and out['marker_abs_max'] <= req['marker'])
    out['pass_gap'] = bool(all_stopped and out['gap_abs_max'] <= req['gap'])
    out['pass_all'] = bool(all_stopped and contact is None
                           and out['pass_disp'] and out['pass_marker']
                           and out['pass_gap'])
    out['requirements'] = dict(req)
    out['runtime_s'] = time.time() - t0
    return out


def run(B, variant='proposed', plant='ideal', apply_filter=None,
        scenario=None, t_loss=None, emergency_unit=None, T_wd=None,
        hifi=None, seed=0, gains=None, h_w=None, v0=None, T_end=None,
        fifo=None, record=False, rec_dt=0.01, W=None, mpc_planner=None,
        mpc_stop='detector', hold_brake=True):
    """One closed-loop run; returns the metrics of the run (module
    docstring).  Units in every output are numbered from 1 (the head);
    emergency_unit likewise.  T_end defaults to T_xi + 60 s (MPC: + 120 s).
    fifo: per-send transport of every hop in units of T_m (row i: hop
    i-1 -> i), else the constant transport of B.  hold_brake (perturbed
    plant): after the latch, which acts on the measured speed, a holding
    brake decelerates the unit at a_b until its true speed is zero, and the
    stop instant and position are those of this true standstill; False
    rests the unit at once at the latch (the convention before
    2026-09-29)."""
    if variant in EXTERNAL:
        return _run_external(B, variant, plant=plant,
                             apply_filter=apply_filter, scenario=scenario,
                             t_loss=t_loss, emergency_unit=emergency_unit,
                             T_wd=T_wd, hifi=hifi, seed=seed, v0=v0,
                             T_end=T_end, fifo=fifo, record=record,
                             rec_dt=rec_dt, W=W, mpc_planner=mpc_planner,
                             mpc_stop=mpc_stop, hold_brake=hold_brake)
    if variant not in VARIANTS:
        raise ValueError('unknown variant %s' % variant)
    if plant not in ('ideal', 'hifi'):
        raise ValueError('plant: ideal or hifi')
    if plant == 'hifi' and scenario is not None:
        raise NotImplementedError('perturbed plant: nominal scenario only')
    if scenario not in (None, 'loss'):
        raise ValueError('scenario: None or loss')
    t0 = time.time()
    law, smode, fdef = VARIANTS[variant]
    # a shaped setup (FB design, 2026-09-29) has no schedule pieces, so no
    # closure onset is ever processed (kcl stays -1) and the desired
    # clearance is DPRE = d_s from takeover in every schedule mode: there
    # 'proposed' and 'feedback' run the same law
    if apply_filter is None:
        apply_filter = fdef
    if variant == 'nofilter' and apply_filter:
        raise ValueError('nofilter runs without the filter')
    n = B['n']
    dt = B['dt']
    H = dict(sb.HIFI_DEFAULT, **(hifi or {}))
    if plant == 'hifi' and float(H.get('ramp_lead', 0.0)) != 0.0:
        raise NotImplementedError('ramp_lead is not supported by the HR '
                                  'variants')
    hold = int(round(H['T_s'] / dt)) if plant == 'hifi' else 1
    if T_end is None:
        T_end = B['T_xi'] + 60.0
    N = int(round(T_end / dt))
    loss = scenario == 'loss'
    k_loss = 0
    em = -1
    k_wd = -1
    if loss:
        if t_loss is None:
            raise ValueError('scenario loss needs t_loss')
        k_loss = int(math.ceil(t_loss / dt - 1e-9))
        if emergency_unit is not None:
            em = int(emergency_unit) - 1
        if T_wd is not None:
            k_wd = int(round(T_wd / dt))
    elif emergency_unit is not None or T_wd is not None:
        raise ValueError('emergency braking and the watchdog belong to the '
                         'loss scenario')
    g = dict(sb.K_PID, **(gains or {}))
    hw = sb.K_CACC['h_w'] if h_w is None else float(h_w)
    # ---- schedules of the variant
    NS = B['NS'].copy()
    SK = B['SK'].copy()
    SR = B['SR'].copy()
    SD = B['SD'].copy()
    SV = B['SV'].copy()
    DPRE = B['DPRE'].copy()
    if smode == 'step':
        NS[:] = 0
        SK[:] = 0
        SR[:] = 0.0
        SD[:] = 0.0
        SV[:] = 0.0
        for i in range(1, n):
            NS[i] = 1
            SD[i, 0] = B['d_s'][i]
    elif smode == 'none':
        NS[:] = 0
        for i in range(1, n):
            DPRE[i] = B['d_s'][i]
    # ---- deliveries
    Ntot = N + 2
    if fifo is None:
        D, Nsend = sc.deliveries(n, Ntot, 1, Ntr=B['Ntr'])
        transport = B['Ntr'] * dt
    else:
        f = _fit(fifo, Ntot)
        D, Nsend = sc.deliveries(n, Ntot, 1, fifo=f)
        transport = None
    D = np.ascontiguousarray(D, np.int64)
    rec_every = max(1, int(round(rec_dt / dt))) if record else 0
    nrec = N // rec_every + 2 if record else 1
    I = np.zeros(I_NI, np.int64)
    I[I_N] = n
    I[I_NSTEP] = N
    I[I_NC] = int(round(B['T_c'] / dt))
    I[I_HOLD] = hold
    I[I_VAR] = law
    I[I_FILT] = 1 if apply_filter else 0
    I[I_PLANT] = 1 if plant == 'hifi' else 0
    I[I_LOSS] = 1 if loss else 0
    I[I_KLOSS] = k_loss
    I[I_EM] = em
    I[I_KWD] = k_wd
    I[I_KXI] = B['k_xi']
    I[I_KT0] = B['kt0']
    I[I_NH] = B['NH']
    I[I_REC] = rec_every
    I[I_NREC] = nrec
    I[I_NSEND] = Nsend
    F = np.zeros(F_NF)
    F[F_DT] = dt
    F[F_AB] = B['a_b']
    F[F_ALPHA] = B['alpha']
    F[F_BETA] = B['beta']
    F[F_GAMMA] = B['gamma']
    F[F_PHI] = B['phi']
    F[F_EPSDET] = B['eps_det']
    F[F_KAPPA] = B['kappa']
    F[F_ELL] = B['ell']
    F[F_SM] = B['s_m']
    F[F_U] = B['U']
    F[F_KP] = g['Kp']
    F[F_KD] = g['Kd']
    F[F_KI] = g['Ki']
    F[F_IMAX] = g['Imax']
    F[F_HW] = hw
    F[F_UACT] = sb.U_ACT
    F[F_DERATE] = H['derate']
    F[F_JERK] = H['jerk_max']
    F[F_A0], F[F_A1], F[F_A2] = [float(x) for x in H['davis']]
    F[F_NGAP] = H['noise_gap']
    F[F_NVEL] = H['noise_vel']
    F[F_BIAS] = H['bias_gap']
    F[F_QUANT] = H['quant_vel']
    F[F_VSTOP] = H['v_stop']
    F[F_GFF] = H['grade_ff']
    F[F_RFF] = H['res_ff']
    F[F_TXI] = B['k_xi'] * dt
    F[F_AHOLD] = float(B['a_b']) if (plant == 'hifi' and hold_brake) else 0.0
    tau_act = np.broadcast_to(np.asarray(H['tau_act'], float),
                              (n,)).astype(float).copy()
    grade_acc = float(H['grade']) * np.array(sb.GRADE_SIGNS[:n])
    if plant == 'hifi':
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((N // hold + 2, n, 4))
    else:
        Z = np.zeros((1, n, 4))
    s = B['s0'].copy()
    v = np.array(B['v0'] if v0 is None else v0, float).copy()
    nan = np.nan
    tstop = np.full(n, nan)
    sstop = np.full(n, nan)
    tact = np.full(n, nan)
    if law != K_NOFB:
        tact[0] = 0.0
    tset = np.full(n, nan)
    tcl = np.full(n, nan)
    trule = np.full(n, nan)
    twd = np.full(n, nan)
    gmin = np.full(n - 1, np.inf)
    tgmin = np.full(n - 1, nan)
    hmin = np.full(n - 1, np.inf)
    thmin = np.full(n - 1, nan)
    mmin = np.full(n - 1, np.inf)
    tmmin = np.full(n - 1, nan)
    tbuf = np.full(n - 1, -1.0)
    umn = np.full(n, np.inf)
    umx = np.full(n, -np.inf)
    umn_pre = np.full(n, np.inf)
    umx_pre = np.full(n, -np.inf)
    uamx = np.zeros(n)
    ucmx = np.zeros(n)
    jpk = np.zeros(n)
    tim = np.zeros((Q_N, n))
    cnt = np.zeros((Q_N + 1, n), np.int64)
    e_hnd = np.full(n, nan)
    eps_hnd = np.full(n, nan)
    fout = np.zeros(O_NO)
    iout = np.zeros(J_NI, np.int64)
    rec = np.zeros((nrec, 1 + 4 * n))
    mode_out = np.zeros(n, np.int64)
    _core(I, F, B['amax'], B['Uminus'], B['bsl'], B['d_s'], s, v, D,
          B['HT'], B['HK'], B['HA'], B['HS'], B['HV'], NS, SK, SR, SD, SV,
          DPRE, Z, tau_act, grade_acc, tstop, sstop, tact, tset, tcl, trule,
          twd, gmin, tgmin, hmin, thmin, mmin, tmmin, tbuf, umn, umx,
          umn_pre, umx_pre, uamx, ucmx, jpk, tim, cnt, e_hnd, eps_hnd, fout,
          iout, rec, mode_out)
    contact = _none(fout[O_CONTACT])
    ulo = [float(-x) for x in B['Uminus']]
    out = dict(
        variant=variant, kernel_law=int(law), schedule_mode=smode,
        plant=plant, apply_filter=bool(apply_filter), scenario=scenario,
        t_loss=t_loss, emergency_unit=emergency_unit, T_wd=T_wd,
        seed=int(seed) if plant == 'hifi' else None,
        hifi=None if plant != 'hifi'
        else {k: (list(x) if isinstance(x, (list, tuple, np.ndarray))
                  else x) for k, x in H.items()},
        lam=B['lam'], beta=B['beta'], gamma=B['gamma'],
        gains=dict(g) if variant == 'pid' else None,
        h_w=hw if variant == 'cacc' else None,
        v0=[float(x) for x in (B['v0'] if v0 is None else v0)],
        T_end=float(T_end), input_set=[ulo, B['U']],
        transport=transport, fifo=fifo is not None,
        design_params=dict(B['params']), control_period=hold * dt,
        simulator='sim_bench_hr',
        hold_brake=bool(plant == 'hifi' and hold_brake))
    out['tstop'] = _lst(tstop)
    out['contact_time'] = contact
    out['contact_pair'] = (int(iout[J_CPAIR]) + 1) if contact is not None \
        else None
    out.update(
        gmin=_lst(gmin), t_gmin=_lst(tgmin),
        gmin_min=float(np.min(gmin)),
        gmin_pair=int(np.argmin(gmin)) + 2,
        hmin=_lst(hmin), t_hmin=_lst(thmin), hmin_min=float(np.min(hmin)),
        mmin=_lst(mmin), t_mmin=_lst(tmmin),
        u_min_unit=_lst(umn), u_max_unit=_lst(umx),
        u_min=_none(np.min(umn)), u_max=_none(np.max(umx)),
        u_abs_max=_none(max(abs(float(np.min(umn))),
                            abs(float(np.max(umx))))),
        u_min_pre_unit=_lst(umn_pre), u_max_pre_unit=_lst(umx_pre),
        n_updates=[int(x) for x in cnt[Q_N]],
        buffer_lost=bool(np.any(tbuf >= 0.0)),
        t_buffer_lost=[(float(x) if x >= 0.0 else None) for x in tbuf],
        tact=_lst(tact), tset=_lst(tset), tcl=_lst(tcl),
        trule=_lst(trule),
        rule_entered=bool(np.any(np.isfinite(trule))),
        rule_units=[int(i) + 1 for i in range(n) if np.isfinite(trule[i])],
        t_watchdog=_lst(twd),
        watchdog_units=[i + 1 for i in range(n) if np.isfinite(twd[i])],
        override_units=[i + 1 for i in range(n)
                        if cnt[Q_OVR, i] > 0],
        mode_final=[MODE_NAMES[int(m)] for m in mode_out],
        e_hnd=_lst(e_hnd), eps_hnd=_lst(eps_hnd),
        in_box_at_Txi=bool(np.all(np.abs(e_hnd) <= B['eps_e'])
                           and np.all(np.abs(eps_hnd) <= B['eps_v']))
        if np.all(np.isfinite(e_hnd)) else None,
        n_latch_before_S2=int(iout[J_NLATPRE]),
        t_end_sim=float(fout[O_TEND]),
        u_act_cmd_abs_max=[float(x) for x in ucmx])
    for q, nm in enumerate(Q_NAMES):
        out[nm] = [float(x) for x in tim[q]]
    if plant == 'hifi':
        out['u_realized_abs_max'] = [float(x) for x in uamx]
        out['jerk_pk'] = [float(x) for x in jpk]
    if record:
        m = int(iout[J_NREC])
        out['rec'] = dict(t=rec[:m, 0].copy(), s=rec[:m, 1:1 + n].T.copy(),
                          v=rec[:m, 1 + n:1 + 2 * n].T.copy(),
                          u=rec[:m, 1 + 2 * n:1 + 3 * n].T.copy(),
                          mode=rec[:m, 1 + 3 * n:1 + 4 * n].T.copy())
    return _finish(B, out, None, s, v, t0)


# =========================================================== comparators
def mpc_selection(path=None):
    """Weights and planner of the cooperative MPC selected by the extended
    protocol of the previous study (data/new_d19/caseD/studies/mpc.json,
    read-only)."""
    import json
    if path is None:
        path = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'studies',
                            'mpc.json')
    M = json.load(open(path))
    P = M['protocols']['extended']
    return dict(W=dict(P['best']['W']), planner=dict(P['best']['planner']),
                source=os.path.relpath(path, CODE).replace('\\', '/'),
                protocol='extended', score=P['best']['score'],
                input_set_of_tuning=M['protocol']['input_bounds'])


def _run_external(B, variant, plant='ideal', apply_filter=None,
                  scenario=None, t_loss=None, emergency_unit=None, T_wd=None,
                  hifi=None, seed=0, v0=None, T_end=None, fifo=None,
                  record=False, rec_dt=0.01, W=None, mpc_planner=None,
                  mpc_stop='detector', hold_brake=True):
    """'plans' (sim_bench 'proposed') and 'mpc' (sim_bench MPC) with the
    input set [-U_i^-, U]; the outputs are mapped to the names of run()."""
    if fifo is not None:
        raise NotImplementedError('sim_bench: constant transport only')
    t0 = time.time()
    Bs = B['sb']
    n = B['n']
    bounds = ([float(-x) for x in B['Uminus']], float(B['U']))
    # the record at every step gives the time with u >= 0 on the ideal plant
    rec_all = plant == 'ideal'
    if variant == 'plans':
        if plant == 'hifi' and float((hifi or {}).get('ramp_lead', 0.0)):
            raise NotImplementedError('ramp_lead not used in these studies')
        R = sb.run(Bs, 'proposed', plant=plant, apply_filter=apply_filter,
                   scenario=scenario, t_loss=t_loss,
                   emergency_unit=emergency_unit, T_wd=T_wd, hifi=hifi,
                   seed=seed, v0=v0, T_end=T_end, record=rec_all or record,
                   rec_dt=B['dt'] if rec_all else rec_dt, bounds=bounds,
                   hold_brake=hold_brake)
        R['sim_bench_variant'] = 'proposed'
    else:
        if scenario is not None:
            raise NotImplementedError('mpc: nominal scenario only')
        if T_end is None:
            T_end = B['T_xi'] + 120.0
        R = sb.run(Bs, 'mpc', plant=plant, hifi=hifi, seed=seed,
                   apply_filter=True if apply_filter is None
                   else apply_filter, v0=v0, T_end=T_end, W=W,
                   mpc_planner=mpc_planner, mpc_stop=mpc_stop,
                   record=rec_all or record,
                   rec_dt=B['dt'] if rec_all else rec_dt, bounds=bounds,
                   hold_brake=hold_brake)
        R['sim_bench_variant'] = 'mpc'
        R['information'] = (
            'absolute position and speed of the unit; head: planned position '
            'and speed of unit 1 (the HR reference minus the head slot); '
            'follower: predicted trajectory of the predecessor over the '
            'horizon, received one control period (0.1 s) late; filter: '
            'received applied command of the predecessor; no follower plans, '
            'no schedules')
    R = dict(R)
    R['variant'] = variant
    R['simulator'] = 'sim_bench'
    R['t_atU'] = R.pop('t_at0')
    R['t_clipU'] = R.pop('t_clip0')
    R['t_nom_above_U'] = R.pop('t_nom_pos')
    if rec_all:
        rc = R['rec']
        u = np.asarray(rc['u'])
        md = np.asarray(rc['mode'])
        R['t_nonneg'] = [float(np.sum((md[i] != sb.LATCH) & (u[i] >= 0.0))
                               * B['dt']) for i in range(n)]
        R['t_nonneg_note'] = 'from the record at every step (ideal plant)'
        if not record:
            R.pop('rec')
    else:
        R['t_nonneg'] = None
        R['t_nonneg_note'] = ('not available: sim_bench records the '
                              'actuator command on the perturbed plant')
    R['lam'] = float(Bs['lam'])
    R['beta'] = float(Bs['beta'])
    R['gamma'] = float(Bs['gamma'])
    R['design_params'] = dict(B['params'])
    R['transport'] = float(Bs['transport'])
    R['runtime_s'] = time.time() - t0
    return R


if __name__ == '__main__':
    D_ = load_design()
    B_ = setup(D_['cfg'])
    for var_ in ('proposed', 'nofilter', 'feedback', 'plans'):
        R_ = run(B_, variant=var_)
        print(var_, 'tstop', R_['tstop'], 'disp', R_['disp'])
        print('   marker', R_['marker_signed'])
        print('   gap', R_['gap_signed'], 'gmin', R_['gmin'])
        print('   tset', R_['tset'], 'u range', R_['u_min'], R_['u_max'],
              'runtime %.2f s' % R_['runtime_s'])
