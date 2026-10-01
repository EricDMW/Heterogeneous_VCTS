#!/usr/bin/env python3
"""sim_bench.py -- study simulator of the planned, deceleration-only arrival
(Case D benchmark): controller variants, perturbed plant, communication
loss, clock offsets and the cooperative arrival MPC (2026-09-28).

The module shares no code with sim_cont.py, sim_root2.py, sim_hifi.py or
sim_mpc.py.  It reads the design through certify_plan.plan(cfg) (read-only).

Model
  ds_i/dt = v_i, dv_i/dt = u_i; net acceleration command of every law in the
  input set [-k, 0]; five units of length ell, unit 1 (index 0) is the head.
  Every unit stores the planned trajectory (s*_i, v*_i, a*_i) of the design
  (piecewise-constant a*_i, common time base, braking at -a_b from T_xi).

Time stepping
  plant step dt = 1 ms; message period T_m = detector period T_c = dt;
  constant transport delay of (dbar - T_m)/dt = 19 steps, so the held
  information age reaches dbar = 20 ms before each delivery.
  plant 'ideal': the command is computed at every grid instant k dt from
    the state at that instant, after the events of that instant, and held
    for one step (pure 1 kHz zero-order hold, T_u = dt).  The double
    integrator is integrated exactly over the step (s += v dt + u dt^2/2,
    v += u dt).  A zero-speed instant inside a step is located exactly
    (tau = -v/u); the unit is latched there with its position at that
    instant.  The clearance of every pair is piecewise quadratic inside a
    step; its minimum, the first instant with g < 0 (buffer lost) and the
    first instant with c <= 0 (contact) are computed in closed form.
  plant 'hifi': the perturbed plant of sim_hifi.py (parameters DEFAULT):
    control every T_s = 10 ms on measured signals, first-order actuator
    lag, jerk limit on the held command, running resistance, per-unit grade
    acceleration with the signs (+,+,-,-,+), braking capability derated,
    Gaussian ranging and speed noise, ranging bias, speed quantization,
    standstill threshold with latch.  Plant integration as in sim_hifi.py
    (explicit Euler for the lag, v <- max(0, v + a dt), s += v dt).
  Event order at a control instant: LATCH, RULE, packet batch (held value,
  flag, activation), planned jump (piece switch, head switch to BRAKE),
  detector sample, command update, send.

Controller variants (argument variant of run())
  'proposed'  the law of case_d.tex Sec. D.II-A: pair errors relative to the
              plans; COAST a*_i + received deviation; S1 adds
              alpha sat(eps_i/phi); S2 adds beta e_i + gamma eps_i; the head
              applies S1/S2 to its errors relative to its own plan and brakes
              open loop at -a_b from T_xi (BRAKE); followers stay in S2
              through T_xi; RULE (-a_b) when the predecessor is sensed at
              standstill; LATCH at the first zero-speed instant.  The message
              carries the deviation u*_i - a*_i formed at the send instant
              and the matching flag.  Filter: sat_[-k,0](min{u_nom, u_cbf})
              with the estimate a*_{i-1}(t) + received deviation, fallback
              -k when u_cbf < -k.
  'nofilter'  'proposed' with the filter removed (input clip kept).
  'nodev'     'proposed' with the received deviation replaced by zero in the
              law and in the filter estimate; flags are delivered as before.
  'planonly'  every unit applies its planned acceleration a*_i(t) until it
              stops (no feedback, no message, no RULE); filter with the
              estimate a*_{i-1}(t).
  'single'    'proposed' without the S1 stage of the followers (S2 from
              activation; the head keeps its S1 stage, as in sim_root2.py).
  'pid'       PID on the pair errors relative to the plans from activation,
              f = Kp e + Kd eps + Ki I, dI/dt = e with |Ki I| <= Imax, with
              the planned feedforward and the received deviation (the head
              runs S1 and then the PID law, as in sim_root2.py).
  'feedback'  closure by feedback, no plans at the followers: pair errors
              relative to the parking gaps, two-stage law with the received
              applied command as feedforward and in the filter; the head
              tracks its own plan and brakes at -a_b from T_xi.
  'cacc'      as 'feedback', single stage, with the spacing target
              d_s_i + h_w v_i.
  'mpc'       cooperative arrival MPC adapted from sim_mpc.py (run_mpc).

Scenarios: communication loss with optional emergency braking of one unit
and a per-unit loss watchdog; clock offsets of the units (multiples of dt).

Numba is required (the kernels are compiled; the cache goes to the
temporary directory, never next to the sources).
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR',
                      os.path.join(tempfile.gettempdir(),
                                   'numba_cache_sim_bench'))
from numba import njit                                   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..'))

# ------------------------------------------------------------------ modes
COAST, S1, S2, RULE, LATCH, BRAKE, PLAN = 0, 1, 2, 3, 4, 5, 6
MODE_NAMES = ('COAST', 'S1', 'S2', 'RULE', 'LATCH', 'BRAKE', 'PLAN')

# --------------------------------------------------------------- variants
V_PROP, V_NODEV, V_PLANONLY, V_SINGLE, V_PID, V_FEEDBACK, V_CACC = range(7)
VARIANTS = dict(proposed=V_PROP, nofilter=V_PROP, nodev=V_NODEV,
                planonly=V_PLANONLY, single=V_SINGLE, pid=V_PID,
                feedback=V_FEEDBACK, cacc=V_CACC)

# gains of the comparators (values of sim_root2.py)
K_PID = dict(Kp=0.003, Kd=0.115, Ki=5e-5, Imax=0.05)
K_CACC = dict(h_w=0.5)

# perturbed plant (values of sim_hifi.DEFAULT)
HIFI_DEFAULT = dict(T_s=0.010, tau_act=0.30, jerk_max=0.60,
                    davis=(0.005, 0.0004, 0.00012), grade=0.003,
                    derate=0.90, noise_gap=0.05, noise_vel=0.02,
                    bias_gap=0.02, quant_vel=0.01, v_stop=0.05,
                    grade_ff=0.0, res_ff=0.0, ramp_lead=0.0)
GRADE_SIGNS = (1.0, 1.0, -1.0, -1.0, 1.0)
U_ACT = 1.6                     # upper limit of the actuator command [m/s^2]

# cooperative arrival MPC (values of sim_mpc.py)
MPC_DEFAULT_W = dict(w_e=3.0, w_v=1.0, w_u=1.0, w_du=0.3)
MPC_SIG = dict(e=1.0, v=1.0, u=1.0, du=0.5)
T_MPC = 0.10
N_H = 20
V_STOP_MPC = 0.02
TAU_C = 5.0
DV_MAX = 0.5

REQ = dict(disp=1.0, marker=3.0, gap=1.0)

# ------------------------------------------------------- parameter slots
(I_N, I_NSTEP, I_NTR, I_NC, I_HOLD, I_VAR, I_FILT, I_PLANT, I_LOSS, I_KLOSS,
 I_EM, I_KWD, I_KXI, I_LEAD, I_REC, I_NREC, I_NI) = range(17)
(F_DT, F_K, F_AB, F_ALPHA, F_BETA, F_GAMMA, F_PHI, F_EPSDET, F_KAPPA, F_ELL,
 F_SM, F_TXI, F_KP, F_KD, F_KI, F_IMAX, F_HW, F_UACT, F_DERATE, F_JERK,
 F_A0, F_A1, F_A2, F_NGAP, F_NVEL, F_BIAS, F_QUANT, F_VSTOP, F_GFF, F_RFF,
 F_TAUSTAR, F_VSTOPMPC, F_UHI, F_AHOLD, F_NF) = range(35)
# counters (rows of cnt): control updates of a unit before its latch
(C_UPD, C_NOMPOS, C_CLIP0, C_CLIPK, C_BAR, C_BARRAW, C_INFEAS, C_OVR,
 C_AT0, C_NC) = range(10)
# fout slots
(O_CONTACT, O_TEND, O_NO) = range(3)
# iout slots
(J_CPAIR, J_KEND, J_NREC, J_NEG, J_NJ) = range(5)


# ================================================================ kernels
@njit(cache=True)
def _pst(i, q, dt, NPc, PK, PA, PS, PV):
    """Planned (s*, v*, a*) of unit i at the local step q (time q dt),
    right-continuous; the first piece is continued to negative times and
    the last one (braking at -a_b) beyond the planned stop."""
    hi = NPc[i] - 1
    if q >= PK[i, hi]:
        j = hi
    else:
        lo = 0
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if PK[i, mid] <= q:
                lo = mid
            else:
                hi = mid
        j = lo
    tau = q * dt - PK[i, j] * dt
    a = PA[i, j]
    return (PS[i, j] + PV[i, j] * tau + 0.5 * a * tau * tau,
            PV[i, j] + a * tau, a)


@njit(cache=True)
def _first_below(q0, d0, r, L):
    """First tau in [0, L] with q0 + d0 tau + r tau^2/2 <= 0 for q0 > 0;
    -1.0 if the quadratic stays positive on [0, L]."""
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
def _pairs_ideal(t, dt, s, v, acc, tz, lat, ell, s_m, gmin, tgmin, tbuf):
    """Exact clearance statistics of every pair over the step [t, t + dt]
    of the ideal plant: unit i moves with the held acceleration acc[i]
    until its zero-speed instant tz[i] (inf: none in the step) and rests
    afterwards.  Updates the smallest clearance margin and its instant and
    the first instant with g < 0; returns the earliest contact instant
    (offset in the step) and its pair index, (-1, -1) if none."""
    n = s.shape[0]
    tc_best = -1.0
    ip_best = -1
    bp = np.empty(4)
    for i in range(1, n):
        g0 = s[i - 1] - s[i] - ell - s_m
        vp = 0.0 if lat[i - 1] else v[i - 1]
        vf = 0.0 if lat[i] else v[i]
        ap = 0.0 if lat[i - 1] else acc[i - 1]
        af = 0.0 if lat[i] else acc[i]
        tzp = tz[i - 1]
        tzf = tz[i]
        nb = 0
        bp[nb] = 0.0
        nb += 1
        a1 = tzp if tzp < tzf else tzf
        a2 = tzf if tzp < tzf else tzp
        if a1 < dt and a1 > 0.0:
            bp[nb] = a1
            nb += 1
        if a2 < dt and a2 > a1:
            bp[nb] = a2
            nb += 1
        bp[nb] = dt
        nb += 1
        if g0 < gmin[i - 1]:
            gmin[i - 1] = g0
            tgmin[i - 1] = t
        done = False
        for q in range(nb - 1):
            ta = bp[q]
            L = bp[q + 1] - ta
            d0 = vp - vf
            pm = ta < tzp and not lat[i - 1]
            fm = ta < tzf and not lat[i]
            r = (ap if pm else 0.0) - (af if fm else 0.0)
            if not pm:
                d0 = -vf
            if not fm:
                d0 = vp if pm else 0.0
            # minimum on the sub-interval
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
                tgmin[i - 1] = t + ta + tm
            if tbuf[i - 1] < 0.0:
                if g0 < 0.0:
                    tbuf[i - 1] = t + ta
                else:
                    tb = _first_below(g0, d0, r, L) if g0 > 0.0 else 0.0
                    if tb >= 0.0:
                        tbuf[i - 1] = t + ta + tb
            c0 = g0 + s_m
            if c0 <= 0.0:
                tcn = ta
            else:
                tcn = _first_below(c0, d0, r, L)
                if tcn >= 0.0:
                    tcn = ta + tcn
            if tcn >= 0.0:
                if tc_best < 0.0 or tcn < tc_best:
                    tc_best = tcn
                    ip_best = i
                done = True
            if done:
                break
            # state of the pair at the end of the sub-interval
            if pm:
                vp = vp + ap * L
                if bp[q + 1] >= tzp:
                    vp = 0.0
            if fm:
                vf = vf + af * L
                if bp[q + 1] >= tzf:
                    vf = 0.0
            g0 = g1
    return tc_best, ip_best


@njit(cache=True)
def _advance_ideal(t, tau, s, v, acc, tz, lat, tstop, sstop):
    """Exact motion of every unit over [t, t + tau] with the held
    accelerations; units whose zero-speed instant lies in the interval are
    latched there."""
    n = s.shape[0]
    for i in range(n):
        if lat[i]:
            continue
        if tz[i] <= tau:
            z = tz[i]
            s[i] = s[i] + v[i] * z + 0.5 * acc[i] * z * z
            v[i] = 0.0
            lat[i] = True
            tstop[i] = t + z
            sstop[i] = s[i]
        else:
            s[i] = s[i] + v[i] * tau + 0.5 * acc[i] * tau * tau
            v[i] = v[i] + acc[i] * tau


@njit(cache=True)
def _zero_times(dt, v, acc, lat, tz):
    n = v.shape[0]
    neg = 0
    for i in range(n):
        tz[i] = np.inf
        if lat[i]:
            continue
        if v[i] <= 0.0:
            # at rest at the start of the step without a latch (takeover
            # speed zero or a positive acceleration afterwards)
            if acc[i] <= 0.0:
                tz[i] = 0.0
            continue
        if acc[i] < 0.0:
            z = -v[i] / acc[i]
            if z <= dt:
                tz[i] = z
    return neg


@njit(cache=True)
def _step_hifi(t, dt, s, v, ua, uc, lat, tau_act, grade_acc, a0, a1, a2,
               ell, s_m, gmin, tgmin, tbuf, uamx, a_hold, tstop, sstop):
    """One step of the perturbed plant (sim_hifi.py): actuator lag
    (explicit Euler), resistance and grade, v <- max(0, v + a dt),
    s += v dt; clearance statistics at the end of the step.  A latched unit
    that still moves decelerates at the holding-brake rate a_hold > 0, with
    resistance and grade, until its true speed is zero, and its stop instant
    and position are recorded then (a_hold = 0: the unit rests at once at
    the latch, as before 2026-09-29).  Returns the index of the first pair
    in contact (-1: none)."""
    n = s.shape[0]
    for i in range(n):
        if tau_act[i] > 0.0:
            ua[i] += dt / tau_act[i] * (uc[i] - ua[i])
        else:
            ua[i] = uc[i]
        if lat[i]:
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
        sg = 0.0
        if v[i] > 0.0:
            sg = 1.0
        elif v[i] < 0.0:
            sg = -1.0
        a = ua[i] - sg * (a0 + a1 * v[i] + a2 * v[i] * v[i]) - grade_acc[i]
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
def _core(I, F, amax, ulo, bsl, d_s, s, v, NPc, PK, PA, PS, PV, dkc, Z,
          tau_act, grade_acc, tstop, sstop, tact, tset, trule, gmin, tgmin,
          hmin,
          thmin, mmin, tmmin, tbuf, umn, umx, uamx, ucmx, jpk, cnt, fout,
          iout, rec, mode_out, twd):
    """Closed loop of every variant but the MPC on the ideal or the
    perturbed plant (module docstring)."""
    n = I[I_N]
    N = I[I_NSTEP]
    Ntr = I[I_NTR]
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
    lead = I[I_LEAD]
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    dt = F[F_DT]
    uhi = F[F_UHI]
    a_b = F[F_AB]
    alpha = F[F_ALPHA]
    beta = F[F_BETA]
    gamma = F[F_GAMMA]
    phi = F[F_PHI]
    eps_det = F[F_EPSDET]
    kappa = F[F_KAPPA]
    ell = F[F_ELL]
    s_m = F[F_SM]
    T_xi = F[F_TXI]
    Kp = F[F_KP]
    Kd = F[F_KD]
    Ki = F[F_KI]
    Imax = F[F_IMAX]
    h_w = F[F_HW]
    T_u = hold * dt

    plan_err = var == V_PROP or var == V_NODEV or var == V_SINGLE \
        or var == V_PID
    two_stage = var == V_PROP or var == V_NODEV or var == V_FEEDBACK
    msg_dev = plan_err
    msg_cmd = var == V_FEEDBACK or var == V_CACC
    has_msg = msg_dev or msg_cmd

    mode = np.zeros(n, np.int64)
    if var == V_PLANONLY:
        for i in range(n):
            mode[i] = PLAN
    else:
        mode[0] = S1
    lat = np.zeros(n, np.bool_)
    kact = np.full(n, -1, np.int64)
    kact[0] = 0
    tact[0] = 0.0
    uh = np.zeros(n)
    rflag = np.zeros(n, np.bool_)
    flag = np.zeros(n, np.bool_)
    unet = np.zeros(n)          # held net command of the law
    uc = np.zeros(n)            # held actuator command
    ua = np.zeros(n)            # actuator state (perturbed plant)
    comp = np.zeros(n)
    pidI = np.zeros(n)
    wd = np.zeros(n, np.bool_)
    k_rx = np.zeros(n, np.int64)
    sent_val = np.zeros((n, N + 2))
    sent_flag = np.zeros((n, N + 2), np.bool_)
    sg = np.zeros(n)
    vim = np.zeros(n)
    vpm = np.zeros(n)
    acc = np.zeros(n)
    tz = np.full(n, np.inf)
    pe0 = 0.0
    lost = False
    contact = np.nan
    cpair = -1
    irec = 0
    k_end = N
    neg = 0

    for k in range(N + 1):
        t = k * dt
        ctrl = k % hold == 0
        # ------------------------------------------- barrier at t_k
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            hb = g + v[i - 1] / bsl[i - 1] - v[i] / bsl[i]
            if hb < hmin[i - 1]:
                hmin[i - 1] = hb
                thmin[i - 1] = t
            if hifi and k == 0:
                gmin[i - 1] = g
                tgmin[i - 1] = 0.0
        # ------------------------------------------- loss onset
        if loss and (not lost) and k >= k_loss:
            lost = True
            for i in range(1, n):
                uh[i] = 0.0
        if ctrl:
            # ---------------------------------------- measurements
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
                        vpm[i] = np.round((v[i - 1] + F[F_NVEL] * Z[jz, i, 2])
                                          / qv) * qv
            else:
                for i in range(n):
                    vim[i] = v[i]
                    if i >= 1:
                        sg[i] = s[i - 1] - s[i]
                        vpm[i] = v[i - 1]
            # ---------------------------------------- (1) LATCH
            if hifi:
                for i in range(n):
                    md = mode[i]
                    if (not lat[i]) and (md == S2 or md == RULE
                                         or md == BRAKE) \
                            and vim[i] <= F[F_VSTOP] and t > T_xi:
                        lat[i] = True
                        mode[i] = LATCH
                        if not (F[F_AHOLD] > 0.0 and v[i] > 0.0):
                            tstop[i] = t
                            sstop[i] = s[i]
                            v[i] = 0.0
                        flag[i] = True
            # ---------------------------------------- (2) RULE
            if var != V_PLANONLY:
                for i in range(1, n):
                    if mode[i] == S2:
                        if hifi:
                            if vpm[i] <= eps_det and t > T_xi:
                                mode[i] = RULE
                                trule[i] = t
                        elif lat[i - 1]:
                            mode[i] = RULE
                            trule[i] = t
            # ---------------------------------------- (3) packet batch
            if has_msg and not lost:
                j = k - Ntr
                if j >= 0:
                    for i in range(1, n):
                        uh[i] = sent_val[i - 1, j]
                        if sent_flag[i - 1, j]:
                            rflag[i] = True
                        k_rx[i] = k
            if has_msg:
                for i in range(1, n):
                    if mode[i] == COAST and rflag[i]:
                        kact[i] = k
                        tact[i] = t
                        if two_stage:
                            mode[i] = S1
                        else:
                            mode[i] = S2
                            tset[i] = t
                            flag[i] = True
            # ---------------------------------------- (4) planned jump
            if var != V_PLANONLY:
                if (mode[0] == S1 or mode[0] == S2) \
                        and k + dkc[0] + lead >= k_xi:
                    mode[0] = BRAKE
                    flag[0] = True
            # ---------------------------------------- (5) detector sample
            for i in range(n):
                if mode[i] == S1 and (k - kact[i]) % Nc == 0:
                    q = k + dkc[i]
                    pso, pvo, pao = _pst(i, q, dt, NPc, PK, PA, PS, PV)
                    if i == 0:
                        x = pvo - vim[0]
                    elif plan_err:
                        psp, pvp, pap = _pst(i - 1, q, dt, NPc, PK, PA, PS,
                                             PV)
                        x = (vpm[i] - vim[i]) - (pvp - pvo)
                    else:
                        x = vpm[i] - vim[i]
                    if abs(x) <= eps_det:
                        mode[i] = S2
                        flag[i] = True
                        tset[i] = t
            # ---------------------------------------- (6) command update
            for i in range(n):
                if lat[i]:
                    unet[i] = 0.0
                    comp[i] = 0.0
                    if hifi:
                        dumax = F[F_JERK] * T_u
                        x = 0.0
                        if x < uc[i] - dumax:
                            x = uc[i] - dumax
                        elif x > uc[i] + dumax:
                            x = uc[i] + dumax
                        uc[i] = x
                    else:
                        uc[i] = 0.0
                    continue
                # loss watchdog and emergency braking (override)
                ovr = False
                if lost and i == em:
                    ovr = True
                elif k_wd >= 0 and i >= 1 and has_msg \
                        and k - k_rx[i] >= k_wd:
                    if not wd[i]:
                        wd[i] = True
                        twd[i] = t
                if wd[i]:
                    ovr = True
                if ovr:
                    unet[i] = -amax[i]
                    uc[i] = -amax[i]
                    comp[i] = 0.0
                    cnt[C_OVR, i] += 1
                    if uc[i] < umn[i]:
                        umn[i] = uc[i]
                    if uc[i] > umx[i]:
                        umx[i] = uc[i]
                    continue
                q = k + dkc[i]
                pso, pvo, pax = _pst(i, q, dt, NPc, PK, PA, PS, PV)
                if lead != 0:
                    psx, pvx, pao = _pst(i, q + lead, dt, NPc, PK, PA, PS,
                                         PV)
                else:
                    pao = pax
                md = mode[i]
                ucbf = np.inf
                if i == 0:
                    e = (pso - s[0]) + pe0
                    x = pvo - vim[0]
                    ff = pao
                    ub = 0.0
                else:
                    psp, pvp, pay = _pst(i - 1, q, dt, NPc, PK, PA, PS, PV)
                    if lead != 0:
                        psx, pvx, pap = _pst(i - 1, q + lead, dt, NPc, PK,
                                             PA, PS, PV)
                    else:
                        pap = pay
                    if plan_err:
                        e = sg[i] - (psp - pso)
                        x = (vpm[i] - vim[i]) - (pvp - pvo)
                        dv = uh[i]
                        if var == V_NODEV:
                            dv = 0.0
                        ff = pao + dv
                        ub = pap + dv
                    elif var == V_PLANONLY:
                        e = 0.0
                        x = 0.0
                        ff = pao
                        ub = pap
                    elif var == V_CACC:
                        e = sg[i] - (d_s[i] + ell) - h_w * vim[i]
                        x = vpm[i] - vim[i]
                        ff = uh[i]
                        ub = uh[i]
                    else:
                        e = sg[i] - (d_s[i] + ell)
                        x = vpm[i] - vim[i]
                        ff = uh[i]
                        ub = uh[i]
                if md == RULE or md == BRAKE:
                    un = -a_b
                elif md == S2:
                    if var == V_PID:
                        lim = Imax / Ki
                        y = pidI[i] + e * T_u
                        if y > lim:
                            y = lim
                        elif y < -lim:
                            y = -lim
                        pidI[i] = y
                        un = ff + Kp * e + Kd * x + Ki * y
                    else:
                        un = ff + beta * e + gamma * x
                elif md == S1:
                    z = x / phi
                    if z < -1.0:
                        z = -1.0
                    elif z > 1.0:
                        z = 1.0
                    un = ff + alpha * z
                else:                       # COAST, PLAN
                    un = ff
                uq = un
                infeas = False
                if i >= 1:
                    g = sg[i] - ell - s_m
                    hb = g + vpm[i] / bsl[i - 1] - vim[i] / bsl[i]
                    ucbf = bsl[i] * ((vpm[i] - vim[i]) + ub / bsl[i - 1]
                                     + kappa * hb)
                    mg = ucbf - un
                    if mg < mmin[i - 1]:
                        mmin[i - 1] = mg
                        tmmin[i - 1] = t
                    if filt:
                        if ucbf < ulo[i] - 1e-12:
                            uq = ulo[i]
                            infeas = True
                        elif ucbf < un:
                            uq = ucbf
                u = uq
                if u < ulo[i]:
                    u = ulo[i]
                elif u > uhi:
                    u = uhi
                unf = un
                if unf < ulo[i]:
                    unf = ulo[i]
                elif unf > uhi:
                    unf = uhi
                cnt[C_UPD, i] += 1
                if un > uhi:
                    cnt[C_NOMPOS, i] += 1
                if uq > uhi:
                    cnt[C_CLIP0, i] += 1
                if u == ulo[i]:
                    cnt[C_CLIPK, i] += 1
                if u == uhi:
                    cnt[C_AT0, i] += 1
                if u < unf - 1e-12:
                    cnt[C_BAR, i] += 1
                if uq < un - 1e-12:
                    cnt[C_BARRAW, i] += 1
                if infeas:
                    cnt[C_INFEAS, i] += 1
                if u < umn[i]:
                    umn[i] = u
                if u > umx[i]:
                    umx[i] = u
                unet[i] = u
                if hifi:
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
                    jk = abs(x - uc[i]) / T_u
                    if jk > jpk[i]:
                        jpk[i] = jk
                    lo = -F[F_DERATE] * amax[i]
                    if x < lo:
                        x = lo
                    elif x > F[F_UACT]:
                        x = F[F_UACT]
                    uc[i] = x
                    if abs(x) > ucmx[i]:
                        ucmx[i] = abs(x)
                else:
                    uc[i] = u
                    if abs(u) > ucmx[i]:
                        ucmx[i] = abs(u)
        # ----------------------------------------------- record
        if rec_every > 0 and k % rec_every == 0 and irec < nrec:
            rec[irec, 0] = t
            for i in range(n):
                rec[irec, 1 + i] = s[i]
                rec[irec, 1 + n + i] = v[i]
                rec[irec, 1 + 2 * n + i] = uc[i]
                rec[irec, 1 + 3 * n + i] = mode[i]
            irec += 1
        # ----------------------------------------------- (7) send
        if has_msg:
            for i in range(n):
                val = uc[i] - comp[i]
                if msg_dev:
                    psx, pvx, pax = _pst(i, k + dkc[i] + lead, dt, NPc, PK,
                                         PA, PS, PV)
                    val = val - pax
                sent_val[i, k] = val
                sent_flag[i, k] = flag[i]
        # ----------------------------------------------- end of the run
        done = True
        for i in range(n):
            if not lat[i] or v[i] > 0.0:        # latched and at true rest
                done = False
                break
        if done or k == N:
            k_end = k
            break
        # ----------------------------------------------- integrate
        if hifi:
            ip = _step_hifi(t, dt, s, v, ua, uc, lat, tau_act, grade_acc,
                            F[F_A0], F[F_A1], F[F_A2], ell, s_m, gmin, tgmin,
                            tbuf, uamx, F[F_AHOLD], tstop, sstop)
            if ip >= 0:
                contact = t + dt
                cpair = ip
                k_end = k + 1
                break
        else:
            for i in range(n):
                acc[i] = uc[i]
            neg += _zero_times(dt, v, acc, lat, tz)
            tc, ip = _pairs_ideal(t, dt, s, v, acc, tz, lat, ell, s_m, gmin,
                                  tgmin, tbuf)
            if ip >= 0:
                _advance_ideal(t, tc, s, v, acc, tz, lat, tstop, sstop)
                contact = t + tc
                cpair = ip
                k_end = k + 1
                for i in range(n):
                    if lat[i]:
                        mode[i] = LATCH
                break
            _advance_ideal(t, dt, s, v, acc, tz, lat, tstop, sstop)
            for i in range(n):
                if lat[i] and mode[i] != LATCH:
                    mode[i] = LATCH
                    flag[i] = True
    for i in range(n):
        mode_out[i] = mode[i]
    fout[O_CONTACT] = contact
    fout[O_TEND] = k_end * dt
    iout[J_CPAIR] = cpair
    iout[J_KEND] = k_end
    iout[J_NREC] = irec
    iout[J_NEG] = neg


@njit(cache=True)
def _mpc_period(k0, hold, I, F, amax, bsl, s, v, ua, uc, lat, tau_act,
                grade_acc, sent_val, tstop, sstop, gmin, tgmin, hmin, thmin,
                tbuf, uamx, acc, tz, stopdet, rec, irec_a):
    """Plant of the MPC over one control period (hold steps) with the held
    commands uc: sends (applied command), motion, stop logic, clearance
    statistics.  stopdet = 1: the stop detector of sim_mpc.py (speed at
    most v_stop once the predecessor, or the reference for the head, is at
    rest, after T_xi) in addition to the zero-speed latch of the ideal
    plant.  Returns (contact instant or nan, pair, steps done)."""
    n = I[I_N]
    N = I[I_NSTEP]
    hifi = I[I_PLANT] == 1
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    dt = F[F_DT]
    ell = F[F_ELL]
    s_m = F[F_SM]
    T_xi = F[F_TXI]
    tau_star = F[F_TAUSTAR]
    vst = F[F_VSTOP] if hifi else F[F_VSTOPMPC]
    for m in range(hold):
        k = k0 + m
        if k > N:
            return np.nan, -1, m
        t = k * dt
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            hb = g + v[i - 1] / bsl[i - 1] - v[i] / bsl[i]
            if hb < hmin[i - 1]:
                hmin[i - 1] = hb
                thmin[i - 1] = t
            if hifi and k == 0:
                gmin[i - 1] = g
                tgmin[i - 1] = 0.0
        if rec_every > 0 and k % rec_every == 0 and irec_a[0] < nrec:
            r = irec_a[0]
            rec[r, 0] = t
            for i in range(n):
                rec[r, 1 + i] = s[i]
                rec[r, 1 + n + i] = v[i]
                rec[r, 1 + 2 * n + i] = uc[i]
                rec[r, 1 + 3 * n + i] = LATCH if lat[i] else S2
            irec_a[0] += 1
        for i in range(n):
            sent_val[i, k] = uc[i]
        done = True
        for i in range(n):
            if not lat[i] or v[i] > 0.0:        # latched and at true rest
                done = False
                break
        if done or k == N:
            return np.nan, -2, m
        if hifi:
            ip = _step_hifi(t, dt, s, v, ua, uc, lat, tau_act, grade_acc,
                            F[F_A0], F[F_A1], F[F_A2], ell, s_m, gmin, tgmin,
                            tbuf, uamx, F[F_AHOLD], tstop, sstop)
            if ip >= 0:
                return t + dt, ip, m + 1
        else:
            for i in range(n):
                acc[i] = uc[i]
            _zero_times(dt, v, acc, lat, tz)
            tc, ip = _pairs_ideal(t, dt, s, v, acc, tz, lat, ell, s_m, gmin,
                                  tgmin, tbuf)
            if ip >= 0:
                _advance_ideal(t, tc, s, v, acc, tz, lat, tstop, sstop)
                return t + tc, ip, m + 1
            _advance_ideal(t, dt, s, v, acc, tz, lat, tstop, sstop)
        if stopdet == 1:
            t1 = (k + 1) * dt
            for i in range(n):
                if lat[i]:
                    continue
                rest = t1 >= tau_star if i == 0 else lat[i - 1]
                if t1 > T_xi and v[i] <= vst and rest:
                    lat[i] = True
                    if not (F[F_AHOLD] > 0.0 and v[i] > 0.0):
                        tstop[i] = t1
                        sstop[i] = s[i]
                        v[i] = 0.0
    return np.nan, -1, hold


# ================================================================== setup
def load_design(path=None):
    """design.json of the benchmark (read-only)."""
    import json
    if path is None:
        path = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')
    return json.load(open(path))


def setup(cfg, dt=1e-3):
    """Simulator inputs of a Case D configuration cfg (the dictionary
    design.json['cfg']); the plan is built by certify_plan.plan(cfg)."""
    sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
    import certify_plan as cq
    Pl = cq.plan(cfg)
    n = int(cfg['n'])
    T_m = float(cfg.get('T_c', 1e-3))
    if abs(T_m - dt) > 1e-15:
        raise ValueError('sim_bench: T_m = T_c = dt = 1 ms required')
    lam = float(cfg['lam'])
    MP = max(len(sg) for sg in Pl.segs)
    NPc = np.zeros(n, np.int64)
    PK = np.zeros((n, MP), np.int64)
    PA = np.zeros((n, MP))
    PS = np.zeros((n, MP))
    PV = np.zeros((n, MP))
    for i in range(n):
        NPc[i] = len(Pl.segs[i])
        for j, (t0, t1, a, s0, v0) in enumerate(Pl.segs[i]):
            q = t0 / cq.F(repr(dt))
            if q.denominator != 1:
                raise ValueError('sim_bench: planned jump off the %g s grid'
                                 % dt)
            PK[i, j] = int(q)
            PA[i, j] = float(a)
            PS[i, j] = float(s0)
            PV[i, j] = float(v0)
    k_xi = int(Pl.T_xi / cq.F(repr(dt)))
    dbar = float(cfg['dbar'])
    Ntr = int(round((dbar - T_m) / dt))
    amax = np.array([float(x) for x in cfg['amax']])
    v_c = float(cq.FIXED['v_c'])
    B = dict(
        n=n, dt=dt, k=float(cfg['k']), a_b=float(cfg['a_b']),
        alpha=float(cfg['alpha']), lam=lam, beta=lam * lam, gamma=2.0 * lam,
        phi=float(cfg['phi']), eps_det=float(cfg['eps_det']),
        kappa=float(cq.FIXED['kappa']), ell=float(cq.FIXED['ell']),
        s_m=float(cfg['s_m']), v_c=v_c, amax=amax, bsl=amax / v_c,
        d_s=np.array([float(x) for x in cfg['d_s']]),
        c0=[float(x) for x in cfg['c0']],
        v0=np.array([float(cq.F(x)) for x in cfg['v0']]),
        s0=np.array([float(x) for x in Pl.s0]),
        T_xi=float(Pl.T_xi), k_xi=k_xi, v_xi=float(Pl.v_xi),
        tau_star=float(Pl.tau_star), v_c1=float(Pl.v_c1),
        s_ref_stop=float(Pl.s_ref_stop),
        markers=np.array([float(x) for x in Pl.markers]),
        T_m=T_m, T_c=T_m, dbar=dbar, Ntr=Ntr, transport=Ntr * dt,
        NPc=NPc, PK=PK, PA=PA, PS=PS, PV=PV, Pl=Pl, cfg=cfg,
        req=dict(disp=float(cfg.get('sync_tol', REQ['disp'])),
                 marker=float(cfg.get('align_tol', REQ['marker'])),
                 gap=float(cfg.get('gap_tol', REQ['gap']))))
    return B


def doubled_mismatch(B):
    """Takeover speeds with the increments v_i(0) - v_{i-1}(0) doubled and
    the head speed unchanged."""
    v0 = np.array(B['v0'], float)
    out = [float(v0[0])]
    for i in range(1, len(v0)):
        out.append(out[-1] + 2.0 * float(v0[i] - v0[i - 1]))
    return out


def plan_state(B, i, t):
    """Planned (s*, v*, a*) of unit i at the time t (float, on the grid)."""
    q = int(round(t / B['dt']))
    return _pst(i, q, B['dt'], B['NPc'], B['PK'], B['PA'], B['PS'], B['PV'])


def _params(B, N, variant_code, apply_filter, plant, H, loss, k_loss, em,
            k_wd, lead, rec_every, nrec, gains, h_w, hold, uhi=0.0,
            a_hold=0.0):
    I = np.zeros(I_NI, np.int64)
    I[I_N] = B['n']
    I[I_NSTEP] = N
    I[I_NTR] = B['Ntr']
    I[I_NC] = int(round(B['T_c'] / B['dt']))
    I[I_HOLD] = hold
    I[I_VAR] = variant_code
    I[I_FILT] = 1 if apply_filter else 0
    I[I_PLANT] = 1 if plant == 'hifi' else 0
    I[I_LOSS] = 1 if loss else 0
    I[I_KLOSS] = k_loss
    I[I_EM] = em
    I[I_KWD] = k_wd
    I[I_KXI] = B['k_xi']
    I[I_LEAD] = lead
    I[I_REC] = rec_every
    I[I_NREC] = nrec
    F = np.zeros(F_NF)
    F[F_DT] = B['dt']
    F[F_K] = B['k']
    F[F_AB] = B['a_b']
    F[F_ALPHA] = B['alpha']
    F[F_BETA] = B['beta']
    F[F_GAMMA] = B['gamma']
    F[F_PHI] = B['phi']
    F[F_EPSDET] = B['eps_det']
    F[F_KAPPA] = B['kappa']
    F[F_ELL] = B['ell']
    F[F_SM] = B['s_m']
    F[F_TXI] = B['k_xi'] * B['dt']
    F[F_KP] = gains['Kp']
    F[F_KD] = gains['Kd']
    F[F_KI] = gains['Ki']
    F[F_IMAX] = gains['Imax']
    F[F_HW] = h_w
    F[F_UACT] = U_ACT
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
    F[F_TAUSTAR] = B['tau_star']
    F[F_VSTOPMPC] = V_STOP_MPC
    F[F_UHI] = uhi
    F[F_AHOLD] = a_hold
    return I, F


def _none(x):
    x = float(x)
    return x if np.isfinite(x) else None


def _lst(a):
    return [_none(x) for x in np.asarray(a, float)]


def _assemble(B, res, s, v, lat, T_u, cfgd, t0):
    """Metrics of one run from the accumulators of the kernels."""
    n = B['n']
    tstop = res['tstop']
    fin = np.isfinite(tstop)
    contact = _none(res['contact'])
    all_stopped = bool(np.all(fin) and contact is None)
    marks = B['markers']
    gap = (s[:-1] - s[1:] - B['ell']) - B['d_s'][1:]
    mark = s - marks
    cnt = res['cnt']
    out = dict(cfgd)
    out.update(
        tstop=_lst(tstop),
        stop_order=[int(x) + 1 for x in np.argsort(tstop, kind='stable')]
        if all_stopped else None,
        all_stopped=all_stopped,
        n_stopped=int(np.sum(fin)),
        disp=float(np.max(tstop) - np.min(tstop)) if all_stopped else None,
        T_f=float(np.max(tstop)) if all_stopped else None,
        first_stop=float(np.min(tstop[fin])) if np.any(fin) else None,
        marker_signed=[float(x) for x in mark],
        marker_abs_max=float(np.max(np.abs(mark))),
        gap_signed=[float(x) for x in gap],
        gap_abs_max=float(np.max(np.abs(gap))),
        terminal_defined=all_stopped,
        gmin=_lst(res['gmin']), t_gmin=_lst(res['tgmin']),
        gmin_min=float(np.min(res['gmin'])),
        gmin_pair=int(np.argmin(res['gmin'])) + 2,
        hmin=_lst(res['hmin']), t_hmin=_lst(res['thmin']),
        hmin_min=float(np.min(res['hmin'])),
        mmin=_lst(res['mmin']), t_mmin=_lst(res['tmmin']),
        u_min_unit=_lst(res['umn']), u_max_unit=_lst(res['umx']),
        u_min=_none(np.min(res['umn'])), u_max=_none(np.max(res['umx'])),
        u_abs_max=_none(max(abs(float(np.min(res['umn']))),
                            abs(float(np.max(res['umx']))))),
        n_updates=[int(x) for x in cnt[C_UPD]],
        t_nom_pos=[float(x) * T_u for x in cnt[C_NOMPOS]],
        t_clip0=[float(x) * T_u for x in cnt[C_CLIP0]],
        t_clipk=[float(x) * T_u for x in cnt[C_CLIPK]],
        t_at0=[float(x) * T_u for x in cnt[C_AT0]],
        t_barrier=[float(x) * T_u for x in cnt[C_BAR]],
        t_barrier_raw=[float(x) * T_u for x in cnt[C_BARRAW]],
        t_infeas=[float(x) * T_u for x in cnt[C_INFEAS]],
        t_override=[float(x) * T_u for x in cnt[C_OVR]],
        contact_time=contact,
        contact_pair=(int(res['cpair']) + 1) if contact is not None else None,
        buffer_lost=bool(np.any(res['tbuf'] >= 0.0)),
        t_buffer_lost=[(float(x) if x >= 0.0 else None) for x in res['tbuf']],
        tact=_lst(res['tact']), tset=_lst(res['tset']),
        trule=_lst(res['trule']),
        rule_entered=bool(np.any(np.isfinite(res['trule']))),
        rule_units=[int(i) + 1 for i in range(n)
                    if np.isfinite(res['trule'][i])],
        s_final=[float(x) for x in s], v_final=[float(x) for x in v],
        t_end_sim=float(res['t_end']),
        control_period=T_u)
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


def _bounds(B, bounds):
    if bounds is None:
        return np.full(B['n'], -B['k']), 0.0
    lo = np.array([float(x) for x in bounds[0]])
    if lo.shape != (B['n'],):
        raise ValueError('bounds: one lower limit per unit')
    return lo, float(bounds[1])


def _hifi_arrays(B, H):
    n = B['n']
    tau_act = np.broadcast_to(np.asarray(H['tau_act'], float),
                              (n,)).astype(float).copy()
    grade_acc = float(H['grade']) * np.array(GRADE_SIGNS[:n])
    return tau_act, grade_acc


def _accumulators(n):
    nan = np.nan
    return dict(
        tstop=np.full(n, nan), sstop=np.full(n, nan), tact=np.full(n, nan),
        tset=np.full(n, nan), trule=np.full(n, nan),
        gmin=np.full(n - 1, np.inf), tgmin=np.full(n - 1, nan),
        hmin=np.full(n - 1, np.inf), thmin=np.full(n - 1, nan),
        mmin=np.full(n - 1, np.inf), tmmin=np.full(n - 1, nan),
        tbuf=np.full(n - 1, -1.0), umn=np.full(n, np.inf),
        umx=np.full(n, -np.inf), uamx=np.zeros(n), ucmx=np.zeros(n),
        jpk=np.zeros(n), cnt=np.zeros((C_NC, n), np.int64),
        twd=np.full(n, nan))


# ==================================================================== run
def run(B, variant='proposed', plant='ideal', apply_filter=None,
        scenario=None, t_loss=None, emergency_unit=None, T_wd=None,
        clock_offsets=None, hifi=None, seed=0, gains=None, h_w=None,
        v0=None, T_end=None, record=False, rec_dt=0.01, W=None,
        mpc_stop='detector', mpc_planner=None, bounds=None, hold_brake=False):
    """One closed-loop run; returns the metrics of the run (module
    docstring).  emergency_unit and the units of every output are numbered
    from 1 (the head); T_end defaults to T_xi + 60 s.  bounds = (lower
    limits per unit, upper limit) replaces the input set [-k, 0]; it
    serves the port checks against sim_cont.py and sim_mpc.py only.
    hold_brake (perturbed plant): after the latch or the stop detector, a
    holding brake decelerates the unit at a_b until its true speed is zero
    (default False: the unit rests at once, the convention of the earlier
    records)."""
    if variant == 'mpc':
        if scenario is not None or clock_offsets is not None:
            raise NotImplementedError('mpc: nominal scenario only')
        return run_mpc(B, W=W, plant=plant, hifi=hifi, seed=seed,
                       apply_filter=True if apply_filter is None
                       else apply_filter, stop=mpc_stop, v0=v0, T_end=T_end,
                       planner=mpc_planner, record=record, rec_dt=rec_dt,
                       bounds=bounds, hold_brake=hold_brake)
    if variant not in VARIANTS:
        raise ValueError('unknown variant %s' % variant)
    if plant not in ('ideal', 'hifi'):
        raise ValueError('plant: ideal or hifi')
    if plant == 'hifi' and (scenario is not None
                            or clock_offsets is not None):
        raise NotImplementedError('perturbed plant: nominal scenario only')
    if scenario not in (None, 'loss'):
        raise ValueError('scenario: None or loss')
    t0 = time.time()
    n = B['n']
    dt = B['dt']
    if apply_filter is None:
        apply_filter = variant != 'nofilter'
    if variant == 'nofilter' and apply_filter:
        raise ValueError('nofilter runs without the filter')
    H = dict(HIFI_DEFAULT, **(hifi or {}))
    hold = int(round(H['T_s'] / dt)) if plant == 'hifi' else 1
    lead = int(round(H['ramp_lead'] / dt)) if plant == 'hifi' else 0
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
    dkc = np.zeros(n, np.int64)
    if clock_offsets is not None:
        for i in range(n):
            q = clock_offsets[i] / dt
            if abs(q - round(q)) > 1e-6:
                raise ValueError('clock offsets are multiples of dt')
            dkc[i] = int(round(q))
    g = dict(K_PID, **(gains or {}))
    hw = K_CACC['h_w'] if h_w is None else float(h_w)
    rec_every = max(1, int(round(rec_dt / dt))) if record else 0
    nrec = N // rec_every + 2 if record else 1
    ulo, uhi = _bounds(B, bounds)
    I, F = _params(B, N, VARIANTS[variant], apply_filter, plant, H, loss,
                   k_loss, em, k_wd, lead, rec_every, nrec, g, hw, hold,
                   uhi, a_hold=float(B['a_b']) if (plant == 'hifi'
                                                  and hold_brake) else 0.0)
    tau_act, grade_acc = _hifi_arrays(B, H)
    if plant == 'hifi':
        rng = np.random.default_rng(seed)
        Z = rng.standard_normal((N // hold + 2, n, 4))
    else:
        Z = np.zeros((1, n, 4))
    s = B['s0'].copy()
    v = np.array(B['v0'] if v0 is None else v0, float).copy()
    A = _accumulators(n)
    fout = np.zeros(O_NO)
    iout = np.zeros(J_NJ, np.int64)
    rec = np.zeros((nrec, 1 + 4 * n))
    mode_out = np.zeros(n, np.int64)
    _core(I, F, B['amax'], ulo, B['bsl'], B['d_s'], s, v, B['NPc'],
          B['PK'], B['PA'], B['PS'], B['PV'], dkc, Z, tau_act, grade_acc,
          A['tstop'], A['sstop'], A['tact'], A['tset'], A['trule'],
          A['gmin'], A['tgmin'], A['hmin'], A['thmin'], A['mmin'],
          A['tmmin'], A['tbuf'], A['umn'], A['umx'], A['uamx'], A['ucmx'],
          A['jpk'], A['cnt'], fout, iout, rec, mode_out, A['twd'])
    res = dict(A, contact=fout[O_CONTACT], cpair=iout[J_CPAIR],
               t_end=fout[O_TEND])
    cfgd = dict(
        variant=variant, plant=plant, apply_filter=bool(apply_filter),
        scenario=scenario, t_loss=t_loss, emergency_unit=emergency_unit,
        T_wd=T_wd, hold_brake=bool(F[F_AHOLD] > 0.0),
        clock_offsets=None if clock_offsets is None
        else [float(x) for x in clock_offsets],
        seed=int(seed) if plant == 'hifi' else None,
        hifi=None if plant != 'hifi'
        else {k: (list(x) if isinstance(x, (list, tuple, np.ndarray))
                  else x) for k, x in H.items()},
        gains=dict(g) if variant == 'pid' else None,
        h_w=hw if variant == 'cacc' else None,
        v0=[float(x) for x in v0] if v0 is not None
        else [float(x) for x in B['v0']],
        T_end=float(T_end),
        input_set=[[float(x) for x in ulo], float(uhi)])
    out = _assemble(B, res, s, v, None, hold * dt, cfgd, t0)
    out['mode_final'] = [MODE_NAMES[int(m)] for m in mode_out]
    out['t_watchdog'] = _lst(A['twd'])
    out['watchdog_units'] = [i + 1 for i in range(n)
                             if np.isfinite(A['twd'][i])]
    out['override_units'] = [i + 1 for i in range(n) if A['cnt'][C_OVR, i] > 0]
    out['u_act_cmd_abs_max'] = [float(x) for x in A['ucmx']]
    if plant == 'hifi':
        out['u_realized_abs_max'] = [float(x) for x in A['uamx']]
        out['jerk_pk'] = [float(x) for x in A['jpk']]
    if record:
        m = int(iout[J_NREC])
        out['rec'] = dict(t=rec[:m, 0].copy(), s=rec[:m, 1:1 + n].T.copy(),
                          v=rec[:m, 1 + n:1 + 2 * n].T.copy(),
                          u=rec[:m, 1 + 2 * n:1 + 3 * n].T.copy(),
                          mode=rec[:m, 1 + 3 * n:1 + 4 * n].T.copy())
    return out


# ==================================================================== MPC
def _planner(s_now, s_tgt, v_tgt, T, tau_c, dv_max):
    """Closing profile toward (s_tgt_k, v_tgt_k) (sim_mpc._planner): the
    speed is the target speed plus a saturated proportional closing term on
    the remaining spacing error, integrated from the current position."""
    N = len(s_tgt)
    s_des = np.zeros(N)
    v_des = np.zeros(N)
    sp = s_now
    for k in range(N):
        err = s_tgt[k] - (sp + v_tgt[k] * T)
        v_des[k] = max(0.0, v_tgt[k] + float(np.clip(err / tau_c,
                                                     -dv_max, dv_max)))
        sp = sp + v_des[k] * T
        s_des[k] = sp
    return s_des, v_des


def _pred_mats(N, T):
    Su = np.zeros((N, N))
    Vu = np.zeros((N, N))
    for k in range(1, N + 1):
        for j in range(k):
            Su[k - 1, j] = T * T * (0.5 + (k - 1 - j))
            Vu[k - 1, j] = T
    return Su, Vu


class _MPC:
    """Box-constrained least-squares MPC of sim_mpc.py (bounded-variable
    least squares, scipy.optimize.lsq_linear)."""

    def __init__(self, W, N=N_H, T=T_MPC):
        self.W = dict(MPC_DEFAULT_W, **(W or {}))
        self.N = N
        self.T = T
        self.Su, self.Vu = _pred_mats(N, T)
        D = np.eye(N) - np.eye(N, k=-1)
        sg = MPC_SIG
        self.A = np.vstack([np.sqrt(self.W['w_e']) / sg['e'] * self.Su,
                            np.sqrt(self.W['w_v']) / sg['v'] * self.Vu,
                            np.sqrt(self.W['w_u']) / sg['u'] * np.eye(N),
                            np.sqrt(self.W['w_du']) / sg['du'] * D])
        self.ks = np.arange(1, N + 1)

    def solve(self, s0, v0, s_tgt, v_tgt, u_tgt, u_prev, lo, hi):
        from scipy.optimize import lsq_linear
        N = self.N
        W = self.W
        sg = MPC_SIG
        b = np.concatenate([
            np.sqrt(W['w_e']) / sg['e'] * (s_tgt - s0
                                           - v0 * self.ks * self.T),
            np.sqrt(W['w_v']) / sg['v'] * (v_tgt - v0),
            np.sqrt(W['w_u']) / sg['u'] * u_tgt,
            np.sqrt(W['w_du']) / sg['du'] * np.r_[u_prev, np.zeros(N - 1)]])
        r = lsq_linear(self.A, b, bounds=(lo, hi), method='bvls',
                       lsq_solver='exact', max_iter=60)
        u = np.minimum(hi, np.maximum(lo, r.x))
        s_plan = s0 + v0 * self.ks * self.T + self.Su @ u
        v_plan = np.maximum(0.0, v0 + self.Vu @ u)
        return u, s_plan, v_plan, int(r.status)


def _head_reference(B, t, ks, T):
    """Planned position and speed of the head (the shared reference) over
    the horizon, held at the planned stop after tau_star."""
    ts = t + ks * T
    tt = np.minimum(ts, B['tau_star'])
    Pl = B['Pl']
    sr, vr, ar = Pl.state_f(0, tt)
    vr = np.where(ts >= B['tau_star'], 0.0, np.maximum(0.0, vr))
    return np.asarray(sr, float), np.asarray(vr, float)


def run_mpc(B, W=None, plant='ideal', hifi=None, seed=0, apply_filter=True,
            stop='detector', v0=None, T_end=None, planner=None, record=False,
            rec_dt=0.01, bounds=None, hold_brake=False):
    """Cooperative arrival MPC adapted from sim_mpc.py: input bounds
    [-k, 0], the plan of the head as the shared reference, control period
    0.1 s, horizon 20, predecessor predictions one period old, planner
    layer (tau_c, dv_max), the filter downstream with the received applied
    command of the predecessor (evaluated once per control period).
    stop = 'detector': stop detector of sim_mpc.py (speed at most
    0.02 m/s on the ideal plant, v_stop on the perturbed plant, once the
    predecessor or, for the head, the reference is at rest, after T_xi)
    in addition to the zero-speed latch; stop = 'latch': zero-speed latch
    only (ideal plant)."""
    if stop not in ('detector', 'latch'):
        raise ValueError('stop: detector or latch')
    if plant == 'hifi' and stop != 'detector':
        raise ValueError('perturbed plant: stop detector required')
    t0 = time.time()
    n = B['n']
    dt = B['dt']
    H = dict(HIFI_DEFAULT, **(hifi or {}))
    pl = dict(tau_c=TAU_C, dv_max=DV_MAX)
    pl.update(planner or {})
    hold = int(round(T_MPC / dt))
    if T_end is None:
        T_end = B['T_xi'] + 60.0
    N = int(round(T_end / dt))
    rec_every = max(1, int(round(rec_dt / dt))) if record else 0
    nrec = N // rec_every + 2 if record else 1
    ulo, uhi = _bounds(B, bounds)
    I, F = _params(B, N, -1, apply_filter, plant, H, False, 0, -1, -1, 0,
                   rec_every, nrec, K_PID, 0.0, hold, uhi,
                   a_hold=float(B['a_b']) if (plant == 'hifi'
                                              and hold_brake) else 0.0)
    tau_act, grade_acc = _hifi_arrays(B, H)
    is_hifi = plant == 'hifi'
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N // hold + 2, n, 4)) if is_hifi else None
    mpc = _MPC(W)
    ell = B['ell']
    s_m = B['s_m']
    kappa = B['kappa']
    bsl = B['bsl']
    d_s = B['d_s']
    amax = B['amax']
    Ntr = B['Ntr']
    A_cap = amax * (H['derate'] if is_hifi else 1.0)
    s = B['s0'].copy()
    v = np.array(B['v0'] if v0 is None else v0, float).copy()
    A = _accumulators(n)
    lat = np.zeros(n, np.bool_)
    ua = np.zeros(n)
    uc = np.zeros(n)
    u_prev = np.zeros(n)
    acc = np.zeros(n)
    tz = np.full(n, np.inf)
    sent_val = np.zeros((n, N + 2))
    rec = np.zeros((nrec, 1 + 4 * n))
    irec_a = np.zeros(1, np.int64)
    plans = [None] * n
    solve_ms = []
    nstat = 0
    contact = np.nan
    cpair = -1
    k_end = N
    cnt = A['cnt']
    stopdet = 1 if stop == 'detector' else 0
    k = 0
    while k <= N:
        t = k * dt
        plans_prev = list(plans)
        tw = time.perf_counter()
        s_ref_h, v_ref_h = _head_reference(B, t, mpc.ks, mpc.T)
        jz = k // hold
        for i in range(n):
            if is_hifi:
                si_m = s[i] + H['noise_gap'] * Z[jz, i, 3]
                vi_m = np.round((v[i] + H['noise_vel'] * Z[jz, i, 1])
                                / H['quant_vel']) * H['quant_vel']
                if i > 0:
                    gap_m = (s[i - 1] - s[i] - ell) + H['bias_gap'] \
                        + H['noise_gap'] * Z[jz, i, 0]
                    vp_m = np.round((v[i - 1] + H['noise_vel'] * Z[jz, i, 2])
                                    / H['quant_vel']) * H['quant_vel']
            else:
                si_m = s[i]
                vi_m = v[i]
                if i > 0:
                    gap_m = s[i - 1] - s[i] - ell
                    vp_m = v[i - 1]
            if lat[i]:
                un = 0.0
                plans[i] = (np.full(N_H, s[i]), np.zeros(N_H))
                if is_hifi:
                    dumax = H['jerk_max'] * T_MPC
                    un = float(np.clip(un, uc[i] - dumax, uc[i] + dumax))
                u_prev[i] = un
                uc[i] = un
                continue
            if i == 0:
                s_tgt = s_ref_h
                v_tgt = v_ref_h
            else:
                pp = plans_prev[i - 1]
                if pp is None:              # no prediction received yet
                    sp = si_m + gap_m + ell
                    s_tgt = sp + vp_m * mpc.ks * mpc.T - (ell + d_s[i])
                    v_tgt = np.full(N_H, vp_m)
                else:
                    ps, pv = pp
                    ps = np.r_[ps[1:], ps[-1] + pv[-1] * mpc.T]
                    pv = np.r_[pv[1:], pv[-1]]
                    s_tgt = ps - (ell + d_s[i])
                    v_tgt = pv
            s_tgt, v_tgt = _planner(si_m, s_tgt, v_tgt, mpc.T, pl['tau_c'],
                                    pl['dv_max'])
            u_tgt = np.diff(np.r_[vi_m, v_tgt]) / mpc.T
            u_seq, s_plan, v_plan, status = mpc.solve(
                si_m, vi_m, s_tgt, v_tgt, u_tgt, u_prev[i], ulo[i], uhi)
            if status <= 0:
                nstat += 1
            un = float(u_seq[0])
            plans[i] = (s_plan, v_plan)
            uq = un
            infeas = False
            if i >= 1:
                j = k - Ntr
                uh = sent_val[i - 1, j] if j >= 0 else 0.0
                g = gap_m - s_m
                hb = g + vp_m / bsl[i - 1] - vi_m / bsl[i]
                ucbf = bsl[i] * ((vp_m - vi_m) + uh / bsl[i - 1]
                                 + kappa * hb)
                mg = ucbf - un
                if mg < A['mmin'][i - 1]:
                    A['mmin'][i - 1] = mg
                    A['tmmin'][i - 1] = t
                if apply_filter:
                    if ucbf < ulo[i] - 1e-12:
                        uq = ulo[i]
                        infeas = True
                    elif ucbf < un:
                        uq = ucbf
            u = min(uhi, max(ulo[i], uq))
            unf = min(uhi, max(ulo[i], un))
            cnt[C_UPD, i] += 1
            if un > uhi:
                cnt[C_NOMPOS, i] += 1
            if uq > uhi:
                cnt[C_CLIP0, i] += 1
            if u == ulo[i]:
                cnt[C_CLIPK, i] += 1
            if u == uhi:
                cnt[C_AT0, i] += 1
            if u < unf - 1e-12:
                cnt[C_BAR, i] += 1
            if uq < un - 1e-12:
                cnt[C_BARRAW, i] += 1
            if infeas:
                cnt[C_INFEAS, i] += 1
            A['umn'][i] = min(A['umn'][i], u)
            A['umx'][i] = max(A['umx'][i], u)
            if is_hifi:
                dumax = H['jerk_max'] * T_MPC
                x = float(np.clip(u, uc[i] - dumax, uc[i] + dumax))
                A['jpk'][i] = max(A['jpk'][i], abs(x - uc[i]) / T_MPC)
                u_prev[i] = x
                uc[i] = float(np.clip(x, -A_cap[i], U_ACT))
            else:
                u_prev[i] = u
                uc[i] = u
            A['ucmx'][i] = max(A['ucmx'][i], abs(uc[i]))
        solve_ms.append((time.perf_counter() - tw) * 1e3)
        tc, ip, done = _mpc_period(k, hold, I, F, amax, bsl, s, v, ua, uc,
                                   lat, tau_act, grade_acc, sent_val,
                                   A['tstop'], A['sstop'], A['gmin'],
                                   A['tgmin'], A['hmin'], A['thmin'],
                                   A['tbuf'], A['uamx'], acc, tz, stopdet,
                                   rec, irec_a)
        if ip >= 0:
            contact = tc
            cpair = ip
            k_end = k + done
            break
        if ip == -2:
            k_end = k + done
            break
        k += hold
    res = dict(A, contact=contact, cpair=cpair, t_end=k_end * dt)
    cfgd = dict(
        variant='mpc', plant=plant, apply_filter=bool(apply_filter),
        scenario=None, t_loss=None, emergency_unit=None, T_wd=None,
        hold_brake=bool(F[F_AHOLD] > 0.0),
        clock_offsets=None, seed=int(seed) if is_hifi else None,
        hifi=None if not is_hifi
        else {k_: (list(x) if isinstance(x, (list, tuple, np.ndarray))
                   else x) for k_, x in H.items()},
        gains=None, h_w=None,
        v0=[float(x) for x in v0] if v0 is not None
        else [float(x) for x in B['v0']],
        T_end=float(T_end),
        input_set=[[float(x) for x in ulo], float(uhi)],
        W=dict(mpc.W), mpc_stop=stop,
        mpc_planner=dict(pl), T_mpc=T_MPC, N_h=N_H, sig=dict(MPC_SIG))
    out = _assemble(B, res, s, v, lat, T_MPC, cfgd, t0)
    out['n_solver_not_converged'] = int(nstat)
    out['runtime_solve_ms_mean'] = float(np.mean(solve_ms)) \
        if solve_ms else None
    out['runtime_solve_ms_max'] = float(np.max(solve_ms)) \
        if solve_ms else None
    out['u_act_cmd_abs_max'] = [float(x) for x in A['ucmx']]
    if is_hifi:
        out['u_realized_abs_max'] = [float(x) for x in A['uamx']]
        out['jerk_pk'] = [float(x) for x in A['jpk']]
    if record:
        m = int(irec_a[0])
        out['rec'] = dict(t=rec[:m, 0].copy(), s=rec[:m, 1:1 + n].T.copy(),
                          v=rec[:m, 1 + n:1 + 2 * n].T.copy(),
                          u=rec[:m, 1 + 2 * n:1 + 3 * n].T.copy(),
                          mode=rec[:m, 1 + 3 * n:1 + 4 * n].T.copy())
    return out


if __name__ == '__main__':
    D = load_design()
    B_ = setup(D['cfg'])
    for var in ('proposed', 'nofilter'):
        R = run(B_, variant=var)
        print(var, 'tstop', R['tstop'], 'disp', R['disp'])
        print('   marker', R['marker_signed'])
        print('   gap', R['gap_signed'], 'gmin', R['gmin'])
        print('   tset', R['tset'], 'u range', R['u_min'], R['u_max'],
              'runtime %.2f s' % R['runtime_s'])
