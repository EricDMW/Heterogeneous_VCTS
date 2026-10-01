#!/usr/bin/env python3
r"""sim_hr.py -- simulator of the head-only reference arrival (HR),
data/new_d19/headref/sim_spec.md (2026-09-28).

Only the head (unit 1, array index 0) stores the arrival reference: the
virtual leader (s_0, v_r, a_r) with a_r = a*_1 (planned acceleration of unit
1 of the round-6 fan plan certify_plan.plan(cfg)) on [0, T_xi), -a_b until
v_r = 0 at tau_r = T_xi + v_r(T_xi)/a_b, rest afterwards, and
s_0 = s*_1 + d_s[0] + ell.  Follower i (array index i >= 1, unit i+1 of the
spec) stores the constants of a gap-closing schedule in its local time
sigma_i = t - t^cl_i, t^cl_i the instant at which it processes the relayed
closure flag CLOSE of its predecessor:
    rho_i(sigma) = a*_i - a*_{i-1}          (0 for sigma >= T_xi - t_0),
    d_i(sigma)   = s*_{i-1} - s*_i - ell,
    dd_i(sigma)  = v*_{i-1} - v*_i          (plan time t_0 + sigma),
and before the onset (sigma < 0) d_i = c_i(0), dd_i = rho_i = 0.  The
schedule is stored as exact rational pieces (sigma_j, rho_j, d_j, dd_j) at
the jumps of rho_i (union of the jumps of a*_i and a*_{i-1} in [t_0, T_xi));
on a piece d_i = d_j + dd_j tau - rho_j tau^2/2, dd_i = dd_j - rho_j tau.
hr_setup() checks the pieces exactly (continuity of d_i, dd_i; d_i(0) =
c_i(0), dd_i(0) = 0; d_i = d_s, dd_i = rho_i = 0 after the join).

Laws (spec Sec. 3; e, eps schedule-relative, eps_raw = v_{i-1} - v_i):
    follower COAST  uhat + rho
             S1     uhat + rho + alpha sat(eps/phi)
             S2     uhat + rho + beta e + gamma eps
             RULE   -a_b
    head     S1     a_r + alpha sat(eps_1/phi),  S2  a_r + beta e_1 + gamma eps_1,
             BRAKE  -a_b from T_xi (head clock event)
    filter   u_cbf = b_i (eps_raw + uhat/b_{i-1} + kappa h_i), u_qp =
             sat_[-Uminus_i, U](min{u_nom, u_cbf}), fallback -Uminus_i when
             u_cbf < -Uminus_i; head sat_[-Uminus_1, U](u_nom); LATCH 0.
Mode logic, detector (local grid t_act + m T_c, activation then detector at
once), RULE (exact standstill sensing, entered at the predecessor's
zero-speed instant), LATCH (first zero speed at or after the S2 switch),
flags (MATCHED_i = mode >= S2, CLOSE_i; flags stay raised), priorities
LATCH > RULE > message batch (deliveries; head CLOSE raise at t_0 and head
switch to BRAKE at T_xi) > detector, one command update per event, sends on
the common grid j T_m (post-event value), FIFO delivery (constant transport
or per-send FIFO schedules, batch delivery): as in sim_cont.py.  The
followers never switch on T_xi; the braking reaches them through the
received commands.

Schemes (numerical conventions of sim_cont.py)
  'cont'  continuous local law evaluated at every RK4 stage between events
          (T_u = h); zero-speed instants localized by Illinois root finding
          on the RK4 step length.
  'zoh'   command computed at k T_u and at every event instant of the unit
          (including its own planned jumps) and held; exact double-integrator
          integration; exact zero-speed instants.
Planned jumps (head reference, followers' rho) are local clock events of
their unit.  They must lie on the h grid: the head's jumps are plan instants
(1 ms grid) and T_xi; a follower's jumps are t^cl_i + sigma_j with t^cl_i a
step instant (a delivery) and sigma_j on the 1 ms grid of local time, so the
kernel keeps them as integer step indices k^cl_i + round(sigma_j/h) (run()
asserts that every sigma_j/h, t_0/h, T_xi/h and every head jump/h is an
integer within 1e-6).  The planned pieces are right-continuous and frozen
over each integration step (RK4 stages at a step end see the left limit).

Unit order inside an instant: events, command update and send of unit 0,
then of unit 1, ...  For a transport of at least one step this is identical
to sim_cont's order (all events, all commands, all sends: no quantity of
unit i depends on a same-instant change of unit j > i), and it gives the
zero transport (transport=0) its meaning: the packet sent at t_j is
delivered at t_j to the successor, which updates and sends at t_j (a
same-instant relay along the chain; information age 0 at the send instants,
at most T_m).  sim_cont itself requires transport >= h.

ref_mode 'frozen' (default) as above.  ref_mode 'stage' reproduces the
convention of sim_cont's default kernel for the Case C profile: the head
reference is evaluated right-continuously at every stage time (a jump at a
step end is seen by the last RK4 stage), the head feedforward is a_r(t) for
t < T_xi and 0 from T_xi, and reference jumps are not head events.  It is
used, with schedules=False, to regress against sim_cont.run(profile=...).

schedules=False removes the schedules: rho_i = dd_i = 0 and
  nosched='step' (default): d_i = c_i(0) before the closure onset and d_i =
     d_s (parking gap) from it, so the gap is closed by feedback from t^cl_i
     (the 'feedback closure' ablation; with t_0 = 0 it gives exactly the
     Case C follower laws e_i = gap - d_s of sim_cont);
  nosched='hold': d_i = c_i(0) for every sigma (no closure at all).

Outputs mirror sim_cont.run where they make sense (see run()).
Numba is used when available; otherwise the same code runs as plain Python.
"""
import math
import os
import sys
import tempfile
from fractions import Fraction as Fr

import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR',
                      os.path.join(tempfile.gettempdir(), 'numba_cache_sim_hr'))
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
_CERT = os.path.join(CODE, 'src', 'certification')
if _CERT not in sys.path:
    sys.path.insert(0, _CERT)
import sim_cont as sc                                     # noqa: E402

# modes (identical to sim_cont)
COAST, S1, S2, RULE, LATCH, BRAKE = 0, 1, 2, 3, 4, 5
MODE_NAMES = ('COAST', 'S1', 'S2', 'RULE', 'LATCH', 'BRAKE')
SCH_ZOH, SCH_CONT = 1, 2

# float parameter slots
F_H = 0
F_TXI = 1
F_AB = 2
F_ALPHA = 3
F_BETA = 4
F_GAMMA = 5
F_PHI = 6
F_EPSDET = 7
F_ELL = 8
F_SM = 9
F_KAPPA = 10
F_U = 11
F_EPSE = 12
F_EPSV = 13
F_TAUR = 14
F_NF = 15
# int parameter slots
I_N = 0
I_NSTEP = 1
I_NM = 2
I_NC = 3
I_NU = 4
I_SCH = 5
I_REC = 6
I_KXI = 7
I_FILT = 8
I_NREC = 9
I_EARLY = 10
I_NSEND = 11
I_KT0 = 12
I_REFM = 13
I_NH = 14
I_NI = 15
# code bits of _law_hr
C_MIN, C_FB, C_SAT, C_HI, C_LO, C_POS = 1, 2, 4, 8, 16, 32
CNT_NAMES = ('min_branch', 'fallback', 'saturation', 'sat_upper', 'sat_lower',
             'positive_nominal')


# ---------------------------------------------------------------- kernels
@njit(cache=True)
def _href(t, j, HT, HA, HS, HV):
    """Head reference (s_0, v_r, a_r) at t on piece j."""
    tau = t - HT[j]
    a = HA[j]
    return HS[j] + HV[j] * tau + 0.5 * a * tau * tau, HV[j] + a * tau, a


@njit(cache=True)
def _href_idx(t, NH, HT):
    """Right-continuous piece index of the head reference at t."""
    j = 0
    for q in range(1, NH):
        if t >= HT[q]:
            j = q
        else:
            break
    return j


@njit(cache=True)
def _headref(t, jh, F, I, HT, HA, HS, HV):
    """Reference (s_0, v_r) and head feedforward seen by the head law at t:
    'frozen' the frozen piece jh; 'stage' the exact right-continuous value
    with the feedforward a_r(t) for t < T_xi and 0 from T_xi (sim_cont)."""
    if I[I_REFM] == 1:
        j = _href_idx(t, I[I_NH], HT)
        s0, vr, ar = _href(t, j, HT, HA, HS, HV)
        ff = ar if t < F[F_TXI] else 0.0
        return s0, vr, ff
    return _href(t, jh, HT, HA, HS, HV)


@njit(cache=True)
def _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE):
    """Schedule (d_i, dd_i, rho_i) of follower i at t on piece js[i]
    (before the closure onset: c_i(0), 0, 0)."""
    if kcl[i] < 0:
        return DPRE[i], 0.0, 0.0
    j = js[i]
    tau = t - (kcl[i] + SK[i, j]) * h
    r = SR[i, j]
    return SD[i, j] + SV[i, j] * tau - 0.5 * r * tau * tau, SV[i, j] - r * tau, r


@njit(cache=True)
def _law_hr(i, s, v, md, uhi, s0r, vr, ffh, d, dd, rho, F, I, d_s, bf, Umin):
    """Nominal law, filter and saturation of unit i at one instant.
    Returns (u_applied, u_nom, u_after_filter, u_cbf, code); code bits
    1 min-branch (filter output below u_nom), 2 fallback, 4 saturation,
    8 saturation at U, 16 saturation at -Uminus_i, 32 u_nom > 0.  With
    d = d_s[i], dd = rho = 0 the float operations are those of sim_cont._law."""
    ell = F[F_ELL]
    if md == LATCH:
        return 0.0, 0.0, 0.0, np.nan, 0
    if i == 0:
        gap = s0r - s[0] - ell
        eps = vr - v[0]
        epsr = eps
        ff = ffh
        e = gap - d_s[0]
    else:
        gap = s[i - 1] - s[i] - ell
        epsr = v[i - 1] - v[i]
        e = gap - d
        eps = epsr - dd
        ff = uhi + rho
    if md == RULE or md == BRAKE:
        un = -F[F_AB]
    elif md == S2:
        un = ff + F[F_BETA] * e + F[F_GAMMA] * eps
    elif md == S1:
        x = eps / F[F_PHI]
        if x < -1.0:
            x = -1.0
        elif x > 1.0:
            x = 1.0
        un = ff + F[F_ALPHA] * x
    else:
        un = ff
    uq = un
    ucbf = np.nan
    code = 0
    if i >= 1 and I[I_FILT] == 1:
        g = gap - F[F_SM]
        hb = g + v[i - 1] / bf[i - 1] - v[i] / bf[i]
        ucbf = bf[i] * (epsr + uhi / bf[i - 1] + F[F_KAPPA] * hb)
        if ucbf < -Umin[i] - 1e-12:
            uq = -Umin[i]
            code |= 2
        elif ucbf < un:
            uq = ucbf
        if uq < un - 1e-12:
            code |= 1
    lo = -Umin[i]
    hi = F[F_U]
    ua = uq
    if ua < lo:
        ua = lo
        code |= 16
    elif ua > hi:
        ua = hi
        code |= 8
    if ua != uq:
        code |= 4
    if un > 0.0:
        code |= 32
    return ua, un, uq, ucbf, code


@njit(cache=True)
def _stage_hr(r, t, s, v, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS, HV, jh,
              kcl, js, SK, SR, SD, SV, DPRE, ws):
    """All units' laws at (t, s, v) with the frozen pieces; row r of ws: the
    applied commands, 4+r the margins, 8+r the codes, 16+r u_nom."""
    h = F[F_H]
    s0r, vr, ffh = _headref(t, jh, F, I, HT, HA, HS, HV)
    n = s.shape[0]
    for i in range(n):
        if i == 0:
            d = 0.0
            dd = 0.0
            rho = 0.0
        else:
            d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
        ua, un, uq, ucbf, code = _law_hr(i, s, v, mode[i], uh[i], s0r, vr, ffh,
                                         d, dd, rho, F, I, d_s, bf, Umin)
        ws[r, i] = ua
        if i >= 1 and mode[i] != LATCH:
            ws[4 + r, i] = ucbf - un
        else:
            ws[4 + r, i] = np.nan
        ws[8 + r, i] = code
        ws[16 + r, i] = un


@njit(cache=True)
def _rk4_hr(t, tau, s, v, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS, HV, jh,
            kcl, js, SK, SR, SD, SV, DPRE, ws, s_out, v_out, st, vt):
    """One classical RK4 step of length tau with the discrete state frozen
    (same operation order as sim_cont._rk4); stage velocities in ws 12..15."""
    n = s.shape[0]
    half = 0.5 * tau
    _stage_hr(0, t, s, v, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS, HV, jh,
              kcl, js, SK, SR, SD, SV, DPRE, ws)
    for i in range(n):
        ws[12, i] = v[i]
        st[i] = s[i] + half * v[i]
        vt[i] = v[i] + half * ws[0, i]
    _stage_hr(1, t + half, st, vt, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS,
              HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws)
    for i in range(n):
        ws[13, i] = vt[i]
        st[i] = s[i] + half * ws[13, i]
        vt[i] = v[i] + half * ws[1, i]
    _stage_hr(2, t + half, st, vt, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS,
              HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws)
    for i in range(n):
        ws[14, i] = vt[i]
        st[i] = s[i] + tau * ws[14, i]
        vt[i] = v[i] + tau * ws[2, i]
    _stage_hr(3, t + tau, st, vt, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS,
              HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws)
    for i in range(n):
        ws[15, i] = vt[i]
        if mode[i] == LATCH:
            s_out[i] = s[i]
            v_out[i] = v[i]
        else:
            s_out[i] = s[i] + tau / 6.0 * (ws[12, i] + 2.0 * ws[13, i]
                                          + 2.0 * ws[14, i] + ws[15, i])
            v_out[i] = v[i] + tau / 6.0 * (ws[0, i] + 2.0 * ws[1, i]
                                          + 2.0 * ws[2, i] + ws[3, i])


@njit(cache=True)
def _root_rk4_hr(j, t, rem, fa, fb, s, v, mode, uh, F, I, d_s, bf, Umin, HT,
                 HA, HS, HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws2, s_try,
                 v_try, st, vt):
    """Zero of tau -> v_j(RK4(t, tau)) on (0, rem] (Illinois method, as
    sim_cont._root_rk4); fa = v_j(t) > 0 >= fb = v_j(RK4(t, rem))."""
    a = 0.0
    b = rem
    c = rem
    fc = fb
    side = 0
    for _ in range(200):
        den = fb - fa
        c = (a * fb - b * fa) / den if den != 0.0 else 0.5 * (a + b)
        if not (c > a and c < b):
            c = 0.5 * (a + b)
        _rk4_hr(t, c, s, v, mode, uh, F, I, d_s, bf, Umin, HT, HA, HS, HV, jh,
                kcl, js, SK, SR, SD, SV, DPRE, ws2, s_try, v_try, st, vt)
        fc = v_try[j]
        if fc == 0.0 or abs(fc) < 1e-15 or (b - a) < 1e-15:
            break
        if fc < 0.0:
            b = c
            fb = fc
            if side == -1:
                fa *= 0.5
            side = -1
        else:
            a = c
            fa = fc
            if side == 1:
                fb *= 0.5
            side = 1
    return c, abs(fc)


@njit(cache=True)
def _hermite_min(g0, g1, d0, d1, tau):
    """Minimum over [0, tau] of the cubic Hermite interpolant with values
    g0, g1 and slopes d0, d1 (copy of sim_cont._hermite_min)."""
    best = g0 if g0 < g1 else g1
    if not (d0 < 0.0 and d1 > 0.0):
        return best
    m0 = tau * d0
    m1 = tau * d1
    a = 2.0 * g0 + m0 - 2.0 * g1 + m1
    b = -3.0 * g0 - 2.0 * m0 + 3.0 * g1 - m1
    c = m0
    A3 = 3.0 * a
    B2 = 2.0 * b
    if abs(A3) < 1e-300:
        xs0 = -c / B2 if B2 != 0.0 else -1.0
        xs1 = -1.0
    else:
        disc = B2 * B2 - 4.0 * A3 * c
        if disc < 0.0:
            return best
        sq = math.sqrt(disc)
        xs0 = (-B2 + sq) / (2.0 * A3)
        xs1 = (-B2 - sq) / (2.0 * A3)
    for x in (xs0, xs1):
        if 0.0 < x < 1.0:
            p = ((a * x + b) * x + c) * x + g0
            if p < best:
                best = p
    return best


@njit(cache=True)
def _ustats(i, ua, un, md, pre, inf, umn, umx, umn_pre, umx_pre, ulat, unmx,
            unmx_pre, umx_inf):
    """Command range before LATCH (all, before T_xi), |u| after LATCH, the
    largest nominal command (all, before T_xi) and the largest command
    before T_xi once the unit is informed (head: always; follower: from
    the first delivery of a packet whose sender was informed, i.e. without
    the zero commands that COAST relays before the head's information
    arrives)."""
    if md == LATCH:
        x = abs(ua)
        if x > ulat[i]:
            ulat[i] = x
        return
    if ua < umn[i]:
        umn[i] = ua
    if ua > umx[i]:
        umx[i] = ua
    if un > unmx[i]:
        unmx[i] = un
    if pre:
        if ua < umn_pre[i]:
            umn_pre[i] = ua
        if ua > umx_pre[i]:
            umx_pre[i] = ua
        if un > unmx_pre[i]:
            unmx_pre[i] = un
        if inf and ua > umx_inf[i]:
            umx_inf[i] = ua


@njit(cache=True)
def _count(i, md, code, cnt):
    """Event counters of one evaluation (min-branch and fallback: followers
    not latched, as sim_cont; saturation and positive u_nom: every unit)."""
    if i >= 1 and md != LATCH:
        if code & 1:
            cnt[0, i] += 1
        if code & 2:
            cnt[1, i] += 1
    if code & 4:
        cnt[2, i] += 1
    if code & 8:
        cnt[3, i] += 1
    if code & 16:
        cnt[4, i] += 1
    if code & 32:
        cnt[5, i] += 1


@njit(cache=True)
def _core_hr(F, I, d_s, s, v, b, Umin, deliv, kfb,
             HT, HK, HA, HS, HV, NS, SK, SR, SD, SV, DPRE,
             tstop, tact, tset, trule, tcl, esw, xsw,
             gmin, hmin, gmin_grid, mmin_eval, mmin_lim, upk, cnt, tim,
             umn, umx, umn_pre, umx_pre, ulat, unmx, unmx_pre, umx_inf,
             wint, epsmax_b, fb_real, e_hnd, eps_hnd, iout, fout,
             rt, rs, rv, re, reps, ru, rm, rg, rh):
    """Closed loop of the head-only reference arrival ('cont' and 'zoh')."""
    n = I[I_N]
    N = I[I_NSTEP]
    h = F[F_H]
    Nm = I[I_NM]
    Nc = I[I_NC]
    Nu = I[I_NU]
    sch = I[I_SCH]
    k_xi = I[I_KXI]
    kt0 = I[I_KT0]
    refm = I[I_REFM]
    NH = I[I_NH]
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    Nsend = I[I_NSEND]
    T_xi = F[F_TXI]
    a_b = F[F_AB]
    kappa = F[F_KAPPA]
    s_m = F[F_SM]
    ell = F[F_ELL]
    eps_e = F[F_EPSE]
    eps_v = F[F_EPSV]
    eps_det = F[F_EPSDET]
    tau_r = F[F_TAUR]

    mode = np.zeros(n, np.int64)
    mode[0] = S1
    kact = np.full(n, -1, np.int64)
    kact[0] = 0
    uh = np.zeros(n)
    flagm = np.zeros(n, np.bool_)
    flagc = np.zeros(n, np.bool_)
    closed = np.zeros(n, np.bool_)
    informed = np.zeros(n, np.bool_)
    informed[0] = True
    kcl = np.full(n, -1, np.int64)
    js = np.zeros(n, np.int64)
    jsl = np.zeros(n, np.int64)
    sjev = np.zeros(n, np.bool_)
    jh = 0
    ptr = np.zeros(n, np.int64)
    ucmd = np.zeros(n)
    ccur = np.zeros(n, np.int64)
    mlast = np.full(n, np.nan)
    usent = np.zeros((n, Nsend))
    msent = np.zeros((n, Nsend), np.bool_)
    csent = np.zeros((n, Nsend), np.bool_)
    isent = np.zeros((n, Nsend), np.bool_)
    ws = np.zeros((20, n))
    ws2 = np.zeros((20, n))
    st = np.zeros(n)
    vt = np.zeros(n)
    s_new = np.zeros(n)
    v_new = np.zeros(n)
    s_try = np.zeros(n)
    v_try = np.zeros(n)
    hits_true = 0
    n_loc = 0
    neg_speed = 0
    n_left = 0
    n_eval = 0
    contact_time = np.nan
    rbox = np.zeros(4)
    rprev = np.zeros(4)
    lobox = np.full(4, -1.0)
    crbox = np.full(4, -1.0)
    max_resid = 0.0
    irec = 0
    k_end = N
    k = 0
    while k < N:
        t = k * h
        pre = k < k_xi
        # ------------------------------------------ planned pieces at t_k
        # (right limits; jsl keeps the left-limit pieces of the followers)
        hjev = False
        if refm == 0:
            while jh + 1 < NH and HK[jh + 1] >= 0 and HK[jh + 1] <= k:
                jh += 1
                hjev = True
        for i in range(1, n):
            jsl[i] = js[i]
            sjev[i] = False
            if kcl[i] >= 0:
                while js[i] + 1 < NS[i] and kcl[i] + SK[i, js[i] + 1] <= k:
                    js[i] += 1
                    sjev[i] = True
        srf, vrf, ffh = _headref(t, jh, F, I, HT, HA, HS, HV)
        jx = _href_idx(t, NH, HT)
        sx, vx, ax = _href(t, jx, HT, HA, HS, HV)
        # ------------------------------------------ accepted state at t_k
        for i in range(1, n):
            gap = s[i - 1] - s[i] - ell
            g = gap - s_m
            hb = g + v[i - 1] / b[i - 1] - v[i] / b[i]
            if g < gmin[i - 1]:
                gmin[i - 1] = g
            if g < gmin_grid[i - 1]:
                gmin_grid[i - 1] = g
            if hb < hmin[i - 1]:
                hmin[i - 1] = hb
            if gap <= 0.0 and np.isnan(contact_time):
                contact_time = t
        if not np.isnan(contact_time):
            k_end = k
            break
        if pre:
            # handoff-box excess per channel (schedule-relative pair errors,
            # head pair): 0 all units, 1 velocity only, 2 position only,
            # 3 followers only
            for q in range(4):
                rbox[q] = -np.inf
            for i in range(n):
                if i == 0:
                    ei = sx - s[0] - ell - d_s[0]
                    epi = vx - v[0]
                else:
                    d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
                    ei = s[i - 1] - s[i] - ell - d
                    epi = v[i - 1] - v[i] - dd
                xe = abs(ei) - eps_e
                xv = abs(epi) - eps_v
                xm = xe if xe > xv else xv
                if xm > rbox[0]:
                    rbox[0] = xm
                if xv > rbox[1]:
                    rbox[1] = xv
                if xe > rbox[2]:
                    rbox[2] = xe
                if i >= 1 and xm > rbox[3]:
                    rbox[3] = xm
            for q in range(4):
                if rbox[q] > 0.0:
                    lobox[q] = t
                elif k > 0 and rprev[q] > 0.0:
                    crbox[q] = (t - h) + h * rprev[q] / (rprev[q] - rbox[q])
                rprev[q] = rbox[q]
        if k == k_xi:
            for i in range(n):
                if i == 0:
                    e_hnd[0] = sx - s[0] - ell - d_s[0]
                    eps_hnd[0] = vx - v[0]
                else:
                    d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
                    e_hnd[i] = s[i - 1] - s[i] - ell - d
                    eps_hnd[i] = v[i - 1] - v[i] - dd
        # ------------------------------------------ per unit: events at t_k,
        # command update, send (unit order 0, 1, ..., n-1)
        for i in range(n):
            md0 = mode[i]
            lat = False
            if md0 == S2 or md0 == RULE or md0 == BRAKE:
                lat = v[i] <= 0.0
            rul = False
            if (not lat) and md0 == S2 and i >= 1:
                rul = v[i - 1] <= 0.0
            brk = i == 0 and k == k_xi and md0 != LATCH and not lat
            clh = i == 0 and k == kt0 and not closed[0]
            pkt = i >= 1 and ptr[i] < Nsend and deliv[i, ptr[i]] <= k
            det_due = (md0 == S1 and (k - kact[i]) % Nc == 0) or \
                (md0 == COAST and pkt)
            jev = hjev if i == 0 else sjev[i]
            ev = lat or rul or brk or clh or pkt or det_due or jev
            if ev and i >= 1 and md0 != LATCH:
                # left limit: previous mode, received command and schedule
                d, dd, rho = _sched(i, t, h, kcl, jsl, SK, SR, SD, SV, DPRE)
                ua, un, uq, ucbf, code = _law_hr(i, s, v, md0, uh[i], srf, vrf,
                                                 ffh, d, dd, rho, F, I, d_s, b,
                                                 Umin)
                mg = ucbf - un
                if mg < mmin_lim[i - 1]:
                    mmin_lim[i - 1] = mg
                n_left += 1
            if lat:
                mode[i] = LATCH
                tstop[i] = t
                v[i] = 0.0
            elif rul:
                mode[i] = RULE
                trule[i] = t
            if brk:
                mode[0] = BRAKE
            if clh:
                closed[0] = True
                tcl[0] = t
            if i >= 1:
                while ptr[i] < Nsend and deliv[i, ptr[i]] <= k:
                    j = ptr[i]
                    uh[i] = usent[i - 1, j]
                    if msent[i - 1, j]:
                        flagm[i] = True
                    if csent[i - 1, j]:
                        flagc[i] = True
                    if isent[i - 1, j]:
                        informed[i] = True
                    ptr[i] += 1
                if mode[i] == COAST and flagm[i]:
                    mode[i] = S1
                    kact[i] = k
                    tact[i] = t
                if flagc[i] and not closed[i]:
                    closed[i] = True
                    tcl[i] = t
                    if NS[i] > 0:
                        kcl[i] = k
                        js[i] = 0
                        while js[i] + 1 < NS[i] and SK[i, js[i] + 1] <= 0:
                            js[i] += 1
            if mode[i] == S1 and (k - kact[i]) % Nc == 0:
                if i == 0:
                    ei = srf - s[0] - ell - d_s[0]
                    epi = vrf - v[0]
                else:
                    d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
                    ei = s[i - 1] - s[i] - ell - d
                    epi = v[i - 1] - v[i] - dd
                if abs(epi) <= eps_det:
                    mode[i] = S2
                    tset[i] = t
                    esw[i] = ei
                    xsw[i] = epi
            # ---------------------------------- command update
            if sch == SCH_CONT or k % Nu == 0 or ev:
                if i == 0:
                    d = 0.0
                    dd = 0.0
                    rho = 0.0
                else:
                    d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
                ua, un, uq, ucbf, code = _law_hr(i, s, v, mode[i], uh[i], srf,
                                                 vrf, ffh, d, dd, rho, F, I,
                                                 d_s, b, Umin)
                ucmd[i] = ua
                ccur[i] = code
                n_eval += 1
                if abs(ua) > upk[i]:
                    upk[i] = abs(ua)
                if sch == SCH_ZOH:
                    _ustats(i, ua, un, mode[i], pre, informed[i], umn, umx,
                            umn_pre, umx_pre, ulat, unmx, unmx_pre, umx_inf)
                if i >= 1 and mode[i] != LATCH:
                    mg = ucbf - un
                    mlast[i] = mg
                    if mg < mmin_eval[i - 1]:
                        mmin_eval[i - 1] = mg
                    if mg < mmin_lim[i - 1]:
                        mmin_lim[i - 1] = mg
                else:
                    mlast[i] = np.nan
                _count(i, mode[i], code, cnt)
            # ---------------------------------- send (post-event value)
            if k % Nm == 0:
                j = k // Nm
                if j < Nsend:
                    usent[i, j] = ucmd[i]
                    msent[i, j] = mode[i] >= S2
                    csent[i, j] = closed[i]
                    isent[i, j] = informed[i]
        # barrier-condition check with the current commands
        for i in range(1, n):
            g = s[i - 1] - s[i] - ell - s_m
            hb = g + v[i - 1] / b[i - 1] - v[i] / b[i]
            eps = v[i - 1] - v[i]
            if mode[i] != LATCH:
                hdt = eps + ucmd[i - 1] / b[i - 1] - ucmd[i] / b[i]
                if hdt + kappa * hb < -1e-12:
                    hits_true += 1
        # braking-window diagnostics at t_k
        if k >= k_xi:
            for i in range(1, n):
                if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                    eps = v[i - 1] - v[i]
                    if abs(eps) > epsmax_b[i - 1]:
                        epsmax_b[i - 1] = abs(eps)
                if sch != SCH_CONT and k >= kfb[i - 1] and \
                        np.isnan(tstop[i - 1]):
                    fb = abs(ucmd[i - 1] + a_b)
                    if fb > fb_real[i - 1]:
                        fb_real[i - 1] = fb
        # ------------------------------------------ record
        if k % rec_every == 0 and irec < nrec:
            rt[irec] = t
            for i in range(n):
                rs[i, irec] = s[i]
                rv[i, irec] = v[i]
                ru[i, irec] = ucmd[i]
                if i == 0:
                    re[0, irec] = sx - s[0] - ell - d_s[0]
                    reps[0, irec] = vx - v[0]
                else:
                    d, dd, rho = _sched(i, t, h, kcl, js, SK, SR, SD, SV, DPRE)
                    re[i, irec] = s[i - 1] - s[i] - ell - d
                    reps[i, irec] = v[i - 1] - v[i] - dd
                    rg[i - 1, irec] = s[i - 1] - s[i] - ell - s_m
                    rh[i - 1, irec] = rg[i - 1, irec] + v[i - 1] / b[i - 1] \
                        - v[i] / b[i]
                    rm[i - 1, irec] = mlast[i]
            irec += 1
        # ------------------------------------------ integrate to t_{k+1}
        tcur = t
        t_end = (k + 1) * h
        while True:
            rem = t_end - tcur
            if rem <= 0.0:
                break
            pre_s = tcur < T_xi
            bi = -1
            tau = rem
            if sch == SCH_ZOH:
                for i in range(n):
                    mdi = mode[i]
                    if v[i] > 0.0 and ucmd[i] < 0.0:
                        tz = -v[i] / ucmd[i]
                        if mdi == S2 or mdi == RULE or mdi == BRAKE:
                            if tz <= tau:
                                tau = tz
                                bi = i
                        elif mdi != LATCH and tz < rem:
                            neg_speed += 1
                for i in range(n):
                    if mode[i] != LATCH:
                        s_new[i] = s[i] + v[i] * tau + 0.5 * ucmd[i] * tau * tau
                        v_new[i] = v[i] + ucmd[i] * tau
                    else:
                        s_new[i] = s[i]
                        v_new[i] = v[i]
                    c = ccur[i]
                    for q in range(6):
                        if c & (1 << q):
                            tim[q, i] += tau
                if k >= k_xi:
                    for i in range(1, n):
                        if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                            wint[i - 1] += tau * abs(ucmd[i - 1] - uh[i])
            else:
                _rk4_hr(tcur, rem, s, v, mode, uh, F, I, d_s, b, Umin, HT, HA,
                        HS, HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws, s_new,
                        v_new, st, vt)
                for i in range(n):
                    mdi = mode[i]
                    if v[i] > 0.0 and v_new[i] <= 0.0:
                        if mdi == S2 or mdi == RULE or mdi == BRAKE:
                            tz, rsd = _root_rk4_hr(i, tcur, rem, v[i], v_new[i],
                                                   s, v, mode, uh, F, I, d_s, b,
                                                   Umin, HT, HA, HS, HV, jh,
                                                   kcl, js, SK, SR, SD, SV,
                                                   DPRE, ws2, s_try, v_try, st,
                                                   vt)
                            if rsd > max_resid:
                                max_resid = rsd
                            if tz < tau:
                                tau = tz
                                bi = i
                        elif mdi != LATCH:
                            neg_speed += 1
                if bi >= 0:
                    _rk4_hr(tcur, tau, s, v, mode, uh, F, I, d_s, b, Umin, HT,
                            HA, HS, HV, jh, kcl, js, SK, SR, SD, SV, DPRE, ws,
                            s_new, v_new, st, vt)
                # stage statistics of the accepted (sub)step
                for i in range(n):
                    for q in range(6):
                        bit = 1 << q
                        w = 0.0
                        if int(ws[8, i]) & bit:
                            w += 1.0
                        if int(ws[9, i]) & bit:
                            w += 2.0
                        if int(ws[10, i]) & bit:
                            w += 2.0
                        if int(ws[11, i]) & bit:
                            w += 1.0
                        if w > 0.0:
                            tim[q, i] += tau * w / 6.0
                for r in range(4):
                    for i in range(n):
                        ua = ws[r, i]
                        if abs(ua) > upk[i]:
                            upk[i] = abs(ua)
                        code = int(ws[8 + r, i])
                        n_eval += 1
                        _ustats(i, ua, ws[16 + r, i], mode[i], pre_s,
                                informed[i], umn, umx, umn_pre, umx_pre, ulat,
                                unmx, unmx_pre, umx_inf)
                        if i >= 1 and mode[i] != LATCH:
                            mg = ws[4 + r, i]
                            if mg < mmin_eval[i - 1]:
                                mmin_eval[i - 1] = mg
                            if mg < mmin_lim[i - 1]:
                                mmin_lim[i - 1] = mg
                        _count(i, mode[i], code, cnt)
                        if i >= 1 and k >= kfb[i - 1] and \
                                np.isnan(tstop[i - 1]):
                            fb = abs(ws[r, i - 1] + a_b)
                            if fb > fb_real[i - 1]:
                                fb_real[i - 1] = fb
                if k >= k_xi:
                    for i in range(1, n):
                        if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                            w1 = abs(ws[0, i - 1] - uh[i])
                            w2 = abs(ws[1, i - 1] - uh[i])
                            w3 = abs(ws[2, i - 1] - uh[i])
                            w4 = abs(ws[3, i - 1] - uh[i])
                            wint[i - 1] += tau / 6.0 * (w1 + 2.0 * w2
                                                        + 2.0 * w3 + w4)
            # intersample clearance minimum and barrier at the new state
            for i in range(1, n):
                g0 = s[i - 1] - s[i] - ell - s_m
                g1 = s_new[i - 1] - s_new[i] - ell - s_m
                d0 = v[i - 1] - v[i]
                d1 = v_new[i - 1] - v_new[i]
                gm = _hermite_min(g0, g1, d0, d1, tau)
                if gm < gmin[i - 1]:
                    gmin[i - 1] = gm
                hb1 = g1 + v_new[i - 1] / b[i - 1] - v_new[i] / b[i]
                if hb1 < hmin[i - 1]:
                    hmin[i - 1] = hb1
                if g1 + s_m <= 0.0 and np.isnan(contact_time):
                    contact_time = tcur + tau
            for i in range(n):
                s[i] = s_new[i]
                v[i] = v_new[i]
            if bi < 0:
                break
            # ---- localized zero-speed event of unit bi at tcur + tau
            tcur = tcur + tau
            n_loc += 1
            pre_z = tcur < T_xi
            ev_units = (bi, bi + 1)
            for jj in range(2):
                q = ev_units[jj]
                if q >= 1 and q < n and mode[q] != LATCH:
                    d, dd, rho = _sched(q, tcur, h, kcl, js, SK, SR, SD, SV,
                                        DPRE)
                    ua, un, uq, ucbf, code = _law_hr(q, s, v, mode[q], uh[q],
                                                     0.0, 0.0, 0.0, d, dd, rho,
                                                     F, I, d_s, b, Umin)
                    mg = ucbf - un
                    if mg < mmin_lim[q - 1]:
                        mmin_lim[q - 1] = mg
                    n_left += 1
            v[bi] = 0.0
            mode[bi] = LATCH
            tstop[bi] = tcur
            ucmd[bi] = 0.0
            ccur[bi] = 0
            if bi + 1 < n and mode[bi + 1] == S2:
                mode[bi + 1] = RULE
                trule[bi + 1] = tcur
                # event-triggered command update (right limit)
                d, dd, rho = _sched(bi + 1, tcur, h, kcl, js, SK, SR, SD, SV,
                                    DPRE)
                ua, un, uq, ucbf, code = _law_hr(bi + 1, s, v, mode[bi + 1],
                                                 uh[bi + 1], 0.0, 0.0, 0.0, d,
                                                 dd, rho, F, I, d_s, b, Umin)
                if sch == SCH_ZOH:
                    ucmd[bi + 1] = ua
                    ccur[bi + 1] = code
                    n_eval += 1
                    if abs(ua) > upk[bi + 1]:
                        upk[bi + 1] = abs(ua)
                    _ustats(bi + 1, ua, un, mode[bi + 1], pre_z,
                            informed[bi + 1], umn, umx, umn_pre, umx_pre, ulat,
                            unmx, unmx_pre, umx_inf)
                    _count(bi + 1, mode[bi + 1], code, cnt)
                mg = ucbf - un
                if mg < mmin_eval[bi]:
                    mmin_eval[bi] = mg
                if mg < mmin_lim[bi]:
                    mmin_lim[bi] = mg
            # a latched unit's successor in RULE keeps -a_b until its own
            # zero-speed instant; nothing else changes at this instant
        if not np.isnan(contact_time):
            k_end = k + 1
            break
        # ------------------------------------------ early exit
        if I[I_EARLY] == 1 and t > tau_r + 2.0 * h:
            done = True
            for i in range(n):
                if mode[i] != LATCH:
                    done = False
                    break
            if done:
                k_end = k + 1
                break
        k += 1
    iout[0] = 0
    iout[1] = hits_true
    iout[2] = n_loc
    iout[3] = neg_speed
    iout[4] = n_left
    iout[5] = n_eval
    iout[6] = k_end
    iout[7] = irec
    fout[0] = contact_time
    for q in range(4):
        fout[2 + q] = lobox[q]
        fout[8 + q] = crbox[q]
    fout[6] = max_resid


# ---------------------------------------------------------------- setups
class HRSetup:
    """Data of one HR configuration (hr_setup / profile_setup).

    Unit indices are 0-based (head 0; follower i >= 1 pairs with i-1).
    Attributes: n, ell, d_s, c0 (pair clearances at takeover, length n-1),
    s0, v0 (initial state), amax, b (barrier slopes amax/v_c), U, Uminus,
    gains, tolerances, t_0, T_xi, tau_r, s_ref_stop, markers, head (pieces
    (t0, a_r, s_0, v_r) floats; head_exact rationals), sched (per follower
    i >= 1: dict with exact lists sigma, rho, d, dd and float arrays
    sigma_f, rho_f, d_f, dd_f), d_pre (c_i(0) floats, index i >= 1),
    ref_mode, T_END, kind ('hr' or 'profile'), plan (certify_plan.Plan).
    Evaluators: ref(t), d(i, sigma), dd(i, sigma), rho(i, sigma) (floats,
    vectorized) and sched_exact(i, sigma) (rationals from the plan)."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    # ---- head reference
    def ref(self, t):
        """(s_0, v_r, a_r) at t (right-continuous), arrays."""
        t = np.atleast_1d(np.asarray(t, float))
        T = np.array([p[0] for p in self.head])
        j = np.clip(np.searchsorted(T, t, side='right') - 1, 0, len(T) - 1)
        A = np.array([p[1] for p in self.head])[j]
        S = np.array([p[2] for p in self.head])[j]
        V = np.array([p[3] for p in self.head])[j]
        tau = t - T[j]
        return S + V * tau + 0.5 * A * tau * tau, V + A * tau, A

    # ---- follower schedules (local time sigma)
    def _piece(self, i, sigma):
        sg = np.atleast_1d(np.asarray(sigma, float))
        S = self.sched[i]
        j = np.clip(np.searchsorted(S['sigma_f'], sg, side='right') - 1, 0,
                    len(S['sigma_f']) - 1)
        return sg, S, j, sg - S['sigma_f'][j]

    def rho(self, i, sigma):
        sg, S, j, tau = self._piece(i, sigma)
        return np.where(sg < 0.0, 0.0, S['rho_f'][j])

    def dd(self, i, sigma):
        sg, S, j, tau = self._piece(i, sigma)
        return np.where(sg < 0.0, 0.0, S['dd_f'][j] - S['rho_f'][j] * tau)

    def d(self, i, sigma):
        sg, S, j, tau = self._piece(i, sigma)
        val = S['d_f'][j] + S['dd_f'][j] * tau - 0.5 * S['rho_f'][j] * tau * tau
        return np.where(sg < 0.0, self.d_pre[i], val)

    def sched_exact(self, i, sigma):
        """Exact (d_i, dd_i, rho_i) from the plan (spec Sec. 2)."""
        sigma = Fr(sigma) if not isinstance(sigma, Fr) else sigma
        if sigma < 0:
            return self.c0_exact[i - 1], Fr(0), Fr(0)
        Pl = self.plan
        t = self.t_0_exact + sigma
        sp, vp, ap = Pl.state(i - 1, t)
        sf, vf, af = Pl.state(i, t)
        rho = (af - ap) if t < Pl.T_xi else Fr(0)
        return sp - sf - Pl.ell, vp - vf, rho


def _fr(x):
    if isinstance(x, Fr):
        return x
    if isinstance(x, int):
        return Fr(x)
    return Fr(str(x))


def hr_setup(cfg, U=0.02, Uminus=None):
    """HR configuration of a Case D cfg (certify_plan format): reference of
    the head = plan of unit 1, schedules of the followers (spec Sec. 2),
    input sets [-Uminus_i, U] (Uminus None: 1.2 m/s^2 for every unit)."""
    import certify_plan as cq
    Pl = cq.plan(cfg)
    n = int(Pl.n)
    ell = Pl.ell
    d_s = list(Pl.d_s)
    t_0 = Pl.extra.get('t_0', Fr(0))
    T_xi = Pl.T_xi
    a_b = Pl.a_b
    c0 = [_fr(x) for x in cfg['c0']]
    shaped = Pl.family == 'shaped'
    if shaped:
        # FB design (2026-09-29): the plan keeps the parking gaps; the units
        # take over at the clearances c0 with the head at its slot, and the
        # followers regulate to the parking gaps from takeover (no
        # schedules; run(..., schedules=False, nosched='park' or 'hold'))
        s0 = list(cq.takeover_positions(d_s, c0, ell))
        assert s0[0] == Pl.s0[0], 'head at its slot'
    else:
        s0 = list(Pl.s0)
    # takeover clearances of the units equal cfg['c0'] exactly
    for i in range(1, n):
        assert s0[i - 1] - s0[i] - ell == c0[i - 1], 'takeover clearance'
    # ---- head reference (virtual leader), exact
    off = d_s[0] + ell
    head_exact = [(t0, a, s + off, v) for (t0, t1, a, s, v) in Pl.segs[0]]
    assert head_exact[-1][0] == T_xi and head_exact[-1][1] == -a_b
    assert head_exact[0][2] == 0, 's_0(0) = 0'
    tau_r = Pl.tau_star
    s_stop = Pl.s_ref_stop
    sx, vx, _ = Pl.state(0, T_xi)
    assert tau_r == T_xi + vx / a_b
    assert sx + off + vx * vx / (2 * a_b) == s_stop, 's_0(tau_r)'
    head_exact.append((tau_r, Fr(0), s_stop, Fr(0)))
    # ---- follower schedules, exact pieces at the jumps of rho_i
    sched = {}
    d_pre = np.full(n, np.nan)
    for i in range(1, n):
        if shaped:
            d_pre[i] = float(d_s[i])
            continue
        br = set([t_0])
        for u in (Pl.units[i], Pl.units[i - 1]):
            for (tj, a) in u:
                if t_0 < tj < T_xi:
                    br.add(tj)
        br.add(T_xi)
        rows = []
        for tb in sorted(br):
            sp, vp, ap = Pl.state(i - 1, tb)
            sf, vf, af = Pl.state(i, tb)
            rho = (af - ap) if tb < T_xi else Fr(0)
            if rows and rows[-1]['rho'] == rho:
                continue                    # rho continuous: no piece break
            rows.append(dict(sigma=tb - t_0, rho=rho, d=sp - sf - ell,
                             dd=vp - vf))
        # exact consistency of the pieces
        assert rows[0]['sigma'] == 0
        assert rows[0]['d'] == c0[i - 1] and rows[0]['dd'] == 0, 'd_i(0), dd_i(0)'
        for a, bq in zip(rows[:-1], rows[1:]):
            L = bq['sigma'] - a['sigma']
            assert L > 0
            assert bq['d'] == a['d'] + a['dd'] * L - a['rho'] * L * L / 2
            assert bq['dd'] == a['dd'] - a['rho'] * L
        last = rows[-1]
        assert last['rho'] == 0 and last['dd'] == 0 and last['d'] == d_s[i], \
            'schedule constant after the join'
        sched[i] = dict(sigma=[r['sigma'] for r in rows],
                        rho=[r['rho'] for r in rows],
                        d=[r['d'] for r in rows], dd=[r['dd'] for r in rows],
                        sigma_f=np.array([float(r['sigma']) for r in rows]),
                        rho_f=np.array([float(r['rho']) for r in rows]),
                        d_f=np.array([float(r['d']) for r in rows]),
                        dd_f=np.array([float(r['dd']) for r in rows]),
                        on_1ms=all((r['sigma'] * 1000).denominator == 1
                                   for r in rows),
                        t_join=float(t_0 + rows[-1]['sigma']))
        d_pre[i] = float(c0[i - 1])
    amax = np.array([float(x) for x in cfg['amax']])
    v_c = float(cq.FIXED['v_c'])
    if Uminus is None:
        Umin = np.full(n, 1.2)
    else:
        Umin = np.broadcast_to(np.asarray(Uminus, float), (n,)).copy()
    if not (np.all(Umin > 0.0) and np.all(Umin <= amax)):
        raise ValueError('need 0 < Uminus_i <= amax_i')
    if not float(U) > 0.0:
        raise ValueError('need U > 0')
    lam = float(cfg['lam'])
    head_f = [(float(a), float(b_), float(c), float(d)) for (a, b_, c, d) in
              head_exact]
    return HRSetup(
        kind='hr', n=n, ell=float(ell), d_s=np.array([float(x) for x in d_s]),
        c0=np.array([float(x) for x in c0]), c0_exact=c0,
        s0=np.array([float(x) for x in s0]),
        v0=np.array([float(_fr(x)) for x in cfg['v0']]),
        amax=amax, v_c=v_c, b=amax / v_c, kappa=float(cq.FIXED['kappa']),
        s_m=float(cfg['s_m']), a_b=float(a_b), alpha=float(cfg['alpha']),
        lam=lam, beta=lam * lam, gamma=2.0 * lam, phi=float(cfg['phi']),
        eps_det=float(cfg['eps_det']), T_c=float(cfg['T_c']),
        eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
        dbar=float(cfg['dbar']), U=float(U), Uminus=Umin,
        t_0=float(t_0), t_0_exact=t_0, T_xi=float(T_xi), T_xi_exact=T_xi,
        tau_r=float(tau_r), tau_r_exact=tau_r, s_ref_stop=float(s_stop),
        markers=np.array([float(x) for x in Pl.markers]),
        head=head_f, head_exact=head_exact, sched=None if shaped else sched,
        d_pre=d_pre,
        ref_mode='frozen', T_END=float(T_xi) + 8.0, plan=Pl, cfg=cfg,
        v_c1=float(vx))


def profile_setup(P, d_s, v0, e0, profile, eps_e=0.15, eps_v=0.02):
    """HR setup of a sim_cont profile-mode configuration (Case C): the head
    reference is the piecewise-linear speed profile of sim_cont (start states
    formed with the float operations of sim_cont._ref_profile), no follower
    schedules (use run(..., schedules=False)), CLOSE raised at t_0 = 0,
    input sets [-min(U, amax_i), U] with U = P['U'], initial state
    s = -cumsum(ell + d_s + e0), v = v0, and ref_mode 'stage' (the sim_cont
    convention)."""
    n = int(P['n'])
    vxi = float(P['v_xi'])
    txi = float(P['T_xi'])
    ab = float(P['a_b'])
    td = float(profile['t_d'])
    ad = float(profile['a_d'])
    vc = float(profile['v_c1'])
    tc = td + (vxi - vc) / ad
    if not (ad > 0.0 and 0.0 < vc <= vxi and 0.0 < td and tc <= txi):
        raise ValueError('profile needs a_d > 0, 0 < v_c1 <= v_xi, 0 < t_d, '
                         't_c <= T_xi')
    s_d = vxi * td
    s_c = s_d + 0.5 * (vxi + vc) * (tc - td)
    s_xi = s_c + vc * (txi - tc)
    tr = vc / ab
    head = [(0.0, 0.0, 0.0, vxi), (td, -ad, s_d, vxi), (tc, 0.0, s_c, vc),
            (txi, -ab, s_xi, vc), (txi + tr, 0.0, s_xi + 0.5 * vc * tr, 0.0)]
    ell = float(P['ell'])
    d_s = np.array(d_s, float)
    e0 = np.array(e0, float)
    s = -np.cumsum(ell + d_s + e0)
    amax = np.array(P['amax'], float)
    mark_ref = s_xi + vc ** 2 / (2 * ab)
    d_pre = np.full(n, np.nan)
    for i in range(1, n):
        d_pre[i] = s[i - 1] - s[i] - ell
    return HRSetup(
        kind='profile', n=n, ell=ell, d_s=d_s, c0=d_pre[1:].copy(),
        s0=s, v0=np.array(v0, float), amax=amax, v_c=float(P['v_c']),
        b=amax / P['v_c'], kappa=float(P['kappa']), s_m=float(P['s_m']),
        a_b=ab, alpha=float(P['alpha']), lam=None, beta=float(P['beta']),
        gamma=float(P['gamma']), phi=float(P['phi']),
        eps_det=float(P['eps_det']), T_c=None, eps_e=float(eps_e),
        eps_v=float(eps_v), dbar=float(P.get('dbar', np.nan)),
        U=float(P['U']), Uminus=np.minimum(float(P['U']), amax),
        t_0=0.0, T_xi=txi, tau_r=txi + vc / ab, s_ref_stop=mark_ref,
        markers=mark_ref - np.cumsum(ell + d_s), head=head, head_exact=None,
        sched=None, d_pre=d_pre, ref_mode='stage', T_END=float(P['T_END']),
        plan=None, cfg=None, v_c1=vc, profile=dict(t_d=td, a_d=ad, v_c1=vc,
                                                    t_c=tc))


# ---------------------------------------------------------------- wrapper
def _on_grid(x, h, name, allow_zero=True):
    q = float(x) / h
    k = int(round(q))
    if abs(q - k) > 1e-6 or k < (0 if allow_zero else 1):
        raise ValueError('%s = %.12g is not an integer multiple of h = %g'
                         % (name, float(x), h))
    return k


def _fit(fifo, Nsend):
    """FIFO rows cover every send of the run (pad by the last value)."""
    f = np.asarray(fifo)
    if f.shape[1] >= Nsend:
        return f[:, :Nsend]
    pad = np.repeat(f[:, -1:], Nsend - f.shape[1], axis=1)
    return np.concatenate([f, pad], axis=1)


def run(setup, scheme='cont', h=None, T_m=1e-3, T_c=None, T_u=None,
        transport=0.019, fifo=None, T_end=None, rec_dt=0.01, schedules=True,
        filter=True, nosched='step', ref_mode=None, early_exit=True,
        eps_e=None, eps_v=None):
    """One closed-loop run of the head-only reference arrival.

    setup      hr_setup(cfg) or profile_setup(...).
    scheme     'cont' (RK4 between events, T_u = h) or 'zoh' (digital, T_u).
    h          integration step (default 1e-4 s for 'cont', T_u for 'zoh').
    T_m, T_c, T_u  message, detector (default setup.T_c or T_m) and update
               (default T_m for 'zoh') periods; integer multiples of h.
    transport  constant transport in seconds (integer multiple of h; 0 is
               the same-instant relay), or fifo = per-send FIFO transport
               delays in units of T_m (row i: hop i-1 -> i; padded).
    schedules  False: no schedules (rho = dd = 0; nosched 'step': d_i =
               c_i(0) before the closure onset and d_s from it; 'hold': d_i =
               c_i(0) always).
    filter     False: the safety filter is bypassed (saturation only).
    ref_mode   'frozen' / 'stage' (default: setup.ref_mode).
    Returns a dict (arrays per unit i = 0..n-1, per pair i-1 = 0..n-2):
      tstop (tau_i), disp, T_f, stop_order, all_stopped, contact_time,
      marker_signed, align (max |marker|), egap (head slot, pairs),
      gap_signed (pairs), gaperr, head_slot_err;
      gmin (smallest g_i incl. the cubic-Hermite intersample minimum),
      gmin_grid, hmin (smallest barrier h_i);
      mmin (smallest m_i = u_cbf - u_nom over evaluations), mmin_lim (also
      the left limits at events);
      u_min, u_max (before LATCH), u_min_pre, u_max_pre (before T_xi),
      u_max_pre_all, u_max_pre_informed (before T_xi, from the first
      delivery of head-originated information on, i.e. without the zero
      commands of COAST relayed before it arrives),
      u_abs_latched, unom_max, unom_max_pre, upk;
      counts / time_spent per unit for min_branch, fallback, saturation,
      sat_upper, sat_lower, positive_nominal ('cont': every RK4 stage and
      step-instant evaluation is counted, times are RK4-weighted stage
      indicators times the step; 'zoh': every update, hold durations), and
      the totals n_mod, n_fallback, n_sat, n_sat_hi, n_sat_lo, n_pos_nom,
      t_mod, t_fallback, t_sat, t_pos_nom, filter_modifications;
      tact, tset (= t_sw), tcl (closure onsets t^cl), trule, rule_entered,
      e_sw, eps_sw (switch states), e_hnd, eps_hnd (errors at T_xi);
      t_rmp (ramp receipts: head T_xi; unit i the first delivery of a
      packet sent by i-1 at or after its own receipt), wint (int |u_{i-1} -
      uhat_{i-1}| on [T_xi, min(tau_{i-1}, tau_i))), epsmax_b, fb_real;
      entry_obs (persistent entry into the handoff box before T_xi: all
      units; _vel_only, _pos_only, _followers), box_last_out;
      age (sim_cont.age_audit plus max_left_limit_with_notional);
      hits_true, n_eval, n_left_limits, n_localized, neg_speed,
      root_resid, t_end_sim, s, v (final), schedule_jumps_abs, config;
      rec: t, s, v, u, e, eps (schedule-relative; head pair in row 0),
      g, h, m (decimated every rec_dt)."""
    n = int(setup.n)
    if scheme not in ('cont', 'zoh'):
        raise ValueError("scheme 'cont' or 'zoh'")
    sch = SCH_ZOH if scheme == 'zoh' else SCH_CONT
    if T_u is None:
        T_u = T_m if scheme == 'zoh' else h
    if h is None:
        h = 1e-4 if scheme == 'cont' else T_u
    if T_u is None:
        T_u = h
    if T_c is None:
        T_c = setup.T_c if getattr(setup, 'T_c', None) else T_m
    if ref_mode is None:
        ref_mode = setup.ref_mode
    if ref_mode not in ('frozen', 'stage'):
        raise ValueError("ref_mode 'frozen' or 'stage'")
    if T_end is None:
        T_end = setup.T_END
    Nm = _on_grid(T_m, h, 'T_m', allow_zero=False)
    Nc = _on_grid(T_c, h, 'T_c', allow_zero=False)
    Nu = _on_grid(T_u, h, 'T_u', allow_zero=False)
    if sch == SCH_CONT and Nu != 1:
        raise ValueError("scheme 'cont' evaluates the law at every step "
                         "(T_u = h); use 'zoh' for T_u > h")
    Ntr = None
    if fifo is None:
        Ntr = _on_grid(transport, h, 'transport', allow_zero=True)
    N = int(round(T_end / h))
    k_xi = _on_grid(setup.T_xi, h, 'T_xi', allow_zero=False)
    kt0 = _on_grid(setup.t_0, h, 't_0', allow_zero=True)
    rec_every = max(1, int(round(rec_dt / h)))
    nrec = N // rec_every + 1
    Nsend = (N - 1) // Nm + 1
    fifo_used = None
    if fifo is not None:
        fifo_used = _fit(fifo, Nsend)
    D, Nsend = sc.deliveries(n, N, Nm, Ntr=Ntr, fifo=fifo_used)
    k_rmp = sc.ramp_receipts(D, Nm, k_xi)
    kfb = k_rmp[:n - 1].copy()

    # ---- head reference pieces
    NH = len(setup.head)
    HT = np.zeros(NH)
    HK = np.full(NH, -1, np.int64)
    HA = np.zeros(NH)
    HS = np.zeros(NH)
    HV = np.zeros(NH)
    head_grid = []
    for j, (t0, a, s0, v0) in enumerate(setup.head):
        HA[j] = a
        HS[j] = s0
        HV[j] = v0
        HT[j] = t0
        if ref_mode == 'frozen' and t0 <= setup.T_xi + 1e-12:
            kj = _on_grid(t0, h, 'head jump %d' % j, allow_zero=True)
            HK[j] = kj
            HT[j] = float(kj) * h
            head_grid.append(kj)
    if ref_mode == 'frozen' and HK[0] != 0:
        raise ValueError('the head reference must start at t = 0')
    # ---- follower schedules
    MS = 1
    if schedules:
        if setup.sched is None:
            raise ValueError('this setup has no schedules; use schedules=False')
        MS = max(len(setup.sched[i]['sigma_f']) for i in range(1, n))
    NS = np.zeros(n, np.int64)
    SK = np.zeros((n, MS), np.int64)
    SR = np.zeros((n, MS))
    SD = np.zeros((n, MS))
    SV = np.zeros((n, MS))
    DPRE = np.array(setup.d_pre, float)
    DPRE[0] = 0.0
    sched_on_h = True
    for i in range(1, n):
        if schedules:
            S = setup.sched[i]
            m = len(S['sigma_f'])
            NS[i] = m
            for j in range(m):
                SK[i, j] = _on_grid(S['sigma_f'][j], h,
                                    'schedule jump %d of unit %d' % (j, i))
                SR[i, j] = S['rho_f'][j]
                SD[i, j] = S['d_f'][j]
                SV[i, j] = S['dd_f'][j]
        elif nosched == 'step':
            NS[i] = 1
            SD[i, 0] = setup.d_s[i]
        elif nosched == 'hold':
            NS[i] = 0
        elif nosched == 'park':
            # FB law (2026-09-29): the desired clearance is the parking gap
            # from takeover on (the laws of the frozen paper)
            NS[i] = 0
            DPRE[i] = setup.d_s[i]
        else:
            raise ValueError("nosched 'step', 'hold' or 'park'")

    F = np.zeros(F_NF)
    F[F_H] = h
    F[F_TXI] = float(k_xi) * h if ref_mode == 'frozen' else setup.T_xi
    F[F_AB] = setup.a_b
    F[F_ALPHA] = setup.alpha
    F[F_BETA] = setup.beta
    F[F_GAMMA] = setup.gamma
    F[F_PHI] = setup.phi
    F[F_EPSDET] = setup.eps_det
    F[F_ELL] = setup.ell
    F[F_SM] = setup.s_m
    F[F_KAPPA] = setup.kappa
    F[F_U] = setup.U
    F[F_EPSE] = setup.eps_e if eps_e is None else eps_e
    F[F_EPSV] = setup.eps_v if eps_v is None else eps_v
    F[F_TAUR] = setup.tau_r
    I = np.zeros(I_NI, np.int64)
    I[I_N] = n
    I[I_NSTEP] = N
    I[I_NM] = Nm
    I[I_NC] = Nc
    I[I_NU] = Nu
    I[I_SCH] = sch
    I[I_REC] = rec_every
    I[I_KXI] = k_xi
    I[I_FILT] = 1 if filter else 0
    I[I_NREC] = nrec
    I[I_EARLY] = 1 if early_exit else 0
    I[I_NSEND] = Nsend
    I[I_KT0] = kt0
    I[I_REFM] = 1 if ref_mode == 'stage' else 0
    I[I_NH] = NH

    d_s = np.array(setup.d_s, float)
    s = np.array(setup.s0, float).copy()
    v = np.array(setup.v0, float).copy()
    s_init = s.copy()
    b = np.array(setup.b, float)
    Umin = np.array(setup.Uminus, float)
    nan = np.nan
    tstop = np.full(n, nan)
    tact = np.full(n, nan)
    tact[0] = 0.0
    tset = np.full(n, nan)
    trule = np.full(n, nan)
    tcl = np.full(n, nan)
    esw = np.full(n, nan)
    xsw = np.full(n, nan)
    gmin = np.full(n - 1, 1e9)
    hmin = np.full(n - 1, 1e9)
    gmin_grid = np.full(n - 1, 1e9)
    mmin_eval = np.full(n - 1, np.inf)
    mmin_lim = np.full(n - 1, np.inf)
    upk = np.zeros(n)
    cnt = np.zeros((6, n), np.int64)
    tim = np.zeros((6, n))
    umn = np.full(n, np.inf)
    umx = np.full(n, -np.inf)
    umn_pre = np.full(n, np.inf)
    umx_pre = np.full(n, -np.inf)
    ulat = np.zeros(n)
    unmx = np.full(n, -np.inf)
    unmx_pre = np.full(n, -np.inf)
    umx_inf = np.full(n, -np.inf)
    wint = np.zeros(n - 1)
    epsmax_b = np.zeros(n - 1)
    fb_real = np.zeros(n - 1)
    e_hnd = np.full(n, nan)
    eps_hnd = np.full(n, nan)
    iout = np.zeros(8, np.int64)
    fout = np.zeros(16)
    rt = np.full(nrec, nan)
    rs = np.full((n, nrec), nan)
    rv = np.full((n, nrec), nan)
    re = np.full((n, nrec), nan)
    reps = np.full((n, nrec), nan)
    ru = np.full((n, nrec), nan)
    rm = np.full((n - 1, nrec), nan)
    rg = np.full((n - 1, nrec), nan)
    rh = np.full((n - 1, nrec), nan)

    _core_hr(F, I, d_s, s, v, b, Umin, D, kfb,
             HT, HK, HA, HS, HV, NS, SK, SR, SD, SV, DPRE,
             tstop, tact, tset, trule, tcl, esw, xsw,
             gmin, hmin, gmin_grid, mmin_eval, mmin_lim, upk, cnt, tim,
             umn, umx, umn_pre, umx_pre, ulat, unmx, unmx_pre, umx_inf,
             wint, epsmax_b, fb_real, e_hnd, eps_hnd, iout, fout,
             rt, rs, rv, re, reps, ru, rm, rg, rh)

    irec = int(iout[7])
    # frozen continuation of the record after the early exit
    if irec < nrec and irec > 0:
        last = irec - 1
        rt[irec:] = np.arange(irec, nrec) * rec_every * h
        for arr in (rs, rv, re, reps, ru, rg, rh):
            arr[:, irec:] = arr[:, last:last + 1]
        ru[:, irec:] = 0.0
        rm[:, irec:] = nan
        rs[:, irec:] = np.where(np.isfinite(tstop), s, rs[:, last])[:, None]
        rv[:, irec:] = np.where(np.isfinite(tstop), v, rv[:, last])[:, None]
    marks = np.array(setup.markers, float)
    egap = np.empty(n)
    egap[0] = setup.s_ref_stop - s[0] - setup.ell - d_s[0]
    egap[1:] = (s[:-1] - s[1:] - setup.ell) - d_s[1:]
    contact_time = None if not np.isfinite(fout[0]) else float(fout[0])
    all_stopped = bool(np.all(np.isfinite(tstop)) and contact_time is None)

    def entry(q):
        """Observed persistent entry into the handoff box before T_xi (the
        last exit -> entry crossing, linear interpolation of the box excess
        between steps; 0 if never outside; NaN if outside at the last step
        before T_xi), as sim_cont."""
        lo, cr = fout[2 + q], fout[8 + q]
        if lo < 0:
            return 0.0
        return float(cr) if cr > lo else float('nan')

    def fin(x):
        y = np.array(x, float)
        y[~np.isfinite(y)] = nan
        return y

    # schedule jump instants in absolute time (on the h grid by construction)
    kcl = np.where(np.isfinite(tcl), np.round(tcl / h), -1).astype(np.int64)
    jumps_abs = {}
    for i in range(1, n):
        if kcl[i] >= 0 and NS[i] > 0:
            jumps_abs[i] = ((kcl[i] + SK[i, :NS[i]]) * h).tolist()
    # age audit: held age (sim_cont.age_audit) plus the notional packet
    # before the first delivery (send time in [-T_m, 0): left limit at the
    # first delivery below first_delivery + T_m)
    age = sc.age_audit(D, Nm, h)
    first = max(o['first_delivery'] for o in age['per_hop'])
    age['max_left_limit_with_notional'] = max(age['max_left_limit'],
                                              first + Nm * h)
    order = [int(x) + 1 for x in np.argsort(tstop)] if all_stopped else None
    upre = fin(umx_pre)
    cnt_d = {nm: cnt[q].tolist() for q, nm in enumerate(CNT_NAMES)}
    tim_d = {nm: tim[q].tolist() for q, nm in enumerate(CNT_NAMES)}
    out = dict(
        tstop=tstop,
        disp=float(np.nanmax(tstop) - np.nanmin(tstop))
        if np.any(np.isfinite(tstop)) else nan,
        T_f=float(np.nanmax(tstop)) if np.any(np.isfinite(tstop)) else nan,
        stop_order=order, all_stopped=all_stopped,
        contact_time=contact_time,
        marker_signed=s - marks, align=float(np.max(np.abs(s - marks))),
        egap=egap, gap_signed=egap[1:].copy(),
        gaperr=float(np.max(np.abs(egap[1:]))), head_slot_err=float(egap[0]),
        gmin=gmin, gmin_grid=gmin_grid, hmin=hmin,
        mmin=mmin_eval, mmin_lim=mmin_lim, upk=upk,
        u_min=fin(umn), u_max=fin(umx), u_min_pre=fin(umn_pre),
        u_max_pre=upre, u_max_pre_all=float(np.nanmax(upre)),
        u_max_pre_informed=fin(umx_inf),
        u_max_pre_informed_all=float(np.nanmax(fin(umx_inf))),
        u_abs_latched=ulat, unom_max=fin(unmx), unom_max_pre=fin(unmx_pre),
        counts=cnt_d, time_spent=tim_d,
        n_pos_nom=int(cnt[5].sum()), t_pos_nom=float(tim[5].sum()),
        n_mod=int(cnt[0].sum()), n_fallback=int(cnt[1].sum()),
        n_sat=int(cnt[2].sum()), n_sat_hi=int(cnt[3].sum()),
        n_sat_lo=int(cnt[4].sum()),
        n_mod_unit=cnt[0].tolist(), n_fallback_unit=cnt[1].tolist(),
        n_sat_unit=cnt[2].tolist(),
        t_mod=float(tim[0].sum()), t_fallback=float(tim[1].sum()),
        t_sat=float(tim[2].sum()),
        filter_modifications=int(cnt[0].sum() + cnt[1].sum() + cnt[2].sum()),
        tact=tact, tset=tset, t_sw=tset, tcl=tcl, trule=trule,
        rule_entered=bool(np.any(np.isfinite(trule))),
        e_sw=esw, eps_sw=xsw, e_hnd=e_hnd, eps_hnd=eps_hnd,
        t_rmp=np.where(k_rmp < sc.RMP_NEVER, k_rmp * h, np.nan),
        wint=wint, epsmax_b=epsmax_b, fb_real=fb_real,
        entry_obs=entry(0), entry_obs_vel_only=entry(1),
        entry_obs_pos_only=entry(2), entry_obs_followers=entry(3),
        box_last_out=[float(fout[2 + q]) if fout[2 + q] >= 0 else None
                      for q in range(4)],
        hits_true=int(iout[1]), n_eval=int(iout[5]),
        n_left_limits=int(iout[4]), n_localized=int(iout[2]),
        neg_speed=int(iout[3]), root_resid=float(fout[6]),
        t_end_sim=float(iout[6] * h), s=s.copy(), v=v.copy(), s_init=s_init,
        marks=marks, s_ref_stop=float(setup.s_ref_stop), tau_r=setup.tau_r,
        T_xi=float(k_xi * h), t_0=float(kt0 * h),
        schedule_jumps_abs=jumps_abs,
        jumps_on_h_grid=dict(head=True, schedules=True,
                             closure_onsets=bool(np.allclose(
                                 tcl[np.isfinite(tcl)] / h,
                                 np.round(tcl[np.isfinite(tcl)] / h),
                                 rtol=0, atol=1e-6))),
        age=age,
        config=dict(scheme=scheme, h=h, T_m=T_m, T_c=T_c, T_u=T_u,
                    transport=None if fifo is not None else transport,
                    fifo=fifo is not None, schedules=bool(schedules),
                    nosched=None if schedules else nosched,
                    filter=bool(filter), ref_mode=ref_mode, U=setup.U,
                    Uminus=np.array(setup.Uminus).tolist(), T_end=T_end,
                    eps_e=F[F_EPSE], eps_v=F[F_EPSV], numba=HAVE_NUMBA,
                    kind=setup.kind),
        rec=dict(t=rt, s=rs, v=rv, u=ru, e=re, eps=reps, g=rg, h=rh, m=rm))
    return out
