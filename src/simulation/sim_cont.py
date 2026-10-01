#!/usr/bin/env python3
"""sim_cont.py -- canonical simulator of the closed loop covered by
Theorem 1 (second-round audit R2-B02 / R2-B03, 2026-09-25).

The theorem's system has continuous local feedback between discrete
communication and detector events.  sim_root2.py (the 1 kHz
event-level simulator) holds every command for one 1 ms step, which
is a digital approximation of that system.  This module separates the
time scales that sim_root2 ties together:

  T_m        message period: each unit sends its applied command u*_i
             and its settling flag at t = j T_m (common send grid),
  transport  transport delay in seconds (constant), or a FIFO schedule
             of per-send delays in units of T_m (hop i uses row i);
             FIFO projection a_j = max_{l<=j}(l + d_l), batch delivery,
  T_c        detector period; the settling detector of unit i samples
             |eps_i| on the local grid t_act_i + m T_c (m >= 0),
             anchored at activation ("activation then detector at once"),
  T_u        local control-update period (digital implementation;
             zero-order hold between updates),
  h          integration step (T_m, T_c, T_u, transport are integer
             multiples of h).

Schemes
  'cont'    the theorem's system: T_u = h and the continuous local law
            (COAST / S1 / S2 / RULE / head BRAKE, filter (eq:qp) in
            closed form with fallback, interlock) is evaluated at every
            Runge-Kutta stage; the closed loop is integrated with
            classical RK4 between events.
  'zoh'     digital implementation: the command is computed at t = k T_u
            and at every event instant of the unit ("one command update
            per event") and held in between; the double integrator is
            integrated exactly (s += v h + u h^2/2, v += u h).
  'legacy'  sim_root2.run semantics, operation for operation, for the
            regression (h = T_u = T_m = T_c): semi-implicit Euler with
            the zero clamp v <- max(0, v + u h), s += v h; Riemann-sum
            reference; RULE threshold v_{i-1} <= v_rule (sim_root2
            default eps_det = 0.015 m/s); LATCH threshold v_i <= 1e-4
            m/s at grid instants; both guarded by t > T_xi; head tracks
            the reference ramp with feedforward -a_b (sim_root2) or
            brakes open loop (head='brake'); commands clipped to
            [-a_max_i, a_max_i].

Event semantics of 'cont' and 'zoh' (main text Sec. III-A):
  * priorities at an instant: LATCH > RULE > packet batch (activation
    flag; head switch to BRAKE at T_xi) > detector sample; the unit then
    updates its command once and sends at its next send time;
  * RULE is entered when a follower in S2 senses v_{i-1} = 0 (exact
    standstill sensing), u_nom = -a_b 1[v_i > 0];
  * LATCH (interlock I_i): at the first zero-speed instant at or after
    the S2 switch, the output becomes 0 forever;
  * zero-speed instants are localized inside the integration step
    (exact root for 'zoh'; Illinois root finding on the RK4 step length
    for 'cont'); a predecessor's zero-speed instant triggers the
    follower's RULE at the same instant;
  * the filter is evaluated wherever the unit is not latched:
    u_qp = sat_[-U_i^-, U](min{u_nom, u_cbf}), fallback -U_i^- when
    u_cbf < -U_i^-; the head applies sat_[-U_1^-, U](u_nom).

Online diagnostics (never fed back): filter modifications (min-branch,
fallback, saturation) per evaluation; the inactivity margin
m_i = u_cbf - u_nom at every evaluation (all RK4 stages; update instants
for 'zoh') and, separately, including the left limits at every event
instant; clearance margins g_i at every accepted state plus the exact
(cubic-Hermite) intersample minimum; barrier h_i; max |u|; stop, S1,
S2, RULE times; handoff errors at T_xi; the braking-window mismatch
integral int |u*_{i-1} - uhat_{i-1}| (RK4 quadrature of the stage
values); the post-receipt envelope max|u*_{i-1} + a_b|; the observed
persistent-entry time into the handoff box (all units, head included,
position and velocity errors) and its velocity-only / position-only /
followers-only variants; signed marker and terminal gap errors;
decimated trajectories.

Post-receipt envelope window (round-4 fix M1, 2026-09-27): fb_real[i-1]
is the largest |u*_{i-1} + a_b| (all RK4 stages for 'cont', update
values on the grid for 'zoh') from the ramp receipt of unit i-1 until
its standstill, the window of Proposition 1(e) and Lemma S(rule)(a) with
F^b_{i-1}.  The ramp receipts are computed from the actual delivery
schedule (ramp_receipts): t_rmp,1 = T_xi, and t_rmp,i is the first
delivery to unit i of a packet that unit i-1 sent at or after t_rmp,i-1
(output t_rmp).  Until 2026-09-26 the window opened at T_xi + i
transport, the receipt of unit i under the constant transport, which
under a FIFO schedule with delays above the nominal transport opened
before the predecessor's receipt.  The 'legacy' scheme keeps the window
of sim_root2.run (from T_xi + i transport) so that the regression
against sim_root2 stays bit-identical.  Only this diagnostic changed;
the closed loop and every other output are unchanged.

Reference-speed profile (Case C, Assumption 1', 2026-09-26): the optional
argument profile = dict(t_d, a_d, v_c1) of run() replaces the constant
approach speed by the piecewise-linear profile
  v_r(t) = v^xi                   on [0, t_d),
           v^xi - a_d (t - t_d)   on [t_d, t_c),  t_c = t_d + (v^xi - v_c1)/a_d,
           v_c1                   on [t_c, T_xi),
           v_c1 - a_b (t - T_xi)  on [T_xi, tau_r),  tau_r = T_xi + v_c1/a_b,
           0                      after tau_r,
s_0 = int v_r, and the markers are s_ref,stop - Delta_i with
s_ref,stop = s_0(T_xi) + v_c1^2/(2 a_b).  The head law adds the reference
acceleration a_r(t) as feedforward in COAST / S1 / S2 before T_xi (-a_d on
the deceleration segment, 0 elsewhere) and brakes open loop at T_xi
(BRAKE); the followers are unchanged (the forwarded command carries the
feedforward).  T_xi remains the handoff instant of every statistic.  With
profile=None (default) every scheme is bit-identical to the version of
2026-09-25; the profile is not available in the 'legacy' scheme.

Planned references (Case D, Assumption 1'' of
data/new_d19/caseD/theory_caseD.md, round 4, 2026-09-27): the optional
argument plan = dict(segs, k, markers, tau_star, s_ref_stop) of run()
(plan_from(certify_plan.Plan, k) builds it; segs[i] lists the pieces
(t0, a*, s*(t0), v*(t0)) of unit i, the last one starting at T_xi with
a* = -a_b and continued beyond the planned stop) runs the Case D laws of
Sec. 2.4 of the theory note in a separate kernel (_core_plan):
  * pair errors relative to the plan, e_1 = s*_1 - s_1, eps_1 = v*_1 - v_1,
    e_i = (s_{i-1} - s_i) - (s*_{i-1} - s*_i), eps_i likewise (i >= 2);
  * local planned feedforward: head a*_1 + f_1 before T_xi and BRAKE -a_b
    from T_xi; followers a*_i + uhat_{i-1} + f_i in COAST / S1 / S2, RULE
    -a_b, LATCH 0 (mode logic, detector, flags and priorities unchanged);
  * the message of unit i carries the deviation u*_i - a*_i(s) formed at
    the send instant s (post-event planned acceleration);
  * filter estimate a*_{i-1}(t) + uhat_{i-1}, filter output
    sat_[-k, 0](min{u_nom, u_cbf}) with fallback -k, head sat_[-k, 0];
  * every plan brakes at -a_b from T_xi (a local clock event), so the
    ramp receipts are T_xi for every unit and the post-receipt envelope
    window of unit i runs from T_xi to its standstill;
  * a planned jump is a local clock event of its unit: the integration
    step is split at every jump instant (jumps on the step grid fall on
    step instants), the planned acceleration is right-continuous and
    frozen over each (sub)step (RK4 stages at a step end see the left
    limit), and in 'zoh' a unit updates its command at its own jumps.
Extra outputs: command range per unit (before LATCH, before T_xi) and
|u| after LATCH, deviations u - a*_i per plan piece before T_xi (positive
and negative parts), |u - a*_i| from T_xi to the standstill (fb_unit),
margins per phase (COAST/S1, S2, braking), barrier deviations h_i - h*_i
per phase and the braking barrier minimum, switch states, braking bands,
speed minima before T_xi, stop order, and the planned states and
deviations on the decimated record.  plan=None (default) leaves every
scheme bit-identical: the default kernel _core and the default code path
of run() are unchanged.

Numba is used when available; otherwise the same code runs as plain
Python (slow, but identical semantics).
"""
import math
import os
import tempfile

import numpy as np

os.environ.setdefault('NUMBA_CACHE_DIR',
                      os.path.join(tempfile.gettempdir(), 'numba_cache_sim_cont'))
try:
    from numba import njit
    HAVE_NUMBA = True
except ImportError:                                  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

# modes
COAST, S1, S2, RULE, LATCH, BRAKE = 0, 1, 2, 3, 4, 5
MODE_NAMES = ('COAST', 'S1', 'S2', 'RULE', 'LATCH', 'BRAKE')
# schemes
SCH_LEGACY, SCH_ZOH, SCH_CONT = 0, 1, 2

# float parameter slots
F_H = 0
F_TXI = 1
F_AB = 2
F_VXI = 3
F_ALPHA = 4
F_BETA = 5
F_GAMMA = 6
F_PHI = 7
F_EPSDET = 8
F_ELL = 9
F_SM = 10
F_KAPPA = 11
F_U = 12
F_VRULE = 13
F_LTHR = 14
F_EPSE = 15
F_EPSV = 16
F_PROF = 17         # 1 = piecewise-linear reference-speed profile (Case C)
F_TD = 18           # profile: deceleration start t_d
F_AD = 19           # profile: deceleration rate a_d > 0
F_VC1 = 20          # profile: crawl speed v_c1
F_NF = 21
# int parameter slots
I_N = 0
I_NSTEP = 1
I_NM = 2
I_NC = 3
I_NU = 4
I_SCH = 5
I_HEAD = 6          # 0 = open-loop BRAKE at T_xi (theorem), 1 = ramp tracking
I_GUARD = 7         # 1 = LATCH/RULE only for t > T_xi (sim_root2)
I_REF = 8           # 0 = exact reference, 1 = Riemann sum (sim_root2)
I_CLIP = 9          # 0 = sat[-U_i^-, U], 1 = clip[-amax_i, amax_i]
I_REC = 10
I_KXI = 11
I_NTR = 12
I_FILT = 13
I_NREC = 14
I_EARLY = 15
I_NSEND = 16
I_NI = 17


# ---------------------------------------------------------------- kernels
@njit(cache=True)
def _ref_profile(t, F):
    """Exact reference (s_0, v_r, a_r) of Assumption 1' (piecewise-linear
    speed profile: v^xi until t_d, deceleration at a_d to the crawl speed
    v_c1, crawl until T_xi, ramp -a_b to standstill) with s_0(0) = 0."""
    vxi = F[F_VXI]
    txi = F[F_TXI]
    ab = F[F_AB]
    td = F[F_TD]
    ad = F[F_AD]
    vc = F[F_VC1]
    tc = td + (vxi - vc) / ad
    if t < td:
        return vxi * t, vxi, 0.0
    s_d = vxi * td
    if t < tc:
        d = t - td
        return s_d + vxi * d - 0.5 * ad * d * d, vxi - ad * d, -ad
    s_c = s_d + 0.5 * (vxi + vc) * (tc - td)
    if t < txi:
        return s_c + vc * (t - tc), vc, 0.0
    s_xi = s_c + vc * (txi - tc)
    d = t - txi
    tr = vc / ab
    if d < tr:
        return s_xi + vc * d - 0.5 * ab * d * d, vc - ab * d, -ab
    return s_xi + 0.5 * vc * tr, 0.0, 0.0


@njit(cache=True)
def _ref(t, F):
    """Exact reference (s_0, v_r, a_r) of Assumption 1 with s_0(0) = 0
    (F[F_PROF] = 1: the profile of Assumption 1', see _ref_profile)."""
    if F[F_PROF] == 1.0:
        return _ref_profile(t, F)
    vxi = F[F_VXI]
    txi = F[F_TXI]
    ab = F[F_AB]
    if t < txi:
        return vxi * t, vxi, 0.0
    d = t - txi
    tr = vxi / ab
    if d < tr:
        return vxi * txi + vxi * d - 0.5 * ab * d * d, vxi - ab * d, -ab
    return vxi * txi + 0.5 * vxi * tr, 0.0, 0.0


@njit(cache=True)
def _law(i, s, v, md, uhi, srf, vrf, urf, F, I, d_s, bf, Umin, A):
    """Nominal law, filter and saturation of unit i at one instant.
    Returns (u_applied, u_nom, u_after_filter, u_cbf, code) with code
    bits 1 = filter output below u_nom, 2 = infeasible (fallback),
    4 = saturation active.  The float operation order follows
    sim_root2.run exactly."""
    ell = F[F_ELL]
    if i == 0:
        gap = srf - s[0] - ell
        eps = vrf - v[0]
        ff = urf
    else:
        gap = s[i - 1] - s[i] - ell
        eps = v[i - 1] - v[i]
        ff = uhi
    if md == LATCH:
        return 0.0, 0.0, 0.0, np.nan, 0
    e = gap - d_s[i]
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
        ucbf = bf[i] * (eps + uhi / bf[i - 1] + F[F_KAPPA] * hb)
        if ucbf < -Umin[i] - 1e-12:
            uq = -Umin[i]
            code |= 2
        elif ucbf < un:
            uq = ucbf
        if uq < un - 1e-12:
            code |= 1
    if I[I_CLIP] == 1:
        lo = -A[i]
        hi = A[i]
    else:
        lo = -Umin[i]
        hi = F[F_U]
    ua = uq
    if ua < lo:
        ua = lo
    elif ua > hi:
        ua = hi
    if ua != uq:
        code |= 4
    return ua, un, uq, ucbf, code


@njit(cache=True)
def _headff(t, arf, F, I):
    """Head feedforward: the reference acceleration for the ramp-tracking
    head (head='track'); for the braking head (theorem) 0, except that the
    profile of Assumption 1' is fed forward before T_xi (a_r = -a_d on the
    deceleration segment, 0 elsewhere; after T_xi the head is in BRAKE)."""
    if I[I_HEAD] == 1:
        return arf
    if F[F_PROF] == 1.0 and t < F[F_TXI]:
        return arf
    return 0.0


@njit(cache=True)
def _stage(r, t, s, v, mode, uh, F, I, d_s, bf, Umin, A, ws):
    """Evaluate all units' laws at (t, s, v); row r of ws receives the
    applied commands, row 4+r the margins, row 8+r the codes."""
    srf, vrf, arf = _ref(t, F)
    urf = _headff(t, arf, F, I)
    n = s.shape[0]
    for i in range(n):
        ua, un, uq, ucbf, code = _law(i, s, v, mode[i], uh[i], srf, vrf, urf,
                                      F, I, d_s, bf, Umin, A)
        ws[r, i] = ua
        if i >= 1 and mode[i] != LATCH:
            ws[4 + r, i] = ucbf - un
        else:
            ws[4 + r, i] = np.nan
        ws[8 + r, i] = code


@njit(cache=True)
def _rk4(t, tau, s, v, mode, uh, F, I, d_s, bf, Umin, A, ws, s_out, v_out,
         st, vt):
    """One classical RK4 step of length tau of the closed loop with the
    discrete state (modes, received commands) frozen.  Stage velocities
    are kept in ws rows 12..15."""
    n = s.shape[0]
    half = 0.5 * tau
    _stage(0, t, s, v, mode, uh, F, I, d_s, bf, Umin, A, ws)
    for i in range(n):
        ws[12, i] = v[i]
        st[i] = s[i] + half * v[i]
        vt[i] = v[i] + half * ws[0, i]
    _stage(1, t + half, st, vt, mode, uh, F, I, d_s, bf, Umin, A, ws)
    for i in range(n):
        ws[13, i] = vt[i]
        st[i] = s[i] + half * ws[13, i]
        vt[i] = v[i] + half * ws[1, i]
    _stage(2, t + half, st, vt, mode, uh, F, I, d_s, bf, Umin, A, ws)
    for i in range(n):
        ws[14, i] = vt[i]
        st[i] = s[i] + tau * ws[14, i]
        vt[i] = v[i] + tau * ws[2, i]
    _stage(3, t + tau, st, vt, mode, uh, F, I, d_s, bf, Umin, A, ws)
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
def _root_rk4(j, t, rem, fa, fb, s, v, mode, uh, F, I, d_s, bf, Umin, A,
              ws2, s_try, v_try, st, vt):
    """Zero of tau -> v_j(RK4(t, tau)) on (0, rem] (Illinois method);
    fa = v_j(t) > 0 >= fb = v_j(RK4(t, rem))."""
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
        _rk4(t, c, s, v, mode, uh, F, I, d_s, bf, Umin, A, ws2, s_try, v_try,
             st, vt)
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
    g0, g1 and slopes d0, d1 (exact for quadratics, i.e. held commands)."""
    best = g0 if g0 < g1 else g1
    if not (d0 < 0.0 and d1 > 0.0):
        return best
    m0 = tau * d0
    m1 = tau * d1
    a = 2.0 * g0 + m0 - 2.0 * g1 + m1
    b = -3.0 * g0 - 2.0 * m0 + 3.0 * g1 - m1
    c = m0
    # p'(x) = 3a x^2 + 2b x + c on (0, 1)
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
def _core(F, I, d_s, s, v, A, b, bf, Umin, deliv,
          tstop, tact, tset, trule, gmin, hmin, gmin_grid, mmin_eval,
          mmin_lim, upk, cnt_mod, cnt_fb, cnt_sat, act_time, wint, epsmax_b,
          fb_real, e_hnd, eps_hnd, iout, fout,
          rt, rv, re, reps, ru, rm, rg, rh, kfb):
    n = I[I_N]
    N = I[I_NSTEP]
    h = F[F_H]
    Nm = I[I_NM]
    Nc = I[I_NC]
    Nu = I[I_NU]
    sch = I[I_SCH]
    headm = I[I_HEAD]
    guard = I[I_GUARD] == 1
    refm = I[I_REF]
    k_xi = I[I_KXI]
    Ntr = I[I_NTR]
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    Nsend = I[I_NSEND]
    T_xi = F[F_TXI]
    a_b = F[F_AB]
    v_xi = F[F_VXI]
    kappa = F[F_KAPPA]
    s_m = F[F_SM]
    ell = F[F_ELL]
    eps_e = F[F_EPSE]
    eps_v = F[F_EPSV]
    v_rule = F[F_VRULE]
    lthr = F[F_LTHR]
    eps_det = F[F_EPSDET]
    v_end = F[F_VC1] if F[F_PROF] == 1.0 else v_xi   # speed at T_xi
    tau_r = T_xi + v_end / a_b

    mode = np.zeros(n, np.int64)
    mode[0] = S1
    kact = np.full(n, -1, np.int64)
    kact[0] = 0
    uh = np.zeros(n)
    flag = np.zeros(n, np.bool_)
    ptr = np.zeros(n, np.int64)
    ucmd = np.zeros(n)
    uprev = np.zeros(n)
    mlast = np.full(n, np.nan)
    usent = np.zeros((n, Nsend))
    fsent = np.zeros((n, Nsend), np.bool_)
    ws = np.zeros((16, n))
    ws2 = np.zeros((16, n))
    st = np.zeros(n)
    vt = np.zeros(n)
    s_new = np.zeros(n)
    v_new = np.zeros(n)
    s_try = np.zeros(n)
    v_try = np.zeros(n)
    evs = np.zeros(n, np.bool_)
    sref = 0.0
    hits_leg = 0
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
        if refm == 1:
            vref = v_xi if t < T_xi else max(0.0, v_xi - a_b * (t - T_xi))
            uref = 0.0 if t < T_xi else (-a_b if vref > 1e-12 else 0.0)
            srf = sref
        else:
            srf, vref, arf = _ref(t, F)
            uref = _headff(t, arf, F, I)
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
        if t < T_xi:
            # handoff-box excess per channel: 0 all units (e and eps),
            # 1 velocity errors only, 2 position errors only, 3 followers
            for q in range(4):
                rbox[q] = -np.inf
            for i in range(n):
                if i == 0:
                    ei = srf - s[0] - ell - d_s[0]
                    epi = vref - v[0]
                else:
                    ei = s[i - 1] - s[i] - ell - d_s[i]
                    epi = v[i - 1] - v[i]
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
                    # last exit -> entry: linear interpolation of the excess
                    crbox[q] = (t - h) + h * rprev[q] / (rprev[q] - rbox[q])
                rprev[q] = rbox[q]
        if k == k_xi:
            for i in range(n):
                if i == 0:
                    gap = srf - s[0] - ell
                    e_hnd[0] = gap - d_s[0]
                    eps_hnd[0] = vref - v[0]
                else:
                    gap = s[i - 1] - s[i] - ell
                    e_hnd[i] = gap - d_s[i]
                    eps_hnd[i] = v[i - 1] - v[i]
        # ------------------------------------------ events at t_k
        for i in range(n):
            md0 = mode[i]
            lat = False
            if md0 == S2 or md0 == RULE or md0 == BRAKE:
                if sch == SCH_LEGACY:
                    lat = v[i] <= lthr and ((not guard) or t > T_xi)
                else:
                    lat = v[i] <= 0.0 and ((not guard) or t > T_xi)
            rul = False
            if (not lat) and md0 == S2 and ((not guard) or t > T_xi):
                if i == 0:
                    rul = headm == 1 and vref <= 1e-12
                else:
                    rul = v[i - 1] <= v_rule
            brk = (i == 0 and headm == 0 and k == k_xi and md0 != LATCH
                   and not lat)
            pkt = i >= 1 and ptr[i] < Nsend and deliv[i, ptr[i]] <= k
            det_due = (md0 == S1 and (k - kact[i]) % Nc == 0) or \
                (md0 == COAST and pkt)
            ev = lat or rul or brk or pkt or det_due
            if ev and i >= 1 and md0 != LATCH:
                ua, un, uq, ucbf, code = _law(i, s, v, md0, uh[i], srf, vref,
                                              uref, F, I, d_s, bf, Umin, A)
                mg = ucbf - un
                if mg < mmin_lim[i - 1]:
                    mmin_lim[i - 1] = mg
                n_left += 1
            if lat:
                mode[i] = LATCH
                tstop[i] = t
                if sch != SCH_LEGACY:
                    v[i] = 0.0
            elif rul:
                mode[i] = RULE
                trule[i] = t
            if brk:
                mode[0] = BRAKE
            if i >= 1:
                while ptr[i] < Nsend and deliv[i, ptr[i]] <= k:
                    j = ptr[i]
                    uh[i] = usent[i - 1, j]
                    if fsent[i - 1, j]:
                        flag[i] = True
                    ptr[i] += 1
                if mode[i] == COAST and flag[i]:
                    mode[i] = S1
                    kact[i] = k
                    tact[i] = t
            if mode[i] == S1 and (k - kact[i]) % Nc == 0:
                if i == 0:
                    epi = vref - v[0]
                else:
                    epi = v[i - 1] - v[i]
                if abs(epi) <= eps_det:
                    mode[i] = S2
                    tset[i] = t
            evs[i] = ev
        # ------------------------------------------ command update
        for i in range(n):
            if sch == SCH_ZOH and not (k % Nu == 0 or evs[i]):
                continue
            ua, un, uq, ucbf, code = _law(i, s, v, mode[i], uh[i], srf, vref,
                                          uref, F, I, d_s, bf, Umin, A)
            ucmd[i] = ua
            n_eval += 1
            if abs(ua) > upk[i]:
                upk[i] = abs(ua)
            if i >= 1 and mode[i] != LATCH:
                mg = ucbf - un
                mlast[i] = mg
                if mg < mmin_eval[i - 1]:
                    mmin_eval[i - 1] = mg
                if mg < mmin_lim[i - 1]:
                    mmin_lim[i - 1] = mg
                if code & 1:
                    cnt_mod[i] += 1
                    if sch == SCH_LEGACY:
                        act_time[i] += h
                if code & 2:
                    cnt_fb[i] += 1
            else:
                mlast[i] = np.nan
            if code & 4:
                cnt_sat[i] += 1
        # barrier-condition checks (true current commands; legacy: uprev)
        for i in range(1, n):
            gap = s[i - 1] - s[i] - ell
            g = gap - s_m
            hb = g + v[i - 1] / b[i - 1] - v[i] / b[i]
            eps = v[i - 1] - v[i]
            if sch == SCH_LEGACY:
                hdot = uprev[i - 1] / b[i - 1] - ucmd[i] / b[i] + eps
                if hdot + kappa * hb < -1e-12:
                    hits_leg += 1
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
                    if sch == SCH_LEGACY:
                        wint[i - 1] += abs(ucmd[i - 1] - uh[i]) * h
                if sch != SCH_CONT and k >= kfb[i - 1] and \
                        np.isnan(tstop[i - 1]):
                    fb = abs(ucmd[i - 1] + a_b)
                    if fb > fb_real[i - 1]:
                        fb_real[i - 1] = fb
        # ------------------------------------------ record
        if k % rec_every == 0 and irec < nrec:
            rt[irec] = t
            for i in range(n):
                rv[i, irec] = v[i]
                ru[i, irec] = ucmd[i]
                if i == 0:
                    re[0, irec] = srf - s[0] - ell - d_s[0]
                    reps[0, irec] = vref - v[0]
                else:
                    re[i, irec] = s[i - 1] - s[i] - ell - d_s[i]
                    reps[i, irec] = v[i - 1] - v[i]
                    rg[i - 1, irec] = s[i - 1] - s[i] - ell - s_m
                    rh[i - 1, irec] = rg[i - 1, irec] + v[i - 1] / b[i - 1] \
                        - v[i] / b[i]
                    rm[i - 1, irec] = mlast[i]
            irec += 1
        # ------------------------------------------ sends
        if k % Nm == 0:
            j = k // Nm
            if j < Nsend:
                for i in range(n):
                    usent[i, j] = ucmd[i]
                    fsent[i, j] = mode[i] >= S2
        # ------------------------------------------ integrate to t_{k+1}
        if sch == SCH_LEGACY:
            for i in range(n):
                if mode[i] != LATCH:
                    v[i] = max(0.0, v[i] + ucmd[i] * h)
                    s[i] += v[i] * h
            sref += vref * h
            for i in range(n):
                uprev[i] = ucmd[i]
        else:
            tcur = t
            t_end = (k + 1) * h
            while True:
                rem = t_end - tcur
                if rem <= 0.0:
                    break
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
                    if k >= k_xi:
                        for i in range(1, n):
                            if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                                wint[i - 1] += tau * abs(ucmd[i - 1] - uh[i])
                else:
                    _rk4(tcur, rem, s, v, mode, uh, F, I, d_s, bf, Umin, A,
                         ws, s_new, v_new, st, vt)
                    for i in range(n):
                        mdi = mode[i]
                        if v[i] > 0.0 and v_new[i] <= 0.0:
                            if mdi == S2 or mdi == RULE or mdi == BRAKE:
                                tz, rs = _root_rk4(i, tcur, rem, v[i], v_new[i],
                                                   s, v, mode, uh, F, I, d_s,
                                                   bf, Umin, A, ws2, s_try,
                                                   v_try, st, vt)
                                if rs > max_resid:
                                    max_resid = rs
                                if tz < tau:
                                    tau = tz
                                    bi = i
                            elif mdi != LATCH:
                                neg_speed += 1
                    if bi >= 0:
                        _rk4(tcur, tau, s, v, mode, uh, F, I, d_s, bf, Umin,
                             A, ws, s_new, v_new, st, vt)
                    # stage statistics of the accepted (sub)step
                    for r in range(4):
                        for i in range(n):
                            ua = ws[r, i]
                            if abs(ua) > upk[i]:
                                upk[i] = abs(ua)
                            code = int(ws[8 + r, i])
                            n_eval += 1
                            if i >= 1 and mode[i] != LATCH:
                                mg = ws[4 + r, i]
                                if mg < mmin_eval[i - 1]:
                                    mmin_eval[i - 1] = mg
                                if mg < mmin_lim[i - 1]:
                                    mmin_lim[i - 1] = mg
                                if code & 1:
                                    cnt_mod[i] += 1
                                if code & 2:
                                    cnt_fb[i] += 1
                            if code & 4:
                                cnt_sat[i] += 1
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
                ev_units = (bi, bi + 1)
                for jj in range(2):
                    q = ev_units[jj]
                    if q >= 1 and q < n and mode[q] != LATCH:
                        ua, un, uq, ucbf, code = _law(q, s, v, mode[q], uh[q],
                                                      0.0, 0.0, 0.0, F, I,
                                                      d_s, bf, Umin, A)
                        mg = ucbf - un
                        if mg < mmin_lim[q - 1]:
                            mmin_lim[q - 1] = mg
                        n_left += 1
                v[bi] = 0.0
                mode[bi] = LATCH
                tstop[bi] = tcur
                ucmd[bi] = 0.0
                if bi + 1 < n and mode[bi + 1] == S2 and v_rule <= 0.0:
                    mode[bi + 1] = RULE
                    trule[bi + 1] = tcur
                    # event-triggered command update (right limit)
                    ua, un, uq, ucbf, code = _law(bi + 1, s, v, mode[bi + 1],
                                                  uh[bi + 1], 0.0, 0.0, 0.0,
                                                  F, I, d_s, bf, Umin, A)
                    if sch == SCH_ZOH:
                        ucmd[bi + 1] = ua
                        n_eval += 1
                        if abs(ua) > upk[bi + 1]:
                            upk[bi + 1] = abs(ua)
                        if code & 1:
                            cnt_mod[bi + 1] += 1
                        if code & 2:
                            cnt_fb[bi + 1] += 1
                        if code & 4:
                            cnt_sat[bi + 1] += 1
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
    iout[0] = hits_leg
    iout[1] = hits_true
    iout[2] = n_loc
    iout[3] = neg_speed
    iout[4] = n_left
    iout[5] = n_eval
    iout[6] = k_end
    iout[7] = irec
    fout[0] = contact_time
    fout[1] = sref
    for q in range(4):
        fout[2 + q] = lobox[q]
        fout[8 + q] = crbox[q]
    fout[6] = max_resid


# ------------------------------------ Case D: planned references (plan=)
@njit(cache=True)
def _pstates(t, pidx, PT, PA, PS, PV, ps, pv, pa):
    """Planned states (s*_i, v*_i, a*_i) of every unit at t on the pieces
    pidx.  The pieces are frozen over an integration (sub)step, so the
    value at the right end of a step is the left limit there."""
    for i in range(pidx.shape[0]):
        j = pidx[i]
        tau = t - PT[i, j]
        a = PA[i, j]
        ps[i] = PS[i, j] + PV[i, j] * tau + 0.5 * a * tau * tau
        pv[i] = PV[i, j] + a * tau
        pa[i] = a


@njit(cache=True)
def _law_plan(i, s, v, md, uhi, ps, pv, pa, F, I, bf, Umin):
    """Case D law of unit i at one instant (theory_caseD.md Sec. 2.4):
    pair errors relative to the plan, local planned feedforward a*_i(t),
    received deviation uhi (followers), filter estimate
    a*_{i-1}(t) + uhi, input set [-Umin_i, F_U] (Case D: [-k, 0]) with
    fallback -Umin_i.  Returns (u_applied, u_nom, u_after_filter, u_cbf,
    code) with the code bits of _law."""
    if md == LATCH:
        return 0.0, 0.0, 0.0, np.nan, 0
    if i == 0:
        e = ps[0] - s[0]
        eps = pv[0] - v[0]
        ff = pa[0]
    else:
        e = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
        eps = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
        ff = pa[i] + uhi
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
        g = s[i - 1] - s[i] - F[F_ELL] - F[F_SM]
        hb = g + v[i - 1] / bf[i - 1] - v[i] / bf[i]
        ucbf = bf[i] * ((v[i - 1] - v[i]) + (pa[i - 1] + uhi) / bf[i - 1]
                        + F[F_KAPPA] * hb)
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
    elif ua > hi:
        ua = hi
    if ua != uq:
        code |= 4
    return ua, un, uq, ucbf, code


@njit(cache=True)
def _stage_plan(r, t, s, v, mode, uh, pidx, PT, PA, PS, PV, F, I, bf, Umin,
                ws, qs, qv, qa):
    """_stage for the Case D law (planned states on the frozen pieces)."""
    _pstates(t, pidx, PT, PA, PS, PV, qs, qv, qa)
    n = s.shape[0]
    for i in range(n):
        ua, un, uq, ucbf, code = _law_plan(i, s, v, mode[i], uh[i], qs, qv,
                                           qa, F, I, bf, Umin)
        ws[r, i] = ua
        if i >= 1 and mode[i] != LATCH:
            ws[4 + r, i] = ucbf - un
        else:
            ws[4 + r, i] = np.nan
        ws[8 + r, i] = code


@njit(cache=True)
def _rk4_plan(t, tau, s, v, mode, uh, pidx, PT, PA, PS, PV, F, I, bf, Umin,
              ws, s_out, v_out, st, vt, qs, qv, qa):
    """_rk4 for the Case D law; the plan pieces stay frozen on the step
    (steps are split at the planned jumps)."""
    n = s.shape[0]
    half = 0.5 * tau
    _stage_plan(0, t, s, v, mode, uh, pidx, PT, PA, PS, PV, F, I, bf, Umin,
                ws, qs, qv, qa)
    for i in range(n):
        ws[12, i] = v[i]
        st[i] = s[i] + half * v[i]
        vt[i] = v[i] + half * ws[0, i]
    _stage_plan(1, t + half, st, vt, mode, uh, pidx, PT, PA, PS, PV, F, I,
                bf, Umin, ws, qs, qv, qa)
    for i in range(n):
        ws[13, i] = vt[i]
        st[i] = s[i] + half * ws[13, i]
        vt[i] = v[i] + half * ws[1, i]
    _stage_plan(2, t + half, st, vt, mode, uh, pidx, PT, PA, PS, PV, F, I,
                bf, Umin, ws, qs, qv, qa)
    for i in range(n):
        ws[14, i] = vt[i]
        st[i] = s[i] + tau * ws[14, i]
        vt[i] = v[i] + tau * ws[2, i]
    _stage_plan(3, t + tau, st, vt, mode, uh, pidx, PT, PA, PS, PV, F, I,
                bf, Umin, ws, qs, qv, qa)
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
def _root_rk4_plan(j, t, rem, fa, fb, s, v, mode, uh, pidx, PT, PA, PS, PV,
                   F, I, bf, Umin, ws2, s_try, v_try, st, vt, qs, qv, qa):
    """_root_rk4 for the Case D law."""
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
        _rk4_plan(t, c, s, v, mode, uh, pidx, PT, PA, PS, PV, F, I, bf, Umin,
                  ws2, s_try, v_try, st, vt, qs, qv, qa)
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
def _phase(md, pre):
    """Margin phase: 0 = COAST/S1 before T_xi, 1 = S2 before T_xi,
    2 = from T_xi on."""
    if not pre:
        return 2
    if md == COAST or md == S1:
        return 0
    return 1


@njit(cache=True)
def _ustats(i, ua, md, pre, j, a, umn, umx, umn_pre, umx_pre, ulat, dpos,
            dneg, fbu):
    """Command statistics of unit i: range before LATCH (all, before
    T_xi), |u| after LATCH, deviation u - a*_i per plan piece j before
    T_xi (positive and negative parts), |u - a*_i| from T_xi on."""
    if md == LATCH:
        x = abs(ua)
        if x > ulat[i]:
            ulat[i] = x
        return
    if ua < umn[i]:
        umn[i] = ua
    if ua > umx[i]:
        umx[i] = ua
    d = ua - a
    if pre:
        if ua < umn_pre[i]:
            umn_pre[i] = ua
        if ua > umx_pre[i]:
            umx_pre[i] = ua
        if d > dpos[i, j]:
            dpos[i, j] = d
        if -d > dneg[i, j]:
            dneg[i, j] = -d
    else:
        x = abs(d)
        if x > fbu[i]:
            fbu[i] = x


@njit(cache=True)
def _core_plan(F, I, FP, d_s, s, v, A, b, bf, Umin, deliv, kfb,
               NPc, PT, PA, PS, PV, PJ,
               tstop, tact, tset, trule, gmin, hmin, gmin_grid, mmin_eval,
               mmin_lim, upk, cnt_mod, cnt_fb, cnt_sat, wint, epsmax_b,
               fb_real, e_hnd, eps_hnd, iout, fout,
               rt, rv, re, reps, ru, rm, rg, rh,
               umn, umx, umn_pre, umx_pre, ulat, dpos, dneg, fbu, mph,
               htmin, hbrk, ebmax, esw, xsw, vmin_pre):
    """Closed loop of Case D ('cont' and 'zoh'): _core with the planned
    references of Assumption 1'' (theory_caseD.md Sec. 2.4).  Pair errors
    relative to the plan; the deviation u*_i - a*_i(s) is formed at the
    send instant s; the filter estimate is a*_{i-1}(t) + received
    deviation; every plan brakes at -a_b from T_xi (head BRAKE, followers
    stay in S2 until RULE or LATCH).  A planned jump is a local clock
    event of its unit: integration steps are split at every jump instant
    (a jump on the step grid is processed at the step instant), the
    planned acceleration is right-continuous, and in 'zoh' the unit
    updates its command at its own jumps."""
    n = I[I_N]
    N = I[I_NSTEP]
    h = F[F_H]
    Nm = I[I_NM]
    Nc = I[I_NC]
    Nu = I[I_NU]
    sch = I[I_SCH]
    k_xi = I[I_KXI]
    rec_every = I[I_REC]
    nrec = I[I_NREC]
    Nsend = I[I_NSEND]
    T_xi = F[F_TXI]
    kappa = F[F_KAPPA]
    s_m = F[F_SM]
    ell = F[F_ELL]
    eps_e = F[F_EPSE]
    eps_v = F[F_EPSV]
    eps_det = F[F_EPSDET]
    tau_r = FP[0]
    NJ = PJ.shape[0]

    mode = np.zeros(n, np.int64)
    mode[0] = S1
    kact = np.full(n, -1, np.int64)
    kact[0] = 0
    uh = np.zeros(n)
    flag = np.zeros(n, np.bool_)
    ptr = np.zeros(n, np.int64)
    ucmd = np.zeros(n)
    mlast = np.full(n, np.nan)
    usent = np.zeros((n, Nsend))
    fsent = np.zeros((n, Nsend), np.bool_)
    ws = np.zeros((16, n))
    ws2 = np.zeros((16, n))
    st = np.zeros(n)
    vt = np.zeros(n)
    s_new = np.zeros(n)
    v_new = np.zeros(n)
    s_try = np.zeros(n)
    v_try = np.zeros(n)
    evs = np.zeros(n, np.bool_)
    jev = np.zeros(n, np.bool_)
    pidx = np.zeros(n, np.int64)
    ps = np.zeros(n)
    pv = np.zeros(n)
    pa = np.zeros(n)
    pal = np.zeros(n)
    qs = np.zeros(n)
    qv = np.zeros(n)
    qa = np.zeros(n)
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
    jp = 0
    k = 0
    while k < N:
        t = k * h
        pre = k < k_xi
        # ------------------------------------------ plan pieces at t_k
        for i in range(n):
            pal[i] = PA[i, pidx[i]]
            jev[i] = False
            while pidx[i] + 1 < NPc[i] and PT[i, pidx[i] + 1] <= t:
                pidx[i] += 1
                jev[i] = True
        _pstates(t, pidx, PT, PA, PS, PV, ps, pv, pa)
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
            if mode[i] != LATCH:
                if pre:
                    hs = (ps[i - 1] - ps[i] - ell - s_m) + pv[i - 1] / b[i - 1] \
                        - pv[i] / b[i]
                    q = _phase(mode[i], True)
                    if hb - hs < htmin[q, i - 1]:
                        htmin[q, i - 1] = hb - hs
                elif hb < hbrk[i - 1]:
                    hbrk[i - 1] = hb
        if not np.isnan(contact_time):
            k_end = k
            break
        if pre:
            # handoff-box excess per channel (errors relative to the plan):
            # 0 all units, 1 velocity errors only, 2 position errors only,
            # 3 followers
            for q in range(4):
                rbox[q] = -np.inf
            for i in range(n):
                if v[i] < vmin_pre[i]:
                    vmin_pre[i] = v[i]
                if i == 0:
                    ei = ps[0] - s[0]
                    epi = pv[0] - v[0]
                else:
                    ei = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
                    epi = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
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
                    e_hnd[0] = ps[0] - s[0]
                    eps_hnd[0] = pv[0] - v[0]
                else:
                    e_hnd[i] = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
                    eps_hnd[i] = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
        # ------------------------------------------ events at t_k
        for i in range(n):
            md0 = mode[i]
            lat = False
            if md0 == S2 or md0 == RULE or md0 == BRAKE:
                lat = v[i] <= 0.0
            rul = False
            if (not lat) and md0 == S2 and i >= 1:
                rul = v[i - 1] <= 0.0
            brk = i == 0 and k == k_xi and md0 != LATCH and not lat
            pkt = i >= 1 and ptr[i] < Nsend and deliv[i, ptr[i]] <= k
            det_due = (md0 == S1 and (k - kact[i]) % Nc == 0) or \
                (md0 == COAST and pkt)
            ev = lat or rul or brk or pkt or det_due or jev[i]
            lev = ev or (i >= 1 and jev[i - 1])
            if lev and i >= 1 and md0 != LATCH:
                # left limit: previous mode, received deviation and planned
                # accelerations
                ua, un, uq, ucbf, code = _law_plan(i, s, v, md0, uh[i], ps, pv,
                                                   pal, F, I, bf, Umin)
                mg = ucbf - un
                if mg < mmin_lim[i - 1]:
                    mmin_lim[i - 1] = mg
                q = _phase(md0, k <= k_xi)
                if mg < mph[q, i - 1]:
                    mph[q, i - 1] = mg
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
            if i >= 1:
                while ptr[i] < Nsend and deliv[i, ptr[i]] <= k:
                    j = ptr[i]
                    uh[i] = usent[i - 1, j]
                    if fsent[i - 1, j]:
                        flag[i] = True
                    ptr[i] += 1
                if mode[i] == COAST and flag[i]:
                    mode[i] = S1
                    kact[i] = k
                    tact[i] = t
            if mode[i] == S1 and (k - kact[i]) % Nc == 0:
                if i == 0:
                    ei = ps[0] - s[0]
                    epi = pv[0] - v[0]
                else:
                    ei = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
                    epi = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
                if abs(epi) <= eps_det:
                    mode[i] = S2
                    tset[i] = t
                    esw[i] = ei
                    xsw[i] = epi
            evs[i] = ev
        # ------------------------------------------ command update
        for i in range(n):
            if sch == SCH_ZOH and not (k % Nu == 0 or evs[i]):
                continue
            ua, un, uq, ucbf, code = _law_plan(i, s, v, mode[i], uh[i], ps, pv,
                                               pa, F, I, bf, Umin)
            ucmd[i] = ua
            n_eval += 1
            if abs(ua) > upk[i]:
                upk[i] = abs(ua)
            if sch == SCH_ZOH:
                _ustats(i, ua, mode[i], pre, pidx[i], pa[i], umn, umx, umn_pre,
                        umx_pre, ulat, dpos, dneg, fbu)
            if i >= 1 and mode[i] != LATCH:
                mg = ucbf - un
                mlast[i] = mg
                if mg < mmin_eval[i - 1]:
                    mmin_eval[i - 1] = mg
                if mg < mmin_lim[i - 1]:
                    mmin_lim[i - 1] = mg
                q = _phase(mode[i], pre)
                if mg < mph[q, i - 1]:
                    mph[q, i - 1] = mg
                if code & 1:
                    cnt_mod[i] += 1
                if code & 2:
                    cnt_fb[i] += 1
            else:
                mlast[i] = np.nan
            if code & 4:
                cnt_sat[i] += 1
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
                    epr = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
                    er = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
                    if abs(epr) > epsmax_b[i - 1]:
                        epsmax_b[i - 1] = abs(epr)
                    if abs(er) > ebmax[i - 1]:
                        ebmax[i - 1] = abs(er)
                if sch != SCH_CONT and k >= kfb[i - 1] and \
                        np.isnan(tstop[i - 1]):
                    fb = abs(ucmd[i - 1] - pa[i - 1])
                    if fb > fb_real[i - 1]:
                        fb_real[i - 1] = fb
        # ------------------------------------------ record
        if k % rec_every == 0 and irec < nrec:
            rt[irec] = t
            for i in range(n):
                rv[i, irec] = v[i]
                ru[i, irec] = ucmd[i]
                if i == 0:
                    re[0, irec] = ps[0] - s[0]
                    reps[0, irec] = pv[0] - v[0]
                else:
                    re[i, irec] = (s[i - 1] - s[i]) - (ps[i - 1] - ps[i])
                    reps[i, irec] = (v[i - 1] - v[i]) - (pv[i - 1] - pv[i])
                    rg[i - 1, irec] = s[i - 1] - s[i] - ell - s_m
                    rh[i - 1, irec] = rg[i - 1, irec] + v[i - 1] / b[i - 1] \
                        - v[i] / b[i]
                    rm[i - 1, irec] = mlast[i]
            irec += 1
        # ------------------------------------------ sends: deviation formed
        # at the send instant (post-event planned acceleration)
        if k % Nm == 0:
            j = k // Nm
            if j < Nsend:
                for i in range(n):
                    usent[i, j] = ucmd[i] - pa[i]
                    fsent[i, j] = mode[i] >= S2
        # ------------------------------------------ integrate to t_{k+1}
        tcur = t
        t_end = (k + 1) * h
        while True:
            rem = t_end - tcur
            if rem <= 0.0:
                break
            while jp < NJ and PJ[jp] <= tcur:
                jp += 1
            jflag = jp < NJ and PJ[jp] < t_end
            span = (PJ[jp] - tcur) if jflag else rem
            pre_s = tcur < T_xi
            bi = -1
            tau = span
            if sch == SCH_ZOH:
                for i in range(n):
                    mdi = mode[i]
                    if v[i] > 0.0 and ucmd[i] < 0.0:
                        tz = -v[i] / ucmd[i]
                        if mdi == S2 or mdi == RULE or mdi == BRAKE:
                            if tz <= tau:
                                tau = tz
                                bi = i
                        elif mdi != LATCH and tz < span:
                            neg_speed += 1
                for i in range(n):
                    if mode[i] != LATCH:
                        s_new[i] = s[i] + v[i] * tau + 0.5 * ucmd[i] * tau * tau
                        v_new[i] = v[i] + ucmd[i] * tau
                    else:
                        s_new[i] = s[i]
                        v_new[i] = v[i]
                if k >= k_xi:
                    for i in range(1, n):
                        if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                            wint[i - 1] += tau * abs(ucmd[i - 1] - pa[i - 1]
                                                     - uh[i])
            else:
                _rk4_plan(tcur, span, s, v, mode, uh, pidx, PT, PA, PS, PV, F,
                          I, bf, Umin, ws, s_new, v_new, st, vt, qs, qv, qa)
                for i in range(n):
                    mdi = mode[i]
                    if v[i] > 0.0 and v_new[i] <= 0.0:
                        if mdi == S2 or mdi == RULE or mdi == BRAKE:
                            tz, rs = _root_rk4_plan(i, tcur, span, v[i],
                                                    v_new[i], s, v, mode, uh,
                                                    pidx, PT, PA, PS, PV, F, I,
                                                    bf, Umin, ws2, s_try, v_try,
                                                    st, vt, qs, qv, qa)
                            if rs > max_resid:
                                max_resid = rs
                            if tz < tau:
                                tau = tz
                                bi = i
                        elif mdi != LATCH:
                            neg_speed += 1
                if bi >= 0:
                    _rk4_plan(tcur, tau, s, v, mode, uh, pidx, PT, PA, PS, PV,
                              F, I, bf, Umin, ws, s_new, v_new, st, vt, qs, qv,
                              qa)
                # stage statistics of the accepted (sub)step
                for r in range(4):
                    for i in range(n):
                        ua = ws[r, i]
                        if abs(ua) > upk[i]:
                            upk[i] = abs(ua)
                        code = int(ws[8 + r, i])
                        n_eval += 1
                        _ustats(i, ua, mode[i], pre_s, pidx[i], pa[i], umn, umx,
                                umn_pre, umx_pre, ulat, dpos, dneg, fbu)
                        if i >= 1 and mode[i] != LATCH:
                            mg = ws[4 + r, i]
                            if mg < mmin_eval[i - 1]:
                                mmin_eval[i - 1] = mg
                            if mg < mmin_lim[i - 1]:
                                mmin_lim[i - 1] = mg
                            q = _phase(mode[i], pre_s)
                            if mg < mph[q, i - 1]:
                                mph[q, i - 1] = mg
                            if code & 1:
                                cnt_mod[i] += 1
                            if code & 2:
                                cnt_fb[i] += 1
                        if code & 4:
                            cnt_sat[i] += 1
                        if i >= 1 and k >= kfb[i - 1] and \
                                np.isnan(tstop[i - 1]):
                            fb = abs(ws[r, i - 1] - pa[i - 1])
                            if fb > fb_real[i - 1]:
                                fb_real[i - 1] = fb
                if k >= k_xi:
                    for i in range(1, n):
                        if np.isnan(tstop[i - 1]) and np.isnan(tstop[i]):
                            w1 = abs(ws[0, i - 1] - pa[i - 1] - uh[i])
                            w2 = abs(ws[1, i - 1] - pa[i - 1] - uh[i])
                            w3 = abs(ws[2, i - 1] - pa[i - 1] - uh[i])
                            w4 = abs(ws[3, i - 1] - pa[i - 1] - uh[i])
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
                if jflag:
                    # ---- planned jump(s) strictly inside the step
                    tcur = PJ[jp]
                    for i in range(n):
                        pal[i] = pa[i]
                        jev[i] = False
                        while pidx[i] + 1 < NPc[i] and \
                                PT[i, pidx[i] + 1] <= tcur:
                            pidx[i] += 1
                            jev[i] = True
                    _pstates(tcur, pidx, PT, PA, PS, PV, ps, pv, pa)
                    pre_j = tcur < T_xi
                    for q in range(n):
                        if q >= 1 and mode[q] != LATCH and (jev[q] or jev[q - 1]):
                            ua, un, uq, ucbf, code = _law_plan(
                                q, s, v, mode[q], uh[q], ps, pv, pal, F, I, bf,
                                Umin)
                            mg = ucbf - un
                            if mg < mmin_lim[q - 1]:
                                mmin_lim[q - 1] = mg
                            ph = _phase(mode[q], True)
                            if mg < mph[ph, q - 1]:
                                mph[ph, q - 1] = mg
                            n_left += 1
                        if jev[q] and sch == SCH_ZOH:
                            # event-triggered update at the unit's own jump
                            ua, un, uq, ucbf, code = _law_plan(
                                q, s, v, mode[q], uh[q], ps, pv, pa, F, I, bf,
                                Umin)
                            ucmd[q] = ua
                            n_eval += 1
                            if abs(ua) > upk[q]:
                                upk[q] = abs(ua)
                            _ustats(q, ua, mode[q], pre_j, pidx[q], pa[q], umn,
                                    umx, umn_pre, umx_pre, ulat, dpos, dneg,
                                    fbu)
                            if q >= 1 and mode[q] != LATCH:
                                mg = ucbf - un
                                mlast[q] = mg
                                if mg < mmin_eval[q - 1]:
                                    mmin_eval[q - 1] = mg
                                if mg < mmin_lim[q - 1]:
                                    mmin_lim[q - 1] = mg
                                ph = _phase(mode[q], pre_j)
                                if mg < mph[ph, q - 1]:
                                    mph[ph, q - 1] = mg
                                if code & 1:
                                    cnt_mod[q] += 1
                                if code & 2:
                                    cnt_fb[q] += 1
                            if code & 4:
                                cnt_sat[q] += 1
                    continue
                break
            # ---- localized zero-speed event of unit bi at tcur + tau
            tcur = tcur + tau
            n_loc += 1
            _pstates(tcur, pidx, PT, PA, PS, PV, ps, pv, pa)
            pre_z = tcur < T_xi
            ev_units = (bi, bi + 1)
            for jj in range(2):
                q = ev_units[jj]
                if q >= 1 and q < n and mode[q] != LATCH:
                    ua, un, uq, ucbf, code = _law_plan(q, s, v, mode[q], uh[q],
                                                       ps, pv, pa, F, I, bf,
                                                       Umin)
                    mg = ucbf - un
                    if mg < mmin_lim[q - 1]:
                        mmin_lim[q - 1] = mg
                    ph = _phase(mode[q], pre_z)
                    if mg < mph[ph, q - 1]:
                        mph[ph, q - 1] = mg
                    n_left += 1
            v[bi] = 0.0
            mode[bi] = LATCH
            tstop[bi] = tcur
            ucmd[bi] = 0.0
            if bi + 1 < n and mode[bi + 1] == S2:
                mode[bi + 1] = RULE
                trule[bi + 1] = tcur
                # event-triggered command update (right limit)
                ua, un, uq, ucbf, code = _law_plan(bi + 1, s, v, mode[bi + 1],
                                                   uh[bi + 1], ps, pv, pa, F, I,
                                                   bf, Umin)
                if sch == SCH_ZOH:
                    ucmd[bi + 1] = ua
                    n_eval += 1
                    if abs(ua) > upk[bi + 1]:
                        upk[bi + 1] = abs(ua)
                    _ustats(bi + 1, ua, mode[bi + 1], pre_z, pidx[bi + 1],
                            pa[bi + 1], umn, umx, umn_pre, umx_pre, ulat, dpos,
                            dneg, fbu)
                    if code & 1:
                        cnt_mod[bi + 1] += 1
                    if code & 2:
                        cnt_fb[bi + 1] += 1
                    if code & 4:
                        cnt_sat[bi + 1] += 1
                mg = ucbf - un
                if mg < mmin_eval[bi]:
                    mmin_eval[bi] = mg
                if mg < mmin_lim[bi]:
                    mmin_lim[bi] = mg
                ph = _phase(RULE, pre_z)
                if mg < mph[ph, bi]:
                    mph[ph, bi] = mg
            if not np.isnan(contact_time):
                break
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
    fout[1] = 0.0
    for q in range(4):
        fout[2 + q] = lobox[q]
        fout[8 + q] = crbox[q]
    fout[6] = max_resid


# ---------------------------------------------------------------- wrapper
def _steps(x, h, name):
    q = x / h
    k = int(round(q))
    if abs(q - k) > 1e-6 or k < 1:
        raise ValueError('%s = %g is not a positive integer multiple of h = %g'
                         % (name, x, h))
    return k


def deliveries(n, N, Nm, Ntr=None, fifo=None):
    """Delivery step (in units of h) of every send j of unit i-1 to unit i
    (row i, i >= 1).  Constant transport: j Nm + Ntr.  FIFO schedule
    fifo[i, j] (transport of send j in units of T_m): FIFO projection
    a_j = max_{l <= j} (l + d_l), delivered at a_j Nm."""
    Nsend = (N - 1) // Nm + 1
    D = np.zeros((n, Nsend), np.int64)
    j = np.arange(Nsend, dtype=np.int64)
    for i in range(1, n):
        if fifo is None:
            D[i] = j * Nm + Ntr
        else:
            d = np.asarray(fifo[i][:Nsend], np.int64)
            if np.any(d < 1):
                raise ValueError('FIFO transport must be >= 1 message period')
            D[i] = np.maximum.accumulate(j + d) * Nm
    return D, Nsend


RMP_NEVER = np.iinfo(np.int64).max // 4      # receipt beyond the horizon


def ramp_receipts(D, Nm, k_xi):
    """Ramp receipts t_rmp,i of main-text (B1) as step indices (units of
    h) for the delivery table D of deliveries(): row 0 (the head) k_xi;
    row i >= 1 the first delivery to unit i of a packet that unit i-1
    sent at or after the ramp receipt of unit i-1 (sends at the steps
    j Nm; a packet sent at the receipt step already carries the command
    updated with the received ramp).  RMP_NEVER marks a receipt beyond
    the horizon."""
    n, Nsend = D.shape
    K = np.empty(n, np.int64)
    K[0] = k_xi
    for i in range(1, n):
        if K[i - 1] >= RMP_NEVER:
            K[i] = RMP_NEVER
            continue
        j0 = -(-int(K[i - 1]) // Nm)          # first send at or after
        K[i] = int(D[i, j0:].min()) if j0 < Nsend else RMP_NEVER
    return K


def age_audit(D, Nm, h):
    """Held information age: right after a delivery the age equals the
    packet's transport delay; just before the next delivery (left limit)
    it equals next delivery time minus the send time of the held packet."""
    out = []
    for i in range(1, D.shape[0]):
        a = D[i]
        send = np.arange(len(a)) * Nm
        # distinct delivery instants; the held packet is the last one of
        # each batch (FIFO projection)
        last = np.r_[a[1:] != a[:-1], True]
        idx = np.nonzero(last)[0]
        left = (a[idx[1:]] - send[idx[:-1]]) * h
        at_update = (a[idx] - send[idx]) * h
        out.append(dict(max_left_limit=float(left.max()) if len(left) else None,
                        max_at_delivery=float(at_update.max()),
                        first_delivery=float(a[0] * h)))
    return dict(max_left_limit=max(o['max_left_limit'] for o in out),
                max_at_delivery=max(o['max_at_delivery'] for o in out),
                per_hop=out)


def _profile(P, profile, legacy):
    """Validated copy of the reference-speed profile (None passes)."""
    if profile is None:
        return None
    if legacy:
        raise ValueError("the reference-speed profile is not available in "
                         "the 'legacy' scheme")
    prof = dict(t_d=float(profile['t_d']), a_d=float(profile['a_d']),
                v_c1=float(profile['v_c1']))
    if not (prof['a_d'] > 0.0 and 0.0 < prof['v_c1'] <= P['v_xi']
            and prof['t_d'] >= 0.0):
        raise ValueError('profile needs a_d > 0, 0 < v_c1 <= v_xi, t_d >= 0')
    prof['t_c'] = prof['t_d'] + (P['v_xi'] - prof['v_c1']) / prof['a_d']
    if prof['t_c'] > P['T_xi']:
        raise ValueError('the deceleration segment must end at or before '
                         'T_xi (t_c = %g > T_xi = %g)' % (prof['t_c'], P['T_xi']))
    return prof


def ref_profile(t, P, profile=None):
    """Reference (s_0, v_r, a_r) at the instants t (scalar or array) for
    the constants P and the optional profile, from the same kernel as
    run(); returns an array of shape (3, len(t))."""
    prof = _profile(P, profile, False)
    F = np.zeros(F_NF)
    F[F_TXI] = P['T_xi']
    F[F_AB] = P['a_b']
    F[F_VXI] = P['v_xi']
    if prof is not None:
        F[F_PROF] = 1.0
        F[F_TD] = prof['t_d']
        F[F_AD] = prof['a_d']
        F[F_VC1] = prof['v_c1']
    tt = np.atleast_1d(np.asarray(t, float)).ravel()
    out = np.empty((3, tt.size))
    for j in range(tt.size):
        out[:, j] = _ref(float(tt[j]), F)
    return out


def plan_from(Pobj, k):
    """Plan dictionary for run(plan=...) from a certify_plan.Plan object
    (exact rational pieces, converted to double here) and the input bound
    k (commands in [-k, 0])."""
    return dict(segs=[[(float(t0), float(a), float(s0), float(v0))
                       for (t0, t1, a, s0, v0) in Pobj.segs[i]]
                      for i in range(Pobj.n)],
                markers=[float(x) for x in Pobj.markers],
                tau_star=float(Pobj.tau_star),
                s_ref_stop=float(Pobj.s_ref_stop), k=float(k))


def _plan_arrays(plan, n, h, T_xi, a_b, T_m):
    """Piece tables of the plan: NPc[i] pieces of unit i with start times
    PT (jump instants within 1e-6 steps of the h grid are placed on the
    double k h, which is how the kernel forms its step instants), rates
    PA and planned states PS, PV at the piece starts; PJ the sorted jump
    instants of all units."""
    segs = plan['segs']
    if len(segs) != n:
        raise ValueError('plan: one piece list per unit required')
    MP = max(len(sg) for sg in segs)
    NPc = np.zeros(n, np.int64)
    PT = np.full((n, MP), np.inf)
    PA = np.zeros((n, MP))
    PS = np.zeros((n, MP))
    PV = np.zeros((n, MP))
    on_h = True
    on_tm = True
    k_xi = int(round(T_xi / h))
    for i, sg in enumerate(segs):
        NPc[i] = len(sg)
        for j, (t0, a, s0, v0) in enumerate(sg):
            t0 = float(t0)
            q = t0 / h
            kj = int(round(q))
            if abs(q - kj) <= 1e-6:
                t0 = float(kj) * h
            elif j > 0:
                on_h = False
            qm = float(sg[j][0]) / T_m
            if j > 0 and abs(qm - round(qm)) > 1e-6:
                on_tm = False
            PT[i, j] = t0
            PA[i, j] = float(a)
            PS[i, j] = float(s0)
            PV[i, j] = float(v0)
        m = int(NPc[i])
        if PT[i, 0] != 0.0:
            raise ValueError('plan: the first piece must start at t = 0')
        if np.any(np.diff(PT[i, :m]) <= 0.0):
            raise ValueError('plan: piece start times must increase')
        if PT[i, m - 1] != float(k_xi) * h or PA[i, m - 1] != -a_b:
            raise ValueError('plan: every plan brakes at -a_b from T_xi, and '
                             'T_xi must be an integer multiple of h')
    PJ = np.unique(np.concatenate([PT[i, 1:NPc[i]] for i in range(n)]))
    return NPc, PT, PA, PS, PV, PJ, on_h, on_tm


def plan_states(PT, PA, PS, PV, NPc, t):
    """Planned (s*, v*, a*) of every unit at the instants t (right-
    continuous planned acceleration); arrays of shape (n, len(t))."""
    t = np.atleast_1d(np.asarray(t, float))
    n = len(NPc)
    S = np.empty((n, t.size))
    V = np.empty((n, t.size))
    Aa = np.empty((n, t.size))
    for i in range(n):
        m = int(NPc[i])
        j = np.clip(np.searchsorted(PT[i, :m], t, side='right') - 1, 0, m - 1)
        tau = t - PT[i, j]
        a = PA[i, j]
        S[i] = PS[i, j] + PV[i, j] * tau + 0.5 * a * tau * tau
        V[i] = PV[i, j] + a * tau
        Aa[i] = a
    return S, V, Aa


def _run_plan(P, d_s, v0, e0, plan, h, T_m, T_c, T_u, transport, fifo,
              scheme, apply_filter, eps_e, eps_v, rec_dt, early_exit, T_END):
    """run() with the planned references of Case D (see run)."""
    n = int(P['n'])
    if scheme not in ('cont', 'zoh'):
        raise ValueError("plan: schemes 'cont' and 'zoh' only")
    if h is None:
        h = 2.5e-5
    if T_c is None:
        T_c = T_m
    if T_u is None:
        T_u = T_m if scheme == 'zoh' else h
    if transport is None:
        transport = P['dbar']
    if T_END is None:
        T_END = P['T_END']
    sch = SCH_ZOH if scheme == 'zoh' else SCH_CONT
    Nm = _steps(T_m, h, 'T_m')
    Nc = _steps(T_c, h, 'T_c')
    Nu = _steps(T_u, h, 'T_u')
    Ntr = _steps(transport, h, 'transport')
    if sch == SCH_CONT and Nu != 1:
        raise ValueError("scheme 'cont' evaluates the law at every step "
                         "(T_u = h); use 'zoh' for T_u > h")
    N = int(round(T_END / h))
    k_xi = int(round(P['T_xi'] / h))
    if abs(k_xi * h - P['T_xi']) > 1e-9:
        raise ValueError('plan: T_xi must be an integer multiple of h')
    rec_every = max(1, int(round(rec_dt / h)))
    nrec = N // rec_every + 1
    a_b = float(P['a_b'])
    kbound = float(plan['k'])
    NPc, PT, PA, PS, PV, PJ, on_h, on_tm = _plan_arrays(
        plan, n, h, P['T_xi'], a_b, T_m)
    MP = PT.shape[1]
    T_xi_g = float(k_xi) * h

    d_s = np.array(d_s, float)
    e0 = np.array(e0, float)
    s = -np.cumsum(P['ell'] + d_s + e0)
    v = np.array(v0, float).copy()
    s_init = s.copy()
    A = np.array(P['amax'], float)
    b = A / P['v_c']
    bf = b.copy()
    Umin = np.full(n, kbound)
    D, Nsend = deliveries(n, N, Nm, Ntr=Ntr, fifo=fifo)
    # every unit starts its ramp at T_xi on the common time base (premise
    # (I2)): the ramp receipts are T_xi, and the post-receipt envelope
    # window of unit i runs from T_xi to its standstill
    k_rmp = np.full(n, k_xi, np.int64)
    kfb = k_rmp[:n - 1].copy()
    # planned stop, reference stop and markers
    v_c1 = float(PV[0, NPc[0] - 1])
    tau_star = float(plan.get('tau_star', T_xi_g + v_c1 / a_b))
    if plan.get('s_ref_stop') is not None:
        s_ref_stop = float(plan['s_ref_stop'])
    else:
        s_ref_stop = float(PS[0, NPc[0] - 1] + v_c1 ** 2 / (2 * a_b)
                           + d_s[0] + P['ell'])
    if plan.get('markers') is not None:
        marks = np.array(plan['markers'], float)
    else:
        marks = s_ref_stop - np.cumsum(P['ell'] + d_s)

    F = np.zeros(F_NF)
    F[F_H] = h
    F[F_TXI] = T_xi_g
    F[F_AB] = a_b
    F[F_VXI] = float(PV[0, 0])
    F[F_ALPHA] = P['alpha']
    F[F_BETA] = P['beta']
    F[F_GAMMA] = P['gamma']
    F[F_PHI] = P['phi']
    F[F_EPSDET] = P['eps_det']
    F[F_ELL] = P['ell']
    F[F_SM] = P['s_m']
    F[F_KAPPA] = P['kappa']
    F[F_U] = 0.0                                    # input set [-k, 0]
    F[F_VRULE] = 0.0
    F[F_LTHR] = 1e-4
    F[F_EPSE] = eps_e
    F[F_EPSV] = eps_v
    I = np.zeros(I_NI, np.int64)
    I[I_N] = n
    I[I_NSTEP] = N
    I[I_NM] = Nm
    I[I_NC] = Nc
    I[I_NU] = Nu
    I[I_SCH] = sch
    I[I_REC] = rec_every
    I[I_KXI] = k_xi
    I[I_NTR] = Ntr
    I[I_FILT] = 1 if apply_filter else 0
    I[I_NREC] = nrec
    I[I_EARLY] = 1 if early_exit else 0
    I[I_NSEND] = Nsend
    FP = np.array([tau_star, s_ref_stop, kbound])

    nan = np.nan
    tstop = np.full(n, nan)
    tact = np.full(n, nan)
    tact[0] = 0.0
    tset = np.full(n, nan)
    trule = np.full(n, nan)
    gmin = np.full(n - 1, 1e9)
    hmin = np.full(n - 1, 1e9)
    gmin_grid = np.full(n - 1, 1e9)
    mmin_eval = np.full(n - 1, np.inf)
    mmin_lim = np.full(n - 1, np.inf)
    upk = np.zeros(n)
    cnt_mod = np.zeros(n, np.int64)
    cnt_fb = np.zeros(n, np.int64)
    cnt_sat = np.zeros(n, np.int64)
    wint = np.zeros(n - 1)
    epsmax_b = np.zeros(n - 1)
    fb_real = np.zeros(n - 1)
    e_hnd = np.full(n, nan)
    eps_hnd = np.full(n, nan)
    iout = np.zeros(8, np.int64)
    fout = np.zeros(16)
    rt = np.full(nrec, nan)
    rv = np.full((n, nrec), nan)
    re = np.full((n, nrec), nan)
    reps = np.full((n, nrec), nan)
    ru = np.full((n, nrec), nan)
    rm = np.full((n - 1, nrec), nan)
    rg = np.full((n - 1, nrec), nan)
    rh = np.full((n - 1, nrec), nan)
    umn = np.full(n, np.inf)
    umx = np.full(n, -np.inf)
    umn_pre = np.full(n, np.inf)
    umx_pre = np.full(n, -np.inf)
    ulat = np.zeros(n)
    dpos = np.full((n, MP), -np.inf)
    dneg = np.full((n, MP), -np.inf)
    fbu = np.zeros(n)
    mph = np.full((3, n - 1), np.inf)
    htmin = np.full((2, n - 1), np.inf)
    hbrk = np.full(n - 1, np.inf)
    ebmax = np.zeros(n - 1)
    esw = np.full(n, nan)
    xsw = np.full(n, nan)
    vmin_pre = np.full(n, np.inf)

    _core_plan(F, I, FP, d_s, s, v, A, b, bf, Umin, D, kfb,
               NPc, PT, PA, PS, PV, PJ,
               tstop, tact, tset, trule, gmin, hmin, gmin_grid, mmin_eval,
               mmin_lim, upk, cnt_mod, cnt_fb, cnt_sat, wint, epsmax_b,
               fb_real, e_hnd, eps_hnd, iout, fout,
               rt, rv, re, reps, ru, rm, rg, rh,
               umn, umx, umn_pre, umx_pre, ulat, dpos, dneg, fbu, mph,
               htmin, hbrk, ebmax, esw, xsw, vmin_pre)

    irec = int(iout[7])
    # frozen continuation of the record after the early exit
    if irec < nrec and irec > 0:
        last = irec - 1
        rt[irec:] = np.arange(irec, nrec) * rec_every * h
        for arr in (rv, re, reps, ru, rg, rh):
            arr[:, irec:] = arr[:, last:last + 1]
        ru[:, irec:] = 0.0
        rm[:, irec:] = nan
        rv[:, irec:] = np.where(np.isfinite(tstop), v, rv[:, last])[:, None]
    okr = np.isfinite(rt)
    rps = np.full((n, nrec), nan)
    rpv = np.full((n, nrec), nan)
    rpa = np.full((n, nrec), nan)
    if okr.any():
        rps[:, okr], rpv[:, okr], rpa[:, okr] = plan_states(
            PT, PA, PS, PV, NPc, rt[okr])
    egap = np.empty(n)
    egap[0] = s_ref_stop - s[0] - P['ell'] - d_s[0]
    egap[1:] = (s[:-1] - s[1:] - P['ell']) - d_s[1:]
    contact_time = None if not np.isfinite(fout[0]) else float(fout[0])
    all_stopped = bool(np.all(np.isfinite(tstop)) and contact_time is None)

    def entry(q):
        lo, cr = fout[2 + q], fout[8 + q]
        if lo < 0:
            return 0.0
        return float(cr) if cr > lo else float('nan')

    def fin(x, empty):
        y = np.array(x, float)
        y[~np.isfinite(y)] = nan
        return y if empty is None else np.where(np.isfinite(y), y, empty)

    pieces = [[dict(t0=float(PT[i, j]), a=float(PA[i, j]))
               for j in range(int(NPc[i]))] for i in range(n)]
    order = [int(x) + 1 for x in np.argsort(tstop)] if all_stopped else None
    out = dict(
        tstop=tstop,
        disp=float(np.nanmax(tstop) - np.nanmin(tstop))
        if np.any(np.isfinite(tstop)) else nan,
        align=float(np.max(np.abs(s - marks))),
        gaperr=float(np.max(np.abs(egap[1:]))),
        head_slot_err=float(egap[0]),
        egap=egap, hits=int(iout[1]), hits_true=int(iout[1]),
        infeas=int(cnt_fb.sum()),
        n_mod=int(cnt_mod.sum()), n_fallback=int(cnt_fb.sum()),
        n_sat=int(cnt_sat.sum()), n_mod_unit=cnt_mod.tolist(),
        n_fallback_unit=cnt_fb.tolist(), n_sat_unit=cnt_sat.tolist(),
        n_eval=int(iout[5]), n_left_limits=int(iout[4]),
        n_localized=int(iout[2]), neg_speed=int(iout[3]),
        root_resid=float(fout[6]),
        gmin=gmin, gmin_grid=gmin_grid, hmin=hmin, upk=upk, s=s.copy(),
        marks=marks, tact=tact, tset=tset, trule=trule,
        rule_entered=bool(np.any(np.isfinite(trule))),
        contact_time=contact_time, all_stopped=all_stopped,
        stop_order=order,
        mmin=mmin_eval, mmin_lim=mmin_lim, mmin_phase=fin(mph, None),
        wint=wint, epsmax_b=epsmax_b, ebmax_b=ebmax,
        fb_real=fb_real, fb_unit=fbu, e_hnd=e_hnd, eps_hnd=eps_hnd,
        e_sw=esw, eps_sw=xsw,
        htil_min=fin(htmin, None), h_brk_min=fin(hbrk, None),
        vmin_pre=vmin_pre,
        u_min=fin(umn, None), u_max=fin(umx, None),
        u_min_pre=fin(umn_pre, None), u_max_pre=fin(umx_pre, None),
        u_abs_latched=ulat,
        dev_pos=fin(dpos, None), dev_neg=fin(dneg, None),
        t_rmp=np.full(n, T_xi_g),
        fb_window_start=np.full(n - 1, T_xi_g),
        fb_window=[[T_xi_g, float(tstop[i])] for i in range(n)],
        sched=float(np.nanmax(np.abs(tstop - tau_star)))
        if np.any(np.isfinite(tstop)) else nan,
        T_f=float(np.nanmax(tstop)) if np.any(np.isfinite(tstop)) else nan,
        entry_obs=entry(0), entry_obs_vel_only=entry(1),
        entry_obs_pos_only=entry(2), entry_obs_followers=entry(3),
        t_end_sim=float(iout[6] * h), sref_final=s_ref_stop,
        marker_signed=s - marks, tau_r=tau_star, mark_ref=s_ref_stop,
        s_init=s_init,
        plan=dict(T_xi=T_xi_g, tau_star=tau_star, s_ref_stop=s_ref_stop,
                  markers=marks.tolist(), v_c1=v_c1, k=kbound,
                  s0_plan=PS[:, 0].tolist(), v0_plan=PV[:, 0].tolist(),
                  s0_offset=(s_init - PS[:, 0]).tolist(),
                  jumps=PJ.tolist(), jumps_on_h_grid=bool(on_h),
                  jumps_on_Tm_grid=bool(on_tm), pieces=pieces),
        profile=None,
        config=dict(scheme=scheme, h=h, T_m=T_m, T_c=T_c, T_u=T_u,
                    transport=transport, fifo=fifo is not None, head='brake',
                    v_rule=0.0, apply_filter=apply_filter, localized=True,
                    eps_e=eps_e, eps_v=eps_v, T_END=T_END, numba=HAVE_NUMBA,
                    profile=None, plan=True, input_set=[-kbound, 0.0]),
        age=age_audit(D, Nm, h),
        rec=dict(t=rt, v=rv, e=re, eps=reps, u=ru, m=rm, g=rg, h=rh,
                 ps=rps, pv=rpv, pa=rpa, dev=ru - rpa))
    return out


def run(P, d_s, v0, e0, h=None, T_m=1e-3, T_c=None, T_u=None,
        transport=None, fifo=None, scheme='cont', head=None, v_rule=None,
        apply_filter=True, eps_e=0.15, eps_v=0.02, rec_dt=0.01,
        early_exit=True, T_END=None, profile=None, plan=None):
    """One closed-loop run.

    P: simulator constants in the sim_root2 format (n, dt, dbar, a_b,
       v_xi, T_xi, T_END, alpha, beta, gamma, phi, eps_det, ell, s_m,
       kappa, U, amax, v_c); P['dbar'] is the transport delay when
       `transport` is None (run_new_d19 convention).
    scheme: 'cont' (theorem), 'zoh' (digital, T_u), 'legacy' (sim_root2).
    head: 'brake' (theorem, default for cont/zoh) or 'track' (sim_root2,
       default for legacy).
    v_rule: RULE threshold on the sensed predecessor speed; None -> 0
       (exact standstill sensing, the theorem) for cont/zoh and eps_det
       (sim_root2 default) for legacy.
    profile: None (Assumption 1, constant approach speed v_xi) or
       dict(t_d, a_d, v_c1): the piecewise-linear reference-speed profile
       of Assumption 1' (deceleration start, rate, crawl speed; see the
       module docstring); cont/zoh only.
    plan: None (default) or the planned references of Case D (module
       docstring; plan_from() builds it from a certify_plan.Plan); cont/zoh
       only, not combined with profile, head or v_rule.
    Returns a dict with the sim_root2 output fields plus diagnostics and
    the decimated trajectory (every rec_dt)."""
    if plan is not None:
        if profile is not None or scheme == 'legacy' or head not in (
                None, 'brake') or v_rule not in (None, 0.0):
            raise ValueError('plan: cont/zoh only, without profile, with the '
                             'braking head and exact standstill sensing')
        return _run_plan(P, d_s, v0, e0, plan, h, T_m, T_c, T_u, transport,
                         fifo, scheme, apply_filter, eps_e, eps_v, rec_dt,
                         early_exit, T_END)
    n = int(P['n'])
    legacy = scheme == 'legacy'
    prof = _profile(P, profile, legacy)
    if h is None:
        h = P['dt'] if legacy else 2.5e-5
    if T_c is None:
        T_c = T_m
    if T_u is None:
        T_u = T_m if scheme == 'zoh' else h
    if transport is None:
        transport = P['dbar']
    if T_END is None:
        T_END = P['T_END']
    if head is None:
        head = 'track' if legacy else 'brake'
    if v_rule is None:
        v_rule = P['eps_det'] if legacy else 0.0
    sch = dict(legacy=SCH_LEGACY, zoh=SCH_ZOH, cont=SCH_CONT)[scheme]
    Nm = _steps(T_m, h, 'T_m')
    Nc = _steps(T_c, h, 'T_c')
    Nu = _steps(T_u, h, 'T_u')
    Ntr = _steps(transport, h, 'transport')
    if legacy and not (Nm == Nc == Nu == 1):
        raise ValueError('legacy scheme requires h = T_m = T_c = T_u')
    if sch == SCH_CONT and Nu != 1:
        raise ValueError("scheme 'cont' evaluates the law at every step "
                         "(T_u = h); use 'zoh' for T_u > h")
    if not legacy and v_rule > 0.0:
        raise ValueError('cont/zoh implement exact standstill sensing '
                         '(v_rule = 0)')
    N = int(T_END / h) if legacy else int(round(T_END / h))
    k_xi = int(round(P['T_xi'] / h))
    rec_every = max(1, int(round(rec_dt / h)))
    nrec = N // rec_every + 1

    d_s = np.array(d_s, float)
    e0 = np.array(e0, float)
    s = -np.cumsum(P['ell'] + d_s + e0)
    v = np.array(v0, float).copy()
    A = np.array(P['amax'], float)
    b = A / P['v_c']
    bf = b.copy()
    Umin = np.minimum(P['U'], A)
    if prof is None:
        mark_ref = P['v_xi'] * P['T_xi'] + P['v_xi'] ** 2 / (2 * P['a_b'])
        marks = mark_ref - np.cumsum(P['ell'] + d_s)
    D, Nsend = deliveries(n, N, Nm, Ntr=Ntr, fifo=fifo)
    # post-receipt envelope window of fb_real[i-1] (diagnostic only):
    # from the ramp receipt of unit i-1 ('cont', 'zoh'); sim_root2's
    # window from k_xi + i Ntr ('legacy', regression)
    k_rmp = ramp_receipts(D, Nm, k_xi)
    if legacy:
        kfb = k_xi + np.arange(1, n, dtype=np.int64) * Ntr
    else:
        kfb = k_rmp[:n - 1].copy()

    F = np.zeros(F_NF)
    F[F_H] = h
    F[F_TXI] = P['T_xi']
    F[F_AB] = P['a_b']
    F[F_VXI] = P['v_xi']
    F[F_ALPHA] = P['alpha']
    F[F_BETA] = P['beta']
    F[F_GAMMA] = P['gamma']
    F[F_PHI] = P['phi']
    F[F_EPSDET] = P['eps_det']
    F[F_ELL] = P['ell']
    F[F_SM] = P['s_m']
    F[F_KAPPA] = P['kappa']
    F[F_U] = P['U']
    F[F_VRULE] = v_rule
    F[F_LTHR] = 1e-4
    F[F_EPSE] = eps_e
    F[F_EPSV] = eps_v
    if prof is not None:
        F[F_PROF] = 1.0
        F[F_TD] = prof['t_d']
        F[F_AD] = prof['a_d']
        F[F_VC1] = prof['v_c1']
        # markers: s_ref,stop = s_0(T_xi) + v_c1^2 / (2 a_b)
        mark_ref = float(_ref(P['T_xi'], F)[0]) \
            + prof['v_c1'] ** 2 / (2 * P['a_b'])
        marks = mark_ref - np.cumsum(P['ell'] + d_s)
        prof['s_ref_stop'] = mark_ref
        prof['tau_r'] = P['T_xi'] + prof['v_c1'] / P['a_b']
    I = np.zeros(I_NI, np.int64)
    I[I_N] = n
    I[I_NSTEP] = N
    I[I_NM] = Nm
    I[I_NC] = Nc
    I[I_NU] = Nu
    I[I_SCH] = sch
    I[I_HEAD] = 0 if head == 'brake' else 1
    I[I_GUARD] = 1 if legacy else 0
    I[I_REF] = 1 if legacy else 0
    I[I_CLIP] = 1 if legacy else 0
    I[I_REC] = rec_every
    I[I_KXI] = k_xi
    I[I_NTR] = Ntr
    I[I_FILT] = 1 if apply_filter else 0
    I[I_NREC] = nrec
    I[I_EARLY] = 1 if early_exit else 0
    I[I_NSEND] = Nsend

    nan = np.nan
    tstop = np.full(n, nan)
    tact = np.full(n, nan)
    tact[0] = 0.0
    tset = np.full(n, nan)
    trule = np.full(n, nan)
    gmin = np.full(n - 1, 1e9)
    hmin = np.full(n - 1, 1e9)
    gmin_grid = np.full(n - 1, 1e9)
    mmin_eval = np.full(n - 1, np.inf)
    mmin_lim = np.full(n - 1, np.inf)
    upk = np.zeros(n)
    cnt_mod = np.zeros(n, np.int64)
    cnt_fb = np.zeros(n, np.int64)
    cnt_sat = np.zeros(n, np.int64)
    act_time = np.zeros(n)
    wint = np.zeros(n - 1)
    epsmax_b = np.zeros(n - 1)
    fb_real = np.zeros(n - 1)
    e_hnd = np.full(n, nan)
    eps_hnd = np.full(n, nan)
    iout = np.zeros(8, np.int64)
    fout = np.zeros(16)
    rt = np.full(nrec, nan)
    rv = np.full((n, nrec), nan)
    re = np.full((n, nrec), nan)
    reps = np.full((n, nrec), nan)
    ru = np.full((n, nrec), nan)
    rm = np.full((n - 1, nrec), nan)
    rg = np.full((n - 1, nrec), nan)
    rh = np.full((n - 1, nrec), nan)

    _core(F, I, d_s, s, v, A, b, bf, Umin, D,
          tstop, tact, tset, trule, gmin, hmin, gmin_grid, mmin_eval,
          mmin_lim, upk, cnt_mod, cnt_fb, cnt_sat, act_time, wint, epsmax_b,
          fb_real, e_hnd, eps_hnd, iout, fout, rt, rv, re, reps, ru, rm, rg,
          rh, kfb)

    irec = int(iout[7])
    # frozen continuation of the record after the early exit
    if irec < nrec and irec > 0:
        last = irec - 1
        rt[irec:] = np.arange(irec, nrec) * rec_every * h
        for arr in (rv, re, reps, ru, rg, rh):
            arr[:, irec:] = arr[:, last:last + 1]
        ru[:, irec:] = 0.0
        rm[:, irec:] = nan
        rv[:, irec:] = np.where(np.isfinite(tstop), v, rv[:, last])[:, None]
    if legacy:
        sref = float(fout[1])
    else:
        sref = float(_ref(T_END, F)[0])
    egap = (np.concatenate(([sref], s[:-1])) - s - P['ell']) - d_s
    contact_time = None if not np.isfinite(fout[0]) else float(fout[0])
    if prof is None:
        tau_r = P['T_xi'] + P['v_xi'] / P['a_b']
    else:
        tau_r = prof['tau_r']

    def entry(q):
        """Observed persistent entry into the handoff box up to T_xi
        (audit R2-E01): the last exit->entry crossing, localized by
        linear interpolation of the box excess between samples; 0 if the
        errors never leave the box; NaN if they are outside at the last
        sample before T_xi."""
        lo, cr = fout[2 + q], fout[8 + q]
        if lo < 0:
            return 0.0
        return float(cr) if cr > lo else float('nan')

    out = dict(
        tstop=tstop,
        disp=float(np.nanmax(tstop) - np.nanmin(tstop)),
        align=float(np.max(np.abs(s - marks))),
        gaperr=float(np.max(np.abs(egap))),
        egap=egap, hits=int(iout[0]) if legacy else int(iout[1]),
        hits_legacy=int(iout[0]), hits_true=int(iout[1]),
        infeas=int(cnt_fb.sum()), act_time=act_time.tolist(),
        n_mod=int(cnt_mod.sum()), n_fallback=int(cnt_fb.sum()),
        n_sat=int(cnt_sat.sum()), n_mod_unit=cnt_mod.tolist(),
        n_eval=int(iout[5]), n_left_limits=int(iout[4]),
        n_localized=int(iout[2]), neg_speed=int(iout[3]),
        root_resid=float(fout[6]),
        gmin=gmin, gmin_grid=gmin_grid, hmin=hmin, upk=upk, s=s.copy(),
        marks=marks, tact=tact, tset=tset, trule=trule,
        contact_time=contact_time,
        all_stopped=bool(np.all(np.isfinite(tstop)) and contact_time is None),
        mmin=mmin_eval, mmin_lim=mmin_lim, wint=wint, epsmax_b=epsmax_b,
        fb_real=fb_real, e_hnd=e_hnd, eps_hnd=eps_hnd,
        t_rmp=np.where(k_rmp < RMP_NEVER, k_rmp * h, np.nan),
        fb_window_start=np.where(kfb < RMP_NEVER, kfb * h, np.nan),
        sched=float(np.nanmax(np.abs(tstop - tau_r))),
        T_f=float(np.nanmax(tstop)),
        entry_obs=entry(0), entry_obs_vel_only=entry(1),
        entry_obs_pos_only=entry(2), entry_obs_followers=entry(3),
        t_end_sim=float(iout[6] * h), sref_final=sref,
        marker_signed=s - marks, tau_r=tau_r, mark_ref=mark_ref,
        profile=None if prof is None else dict(prof),
        config=dict(scheme=scheme, h=h, T_m=T_m, T_c=T_c, T_u=T_u,
                    transport=transport, fifo=fifo is not None, head=head,
                    v_rule=v_rule, apply_filter=apply_filter,
                    localized=not legacy, eps_e=eps_e, eps_v=eps_v,
                    T_END=T_END, numba=HAVE_NUMBA,
                    profile=None if prof is None else dict(prof)),
        age=age_audit(D, Nm, h),
        rec=dict(t=rt, v=rv, e=re, eps=reps, u=ru, m=rm, g=rg, h=rh))
    return out
