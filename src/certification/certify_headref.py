#!/usr/bin/env python3
"""certify_headref.py -- dispatch-certificate evaluator of the arrival with a
reference known only to the head ("HR"), specification
data/new_d19/headref/theory_headref.md (2026-09-28), with the refinements
and completions documented in data/new_d19/headref/evaluator_notes.md.

Information structure of the frozen paper (OT, certify_v30.py) and of Case C
(CT, certify_profile.py): only unit 1 stores the reference (a_r = planned
acceleration of unit 1 of the round-6 fan plan certify_plan.plan(cfg)),
every follower stores a gap-closing schedule rho_i (constants only) that it
starts when it processes the relayed closure flag, and the braking onset is
relayed through the commands.  The input set is [-U_i^-, U] with a small
traction limit U and U_i^- = min(U^-, a^max_i).

The evaluator combines
  * the acquisition chain and the relayed braking recursion of OT (frozen
    namespace b2_design._namespace through certify_plan.namespace, frozen
    helpers corner_stats, D_quad, DB_quad), extended by the planned L1
    budgets A^(1), A^(2), the planned forced-response norms P^e, P^eps,
    P^f of Lemma 7.1, and the braking-jump size a_b - a_crawl;
  * the one-sided switch box, the sign-resolved envelopes F^+-, the velocity
    envelopes and the planned-residual admission of PT (certify_plan.py),
    extended by the time-resolved planned forced responses and by the
    realization plan over all closure offsets o_k in [0, (k-1) dbar];
  * the tail-variation recursion of OT/PT with the L1 tails of the planned
    pulses, and a time-resolved entry deadline (running supremum from the
    right of the tail bound plus the planned forced response).

Conditions (one row per test; Sec. 11 of the theory note, rows added by the
evaluator notes are marked *):
  A_R, A_S   Assumptions R and S (reference, schedules; exact rationals)
  C0a, C0b   timing premises
  H8         ramp before stop with the handoff floor v^xi - n eps_v
  C1..C6     sampled event, braking S1 stages, hold, velocity floor,
             handoff pad, headroom
  C7a, C7b   authority before T_xi (upper, lower), per 10 ms cell
  C7c_brk_lo, C7c_brk_up   braking branch  a_b + F^b_i <= U_i^-,
                           -a_b + F^b_i <= U
  C7c_pre_lo pre-receipt  P^pre_i <= U_i^-
  C7c_pre_up* pre-receipt upper bound  -a_crawl + Q^+_i <= U
  C8a..C8c   admission, C9a clearance floors, C9b barrier floors
  C10..C12   dispersion, marker, terminal gap
  NONPOS     reported only: commands certified nonpositive (all phases)

API
  config(**kw)              HR configuration (default: round-6 fan design)
  setup(cfg)                HRSetup (reference, schedules, jumps; exact)
  evaluate(cfg)             dict(conds, consts, bounds, admitted, failed,
                            timing)
  age_boundary_ms(cfg)      largest admitted age bound on the 1 ms grid
  cond_boundaries_ms(cfg)   per-condition boundaries on the 1 ms grid
  regress()                 regression and brute-force checks
"""
import copy
import json
import math
import os
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import certify_plan as cq                                # noqa: E402
import certify_profile as cp                             # noqa: E402

CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_HR = os.path.join(CODE, 'data', 'new_d19', 'headref')
DESIGN_D = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')
B2_RESULTS = os.path.join(CODE, 'data', 'new_d19', 'b2_results.json')

# constants of certify_v30.py without an override hook (U is not frozen
# here: the evaluator uses its own U and U_i^-)
FIXED = dict(ell=25.0, kappa=1.0, v_c=1.2)
BIG = 1.0e9
F = cq.F

# HR keys added to the round-6 design configuration
HR_KEYS = dict(U=0.02, Uminus=None, cell_dt=0.01, e0=None, lags=[15, 20])

COND_NAMES = ('A_R', 'A_S', 'C0a', 'C0b', 'H8', 'C1', 'C2', 'C3', 'C4', 'C5',
              'C6', 'C7a', 'C7b', 'C7c_brk_lo', 'C7c_brk_up', 'C7c_pre_lo',
              'C7c_pre_up', 'C8a', 'C8b', 'C8c', 'C9a', 'C9b', 'C10', 'C11',
              'C12')
REPORT_ONLY = ('NONPOS',)


# ===================================================================== config
_BASE = {}


def base_cfg():
    """cfg of data/new_d19/caseD/design.json (round-6 fan plan)."""
    if 'cfg' not in _BASE:
        _BASE['cfg'] = json.load(open(DESIGN_D))['cfg']
    return copy.deepcopy(_BASE['cfg'])


def config(**kw):
    """HR configuration: the round-6 design cfg, the HR keys (U, Uminus,
    cell_dt, e0, lags) and kw (plan entries merged).  Uminus defaults to
    the braking limit k of the design (1.2 m/s^2)."""
    cfg = base_cfg()
    for k, v in HR_KEYS.items():
        if k not in cfg:
            cfg[k] = copy.deepcopy(v)
    kw = copy.deepcopy(kw)
    pl = kw.pop('plan', None)
    cfg.update(kw)
    if pl is not None:
        if pl.get('family') and pl.get('family') != cfg['plan'].get('family'):
            cfg['plan'] = dict(pl)
        else:
            cfg['plan'] = dict(cfg['plan'], **pl)
    if cfg.get('Uminus') is None:
        cfg['Uminus'] = float(cfg.get('k', 1.2))
    return cfg


# ============================================== piecewise polynomials (float)
class PW:
    """Piecewise quadratic f(x) = c0[j] + c1[j] tau + c2[j] tau^2 with
    tau = x - X[j] on [X[j], X[j+1]) (the last piece extends to +infinity,
    the first starts at -BIG).  ext(a, b, mode) returns the exact minimum or
    maximum over the closed interval [a, b] (vectorized), with both one-sided
    limits at every breakpoint inside the interval (each piece is evaluated on
    its closure)."""

    def __init__(self, X, c0, c1=None, c2=None):
        self.X = np.asarray(X, float)
        m = len(self.X)
        self.c0 = np.asarray(c0, float)
        self.c1 = np.zeros(m) if c1 is None else np.asarray(c1, float)
        self.c2 = np.zeros(m) if c2 is None else np.asarray(c2, float)
        self.Xe = np.append(self.X[1:], np.inf)
        assert len(self.c0) == m and np.all(np.diff(self.X) > 0)

    def __call__(self, x):
        x = np.asarray(x, float)
        j = np.clip(np.searchsorted(self.X, x, 'right') - 1, 0, len(self.X) - 1)
        tau = x - self.X[j]
        return self.c0[j] + self.c1[j] * tau + self.c2[j] * tau * tau

    def ext(self, a, b, mode='max'):
        a = np.atleast_1d(np.asarray(a, float))
        b = np.atleast_1d(np.asarray(b, float))
        a, b = np.broadcast_arrays(a, b)
        m = len(self.X)
        sgn = 1.0 if mode == 'max' else -1.0
        ja = np.clip(np.searchsorted(self.X, a, 'right') - 1, 0, m - 1)
        jb = np.clip(np.searchsorted(self.X, b, 'right') - 1, 0, m - 1)
        out = np.full(a.shape, -np.inf)
        K = int(np.max(jb - ja)) + 1 if a.size else 0
        for d in range(K):
            j = ja + d
            ok = j <= jb
            j = np.minimum(j, jb)
            lo = np.maximum(a, self.X[j])
            hi = np.minimum(b, self.Xe[j])
            tl = lo - self.X[j]
            th = hi - self.X[j]
            c0, c1, c2 = self.c0[j], self.c1[j], self.c2[j]
            fl = c0 + c1 * tl + c2 * tl * tl
            fh = c0 + c1 * th + c2 * th * th
            v = np.maximum(sgn * fl, sgn * fh)
            nz = c2 != 0.0
            if np.any(nz):
                with np.errstate(divide='ignore', invalid='ignore'):
                    tv = np.where(nz, -c1 / (2.0 * np.where(nz, c2, 1.0)), 0.0)
                ins = nz & (tv > tl) & (tv < th)
                fv = c0 + c1 * tv + c2 * tv * tv
                v = np.where(ins, np.maximum(v, sgn * fv), v)
            out = np.where(ok, np.maximum(out, v), out)
        return sgn * out


# ======================================================================= setup
class HRSetup:
    """Reference, schedules and planned jumps of an HR configuration.

    Exact data (fractions.Fraction): ref_t, ref_a (pieces of a_r on
    [0, T_xi)), ref_v (v_r at the piece starts), sched[u] (code unit u >= 1:
    piece starts x_j, values rho_j, D_j = int_0^x rho, S_j = int int rho),
    jumps, markers.  Float views: PW objects for a_r (with the zero command
    before 0 and the continuation -a_crawl after T_xi), v_r, a_r + kappa v_r,
    rho_k, D_k = -dd_k, rho_k + kappa D_k, and the own-schedule terms
    A_k = d_k - s_m + dd_k/b_k, B_k = dd_k - rho_k/b_k + kappa A_k and
    gs_k = d_k - s_m of the realization plan (theory Sec. 10)."""

    def pair_jumps(self, c):
        """planned jumps whose pulses enter w^p of code pair c (1-based pair
        c+1): the reference jumps on (0, T_xi) and the schedule jumps of the
        code units 1..c-1; returns (h, lo) with lo = t_J (reference) or
        t_0 + x_J (schedules), the left end of the admissible interval
        K_J = [lo, lo + c dbar]."""
        hs = [h for (t, h) in self.ref_jumps_f]
        los = [t for (t, h) in self.ref_jumps_f]
        for u in range(1, c):
            for (x, h) in self.sched_jumps_f[u]:
                hs.append(h)
                los.append(self.t_0_f + x)
        return np.array(hs, float), np.array(los, float)

    def first_lo(self, c):
        """earliest planned jump that can reach code pair c or change the
        command of its own unit (own schedule included), for (C0a)."""
        los = [t for (t, h) in self.ref_jumps_f]
        for u in range(1, c + 1):
            los += [self.t_0_f + x for (x, h) in self.sched_jumps_f[u]]
        return min(los) if los else np.inf

    def last_lo(self):
        los = [t for (t, h) in self.ref_jumps_f]
        for u in range(1, self.n):
            los += [self.t_0_f + x for (x, h) in self.sched_jumps_f[u]]
        return max(los) if los else -np.inf


_SETUP = OrderedDict()


def _setup_key(cfg):
    keys = ('plan', 'c0', 'd_s', 'e0', 'a_b', 'n', 'amax', 's_m')
    return json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True,
                      default=str)


def setup(cfg):
    """HRSetup of cfg (cached).  Reference = plan of unit 1 of
    certify_plan.plan(cfg); schedules rho_i(x) = a*_i(t_0 + x) -
    a*_{i-1}(t_0 + x) on [0, T_xi - t_0) (theory Sec. 3); exact checks of
    Assumptions R and S."""
    key = _setup_key(cfg)
    if key in _SETUP:
        _SETUP.move_to_end(key)
        return _SETUP[key]
    P = cq.plan(cfg)
    n = P.n
    ell = P.ell
    a_b = F(cfg['a_b'])
    T_xi = P.T_xi
    t_0 = P.extra.get('t_0')
    t_0 = Fr(0) if t_0 is None else Fr(t_0)
    d_s = [F(x) for x in cfg['d_s']]
    c0 = [F(x) for x in cfg['c0']]
    e0 = [Fr(0)] * n if cfg.get('e0') is None else [F(x) for x in cfg['e0']]
    amax = [F(x) for x in cfg['amax']]
    b = [a / F(FIXED['v_c']) for a in amax]
    kap = F(FIXED['kappa'])
    s_m = F(cfg['s_m'])
    S = HRSetup()
    S.P = P
    S.n = n
    S.T_xi, S.t_0, S.a_b = T_xi, t_0, a_b
    # ---------------------------------------------------------- reference
    rp = P.acc_pieces(0)
    S.ref_t = [p[0] for p in rp]
    S.ref_a = [p[2] for p in rp]
    S.ref_v = [P.state(0, t)[1] for t in S.ref_t]
    S.v_in = P.v_xi
    S.v_xi_h = P.v_c1                          # handoff speed v^xi = v_r(T_xi)
    S.a_crawl = -S.ref_a[-1]
    S.a_r0 = S.ref_a[0]
    S.tau_r = P.tau_star
    ref_jumps = [(S.ref_t[j], S.ref_a[j] - S.ref_a[j - 1])
                 for j in range(1, len(S.ref_t)) if S.ref_a[j] != S.ref_a[j - 1]]
    S.ref_jumps = ref_jumps
    S.takeover_jump = (Fr(0), S.a_r0)
    S.braking_jump = (T_xi, -a_b + S.a_crawl)
    # ---------------------------------------------------------- schedules
    S.sched = [None]
    S.sched_jumps = [None]
    S.d0 = [None] + [c0[u - 1] - e0[u] for u in range(1, n)]
    prem_S = dict(ok=True, closure=[], rate_end=[], dd_max=[], plan_match=[])
    for u in range(1, n):
        bps = sorted(set([t for (t, a) in P.units[u - 1]]
                         + [t for (t, a) in P.units[u]] + [t_0]))
        bps = [t for t in bps if t_0 <= t < T_xi]
        xs, rs = [], []
        for t in bps:
            r = P.state(u, t)[2] - P.state(u - 1, t)[2]
            if rs and r == rs[-1]:
                continue
            xs.append(t - t_0)
            rs.append(r)
        X_end = T_xi - t_0
        # drop trailing zero pieces: the schedule ends at x^f
        while rs and rs[-1] == 0:
            X_end = xs[-1]
            xs.pop()
            rs.pop()
        x_f = X_end if rs else Fr(0)
        # exact single and double integrals at the piece starts
        Dl, Sl = [], []
        Dv, Sv = Fr(0), Fr(0)
        for j in range(len(xs)):
            Dl.append(Dv)
            Sl.append(Sv)
            L = (xs[j + 1] if j + 1 < len(xs) else x_f) - xs[j]
            Sv = Sv + Dv * L + rs[j] * L * L / 2
            Dv = Dv + rs[j] * L
        D_end, S_end = Dv, Sv
        jumps = []
        prev = Fr(0)
        for j in range(len(xs)):
            if rs[j] != prev:
                jumps.append((xs[j], rs[j] - prev))
            prev = rs[j]
        if prev != 0:
            jumps.append((x_f, -prev))
        S.sched.append(dict(x=xs, rho=rs, D=Dl, S=Sl, x_f=x_f, D_end=D_end,
                            S_end=S_end))
        S.sched_jumps.append(jumps)
        # (S1): int rho = 0, int int rho = d0 - d_s (= C_i with e0 = 0)
        ok1 = (D_end == 0) and (S_end == S.d0[u] - d_s[u])
        # (S2): dd = -D <= 0 at every breakpoint (D piecewise linear)
        ok2 = all(x >= 0 for x in Dl + [D_end])
        # consistency with the plan: d = d0 - S = s*_{u-1} - s*_u - ell and
        # dd = -D = v*_{u-1} - v*_u at the breakpoints (theory Sec. 3)
        okp = True
        for j in range(len(xs)):
            t = t_0 + xs[j]
            sp, vp, _ = P.state(u - 1, t)
            sf, vf, _ = P.state(u, t)
            if (S.d0[u] - Sl[j] != sp - sf - ell) or (-Dl[j] != vp - vf):
                okp = False
        prem_S['closure'].append(dict(D_end=str(D_end), S_end=str(S_end),
                                      target=str(S.d0[u] - d_s[u]), ok=ok1))
        prem_S['dd_max'].append(float(-min(Dl + [D_end])) if Dl else 0.0)
        prem_S['plan_match'].append(okp)
        prem_S['ok'] = bool(prem_S['ok'] and ok1 and ok2 and okp)
    S.prem_S = prem_S
    # ---------------------------------------------------------- Assumption R
    vmin = min(S.ref_v + [S.v_xi_h])
    S.prem_R = dict(v_in=float(S.v_in), v_min_before_T_xi=float(vmin),
                    v_xi=float(S.v_xi_h), t_0=float(t_0), T_xi=float(T_xi),
                    t_0_in_range=bool(0 <= t_0 < T_xi),
                    deceleration_only=bool(all(a <= 0 for a in S.ref_a)),
                    head_error_zero=bool(e0[0] == 0),
                    ok=bool(S.v_in > 0 and vmin > 0 and S.v_xi_h > 0
                            and 0 <= t_0 < T_xi))
    S.markers = list(P.markers)
    S.s_ref_stop = P.s_ref_stop
    S.C = list(P.C)
    # ---------------------------------------------------------- float views
    S.T_xi_f = float(T_xi)
    S.t_0_f = float(t_0)
    S.v_in_f = float(S.v_in)
    S.v_xi_h_f = float(S.v_xi_h)
    S.a_crawl_f = float(S.a_crawl)
    S.a_r0_f = float(S.a_r0)
    S.e0_f = np.array([float(x) for x in e0])
    S.ref_jumps_f = [(float(t), float(h)) for (t, h) in ref_jumps]
    S.sched_jumps_f = [None] + [[(float(x), float(h)) for (x, h) in S.sched_jumps[u]]
                                for u in range(1, n)]
    rt = [float(t) for t in S.ref_t]
    ra = [float(a) for a in S.ref_a]
    rv = [float(v) for v in S.ref_v]
    S.arA = PW([-BIG] + rt, [0.0] + ra)            # a_r, 0 before 0, -a_crawl after
    S.vr = PW([-BIG] + rt, [S.v_in_f] + rv, [0.0] + ra)
    kf = float(kap)
    S.arkvr = PW([-BIG] + rt, [kf * S.v_in_f] + [a + kf * v for a, v in zip(ra, rv)],
                 [0.0] + [kf * a for a in ra])
    S.rho_pw, S.negdd_pw, S.rkd_pw = [None], [None], [None]
    S.A_pw, S.B_pw, S.gs_pw = [None], [None], [None]
    for u in range(1, n):
        sc = S.sched[u]
        bu = b[u]
        d0u = S.d0[u]
        xs = sc['x'] + [sc['x_f']]
        X = [-BIG] + [float(x) for x in xs]
        rho0 = [Fr(0)] + sc['rho'] + [Fr(0)]
        D0 = [Fr(0)] + sc['D'] + [sc['D_end']]
        S0 = [Fr(0)] + sc['S'] + [sc['S_end']]
        if not sc['x']:                          # no schedule
            X = [-BIG]
            rho0, D0, S0 = [Fr(0)], [Fr(0)], [Fr(0)]
        # A = d0 - S - s_m - D/b on a piece: S = S_j + D_j tau + rho_j tau^2/2,
        # D = D_j + rho_j tau
        A0 = [d0u - s_m - Sj - Dj / bu for Sj, Dj in zip(S0, D0)]
        A1 = [-Dj - rj / bu for Dj, rj in zip(D0, rho0)]
        A2 = [-rj / 2 for rj in rho0]
        B0 = [-Dj - rj / bu + kap * a0 for Dj, rj, a0 in zip(D0, rho0, A0)]
        B1 = [-rj + kap * a1 for rj, a1 in zip(rho0, A1)]
        B2 = [kap * a2 for a2 in A2]
        fl = lambda L: [float(x) for x in L]     # noqa: E731
        S.rho_pw.append(PW(X, fl(rho0)))
        S.negdd_pw.append(PW(X, fl(D0), fl(rho0)))
        S.rkd_pw.append(PW(X, fl([rj + kap * Dj for rj, Dj in zip(rho0, D0)]),
                           fl([kap * rj for rj in rho0])))
        S.A_pw.append(PW(X, fl(A0), fl(A1), fl(A2)))
        S.B_pw.append(PW(X, fl(B0), fl(B1), fl(B2)))
        S.gs_pw.append(PW(X, fl([d0u - s_m - Sj for Sj in S0]), fl([-Dj for Dj in D0]),
                          fl([-rj / 2 for rj in rho0])))
    _SETUP[key] = S
    while len(_SETUP) > 8:
        _SETUP.popitem(last=False)
    return S


# ================================================ planned pulses (Lemma 7.1)
def pulse_bounds(lam, dbar, h, lo, hi, tg, tg1, chunk=40):
    """Cell bounds of the planned forced response of one pair (Lemma 7.1,
    cell-extended form of Sec. 7).  Jump J has height h[J] and admissible
    pulse interval K_J = [lo[J], hi[J]]; the pulse has length at most dbar.
    For the cell [tg, tg1] every t in the cell has t - s in
    [max(0, tg - hi), tg1 - lo] for s in K_J, s <= t, so
    M_J(cell) = dbar max of the impulse response over that lag interval
    bounds the pulse integral for every t in the cell.  Returns the arrays
    e_hi >= 0 >= e_lo, x_hi, x_lo (eps), f_hi, f_lo (feedback)."""
    nC = len(tg)
    out = {k: np.zeros(nC) for k in ('e_hi', 'e_lo', 'x_hi', 'x_lo', 'f_hi', 'f_lo')}
    h = np.asarray(h, float)
    if h.size == 0:
        return out
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    r1, r2, r3 = 1.0 / lam, 2.0 / lam, 3.0 / lam
    he_star = 1.0 / (lam * math.e)
    hx_min = -math.exp(-2.0)
    hf_min = -lam * math.exp(-3.0)
    g0 = int(np.searchsorted(tg1, float(np.min(lo)), 'left'))
    if g0 >= nC:
        return out
    ts = tg[g0:]
    ts1 = tg1[g0:]
    acc = {k: np.zeros(nC - g0) for k in out}
    for s in range(0, h.size, chunk):
        hh = h[s:s + chunk][:, None]
        ll = lo[s:s + chunk][:, None]
        uu = hi[s:s + chunk][:, None]
        rh = ts1[None, :] - ll
        act = rh >= 0.0
        rl = np.maximum(ts[None, :] - uu, 0.0)
        rh = np.where(act, rh, 0.0)
        rl = np.where(act, rl, 0.0)
        El = np.exp(-lam * rl)
        Eh = np.exp(-lam * rh)
        # h_e(r) = r e^{-lam r} >= 0, maximum 1/(lam e) at r = 1/lam
        me = np.maximum(rl * El, rh * Eh)
        me = np.where((rl < r1) & (rh > r1), he_star, me)
        # h_eps(r) = (1 - lam r) e^{-lam r}, minimum -e^{-2} at r = 2/lam
        a_ = (1.0 - lam * rl) * El
        b_ = (1.0 - lam * rh) * Eh
        xmax = np.maximum(a_, b_)
        xmin = np.where((rl < r2) & (rh > r2), hx_min, np.minimum(a_, b_))
        # h_f(r) = lam (2 - lam r) e^{-lam r}, minimum -lam e^{-3} at 3/lam
        a_ = lam * (2.0 - lam * rl) * El
        b_ = lam * (2.0 - lam * rh) * Eh
        fmax = np.maximum(a_, b_)
        fmin = np.where((rl < r3) & (rh > r3), hf_min, np.minimum(a_, b_))
        af = act.astype(float) * dbar
        Me = me * af
        Mxp = np.maximum(xmax, 0.0) * af
        Mxm = np.maximum(-xmin, 0.0) * af
        Mfp = np.maximum(fmax, 0.0) * af
        Mfm = np.maximum(-fmin, 0.0) * af
        acc['e_hi'] += (np.maximum(hh, 0.0) * Me).sum(0)
        acc['e_lo'] += (np.minimum(hh, 0.0) * Me).sum(0)
        t1 = hh * Mxp
        t2 = -hh * Mxm
        acc['x_hi'] += np.maximum(t1, t2).sum(0)
        acc['x_lo'] += np.minimum(t1, t2).sum(0)
        t1 = hh * Mfp
        t2 = -hh * Mfm
        acc['f_hi'] += np.maximum(t1, t2).sum(0)
        acc['f_lo'] += np.minimum(t1, t2).sum(0)
    for k in out:
        out[k][g0:] = acc[k]
    return out


def tail_l1(dbar, h, hi, TGRID):
    """A^tail(T) = dbar sum_{J: hi_J >= T} |h_J| >= int_T^inf |w^p| (every
    pulse lies in K_J = [lo_J, hi_J] and has length at most dbar); a
    nonincreasing step function evaluated on the grid."""
    out = np.zeros(len(TGRID))
    if len(h) == 0:
        return out
    order = np.argsort(hi)
    hs = np.asarray(hi, float)[order]
    w = np.abs(np.asarray(h, float)[order])
    suf = np.cumsum(w[::-1])[::-1]
    k = np.searchsorted(hs, TGRID, 'left')
    ok = k < len(hs)
    out[ok] = dbar * suf[k[ok]]
    return out


def win_max(arr, m):
    """out[g] = max(arr[max(0, g - m) .. g])."""
    out = np.array(arr, float, copy=True)
    for d in range(1, m + 1):
        if d >= len(arr):
            break
        out[d:] = np.maximum(out[d:], arr[:-d])
    return out


# ============================================================ forward chain
FIELDS_R1 = cq.FIELDS_R1


def chain(G, e0, eps0, dbar, A1, A2, Pe, Px, Pf, ramp, v_brk):
    """Acquisition chain and relayed braking recursion: a statement-for-
    statement copy of certify_plan.chain (= certify_v30.forward_pass) with
      Lambda^(1)_k = dbar sum_{j<k} V^acq_j + A^(1)_k        (takeover pulse)
      V^S2_k       = V^free_k + zeta_k (Lambda^(2)_k + A^(2)_k)
      eps-bar^(2)  = eps-hat + Lambda^(2) + P^eps,  e-bar^(2) = e-hat +
                     Lambda^(2)/(lam e) + P^e,  f-bar = f-hat + gamma
                     Lambda^(2) + P^f                      (theory Sec. 8.2)
      W_i          = dbar (V^hnd + ramp + f-bar^xi + sum (Theta_m + F^b_m))
    with ramp = a_b - a_crawl (theory Sec. 9) and the braking speed v_brk =
    v^xi = v_r(T_xi) in T_brk.  With A1 = A2 = Pe = Px = Pf = 0, ramp = a_b
    and v_brk = G['v_xi'] every field is bit-identical to the frozen
    forward pass (regress (R1))."""
    n = G['n']; alpha_g = G['alpha_g']; T_c = G['T_c']
    eps_det = G['eps_det']; beta = G['beta']; gamma = G['gamma']
    b = G['b']; a_b = G['a_b']
    eps_e = G['eps_e']; eps_v = G['eps_v']; VHND_PAD = G['VHND_PAD']
    corner_stats = G['corner_stats']; D_quad = G['D_quad']
    DB_quad = G['DB_quad']
    GAIN_X = G['GAIN_X']; GAIN_F = G['GAIN_F']; GAIN_E = G['GAIN_E']
    TV_KIMP = G['TV_KIMP']; XFREE_PK = G['XFREE_PK']
    EFREE_PK = G['EFREE_PK']; IE_BOX = G['IE_BOX']; IX_BOX = G['IX_BOX']
    L1_GE = G['L1_GE']; L1_GX = G['L1_GX']
    e0 = np.asarray(e0, float); eps0 = np.asarray(eps0, float)
    A1 = np.asarray(A1, float); A2 = np.asarray(A2, float)
    Pe = np.asarray(Pe, float); Px = np.asarray(Px, float); Pf = np.asarray(Pf, float)

    Lam1 = np.zeros(n); Lam2 = np.zeros(n); taub = np.zeros(n)
    tact = np.zeros(n); tset = np.zeros(n); E1 = np.zeros(n)
    ebar = np.zeros(n); fbar = np.zeros(n); fsw = np.zeros(n)
    tvS2 = np.zeros(n); x2tot = np.zeros(n); Tk = np.zeros(n)
    Dq = np.zeros(n); DBq = np.zeros(n); e2bar = np.zeros(n)
    sup_e = np.zeros(n); sup_x = np.zeros(n); sup_f = np.zeros(n)
    tv_f = np.zeros(n)

    for k in range(n):
        Lam1[k] = dbar*np.sum(Tk[:k]) + A1[k]
        m_k     = abs(eps0[k]) + Lam1[k]
        taub[k] = m_k/alpha_g[k] + T_c
        tact[k] = 0.0 if k == 0 else tset[k-1] + dbar
        tset[k] = tact[k] + taub[k]
        Lk_eps  = alpha_g[k] + np.sum(Tk[:k])
        E1[k]   = m_k*tact[k] \
                  + max(m_k**2 - eps_det**2, 0.0)/(2.0*alpha_g[k]) \
                  + (eps_det + Lk_eps*T_c)*T_c
        ebar[k] = abs(e0[k]) + E1[k]
        st      = corner_stats(ebar[k], eps_det)
        Lam2[k] = dbar*np.sum(tvS2[:k])
        x2tot[k] = st['sup_x'] + GAIN_X*Lam2[k] + Px[k]
        fbar[k]  = st['sup_f'] + GAIN_F*Lam2[k] + Pf[k]
        e2bar[k] = st['sup_e'] + GAIN_E*Lam2[k] + Pe[k]
        tvS2[k]  = st['tv_f'] + TV_KIMP*(Lam2[k] + A2[k])
        fsw[k]   = beta*ebar[k] + gamma*eps_det
        Tk[k]    = 2.0*alpha_g[k] + fsw[k] + tvS2[k]
        Dq[k]    = D_quad(ebar[k], eps_det, b[k])
        DBq[k]   = DB_quad(b[k])
        sup_e[k] = st['sup_e']; sup_x[k] = st['sup_x']
        sup_f[k] = st['sup_f']; tv_f[k] = st['tv_f']

    Fv = np.array([np.sum(fbar[:i]) for i in range(n)])

    f_xi = beta*eps_e + gamma*eps_v
    T_brk = (v_brk + eps_v)/a_b + (n-1)*dbar + 1.0
    diverged = False
    for _ in range(30):
        W = np.zeros(n); epsb = np.zeros(n); eb = np.zeros(n)
        fb = np.zeros(n); Th = np.zeros(n); Wphi = np.zeros(n)
        IE = np.zeros(n); IX = np.zeros(n); IF = np.zeros(n)
        Fcum = np.zeros(n)
        for k in range(1, n):
            W[k] = dbar*(VHND_PAD + ramp + f_xi
                         + np.sum(Th[1:k]) + np.sum(Fcum[1:k]))
            Wphi[k] = dbar*(VHND_PAD + f_xi
                            + np.sum(Th[1:k]) + np.sum(Fcum[1:k]))
            epsb[k] = XFREE_PK + GAIN_X*W[k]
            eb[k]   = EFREE_PK + GAIN_E*W[k]
            fb[k]   = beta*eb[k] + gamma*epsb[k]
            Fcum[k] = f_xi + np.sum(fb[1:k+1])
            IE[k]   = IE_BOX + L1_GE*W[k]
            IX[k]   = IX_BOX + L1_GX*W[k]
            IF[k]   = beta*IE[k] + gamma*IX[k]
            Th[k]   = beta*IX[k] + gamma*(W[k] + IF[k])
        epsb_max = np.max(epsb)
        if not np.isfinite(epsb_max) or epsb_max > 50.0:
            diverged = True; break
        T_new = (v_brk + eps_v)/a_b + (n-1)*(dbar + epsb_max/a_b)
        if abs(T_new - T_brk) < 1e-10:
            break
        T_brk = T_new
    closed = not (diverged or np.min(a_b - Fcum[:n-1]) <= 0)
    Fb = np.array([f_xi + np.sum(fb[1:i]) for i in range(n)])
    hops = epsb[1:]/(a_b - Fb[1:n])
    disp = np.sum(hops)
    late = np.sum(epsb[1:]/a_b)
    sched = eps_v/a_b + np.sum(hops)
    gaperr = np.array([EFREE_PK + GAIN_E*W[i]
                       + epsb[i]**2/(2*(a_b - Fb[i]))
                       for i in range(1, n)])
    Leps = np.max(alpha_g) + np.sum(Tk)
    return dict(closed=bool(closed), diverged=bool(diverged),
                Lam1=Lam1, Lam2=Lam2, taub=taub, tact=tact, tset=tset,
                E1=E1, ebar=ebar, fbar=fbar, fsw=fsw, tvS2=tvS2,
                x2tot=x2tot, e2bar=e2bar, Tk=Tk, Dq=Dq, DBq=DBq, F=Fv,
                sup_e=sup_e, sup_x=sup_x, sup_f=sup_f, tv_f=tv_f,
                W=W, Wphi=Wphi, epsb=epsb, eb=eb, fb=fb, Th=Th,
                Fcum=Fcum, Fb=Fb, IE=IE, IX=IX, IF=IF, f_xi=f_xi,
                T_brk=T_brk, hops=hops, disp=disp, late=late, sched=sched,
                gaperr=gaperr, Leps=Leps)


# ============================================================ entry curves
def entry_curves(G, R, Lam2, A2, Atail, dbar, lags):
    """Tail-variation recursion of OT/PT (supplement S.II-E; frozen
    certified_entry, certify_profile._entry_curves) on the deviation part
    Phi, with the planned pulses of every pair as additional forcing of its
    feedback (theory Sec. 8.3):
      V^f_j(t) = FT_j((t - tswb_j)^+) + atoms
                 + gamma (dbar V_{j-1}((t - dbar)^+) + A^tail_j(t))
                 + min_B [Psi(B) (Lambda^(2)_j + A^(2)_j)
                          + Psi(0) (dbar V_{j-1}((t-B-dbar)^+) + A^tail_j((t-B)^+))],
      V_j(t)   = V_{j-1}((t - dbar)^+) + V^f_j(t),
    and the state tails of the free part and of the Phi-forced part (the
    planned forced part is added time resolved by the caller).  With A2 = 0
    and Atail = 0 the curves are bit-identical to the frozen recursion."""
    nB, nA = lags
    n = G['n']; NT = G['NT']; TGRID = G['TGRID']; dtg = G['dtg']
    alpha_g = G['alpha_g']; gamma = G['gamma']; lam = G['lam']
    EULER = G['EULER']
    _corner_tails = G['_corner_tails']; _shift_ceil = G['_shift_ceil']
    _Psi = G['_Psi']; L1FP_X = G['L1FP_X']; _tail_sup = G['_tail_sup']
    WtotV = (np.asarray(Lam2, float) + np.asarray(A2, float)) + 1e-15
    WtotS = np.asarray(Lam2, float) + 1e-15
    SeL, SxL, FTL = [], [], []
    for k in range(n):
        Se, Sx, FT = _corner_tails(R['ebar'][k])
        SeL.append(Se); SxL.append(Sx); FTL.append(FT)
    V = [np.zeros(NT)]
    Bset = [2.0 * m for m in range(nB)]
    for j in range(1, n + 1):
        jj = j - 1
        Vs = _shift_ceil(V[jj], dbar)
        sh = R['tset'][jj]
        idx = np.maximum(np.floor((TGRID - sh) / dtg).astype(int), 0)
        idx = np.minimum(idx, NT - 1)
        ft = FTL[jj][idx]
        jmp = np.where(TGRID <= R['tact'][jj] + 1e-12, alpha_g[jj], 0.0) \
            + np.where(TGRID <= R['tset'][jj] + 1e-12,
                       alpha_g[jj] + R['fsw'][jj], 0.0)
        if jj == 0:
            forc = np.zeros(NT)
        else:
            At = Atail[jj]
            forc = gamma * dbar * Vs + gamma * At
            ct = np.full(NT, np.inf)
            for B in Bset:
                ct = np.minimum(ct, _Psi(B) * WtotV[jj]
                                + L1FP_X * dbar * _shift_ceil(V[jj], B + dbar)
                                + L1FP_X * _shift_ceil(At, B))
            forc = forc + ct
        V.append(Vs + ft + jmp + forc)
    totEL = []; totXL = []
    Aset = [2.0 * m for m in range(nA)]
    for i in range(n):
        totE = SeL[i].copy(); totX = SxL[i].copy()
        if i >= 1:
            fbE = np.full(NT, np.inf); fbX = np.full(NT, np.inf)
            for A in Aset:
                Vsh = dbar * _shift_ceil(V[i], A + dbar)
                kE = float(_tail_sup(0.0, 1.0, np.array([A]))[0])
                kX = float(_tail_sup(1.0, -lam, np.array([A]))[0])
                fbE = np.minimum(fbE, kE * WtotS[i] + (1.0 / (lam * EULER)) * Vsh)
                fbX = np.minimum(fbX, kX * WtotS[i] + 1.0 * Vsh)
            totE = totE + fbE; totX = totX + fbX
        totEL.append(totE); totXL.append(totX)
    return dict(V=V, SeL=SeL, SxL=SxL, totE=totEL, totX=totXL)


# ======================================================== plan lower bounds
def plan_lower_bounds(S, c, ta, tb, dbar, Pi_c):
    """Lower bounds, for each window [ta, tb] (arrays) and over every closure
    offset o_k in [0, (k-1) dbar], of the planned barrier h*_i, the planned
    residual mu*_i and the planned clearance margin g*_i = d_i - s_m of code
    pair c (1-based pair i = c+1) along the realization plan (theory
    Sec. 10):
      h*_i  = A_i(x_i) + Pi_i V_{i-1}(t),   V_{i-1} = v_r + sum_k D_k(x_k),
      mu*_i = B_i(x_i) + Pi_i (a*_{i-1} + kappa V_{i-1}),
              a*_{i-1} + kappa V_{i-1} = (a_r + kappa v_r) + sum_k (rho_k + kappa D_k),
      g*_i  = gs_i(x_i) = d_i(x_i) - s_m,
    x_k = t - t_0 - o_k.  Each own term is minimized over x in
    [ta - t_0 - (i-1) dbar, tb - t_0]; each predecessor term (k = 2..i-1) is
    maximized (Pi_i < 0) or minimized (Pi_i > 0) over its own x-window
    [ta - t_0 - (k-1) dbar, tb - t_0] and v_r, a_r + kappa v_r over
    [ta, tb]; the offsets are treated as independent (conservative)."""
    t_0 = S.t_0_f
    xa = ta - t_0 - c * dbar
    xb = tb - t_0
    Aown = S.A_pw[c].ext(xa, xb, 'min')
    Bown = S.B_pw[c].ext(xa, xb, 'min')
    gown = S.gs_pw[c].ext(xa, xb, 'min')
    mode = 'max' if Pi_c < 0 else 'min'
    Vx = S.vr.ext(ta, tb, mode)
    Mx = S.arkvr.ext(ta, tb, mode)
    for u in range(1, c):
        xau = ta - t_0 - u * dbar
        Vx = Vx + S.negdd_pw[u].ext(xau, xb, mode)
        Mx = Mx + S.rkd_pw[u].ext(xau, xb, mode)
    return Aown + Pi_c * Vx, Bown + Pi_c * Mx, gown


def q_bound(S, c, ta, tb, dbar):
    """Q_{i-1} of theory Sec. 10 for code pair c (predecessor = 1-based unit
    i-1 = c): the oscillation of a_r over [t - (i-1) dbar, t] plus, for the
    schedules k = 2..i-1, the oscillation of rho_k over
    [t - t_0 - (i-1) dbar, t - t_0], evaluated for t in the window
    [ta, tb] (arrays); a_r is the zero command before 0."""
    wa = ta - c * dbar
    Q = S.arA.ext(wa, tb, 'max') - S.arA.ext(wa, tb, 'min')
    for u in range(1, c):
        xa = wa - S.t_0_f
        xb = tb - S.t_0_f
        Q = Q + S.rho_pw[u].ext(xa, xb, 'max') - S.rho_pw[u].ext(xa, xb, 'min')
    return Q


# ================================================================ evaluate
def _cond(ok, slack, value=None, threshold=None, sense=None, certifying=True,
          **kw):
    d = dict(ok=bool(ok), slack=None if slack is None else float(slack),
             value=None if value is None else float(value),
             threshold=None if threshold is None else float(threshold),
             sense=sense, certifying=bool(certifying))
    d.update(kw)
    return d


def _argmin2(arr2):
    """(value, row, col) of the minimum of a 2-d array."""
    j = int(np.argmin(arr2))
    r, c = np.unravel_index(j, arr2.shape)
    return float(arr2[r, c]), int(r), int(c)


def evaluate(cfg, S=None, want_entry=True, keep_curves=False):
    """Full evaluation of one HR configuration at the age bound cfg['dbar']."""
    if 'n' not in cfg or 'U' not in cfg:
        cfg = config(**cfg)
    for kf, vf in FIXED.items():
        if kf in cfg and abs(float(cfg[kf]) - vf) > 0:
            raise ValueError('%s is frozen at %g' % (kf, vf))
    t_start = time.time()
    timing = OrderedDict()
    S = S or setup(cfg)
    timing['setup_s'] = time.time() - t_start
    n = S.n
    dbar = float(cfg['dbar'])
    G = cq.namespace(cfg)
    lam = G['lam']; beta = G['beta']; gamma = G['gamma']
    alpha_g = G['alpha_g']; b = G['b']; Pi = G['Pi']; eta = G['eta']
    kappa = G['kappa']; a_b = G['a_b']; eps_e = G['eps_e']
    eps_v = G['eps_v']; eps_det = G['eps_det']; phi = G['phi']
    T_c = G['T_c']; A_run = G['A_run']
    TGRID = G['TGRID']; dtg = G['dtg']; NT = G['NT']
    U = float(cfg['U'])
    Uminus = float(cfg['Uminus'])
    amax = np.array([float(x) for x in cfg['amax']])
    Umin = np.minimum(Uminus, amax)
    s_m = float(cfg['s_m'])
    NUMPAD = float(cfg['numpad'])
    T_xi = S.T_xi_f; t_0 = S.t_0_f; v_brk = S.v_xi_h_f
    a_crawl = S.a_crawl_f; v_in = S.v_in_f
    eps0 = np.array(cq.eps0_of(cfg))
    e0 = S.e0_f.copy()
    v0 = np.array([float(F(x)) for x in cfg['v0']])
    vt0 = v0 - v_in
    d_s = np.array([float(x) for x in cfg['d_s']])
    one_sided = bool(cfg.get('one_sided', True))
    lags = tuple(int(x) for x in cfg.get('lags', (15, 20)))
    res = dict(case='HR', dbar=dbar, T_xi=T_xi, lam=lam, U=U, Uminus=Uminus)
    conds = OrderedDict()

    # ------------------------------------------------------ cells (Sec. 7)
    Dg = float(cfg['cell_dt'])
    NGc = int(math.ceil(T_xi / Dg - 1e-9))
    gidx = np.arange(NGc)
    tg = gidx * Dg
    tg1 = (gidx + 1) * Dg
    cell_hi = np.minimum(tg1, T_xi)

    # ------------------------------------------------ planned pulses (Sec. 7)
    t1 = time.time()
    zero = dict(e_hi=np.zeros(NGc), e_lo=np.zeros(NGc), x_hi=np.zeros(NGc),
                x_lo=np.zeros(NGc), f_hi=np.zeros(NGc), f_lo=np.zeros(NGc))
    pb = [zero]
    jl = [None]
    Pe = np.zeros(n); Px = np.zeros(n); Pf = np.zeros(n)
    A1 = np.zeros(n); A2 = np.zeros(n)
    for c in range(1, n):
        h, lo = S.pair_jumps(c)
        hi = lo + c * dbar
        jl.append((h, lo, hi))
        pb.append(pulse_bounds(lam, dbar, h, lo, hi, tg, tg1))
        Pe[c] = max(float(np.max(pb[c]['e_hi'])), float(-np.min(pb[c]['e_lo'])))
        Px[c] = max(float(np.max(pb[c]['x_hi'])), float(-np.min(pb[c]['x_lo'])))
        Pf[c] = max(float(np.max(pb[c]['f_hi'])), float(-np.min(pb[c]['f_lo'])))
        A2[c] = dbar * float(np.sum(np.abs(h)))
        A1[c] = dbar * abs(S.a_r0_f)
    timing['pulses_s'] = time.time() - t1

    # ------------------------------------------------------ chain (Secs. 8, 9)
    R = chain(G, e0, eps0, dbar, A1, A2, Pe, Px, Pf, ramp=a_b - a_crawl,
              v_brk=v_brk)
    tset = R['tset']; tact = R['tact']; Lam1 = R['Lam1']; Lam2 = R['Lam2']
    E1 = R['E1']
    res['closed'] = R['closed']

    # ------------------------------------------------ assumptions and premises
    conds['A_R'] = _cond(S.prem_R['ok'], S.prem_R['v_min_before_T_xi'],
                         S.prem_R['v_min_before_T_xi'], 0.0, '>', unit='m/s',
                         detail=S.prem_R)
    conds['A_S'] = _cond(S.prem_S['ok'], None, None, None, None,
                         detail=dict(closure=S.prem_S['closure'],
                                     dd_max=S.prem_S['dd_max'],
                                     plan_match=S.prem_S['plan_match']))
    first = np.array([S.first_lo(c) for c in range(1, n)])
    c0a = first - tset[1:]
    s_c0a = float(np.min(c0a)) if np.all(np.isfinite(c0a)) else \
        (float(np.min(c0a[np.isfinite(c0a)])) if np.any(np.isfinite(c0a)) else np.inf)
    conds['C0a'] = _cond(s_c0a >= 0, s_c0a, float(np.max(tset)),
                         float(np.min(first)) if np.any(np.isfinite(first)) else None,
                         '<=', unit='s', per_pair=[float(x) for x in c0a])
    last = S.last_lo()
    lhs0b = last + 2 * (n - 1) * dbar
    s_c0b = T_xi - lhs0b if np.isfinite(last) else np.inf
    conds['C0b'] = _cond(s_c0b >= 0, s_c0b,
                         lhs0b if np.isfinite(last) else None, T_xi, '<=',
                         unit='s', last_jump=float(last) if np.isfinite(last) else None)
    vflr_c = v_brk - n * eps_v
    lhs8 = (n - 1) * dbar * float(np.max(Umin))
    conds['H8'] = _cond(lhs8 < vflr_c, vflr_c - lhs8, lhs8, vflr_c, '<',
                        unit='m/s')
    Leps = float(R['Leps'])
    ev_ok = (eps_det - phi >= Leps * T_c) and (eps_det > Leps * T_c)
    conds['C1'] = _cond(ev_ok, eps_det - phi - Leps * T_c, Leps * T_c,
                        eps_det - phi, '<=', unit='m/s')
    ps1 = eps0 + Lam1
    s_c2 = -float(np.max(ps1))
    conds['C2'] = _cond(s_c2 >= 0 or not one_sided, s_c2, float(np.max(ps1)),
                        0.0, '<=', unit='m/s', required=one_sided)

    if not R['closed']:
        for nm in COND_NAMES:
            if nm not in conds:
                conds[nm] = _cond(False, None)
        conds['C6'] = _cond(False, float(np.min(a_b - R['Fb'][1:n])))
        conds['NONPOS'] = _cond(False, None, certifying=False)
        res.update(conds=conds, admitted=False,
                   failed=[k for k in COND_NAMES if not conds[k]['ok']],
                   note='braking recursion diverged or headroom <= 0',
                   timing=timing)
        return res

    # ------------------------------------------ entry and handoff pad (8.3)
    t1 = time.time()
    Atail = [None]
    for c in range(1, n):
        h, lo, hi = jl[c]
        Atail.append(tail_l1(dbar, h, hi, TGRID))
    Tent_lag = np.full(n, np.inf)
    Tent_abs = np.full(n, np.inf)
    entry_planned = [0.0] * n
    V = None
    if want_entry:
        ec = entry_curves(G, R, Lam2, A2, Atail, dbar, lags)
        V = ec['V']
        # candidate instants: the points t_k = k dtg of the fixed grid in
        # [tswb_i, T_xi]; they do not move with dbar (supplement S.II-E,
        # eq. (seq:tent)), which the monotonicity in dbar uses.  The
        # envelopes of entry_curves are functions of the lag and are read at
        # floor(t_k - tswb_i) on the grid (nonincreasing: a rounded-down lag
        # does not decrease them); the planned part is the largest cell value
        # over the closed cells [g Dg, (g+1) Dg] that meet [t_k, t_k + dtg],
        # in integer arithmetic with r = Dg/dtg grid steps per cell: cell
        # k//r, the cell before it if r divides k, and the cell after it if
        # r divides k+1 (this set also contains every cell with the
        # floating-point ends of interval_headref.make_cells that contains a
        # point of [t_k, t_k + dtg]).
        kK = min(int(math.floor(T_xi / dtg + 1e-9)), NT - 1)
        rc = int(round(Dg / dtg))
        assert rc >= 2 and abs(rc * dtg - Dg) < 1e-12
        for i in range(n):
            k0 = int(math.ceil(tset[i] / dtg - 1e-9))
            if kK < k0:
                continue
            kk_ = np.arange(k0, kK + 1)
            tk = TGRID[kk_]
            lag = np.clip(np.floor((tk - tset[i]) / dtg).astype(int), 0, NT - 1)
            BE = ec['totE'][i][lag].copy()
            BX = ec['totX'][i][lag].copy()
            if i >= 1:
                ga = np.clip(kk_ // rc - (kk_ % rc == 0).astype(int), 0, NGc - 1)
                gb = np.clip(kk_ // rc + ((kk_ + 1) % rc == 0).astype(int), 0, NGc - 1)
                pe = np.maximum(pb[i]['e_hi'], -pb[i]['e_lo'])
                px = np.maximum(pb[i]['x_hi'], -pb[i]['x_lo'])
                BE = BE + np.maximum(pe[ga], pe[gb])
                BX = BX + np.maximum(px[ga], px[gb])
            RSE = np.maximum.accumulate(BE[::-1])[::-1]
            RSX = np.maximum.accumulate(BX[::-1])[::-1]
            ok = (RSE <= eps_e) & (RSX <= eps_v)
            if ok.any():
                m0 = int(np.argmax(ok))
                Tent_abs[i] = float(tk[m0])
                Tent_lag[i] = float(tk[m0]) - tset[i]
                if i >= 1:
                    g0 = min(NGc - 1, int(tk[m0] / Dg))
                    entry_planned[i] = float(np.max(np.maximum(pb[i]['e_hi'], -pb[i]['e_lo'])[g0:]))
        T_ent_bar = float(np.max(Tent_abs))
        conds['C3'] = _cond(T_ent_bar + (n - 1) * dbar <= T_xi,
                            T_xi - T_ent_bar - (n - 1) * dbar,
                            T_ent_bar + (n - 1) * dbar, T_xi, '<=', unit='s')
        ixh = min(NT - 1, int((T_xi - dbar) / dtg))
        Vhnd = np.array([0.0] + [float(V[i][ixh]) for i in range(1, n)])
        vh = float(np.max(Vhnd))
        conds['C5'] = _cond(vh <= G['VHND_PAD'], G['VHND_PAD'] - vh, vh,
                            G['VHND_PAD'], '<=', unit='m/s^2')
    else:
        T_ent_bar = None; Vhnd = None
        conds['C3'] = _cond(False, None)
        conds['C5'] = _cond(False, None)
    timing['entry_s'] = time.time() - t1

    # ------------------------------------------------- velocity floor (C4)
    t1 = time.time()
    corners = [cq.box_corners(e0[k], E1[k], eps_det, one_sided) for k in range(n)]
    sig_p = np.array([1.0 if eps0[k] + Lam1[k] > eps_det else 0.0 for k in range(n)])
    sig_m = np.array([1.0 if eps0[k] - Lam1[k] < -eps_det else 0.0 for k in range(n)])
    Lam2h = Lam2.copy(); Lam2h[0] = 0.0
    Ek = []
    for k in range(n):
        Rg = np.maximum(tg - tset[k], 0.0)
        env = cq.EnvX(lam, corners[k], Rg) + Lam2[k] + pb[k]['x_hi']
        if not one_sided:
            M1k = abs(eps0[k]) + Lam1[k]
            pre = max(max(eps0[k] + Lam1[k], 0.0), M1k * sig_p[k],
                      float(cq.EnvX(lam, corners[k], 0.0)) + Lam2[k])
            env = np.where(tg < tset[k], pre, env)
        Ek.append(env)
    Ecum = np.cumsum(np.array(Ek), axis=0)
    vrmin = S.vr.ext(tg, cell_hi, 'min')
    flr = vrmin[None, :] - Ecum
    vfl, vi, vg = _argmin2(flr)
    conds['C4'] = _cond(vfl > 0, vfl, vfl, 0.0, '>', unit='m/s',
                        at=dict(unit=vi + 1, t=float(tg[vg])))

    # --------------------------------------- authority before T_xi (C7a, b)
    fpp = [pb[k]['f_hi'] for k in range(n)]
    fpm = [-pb[k]['f_lo'] for k in range(n)]
    up_all = np.zeros((n, NGc)); lo_all = np.zeros((n, NGc))
    Fp_all = np.zeros((n, NGc)); Fm_all = np.zeros((n, NGc))
    Pp_all = np.zeros((n, NGc)); Pm_all = np.zeros((n, NGc))
    for i in range(n):
        Fp = np.zeros(NGc); Fm = np.zeros(NGc)
        s1p = np.zeros(NGc); s1m = np.zeros(NGc)
        for k in range(i + 1):
            Rk = np.maximum(tg - (i - k) * dbar - tset[k], 0.0)
            chi = (tg < tset[k] + (i - k) * dbar).astype(float)
            tp = cq.Tf(lam, corners[k], Rk, +1)
            tm = cq.Tf(lam, corners[k], Rk, -1)
            forced = gamma * Lam2h[k]
            m = int(math.ceil((i - k) * dbar / Dg - 1e-9))
            Fp = Fp + tp + forced + win_max(fpp[k], m)
            Fm = Fm + tm + forced + win_max(fpm[k], m)
            s1p = np.maximum(s1p, alpha_g[k] * sig_p[k] * chi)
            s1m = np.maximum(s1m, alpha_g[k] * sig_m[k] * chi)
        Fp = Fp + s1p; Fm = Fm + s1m
        wa = tg - i * dbar
        Pp = S.arA.ext(wa, cell_hi, 'max')
        Pm = S.arA.ext(wa, cell_hi, 'min')
        for u in range(1, i + 1):
            Pp = Pp + S.rho_pw[u].ext(wa - t_0, cell_hi - t_0, 'max')
            Pm = Pm + S.rho_pw[u].ext(wa - t_0, cell_hi - t_0, 'min')
        Fp_all[i] = Fp; Fm_all[i] = Fm; Pp_all[i] = Pp; Pm_all[i] = Pm
        up_all[i] = Pp + Fp
        lo_all[i] = Pm - Fm
    s7a, ia, ga_ = _argmin2(U - up_all)
    s7b, ib, gb_ = _argmin2(lo_all + Umin[:, None])
    snp_pre, inp, gnp = _argmin2(-up_all)
    conds['C7a'] = _cond(s7a >= 0, s7a, float(np.max(up_all)), U, '<=',
                         unit='m/s^2', at=dict(unit=ia + 1, t=float(tg[ga_])),
                         per_unit_max=[float(x) for x in np.max(up_all, axis=1)])
    conds['C7b'] = _cond(s7b >= 0, s7b, float(np.min(lo_all)), None, '>=',
                         unit='m/s^2', at=dict(unit=ib + 1, t=float(tg[gb_])),
                         per_unit_min=[float(x) for x in np.min(lo_all, axis=1)])
    timing['envelopes_s'] = time.time() - t1

    # --------------------------------------------- braking and pre-receipt
    Fb = R['Fb']; fb = R['fb']; epsb = R['epsb']
    eb = R['eb']; W = R['W']; Wphi = R['Wphi']; Fv = R['F']; fbar = R['fbar']
    # F^b of every unit: F^b_1 = f-bar^xi (head), F^b_i = Fcum (followers);
    # the frozen Fcum[0] is 0 and is not used for the head
    Fcum = R['Fcum'].copy()
    Fcum[0] = R['f_xi']
    anet = a_b - Fb[1:n]
    conds['C6'] = _cond(float(np.min(anet)) > 0, float(np.min(anet)),
                        float(np.min(anet)), 0.0, '>', unit='m/s^2')
    s_blo = np.array([Umin[u] - (a_b + Fcum[u]) for u in range(n)])
    s_bup = np.array([U - (-a_b + Fcum[u]) for u in range(n)])
    conds['C7c_brk_lo'] = _cond(np.min(s_blo) >= 0, float(np.min(s_blo)),
                                float(np.max(a_b + Fcum)), None, '<=',
                                unit='m/s^2', per_unit=s_blo.tolist())
    conds['C7c_brk_up'] = _cond(np.min(s_bup) >= 0, float(np.min(s_bup)),
                                float(np.max(-a_b + Fcum)), U, '<=',
                                unit='m/s^2', per_unit=s_bup.tolist())
    # pre-receipt magnitude envelope of the theory note:
    # P^pre_i = a_crawl + Q_i, Q_i = f-bar^b_i + max{F_i, Q_{i-1}}, Q_1 = 0
    Qm = np.zeros(n)
    for u in range(1, n):
        Qm[u] = fb[u] + max(Fv[u], Qm[u - 1])
    Ppre = a_crawl + Qm
    s_plo = np.array([Umin[u] - Ppre[u] for u in range(1, n)])
    conds['C7c_pre_lo'] = _cond(np.min(s_plo) >= 0, float(np.min(s_plo)),
                                float(np.max(Ppre[1:])), None, '<=',
                                unit='m/s^2', per_unit=s_plo.tolist())
    # sign-resolved pre-receipt envelopes (evaluator notes, item N7):
    # in-flight acquisition commands carry layers formed in the handoff box,
    # |f_k| <= f_box; the own feedback on the pre-receipt window starts in
    # the box and is pushed down by the negative braking pulse only
    f_box = beta * eps_e + gamma * eps_v
    fpre_p = np.zeros(n); fpre_m = np.zeros(n)
    Qp = np.full(n, -np.inf); Qn = np.full(n, -np.inf)
    for u in range(1, n):
        base = f_box + u * dbar * (beta * epsb[u] + gamma * fb[u])
        fpre_p[u] = min(fb[u], base + gamma * Wphi[u])
        fpre_m[u] = min(fb[u], base + gamma * W[u])
        inflight = u * f_box
        Qp[u] = fpre_p[u] + max(inflight, Qp[u - 1] if u >= 2 else -np.inf)
        Qn[u] = fpre_m[u] + max(inflight, Qn[u - 1] if u >= 2 else -np.inf)
    s_pup = np.array([U - (-a_crawl + Qp[u]) for u in range(1, n)])
    conds['C7c_pre_up'] = _cond(np.min(s_pup) >= 0, float(np.min(s_pup)),
                                float(np.max(-a_crawl + Qp[1:])), U, '<=',
                                unit='m/s^2', per_unit=s_pup.tolist(),
                                lower_refined=[float(-a_crawl - Qn[u])
                                               for u in range(1, n)])

    # ------------------------------------------- admission, floors (Sec. 10)
    t1 = time.time()
    x2tot = R['x2tot']; e2bar = R['e2bar']; Dq = R['Dq']; DBq = R['DBq']
    IX = R['IX']; IF = R['IF']
    pairs = OrderedDict()
    adm = {'C8a': [], 'C8b': [], 'C8c': []}
    clr = []; bfl = []
    in2 = tg < T_xi
    for c in range(1, n):
        V1 = np.sum(np.abs(eps0[:c]) + Lam1[:c] + x2tot[:c])
        H1 = E1[c] + (alpha_g[c]/b[c])*R['taub'][c] + V1*abs(Pi[c]) \
            + Lam1[c]/b[c]
        V2 = 2.0*np.sum(x2tot[:c])
        L2tot = Lam2[c] + A2[c]
        Ht = Dq[c] + V2*abs(Pi[c]) + L2tot*(1.0/b[c] + DBq[c])
        M1 = abs(eps0[c]) + Lam1[c]
        in1 = tg < tset[c]
        hi1 = np.minimum(tg1, tset[c])
        Q1 = float(np.max(q_bound(S, c, tg[in1], hi1[in1], dbar)))
        Q2 = float(np.max(q_bound(S, c, tg[in2], cell_hi[in2], dbar)))
        hm1_coast = (b[c]*M1 + (Fv[c] + A_run[c-1] + Q1)*eta[c])/(kappa*b[c])
        hm1_s1 = (alpha_g[c] + b[c]*M1 + (Fv[c] + Q1)*eta[c])/(kappa*b[c])
        hm1 = max(hm1_coast, hm1_s1)
        hm2 = (fbar[c] + b[c]*x2tot[c] + (Fv[c] + Q2)*eta[c])/(kappa*b[c])
        bmin = min(b[c], b[c-1])
        Vbrk = v_brk + c*eps_v
        Vrise = (c - 1)*dbar*max(-a_crawl + Qm[c-1], 0.0)
        Ph2 = IX[c] + epsb[c]/bmin + epsb[c]**2/(2.0*anet[c-1]) + IF[c]/b[c] \
            + Vbrk*max(Pi[c], 0.0) + Vrise*max(-Pi[c], 0.0) \
            + W[c]*(1.0/b[c] + DBq[c])
        dem_prev = a_crawl + Fv[c-1] + max(A_run[c-1], fbar[c-1])
        Ubrk = max(dem_prev, a_crawl + Qm[c-1], a_b + Fb[c],
                   (a_b + Fb[c-1]) + fbar[c-1])
        hmb_s2 = (fb[c] + b[c]*epsb[c] + Ubrk*eta[c])/(kappa*b[c])
        hmb_rule = max(0.0, b[c]*epsb[c]
                       + (b[c]/b[c-1])*(a_b + Fb[c]) - a_b)/(kappa*b[c])
        hmb = max(hmb_s2, hmb_rule)
        ht0 = e0[c] + eps0[c]/b[c] + vt0[c-1]*Pi[c]
        hl1, ml1, gl1 = plan_lower_bounds(S, c, tg[in1], hi1[in1], dbar, Pi[c])
        hl2, ml2, gl2 = plan_lower_bounds(S, c, tg[in2], cell_hi[in2], dbar, Pi[c])
        mu1 = float(np.min(ml1)); mu2 = float(np.min(ml2))
        hs1 = float(np.min(hl1)); hs2 = float(np.min(hl2))
        gs1 = float(np.min(gl1))
        hstar_T = (d_s[c] - s_m) + v_brk*Pi[c]
        s_i = ht0 - H1 + mu1/kappa - hm1 - NUMPAD
        s_ii = ht0 - H1 - Ht + mu2/kappa - hm2 - NUMPAD
        hb = hstar_T + ht0 - H1 - Ht - Ph2
        s_iii = hb - hmb - NUMPAD
        adm['C8a'].append(s_i); adm['C8b'].append(s_ii); adm['C8c'].append(s_iii)
        gmin1 = gs1 + e0[c] - E1[c]
        gmin2 = (d_s[c] - s_m) - e2bar[c]
        gmin3 = (d_s[c] - s_m) - eb[c] - epsb[c]**2/(2.0*a_b)
        hf1 = hs1 + ht0 - H1
        hf2 = hs2 + ht0 - H1 - Ht
        bfl.append(min(hf1, hf2))
        # clearance floor: the direct identity of theory Sec. 10 and the
        # barrier-based floor of OT (Corollary 1: g' + b_i g = b_i h
        # - b_i Pi_i v_{i-1}, so with Pi_i <= 0 the clearance margin stays
        # above min{g_i(0), barrier floor} phase by phase); both are valid
        # lower bounds, the larger is used (evaluator notes, item N9)
        g_dir = min(gmin1, gmin2, gmin3)
        g_init = float(cfg['c0'][c-1]) - s_m
        g_bar = min(g_init, hf1, hf2, hb) if Pi[c] <= 0.0 else -np.inf
        clr.append(max(g_dir, g_bar))
        sigma = b[c]*min(s_i + NUMPAD, s_ii + NUMPAD, s_iii + NUMPAD)
        h0 = (float(cfg['c0'][c-1]) - s_m) + v0[c-1]/b[c-1] - v0[c]/b[c]
        pairs[c + 1] = dict(
            h0=h0, h_tilde0=ht0, H1=H1, H2=Ht, Hb=Ph2, hm1=hm1, hm2=hm2, hmb=hmb,
            hmb_s2=hmb_s2, hmb_rule=hmb_rule, Ubrk=Ubrk, V1=V1, V2=V2,
            Vbrk=Vbrk, Vrise=Vrise, Q1=Q1, Q2=Q2, mu1=mu1, mu2=mu2,
            hstar_min1=hs1, hstar_min2=hs2, gstar_min1=gs1, hstar_T=hstar_T,
            slack_a=s_i, slack_b=s_ii, slack_c=s_iii, h_b=hb,
            gmin_acq=gmin1, gmin_s2=gmin2, gmin_brk=gmin3, gmin_direct=g_dir,
            gmin_barrier=g_bar, g_init=g_init, gmin=max(g_dir, g_bar),
            e2bar=e2bar[c], hfloor_acq=hf1, hfloor_s2=hf2, sigma=sigma,
            L2tot=L2tot)
    for nm in ('C8a', 'C8b', 'C8c'):
        v = adm[nm]
        conds[nm] = _cond(min(v) > 0, min(v), None, 0.0, '>', unit='m',
                          per_pair=[float(x) for x in v])
    conds['C9a'] = _cond(min(clr) > 0, min(clr), min(clr), 0.0, '>', unit='m',
                         per_pair=[float(x) for x in clr])
    conds['C9b'] = _cond(min(bfl) >= 0, min(bfl), min(bfl), 0.0, '>=', unit='m',
                         per_pair=[float(x) for x in bfl])
    timing['admission_s'] = time.time() - t1

    # --------------------------------------------------- terminal (C10-C12)
    disp = float(R['disp'])
    gaperr = R['gaperr']
    m1 = eps_e + v_brk*eps_v/a_b + eps_v**2/(2*a_b)
    marker = [m1] + [m1 + float(np.sum(gaperr[:i])) for i in range(1, n)]
    align = marker[-1]
    conds['C10'] = _cond(disp <= cfg['sync_tol'], cfg['sync_tol'] - disp, disp,
                         cfg['sync_tol'], '<=', unit='s')
    conds['C11'] = _cond(align <= cfg['align_tol'], cfg['align_tol'] - align,
                         align, cfg['align_tol'], '<=', unit='m')
    ge = float(np.max(gaperr))
    conds['C12'] = _cond(ge <= cfg['gap_tol'], cfg['gap_tol'] - ge, ge,
                         cfg['gap_tol'], '<=', unit='m')
    epsb_max = float(np.max(epsb[1:]))
    T_f_bar = T_xi + (v_brk + eps_v)/a_b + (n - 1)*epsb_max/a_b
    sched = eps_v/a_b + disp

    # ------------------------------------------ report-only: nonpositive
    s_np_pre = float(np.min(a_crawl - Qp[1:]))
    s_np_post = float(np.min(a_b - Fcum))
    s_np = min(snp_pre, s_np_pre, s_np_post)
    conds_np = _cond(s_np >= 0, s_np, None, 0.0, '<=', certifying=False,
                     unit='m/s^2', before_T_xi=snp_pre,
                     before_T_xi_at=dict(unit=inp + 1, t=float(tg[gnp])),
                     pre_receipt=s_np_pre, post_receipt=s_np_post)

    conds = OrderedDict((k, conds[k]) for k in COND_NAMES)
    admitted = all(c['ok'] for c in conds.values())
    conds['NONPOS'] = conds_np
    failed = [k for k in COND_NAMES if not conds[k]['ok']]

    bounds = OrderedDict(
        dispersion=disp, T_f_bar=T_f_bar, marker=marker, marker_max=align,
        gap=[float(x) for x in gaperr], gap_max=ge,
        schedule_error=sched, entry_deadline=T_ent_bar,
        T_hold=None if T_ent_bar is None else T_ent_bar + (n - 1)*dbar,
        clearance_floor=[float(x) for x in clr], clearance_floor_min=min(clr),
        switch_deadline=tset.tolist(), activation_deadline=tact.tolist(),
        closure_offset_bound=[k*dbar for k in range(n)],
        ramp_receipt_bound=[T_xi + k*dbar for k in range(n)],
        u_max_before_T_xi=[float(x) for x in np.max(up_all, axis=1)],
        u_min_before_T_xi=[float(x) for x in np.min(lo_all, axis=1)],
        u_max_pre_receipt=[None] + [float(-a_crawl + Qp[u]) for u in range(1, n)],
        u_min_pre_receipt=[None] + [float(-a_crawl - Qn[u]) for u in range(1, n)],
        u_band_post_receipt=[[float(-a_b - Fcum[u]), float(-a_b + Fcum[u])]
                             for u in range(n)],
        velocity_floor=vfl, handoff_floor=vflr_c,
        admission_slack_sigma=[pairs[c + 1]['sigma'] for c in range(1, n)],
        epsb=[float(x) for x in epsb[1:]], eb=[float(x) for x in eb[1:]],
        q=[float(x) for x in R['hops']])
    consts = OrderedDict(
        a_crawl=a_crawl, v_xi=v_brk, v_in=v_in, t_0=t_0, T_xi=T_xi,
        a_r0=S.a_r0_f, f_box=f_box, U=U, Umin=Umin.tolist(),
        eps0=eps0.tolist(), e0=e0.tolist(), vt0=vt0.tolist(),
        one_sided=one_sided, A1=A1.tolist(), A2=A2.tolist(),
        P_e=Pe.tolist(), P_eps=Px.tolist(), P_f=Pf.tolist(),
        n_jumps=[0] + [len(jl[c][0]) for c in range(1, n)],
        Lam1=Lam1.tolist(), Lam2=Lam2.tolist(), taub=R['taub'].tolist(),
        tact=tact.tolist(), tset=tset.tolist(), E1=E1.tolist(),
        ebar=R['ebar'].tolist(), sup_e=R['sup_e'].tolist(),
        sup_x=R['sup_x'].tolist(), sup_f=R['sup_f'].tolist(),
        tv_f=R['tv_f'].tolist(), x2tot=x2tot.tolist(),
        e2bar=e2bar.tolist(), fbar=fbar.tolist(), tvS2=R['tvS2'].tolist(),
        Tk=R['Tk'].tolist(), fsw=R['fsw'].tolist(), F=Fv.tolist(),
        W=W[1:].tolist(), Wphi=Wphi[1:].tolist(), epsb=epsb[1:].tolist(),
        eb=eb[1:].tolist(), fb=fb[1:].tolist(), Theta=R['Th'][1:].tolist(),
        Fb_units=Fcum.tolist(), f_xi=float(R['f_xi']), anet=anet.tolist(),
        q=R['hops'].tolist(), T_brk=float(R['T_brk']), Leps=Leps,
        Qm=Qm.tolist(), P_pre=Ppre.tolist(), Q_plus=[None] + Qp[1:].tolist(),
        Q_minus=[None] + Qn[1:].tolist(), fpre_plus=fpre_p.tolist(),
        fpre_minus=fpre_m.tolist(),
        Tent_lag=[float(x) for x in Tent_lag], Tent_abs=[float(x) for x in Tent_abs],
        T_ent_bar=T_ent_bar, entry_planned_part=entry_planned,
        Vhnd=None if Vhnd is None else Vhnd[1:].tolist(),
        Vhnd_max=None if Vhnd is None else float(np.max(Vhnd)),
        F_plus_0=[float(Fp_all[i][0]) for i in range(n)],
        F_minus_0=[float(Fm_all[i][0]) for i in range(n)],
        pairs=pairs, NUMPAD=NUMPAD, VHND_PAD=float(G['VHND_PAD']),
        cell_dt=Dg, n_cells=NGc)
    res.update(conds=conds, admitted=bool(admitted), failed=failed,
               bounds=bounds, consts=consts)
    if keep_curves:
        res['curves'] = dict(tg=tg, pb=pb, Fp=Fp_all, Fm=Fm_all, Pp=Pp_all,
                             Pm=Pm_all, up=up_all, lo=lo_all, Ek=np.array(Ek),
                             vrmin=vrmin, V=V, R=R)
    timing['total_s'] = time.time() - t_start
    res['timing'] = timing
    return res


def summary(res):
    b = res.get('bounds', {})
    return OrderedDict(
        admitted=res['admitted'], failed=res['failed'],
        slacks=OrderedDict((k, c['slack']) for k, c in res['conds'].items()),
        dispersion=b.get('dispersion'), T_f_bar=b.get('T_f_bar'),
        marker=b.get('marker_max'), gap=b.get('gap_max'),
        entry_deadline=b.get('entry_deadline'),
        clearance_floor=b.get('clearance_floor_min'))


# =========================================================== age boundaries
def _ev_factory(cfg, memo):
    S = setup(cfg)

    def ev(k):
        if k not in memo:
            memo[k] = evaluate(dict(cfg, dbar=k * 1e-3), S=S)
        return memo[k]
    return ev


def age_boundary_ms(cfg, kmax=200, memo=None, verify_grid=None, verbose=False):
    """Largest k with the certificate admitted at dbar = k ms (bisection on
    the 1 ms grid, justified by the monotonicity of theory Sec. 12), with
    the failing conditions at k + 1, and a brute-force check that no age of
    the coarse grid verify_grid below the boundary is rejected."""
    cfg = config(**cfg) if 'U' not in cfg else cfg
    memo = {} if memo is None else memo
    ev = _ev_factory(cfg, memo)
    out = OrderedDict()
    if not ev(1)['admitted']:
        out.update(k_ms=0, failed_next=ev(1)['failed'])
        return out
    lo, hi = 1, kmax
    if ev(hi)['admitted']:
        out.update(k_ms=hi, note='admitted at kmax')
        lo = hi
    else:
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ev(mid)['admitted']:
                lo = mid
            else:
                hi = mid
        r0 = ev(lo); r1 = ev(lo + 1)
        assert r0['admitted'] and not r1['admitted']
        out.update(k_ms=lo, failed_next=r1['failed'],
                   slacks_at_k={nm: r0['conds'][nm]['slack'] for nm in COND_NAMES},
                   slacks_at_k1={nm: r1['conds'][nm]['slack'] for nm in COND_NAMES})
    grid = verify_grid if verify_grid is not None else \
        sorted(set([1, 2, 3, 5] + list(range(10, lo, 10)) + [max(1, lo - 1)]))
    bad = [k for k in grid if k <= lo and not ev(k)['admitted']]
    out['coarse_below'] = [k for k in grid if k <= lo]
    out['coarse_below_all_admitted'] = len(bad) == 0
    out['coarse_below_rejected'] = bad
    if verbose:
        print('  boundary %s ms, next fails %s, coarse check %s'
              % (out['k_ms'], out.get('failed_next'),
                 out['coarse_below_all_admitted']), flush=True)
    return out


def cond_boundaries_ms(cfg, kmax=200, memo=None, names=None):
    """Per-condition largest k (ms) at which the condition holds (bisection
    on the 1 ms grid for each condition separately)."""
    cfg = config(**cfg) if 'U' not in cfg else cfg
    memo = {} if memo is None else memo
    ev = _ev_factory(cfg, memo)
    out = OrderedDict()
    for nm in (names or COND_NAMES + REPORT_ONLY):
        def ok(k):
            return bool(ev(k)['conds'][nm]['ok'])
        if not ok(1):
            out[nm] = dict(k_ms=0); continue
        if ok(kmax):
            out[nm] = dict(k_ms=kmax, note='holds at kmax'); continue
        lo, hi = 1, kmax
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        out[nm] = dict(k_ms=lo, slack_at_k=ev(lo)['conds'][nm]['slack'],
                       slack_at_k1=ev(lo + 1)['conds'][nm]['slack'])
    return out


# OT condition families of b2_results.json and their HR rows
OT_GROUPS = OrderedDict(
    sync=('C10',), align=('C11',), gap=('C12',), adm=('C8a', 'C8b', 'C8c'),
    auth=('C7a', 'C7b', 'C7c_brk_lo', 'C7c_brk_up', 'C7c_pre_lo',
          'C7c_pre_up'),
    clr=('C9a',), event=('C1',), vel=('C4',), ramp=('H8',), hold=('C3',),
    hnd=('C5',))


def age_ceilings(cfg, groups=None, lo=0.002, hi=0.45, tol=2e-4, round_mid=6):
    """Per-family feasible endpoints by bisection in the age bound, the
    procedure of run_b2.part_cert / certify_profile.age_ceilings (lo = 2 ms,
    mid rounded to 6 decimals); a family holds when all its rows hold."""
    cfg = config(**cfg) if 'U' not in cfg else cfg
    groups = groups or OT_GROUPS
    S = setup(cfg)
    memo = {}

    def ev(d):
        if d not in memo:
            memo[d] = evaluate(dict(cfg, dbar=d), S=S)
        return memo[d]

    def ok(nm, d):
        r = ev(d)
        if not r.get('closed', True):
            return False
        return all(r['conds'][c]['ok'] for c in groups[nm])

    out = OrderedDict()
    for nm in groups:
        a, bnd = lo, hi
        if not ok(nm, a):
            out[nm] = 0.0; continue
        if ok(nm, bnd):
            out[nm] = bnd; continue
        while bnd - a > tol:
            mid = 0.5 * (a + bnd)
            if round_mid is not None:
                mid = round(mid, round_mid)
            if ok(nm, mid):
                a = mid
            else:
                bnd = mid
        out[nm] = a
    return out


# ================================================================ regression
def caseB_config(dbar=None):
    """Degenerate HR configuration that reduces to OT (task (a)): constant
    reference before T_xi (a_r = 0, no onset jumps), no schedules (rho = 0;
    the desired clearance equals the parking gap from the start, so
    e_i(0) = c_i(0) - d_{s,i} and the gaps are closed by the S2 feedback as
    in OT), the Case B takeover state, gaps and constants
    (certify_profile.config()), and the frozen input set U = U^- = 1.6
    (U_i^- = min{U^-, a^max_i} = min{U, a^max_i}).  The one-sided box is
    not available (the Case B followers take over slower than their
    predecessors, (C2) fails), so the symmetric box is used, as in the
    constant-plan regression (R3) of certify_plan."""
    cB = cp.config()
    n = int(cB['n'])
    ell = F(FIXED['ell'])
    d_s = [F(x) for x in cB['d_s']]
    acc = Fr(0)
    s0 = []
    for i in range(n):
        acc += d_s[i] + ell
        s0.append(-acc)
    v_xi = F(str(cB['v_xi']))
    v0 = []
    vp = v_xi
    for x in cB['eps0']:
        vp = vp - F(x)
        v0.append(vp)
    c0 = [F(x) for x in cB['c0']]
    e0 = [F(cB['e0_head'])] + [c0[i - 1] - d_s[i] for i in range(1, n)]
    return config(n=n, amax=[float(x) for x in cB['amax']], a_b=cB['a_b'],
                  alpha=cB['alpha'], lam=cB['lam'], s_m=cB['s_m'],
                  eps_det=cB['eps_det'], phi=cB['phi'], T_c=cB['T_c'],
                  dbar=cB['dbar'] if dbar is None else dbar,
                  eps_e=cB['eps_e'], eps_v=cB['eps_v'], vhnd=cB['vhnd'],
                  d_s=[float(x) for x in cB['d_s']],
                  c0=[float(x) for x in cB['c0']],
                  v0=[str(x) for x in v0], e0=[str(x) for x in e0],
                  U=1.6, Uminus=1.6, k=1.6, one_sided=False,
                  sync_tol=cB['sync_tol'], align_tol=cB['align_tol'],
                  gap_tol=cB['gap_tol'], numpad=0.01,
                  plan=dict(family='constant', v_xi=str(cB['v_xi']),
                            T_xi=str(cB['T_xi']),
                            s0=[str(x) for x in s0]))


def _bitcmp(a, b, path, bad):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if a.shape != b.shape or not np.array_equal(a, b):
        bad.append(path)


def regress_R1_R2(verbose=True):
    """(R1) chain() with no planned parts (A1 = A2 = P = 0), ramp = a_b and
    v_brk = v^xi is bit-identical to the frozen forward pass
    (certify_v30.forward_pass) for Case B, Case A and the Case D constants;
    (R2) entry_curves() with A2 = A^tail = 0 is bit-identical to the frozen
    tail recursion (certify_profile._entry with ff=None, = certified_entry)."""
    out = OrderedDict()
    cases = OrderedDict()
    cfgB = cp.config()
    GB = cp.namespace(cfgB)
    geoB = cp.geometry(GB, cfgB)
    cases['caseB'] = (GB, geoB['e0'], np.array(GB['eps0']), float(cfgB['dbar']))
    A = json.load(open(os.path.join(HERE, 'v30_values.json')))
    cfgA = cp.config(v_xi=10.0, a_b=0.70, alpha=0.30, lam=0.075, s_m=3.0,
                     T_xi=180.0, dbar=0.06, eps_e=0.20, eps_v=0.02,
                     eps0=[-0.05, 0.025, 0.025, 0.025, 0.025],
                     e0=[-0.5, 40.0, 40.0, 40.0, 40.0], c0=None,
                     d_s=[0.0] + list(A['d_s']), sync_tol=1.0,
                     align_tol=3.0, gap_tol=1.0, vhnd=1e-3)
    GA = cp.namespace(cfgA)
    geoA = cp.geometry(GA, cfgA)
    cases['caseA'] = (GA, geoA['e0'], np.array(GA['eps0']), float(cfgA['dbar']))
    cfgD = cq.config()
    GD = cq.namespace(cfgD)
    cases['caseD_constants'] = (GD, np.zeros(5), np.array(cq.eps0_of(cfgD)),
                                float(cfgD['dbar']))
    for name, (G, e0, eps0, dbar) in cases.items():
        n = G['n']
        e_save = np.array(G['e0'], float).copy()
        G['e0'][:] = e0
        Rf = G['forward_pass'](dbar)
        G['e0'][:] = e_save
        z = np.zeros(n)
        Rc = chain(G, e0, eps0, dbar, z, z, z, z, z, ramp=G['a_b'],
                   v_brk=G['v_xi'])
        if Rf is None:
            out[name] = dict(frozen_closed=False, copy_closed=Rc['closed'])
            continue
        bad = []
        for k in FIELDS_R1:
            _bitcmp(Rf[k], Rc[k], name + '/' + k, bad)
        Tf_, Vf_, _ = cp._entry(G, dict(Rf), Rf['Lam2'], dbar, ff=None,
                                lags=cp.LAG_FROZEN)
        ec = entry_curves(G, Rc, Rc['Lam2'], z, [np.zeros(G['NT'])] * n,
                          dbar, cp.LAG_FROZEN)
        bad_e = []
        TGRID = G['TGRID']
        Tent_c = np.zeros(n)
        for i in range(n):
            ok = (ec['totE'][i] <= G['eps_e']) & (ec['totX'][i] <= G['eps_v'])
            Tent_c[i] = TGRID[int(np.argmax(ok))] if ok.any() else np.inf
        _bitcmp(Tf_, Tent_c, name + '/Tent', bad_e)
        for j in range(len(Vf_)):
            _bitcmp(Vf_[j], ec['V'][j], name + '/V%d' % j, bad_e)
        out[name] = dict(fields=len(FIELDS_R1), bit_identical=len(bad) == 0,
                         mismatches=bad[:5], entry_bit_identical=len(bad_e) == 0,
                         entry_mismatches=bad_e[:5])
        if verbose:
            print('(R1/R2) %-16s chain %s, entry recursion %s'
                  % (name, 'BIT-IDENTICAL' if not bad else 'MISMATCH %s' % bad[:3],
                     'BIT-IDENTICAL' if not bad_e else 'MISMATCH %s' % bad_e[:3]),
                  flush=True)
    out['ok'] = all(v.get('bit_identical') and v.get('entry_bit_identical')
                    for v in out.values() if isinstance(v, dict))
    return out


# reasons for the differences between the degenerate HR certificate and the
# frozen Case B certificate (fields of b2_results.json 'cert')
CASEB_REASONS = OrderedDict(
    margins='admission: the HR slacks are the PT forms (C8a-c) with the '
            'planned residual mu* = kappa h* of the constant plan; they '
            'equal h(0) - thr_j - NUMPAD branch by branch up to rounding, and '
            'the reported margin is the minimum of the three',
    gmin='clearance floor: HR takes the larger of the direct floor of theory '
         'Sec. 10 (g = (d_i - s_m) + e_i with the magnitude bounds of e_i) and '
         'the barrier-based floor of OT (Corollary 1); with the Case B '
         'closures of 37-44 m by feedback the direct floor is negative and '
         'the barrier-based floor is the OT floor up to rounding',
    auth_slack='authority: HR tests the sign-resolved envelopes P^+- + F^+- '
               'per 10 ms cell against [-U_i^-, U] and the braking and '
               'pre-receipt branches separately; OT tests one magnitude '
               'demand max{F_i + max(alpha, fbar_i), a_b + F^b_i, P^pre_i, '
               '(a_b + F^b_{i-1}) + fbar_i} <= U_i^-; the reported HR slack '
               'is the smallest slack of the C7 rows of the unit',
    vfloor='velocity floor: HR uses the time-resolved cell floor '
           'v_r(t) - sum_k E^vel_k(t) (PT) instead of the constant OT floor '
           'v^xi - max_i sum_k max{M1_k, eps2_k}',
    T_hold='hold: HR tests T_ent_bar + (n-1) dbar <= T_xi (the last age '
           'windows of every layer, theory Sec. 8.3) instead of + dbar',
    Tent_cert='entry deadlines: HR takes the candidates on the fixed 1 ms '
              'grid in [tswb_i, T_xi] (supplement S.II-E, needed for the '
              'monotonicity in dbar) instead of the 1 ms lag grid anchored at '
              'tswb_i; the stored lag is Tent_i - tswb_i, which differs from '
              'the frozen grid lag by less than one grid step',
    Tent_bar='largest entry deadline: a point of the fixed 1 ms grid (see '
             'Tent_cert); it differs from the frozen value by less than one '
             'grid step',
    ramp='(H8): HR tests the handoff floor v^xi - n eps_v (CT) instead of the '
         'acquisition floor of OT',
)


# fields and age ceilings of the degenerate reduction that differ from the
# frozen Case B record by construction (formulas of the HR certificate);
# every other field must be equal to 1e-12 and every pass flag identical
EXPECTED_FIELD_DIFFS = {'auth_slack', 'vfloor', 'entry/T_hold', 'entry/Tent_cert',
                        'entry/Tent_bar'}
EXPECTED_CEIL_DIFFS = {'auth', 'hold'}
CASEB_REASONS['ceiling_auth'] = (
    'authority ceiling: the sign-resolved HR tests are less conservative than '
    'the magnitude demand of OT (S1 level and the pre-ramp-negative candidate '
    'enter only with their sign), so the authority family holds up to a '
    'larger age bound')
CASEB_REASONS['ceiling_hold'] = (
    'hold ceiling: T_ent_bar(dbar) + (n-1) dbar <= T_xi is stricter than '
    'T_ent_bar(dbar) + dbar <= T_xi')


def regress_caseB(verbose=True, ceilings=True):
    """(R3) the degenerate HR configuration (caseB_config) against the frozen
    Case B certificate of b2_results.json 'cert': every field and pass flag
    is compared; differing fields are listed with their reason."""
    B = json.load(open(B2_RESULTS))['cert']
    cfg = caseB_config()
    r = evaluate(cfg)
    K = r['consts']; bnd = r['bounds']
    n = int(cfg['n'])
    auth_units = []
    for u in range(n):
        s = [r['conds']['C7a']['slack'], r['conds']['C7b']['slack']]
        s.append(r['conds']['C7c_brk_lo']['per_unit'][u])
        s.append(r['conds']['C7c_brk_up']['per_unit'][u])
        if u >= 1:
            s.append(r['conds']['C7c_pre_lo']['per_unit'][u - 1])
            s.append(r['conds']['C7c_pre_up']['per_unit'][u - 1])
        auth_units.append(min(s))
    margins = [min(K['pairs'][c]['slack_a'], K['pairs'][c]['slack_b'],
                   K['pairs'][c]['slack_c']) for c in range(2, n + 1)]
    got = OrderedDict(
        disp=bnd['dispersion'], late=None, sched=bnd['schedule_error'],
        align=bnd['marker_max'], gaperr=bnd['gap'], T_f=bnd['T_f_bar'],
        T_brk=K['T_brk'], taub=K['taub'], tset=K['tset'], E1=K['E1'],
        epsb=K['epsb'], anet=K['anet'], hops=K['q'], Leps=K['Leps'],
        margins=margins, gmin=bnd['clearance_floor_min'],
        auth_slack=auth_units, vfloor=bnd['velocity_floor'],
        h0=[K['pairs'][c]['h0'] for c in range(2, n + 1)],
        e0=K['e0'])
    # late (directional bound) from the chain
    got['late'] = float(np.sum(np.array(K['epsb']) / float(cfg['a_b'])))
    ent = dict(Tent_cert=K['Tent_lag'], Tent_bar=K['T_ent_bar'],
               T_hold=bnd['T_hold'], hold_ok=r['conds']['C3']['ok'],
               Vhnd_max=K['Vhnd_max'], hnd_ok=r['conds']['C5']['ok'])
    rows = []
    for k, v in got.items():
        if k not in B:
            continue
        ref = B[k]
        a = np.atleast_1d(np.asarray(v, float))
        bb = np.atleast_1d(np.asarray(ref, float))
        note = None
        if a.shape != bb.shape and a.ndim == 1 and len(bb) == len(a) - 1:
            # the archived record stores this field from its second entry on
            # (gaperr: pairs 3..n; epsb: the frozen array without index 0)
            a = a[1:]
            note = 'archived record stores entries 2..; compared on them'
        same = a.shape == bb.shape and np.array_equal(a, bb)
        close = a.shape == bb.shape and bool(np.allclose(a, bb, rtol=1e-12, atol=1e-12))
        rows.append(OrderedDict(field=k, frozen=ref, hr=v, bit_identical=bool(same),
                                equal_1e12=close,
                                max_abs_diff=float(np.max(np.abs(a - bb)))
                                if a.shape == bb.shape else None,
                                note=note,
                                reason=None if close else CASEB_REASONS.get(k)))
    for k, v in ent.items():
        ref = B['entry'][k]
        if isinstance(ref, bool):
            same = ref == v
            rows.append(OrderedDict(field='entry/' + k, frozen=ref, hr=v,
                                    bit_identical=same, equal_1e12=same,
                                    reason=None if same else CASEB_REASONS.get(k)))
        else:
            a = np.atleast_1d(np.asarray(v, float)); bb = np.atleast_1d(np.asarray(ref, float))
            same = a.shape == bb.shape and np.array_equal(a, bb)
            close = a.shape == bb.shape and bool(np.allclose(a, bb, rtol=1e-12, atol=1e-12))
            rows.append(OrderedDict(field='entry/' + k, frozen=ref, hr=v,
                                    bit_identical=bool(same), equal_1e12=close,
                                    max_abs_diff=float(np.max(np.abs(a - bb)))
                                    if a.shape == bb.shape else None,
                                    reason=None if close else CASEB_REASONS.get(k)))
    flags = OrderedDict()
    for nm, grp in OT_GROUPS.items():
        hr_ok = all(r['conds'][c]['ok'] for c in grp)
        flags[nm] = dict(frozen=B['conds'][nm], hr=hr_ok,
                         same=bool(B['conds'][nm] == hr_ok))
    out = OrderedDict(admitted=r['admitted'], failed=r['failed'], rows=rows,
                      flags=flags)
    diff_fields = sorted(row['field'] for row in rows if not row['equal_1e12'])
    out['differing_fields'] = diff_fields
    out['expected_differences'] = sorted(EXPECTED_FIELD_DIFFS)
    ok = (set(diff_fields) <= EXPECTED_FIELD_DIFFS
          and all(v['same'] for v in flags.values()) and r['admitted'])
    if ceilings:
        t0 = time.time()
        ce = age_ceilings(cfg)
        out['ceilings'] = OrderedDict(
            (k, dict(frozen=B['ceils'][k], hr=ce[k],
                     same=bool(abs(B['ceils'][k] - ce[k]) < 1e-12)))
            for k in ce)
        out['ceilings_runtime_s'] = time.time() - t0
        diff_ce = sorted(k for k, v in out['ceilings'].items() if not v['same'])
        out['differing_ceilings'] = diff_ce
        ok = ok and set(diff_ce) <= EXPECTED_CEIL_DIFFS
    out['ok'] = bool(ok)
    if verbose:
        print('(R3) Case B degenerate HR: admitted %s (failed %s)'
              % (r['admitted'], r['failed']), flush=True)
        for row in rows:
            print('   %-18s %s%s' % (row['field'],
                                     'BIT-IDENTICAL' if row['bit_identical'] else
                                     ('equal to 1e-12' if row['equal_1e12'] else
                                      'DIFFERS (max |diff| %s)' % row.get('max_abs_diff')),
                                     '' if row['equal_1e12'] else ' -- ' + str(row['reason'])))
        print('   pass flags:', {k: (v['frozen'], v['hr']) for k, v in flags.items()})
        if ceilings:
            print('   ceilings:', {k: (v['frozen'], v['hr']) for k, v in out['ceilings'].items()})
    return out


def _He(lam, r):
    r = np.maximum(r, 0.0)
    return (1.0 - (1.0 + lam * r) * np.exp(-lam * r)) / lam ** 2


def _Hx(lam, r):
    r = np.maximum(r, 0.0)
    return r * np.exp(-lam * r)


def _Hf(lam, r):
    r = np.maximum(r, 0.0)
    return (lam * r - 1.0) * np.exp(-lam * r) + 1.0


def regress_pulses(cfg=None, n_real=40, n_t=3000, seed=20260928, verbose=True):
    """(R4) Lemma 7.1 and its cell-extended evaluation against the exact
    forced response of random and adversarial pulse placements: every jump J
    of a pair gets a pulse [s, s + l) with l in [0, dbar] inside
    K_J = [lo_J, lo_J + (i-1) dbar]; the responses e^p, eps^p, f^p are the
    closed-form integrals of the impulse responses; at random instants the
    response must lie in the bounds of the cell that contains the instant.
    Adversarial placements put every pulse at the end of K_J that maximizes
    (or minimizes) the signed response at a target instant."""
    cfg = cfg or config(lam=0.10)
    S = setup(cfg)
    lam = float(cfg['lam']); dbar = float(cfg['dbar'])
    Dg = float(cfg['cell_dt'])
    T_xi = S.T_xi_f
    NGc = int(math.ceil(T_xi / Dg - 1e-9))
    tg = np.arange(NGc) * Dg; tg1 = (np.arange(NGc) + 1) * Dg
    rng = np.random.default_rng(seed)
    worst = dict(e=-np.inf, x=-np.inf, f=-np.inf)
    ok = True
    tight = dict(e=0.0, x=0.0, f=0.0)
    for c in range(1, S.n):
        h, lo = S.pair_jumps(c)
        hi = lo + c * dbar
        pb = pulse_bounds(lam, dbar, h, lo, hi, tg, tg1)
        ts = np.sort(rng.uniform(S.t_0_f, T_xi, n_t))
        g = np.minimum((ts / Dg).astype(int), NGc - 1)
        for rr in range(n_real + 6):
            if rr < n_real:
                ell = rng.uniform(0.0, dbar, len(h))
                s = lo + rng.uniform(0.0, 1.0, len(h)) * (hi - lo - ell)
            else:
                # adversarial: full-length pulses at one end of K_J, chosen
                # by the sign of h_J (both combinations) or at random ends
                ell = np.full(len(h), dbar)
                mode = rr - n_real
                if mode == 0:
                    s = np.where(h > 0, hi - dbar, lo)
                elif mode == 1:
                    s = np.where(h > 0, lo, hi - dbar)
                elif mode == 2:
                    s = lo.copy()
                elif mode == 3:
                    s = hi - dbar
                else:
                    s = np.where(rng.uniform(size=len(h)) < 0.5, lo, hi - dbar)
            s1 = s[None, :]; s2 = (s + ell)[None, :]
            T = ts[:, None]
            # integral of h(t - sigma) over [s1, min(s2, t)]
            up = np.minimum(s2, T)
            act = up > s1
            ev_ = (_He(lam, T - s1) - _He(lam, T - up)) * act
            xv_ = (_Hx(lam, T - s1) - _Hx(lam, T - up)) * act
            fv_ = (_Hf(lam, T - s1) - _Hf(lam, T - up)) * act
            ep = ev_ @ h; xp = xv_ @ h; fp = fv_ @ h
            for key, val, lo_b, hi_b in (('e', ep, pb['e_lo'][g], pb['e_hi'][g]),
                                         ('x', xp, pb['x_lo'][g], pb['x_hi'][g]),
                                         ('f', fp, pb['f_lo'][g], pb['f_hi'][g])):
                exc = float(np.max(np.maximum(val - hi_b, lo_b - val)))
                worst[key] = max(worst[key], exc)
                if exc > 1e-12:
                    ok = False
                rel = np.max(np.abs(val) / np.maximum(np.maximum(hi_b, -lo_b), 1e-15))
                tight[key] = max(tight[key], float(rel))
    out = OrderedDict(ok=bool(ok), max_excess_over_bound=worst,
                      max_attained_fraction_of_bound=tight,
                      realizations=n_real + 6, instants_per_pair=n_t,
                      lam=lam, dbar=dbar)
    if verbose:
        print('(R4) planned-pulse bounds: never exceeded %s; largest excess %s; '
              'largest attained fraction %s' % (ok, worst, tight), flush=True)
    return out


def regress_plan_bounds(cfg=None, n_s=4000, seed=7, verbose=True):
    """(R5) (a) PW.ext against dense sampling; (b) the lower bounds of h*,
    mu*, g* over the closure offsets (theory Sec. 10) against the
    realization plan evaluated at random instants and random offsets
    o_k in [0, (k-1) dbar]; (c) the bound Q_{i-1} on |p_{i-1}(theta_i(t)) -
    a*_{i-1}(t)| against random composed arguments."""
    cfg = cfg or config(lam=0.10)
    S = setup(cfg)
    dbar = float(cfg['dbar']); n = S.n
    rng = np.random.default_rng(seed)
    b = np.array([float(a) for a in cfg['amax']]) / FIXED['v_c']
    Pi = np.array([0.0] + [1.0 / b[i - 1] - 1.0 / b[i] for i in range(1, n)])
    kap = FIXED['kappa']; s_m = float(cfg['s_m'])
    t_0 = S.t_0_f; T_xi = S.T_xi_f
    out = OrderedDict()
    # (a) PW.ext
    worst_a = 0.0
    for pw in [S.arA, S.vr, S.arkvr] + S.A_pw[1:] + S.B_pw[1:] + S.rho_pw[1:]:
        a = rng.uniform(-1.0, T_xi, 300)
        w = rng.uniform(0.0, 0.5, 300)
        bb = a + w
        mx = pw.ext(a, bb, 'max'); mn = pw.ext(a, bb, 'min')
        for j in range(300):
            xs = np.linspace(a[j], bb[j], 2001)
            v = pw(xs)
            worst_a = max(worst_a, float(np.max(v) - mx[j]), float(mn[j] - np.min(v)))
    out['a_pw_ext_worst'] = worst_a
    # (b) realization plan: h*, mu*, g* at random t and offsets
    Dg = float(cfg['cell_dt'])
    worst_b = dict(h=-np.inf, mu=-np.inf, g=-np.inf)
    for c in range(1, n):
        ts = rng.uniform(0.0, T_xi, n_s)
        g = (ts / Dg).astype(int)
        ta = g * Dg; tb = np.minimum((g + 1) * Dg, T_xi)
        hl, ml, gl = plan_lower_bounds(S, c, ta, tb, dbar, Pi[c])
        o = [rng.uniform(0.0, k * dbar, n_s) for k in range(n)]  # o[k]: code unit k
        x_own = ts - t_0 - o[c]
        d_own = S.gs_pw[c](x_own) + s_m
        dd_own = -S.negdd_pw[c](x_own)
        rho_own = S.rho_pw[c](x_own)
        V = S.vr(ts); ar = S.arA(ts)
        for u in range(1, c):
            xu = ts - t_0 - o[u]
            V = V + S.negdd_pw[u](xu)
            ar = ar + S.rho_pw[u](xu)
        hstar = d_own - s_m + dd_own / b[c] + Pi[c] * V
        mustar = dd_own - rho_own / b[c] + Pi[c] * ar + kap * hstar
        gstar = d_own - s_m
        worst_b['h'] = max(worst_b['h'], float(np.max(hl - hstar)))
        worst_b['mu'] = max(worst_b['mu'], float(np.max(ml - mustar)))
        worst_b['g'] = max(worst_b['g'], float(np.max(gl - gstar)))
    out['b_bound_minus_sample_max'] = worst_b
    # (c) Q bound
    worst_c = -np.inf
    for c in range(2, n):
        ts = rng.uniform(t_0, T_xi, n_s)
        g = (ts / Dg).astype(int)
        ta = g * Dg; tb = np.minimum((g + 1) * Dg, T_xi)
        Q = q_bound(S, c, ta, tb, dbar)
        s_ar = ts - rng.uniform(0.0, c * dbar, n_s)
        diff = np.abs(S.arA(s_ar) - S.arA(ts))
        tot = diff.copy()
        for u in range(1, c):
            o_u = rng.uniform(0.0, u * dbar, n_s)
            arg = ts - rng.uniform(0.0, (c - u) * dbar, n_s) - t_0 - o_u
            tot = tot + np.abs(S.rho_pw[u](arg) - S.rho_pw[u](ts - t_0 - o_u))
        worst_c = max(worst_c, float(np.max(tot - Q)))
    out['c_sample_minus_Q_max'] = worst_c
    out['ok'] = bool(worst_a <= 1e-12 and all(v <= 1e-9 for v in worst_b.values())
                     and worst_c <= 1e-12)
    if verbose:
        print('(R5) PW.ext worst %.2e; plan bounds (bound - sample) max %s; Q (sample - '
              'bound) max %.2e; ok %s' % (worst_a, worst_b, worst_c, out['ok']),
              flush=True)
    return out


def regress_setup(verbose=True):
    """(R6) exactness of the reference and the schedules of the round-6
    design: Assumption S (exact closure int rho = 0, int int rho = C_i,
    dd <= 0), consistency of d_i, dd_i with the plan at every breakpoint,
    the sum of the planned jumps of each schedule is zero, the reference
    jumps sum to a_r(T_xi^-) - a_r(0), and the markers are those of the
    plan."""
    cfg = config()
    S = setup(cfg)
    out = OrderedDict(prem_R=S.prem_R, prem_S=S.prem_S)
    sums = [str(sum(h for (x, h) in S.sched_jumps[u])) for u in range(1, S.n)]
    out['schedule_jump_sums'] = sums
    out['reference_jump_sum_ok'] = bool(sum(h for (t, h) in S.ref_jumps)
                                        == S.ref_a[-1] - S.ref_a[0])
    out['n_ref_jumps'] = len(S.ref_jumps)
    out['n_sched_jumps'] = [len(S.sched_jumps[u]) for u in range(1, S.n)]
    out['x_f'] = [float(S.sched[u]['x_f']) for u in range(1, S.n)]
    out['a_crawl'] = str(S.a_crawl)
    out['v_xi'] = str(S.v_xi_h)
    out['t_0'] = str(S.t_0)
    out['ok'] = bool(S.prem_R['ok'] and S.prem_S['ok']
                     and all(x == '0' for x in sums)
                     and out['reference_jump_sum_ok'])
    if verbose:
        print('(R6) setup: Assumption R %s, Assumption S %s, schedule jump sums %s, '
              'ok %s' % (S.prem_R['ok'], S.prem_S['ok'], sums, out['ok']), flush=True)
    return out


def regress(verbose=True, ceilings=True):
    """(R1)-(R6)."""
    out = OrderedDict()
    t0 = time.time()
    out['R1_R2'] = regress_R1_R2(verbose=verbose)
    out['R3_caseB'] = regress_caseB(verbose=verbose, ceilings=ceilings)
    out['R4_pulses'] = regress_pulses(verbose=verbose)
    out['R5_plan_bounds'] = regress_plan_bounds(verbose=verbose)
    out['R6_setup'] = regress_setup(verbose=verbose)
    out['ok'] = bool(out['R1_R2']['ok'] and out['R3_caseB']['ok']
                     and out['R4_pulses']['ok']
                     and out['R5_plan_bounds']['ok'] and out['R6_setup']['ok'])
    out['runtime_s'] = time.time() - t0
    if verbose:
        print('regression ok (R1-R6; R3: only the expected fields %s and '
              'ceilings %s differ): %s' % (sorted(EXPECTED_FIELD_DIFFS),
                                           sorted(EXPECTED_CEIL_DIFFS), out['ok']),
              flush=True)
    return out


if __name__ == '__main__':
    args = sys.argv[1:] or ['eval']
    for a in args:
        if a == 'eval':
            r = evaluate(config())
            print(json.dumps(cp._py(summary(r)), indent=1))
        elif a == 'regress':
            rep = regress()
            print(json.dumps(cp._py(dict(ok=rep['ok'])), indent=1))
        else:
            raise SystemExit('unknown part %s' % a)
