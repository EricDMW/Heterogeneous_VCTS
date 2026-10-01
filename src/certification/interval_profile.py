#!/usr/bin/env python3
"""interval_profile.py -- outward-interval enclosure of the closed-form
conditions of certify_profile.evaluate (Assumption 1', Case C).

Method (as witness_core_v7.py): two-sided intervals of IEEE doubles with
one-ulp outward rounding of every elementary operation; exponentials
from python-flint Arb balls (120 bits) stepped one ulp outward; sup, TV
and L1 of the modal family (p + q t) e^{-lam t} by branch-union upper
bounds.  Unlike witness_core_v7.py, every derived constant (Euler's
number, 1/(lam e), 2 gamma + 2 lam e^{-3}, the handoff-box peaks and
L1 norms) is itself an interval.

Scope (interval-verified): the acquisition chain to the switch
deadlines, the S2-window chain with the feedforward charge, the braking
recursion, the budgets, thresholds and admission margins (H4), the
clearance floors (H5), the authority demands (H1), the headroom (H2),
the sampled-event margin (H7), the pre-switch velocity floor and the
time-resolved velocity floor on a rigorous cell partition of
[T_sw_bar, T_xi] (H6), ramp-before-stop (H8), the dispersion, schedule,
alignment and gap bounds (C_sync, C_al, C_gap), and the premise P_seg.
Out of scope (grid-evaluated in floating point, as in the archived
witness): H3 and C_hnd (entry recursion).

Refinements (cfg['refine'], 2026-09-26): (R1) the segment demand of H1
at t = t_d from the corner tail suprema of |f_free| with interval
arguments t_d - (i-k) dbar - tset_k; (R2) the signed tail supremum of
eps_free over the asymmetric switch-state box in the H6 cells; P_seg with
the premise t_d >= T_sw_bar + (n-1) dbar.  The handoff pad is cfg['vhnd'].

API: enclose(cfg, cell=0.02) -> dict with, per condition, the interval
[lo, hi] of the slack and 'verified' = lo > 0; 'values' holds the
interval bounds of disp, sched, late, align, gap errors, T_f.
regress() checks Case B against the archived witness and the enclosure
of the floating-point evaluator.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_profile as cp                              # noqa: E402

INF = math.inf


def dn(x):
    return x if x in (0.0, -INF) else math.nextafter(x, -INF)


def up(x):
    return x if x in (0.0, INF) else math.nextafter(x, INF)


class IV:
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        if hi is None:
            hi = lo
        self.lo, self.hi = float(lo), float(hi)
        assert self.lo <= self.hi, (lo, hi)

    def __add__(s, o):
        o = _iv(o); return IV(dn(s.lo + o.lo), up(s.hi + o.hi))
    __radd__ = __add__

    def __sub__(s, o):
        o = _iv(o); return IV(dn(s.lo - o.hi), up(s.hi - o.lo))

    def __rsub__(s, o):
        return _iv(o) - s

    def __mul__(s, o):
        o = _iv(o)
        c = (s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi)
        return IV(dn(min(c)), up(max(c)))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = _iv(o)
        assert o.lo > 0.0, "interval division needs positive denominator"
        c = (s.lo / o.lo, s.lo / o.hi, s.hi / o.lo, s.hi / o.hi)
        return IV(dn(min(c)), up(max(c)))

    def __rtruediv__(s, o):
        return _iv(o) / s

    def __neg__(s):
        return IV(-s.hi, -s.lo)

    def __repr__(self):
        return 'IV(%r, %r)' % (self.lo, self.hi)


def _iv(x):
    return x if isinstance(x, IV) else IV(x)


def _exp_dn(v):
    from flint import arb, ctx
    ctx.prec = 120
    return math.nextafter(float(arb(v).exp().lower()), -INF)


def _exp_up(v):
    from flint import arb, ctx
    ctx.prec = 120
    return math.nextafter(float(arb(v).exp().upper()), INF)


def iexp(x):
    x = _iv(x); return IV(_exp_dn(x.lo), _exp_up(x.hi))


def iabs(x):
    x = _iv(x)
    lo = 0.0 if x.lo <= 0.0 <= x.hi else min(abs(x.lo), abs(x.hi))
    return IV(lo, up(max(abs(x.lo), abs(x.hi))))


def imax(*xs):
    xs = [_iv(x) for x in xs]
    return IV(max(x.lo for x in xs), max(x.hi for x in xs))


def imin(*xs):
    xs = [_iv(x) for x in xs]
    return IV(min(x.lo for x in xs), min(x.hi for x in xs))


def ipos(x):
    x = _iv(x); return IV(max(x.lo, 0.0), max(x.hi, 0.0))


def isum(xs):
    t = IV(0.0)
    for x in xs:
        t = t + x
    return t


def isqr(x):
    x = _iv(x)
    a = iabs(x)
    return IV(dn(a.lo * a.lo), up(a.hi * a.hi))


# ---------------------------------------------------------- modal family
class Modal:
    """sup/TV/L1 upper bounds of (p + q t) e^{-lam t} on [0, inf) and the
    tail sup on [R, inf), with lam an interval."""

    def __init__(self, lam):
        self.lam = _iv(lam)
        self.ilam = IV(1.0) / self.lam
        self.ilam2 = self.ilam * self.ilam

    def cf_upper(self, p, q):
        lam, ilam, ilam2 = self.lam, self.ilam, self.ilam2
        p, q = _iv(p), _iv(q)
        ap = iabs(p)
        qz = not (q.lo > 0.0 or q.hi < 0.0)
        if qz:
            gs = q * ilam
            sup_u = imax(ap, iabs(gs)).hi
            tv_u = max(ap.hi, (ap + 2.0 * iabs(gs)).hi)
            l1_u = (iabs(p) * ilam + iabs(q) * ilam2).hi
            return IV(0, sup_u), IV(0, tv_u), IV(0, l1_u)
        sq = 1.0 if q.lo > 0 else -1.0
        ts = ((q - lam * p) / iabs(q)) * (IV(sq) * ilam)
        gs = (q * ilam) * iexp(-(lam * ipos(ts)))
        if ts.hi <= 0.0:
            sup_u, tv_u = ap.hi, ap.hi
        elif ts.lo > 0.0:
            sup_u = imax(ap, iabs(gs)).hi
            tv_u = (iabs(p - gs) + iabs(gs)).hi
        else:
            sup_u = imax(ap, iabs(gs)).hi
            tv_u = max(ap.hi, (iabs(p - gs) + iabs(gs)).hi)
        I0 = p * ilam + q * ilam2
        t0 = (-p) / q if q.lo > 0.0 else p / (-q)
        It = q * ilam2 * iexp(-(lam * ipos(t0)))
        if t0.hi <= 0.0:
            l1_u = iabs(I0).hi
        elif t0.lo > 0.0:
            l1_u = (iabs(I0 - It) + iabs(It)).hi
        else:
            l1_u = max(iabs(I0).hi, (iabs(I0 - It) + iabs(It)).hi)
        return IV(0, sup_u), IV(0, tv_u), IV(0, l1_u)

    def tail_sup_signed_hi(self, p, q, R):
        """upper bound on sup_{r >= R} (p + q r) e^{-lam r}, R >= 0 (signed;
        the limit at infinity is zero, so the supremum is at least zero;
        for q > 0 the stationary point ts = (q - lam p)/(lam q) is a
        maximum of value (q/lam) e^{-lam ts}, for q < 0 a minimum)."""
        lam, ilam = self.lam, self.ilam
        p, q, R = _iv(p), _iv(q), _iv(R)
        out = max(((p + q * R) * iexp(-(lam * R))).hi, 0.0)
        if q.lo > 0.0:
            ts = ((q - lam * p) / q) * ilam
            if ts.hi > R.lo:
                tlo = max(ts.lo, R.lo)
                out = max(out, (q * ilam * iexp(-(lam * IV(tlo)))).hi)
        elif q.hi > 0.0:
            # q may vanish: (p + q r) e^{-lam r} <= |p| e^{-lam R}
            #               + q^+ sup_r r e^{-lam r}
            out = max(out, (iabs(p) * iexp(-(lam * R))
                            + ipos(q) * ilam * iexp(-1.0)).hi)
        return out

    def tail_sup_hi(self, p, q, R):
        """upper bound on sup_{r >= R} |(p + q r) e^{-lam r}|, R >= 0."""
        lam, ilam = self.lam, self.ilam
        p, q, R = _iv(p), _iv(q), _iv(R)
        gR = (iabs(p + q * R) * iexp(-(lam * R))).hi
        if q.lo > 0.0 or q.hi < 0.0:
            sq = 1.0 if q.lo > 0 else -1.0
            ts = ((q - lam * p) / iabs(q)) * (IV(sq) * ilam)
            if ts.hi > R.lo:
                tlo = max(ts.lo, R.lo)
                gs = (iabs(q) * ilam * iexp(-(lam * IV(tlo)))).hi
                return max(gR, gs)
            return gR
        # q may vanish: |g| <= (|p| + |q| r) e^{-lam r} <= |p| e^{-lam R} +
        # |q| sup r e^{-lam r}
        gs = (iabs(p) * iexp(-(lam * R)) + iabs(q) * ilam * iexp(-1.0)).hi
        return max(gR, gs)


# ---------------------------------------------------------- the chain
def enclose(cfg, cell=0.02, verbose=False):
    cfg = cp.config(**cfg)
    n = int(cfg['n'])
    seg, t_d, t_c, a_d, v_c1 = cp._segment(cfg)
    lam = IV(float(cfg['lam']))
    beta = lam * lam
    gamma = IV(2.0) * lam
    M = Modal(lam)
    ilam = M.ilam
    E = iexp(IV(1.0))                     # Euler's number
    a_b = IV(float(cfg['a_b'])); v_xi = IV(float(cfg['v_xi']))
    v_c1i = IV(v_c1); a_di = IV(a_d)
    dv = v_xi - v_c1i
    alpha = [IV(float(cfg['alpha']))] * n
    A_run = [imax(*alpha[:i + 1]) for i in range(n)]
    eps0 = [IV(float(x)) for x in cfg['eps0']]
    eps_e = IV(float(cfg['eps_e'])); eps_v = IV(float(cfg['eps_v']))
    eps_det = IV(float(cfg['eps_det'])); phi = IV(float(cfg['phi']))
    T_c = IV(float(cfg['T_c'])); kappa = IV(cp.FIXED['kappa'])
    U = IV(cp.FIXED['U']); v_c = IV(cp.FIXED['v_c'])
    s_m = IV(float(cfg['s_m'])); d = IV(float(cfg['dbar']))
    dbar = float(cfg['dbar'])
    T_xi = float(cfg['T_xi'])
    amax = [IV(float(x)) for x in cfg['amax']]
    b = [a / v_c for a in amax]
    Umin = [imin(U, a) for a in amax]
    Pi = [IV(0.0)] + [IV(1.0) / b[i - 1] - IV(1.0) / b[i] for i in range(1, n)]
    eta = [IV(0.0)] + [iabs(IV(1.0) - b[i] / b[i - 1]) for i in range(1, n)]
    NUMPAD = IV(0.01); VHND_PAD = float(cfg.get('vhnd', 1e-3))
    refine = bool(cfg.get('refine', True)) and seg
    GAIN_E = IV(1.0) / (lam * E); GAIN_X = IV(1.0); GAIN_F = gamma
    L1_GE = M.ilam2; L1_GX = IV(2.0) / (lam * E)
    TV_KIMP = IV(2.0) * gamma + IV(2.0) * lam * iexp(-3.0)
    XFREE_PK = eps_v + (lam / E) * eps_e
    EFREE_PK = eps_e + eps_v / (lam * E)

    def corner_stats(ebar, xbar):
        best = dict(sup_x=IV(0), sup_f=IV(0), tv_f=IV(0), int_e=IV(0),
                    int_x=IV(0))
        eb, xb = _iv(ebar), _iv(xbar)
        for se in (+1.0, -1.0):
            for sx in (+1.0, -1.0):
                e_0 = IV(-eb.hi, -eb.lo) if se < 0 else IV(eb.lo, eb.hi)
                x_0 = IV(-xb.hi, -xb.lo) if sx < 0 else IV(xb.lo, xb.hi)
                c = x_0 + lam * e_0
                _, _, l1e = M.cf_upper(e_0, c)
                spx, _, l1x = M.cf_upper(x_0, (-lam) * c)
                spf, tvf, _ = M.cf_upper(beta * e_0 + gamma * x_0,
                                         (-(lam * lam)) * c)
                best['sup_x'] = imax(best['sup_x'], spx)
                best['sup_f'] = imax(best['sup_f'], spf)
                best['tv_f'] = imax(best['tv_f'], tvf)
                best['int_e'] = imax(best['int_e'], l1e)
                best['int_x'] = imax(best['int_x'], l1x)
        return best

    def D_quad(ebar, xbar, bi):
        c1, c2 = (-beta) / bi, IV(1.0) - gamma / bi
        out = IV(0)
        eb, xb = _iv(ebar), _iv(xbar)
        for se in (+1.0, -1.0):
            for sx in (+1.0, -1.0):
                e_0 = IV(-eb.hi, -eb.lo) if se < 0 else IV(eb.lo, eb.hi)
                x_0 = IV(-xb.hi, -xb.lo) if sx < 0 else IV(xb.lo, xb.hi)
                c = x_0 + lam * e_0
                _, _, l1 = M.cf_upper(c1 * e_0 + c2 * x_0, c * (c1 - lam * c2))
                out = imax(out, l1)
        return out

    def DB_quad(bi):
        c1, c2 = (-beta) / bi, IV(1.0) - gamma / bi
        _, _, l1 = M.cf_upper(c2, c1 - lam * c2)
        return l1

    bx = corner_stats(eps_e, eps_v)
    IE_BOX, IX_BOX = bx['int_e'], bx['int_x']

    # ---- geometry
    d_s = [IV(float(x)) for x in cfg['d_s']]
    if cfg.get('c0') is not None:
        c0 = [IV(float(x)) for x in cfg['c0']]
        e0 = [IV(float(cfg['e0_head']))] + [c0[i - 1] - d_s[i]
                                            for i in range(1, n)]
        g_init = [IV(0)] + [c0[i - 1] - s_m for i in range(1, n)]
    else:
        e0 = [IV(float(x)) for x in cfg['e0']]
        g_init = [IV(0)] + [d_s[i] - s_m + e0[i] for i in range(1, n)]
    v = []; vp = v_xi
    for i in range(n):
        vi = vp - eps0[i]; v.append(vi); vp = vi
    h0 = [IV(0)] + [g_init[i] + v[i - 1] / b[i - 1] - v[i] / b[i]
                    for i in range(1, n)]

    # ---- acquisition chain to the switch deadlines (pre-values)
    Lam1 = [IV(0)] * n; Lam2p = [IV(0)] * n; taub = [IV(0)] * n
    tact = [IV(0)] * n; tset = [IV(0)] * n; E1 = [IV(0)] * n
    ebar = [IV(0)] * n; fbp = [IV(0)] * n; tvp = [IV(0)] * n
    x2p = [IV(0)] * n; Tk = [IV(0)] * n; Dq = [IV(0)] * n; DBq = [IV(0)] * n
    st_k = []
    for k in range(n):
        Lam1[k] = d * isum(Tk[:k])
        m = iabs(eps0[k]) + Lam1[k]
        taub[k] = m / alpha[k] + T_c
        tact[k] = IV(0) if k == 0 else tset[k - 1] + d
        tset[k] = tact[k] + taub[k]
        Lk = alpha[k] + isum(Tk[:k])
        E1[k] = m * tact[k] + ipos(m * m - eps_det * eps_det) / (IV(2.0) * alpha[k]) \
            + (eps_det + Lk * T_c) * T_c
        ebar[k] = iabs(e0[k]) + E1[k]
        st = corner_stats(ebar[k], eps_det)
        st_k.append(st)
        Lam2p[k] = d * isum(tvp[:k])
        x2p[k] = st['sup_x'] + GAIN_X * Lam2p[k]
        fbp[k] = st['sup_f'] + GAIN_F * Lam2p[k]
        tvp[k] = st['tv_f'] + TV_KIMP * Lam2p[k]
        Tk[k] = IV(2.0) * alpha[k] + (beta * ebar[k] + gamma * eps_det) + tvp[k]
        Dq[k] = D_quad(ebar[k], eps_det, b[k])
        DBq[k] = DB_quad(b[k])
    T_sw_bar = imax(*tset)
    # ---- S2-window chain with the feedforward charge
    Lam2 = [IV(0)] * n; x2 = [IV(0)] * n; fbar = [IV(0)] * n; tv2 = [IV(0)] * n
    for k in range(n):
        s = isum(tv2[:k])
        if seg and k >= 1:
            s = s + IV(2.0) * a_di
        Lam2[k] = d * s
        x2[k] = st_k[k]['sup_x'] + GAIN_X * Lam2[k]
        fbar[k] = st_k[k]['sup_f'] + GAIN_F * Lam2[k]
        tv2[k] = st_k[k]['tv_f'] + TV_KIMP * Lam2[k]
    F = [isum(fbar[:i]) for i in range(n)]
    # ---- braking recursion
    f_xi = beta * eps_e + gamma * eps_v
    W = [IV(0)] * n; epsb = [IV(0)] * n; eb_ = [IV(0)] * n; fb = [IV(0)] * n
    Th = [IV(0)] * n; IE = [IV(0)] * n; IX = [IV(0)] * n; IF = [IV(0)] * n
    Fcum = [IV(0)] * n
    for k in range(1, n):
        W[k] = d * (IV(VHND_PAD) + a_b + f_xi + isum(Th[1:k]) + isum(Fcum[1:k]))
        epsb[k] = XFREE_PK + GAIN_X * W[k]
        eb_[k] = EFREE_PK + GAIN_E * W[k]
        fb[k] = beta * eb_[k] + gamma * epsb[k]
        Fcum[k] = f_xi + isum(fb[1:k + 1])
        IE[k] = IE_BOX + L1_GE * W[k]
        IX[k] = IX_BOX + L1_GX * W[k]
        IF[k] = beta * IE[k] + gamma * IX[k]
        Th[k] = beta * IX[k] + gamma * (W[k] + IF[k])
    Fb = [f_xi + isum(fb[1:i]) for i in range(n)]
    anet = [a_b - Fb[i] for i in range(1, n)]
    closed = all(a.lo > 0 for a in anet) and all(
        (a_b - Fcum[i]).lo > 0 for i in range(1, n - 1))
    out = dict(cfg={k: v for k, v in cfg.items() if k != 'G'},
               closed=bool(closed), cell=cell, conds={}, values={})
    if not closed:
        out['conds']['H2'] = dict(lo=min(a.lo for a in anet),
                                  hi=min(a.hi for a in anet), verified=False)
        out['verified'] = False
        return out
    Pp = [IV(0)] * n
    for u in range(1, n):
        Pp[u] = fb[u] + imax(F[u], Pp[u - 1])
    # ---- budgets, thresholds, margins
    pairs = {}
    for i in range(1, n):
        V1 = isum([iabs(eps0[k]) + Lam1[k] + x2p[k] for k in range(i)])
        H1 = E1[i] + (alpha[i] / b[i]) * taub[i] + V1 * iabs(Pi[i]) + Lam1[i] / b[i]
        V2 = IV(2.0) * isum(x2[:i])
        Ht = Dq[i] + V2 * iabs(Pi[i]) + Lam2[i] * (IV(1.0) / b[i] + DBq[i])
        if seg:
            Ht = Ht + dv * ipos(Pi[i])
        Vbrk = v_c1i + IV(float(i)) * eps_v
        Vrise = IV(float(i - 1)) * d * Pp[i - 1]
        bmin = imin(b[i], b[i - 1])
        an = a_b - Fb[i]
        Ph2 = IX[i] + epsb[i] / bmin + isqr(epsb[i]) / (IV(2.0) * an) \
            + IF[i] / b[i] + Vbrk * ipos(Pi[i]) + Vrise * ipos(-Pi[i]) \
            + W[i] * (IV(1.0) / b[i] + DBq[i])
        M1 = iabs(eps0[i]) + Lam1[i]
        hm1 = imax((b[i] * M1 + (F[i] + A_run[i - 1]) * eta[i]) / (kappa * b[i]),
                   (alpha[i] + b[i] * M1 + F[i] * eta[i]) / (kappa * b[i]))
        Fi = F[i] + a_di if seg else F[i]
        hm2 = (fbar[i] + b[i] * x2[i] + Fi * eta[i]) / (kappa * b[i])
        dem_prev = F[i - 1] + imax(A_run[i - 1], fbar[i - 1])
        Ubrk = imax(dem_prev, Pp[i - 1], a_b + Fb[i], (a_b + Fb[i - 1]) + fbar[i - 1])
        hmb = imax((fb[i] + b[i] * epsb[i] + Ubrk * eta[i]) / (kappa * b[i]),
                   ipos(b[i] * epsb[i] + (b[i] / b[i - 1]) * (a_b + Fb[i]) - a_b)
                   / (kappa * b[i]))
        thr = imax(H1 + hm1, H1 + Ht + hm2, H1 + Ht + Ph2 + hmb)
        uh = h0[i] - H1 - Ht
        bf = uh - Ph2
        pairs[i] = dict(H1=H1, Ht=Ht, Ph2=Ph2, thr=thr, V1=V1, uh=uh, bf=bf,
                        margin=h0[i] - thr - NUMPAD)
    # ---- fleet bounds
    hops = [epsb[i] / (a_b - Fb[i]) for i in range(1, n)]
    disp = isum(hops)
    late = isum([epsb[i] / a_b for i in range(1, n)])
    sched = eps_v / a_b + disp
    gaperr = [EFREE_PK + GAIN_E * W[i] + isqr(epsb[i]) / (IV(2.0) * (a_b - Fb[i]))
              for i in range(1, n)]
    align = eps_e + v_c1i * eps_v / a_b + isqr(eps_v) / (IV(2.0) * a_b) + isum(gaperr)
    epsb_max = imax(*epsb[1:])
    T_f = IV(T_xi) + (v_c1i + eps_v) / a_b + IV(float(n - 1)) * epsb_max / a_b
    # ---- authority
    # (R1) segment demand at t = t_d: a_d plus, for every layer k <= i,
    # the corner tail supremum of |f_free,k| from r >= t_d - (i-k) dbar -
    # tset_k (interval) plus the forced part gamma Lambda2_k; premise
    # t_d >= T_sw_bar + (n-1) dbar (P_seg)
    seg_dem = [None] * n
    if refine:
        for i in range(n):
            tot = a_di
            for k in range(i + 1):
                Rk = IV(t_d) - IV(float(i - k)) * d - tset[k]
                Rk = IV(max(Rk.lo, 0.0), max(Rk.hi, 0.0))
                best = 0.0
                eb = ebar[k]
                for se in (+1.0, -1.0):
                    for sx in (+1.0, -1.0):
                        e_0 = IV(-eb.hi, -eb.lo) if se < 0 else IV(eb.lo, eb.hi)
                        x_0 = IV(-eps_det.hi, -eps_det.lo) if sx < 0 else eps_det
                        c = x_0 + lam * e_0
                        best = max(best, M.tail_sup_hi(beta * e_0 + gamma * x_0,
                                                       (-(lam * lam)) * c, Rk))
                tot = tot + IV(0.0, best) + GAIN_F * Lam2[k]
            seg_dem[i] = tot
    dem = []
    for i in range(n):
        acq = F[i] + imax(A_run[i], fbar[i])
        if seg and not refine:
            acq = acq + a_di
        brk = a_b + (Fcum[i] if i >= 1 else f_xi)
        pre = (a_b + Fb[i]) + fbar[i] if i >= 1 else IV(0)
        cands = [acq, brk, Pp[i], pre]
        if refine:
            cands.append(seg_dem[i])
        dem.append(imax(*cands))
    # ---- sampled event, velocity floors
    Leps = imax(*alpha) + isum(Tk)
    # pre-switch bound with the S2-window band x2 (feedforward ledger
    # included), as certify_profile.core(): valid on [0, t_d), where every
    # signal coincides with the acquisition continuation of Assumption 1,
    # and nonincreasing in dbar together with the time-resolved curve;
    # without a segment x2 == x2p
    supx = [imax(iabs(eps0[k]) + Lam1[k], x2[k]) for k in range(n)]
    cum = []; t = IV(0)
    for s in supx:
        t = t + s; cum.append(t)
    vfloor_pre = v_xi - imax(*cum)
    vflr_c = v_c1i - IV(float(n)) * eps_v

    def vr_lo(t):
        """lower bound of v_r on the acquisition profile at time t."""
        if not seg or t < t_d:
            return v_xi.lo
        if t < t_c:
            return (v_xi - a_di * (IV(t) - IV(t_d))).lo
        return v_c1i.lo

    def env_hi(t, k):
        """upper bound on eps_k(t) for t >= T_sw_bar: with (R2) the signed
        corner supremum of eps_free over the asymmetric switch-state box
        [e_k(0) - E1_k, e_k(0) + E1_k] x {+-eps_det}, otherwise the
        supremum of |eps_free| over the symmetric box; plus Lambda2_k."""
        R = IV(max(t - tset[k].hi, 0.0))
        best = 0.0
        if refine:
            for e_0 in (e0[k] - E1[k], e0[k] + E1[k]):
                for sx in (+1.0, -1.0):
                    x_0 = IV(-eps_det.hi, -eps_det.lo) if sx < 0 else eps_det
                    c = x_0 + lam * e_0
                    best = max(best, M.tail_sup_signed_hi(x_0, (-lam) * c, R))
            return up(best + Lam2[k].hi)
        eb = ebar[k]
        for se in (+1.0, -1.0):
            for sx in (+1.0, -1.0):
                e_0 = IV(-eb.hi, -eb.lo) if se < 0 else IV(eb.lo, eb.hi)
                x_0 = IV(-eps_det.hi, -eps_det.lo) if sx < 0 else eps_det
                c = x_0 + lam * e_0
                best = max(best, M.tail_sup_hi(x_0, (-lam) * c, R))
        return up(best + Lam2[k].hi)

    t_a = T_sw_bar.hi
    ncell = int(math.ceil((T_xi - t_a) / cell))
    vflr_time = INF; t_at = None
    if ncell > 0:
        for j in range(ncell):
            tj = t_a + j * cell
            tj1 = min(t_a + (j + 1) * cell, T_xi)
            env = 0.0
            for k in range(n):
                env = up(env + env_hi(tj, k))
            f = dn(vr_lo(tj1) - env)
            if f < vflr_time:
                vflr_time = f; t_at = tj
    vflr_lo = min(vfloor_pre.lo, vflr_time)
    # ---- clearance floors
    gmin = IV(INF, INF)
    rows = {}
    for i in range(1, n):
        P = pairs[i]; uh = P['uh']; bf = P['bf']
        if Pi[i].hi > 0:
            V1f = isum([iabs(eps0[k]) + Lam1[k] + x2[k] for k in range(i)])
            vs_a = v_xi + V1f
            vs_b = v_c1i + eps_v + isum(epsb[1:i])
            gspd = imin(g_init[i], uh - ipos(Pi[i]) * vs_a, bf - ipos(Pi[i]) * vs_b)
            gm = imin(gspd, bf)
        else:
            gm = imin(g_init[i], bf)
        rows[i] = gm
        gmin = imin(gmin, gm)
    # ---- condition intervals (slack lower / upper)
    C = {}

    def put(name, lst):
        C[name] = dict(lo=min(x.lo for x in lst), hi=min(x.hi for x in lst))
        C[name]['verified'] = bool(C[name]['lo'] > 0)

    put('H1', [Umin[i] - dem[i] for i in range(n)])
    put('H2', anet)
    put('H4', [pairs[i]['margin'] for i in range(1, n)])
    put('H5', [gmin])
    put('H7', [(eps_det - phi) - Leps * T_c])
    C['H6'] = dict(lo=float(vflr_lo), hi=min(vfloor_pre.hi, INF),
                   verified=bool(vflr_lo > 0), pre_lo=vfloor_pre.lo,
                   time_lo=vflr_time, time_at=t_at)
    put('H8', [vflr_c - IV(float(n - 1)) * d * imax(*Umin)])
    put('C_sync', [IV(float(cfg['sync_tol'])) - disp])
    put('C_al', [IV(float(cfg['align_tol'])) - align])
    put('C_gap', [IV(float(cfg['gap_tol'])) - g for g in gaperr])
    if seg:
        s1 = IV(t_d) - T_sw_bar
        if refine:
            s1 = s1 - IV(float(n - 1)) * d
        s2 = IV(T_xi) - (IV(t_c) + IV(float(n - 1)) * d)
        put('P_seg', [imin(s1, s2)])
    else:
        C['P_seg'] = dict(lo=INF, hi=INF, verified=True)
    out['conds'] = C
    out['refine'] = refine
    out['values'] = dict(disp=[disp.lo, disp.hi], late=[late.lo, late.hi],
                         sched=[sched.lo, sched.hi], align=[align.lo, align.hi],
                         gaperr=[[g.lo, g.hi] for g in gaperr],
                         T_f=[T_f.lo, T_f.hi], T_sw_bar=[T_sw_bar.lo, T_sw_bar.hi],
                         Lam2=[[x.lo, x.hi] for x in Lam2],
                         margins=[[pairs[i]['margin'].lo, pairs[i]['margin'].hi]
                                  for i in range(1, n)],
                         gmin=[gmin.lo, gmin.hi],
                         auth=[[(Umin[i] - dem[i]).lo, (Umin[i] - dem[i]).hi]
                               for i in range(n)],
                         seg_dem=None if not refine else
                         [[s.lo, s.hi] for s in seg_dem],
                         vfloor_pre=[vfloor_pre.lo, vfloor_pre.hi],
                         vflr_time_lo=vflr_time, vflr_time_at=t_at,
                         vflr_c=[vflr_c.lo, vflr_c.hi])
    out['verified'] = bool(all(c['verified'] for c in C.values()))
    out['scope'] = ('closed-form chain, H6 on a %g s cell partition; H3 and '
                    'C_hnd are grid-evaluated in floating point' % cell)
    if verbose:
        for k, c in C.items():
            print('  %-6s slack in [%+.6f, %s]  %s'
                  % (k, c['lo'], ('%+.6f' % c['hi']) if c['hi'] < INF else 'inf',
                     'VERIFIED' if c['verified'] else 'not verified'))
    return out


def check_enclosure(cfg, cell=0.02):
    """Interval enclosure of a configuration and the check that every
    floating-point slack of certify_profile.evaluate lies inside it."""
    enc = enclose(cfg, cell=cell)
    fp = cp.evaluate(cfg)
    inside = {}
    for k, c in enc['conds'].items():
        s = fp['conds'][k]['slack']
        if s is None or not np.isfinite(s) or not np.isfinite(c['lo']):
            inside[k] = None
        elif k == 'H6':
            # the cell partition loses at most a_d*cell; the floating
            # value must not lie below the interval lower bound
            inside[k] = bool(c['lo'] <= s + 1e-12)
        else:
            inside[k] = bool(c['lo'] - 1e-15 <= s <= c['hi'] + 1e-15)
    return dict(enclosure=enc, fp_slack={k: fp['conds'][k]['slack']
                                         for k in enc['conds']},
                inside=inside, all_inside=all(v in (True, None)
                                              for v in inside.values()))


def regress(verbose=True):
    """Case B at 20 ms against the archived witness (b2_results.json
    'witness'/'primary') and the enclosure of the floating-point values."""
    B = json.load(open(os.path.join(cp.CODE, 'data', 'new_d19',
                                    'b2_results.json')))['witness']['primary']
    r = check_enclosure(cp.config())
    enc = r['enclosure']
    ref = B['margins']
    names = dict(adm='H4', auth='H1', clr='H5', event='H7', vel='H6',
                 head='H2', sync='C_sync', align='C_al', gap='C_gap')
    rep = {}
    for k, nm in names.items():
        lo, hi = enc['conds'][nm]['lo'], enc['conds'][nm]['hi']
        rep[k] = dict(archived_lo=ref[k], archived_hi=ref['#' + k], lo=lo,
                      hi=hi, dlo=lo - ref[k])
        if verbose:
            print('  %-6s archived [%+.9f, %+.9f]  new [%+.9f, %+.9f]'
                  % (k, ref[k], ref['#' + k], lo, hi))
    if verbose:
        print('  disp hi archived %.15f new %.15f' % (B['margins']['_disp_hi'],
                                                        enc['values']['disp'][1]))
        print('  floating-point slacks inside:', r['inside'])
    return dict(compare=rep, inside=r['inside'], all_inside=r['all_inside'],
                verified=enc['verified'],
                disp_hi=enc['values']['disp'][1],
                disp_hi_archived=B['margins']['_disp_hi'])


if __name__ == '__main__':
    rep = regress()
    print('verified:', rep['verified'], ' enclosure ok:', rep['all_inside'])
