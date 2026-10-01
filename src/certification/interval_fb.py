#!/usr/bin/env python3
"""interval_fb.py -- outward-interval enclosure of the constants and
conditions of certify_fb.evaluate (feedback-only followers and a shaped
head reference, 2026-09-29).  This module is interval_headref.py (copied
2026-09-29; the original stays unchanged for the schedule version in
new/future/) evaluated on certify_fb, with the FB refinement of certify_fb
(earliest S2 switch instants tlow_k, marked "FB refinement") enclosed in
interval arithmetic: tlow_k is an interval, and the S2 terms of layer k are
dropped only on the cells whose end lies strictly below its lower end.
The remaining docstring is that of interval_headref.py.

interval_headref.py -- outward-interval enclosure of the constants and
conditions of certify_headref.evaluate (arrival with a reference known only
to the head, HR; formulas and code lines in
data/new_d19/headref/evaluator_notes.md, items N1-N20).

Arithmetic (the conventions of interval_plan.py and interval_profile.py,
whose interval class IV, helpers and Arb exponentials are imported
read-only):
  * every design constant is the exact rational of its decimal value
    (certify_plan.F), and every plan quantity (reference pieces, schedules,
    jump times and sizes, handoff speed, crawl rate) the exact rational of
    certify_headref.setup; each enters as the tightest interval of doubles
    (interval_plan.ivF);
  * scalars are intervals of doubles with one-ulp outward rounding of every
    operation (interval_profile.IV); exponentials are Arb balls
    (python-flint, 120 bits) stepped one ulp outward;
  * cell functions are arrays of intervals (class IA below) with one
    outward nextafter step after every rounded operation; e^{-lam r} at a
    lag r = t - s is the product of the Arb enclosures of e^{-lam t} and
    e^{lam s};
  * the closed forms (sup, TV, L1 of the modal family (p + q t) e^{-lam t},
    the signed tail supremum, the piecewise-quadratic window extrema) are
    evaluated branch by branch; a branch that the intervals cannot decide
    contributes the union of both branch values.  Unlike interval_plan.py,
    every enclosure is two-sided, so that a condition can be refuted (upper
    end of the slack on the failing side) as well as verified.

Scope (interval-enclosed, two-sided):
  premises A_R, A_S (exact), C0a, C0b, H8, C1, C2; the planned L1 budgets
  A^(1), A^(2) (exact rationals); the planned pulse bounds of Lemma 7.1 on
  every 10 ms cell of [0, T_xi] for every pair (all jumps, all cells) and
  their sup-norms P^e, P^eps, P^f; the acquisition chain (Lambda^(1),
  deadlines, E^(1), corner constants, V^acq, Lambda^(2), V^S2, eps-bar^(2),
  e-bar^(2), f-bar, F); the relayed braking recursion (W, W^Phi, bands,
  integrals, Theta, F^b, headroom, q, dispersion, T_f bound, marker and gap
  bounds); the velocity floor C4 and the authority tests C7a, C7b on the
  10 ms cells (signed tail suprema of the free feedback, planned forced
  feedback, window maxima with the exact cell coverage); C6, C7c (braking
  and pre-receipt); the offset-robust planned minima of h*, mu*, g* and the
  window oscillations Q (exact-coefficient piecewise quadratics); the
  budgets, thresholds and admission tests C8a/b/c; the floors C9a/b;
  C10-C12.
  The entry layer (C3 and C5: tail recursion on the 1 ms lag grid, 210 s
  horizon) is enclosed by entry_layer() below with the method of
  run_witness_caseB_entry.py (its closed-form helpers imported unchanged):
  exact grid k/1000 s, Arb scalars, outward-rounded double arrays.

Status of a condition: verified (lower end of its slack > 0 for a strict
test, >= 0 otherwise), refuted (upper end <= 0, resp. < 0), else undecided.

API
  enclose(cfg, dbar=None, entry=True, keep=False)
                          -> dict(conds, consts, bounds, entry, timing, ...)
  selfcheck(cfg, enc)     -> independent checks (acquisition chain against
                             Arb, window extrema against Fractions, pulse
                             cell bounds against Arb)
  float_crosscheck(cfg, enc)
                          -> certify_headref.evaluate against the enclosure:
                             every slack and constant, the pulse bounds cell
                             by cell, the entry lags and command tails (with
                             the exact planned L1 tail and exact grid shifts
                             substituted to trace every excursion), and the
                             margins in units of the rounding estimate
Driver: code/run_hr_interval.py (data/new_d19/headref/interval_hr.json).
"""
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
import certify_fb as hr                                   # noqa: E402
import certify_plan as cq                                 # noqa: E402
from interval_profile import (IV, iexp, iabs, imax, imin, ipos, isum,  # noqa: E402
                              isqr, INF)
from interval_plan import ivF                             # noqa: E402

F = cq.F
U_RND = 2.0 ** -53
ARB_BITS = 120


# ============================================================ scalars
def _iv(x):
    return x if isinstance(x, IV) else IV(x)


def ivQ(x):
    """tightest double interval of the exact rational of a design value."""
    return ivF(x if isinstance(x, Fr) else F(x))


def idiv(a, b):
    """a / b for an interval b of one sign."""
    a, b = _iv(a), _iv(b)
    if b.lo > 0.0:
        return a / b
    if b.hi < 0.0:
        return -(a / (-b))
    raise ZeroDivisionError('interval division by an interval containing 0')


def hull(xs):
    return IV(min(x.lo for x in xs), max(x.hi for x in xs))


def pr(x):
    """json pair of an interval."""
    return [float(x.lo), float(x.hi)]


# ============================================================ arrays
def nd(x):
    return np.nextafter(x, -np.inf)


def nu(x):
    return np.nextafter(x, np.inf)


class IA:
    """array of intervals [lo, hi] of IEEE doubles; every rounded operation
    is followed by one outward nextafter step."""
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo = np.asarray(lo, dtype=float)
        self.hi = self.lo if hi is None else np.asarray(hi, dtype=float)

    def __add__(s, o):
        o = _ia(o)
        return IA(nd(s.lo + o.lo), nu(s.hi + o.hi))
    __radd__ = __add__

    def __sub__(s, o):
        o = _ia(o)
        return IA(nd(s.lo - o.hi), nu(s.hi - o.lo))

    def __rsub__(s, o):
        return _ia(o) - s

    def __neg__(s):
        return IA(-s.hi, -s.lo)

    def __mul__(s, o):
        o = _ia(o)
        c1 = s.lo * o.lo
        c2 = s.lo * o.hi
        c3 = s.hi * o.lo
        c4 = s.hi * o.hi
        return IA(nd(np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))),
                  nu(np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))))
    __rmul__ = __mul__

    def __getitem__(s, k):
        return IA(s.lo[k], s.hi[k])


def _ia(x):
    if isinstance(x, IA):
        return x
    if isinstance(x, IV):
        return IA(x.lo, x.hi)
    return IA(float(x), float(x))


def iamax(*xs):
    xs = [_ia(x) for x in xs]
    lo, hi = xs[0].lo, xs[0].hi
    for x in xs[1:]:
        lo = np.maximum(lo, x.lo)
        hi = np.maximum(hi, x.hi)
    return IA(lo, hi)


def iamin(*xs):
    xs = [_ia(x) for x in xs]
    lo, hi = xs[0].lo, xs[0].hi
    for x in xs[1:]:
        lo = np.minimum(lo, x.lo)
        hi = np.minimum(hi, x.hi)
    return IA(lo, hi)


def iapos(x):
    return IA(np.maximum(x.lo, 0.0), np.maximum(x.hi, 0.0))


def ia_min_all(x):
    """interval of the minimum over all entries."""
    return IV(float(np.min(x.lo)), float(np.min(x.hi)))


def ia_max_all(x):
    return IV(float(np.max(x.lo)), float(np.max(x.hi)))


def _arb():
    from flint import arb, ctx
    ctx.prec = ARB_BITS
    return arb


def exp_neg_lam_t(lam, t):
    """enclosure of e^{-lam t} for every double t >= 0 of the array t and
    every lam in the interval lam (Arb, one ulp outward)."""
    arb = _arb()
    Ll, Lh = arb(lam.lo), arb(lam.hi)
    t = np.asarray(t, float)
    lo = np.empty(t.shape)
    hi = np.empty(t.shape)
    for j, x in enumerate(t.tolist()):
        a = arb(x)
        lo[j] = float((-(Lh * a)).exp().lower())
        hi[j] = float((-(Ll * a)).exp().upper())
    return IA(nd(lo), np.minimum(nu(hi), 1.0))


def exp_neg_lam_R(ET, Rraw, shift_exp, lam):
    """e^{-lam R} for R = max(t - s, 0) on the cells: ET encloses e^{-lam t}
    (array), shift_exp encloses e^{lam s} (scalar), Rraw encloses t - s.
    Certainly positive lags use the product, certainly nonpositive lags the
    value 1, undecided ones [1 - lam R^hi, 1] (e^{-x} >= 1 - x)."""
    pos = Rraw.lo > 0.0
    neg = Rraw.hi <= 0.0
    Rhi = np.maximum(Rraw.hi, 0.0)
    lo = np.where(pos, nd(ET.lo * shift_exp.lo),
                  np.where(neg, 1.0, nd(1.0 - nu(lam.hi * Rhi))))
    hi = np.where(pos, np.minimum(nu(ET.hi * shift_exp.hi), 1.0), 1.0)
    return IA(lo, hi)


# ============================================ two-sided modal closed forms
class Modal2:
    """two-sided enclosures of sup|g|, TV(g), ||g||_1 on [0, inf) of
    g(t) = (p + q t) e^{-lam t} (the closed forms of certify_v30._cf) and of
    the signed tail supremum sup_{r >= R} g(r) (certify_profile.
    _tail_sup_signed) for interval p, q, lam."""

    def __init__(self, lam):
        self.lam = lam
        self.ilam = IV(1.0) / lam
        self.ilam2 = self.ilam * self.ilam
        self.inv_e = iexp(IV(-1.0))
        self.undecided = 0
        self.undecided_calls = []

    def cf(self, p, q):
        lam, ilam, ilam2 = self.lam, self.ilam, self.ilam2
        p, q = _iv(p), _iv(q)
        ap = iabs(p)
        if q.lo == 0.0 and q.hi == 0.0:
            return ap, ap, ap * ilam
        if q.lo > 0.0 or q.hi < 0.0:
            ts = idiv(q - lam * p, lam * q)
            sups, tvs = [], []
            if ts.lo <= 0.0:                        # branch ts <= 0
                sups.append(ap)
                tvs.append(ap)
            if ts.hi > 0.0:                         # branch ts > 0
                tsp = IV(max(ts.lo, 0.0), ts.hi)
                gs = (q * ilam) * iexp(-(lam * tsp))
                ags = iabs(gs)
                sups.append(imax(ap, ags))
                tvs.append(iabs(p - gs) + ags)
            I0 = p * ilam + q * ilam2
            t0 = idiv(-p, q)
            l1s = []
            if t0.lo <= 0.0:
                l1s.append(iabs(I0))
            if t0.hi > 0.0:
                t0p = IV(max(t0.lo, 0.0), t0.hi)
                It = (q * ilam2) * iexp(-(lam * t0p))
                l1s.append(iabs(I0 - It) + iabs(It))
            if len(sups) > 1 or len(l1s) > 1:
                self.undecided += 1
                self.undecided_calls.append(dict(
                    p=pr(p), q=pr(q), ts=pr(ts), t0=pr(t0),
                    sup=pr(hull(sups)), tv=pr(hull(tvs)), l1=pr(hull(l1s))))
            return hull(sups), hull(tvs), hull(l1s)
        # q of undecided sign (not identically zero): bounds valid for
        # every q in the interval
        self.undecided += 1
        self.undecided_calls.append(dict(p=pr(p), q=pr(q), note='q of undecided sign'))
        aq = iabs(q)
        sup = IV(ap.lo, (ap + aq * ilam * self.inv_e).hi)
        tv = IV(ap.lo, (ap + IV(2.0) * aq * ilam).hi)
        l1 = IV(iabs(p * ilam + q * ilam2).lo, (ap * ilam + aq * ilam2).hi)
        return sup, tv, l1

    def tail_sup_signed(self, p, q, R, ER):
        """sup_{r >= R} (p + q r) e^{-lam r} for the lag array R (IA, R >= 0)
        with ER an enclosure of e^{-lam R}: max{g(R), 0} and, for q > 0 and
        R below the stationary point ts, g(ts) = (q/lam) e^{-lam ts}."""
        lam, ilam = self.lam, self.ilam
        p, q = _iv(p), _iv(q)
        g = (_ia(p) + _ia(q) * R) * ER
        lo = np.maximum(g.lo, 0.0)
        hi = np.maximum(g.hi, 0.0)
        if q.lo > 0.0:
            ts = idiv(q - lam * p, lam * q)
            gs = (q * ilam) * iexp(-(lam * ipos(ts)))
            hi = np.where(R.lo < ts.hi, np.maximum(hi, gs.hi), hi)
            lo = np.where(R.hi < ts.lo, np.maximum(lo, gs.lo), lo)
        elif q.hi > 0.0:
            # q may vanish: g <= |p| e^{-lam r} + q^+ r e^{-lam r}
            b = nu(nu(iabs(p).hi * ER.hi) + (q * ilam * self.inv_e).hi)
            hi = np.maximum(hi, b)
        return IA(lo, hi)


def corner_stats2(M, beta, gamma, ebar, xbar):
    """two-sided corner-exact sup/TV/L1 of the free response from the sign
    corners of the box |e| <= ebar, |x| <= xbar (certify_v30.corner_stats)."""
    lam = M.lam
    best = None
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0 = ebar if se > 0 else -ebar
            x_0 = xbar if sx > 0 else -xbar
            c = x_0 + lam * e_0
            spe, _, l1e = M.cf(e_0, c)
            spx, _, l1x = M.cf(x_0, (-lam) * c)
            spf, tvf, _ = M.cf(beta * e_0 + gamma * x_0, (-(lam * lam)) * c)
            cur = dict(sup_e=spe, sup_x=spx, sup_f=spf, tv_f=tvf, int_e=l1e,
                       int_x=l1x)
            if best is None:
                best = cur
            else:
                best = {k: imax(best[k], cur[k]) for k in best}
    return best


def D_quad2(M, beta, gamma, ebar, xbar, bi):
    lam = M.lam
    c1, c2 = (-beta) / bi, IV(1.0) - gamma / bi
    out = None
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0 = ebar if se > 0 else -ebar
            x_0 = xbar if sx > 0 else -xbar
            c = x_0 + lam * e_0
            _, _, l1 = M.cf(c1 * e_0 + c2 * x_0, c * (c1 - lam * c2))
            out = l1 if out is None else imax(out, l1)
    return out


def DB_quad2(M, beta, gamma, bi):
    lam = M.lam
    c1, c2 = (-beta) / bi, IV(1.0) - gamma / bi
    _, _, l1 = M.cf(c2, c1 - lam * c2)
    return l1


# ================================== exact piecewise quadratics (plan views)
class IPW:
    """f(x) = c0[j] + c1[j] tau + c2[j] tau^2, tau = x - X[j], on the piece
    [X[j], X[j+1]) (first piece from X[0] = -BIG, last to +infinity), exact
    rational breakpoints and coefficients, f right-continuous.  ext(a, b,
    mode) encloses the minimum or maximum over the closed window [a, b] as
    certify_headref.PW.ext defines it: every piece that meets [a, b]
    (X_j <= b and X_{j+1} > a) evaluated on the closure of its part in the
    window, i.e. f(a), both one-sided limits at every breakpoint in (a, b]
    and the vertex, but not the left limit at a.  The window ends are
    interval arrays: the lower (upper) end of a minimum is taken over every
    piece that possibly meets the window and every candidate that possibly
    lies in it, the upper end over the pieces that certainly meet it and the
    candidates that certainly lie in it."""

    def __init__(self, X, c0, c1=None, c2=None):
        m = len(X)
        c1 = [Fr(0)] * m if c1 is None else list(c1)
        c2 = [Fr(0)] * m if c2 is None else list(c2)
        c0 = list(c0)
        assert len(c0) == len(c1) == len(c2) == m
        assert all(X[j + 1] > X[j] for j in range(m - 1))
        self.m = m
        self.X = list(X)
        self.c = (c0, c1, c2)

        def arr(vals):
            ivs = [ivF(v) for v in vals]
            return (np.array([v.lo for v in ivs]), np.array([v.hi for v in ivs]))
        self.Xl, self.Xh = arr(X)
        self.Xel = np.append(self.Xl[1:], np.inf)
        self.Xeh = np.append(self.Xh[1:], np.inf)
        Ll, Lh = arr([X[j + 1] - X[j] for j in range(m - 1)]) if m > 1 \
            else (np.zeros(0), np.zeros(0))
        self.Ll = np.append(Ll, np.inf)
        self.Lh = np.append(Lh, np.inf)
        self.c0l, self.c0h = arr(c0)
        self.c1l, self.c1h = arr(c1)
        self.c2l, self.c2h = arr(c2)
        self.c2s = np.array([(v > 0) - (v < 0) for v in c2], int)
        tv = [(-b / (2 * a)) if a != 0 else Fr(0) for b, a in zip(c1, c2)]
        fv = [(c - b * b / (4 * a)) if a != 0 else c for c, b, a in zip(c0, c1, c2)]
        self.tvl, self.tvh = arr(tv)
        self.fvl, self.fvh = arr(fv)

    def _q(self, j, tl, th):
        T = IA(tl, th)
        c2 = IA(self.c2l[j], self.c2h[j])
        c1 = IA(self.c1l[j], self.c1h[j])
        c0 = IA(self.c0l[j], self.c0h[j])
        return c0 + T * (c1 + c2 * T)

    def ext(self, a, b, mode):
        a, b = _ia(a), _ia(b)
        a_lo, a_hi = np.broadcast_arrays(a.lo, a.hi)
        b_lo, b_hi = np.broadcast_arrays(b.lo, b.hi)
        shape = np.broadcast(a_lo, b_lo).shape
        a_lo = np.broadcast_to(a_lo, shape); a_hi = np.broadcast_to(a_hi, shape)
        b_lo = np.broadcast_to(b_lo, shape); b_hi = np.broadcast_to(b_hi, shape)
        m = self.m
        # piece j meets the window of the right-continuous function iff
        # X_j <= b and X_{j+1} > a (the left limit at a is not attained);
        # possibly: X_j.lo <= b.hi, X_{j+1}.hi > a.lo; certainly: X_j.hi <=
        # b.lo, X_{j+1}.lo > a.hi
        js = np.clip(np.searchsorted(self.Xeh, a_lo, 'right'), 0, m - 1)
        je = np.clip(np.searchsorted(self.Xl, b_hi, 'right') - 1, 0, m - 1)
        K = int(np.max(je - js)) + 1 if a_lo.size else 0
        mn = mode == 'min'
        big = np.inf if mn else -np.inf
        out_lo = np.full(shape, big)
        out_hi = np.full(shape, big)
        for dd in range(max(K, 1)):
            j = js + dd
            valid = j <= je
            j = np.minimum(j, je)
            cert = valid & (self.Xh[j] <= b_lo) & (self.Xel[j] > a_hi)
            tl_lo = np.maximum(nd(a_lo - self.Xh[j]), 0.0)
            tl_hi = np.maximum(nu(a_hi - self.Xl[j]), 0.0)
            th_lo = np.minimum(nd(b_lo - self.Xh[j]), self.Ll[j])
            th_hi = np.minimum(nu(b_hi - self.Xl[j]), self.Lh[j])
            ql = self._q(j, tl_lo, tl_hi)
            qh = self._q(j, th_lo, th_hi)
            if mn:
                c_lo = np.minimum(ql.lo, qh.lo)
                c_hi = np.minimum(ql.hi, qh.hi)
                vx = self.c2s[j] > 0
            else:
                c_lo = np.maximum(ql.lo, qh.lo)
                c_hi = np.maximum(ql.hi, qh.hi)
                vx = self.c2s[j] < 0
            if np.any(vx):
                vposs = vx & (self.tvh[j] > tl_lo) & (self.tvl[j] < th_hi)
                vcert = vx & (self.tvl[j] >= tl_hi) & (self.tvh[j] <= th_lo)
                if mn:
                    c_lo = np.where(vposs, np.minimum(c_lo, self.fvl[j]), c_lo)
                    c_hi = np.where(vcert, np.minimum(c_hi, self.fvh[j]), c_hi)
                else:
                    c_hi = np.where(vposs, np.maximum(c_hi, self.fvh[j]), c_hi)
                    c_lo = np.where(vcert, np.maximum(c_lo, self.fvl[j]), c_lo)
            if mn:
                out_lo = np.where(valid, np.minimum(out_lo, c_lo), out_lo)
                out_hi = np.where(cert, np.minimum(out_hi, c_hi), out_hi)
            else:
                out_hi = np.where(valid, np.maximum(out_hi, c_hi), out_hi)
                out_lo = np.where(cert, np.maximum(out_lo, c_lo), out_lo)
        if mn:
            return IA(out_lo, out_hi)
        return IA(out_lo, out_hi)


def plan_views(S, cfg):
    """exact-coefficient views of certify_headref.HRSetup (the float PW
    objects arA, vr, arkvr, rho_k, D_k, rho_k + kappa D_k, A_k, B_k, gs_k,
    rebuilt from the exact rationals)."""
    n = S.n
    b = [F(a) / F(hr.FIXED['v_c']) for a in cfg['amax']]
    kap = F(hr.FIXED['kappa'])
    s_m = F(cfg['s_m'])
    BIGF = Fr(int(hr.BIG))
    rt, ra, rv = list(S.ref_t), list(S.ref_a), list(S.ref_v)
    V = dict(arA=IPW([-BIGF] + rt, [Fr(0)] + ra),
             vr=IPW([-BIGF] + rt, [S.v_in] + rv, [Fr(0)] + ra),
             arkvr=IPW([-BIGF] + rt,
                       [kap * S.v_in] + [a + kap * v for a, v in zip(ra, rv)],
                       [Fr(0)] + [kap * a for a in ra]))
    for nm in ('rho', 'negdd', 'rkd', 'A', 'B', 'gs'):
        V[nm] = [None]
    for u in range(1, n):
        sc = S.sched[u]
        bu = b[u]
        d0u = S.d0[u]
        if sc['x']:
            X = [-BIGF] + list(sc['x']) + [sc['x_f']]
            r0 = [Fr(0)] + list(sc['rho']) + [Fr(0)]
            D0 = [Fr(0)] + list(sc['D']) + [sc['D_end']]
            S0 = [Fr(0)] + list(sc['S']) + [sc['S_end']]
        else:
            X, r0, D0, S0 = [-BIGF], [Fr(0)], [Fr(0)], [Fr(0)]
        A0 = [d0u - s_m - Sj - Dj / bu for Sj, Dj in zip(S0, D0)]
        A1 = [-Dj - rj / bu for Dj, rj in zip(D0, r0)]
        A2 = [-rj / 2 for rj in r0]
        B0 = [-Dj - rj / bu + kap * a0 for Dj, rj, a0 in zip(D0, r0, A0)]
        B1 = [-rj + kap * a1 for rj, a1 in zip(r0, A1)]
        B2 = [kap * a2 for a2 in A2]
        V['rho'].append(IPW(X, r0))
        V['negdd'].append(IPW(X, D0, r0))
        V['rkd'].append(IPW(X, [rj + kap * Dj for rj, Dj in zip(r0, D0)],
                            [kap * rj for rj in r0]))
        V['A'].append(IPW(X, A0, A1, A2))
        V['B'].append(IPW(X, B0, B1, B2))
        V['gs'].append(IPW(X, [d0u - s_m - Sj for Sj in S0], [-Dj for Dj in D0],
                           [-rj / 2 for rj in r0]))
    return V


def pair_jumps_exact(S, c):
    """exact version of HRSetup.pair_jumps (same order)."""
    hs = [h for (t, h) in S.ref_jumps]
    los = [t for (t, h) in S.ref_jumps]
    for u in range(1, c):
        for (x, h) in S.sched_jumps[u]:
            hs.append(h)
            los.append(S.t_0 + x)
    return hs, los


# ============================================ planned pulses (Lemma 7.1)
def pulse_enclosure(S, c, dq, K, cells):
    """two-sided enclosure, on every cell [t_g, t_g + Delta_g] of the float
    evaluator, of the cell bounds of certify_headref.pulse_bounds for code
    pair c: the lag interval [max(0, t_g - hi_J), t_g1 - lo_J] with the
    exact K_J = [lo_J, lo_J + c dbar]; activation decided exactly when the
    intervals cannot decide it; the endpoint values and the interior
    extremum (peak of h_e at 1/lam, trough of h_eps at 2/lam, of h_f at
    3/lam) included when possibly (upper ends) or certainly (lower ends)
    inside the lag interval."""
    lam = K['lam']
    dI = K['d']
    lam_lo, lam_hi = lam.lo, lam.hi
    R1, R2, R3 = K['R1'], K['R2'], K['R3']
    PE, TRx, TRf = K['PE'], K['TRx'], K['TRf']
    tg, tg1 = cells['tg'], cells['tg1']
    ET0, ET1 = cells['ET0'], cells['ET1']
    nC = len(tg)
    names = ('e_hi', 'e_lo', 'x_hi', 'x_lo', 'f_hi', 'f_lo')
    acc = {k: [np.zeros(nC), np.zeros(nC)] for k in names}
    hs, los = pair_jumps_exact(S, c)
    n_amb = 0

    def add(key, g0, t_lo, t_hi, sign):
        L, H = acc[key]
        if sign > 0:
            L[g0:] = nd(L[g0:] + t_lo)
            H[g0:] = nu(H[g0:] + t_hi)
        else:
            L[g0:] = nd(L[g0:] - t_hi)
            H[g0:] = nu(H[g0:] - t_lo)

    def times_pos(aI, lo, hi):
        return nd(aI.lo * lo), nu(aI.hi * hi)

    def mul_pos_right(A_lo, A_hi, E_lo, E_hi):
        """A * E with E > 0, A of any sign."""
        lo = nd(np.where(A_lo >= 0.0, A_lo * E_lo, A_lo * E_hi))
        hi = nu(np.where(A_hi >= 0.0, A_hi * E_hi, A_hi * E_lo))
        return lo, hi

    for h, lo in zip(hs, los):
        hi = lo + c * dq
        loI, hiI = ivF(lo), ivF(hi)
        g0 = int(np.searchsorted(tg1, loI.lo, 'left'))
        if g0 >= nC:
            continue
        t1 = tg1[g0:]
        rh_lo = nd(t1 - loI.hi)
        rh_hi = nu(t1 - loI.lo)
        act = rh_lo >= 0.0
        amb = np.nonzero((~act) & (rh_hi >= 0.0))[0]
        for j in amb:                                   # exact decision
            n_amb += 1
            act[j] = Fr(float(t1[j])) >= lo
        if not act.any():
            continue
        ia = int(np.argmax(act))                        # activation is a suffix
        assert act[ia:].all()
        ga = g0 + ia
        t0 = tg[ga:]
        t1 = tg1[ga:]
        rh_lo = np.maximum(rh_lo[ia:], 0.0)
        rh_hi = rh_hi[ia:]
        eLo = iexp(lam * loI)
        eHi = iexp(lam * hiI)
        EH_lo = nd(ET1.lo[ga:] * eLo.lo)
        EH_hi = np.minimum(nu(ET1.hi[ga:] * eLo.hi), 1.0)
        rr = IA(nd(t0 - hiI.hi), nu(t0 - hiI.lo))
        RL_lo = np.maximum(rr.lo, 0.0)
        RL_hi = np.maximum(rr.hi, 0.0)
        EL = exp_neg_lam_R(ET0[ga:], rr, eHi, lam)
        EL_lo, EL_hi = EL.lo, EL.hi
        # h_e(r) = r e^{-lam r}, peak 1/(lam e) at 1/lam
        HL_lo, HL_hi = nd(RL_lo * EL_lo), nu(RL_hi * EL_hi)
        HH_lo, HH_hi = nd(rh_lo * EH_lo), nu(rh_hi * EH_hi)
        poss = (RL_lo < R1.hi) & (rh_hi > R1.lo)
        cert = (RL_hi <= R1.lo) & (rh_lo >= R1.hi)
        me_lo = np.maximum(np.maximum(HL_lo, HH_lo), np.where(cert, PE.lo, 0.0))
        me_hi = np.maximum(np.maximum(HL_hi, HH_hi), np.where(poss, PE.hi, 0.0))
        # h_eps(r) = (1 - lam r) e^{-lam r}, trough -e^{-2} at 2/lam
        AL = (nd(1.0 - nu(lam_hi * RL_hi)), nu(1.0 - nd(lam_lo * RL_lo)))
        AH = (nd(1.0 - nu(lam_hi * rh_hi)), nu(1.0 - nd(lam_lo * rh_lo)))
        XL = mul_pos_right(AL[0], AL[1], EL_lo, EL_hi)
        XH = mul_pos_right(AH[0], AH[1], EH_lo, EH_hi)
        xmax_lo = np.maximum(XL[0], XH[0])
        xmax_hi = np.maximum(XL[1], XH[1])
        poss = (RL_lo < R2.hi) & (rh_hi > R2.lo)
        cert = (RL_hi <= R2.lo) & (rh_lo >= R2.hi)
        xmin_lo = np.minimum(np.minimum(XL[0], XH[0]), np.where(poss, TRx.lo, np.inf))
        xmin_hi = np.minimum(np.minimum(XL[1], XH[1]), np.where(cert, TRx.hi, np.inf))
        # h_f(r) = lam (2 - lam r) e^{-lam r}, trough -lam e^{-3} at 3/lam
        BL = (nd(2.0 - nu(lam_hi * RL_hi)), nu(2.0 - nd(lam_lo * RL_lo)))
        BH = (nd(2.0 - nu(lam_hi * rh_hi)), nu(2.0 - nd(lam_lo * rh_lo)))

        def lamx(B):
            return (nd(np.where(B[0] >= 0.0, lam_lo * B[0], lam_hi * B[0])),
                    nu(np.where(B[1] >= 0.0, lam_hi * B[1], lam_lo * B[1])))
        BLl = lamx(BL)
        BHl = lamx(BH)
        FL = mul_pos_right(BLl[0], BLl[1], EL_lo, EL_hi)
        FH = mul_pos_right(BHl[0], BHl[1], EH_lo, EH_hi)
        fmax_lo = np.maximum(FL[0], FH[0])
        fmax_hi = np.maximum(FL[1], FH[1])
        poss = (RL_lo < R3.hi) & (rh_hi > R3.lo)
        cert = (RL_hi <= R3.lo) & (rh_lo >= R3.hi)
        fmin_lo = np.minimum(np.minimum(FL[0], FH[0]), np.where(poss, TRf.lo, np.inf))
        fmin_hi = np.minimum(np.minimum(FL[1], FH[1]), np.where(cert, TRf.hi, np.inf))
        # M = dbar max(...)
        Me = times_pos(dI, me_lo, me_hi)
        Mxp = times_pos(dI, np.maximum(xmax_lo, 0.0), np.maximum(xmax_hi, 0.0))
        Mxm = times_pos(dI, np.maximum(-xmin_hi, 0.0), np.maximum(-xmin_lo, 0.0))
        Mfp = times_pos(dI, np.maximum(fmax_lo, 0.0), np.maximum(fmax_hi, 0.0))
        Mfm = times_pos(dI, np.maximum(-fmin_hi, 0.0), np.maximum(-fmin_lo, 0.0))
        aI = ivF(abs(h))
        if h > 0:
            add('e_hi', ga, *times_pos(aI, *Me), +1)
            add('x_hi', ga, *times_pos(aI, *Mxp), +1)
            add('x_lo', ga, *times_pos(aI, *Mxm), -1)
            add('f_hi', ga, *times_pos(aI, *Mfp), +1)
            add('f_lo', ga, *times_pos(aI, *Mfm), -1)
        elif h < 0:
            add('e_lo', ga, *times_pos(aI, *Me), -1)
            add('x_hi', ga, *times_pos(aI, *Mxm), +1)
            add('x_lo', ga, *times_pos(aI, *Mxp), -1)
            add('f_hi', ga, *times_pos(aI, *Mfm), +1)
            add('f_lo', ga, *times_pos(aI, *Mfp), -1)
    out = {k: IA(acc[k][0], acc[k][1]) for k in names}
    out['n_jumps'] = len(hs)
    out['n_activation_decided_exactly'] = n_amb
    return out


def pulse_norms(pb):
    """P^e, P^eps, P^f: sup over the cells of the magnitudes."""
    out = {}
    for key, (a, b) in (('P_e', ('e_hi', 'e_lo')), ('P_eps', ('x_hi', 'x_lo')),
                        ('P_f', ('f_hi', 'f_lo'))):
        u, l = pb[a], pb[b]
        mag = IA(np.maximum(u.lo, -l.hi), np.maximum(u.hi, -l.lo))
        out[key] = (ia_max_all(mag), int(np.argmax(mag.hi)), mag)
    return out


# ============================================================ constants
def constants(cfg, S, dq):
    """IV constants of the certificate (exact decimal design values)."""
    n = S.n
    K = OrderedDict()
    lam = ivQ(cfg['lam'])
    K['lam'] = lam
    K['beta'] = lam * lam
    K['gamma'] = IV(2.0) * lam
    M = Modal2(lam)
    K['M'] = M
    E = iexp(IV(1.0))
    K['E'] = E
    K['d'] = ivF(dq)
    for nm in ('a_b', 'eps_e', 'eps_v', 'eps_det', 'phi', 'T_c', 's_m', 'numpad',
               'vhnd', 'U', 'Uminus', 'sync_tol', 'align_tol', 'gap_tol'):
        K[nm] = ivQ(cfg[nm])
    K['alpha'] = [ivQ(cfg['alpha'])] * n
    K['A_run'] = [imax(*K['alpha'][:i + 1]) for i in range(n)]
    bF = [F(a) / F(hr.FIXED['v_c']) for a in cfg['amax']]
    K['bF'] = bF
    K['b'] = [ivF(x) for x in bF]
    K['PiF'] = [Fr(0)] + [1 / bF[i - 1] - 1 / bF[i] for i in range(1, n)]
    K['Pi'] = [ivF(x) for x in K['PiF']]
    K['eta'] = [IV(0.0)] + [ivF(abs(1 - bF[i] / bF[i - 1])) for i in range(1, n)]
    K['kappa'] = ivQ(hr.FIXED['kappa'])
    UminF = [min(F(cfg['Uminus']), F(a)) for a in cfg['amax']]
    K['UminF'] = UminF
    K['Umin'] = [ivF(x) for x in UminF]
    # takeover state (exact)
    v0 = [F(x) for x in cfg['v0']]
    v_plan = F(cfg['plan']['v_xi'])
    eps0F = [v_plan - v0[0]] + [v0[i - 1] - v0[i] for i in range(1, n)]
    e0F = [Fr(0)] * n if cfg.get('e0') is None else [F(x) for x in cfg['e0']]
    K['eps0F'], K['e0F'] = eps0F, e0F
    K['eps0'] = [ivF(x) for x in eps0F]
    K['e0'] = [ivF(x) for x in e0F]
    K['vt0'] = [ivF(v0[i] - S.v_in) for i in range(n)]
    K['d_s'] = [ivQ(x) for x in cfg['d_s']]
    K['c0'] = [ivQ(x) for x in cfg['c0']]
    # plan quantities (exact)
    K['T_xi'] = ivF(S.T_xi)
    K['t_0'] = ivF(S.t_0)
    K['v_brk'] = ivF(S.v_xi_h)
    K['a_crawl'] = ivF(S.a_crawl)
    K['ramp'] = ivF(F(cfg['a_b']) - S.a_crawl)
    # modal constants
    beta, gamma = K['beta'], K['gamma']
    K['GAIN_E'] = IV(1.0) / (lam * E)
    K['GAIN_X'] = IV(1.0)
    K['GAIN_F'] = gamma
    K['L1_GE'] = M.ilam2
    K['L1_GX'] = IV(2.0) / (lam * E)
    K['TV_KIMP'] = IV(2.0) * gamma + IV(2.0) * lam * iexp(IV(-3.0))
    K['XFREE_PK'] = K['eps_v'] + (lam / E) * K['eps_e']
    K['EFREE_PK'] = K['eps_e'] + K['eps_v'] / (lam * E)
    bx = corner_stats2(M, beta, gamma, K['eps_e'], K['eps_v'])
    K['IE_BOX'], K['IX_BOX'] = bx['int_e'], bx['int_x']
    K['f_box'] = beta * K['eps_e'] + gamma * K['eps_v']
    # Lemma 7.1 kernel constants
    K['R1'] = M.ilam
    K['R2'] = IV(2.0) * M.ilam
    K['R3'] = IV(3.0) * M.ilam
    K['PE'] = IV(1.0) / (lam * E)
    K['TRx'] = -iexp(IV(-2.0))
    K['TRf'] = -(lam * iexp(IV(-3.0)))
    return K


# ============================================================ chain
def chain_iv(K, n, A1, A2, Pe, Px, Pf):
    """certify_headref.chain in interval arithmetic (acquisition chain and
    relayed braking recursion; the T_brk fixed point does not change W)."""
    M = K['M']
    lam, beta, gamma = K['lam'], K['beta'], K['gamma']
    d = K['d']
    alpha, b = K['alpha'], K['b']
    eps0, e0 = K['eps0'], K['e0']
    eps_det, T_c = K['eps_det'], K['T_c']
    z = IV(0.0)
    R = {k: [z] * n for k in ('Lam1', 'Lam2', 'taub', 'tact', 'tset', 'E1', 'ebar',
                              'fbar', 'fsw', 'tvS2', 'x2tot', 'e2bar', 'Tk', 'Dq',
                              'DBq', 'sup_e', 'sup_x', 'sup_f', 'tv_f', 'M1')}
    for k in range(n):
        Lam1 = d * isum(R['Tk'][:k]) + A1[k]
        m = iabs(eps0[k]) + Lam1
        taub = m / alpha[k] + T_c
        tact = z if k == 0 else R['tset'][k - 1] + d
        tset = tact + taub
        Lk = alpha[k] + isum(R['Tk'][:k])
        E1 = m * tact + ipos(isqr(m) - isqr(eps_det)) / (IV(2.0) * alpha[k]) \
            + (eps_det + Lk * T_c) * T_c
        ebar = iabs(e0[k]) + E1
        st = corner_stats2(M, beta, gamma, ebar, eps_det)
        Lam2 = d * isum(R['tvS2'][:k])
        x2 = st['sup_x'] + K['GAIN_X'] * Lam2 + Px[k]
        fbar = st['sup_f'] + K['GAIN_F'] * Lam2 + Pf[k]
        e2bar = st['sup_e'] + K['GAIN_E'] * Lam2 + Pe[k]
        tv2 = st['tv_f'] + K['TV_KIMP'] * (Lam2 + A2[k])
        fsw = beta * ebar + gamma * eps_det
        Tk = IV(2.0) * alpha[k] + fsw + tv2
        vals = dict(Lam1=Lam1, M1=m, taub=taub, tact=tact, tset=tset, E1=E1,
                    ebar=ebar, Lam2=Lam2, x2tot=x2, fbar=fbar, e2bar=e2bar,
                    tvS2=tv2, fsw=fsw, Tk=Tk,
                    Dq=D_quad2(M, beta, gamma, ebar, eps_det, b[k]),
                    DBq=DB_quad2(M, beta, gamma, b[k]),
                    sup_e=st['sup_e'], sup_x=st['sup_x'], sup_f=st['sup_f'],
                    tv_f=st['tv_f'])
        for key, v in vals.items():
            R[key][k] = v
    R['F'] = [isum(R['fbar'][:i]) for i in range(n)]
    # relayed braking recursion
    f_xi = beta * K['eps_e'] + gamma * K['eps_v']
    VH = K['vhnd']
    for key in ('W', 'Wphi', 'epsb', 'eb', 'fb', 'Th', 'IE', 'IX', 'IF', 'Fcum'):
        R[key] = [z] * n
    for k in range(1, n):
        s = isum(R['Th'][1:k]) + isum(R['Fcum'][1:k])
        W = d * (VH + K['ramp'] + f_xi + s)
        Wphi = d * (VH + f_xi + s)
        epsb = K['XFREE_PK'] + K['GAIN_X'] * W
        eb = K['EFREE_PK'] + K['GAIN_E'] * W
        fb = beta * eb + gamma * epsb
        R['W'][k], R['Wphi'][k], R['epsb'][k], R['eb'][k], R['fb'][k] = W, Wphi, epsb, eb, fb
        R['Fcum'][k] = f_xi + isum(R['fb'][1:k + 1])
        IE = K['IE_BOX'] + K['L1_GE'] * W
        IX = K['IX_BOX'] + K['L1_GX'] * W
        IF = beta * IE + gamma * IX
        R['IE'][k], R['IX'][k], R['IF'][k] = IE, IX, IF
        R['Th'][k] = beta * IX + gamma * (W + IF)
    R['f_xi'] = f_xi
    R['Fb'] = [f_xi + isum(R['fb'][1:i]) for i in range(n)]
    a_b = K['a_b']
    R['diverged_excluded'] = bool(all(x.hi <= 50.0 for x in R['epsb']))
    R['closed'] = bool(R['diverged_excluded']
                       and all((a_b - R['Fcum'][k]).lo > 0.0 for k in range(n - 1)))
    R['anet'] = [a_b - R['Fb'][i] for i in range(1, n)]
    R['hops'] = [R['epsb'][i] / (a_b - R['Fb'][i]) for i in range(1, n)]
    R['disp'] = isum(R['hops'])
    R['gaperr'] = [K['EFREE_PK'] + K['GAIN_E'] * R['W'][i]
                   + isqr(R['epsb'][i]) / (IV(2.0) * (a_b - R['Fb'][i]))
                   for i in range(1, n)]
    R['Leps'] = imax(*K['alpha']) + isum(R['Tk'])
    return R


# ============================================================ cells
def make_cells(cfg, S, K):
    """the 10 ms cells of certify_headref.evaluate (float cell ends are the
    cell definition; consecutive cells share their ends exactly)."""
    Dg = float(cfg['cell_dt'])
    NGc = int(math.ceil(S.T_xi_f / Dg - 1e-9))
    gidx = np.arange(NGc)
    tg = gidx * Dg
    tg1 = (gidx + 1) * Dg
    assert np.array_equal(tg1[:-1], tg[1:])
    assert Fr(float(tg1[-1])) >= S.T_xi and tg[0] == 0.0
    bnd = np.append(tg, tg1[-1:])
    ETb = exp_neg_lam_t(K['lam'], bnd)
    T = K['T_xi']
    cell_hi = IA(np.minimum(tg1, T.lo), np.minimum(tg1, T.hi))
    return dict(Dg=Dg, NGc=NGc, tg=tg, tg1=tg1, ET0=ETb[:-1], ET1=ETb[1:],
                cell_hi=cell_hi, TG=IA(tg), NT_boundaries=len(bnd))


def win_max_cells(arr, tg1, wl):
    """rigorous window maximum over the cells g' <= g that meet the window
    [wl_g, t_g1] of cell g (wl an interval array): upper ends over every cell
    with tg1[g'] > wl.lo (possibly meeting), lower ends over the cells with
    tg1[g'] > wl.hi (certainly meeting)."""
    nC = len(tg1)
    G = np.arange(nC)
    ga_p = np.searchsorted(tg1, wl.lo, 'right')
    ga_c = np.searchsorted(tg1, wl.hi, 'right')
    lo = arr.lo.copy()
    hi = arr.hi.copy()
    W = int(np.max(G - ga_p)) if nC else 0
    for dd in range(1, W + 1):
        idx = G - dd
        okp = idx >= np.maximum(ga_p, 0)
        okc = idx >= np.maximum(ga_c, 0)
        idx = np.maximum(idx, 0)
        hi = np.where(okp, np.maximum(hi, arr.hi[idx]), hi)
        lo = np.where(okc, np.maximum(lo, arr.lo[idx]), lo)
    return IA(lo, hi)


# ============================================================ evaluate
def _cond(slack, sense, method, **kw):
    lo, hi = float(slack.lo), float(slack.hi)
    if sense == '>':
        ver, ref = lo > 0.0, hi <= 0.0
    else:
        ver, ref = lo >= 0.0, hi < 0.0
    d = OrderedDict(lo=lo, hi=hi, sense=sense, verified=bool(ver),
                    refuted=bool(ref),
                    status='verified' if ver else ('refuted' if ref else 'undecided'),
                    method=method)
    d.update(kw)
    return d


def _exact_cond(value, ok, sense, method, **kw):
    v = ivF(value)
    d = _cond(v, sense, method, exact=str(value), **kw)
    d['verified'] = bool(ok)
    d['refuted'] = not bool(ok)
    d['status'] = 'verified' if ok else 'refuted'
    return d


def enclose(cfg, dbar=None, entry=True, keep=False, verbose=False):
    """interval enclosure of certify_headref.evaluate(cfg) at the exact age
    bound dbar (Fraction or decimal; default the exact rational of
    cfg['dbar'])."""
    t_start = time.time()
    timing = OrderedDict()
    if 'n' not in cfg or 'U' not in cfg:
        cfg = hr.config(**cfg)
    dq = F(cfg['dbar']) if dbar is None else (dbar if isinstance(dbar, Fr) else F(dbar))
    S = hr.setup(cfg)
    n = S.n
    one_sided = bool(cfg.get('one_sided', True))
    if not one_sided:
        raise NotImplementedError('the HR enclosure covers the one-sided box')
    K = constants(cfg, S, dq)
    M = K['M']
    lam, beta, gamma, d = K['lam'], K['beta'], K['gamma'], K['d']
    out = OrderedDict(dbar=str(dq), dbar_float=float(dq), n=n)
    conds = OrderedDict()
    consts = OrderedDict()
    # ------------------------------------------------ views and cells
    t1 = time.time()
    V = plan_views(S, cfg)
    cells = make_cells(cfg, S, K)
    tg, tg1, TG = cells['tg'], cells['tg1'], cells['TG']
    cell_hi = cells['cell_hi']
    NGc = cells['NGc']
    timing['views_cells_s'] = time.time() - t1
    # ------------------------------------------------ planned pulses
    t1 = time.time()
    zero = IA(np.zeros(NGc))
    pb = [dict(e_hi=zero, e_lo=zero, x_hi=zero, x_lo=zero, f_hi=zero, f_lo=zero)]
    Pe, Px, Pf = [IV(0.0)] * n, [IV(0.0)] * n, [IV(0.0)] * n
    A1F, A2F = [Fr(0)] * n, [Fr(0)] * n
    pulse_info = [None]
    for c in range(1, n):
        p = pulse_enclosure(S, c, dq, K, cells)
        pb.append(p)
        nrm = pulse_norms(p)
        Pe[c], Px[c], Pf[c] = nrm['P_e'][0], nrm['P_eps'][0], nrm['P_f'][0]
        hs, _ = pair_jumps_exact(S, c)
        A2F[c] = dq * sum(abs(h) for h in hs)
        A1F[c] = dq * abs(S.a_r0)
        pulse_info.append(dict(n_jumps=p['n_jumps'],
                               activation_decided_exactly=p['n_activation_decided_exactly'],
                               argmax_cell=dict(P_e=nrm['P_e'][1], P_eps=nrm['P_eps'][1],
                                                P_f=nrm['P_f'][1])))
    A1 = [ivF(x) for x in A1F]
    A2 = [ivF(x) for x in A2F]
    timing['pulses_s'] = time.time() - t1
    # ------------------------------------------------ chain
    t1 = time.time()
    R = chain_iv(K, n, A1, A2, Pe, Px, Pf)
    timing['chain_s'] = time.time() - t1
    tset, tact, Lam1, Lam2, E1 = R['tset'], R['tact'], R['Lam1'], R['Lam2'], R['E1']
    # ------------------------------------------------ premises
    vmin = min(list(S.ref_v) + [S.v_xi_h])
    conds['A_R'] = _exact_cond(vmin, S.prem_R['ok'], '>', 'exact rationals',
                               note='min of v_r at the pieces and v^xi (exact)')
    conds['A_S'] = _exact_cond(Fr(1) if S.prem_S['ok'] else Fr(-1), S.prem_S['ok'],
                               '>', 'exact rationals',
                               note='closure, (S2) and plan consistency, exact; '
                                    'slack reported as +-1')
    first = []
    for c in range(1, n):
        los = [t for (t, h) in S.ref_jumps]
        for u in range(1, c + 1):
            los += [S.t_0 + x for (x, h) in S.sched_jumps[u]]
        first.append(min(los))
    c0a = [ivF(first[c - 1]) - tset[c] for c in range(1, n)]
    conds['C0a'] = _cond(imin(*c0a), '>=', 'interval', per_pair=[pr(x) for x in c0a])
    lastF = max([t for (t, h) in S.ref_jumps]
                + [S.t_0 + x for u in range(1, n) for (x, h) in S.sched_jumps[u]])
    s0b = S.T_xi - (lastF + 2 * (n - 1) * dq)
    conds['C0b'] = _exact_cond(s0b, s0b >= 0, '>=', 'exact rationals')
    s_h8 = (S.v_xi_h - n * F(cfg['eps_v'])) - (n - 1) * dq * max(K['UminF'])
    conds['H8'] = _exact_cond(s_h8, s_h8 > 0, '>', 'exact rationals')
    Leps = R['Leps']
    s_c1 = K['eps_det'] - K['phi'] - Leps * K['T_c']
    s_c1b = K['eps_det'] - Leps * K['T_c']
    conds['C1'] = _cond(s_c1, '>=', 'interval',
                        strict_part=pr(s_c1b), strict_part_verified=bool(s_c1b.lo > 0))
    ps1 = [K['eps0'][k] + Lam1[k] for k in range(n)]
    conds['C2'] = _cond(-imax(*ps1), '>=', 'interval')
    # ------------------------------------------------ headroom, braking, pre-receipt
    a_b = K['a_b']
    anet = R['anet']
    conds['C6'] = _cond(imin(*anet), '>', 'interval', per_pair=[pr(x) for x in anet])
    Fcum = list(R['Fcum'])
    Fcum[0] = R['f_xi']
    Umin, U = K['Umin'], K['U']
    s_blo = [Umin[u] - (a_b + Fcum[u]) for u in range(n)]
    s_bup = [U - (-a_b + Fcum[u]) for u in range(n)]
    conds['C7c_brk_lo'] = _cond(imin(*s_blo), '>=', 'interval', per_unit=[pr(x) for x in s_blo])
    conds['C7c_brk_up'] = _cond(imin(*s_bup), '>=', 'interval', per_unit=[pr(x) for x in s_bup])
    fb, epsb, W, Wphi, Fv, fbar = R['fb'], R['epsb'], R['W'], R['Wphi'], R['F'], R['fbar']
    a_crawl = K['a_crawl']
    Qm = [IV(0.0)] * n
    for u in range(1, n):
        Qm[u] = fb[u] + imax(Fv[u], Qm[u - 1])
    Ppre = [a_crawl + q for q in Qm]
    s_plo = [Umin[u] - Ppre[u] for u in range(1, n)]
    conds['C7c_pre_lo'] = _cond(imin(*s_plo), '>=', 'interval', per_unit=[pr(x) for x in s_plo])
    f_box = K['f_box']
    Qp = [None] * n
    Qn = [None] * n
    fpre_p = [IV(0.0)] * n
    fpre_m = [IV(0.0)] * n
    for u in range(1, n):
        base = f_box + IV(float(u)) * d * (beta * epsb[u] + gamma * fb[u])
        fpre_p[u] = imin(fb[u], base + gamma * Wphi[u])
        fpre_m[u] = imin(fb[u], base + gamma * W[u])
        inflight = IV(float(u)) * f_box
        Qp[u] = fpre_p[u] + (imax(inflight, Qp[u - 1]) if u >= 2 else inflight)
        Qn[u] = fpre_m[u] + (imax(inflight, Qn[u - 1]) if u >= 2 else inflight)
    s_pup = [U - (-a_crawl + Qp[u]) for u in range(1, n)]
    conds['C7c_pre_up'] = _cond(imin(*s_pup), '>=', 'interval', per_unit=[pr(x) for x in s_pup])
    # ------------------------------------------------ velocity floor (C4)
    t1 = time.time()
    corners = []
    for k in range(n):
        e0k = K['e0'][k]
        corners.append([(e, x) for e in (e0k - E1[k], e0k)
                        for x in (-K['eps_det'], IV(0.0))])
    ET0 = cells['ET0']
    Ek = []
    for k in range(n):
        Rraw = TG - tset[k]
        Rk = iapos(Rraw)
        ER = exp_neg_lam_R(ET0, Rraw, iexp(lam * tset[k]), lam)
        env = None
        for (e, x) in corners[k]:
            s = M.tail_sup_signed(x, (-lam) * (x + lam * e), Rk, ER)
            env = s if env is None else iamax(env, s)
        Ek.append(env + Lam2[k] + pb[k]['x_hi'])
    vrmin = V['vr'].ext(TG, cell_hi, 'min')
    vfl = None
    Ecum = None
    vfl_at = None
    for i in range(n):
        Ecum = Ek[i] if Ecum is None else Ecum + Ek[i]
        flr = vrmin - Ecum
        m = ia_min_all(flr)
        if vfl is None or m.lo < vfl.lo:
            vfl_at = dict(unit=i + 1, t=float(tg[int(np.argmin(flr.lo))]))
        vfl = m if vfl is None else imin(vfl, m)
    conds['C4'] = _cond(vfl, '>', 'interval cells', at=vfl_at)
    # ------------------------------------------------ authority (C7a, C7b)
    alpha = K['alpha']
    # FB (2026-09-29): non-strict sign indicators (certify_fb), as intervals:
    # (possibly, certainly) of eps_k(0) +- Lambda^(1)_k reaching +-eps_det
    sigp = [((K['eps0'][k] + Lam1[k]).hi >= K['eps_det'].lo,
             (K['eps0'][k] + Lam1[k]).lo >= K['eps_det'].hi) for k in range(n)]
    sigm = [((K['eps0'][k] - Lam1[k]).lo <= -K['eps_det'].lo,
             (K['eps0'][k] - Lam1[k]).hi <= -K['eps_det'].hi) for k in range(n)]
    Lam2h = [IV(0.0)] + Lam2[1:]
    fpp = [pb[k]['f_hi'] for k in range(n)]
    fpm = [-pb[k]['f_lo'] for k in range(n)]
    t_0 = K['t_0']
    # FB refinement (certify_fb): earliest S2 switch instants as intervals;
    # tlow_k.lo is a rigorous lower bound, and a cell counts as before the
    # switch of layer k only when its (float-defined, exact) end lies
    # strictly below tlow_k.lo
    tlow = []
    t_prev = IV(0.0)
    for k in range(n):
        ps = K['eps0'][k] + Lam1[k]
        if ps.hi <= 0.0:                                 # braking S1 stage
            num = iabs(K['eps0'][k]) - K['eps_det'] - Lam1[k]
            dur = ipos(idiv(num, alpha[k]))
        else:
            dur = IV(0.0)
        t_prev = t_prev + dur
        tlow.append(t_prev)
    if not bool(cfg.get('switch_lower', False)):
        tlow = [IV(0.0)] * n
    consts['tlow_sw'] = [pr(x) for x in tlow]

    def _mask(A, act):
        return IA(A.lo * act, A.hi * act)

    s7a = None
    s7b = None
    at7a = at7b = None
    umax_units, umin_units = [], []
    umax_rep, umin_rep = [], []
    for i in range(n):
        Fp = IA(np.zeros(NGc))
        Fm = IA(np.zeros(NGc))
        s1p_lo = np.zeros(NGc); s1p_hi = np.zeros(NGc)
        s1m_lo = np.zeros(NGc); s1m_hi = np.zeros(NGc)
        for k in range(i + 1):
            shift = IV(float(i - k)) * d + tset[k]
            Rraw = TG - shift
            Rk = iapos(Rraw)
            ER = exp_neg_lam_R(ET0, Rraw, iexp(lam * shift), lam)
            tp = tm = None
            for (e, x) in corners[k]:
                p_ = lam * lam * e + IV(2.0) * lam * x
                q_ = -(lam * lam) * (x + lam * e)
                a_ = M.tail_sup_signed(p_, q_, Rk, ER)
                b_ = M.tail_sup_signed(-p_, -q_, Rk, ER)
                tp = a_ if tp is None else iamax(tp, a_)
                tm = b_ if tm is None else iamax(tm, b_)
            forced = gamma * Lam2h[k]
            wl = TG - IV(float(i - k)) * d
            act = (tg1 >= tlow[k].lo).astype(float)      # FB refinement
            Fp = Fp + _mask(tp + forced + win_max_cells(fpp[k], tg1, wl), act)
            Fm = Fm + _mask(tm + forced + win_max_cells(fpm[k], tg1, wl), act)
            chi_p = tg < shift.hi
            chi_c = tg < shift.lo
            s1p_hi = np.maximum(s1p_hi, np.where(chi_p & sigp[k][0], alpha[k].hi, 0.0))
            s1p_lo = np.maximum(s1p_lo, np.where(chi_c & sigp[k][1], alpha[k].lo, 0.0))
            s1m_hi = np.maximum(s1m_hi, np.where(chi_p & sigm[k][0], alpha[k].hi, 0.0))
            s1m_lo = np.maximum(s1m_lo, np.where(chi_c & sigm[k][1], alpha[k].lo, 0.0))
        Fp = Fp + IA(s1p_lo, s1p_hi)
        Fm = Fm + IA(s1m_lo, s1m_hi)
        # FB (2026-09-29): planned-part window truncated at 0 (certify_fb);
        # the reported envelopes add the zero command on the cells whose
        # window possibly reaches before takeover
        wraw = TG - IV(float(i)) * d
        wa = IA(np.maximum(wraw.lo, 0.0), np.maximum(wraw.hi, 0.0))
        Pp = V['arA'].ext(wa, cell_hi, 'max')
        Pm = V['arA'].ext(wa, cell_hi, 'min')
        for u in range(1, i + 1):
            Pp = Pp + V['rho'][u].ext(wa - t_0, cell_hi - t_0, 'max')
            Pm = Pm + V['rho'][u].ext(wa - t_0, cell_hi - t_0, 'min')
        up_ = Pp + Fp
        lo_ = Pm - Fm
        # the head's window [t, t] never reaches before takeover (its lower
        # end can round below 0 only by outward rounding)
        pre0 = (wraw.lo < 0.0) & (i > 0)
        up_r = IA(np.where(pre0, np.maximum(up_.lo, 0.0), up_.lo),
                  np.where(pre0, np.maximum(up_.hi, 0.0), up_.hi))
        lo_r = IA(np.where(pre0, np.minimum(lo_.lo, 0.0), lo_.lo),
                  np.where(pre0, np.minimum(lo_.hi, 0.0), lo_.hi))
        umax_units.append(ia_max_all(up_))          # the tests (C7a), (C7b)
        umin_units.append(ia_min_all(lo_))
        umax_rep.append(ia_max_all(up_r))           # the reported envelopes
        umin_rep.append(ia_min_all(lo_r))
        sa = ia_min_all(_ia(U) - up_)
        sb = ia_min_all(lo_ + Umin[i])
        if s7a is None or sa.lo < s7a.lo:
            at7a = dict(unit=i + 1, t=float(tg[int(np.argmax(up_.hi))]))
        if s7b is None or sb.lo < s7b.lo:
            at7b = dict(unit=i + 1, t=float(tg[int(np.argmin(lo_.lo))]))
        s7a = sa if s7a is None else imin(s7a, sa)
        s7b = sb if s7b is None else imin(s7b, sb)
    conds['C7a'] = _cond(s7a, '>=', 'interval cells', at=at7a,
                         per_unit_max=[pr(x) for x in umax_units])
    conds['C7b'] = _cond(s7b, '>=', 'interval cells', at=at7b,
                         per_unit_min=[pr(x) for x in umin_units])
    timing['cells_s'] = time.time() - t1
    # ------------------------------------------------ admission, floors
    t1 = time.time()
    b, Pi, eta, kappa = K['b'], K['Pi'], K['eta'], K['kappa']
    x2tot, e2bar, Dq, DBq = R['x2tot'], R['e2bar'], R['Dq'], R['DBq']
    IX, IF, eb = R['IX'], R['IF'], R['eb']
    Fb = R['Fb']
    eps0 = K['eps0']
    NUMPAD = K['numpad']
    s_m = K['s_m']
    v_brk = K['v_brk']
    pairs = OrderedDict()
    adm = {'C8a': [], 'C8b': [], 'C8c': []}
    clr, bfl = [], []
    in2 = np.ones(NGc, bool)
    assert np.all(tg < K['T_xi'].lo)

    def q_bound(c, ta, tb):
        wa = ta - IV(float(c)) * d
        Q = V['arA'].ext(wa, tb, 'max') - V['arA'].ext(wa, tb, 'min')
        for u in range(1, c):
            xa, xb = wa - t_0, tb - t_0
            Q = Q + (V['rho'][u].ext(xa, xb, 'max') - V['rho'][u].ext(xa, xb, 'min'))
        return Q

    def plan_lb(c, ta, tb):
        xa = ta - t_0 - IV(float(c)) * d
        xb = tb - t_0
        Aown = V['A'][c].ext(xa, xb, 'min')
        Bown = V['B'][c].ext(xa, xb, 'min')
        gown = V['gs'][c].ext(xa, xb, 'min')
        mode = 'max' if K['PiF'][c] < 0 else 'min'
        Vx = V['vr'].ext(ta, tb, mode)
        Mx = V['arkvr'].ext(ta, tb, mode)
        for u in range(1, c):
            xau = ta - t_0 - IV(float(u)) * d
            Vx = Vx + V['negdd'][u].ext(xau, xb, mode)
            Mx = Mx + V['rkd'][u].ext(xau, xb, mode)
        return Aown + Vx * Pi[c], Bown + Mx * Pi[c], gown

    def red(x, poss, cert, mode):
        """min/max over the cells of a window whose last cell is undecided:
        outer bound over the possible cells, inner bound over the certain
        ones."""
        if mode == 'max':
            return IV(float(np.max(x.lo[cert])) if cert.any() else -INF,
                      float(np.max(x.hi[poss])))
        return IV(float(np.min(x.lo[poss])),
                  float(np.min(x.hi[cert])) if cert.any() else INF)

    for c in range(1, n):
        V1 = isum([iabs(eps0[k]) + Lam1[k] + x2tot[k] for k in range(c)])
        H1 = E1[c] + (alpha[c] / b[c]) * R['taub'][c] + V1 * iabs(Pi[c]) + Lam1[c] / b[c]
        V2 = IV(2.0) * isum(x2tot[:c])
        L2tot = Lam2[c] + A2[c]
        Ht = Dq[c] + V2 * iabs(Pi[c]) + L2tot * (IV(1.0) / b[c] + DBq[c])
        M1 = iabs(eps0[c]) + Lam1[c]
        ts_ = tset[c]
        poss1 = tg < ts_.hi
        cert1 = tg < ts_.lo
        hi1 = IA(np.minimum(tg1, ts_.lo), np.minimum(tg1, ts_.hi))
        sel = poss1
        Qc1 = q_bound(c, TG[sel], hi1[sel])
        Q1 = red(Qc1, np.ones(int(sel.sum()), bool), cert1[sel], 'max')
        Qc2 = q_bound(c, TG, cell_hi)
        Q2 = ia_max_all(Qc2)
        hm1_coast = (b[c] * M1 + (Fv[c] + K['A_run'][c - 1] + Q1) * eta[c]) / (kappa * b[c])
        hm1_s1 = (alpha[c] + b[c] * M1 + (Fv[c] + Q1) * eta[c]) / (kappa * b[c])
        hm1 = imax(hm1_coast, hm1_s1)
        hm2 = (fbar[c] + b[c] * x2tot[c] + (Fv[c] + Q2) * eta[c]) / (kappa * b[c])
        bmin = imin(b[c], b[c - 1])
        Vbrk = v_brk + IV(float(c)) * K['eps_v']
        Vrise = IV(float(c - 1)) * d * ipos(-a_crawl + Qm[c - 1])
        Ph2 = IX[c] + epsb[c] / bmin + isqr(epsb[c]) / (IV(2.0) * anet[c - 1]) \
            + IF[c] / b[c] + Vbrk * ipos(Pi[c]) + Vrise * ipos(-Pi[c]) \
            + W[c] * (IV(1.0) / b[c] + DBq[c])
        dem_prev = a_crawl + Fv[c - 1] + imax(K['A_run'][c - 1], fbar[c - 1])
        Ubrk = imax(dem_prev, a_crawl + Qm[c - 1], a_b + Fb[c], (a_b + Fb[c - 1]) + fbar[c - 1])
        hmb_s2 = (fb[c] + b[c] * epsb[c] + Ubrk * eta[c]) / (kappa * b[c])
        hmb_rule = ipos(b[c] * epsb[c] + (b[c] / b[c - 1]) * (a_b + Fb[c]) - a_b) / (kappa * b[c])
        hmb = imax(hmb_s2, hmb_rule)
        ht0 = K['e0'][c] + eps0[c] / b[c] + K['vt0'][c - 1] * Pi[c]
        hl1, ml1, gl1 = plan_lb(c, TG[sel], hi1[sel])
        hl2, ml2, gl2 = plan_lb(c, TG, cell_hi)
        allp = np.ones(int(sel.sum()), bool)
        mu1 = red(ml1, allp, cert1[sel], 'min')
        hs1 = red(hl1, allp, cert1[sel], 'min')
        gs1 = red(gl1, allp, cert1[sel], 'min')
        mu2 = ia_min_all(ml2)
        hs2 = ia_min_all(hl2)
        hstar_T = (K['d_s'][c] - s_m) + v_brk * Pi[c]
        s_i = ht0 - H1 + mu1 / kappa - hm1 - NUMPAD
        s_ii = ht0 - H1 - Ht + mu2 / kappa - hm2 - NUMPAD
        hb = hstar_T + ht0 - H1 - Ht - Ph2
        s_iii = hb - hmb - NUMPAD
        adm['C8a'].append(s_i)
        adm['C8b'].append(s_ii)
        adm['C8c'].append(s_iii)
        gmin1 = gs1 + K['e0'][c] - E1[c]
        gmin2 = (K['d_s'][c] - s_m) - e2bar[c]
        gmin3 = (K['d_s'][c] - s_m) - eb[c] - isqr(epsb[c]) / (IV(2.0) * a_b)
        hf1 = hs1 + ht0 - H1
        hf2 = hs2 + ht0 - H1 - Ht
        bfl.append(imin(hf1, hf2))
        g_dir = imin(gmin1, gmin2, gmin3)
        g_init = K['c0'][c - 1] - s_m
        if K['PiF'][c] <= 0:
            g_bar = imin(g_init, hf1, hf2, hb)
            gm = imax(g_dir, g_bar)
        else:
            g_bar = None
            gm = g_dir
        clr.append(gm)
        sigma = b[c] * imin(s_i + NUMPAD, s_ii + NUMPAD, s_iii + NUMPAD)
        pairs[c + 1] = OrderedDict(
            (k_, pr(v_)) for k_, v_ in (
                ('h_tilde0', ht0), ('H1', H1), ('H2', Ht), ('Hb', Ph2), ('hm1', hm1),
                ('hm2', hm2), ('hmb', hmb), ('hmb_s2', hmb_s2), ('hmb_rule', hmb_rule),
                ('Ubrk', Ubrk), ('V1', V1), ('V2', V2), ('Vbrk', Vbrk), ('Vrise', Vrise),
                ('Q1', Q1), ('Q2', Q2), ('mu1', mu1), ('mu2', mu2), ('hstar_min1', hs1),
                ('hstar_min2', hs2), ('gstar_min1', gs1), ('hstar_T', hstar_T),
                ('slack_a', s_i), ('slack_b', s_ii), ('slack_c', s_iii), ('h_b', hb),
                ('gmin_acq', gmin1), ('gmin_s2', gmin2), ('gmin_brk', gmin3),
                ('gmin_direct', g_dir), ('gmin', gm), ('hfloor_acq', hf1),
                ('hfloor_s2', hf2), ('sigma', sigma), ('L2tot', L2tot)))
        if g_bar is not None:
            pairs[c + 1]['gmin_barrier'] = pr(g_bar)
    for nm in ('C8a', 'C8b', 'C8c'):
        conds[nm] = _cond(imin(*adm[nm]), '>', 'interval',
                          per_pair=[pr(x) for x in adm[nm]])
    conds['C9a'] = _cond(imin(*clr), '>', 'interval', per_pair=[pr(x) for x in clr])
    conds['C9b'] = _cond(imin(*bfl), '>=', 'interval', per_pair=[pr(x) for x in bfl])
    timing['admission_s'] = time.time() - t1
    # ------------------------------------------------ terminal
    disp = R['disp']
    gaperr = R['gaperr']
    eps_e, eps_v = K['eps_e'], K['eps_v']
    m1 = eps_e + v_brk * eps_v / a_b + isqr(eps_v) / (IV(2.0) * a_b)
    marker = [m1] + [m1 + isum(gaperr[:i]) for i in range(1, n)]
    conds['C10'] = _cond(K['sync_tol'] - disp, '>=', 'interval')
    conds['C11'] = _cond(K['align_tol'] - marker[-1], '>=', 'interval')
    ge = imax(*gaperr)
    conds['C12'] = _cond(K['gap_tol'] - ge, '>=', 'interval')
    epsb_max = imax(*epsb[1:])
    T_f = K['T_xi'] + (v_brk + eps_v) / a_b + IV(float(n - 1)) * epsb_max / a_b
    # ------------------------------------------------ entry layer (C3, C5)
    ent = None
    if entry:
        t1 = time.time()
        ent = entry_layer(cfg, S, K, R, dq, pb, cells, A2F, keep_arrays=keep)
        conds['C3'] = ent['C3']
        conds['C5'] = ent['C5']
        timing['entry_s'] = time.time() - t1
    order = ('A_R', 'A_S', 'C0a', 'C0b', 'H8', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
             'C7a', 'C7b', 'C7c_brk_lo', 'C7c_brk_up', 'C7c_pre_lo', 'C7c_pre_up',
             'C8a', 'C8b', 'C8c', 'C9a', 'C9b', 'C10', 'C11', 'C12')
    out['closed'] = R['closed']
    out['conds'] = OrderedDict((k_, conds[k_]) for k_ in order if k_ in conds)
    out['verified_all'] = bool(all(c_['verified'] for c_ in out['conds'].values())
                               and R['closed'] and len(out['conds']) == len(order))
    out['refuted'] = [k_ for k_, c_ in out['conds'].items() if c_['refuted']]
    out['undecided'] = [k_ for k_, c_ in out['conds'].items() if c_['status'] == 'undecided']
    # ------------------------------------------------ constants and bounds
    for key in ('Lam1', 'Lam2', 'taub', 'tact', 'tset', 'E1', 'ebar', 'sup_e', 'sup_x',
                'sup_f', 'tv_f', 'x2tot', 'e2bar', 'fbar', 'tvS2', 'Tk', 'fsw', 'F',
                'Dq', 'DBq'):
        consts[key] = [pr(x) for x in R[key]]
    for key in ('W', 'Wphi', 'epsb', 'eb', 'fb', 'Th', 'IE', 'IX', 'IF'):
        consts[key] = [pr(x) for x in R[key][1:]]
    consts['A1'] = [pr(x) for x in A1]
    consts['A1_exact'] = [str(x) for x in A1F]
    consts['A2'] = [pr(x) for x in A2]
    consts['A2_exact'] = [str(x) for x in A2F]
    consts['P_e'] = [pr(x) for x in Pe]
    consts['P_eps'] = [pr(x) for x in Px]
    consts['P_f'] = [pr(x) for x in Pf]
    consts['pulses'] = pulse_info[1:]
    consts['Fb_units'] = [pr(x) for x in Fcum]
    consts['f_xi'] = pr(R['f_xi'])
    consts['anet'] = [pr(x) for x in anet]
    consts['q'] = [pr(x) for x in R['hops']]
    consts['Leps'] = pr(Leps)
    consts['Qm'] = [pr(x) for x in Qm]
    consts['P_pre'] = [pr(x) for x in Ppre]
    consts['Q_plus'] = [None] + [pr(x) for x in Qp[1:]]
    consts['Q_minus'] = [None] + [pr(x) for x in Qn[1:]]
    consts['fpre_plus'] = [pr(x) for x in fpre_p]
    consts['fpre_minus'] = [pr(x) for x in fpre_m]
    for key in ('GAIN_E', 'L1_GE', 'L1_GX', 'TV_KIMP', 'XFREE_PK', 'EFREE_PK', 'IE_BOX',
                'IX_BOX', 'f_box', 'lam', 'beta', 'gamma', 'd'):
        consts[key] = pr(K[key])
    consts['pairs'] = pairs
    consts['modal_undecided_branches'] = M.undecided
    consts['modal_undecided_calls'] = M.undecided_calls
    bounds = OrderedDict(
        dispersion=pr(disp), T_f_bar=pr(T_f), marker=[pr(x) for x in marker],
        marker_max=pr(marker[-1]), gap=[pr(x) for x in gaperr], gap_max=pr(ge),
        schedule_error=pr(eps_v / a_b + disp),
        clearance_floor=[pr(x) for x in clr], clearance_floor_min=pr(imin(*clr)),
        velocity_floor=pr(vfl), switch_deadline=[pr(x) for x in tset],
        activation_deadline=[pr(x) for x in tact],
        u_max_before_T_xi=[pr(x) for x in umax_rep],
        u_min_before_T_xi=[pr(x) for x in umin_rep],
        u_max_pre_receipt=[None] + [pr(-a_crawl + Qp[u]) for u in range(1, n)],
        u_band_post_receipt=[[pr(-a_b - Fcum[u]), pr(-a_b + Fcum[u])] for u in range(n)],
        admission_slack_sigma=[pairs[c + 1]['sigma'] for c in range(1, n)])
    if ent is not None:
        bounds['entry_deadline'] = ent['T_ent_bar']
        bounds['T_hold'] = ent['T_hold']
        bounds['Vhnd_max'] = ent['Vhnd_max']
    out['consts'] = consts
    out['bounds'] = bounds
    if ent is not None:
        out['entry'] = {k_: v_ for k_, v_ in ent.items() if k_ not in ('C3', 'C5', 'arrays')}
    timing['total_s'] = time.time() - t_start
    out['timing'] = timing
    if keep:
        out['_pb'] = pb
        out['_R'] = R
        out['_K'] = K
        out['_cells'] = cells
        if ent is not None and 'arrays' in ent:
            out['_entry_arrays'] = ent['arrays']
    if verbose:
        for k_, c_ in out['conds'].items():
            print('  %-11s [%+.9e, %+.9e] %s' % (k_, c_['lo'], c_['hi'], c_['status']))
    return out


# ============================================================ entry layer
def entry_layer(cfg, S, K, R, dq, pb, cells, A2F, keep_arrays=False):
    """Interval enclosure of the entry layer of certify_headref.evaluate
    (tail recursion entry_curves with the planned L1 tails, time-resolved
    entry deadline with the planned cell bounds, hold C3, handoff pad C5),
    with the method of run_witness_caseB_entry.py (closed-form helpers
    tail_sup, tail_tv, Psi, kE, kX, Grid, shift, flo, fhi imported
    unchanged): exact grid T_k = k/1000 s (k < 210000), Arb scalars from the
    exact constants and from the interval chain, outward-rounded double
    arrays.  The float evaluator uses the rounded grid np.arange(0, 210,
    0.001) and the guards 1e-15 (ledgers) and 1e-12 (indicators), which the
    certificate does not contain.

    Upper arrays (VH, BeH, ...) bound the certificate's envelopes from
    above, lower arrays from below.  The entry deadline of unit i is a point
    k/1000 s of the fixed grid in [tswb_i, T_xi] (supplement S.II-E); the
    verified (upper) index is the first candidate at which both upper running
    suprema (from the right, up to the last grid point <= T_xi) pass, and the
    lower index is the first candidate at which the lower running suprema do
    not certainly fail."""
    CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import run_witness_caseB_entry as W                   # noqa: E402
    from flint import arb, ctx
    ctx.prec = 200
    n = S.n
    lags = tuple(int(x) for x in cfg.get('lags', (15, 20)))
    nB, nA = lags
    BSET = [2 * m for m in range(nB)]
    ASET = [2 * m for m in range(nA)]
    N, H = W.N, W.H
    lam_q = F(cfg['lam'])
    P = dict(lam_=W.A(lam_q), beta_=W.A(lam_q * lam_q), gamma_=W.A(2 * lam_q))
    lam, beta, gamma = P['lam_'], P['beta_'], P['gamma_']
    key = ('grid', str(lam_q))
    if key not in _GRID:
        _GRID[key] = W.Grid(lam_q)
    G = _GRID[key]
    kk = G.k

    def ball(x):
        """Arb ball containing the double interval x (IV)."""
        return arb(x.lo).union(arb(x.hi))
    eps_det = W.A(F(cfg['eps_det']))
    d = W.A(dq)
    K_hi = math.ceil(dq / H)
    K_lo = math.floor(dq / H)
    # ---- four-corner closed-form tails per unit (symmetric box) -----------
    SeL, SeH, SxL, SxH, FTL, FTH = [], [], [], [], [], []
    for k in range(n):
        eb = ball(R['ebar'][k])
        acc = None
        for se in (1, -1):
            for sx in (1, -1):
                e0c = se * eb
                x0c = sx * eps_det
                c = x0c + lam * e0c
                r = (W.tail_sup(e0c, c, lam, G),
                     W.tail_sup(x0c, -lam * c, lam, G),
                     W.tail_tv(-(beta * e0c + gamma * x0c),
                               -(beta - gamma * lam) * c, lam, G))
                if acc is None:
                    acc = [list(x) for x in r]
                else:
                    for a_, b_ in zip(acc, r):
                        a_[0] = np.maximum(a_[0], b_[0])
                        a_[1] = np.maximum(a_[1], b_[1])
        SeL.append(acc[0][0]); SeH.append(acc[0][1])
        SxL.append(acc[1][0]); SxH.append(acc[1][1])
        FTL.append(acc[2][0]); FTH.append(acc[2][1])
    # ---- planned L1 tails A^tail_c(T_k) = dbar sum_{hi_J >= T_k} |h_J| -------
    AtL, AtH = [None], [None]
    for c in range(1, n):
        hs, los = pair_jumps_exact(S, c)
        his = [lo + c * dq for lo in los]
        order = sorted(range(len(hs)), key=lambda j: his[j])
        hs_s = [abs(hs[j]) for j in order]
        his_s = [his[j] for j in order]
        suf = [Fr(0)] * (len(hs_s) + 1)
        for j in range(len(hs_s) - 1, -1, -1):
            suf[j] = suf[j + 1] + hs_s[j]
        # grid index of the first T_k > hi_J: jumps with hi_J >= T_k count
        idx = [math.floor(h_ / H) for h_ in his_s]          # T_k <= hi_J <=> k <= idx
        lo_arr = np.zeros(N)
        hi_arr = np.zeros(N)
        # A^tail(T_k) = dbar * sum_{j: idx_j >= k} |h_j|; piecewise constant
        # in k with breaks after each idx_j (sorted)
        prev = 0
        for j in range(len(hs_s)):
            kend = min(idx[j], N - 1)
            if kend >= prev:
                val = dq * suf[j]
                iv = ivF(val)
                lo_arr[prev:kend + 1] = iv.lo
                hi_arr[prev:kend + 1] = iv.hi
                prev = kend + 1
        AtL.append(lo_arr)
        AtH.append(hi_arr)
    # ---- certified command tails V_j (HR recursion, entry_curves) -----------
    gd = gamma * d
    L1FP = W.Psi(0, P)
    c2 = L1FP * d
    PsiB = {B: W.Psi(B, P) for B in BSET}
    VH = [np.zeros(N)]
    VL = [np.zeros(N)]
    alpha = W.A(F(cfg['alpha']))
    for j in range(1, n + 1):
        jj = j - 1
        VsH = W.shift(VH[jj], K_hi)
        VsL = W.shift(VL[jj], K_lo)
        tset_b, tact_b = ball(R['tset'][jj]), ball(R['tact'][jj])
        s_hi = W._ceil_idx(W.fhi(tset_b))
        s_lo = W._ceil_idx(W.flo(tset_b))
        ftH = FTH[jj][np.maximum(kk - s_hi, 0)]
        ftL = FTL[jj][np.maximum(kk - s_lo, 0)]
        ajf = alpha + ball(R['fsw'][jj])
        jmpH = W.upA(np.where(kk <= W._floor_idx(W.fhi(tact_b)), W.fhi(alpha), 0.0)
                     + np.where(kk <= W._floor_idx(W.fhi(tset_b)), W.fhi(ajf), 0.0))
        jmpL = W.dnA(np.where(kk <= W._floor_idx(W.flo(tact_b)), W.flo(alpha), 0.0)
                     + np.where(kk <= W._floor_idx(W.flo(tset_b)), W.flo(ajf), 0.0))
        if jj == 0:
            forcH = np.zeros(N)
            forcL = np.zeros(N)
        else:
            L2 = ball(R['Lam2'][jj])
            Wtot = L2 + W.A(A2F[jj])
            # gamma (dbar V_{j-1}((t - dbar)^+) + A^tail_j(t))
            forcH = W.upA(W.upA(W.fhi(gd) * VsH) + W.upA(W.fhi(gamma) * AtH[jj]))
            forcL = W.dnA(W.dnA(W.flo(gd) * VsL) + W.dnA(W.flo(gamma) * AtL[jj]))
            ctH = np.full(N, np.inf)
            ctL = np.full(N, np.inf)
            for B in BSET:
                c1 = PsiB[B] * Wtot
                sB = 1000 * B
                tH = W.upA(W.upA(W.fhi(c1) + W.upA(W.fhi(c2) * W.shift(VH[jj], K_hi + sB)))
                           + W.upA(W.fhi(L1FP) * W.shift(AtH[jj], sB)))
                tL = W.dnA(W.dnA(W.flo(c1) + W.dnA(W.flo(c2) * W.shift(VL[jj], K_lo + sB)))
                           + W.dnA(W.flo(L1FP) * W.shift(AtL[jj], sB)))
                ctH = np.minimum(ctH, tH)
                ctL = np.minimum(ctL, tL)
            forcH = W.upA(forcH + ctH)
            forcL = W.dnA(forcL + ctL)
        VH.append(W.upA(W.upA(W.upA(VsH + ftH) + jmpH) + forcH))
        VL.append(W.dnA(W.dnA(W.dnA(VsL + ftL) + jmpL) + forcL))
    # ---- state envelopes, planned parts, entry lags --------------------------
    KEinf = 1 / (lam * W.E1_)
    ee_lo, ee_hi = W.flo(W.A(F(cfg['eps_e']))), W.fhi(W.A(F(cfg['eps_e'])))
    ev_lo, ev_hi = W.flo(W.A(F(cfg['eps_v']))), W.fhi(W.A(F(cfg['eps_v'])))
    kEs = {a: W.kE(a, lam) for a in ASET}
    kXs = {a: W.kX(a, lam) for a in ASET}
    T_xi_q = S.T_xi
    tg, tg1 = cells['tg'], cells['tg1']
    Dg_cells = len(tg)
    rows = []
    k_star, k_low = [], []
    for i in range(n):
        BeH, BeL = SeH[i].copy(), SeL[i].copy()
        BxH, BxL = SxH[i].copy(), SxL[i].copy()
        if i >= 1:
            L2 = ball(R['Lam2'][i])
            fEH = np.full(N, np.inf); fEL = np.full(N, np.inf)
            fXH = np.full(N, np.inf); fXL = np.full(N, np.inf)
            dE = KEinf * d
            for a in ASET:
                VshH = W.shift(VH[i], K_hi + 1000 * a)
                VshL = W.shift(VL[i], K_lo + 1000 * a)
                cE = kEs[a] * L2
                cX = kXs[a] * L2
                fEH = np.minimum(fEH, W.upA(W.fhi(cE) + W.upA(W.fhi(dE) * VshH)))
                fEL = np.minimum(fEL, W.dnA(W.flo(cE) + W.dnA(W.flo(dE) * VshL)))
                fXH = np.minimum(fXH, W.upA(W.fhi(cX) + W.upA(W.fhi(d) * VshH)))
                fXL = np.minimum(fXL, W.dnA(W.flo(cX) + W.dnA(W.flo(d) * VshL)))
            BeH = W.upA(BeH + fEH); BeL = W.dnA(BeL + fEL)
            BxH = W.upA(BxH + fXH); BxL = W.dnA(BxL + fXL)
        tset_i = R['tset'][i]
        # candidate instants t_k = k/1000 s of the fixed grid in
        # [tswb_i, T_xi] (supplement S.II-E, eq. (seq:tent)); they do not
        # move with dbar.  Upper test (certain pass): candidates k >= ceil of
        # the upper end of tswb_i, envelopes read at the smallest possible
        # lag floor(t_k - tswb_i) = k - ceil(tswb_i.hi / H).  Lower test
        # (certain failure): candidates k >= ceil(tswb_i.lo / H), envelopes
        # at the largest possible lag k - ceil(tswb_i.lo / H).
        kK = min(math.floor(T_xi_q / H), N - 1)
        cH = math.ceil(Fr(tset_i.hi) / H)
        cL = math.ceil(Fr(tset_i.lo) / H)
        if kK < cH:
            rows.append(dict(unit=i + 1, note='T_xi before the switch deadline'))
            k_star.append(None); k_low.append(None)
            continue
        kkH = np.arange(cH, kK + 1)
        kkL = np.arange(cL, kK + 1)
        BeH, BxH = BeH[kkH - cH].copy(), BxH[kkH - cH].copy()
        BeL, BxL = BeL[kkL - cL].copy(), BxL[kkL - cL].copy()
        if i >= 1:
            pe = IA(np.maximum(pb[i]['e_hi'].lo, -pb[i]['e_lo'].hi),
                    np.maximum(pb[i]['e_hi'].hi, -pb[i]['e_lo'].lo))
            px = IA(np.maximum(pb[i]['x_hi'].lo, -pb[i]['x_lo'].hi),
                    np.maximum(pb[i]['x_hi'].hi, -pb[i]['x_lo'].lo))
            # the closed cells [g Dg, (g+1) Dg] that meet [t_k, t_k + H]
            # (exact, integer arithmetic, rc grid steps per cell): cell
            # k//rc, the cell before it if rc divides k, and the cell after
            # it if rc divides k+1.  The same set serves the upper and the
            # lower envelope (it also contains every cell with the float
            # ends of make_cells that contains a point of [t_k, t_k + H]).
            rc = int(round(float(cfg['cell_dt']) / float(H)))
            assert Fr(str(cfg['cell_dt'])) == rc * H

            def _cells_of(kk):
                a = np.clip(kk // rc - (kk % rc == 0).astype(int), 0, Dg_cells - 1)
                b = np.clip(kk // rc + ((kk + 1) % rc == 0).astype(int), 0, Dg_cells - 1)
                return a, b
            ga, gb = _cells_of(kkH)
            BeH = W.upA(BeH + np.maximum(pe.hi[ga], pe.hi[gb]))
            BxH = W.upA(BxH + np.maximum(px.hi[ga], px.hi[gb]))
            ga, gb = _cells_of(kkL)
            BeL = W.dnA(BeL + np.maximum(pe.lo[ga], pe.lo[gb]))
            BxL = W.dnA(BxL + np.maximum(px.lo[ga], px.lo[gb]))
        RSEH = W._rev_cummax(BeH); RSXH = W._rev_cummax(BxH)
        RSEL = W._rev_cummax(BeL); RSXL = W._rev_cummax(BxL)
        passH = (RSEH <= ee_lo) & (RSXH <= ev_lo)
        failL = (RSEL > ee_hi) | (RSXL > ev_hi)
        ks = cH + int(np.argmax(passH)) if passH.any() else None
        kl = cL + int(np.argmax(~failL)) if (~failL).any() else None
        k_star.append(ks)
        k_low.append(kl)
        rows.append(dict(unit=i + 1, k_star=ks, k_low=kl,
                         identified_exactly=(ks is not None and ks == kl),
                         candidates=[cL, cH, kK],
                         pass_margin_e=None if ks is None else float(W.dnA(ee_lo - RSEH[ks - cH])),
                         pass_margin_eps=None if ks is None else float(W.dnA(ev_lo - RSXH[ks - cH])),
                         fails_one_step_before=None if (ks is None or ks - 1 < cL)
                         else bool(failL[ks - 1 - cL])))
        if ks is not None and ks - 1 >= cL:
            rows[-1]['fail_margin_one_step_before'] = float(W.dnA(max(
                RSEL[ks - 1 - cL] - ee_hi, RSXL[ks - 1 - cL] - ev_hi)))
    # ---- deadline, C3, C5 -----------------------------------------------------
    # the deadlines are grid points k/1000 s (exact)
    T_xi_b = W.A(T_xi_q)
    nm1 = W.A(n - 1)
    if all(k is not None for k in k_star):
        Tb_hi = W.A(Fr(max(k_star), 1000))
        s3_lo_b = T_xi_b - nm1 * d - Tb_hi
        s3_lo = float(W.flo(s3_lo_b))
        Tbar_hi = W.fhi(Tb_hi)
    else:
        s3_lo, Tbar_hi = -INF, INF
    if all(k is not None for k in k_low):
        Tb_lo = W.A(Fr(max(k_low), 1000))
        s3_hi = float(W.fhi(T_xi_b - nm1 * d - Tb_lo))
        Tbar_lo = W.flo(Tb_lo)
    else:
        s3_hi, Tbar_lo = -INF, INF
    assert s3_lo <= s3_hi, (s3_lo, s3_hi)
    C3 = _cond(IV(s3_lo, s3_hi), '>=',
               'interval entry layer (exact 1 ms grid)',
               T_ent_bar=[Tbar_lo, Tbar_hi], entry_idx_ms=dict(upper=k_star, lower=k_low))
    # FB (2026-09-30): (C6) reads the tail at (T_xi - dbar)^+ (zero-time extension)
    kh = max((T_xi_q - dq) / H, 0)
    kh_hi, kh_lo = math.floor(kh), math.ceil(kh)
    VhndH = [float(VH[i][kh_hi]) for i in range(1, n)]
    VhndL = [float(VL[i][min(kh_lo, N - 1)]) for i in range(1, n)]
    pad = W.A(F(cfg['vhnd']))
    s5_lo = float(W.dnA(W.flo(pad) - max(VhndH)))
    s5_hi = float(W.upA(W.fhi(pad) - max(VhndL)))
    C5 = _cond(IV(s5_lo, s5_hi), '>=', 'interval entry layer (exact 1 ms grid)',
               Vhnd_hi=VhndH, Vhnd_lo=VhndL, grid_index=[kh_lo, kh_hi])
    out = dict(C3=C3, C5=C5, rows=rows, T_ent_bar=[Tbar_lo, Tbar_hi],
               T_hold=[float(W.flo(W.A(Fr(Tbar_lo)) + nm1 * d)) if Tbar_lo < INF else INF,
                       float(W.fhi(W.A(Fr(Tbar_hi)) + nm1 * d)) if Tbar_hi < INF else INF],
               Vhnd_max=[max(VhndL), max(VhndH)],
               lags=list(lags), grid='exact k/1000 s, k < %d' % N,
               V_hi_max_increase=[float(np.max(np.diff(v))) for v in VH[1:]])
    if keep_arrays:
        out['arrays'] = dict(VH=VH, VL=VL, AtL=AtL, AtH=AtH)
    return out


_GRID = {}


def _range_max(arr, ga, gb):
    """max of arr over the index ranges [ga_k, gb_k] (gb >= ga assumed
    small ranges)."""
    out = arr[ga].copy()
    w = int(np.max(gb - ga)) if len(ga) else 0
    for dd in range(1, w + 1):
        idx = np.minimum(ga + dd, gb)
        out = np.maximum(out, arr[idx])
    return out


# ============================================================ self-check
def _frac_ext(pw, a, b, mode):
    """brute-force exact extremum of an IPW over [a, b] (Fractions): f(a),
    both one-sided limits at the breakpoints in (a, b], f at b, vertices."""
    X = pw.X
    c0, c1, c2 = pw.c
    m = len(X)
    vals = []
    for j in range(m):
        xe = X[j + 1] if j + 1 < m else None
        if X[j] > b or (xe is not None and xe <= a):
            continue
        lo = max(a, X[j])
        hi = b if xe is None else min(b, xe)

        def q(x):
            t = x - X[j]
            return c0[j] + c1[j] * t + c2[j] * t * t
        vals += [q(lo), q(hi)]
        if c2[j] != 0:
            tv = X[j] - c1[j] / (2 * c2[j])
            if lo < tv < hi:
                vals.append(q(tv))
    return min(vals) if mode == 'min' else max(vals)


def selfcheck(cfg, enc, n_windows=300, cells=None, seed=20260928):
    """independent checks of the enclosure (not against the float
    evaluator): (a) the acquisition chain with zero planned parts against
    run_witness_caseB_entry.acq_chain (Arb, 200 bits, exact constants);
    (b) window extrema of the exact piecewise quadratics against brute-force
    Fraction arithmetic at random windows with dyadic ends; (c) the planned
    pulse bounds of pair 5 (and the other pairs) at sample cells, including
    the cells where P^e, P^eps, P^f are attained, recomputed in Arb from the
    exact rationals (interior extrema decided in exact arithmetic)."""
    CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
    if CODE not in sys.path:
        sys.path.insert(0, CODE)
    import run_witness_caseB_entry as W                   # noqa: E402
    from flint import arb, fmpq, ctx
    rng = np.random.default_rng(seed)
    S = hr.setup(cfg)
    n = S.n
    dq = Fr(enc['dbar'])
    out = OrderedDict()
    # (a) chain against Arb
    K = constants(cfg, S, dq)
    z = [IV(0.0)] * n
    Rz = chain_iv(K, n, z, z, z, z, z)
    ctx.prec = 200
    P = dict(n=n, lam_=W.A(F(cfg['lam'])), beta_=W.A(F(cfg['lam']) ** 2),
             gamma_=W.A(2 * F(cfg['lam'])), eps_det=F(cfg['eps_det']), T_c=F(cfg['T_c']),
             alpha=[F(cfg['alpha'])] * n, eps0=K['eps0F'], e0=K['e0F'])
    ch = W.acq_chain(P, dq)
    worst = 0.0
    bad = []
    for key in ('Lam1', 'taub', 'tact', 'tset', 'E1', 'ebar', 'Lam2', 'tvS2', 'fsw', 'Tk'):
        for k in range(n):
            b_ = ch[key][k]
            lo_, hi_ = W.flo(b_), W.fhi(b_)
            iv = Rz[key][k]
            # the Arb ball and the interval must overlap, and the interval
            # must contain the ball's midpoint up to the ball radius
            if not (iv.lo <= hi_ and lo_ <= iv.hi):
                bad.append('%s[%d]' % (key, k))
            worst = max(worst, (iv.hi - iv.lo) / max(abs(iv.hi), 1e-300))
    out['chain_vs_arb'] = OrderedDict(ok=not bad, disjoint=bad,
                                      max_rel_width_of_interval=worst,
                                      fields=10, units=n)
    # (b) window extrema against Fractions
    V = plan_views(S, cfg)
    names = [('arA', V['arA']), ('vr', V['vr']), ('arkvr', V['arkvr'])]
    for u in range(1, n):
        names += [('rho%d' % u, V['rho'][u]), ('negdd%d' % u, V['negdd'][u]),
                  ('A%d' % u, V['A'][u]), ('B%d' % u, V['B'][u]), ('gs%d' % u, V['gs'][u])]
    nfail = 0
    ntest = 0
    for nm, pw in names:
        X = [float(x) for x in pw.X[1:]] or [0.0]
        for _ in range(max(1, n_windows // len(names))):
            # windows around breakpoints and at random, dyadic ends
            c_ = X[int(rng.integers(len(X)))] if rng.random() < 0.7 else float(rng.uniform(-5, 90))
            a = round(c_ + float(rng.uniform(-0.2, 0.05)), 6)
            b = a + round(float(rng.uniform(0.0, 0.3)), 6)
            a, b = Fr(a).limit_denominator(1 << 20), Fr(b).limit_denominator(1 << 20)
            a = Fr(float(a)); b = Fr(float(b))
            if b < a:
                a, b = b, a
            for mode in ('min', 'max'):
                ex = _frac_ext(pw, a, b, mode)
                e = pw.ext(IA(np.array([float(a)])), IA(np.array([float(b)])), mode)
                ntest += 1
                if not (Fr(e.lo[0]) <= ex <= Fr(e.hi[0])):
                    nfail += 1
    out['window_extrema_vs_fractions'] = OrderedDict(tests=ntest, failures=nfail)
    # (c) pulse bounds against Arb
    pb = enc.get('_pb')
    cellsD = enc.get('_cells')
    if pb is not None:
        tg, tg1 = cellsD['tg'], cellsD['tg1']
        lam_q = F(cfg['lam'])
        lamA = arb(fmpq(lam_q.numerator, lam_q.denominator))
        dA = arb(fmpq(dq.numerator, dq.denominator))

        def A_(x):
            x = Fr(x)
            return arb(fmpq(x.numerator, x.denominator))

        def h_all(r):
            e = (-lamA * A_(r)).exp()
            return (A_(r) * e, (1 - lamA * A_(r)) * e, lamA * (2 - lamA * A_(r)) * e)
        r1, r2, r3 = 1 / lam_q, 2 / lam_q, 3 / lam_q
        pk = (1 / (lamA * arb(1).exp()), -(arb(-2).exp()), -lamA * arb(-3).exp())
        rows = []
        nb = 0
        nt = 0
        for c in range(1, n):
            nrm = pulse_norms(pb[c])
            sample = set(int(g) for g in rng.integers(0, len(tg), 6))
            sample |= {nrm['P_e'][1], nrm['P_eps'][1], nrm['P_f'][1]}
            if cells:
                sample |= set(cells)
            hs, los = pair_jumps_exact(S, c)
            for g in sorted(sample):
                ta, tb = Fr(float(tg[g])), Fr(float(tg1[g]))
                acc = dict(e_hi=arb(0), e_lo=arb(0), x_hi=arb(0), x_lo=arb(0),
                           f_hi=arb(0), f_lo=arb(0))
                for h, lo in zip(hs, los):
                    hi = lo + c * dq
                    rh = tb - lo
                    if rh < 0:
                        continue
                    rl = max(ta - hi, Fr(0))
                    Hl, Hh = h_all(rl), h_all(rh)
                    me = Hl[0].max(Hh[0])
                    if rl < r1 < rh:
                        me = pk[0]
                    xmax = Hl[1].max(Hh[1])
                    xmin = pk[1] if rl < r2 < rh else Hl[1].min(Hh[1])
                    fmax = Hl[2].max(Hh[2])
                    fmin = pk[2] if rl < r3 < rh else Hl[2].min(Hh[2])
                    Me = dA * me
                    Mxp = dA * xmax.max(arb(0))
                    Mxm = dA * (-xmin).max(arb(0))
                    Mfp = dA * fmax.max(arb(0))
                    Mfm = dA * (-fmin).max(arb(0))
                    hA = A_(h)
                    if h > 0:
                        acc['e_hi'] += hA * Me
                        acc['x_hi'] += hA * Mxp
                        acc['x_lo'] -= hA * Mxm
                        acc['f_hi'] += hA * Mfp
                        acc['f_lo'] -= hA * Mfm
                    else:
                        acc['e_lo'] += hA * Me
                        acc['x_hi'] += (-hA) * Mxm
                        acc['x_lo'] -= (-hA) * Mxp
                        acc['f_hi'] += (-hA) * Mfm
                        acc['f_lo'] -= (-hA) * Mfp
                for key, val in acc.items():
                    nt += 1
                    I_ = pb[c][key]
                    ok = bool((arb(float(I_.lo[g])) <= val) and (val <= arb(float(I_.hi[g]))))
                    if not ok:
                        nb += 1
                        rows.append(dict(pair=c + 1, cell=g, bound=key,
                                         arb=str(val), interval=[float(I_.lo[g]),
                                                                  float(I_.hi[g])]))
        out['pulse_cells_vs_arb'] = OrderedDict(tests=nt, failures=nb, failed=rows[:10])
        ctx.prec = ARB_BITS
    out['ok'] = bool(out['chain_vs_arb']['ok'] and
                     out['window_extrema_vs_fractions']['failures'] == 0 and
                     out.get('pulse_cells_vs_arb', {}).get('failures', 0) == 0)
    return out


# ============================================================ float check
def _ulp(x):
    x = abs(float(x))
    return math.ulp(x) if x > 0 else math.ulp(0.0)


def _inside(fv, iv, tol_rel=1e-13):
    """(strictly inside, signed excursion, excursion in ulps of the float
    value, inside within tol_rel * max(1, |fv|))."""
    lo, hi = iv
    if fv is None or lo is None or hi is None:
        return None, None, None, None
    fv = float(fv)
    if not math.isfinite(fv):
        return None, None, None, None
    ins = lo <= fv <= hi
    exc = 0.0 if ins else (fv - hi if fv > hi else fv - lo)
    tol = tol_rel * max(1.0, abs(fv))
    return bool(ins), exc, abs(exc) / _ulp(fv), bool(abs(exc) <= tol)


# rounding model of the floating-point evaluator for the margin ratios:
# every tested quantity is produced by at most N_OPS rounded operations on
# terms of magnitude at most the stated scale (generous: the dependency depth
# of the longest recursion, the tail recursion, is about 10 per layer)
N_OPS = 1000


def float_crosscheck(cfg, enc, fres=None):
    """certify_headref.evaluate at the same age (the double nearest to the
    exact age) against the enclosure: every condition slack, every
    constant of the float record, the planned pulse bounds cell by cell, the
    entry lags and the command tails V_j of the entry layer; plus the
    margins of the time-resolved tests in units of the rounding estimate
    N_OPS * u * scale."""
    dq = Fr(enc['dbar'])
    cfgf = hr.config(**dict(cfg, dbar=float(dq))) if 'U' not in cfg else dict(cfg, dbar=float(dq))
    t0 = time.time()
    fr = fres or hr.evaluate(cfgf, keep_curves=True)
    out = OrderedDict(float_dbar=float(dq), float_runtime_s=time.time() - t0,
                      float_admitted=fr['admitted'], float_failed=fr['failed'])
    out['float_bounds'] = OrderedDict(
        (k, fr['bounds'].get(k)) for k in ('dispersion', 'T_f_bar', 'marker_max', 'gap_max',
                                           'clearance_floor', 'clearance_floor_min',
                                           'entry_deadline', 'T_hold', 'velocity_floor'))
    # ---- conditions
    rows = OrderedDict()
    for nm, c in enc['conds'].items():
        fc = fr['conds'][nm]
        ins, exc, ulps, ins_tol = _inside(fc['slack'], (c['lo'], c['hi']))
        agrees = None
        if c['status'] == 'verified':
            agrees = bool(fc['ok'])
        elif c['status'] == 'refuted':
            agrees = not bool(fc['ok'])
        rows[nm] = OrderedDict(float_slack=fc['slack'], float_ok=fc['ok'],
                               interval=[c['lo'], c['hi']], status=c['status'],
                               inside=ins, excursion=exc, excursion_ulps=ulps,
                               inside_within_1e13_rel=ins_tol,
                               status_agrees_with_float=agrees)
    out['conds'] = rows
    # ---- constants
    K = fr['consts']
    E = enc['consts']
    checked, outside = 0, []

    def chk(name, fv, iv):
        nonlocal checked
        if fv is None or iv is None:
            return
        checked += 1
        ins, exc, ulps, ins_tol = _inside(fv, iv)
        if not ins:
            outside.append(OrderedDict(name=name, float=float(fv), interval=list(iv),
                                       excursion=exc, excursion_ulps=ulps,
                                       within_1e13_rel=ins_tol))
    for key in ('Lam1', 'Lam2', 'taub', 'tact', 'tset', 'E1', 'ebar', 'sup_e', 'sup_x',
                'sup_f', 'tv_f', 'x2tot', 'e2bar', 'fbar', 'tvS2', 'Tk', 'fsw', 'F', 'W',
                'Wphi', 'epsb', 'eb', 'fb', 'A1', 'A2', 'P_e', 'P_eps', 'P_f', 'Fb_units',
                'anet', 'q', 'Qm', 'P_pre', 'fpre_plus', 'fpre_minus', 'Q_plus', 'Q_minus'):
        for j, (a, b) in enumerate(zip(K[key], E[key])):
            chk('%s[%d]' % (key, j), a, b)
    if 'Theta' in K:
        for j, (a, b) in enumerate(zip(K['Theta'], E['Th'])):
            chk('Theta[%d]' % j, a, b)
    chk('Leps', K['Leps'], E['Leps'])
    chk('f_xi', K['f_xi'], E['f_xi'])
    for c, p in K['pairs'].items():
        ip = E['pairs'][c]
        for k_, v_ in p.items():
            if k_ in ip:
                chk('pair%s.%s' % (c, k_), v_, ip[k_])
    B = fr['bounds']
    for key in ('dispersion', 'T_f_bar', 'marker_max', 'gap_max', 'clearance_floor_min',
                'velocity_floor', 'schedule_error'):
        chk(key, B[key], enc['bounds'][key])
    for key in ('marker', 'gap', 'clearance_floor', 'u_max_before_T_xi',
                'u_min_before_T_xi', 'switch_deadline', 'activation_deadline'):
        for j, (a, b) in enumerate(zip(B[key], enc['bounds'][key])):
            chk('%s[%d]' % (key, j), a, b)
    if 'entry_deadline' in enc['bounds']:
        chk('entry_deadline', B['entry_deadline'], enc['bounds']['entry_deadline'])
        chk('Vhnd_max', K['Vhnd_max'], enc['bounds']['Vhnd_max'])
    out['constants'] = OrderedDict(
        checked=checked, outside=len(outside),
        outside_beyond_1e13_rel=sum(1 for o in outside if not o['within_1e13_rel']),
        max_excursion_ulps=max([o['excursion_ulps'] for o in outside], default=0.0),
        items_outside=outside)
    # ---- planned pulse bounds, cell by cell
    pbI = enc.get('_pb')
    if pbI is not None:
        fpb = fr['curves']['pb']
        prow = []
        for c in range(1, len(fpb)):
            for key in ('e_hi', 'e_lo', 'x_hi', 'x_lo', 'f_hi', 'f_lo'):
                f = np.asarray(fpb[c][key])
                I_ = pbI[c][key]
                ins = (f >= I_.lo) & (f <= I_.hi)
                exc = np.maximum(np.maximum(f - I_.hi, I_.lo - f), 0.0)
                prow.append(OrderedDict(pair=c + 1, bound=key, cells=int(len(f)),
                                        inside=int(ins.sum()),
                                        max_excursion=float(exc.max()),
                                        max_width=float(np.max(I_.hi - I_.lo))))
        out['pulse_cells'] = OrderedDict(
            all_inside=all(r['inside'] == r['cells'] for r in prow), rows=prow)
    # ---- entry layer: deadlines (points k/1000 s of the fixed grid) and
    # command tails
    if 'entry' in enc:
        ent = enc['entry']
        er = OrderedDict()
        er['float_entry_deadlines_s'] = [float(x) for x in K['Tent_abs']]
        er['interval_entry_idx_ms'] = [[r.get('k_low'), r.get('k_star')] for r in ent['rows']]
        er['deadlines_equal'] = all(
            r.get('k_star') is not None and r.get('k_star') == r.get('k_low')
            and abs(r['k_star'] / 1000.0 - float(x)) < 5e-10
            for r, x in zip(ent['rows'], K['Tent_abs']))
        arrs = enc.get('_entry_arrays')
        if arrs is not None:
            er['V_tails'] = _entry_array_check(cfgf, fr, arrs, dq)
        out['entry'] = er
    # ---- margins in units of the rounding estimate
    out['rounding_margins'] = _rounding_margins(cfgf, enc)
    return out


def _entry_array_check(cfg, fr, arrs, dq):
    """float command tails V_j of certify_headref.entry_curves against the
    enclosure; the excursions below the enclosure are traced to the ties
    hi_J = T_k of the planned L1 tail (the float evaluator compares rounded
    values), by recomputing the float recursion with the exact A^tail on the
    grid; the float lag shifts ceil((B + dbar)/dtg - 1e-12) are compared with
    the exact shifts."""
    S = hr.setup(cfg)
    G = cq.namespace(cfg)
    n = S.n
    dbar = float(cfg['dbar'])
    TG = G['TGRID']
    dtg = G['dtg']
    R = fr['curves']['R']
    A2 = np.array(fr['consts']['A2'])
    lags = tuple(int(x) for x in cfg.get('lags', (15, 20)))
    VH, VL = arrs['VH'], arrs['VL']

    def compare(V):
        rows = []
        for j in range(1, len(V)):
            f = np.asarray(V[j])
            m = min(len(f), len(VL[j]))
            lo, hi = VL[j][:m], VH[j][:m]
            f = f[:m]
            scale = np.maximum(np.abs(hi), 1e-300)
            above = (f - hi) / scale
            below = (lo - f) / scale
            ka = int(np.argmax(above))
            kb = int(np.argmax(below))
            rows.append(OrderedDict(
                j=j, inside_fraction=float(np.mean((f >= lo) & (f <= hi))),
                above_points=int(np.sum(f > hi)), below_points=int(np.sum(f < lo)),
                above_points_beyond_1e12_rel=int(np.sum(above > 1e-12)),
                below_points_beyond_1e12_rel=int(np.sum(below > 1e-12)),
                max_rel_above=float(max(above[ka], 0.0)), at_above_s=float(TG[ka]),
                max_rel_below=float(max(below[kb], 0.0)), at_below_s=float(TG[kb]),
                median_rel_above=float(np.median(above[above > 0])) if np.any(above > 0) else 0.0))
        return rows
    out = OrderedDict(float_as_evaluated=compare(fr['curves']['V']))
    # float recursion with the exact planned L1 tails on the grid
    AtE = [None] + [0.5 * (arrs['AtL'][c] + arrs['AtH'][c]) for c in range(1, n)]
    ec = hr.entry_curves(G, R, R['Lam2'], A2, AtE, dbar, lags)
    out['float_with_exact_Atail'] = compare(ec['V'])
    out['float_entry_deadlines_with_exact_Atail_s'] = _float_entry_deadlines(cfg, fr, G, S, ec)
    # ... and with the exact grid shifts (every shift of the recursion is a
    # whole number of milliseconds)
    orig = G['_shift_ceil']

    def shift_exact(arr, x):
        k = int(round(float(x) * 1000.0))
        assert abs(float(x) * 1000.0 - k) < 1e-6
        if k <= 0:
            return arr
        o = np.empty_like(arr)
        o[:k] = arr[0]
        o[k:] = arr[:-k]
        return o
    try:
        G['_shift_ceil'] = shift_exact
        ec2 = hr.entry_curves(G, R, R['Lam2'], A2, AtE, dbar, lags)
    finally:
        G['_shift_ceil'] = orig
    out['float_with_exact_Atail_and_shifts'] = compare(ec2['V'])
    # float tail L1 against the exact one: differences beyond rounding
    # (1e-12 relative) and whether each lies at a tie hi_J = T_k
    mism = 0
    at_ties = 0
    below_f = 0
    for c in range(1, n):
        h, lo = S.pair_jumps(c)
        Af = hr.tail_l1(dbar, h, lo + c * dbar, TG)
        m = min(len(Af), len(arrs['AtL'][c]))
        lo_, hi_ = arrs['AtL'][c][:m], arrs['AtH'][c][:m]
        tol = 1e-12 * np.maximum(np.abs(hi_), 1e-300)
        bad = np.nonzero((Af[:m] < lo_ - tol) | (Af[:m] > hi_ + tol))[0]
        hs, los = pair_jumps_exact(S, c)
        ties = set()
        for lo_J in los:
            q = (lo_J + c * dq) * 1000
            if q.denominator == 1:
                ties.add(int(q))
        mism += len(bad)
        at_ties += sum(1 for k in bad if int(k) in ties)
        below_f += int(np.sum(Af[:m][bad] < lo_[bad]))
    out['Atail_float_points_beyond_rounding'] = mism
    out['Atail_float_points_beyond_rounding_at_ties'] = at_ties
    out['Atail_float_points_beyond_rounding_below_exact'] = below_f
    # shifts
    nB, nA = lags
    sh = []
    for B in [2 * m for m in range(max(nB, nA))]:
        kf = int(np.ceil(max(B + dbar, 0.0) / dtg - 1e-12))
        ke = math.ceil((Fr(B) + dq) * 1000)
        kfa = int(np.ceil(max(float(B), 0.0) / dtg - 1e-12)) if B > 0 else 0
        sh.append((B, kf - ke, kfa - 1000 * B))
    out['float_shift_excess_V'] = {str(B): e for (B, e, _) in sh if e != 0}
    out['float_shift_excess_Atail'] = {str(B): e for (B, _, e) in sh if e != 0}
    return out


def _float_entry_deadlines(cfg, fr, G, S, ec):
    """entry deadlines of certify_headref.evaluate (statement-for-statement
    copy of its entry search on the fixed grid) from the tail curves ec."""
    n = S.n
    TGRID, dtg, NT = G['TGRID'], G['dtg'], G['NT']
    R = fr['curves']['R']
    tset = R['tset']
    pb = fr['curves']['pb']
    Dg = float(cfg['cell_dt'])
    NGc = len(pb[0]['e_hi'])
    kK = min(int(math.floor(S.T_xi_f / dtg + 1e-9)), NT - 1)
    rc = int(round(Dg / dtg))
    out = []
    for i in range(n):
        k0 = int(math.ceil(tset[i] / dtg - 1e-9))
        if kK < k0:
            out.append(None)
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
        ok = (RSE <= G['eps_e']) & (RSX <= G['eps_v'])
        out.append(float(tk[int(np.argmax(ok))]) if ok.any() else None)
    return out


def _rounding_margins(cfg, enc):
    """slack / (N_OPS u scale) of the time-resolved tests (reported for the
    record; every one of them is also interval-enclosed).  scale: the
    largest magnitude of the terms of the test (design values)."""
    u = U_RND
    C = enc['conds']
    out = OrderedDict()
    eps_e, eps_v = float(cfg['eps_e']), float(cfg['eps_v'])
    T_xi = float(F(cfg['plan']['T_xi']))
    v_in = float(F(cfg['plan']['v_xi']))
    c0max = max(float(x) for x in cfg['c0'])

    def put(nm, slack, scale, note):
        est = N_OPS * u * scale
        out[nm] = OrderedDict(slack=slack, scale=scale, rounding_estimate=est,
                              ratio=slack / est if est > 0 else None,
                              at_least_1e3=bool(slack >= 1e3 * est), note=note)
    if 'C3' in C:
        for r in enc['entry']['rows']:
            if r.get('k_star') is None:
                continue
            put('C3_entry_pass_unit%d' % r['unit'],
                min(r['pass_margin_e'] / eps_e, r['pass_margin_eps'] / eps_v), 1.0,
                'relative pass margin of the entry test at the entry deadline '
                '(upper envelopes against eps_e, eps_v)')
            if r.get('fail_margin_one_step_before') is not None:
                put('C3_entry_fail_unit%d' % r['unit'],
                    r['fail_margin_one_step_before'] / min(eps_e, eps_v), 1.0,
                    'certain failure margin one grid step before the entry deadline '
                    '(lower envelopes, relative to the smaller threshold)')
        put('C3', C['C3']['lo'], T_xi, 'time slack of the hold (s)')
    if 'C5' in C:
        put('C5', C['C5']['lo'], float(cfg['vhnd']), 'handoff pad slack (m/s^2)')
    put('C4', C['C4']['lo'], v_in, 'velocity floor (m/s)')
    put('C7a', C['C7a']['lo'], float(cfg['Uminus']), 'upper authority slack (m/s^2)')
    put('C7b', C['C7b']['lo'], float(cfg['Uminus']), 'lower authority slack (m/s^2)')
    for nm in ('C8a', 'C8b', 'C9a', 'C9b'):
        put(nm, C[nm]['lo'], c0max, 'planned minima of h*, mu*, g* (m)')
    return out
