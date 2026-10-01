#!/usr/bin/env python3
"""run_witness_caseB_entry.py -- interval witness for the persistent-entry
layer of the Case-B (primary) dispatch certificate (audit item R2-B04).

The closed-form conditions of the Case-B certificate are interval-verified
by witness_root2.py.  Until now the persistent-entry layer -- the certified
command tails V_j (Lemma S-tailB), the state envelopes B^e_i, B^eps_i
(Lemma S-tailC), the entry lags T^ent_i, the entry deadline
bar T_ent = max_i (bar t^set_i + T^ent_i), the hold condition
(H3) bar T_ent + dbar <= T_xi, and the handoff pad
C_hnd: V_{i-1}(T_xi - dbar) <= V^hnd -- was evaluated only in floating point
(certify_v30.certified_entry through b2_design.entry_certify).  This script
encloses that layer.

Arithmetic
  * Every scalar (the design constants, the acquisition recursion that
    produces bar t_j, bar t^set_j, bar e_j, Lambda^(2)_j, bar f^sw_j, and
    the closed-form constants Psi(rho_B), ||f'_imp||_1, sup_{a>=rho_A}|g_e|,
    sup_{a>=rho_A}|g_eps|, 1/(lam e), TV of the causal kernel, and every
    extremum location r_*) is an Arb ball (python-flint, 200-bit
    precision) computed from the EXACT decimal design constants, parsed
    as rationals.  A branch is taken only when Arb decides the comparison;
    an undecided branch takes the union of both branch values.
  * Grid functions live on the exact grid T_k = k/1000 s, k = 0..209999
    (the grid G of Sec. S-entry).  Each is a pair of IEEE-double arrays
    (lo, hi) with one outward nextafter step after every rounded
    operation.  T_k is enclosed by one outward step around the correctly
    rounded k/1000.0, and e^{-lam T_k} by the product of the Arb-enclosed
    anchors e^{-lam m} and e^{-lam r/1000} (k = 1000 m + r).
  * Every scalar enters the arrays through a verified double enclosure
    (flo/fhi: the double is checked against the ball in Arb).
  * Shifts by dbar and by the lags rho_A, rho_B are exact integer grid
    shifts (all are multiples of 1 ms; otherwise ceil/floor are used on
    the conservative side).  The lag of FT_j uses the evaluator's floor
    grid argument k - ceil(bar t^set_j / Delta_g); bar t^set_j is a ball,
    so the upper arrays use the ball's upper end and the lower arrays its
    lower end.  C_hnd is read at the exact grid point T_xi - dbar.
Outputs
  hi arrays bound the certificate's envelopes from above and lo arrays
  from below.  The verified lag R*_i is the first grid point at which
  both UPPER envelopes pass, so the entry conclusion of Prop. 1(c) holds
  from R*_i and T^ent_i <= R*_i.  The lower lag is the first grid point
  at which the LOWER envelopes do not certainly fail, so T^ent_i is no
  smaller.  When the two coincide, T^ent_i is identified exactly.
  (H3) is verified with bar T_ent <= max_i (sup bar t^set_i + R*_i); C_hnd
  with the upper arrays.  Boundary brackets verify H3 / C_hnd at a
  feasible age and their failure at an infeasible age (lower arrays).

A floating-point cross-check reruns the frozen evaluator's
certified_entry (read-only import of b2_design / certify_root2) and
reports how its entry lags and tail arrays sit inside the enclosure.

Usage:  python run_witness_caseB_entry.py [--no-brackets] [--no-float]
                                          [--no-caseA]
Writes: data/new_d19/witness_caseB_entry.json
        data/new_d19/witness_caseA_entry_recheck.json  (same method on
        the Case-A constants, cross-checked against witness_core_v7)
Creates no other files and modifies no existing file.
"""
import hashlib
import json
import math
import os
import re
import sys
import time
from fractions import Fraction as Fr

import numpy as np
from flint import arb, fmpq, ctx

ctx.prec = 200
INF = math.inf
CODE = os.path.dirname(os.path.abspath(__file__))
CERTDIR = os.path.join(CODE, 'src', 'certification')
NEW = os.path.join(CODE, 'data', 'new_d19')
EVAL = os.path.join(CERTDIR, 'certify_v30.py')
B2_JSON = os.path.join(NEW, 'b2_results.json')
V30_JSON = os.path.join(CODE, 'data', 'v30_values.json')
WCERT_A = os.path.join(CODE, 'data', 'witness_certificate.json')
OUT_B = os.path.join(NEW, 'witness_caseB_entry.json')
OUT_A = os.path.join(NEW, 'witness_caseA_entry_recheck.json')

H = Fr(1, 1000)                   # grid step Delta_g (exact)
N = 210000                        # grid G = {0, H, ..., 210 - H}
ASET = [2 * m for m in range(20)]  # rho_A in {0, 2, ..., 38} s
BSET = [2 * m for m in range(15)]  # rho_B in {0, 2, ..., 28} s


# ============================================================ scalars
def Q(x):
    """Exact rational of a design constant: decimal text, int, or the
    shortest-repr double of a decimal literal (e.g. 0.055 -> 11/200)."""
    if isinstance(x, Fr):
        return x
    if isinstance(x, bool):
        raise TypeError('bool is not a constant')
    if isinstance(x, int):
        return Fr(x)
    if isinstance(x, float):
        return Fr(repr(x))
    return Fr(str(x).strip())


def A(x):
    """Arb ball of an exact rational constant (or an arb, unchanged)."""
    if isinstance(x, arb):
        return x
    q = Q(x)
    return arb(fmpq(q.numerator, q.denominator))


def flo(x):
    """A double certainly <= every point of the ball x."""
    x = A(x)
    f = float(x.lower())
    while not (arb(f) <= x):
        f = math.nextafter(f, -INF)
    return f


def fhi(x):
    """A double certainly >= every point of the ball x."""
    x = A(x)
    f = float(x.upper())
    while not (arb(f) >= x):
        f = math.nextafter(f, INF)
    return f


def pair(x):
    return [flo(x), fhi(x)]


def branch(c_true, c_false, f_true, f_false):
    """Decided branch, or the union of both when Arb cannot decide."""
    if c_true:
        return f_true()
    if c_false:
        return f_false()
    return f_true().union(f_false())


def nz(q):
    return bool((q > 0) or (q < 0))


def amax_(xs):
    out = xs[0]
    for x in xs[1:]:
        out = out.max(x)
    return out


E1_ = arb(1).exp()                 # Euler's number e (ball)


# ============================================================ arrays
def dnA(a):
    return np.nextafter(a, -np.inf)


def upA(a):
    return np.nextafter(a, np.inf)


def shift(arr, s):
    """arr at index max(k - s, 0): the value at (T_k - s*H)^+ ."""
    if s <= 0:
        return arr
    if s >= len(arr):
        return np.full_like(arr, arr[0])
    out = np.empty_like(arr)
    out[:s] = arr[0]
    out[s:] = arr[:-s]
    return out


def _abs_hi(lo, hi):
    return np.maximum(np.abs(lo), np.abs(hi))


def _abs_lo(lo, hi):
    return np.where((lo <= 0.0) & (hi >= 0.0), 0.0,
                    np.minimum(np.abs(lo), np.abs(hi)))


def _rev_cummax(a):
    return np.maximum.accumulate(a[::-1])[::-1]


class Grid:
    """Exact grid T_k = k/1000 with enclosures of T_k and e^{-lam T_k}."""

    def __init__(self, lam):
        k = np.arange(N)
        t = k / 1000.0                     # correctly rounded quotient
        self.k = k
        self.Tlo = np.where(k == 0, 0.0, dnA(t))
        self.Thi = np.where(k == 0, 0.0, upA(t))
        lam_ = A(lam)
        M = N // 1000 + 1
        anc = [(-lam_ * m).exp() for m in range(M)]
        frc = [(-lam_ * A(Fr(r, 1000))).exp() for r in range(1000)]
        alo = np.array([flo(a) for a in anc])
        ahi = np.array([fhi(a) for a in anc])
        rlo = np.array([flo(a) for a in frc])
        rhi = np.array([fhi(a) for a in frc])
        m_, r_ = k // 1000, k % 1000
        self.Elo = dnA(alo[m_] * rlo[r_])
        self.Ehi = upA(ahi[m_] * rhi[r_])


def _g(p, q, G):
    """(lo, hi) arrays enclosing g(T_k) = (p + q T_k) e^{-lam T_k}."""
    plo, phi_ = flo(p), fhi(p)
    qlo, qhi = flo(q), fhi(q)
    c1, c2 = qlo * G.Tlo, qlo * G.Thi
    c3, c4 = qhi * G.Tlo, qhi * G.Thi
    qTlo = dnA(np.minimum(np.minimum(c1, c2), np.minimum(c3, c4)))
    qThi = upA(np.maximum(np.maximum(c1, c2), np.maximum(c3, c4)))
    llo = dnA(plo + qTlo)
    lhi = upA(phi_ + qThi)
    glo = dnA(np.where(llo >= 0.0, llo * G.Elo, llo * G.Ehi))
    ghi = upA(np.where(lhi >= 0.0, lhi * G.Ehi, lhi * G.Elo))
    return glo, ghi


def tail_sup(p, q, lam, G):
    """(lo, hi): sup_{r >= T_k} |(p + q r) e^{-lam r}|.  One interior
    extremum at r_* = (q - lam p)/(lam q); on [T, inf) the sup is |g(T)|
    if r_* <= T and max{|g(T)|, |g(r_*)|} otherwise."""
    glo, ghi = _g(p, q, G)
    ah, al = _abs_hi(glo, ghi), _abs_lo(glo, ghi)
    if nz(q):
        ts = (q - lam * p) / (lam * q)
        gs = abs((q / lam) * (-lam * ts).exp())
        tlo, thi = flo(ts), fhi(ts)
        after = G.Tlo >= thi               # T certainly >= r_*
        before = G.Thi < tlo               # T certainly <  r_*
        hi = np.where(after, ah, np.maximum(ah, fhi(gs)))
        lo = np.where(before, np.maximum(al, flo(gs)), al)
    else:                                   # q may vanish: global bound
        bs = abs(p) + abs(q) / (lam * E1_)
        hi = np.maximum(ah, fhi(bs))
        lo = al
    return lo, _rev_cummax(hi)


def tail_tv(p, q, lam, G):
    """(lo, hi): TV_[T_k, inf) of (p + q r) e^{-lam r}: |g(T)| if
    r_* <= T, |g(T) - g(r_*)| + |g(r_*)| otherwise."""
    glo, ghi = _g(p, q, G)
    ah, al = _abs_hi(glo, ghi), _abs_lo(glo, ghi)
    if nz(q):
        ts = (q - lam * p) / (lam * q)
        gs = (q / lam) * (-lam * ts).exp()
        gsl, gsh = flo(gs), fhi(gs)
        agl, agh = flo(abs(gs)), fhi(abs(gs))
        dlo = dnA(glo - gsh)
        dhi = upA(ghi - gsl)
        tvh = upA(_abs_hi(dlo, dhi) + agh)
        tvl = dnA(_abs_lo(dlo, dhi) + agl)
        tlo, thi = flo(ts), fhi(ts)
        after = G.Tlo >= thi
        before = G.Thi < tlo
        hi = np.where(after, ah, np.maximum(ah, tvh))
        lo = np.where(before, np.maximum(al, tvl), al)
    else:
        bs = abs(p) + abs(q) / (lam * E1_)
        hi = upA(ah + 2.0 * fhi(bs))
        lo = al
    return lo, _rev_cummax(hi)


# ============================================================ closed forms
def tv0(p, q, lam):
    """TV on [0, inf) of (p + q t) e^{-lam t} (the evaluator's _cf)."""
    if not nz(q):
        raise ValueError('tv0: q not certainly nonzero')
    ts = (q - lam * p) / (lam * q)
    gs = (q / lam) * (-lam * ts).exp()
    return branch(ts > 0, ts <= 0,
                  lambda: abs(p - gs) + abs(gs), lambda: abs(p))


def Psi(B, P):
    """int_B^inf |f_imp'(r)| dr in closed form (evaluator's _Psi)."""
    lam, beta, gamma = P['lam_'], P['beta_'], P['gamma_']
    PF = -gamma
    QF = -(beta - gamma * lam)
    p = QF - lam * PF
    q = -lam * QF
    B_ = A(B)

    def I(a):
        return (-lam * a).exp() * ((p + q * a) / lam + q / (lam * lam))
    if not nz(q):
        raise ValueError('Psi: q not certainly nonzero')
    t0 = -p / q
    return branch(B_ < t0, B_ >= t0,
                  lambda: abs(I(B_) - I(t0)) + abs(I(t0)),
                  lambda: abs(I(B_)))


def kE(a, lam):
    """sup_{r >= a} |r e^{-lam r}|  (extremum at 1/lam)."""
    a_ = A(a)
    return branch(a_ < 1 / lam, a_ >= 1 / lam,
                  lambda: 1 / (lam * E1_),
                  lambda: a_ * (-lam * a_).exp())


def kX(a, lam):
    """sup_{r >= a} |(1 - lam r) e^{-lam r}|  (extremum at 2/lam)."""
    a_ = A(a)
    ga = abs((1 - lam * a_) * (-lam * a_).exp())
    e2 = arb(-2).exp()
    return branch(a_ < 2 / lam, a_ >= 2 / lam,
                  lambda: ga.max(e2), lambda: ga)


def acq_chain(P, dbar):
    """Acquisition recursion of forward_pass (certify_v30) in Arb:
    returns the per-unit balls needed by the entry layer."""
    n = P['n']
    lam, beta, gamma = P['lam_'], P['beta_'], P['gamma_']
    eps_det, T_c = A(P['eps_det']), A(P['T_c'])
    d = A(dbar)
    TVK = 2 * gamma + 2 * lam * arb(-3).exp()
    zero = arb(0)
    Tk, tvS2, tset = [], [], []
    rec = {k: [] for k in ('Lam1', 'taub', 'tact', 'tset', 'E1', 'ebar',
                           'Lam2', 'tvS2', 'fsw', 'Tk')}
    for k in range(n):
        al = A(P['alpha'][k])
        ST = sum(Tk, zero)
        Lam1 = d * ST
        m = abs(A(P['eps0'][k])) + Lam1
        taub = m / al + T_c
        tact = zero if k == 0 else tset[k - 1] + d
        ts_ = tact + taub
        Lk = al + ST
        E1 = m * tact + (m * m - eps_det * eps_det).nonnegative_part() \
            / (2 * al) + (eps_det + Lk * T_c) * T_c
        eb = abs(A(P['e0'][k])) + E1
        tvf = None
        for se in (1, -1):
            for sx in (1, -1):
                e_0 = se * eb
                x_0 = sx * eps_det
                c = x_0 + lam * e_0
                v = tv0(beta * e_0 + gamma * x_0, -(lam * lam) * c, lam)
                tvf = v if tvf is None else tvf.max(v)
        Lam2 = d * sum(tvS2, zero)
        tv = tvf + TVK * Lam2
        fsw = beta * eb + gamma * eps_det
        T = 2 * al + fsw + tv
        Tk.append(T)
        tvS2.append(tv)
        tset.append(ts_)
        for key, val in (('Lam1', Lam1), ('taub', taub), ('tact', tact),
                         ('tset', ts_), ('E1', E1), ('ebar', eb),
                         ('Lam2', Lam2), ('tvS2', tv), ('fsw', fsw),
                         ('Tk', T)):
            rec[key].append(val)
    return rec


# ============================================================ entry layer
def _ceil_idx(x_float):
    return math.ceil(Fr(x_float) / H)


def _floor_idx(x_float):
    return math.floor(Fr(x_float) / H)


def entry_layer(P, dbar, G, ch, keep_arrays=False):
    n = P['n']
    lam, beta, gamma = P['lam_'], P['beta_'], P['gamma_']
    eps_det = A(P['eps_det'])
    d = A(dbar)
    dq = Q(dbar)
    K_hi = math.ceil(dq / H)       # shift for the upper arrays
    K_lo = math.floor(dq / H)      # shift for the lower arrays
    kk = G.k

    # ---- four-corner closed-form tails per unit (pair clock) --------
    SeL, SeH, SxL, SxH, FTL, FTH = [], [], [], [], [], []
    for k in range(n):
        eb = ch['ebar'][k]
        acc = None
        for se in (1, -1):
            for sx in (1, -1):
                e0c = se * eb
                x0c = sx * eps_det
                c = x0c + lam * e0c
                r = (tail_sup(e0c, c, lam, G),
                     tail_sup(x0c, -lam * c, lam, G),
                     tail_tv(-(beta * e0c + gamma * x0c),
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

    # ---- certified command tails V_j (Lemma S-tailB) ------------------
    gd = gamma * d
    L1FP = Psi(0, P)
    c2 = L1FP * d
    PsiB = {B: Psi(B, P) for B in BSET}
    VH = [np.zeros(N)]
    VL = [np.zeros(N)]
    for j in range(1, n + 1):
        jj = j - 1
        VsH = shift(VH[jj], K_hi)
        VsL = shift(VL[jj], K_lo)
        tset, tact = ch['tset'][jj], ch['tact'][jj]
        s_hi = _ceil_idx(fhi(tset))
        s_lo = _ceil_idx(flo(tset))
        ftH = FTH[jj][np.maximum(kk - s_hi, 0)]
        ftL = FTL[jj][np.maximum(kk - s_lo, 0)]
        aj = A(P['alpha'][jj])
        ajf = aj + ch['fsw'][jj]
        jmpH = upA(np.where(kk <= _floor_idx(fhi(tact)), fhi(aj), 0.0)
                   + np.where(kk <= _floor_idx(fhi(tset)), fhi(ajf), 0.0))
        jmpL = dnA(np.where(kk <= _floor_idx(flo(tact)), flo(aj), 0.0)
                   + np.where(kk <= _floor_idx(flo(tset)), flo(ajf), 0.0))
        if jj == 0:
            forcH = np.zeros(N)
            forcL = np.zeros(N)
        else:
            L2 = ch['Lam2'][jj]
            forcH = upA(fhi(gd) * VsH)
            forcL = dnA(flo(gd) * VsL)
            ctH = np.full(N, np.inf)
            ctL = np.full(N, np.inf)
            for B in BSET:
                c1 = PsiB[B] * L2
                ctH = np.minimum(ctH, upA(fhi(c1) + upA(
                    fhi(c2) * shift(VH[jj], K_hi + 1000 * B))))
                ctL = np.minimum(ctL, dnA(flo(c1) + dnA(
                    flo(c2) * shift(VL[jj], K_lo + 1000 * B))))
            forcH = upA(forcH + ctH)
            forcL = dnA(forcL + ctL)
        VH.append(upA(upA(upA(VsH + ftH) + jmpH) + forcH))
        VL.append(dnA(dnA(dnA(VsL + ftL) + jmpL) + forcL))

    # ---- entry envelopes (Lemma S-tailC) and entry lags ---------------
    KEinf = 1 / (lam * E1_)
    ee_lo, ee_hi = flo(P['eps_e']), fhi(P['eps_e'])
    ev_lo, ev_hi = flo(P['eps_v']), fhi(P['eps_v'])
    kEs = {a: kE(a, lam) for a in ASET}
    kXs = {a: kX(a, lam) for a in ASET}
    k_star, k_low, mono, slack_e, slack_v, fail_prev = [], [], [], [], [], []
    for i in range(n):
        BeH, BeL = SeH[i], SeL[i]
        BxH, BxL = SxH[i], SxL[i]
        if i >= 1:
            L2 = ch['Lam2'][i]
            fEH = np.full(N, np.inf); fEL = np.full(N, np.inf)
            fXH = np.full(N, np.inf); fXL = np.full(N, np.inf)
            dE = KEinf * d
            for a in ASET:
                VshH = shift(VH[i], K_hi + 1000 * a)
                VshL = shift(VL[i], K_lo + 1000 * a)
                cE = kEs[a] * L2
                cX = kXs[a] * L2
                fEH = np.minimum(fEH, upA(fhi(cE) + upA(fhi(dE) * VshH)))
                fEL = np.minimum(fEL, dnA(flo(cE) + dnA(flo(dE) * VshL)))
                fXH = np.minimum(fXH, upA(fhi(cX) + upA(fhi(d) * VshH)))
                fXL = np.minimum(fXL, dnA(flo(cX) + dnA(flo(d) * VshL)))
            BeH = upA(BeH + fEH); BeL = dnA(BeL + fEL)
            BxH = upA(BxH + fXH); BxL = dnA(BxL + fXL)
        passH = (BeH <= ee_lo) & (BxH <= ev_lo)          # certainly passes
        failL = (BeL > ee_hi) | (BxL > ev_hi)            # certainly fails
        ks = int(np.argmax(passH)) if passH.any() else None
        kl = int(np.argmax(~failL)) if (~failL).any() else None
        k_star.append(ks)
        k_low.append(kl)
        mono.append(bool(passH[ks:].all()) if ks is not None else None)
        slack_e.append(None if ks is None else float(dnA(ee_lo - BeH[ks])))
        slack_v.append(None if ks is None else float(dnA(ev_lo - BxH[ks])))
        fail_prev.append(None if (ks is None or ks == 0) else
                         bool(failL[ks - 1]))

    # ---- deadline, (H3), C_hnd ----------------------------------------
    T_xi = A(P['T_xi'])
    if all(k is not None for k in k_star):
        Tbar_hi_ball = amax_([ch['tset'][i] + A(Fr(k_star[i], 1000))
                              for i in range(n)])
        Tbar_hi = fhi(Tbar_hi_ball)
        h3_slack_ball = T_xi - d - Tbar_hi_ball
        h3_slack_lo = float(flo(h3_slack_ball))
        h3_holds = bool(h3_slack_ball >= 0)
    else:
        Tbar_hi, h3_slack_lo, h3_holds = INF, -INF, False
    if all(k is not None for k in k_low):
        Tbar_lo_ball = amax_([ch['tset'][i] + A(Fr(k_low[i], 1000))
                              for i in range(n)])
        Tbar_lo = flo(Tbar_lo_ball)
        h3_fails = bool(Tbar_lo_ball + d > T_xi)
        h3_slack_hi = float(fhi(T_xi - d - Tbar_lo_ball))
    else:                       # some T^ent_i = +inf certainly
        Tbar_lo, h3_fails, h3_slack_hi = INF, True, -INF
    kh = (Q(P['T_xi']) - dq) / H
    kh_hi, kh_lo = math.floor(kh), math.ceil(kh)
    assert 0 <= kh_hi and kh_lo < N, 'T_xi - dbar outside the grid'
    VhndH = [float(VH[i][kh_hi]) for i in range(1, n)]
    VhndL = [float(VL[i][kh_lo]) for i in range(1, n)]
    pad = A(P['VHND'])
    hnd_slack_lo = float(dnA(flo(pad) - max(VhndH)))
    hnd_holds = max(VhndH) <= flo(pad)
    hnd_fails = max(VhndL) > fhi(pad)

    out = dict(
        dbar=str(dq), K_shift=[K_lo, K_hi],
        Tent_verified_upper=[None if k is None else k / 1000.0
                             for k in k_star],
        Tent_lower=[None if k is None else k / 1000.0 for k in k_low],
        Tent_identified_exactly=[(a is not None and a == b)
                                 for a, b in zip(k_star, k_low)],
        Tent_grid_index=[k_low, k_star],
        envelope_slack_at_Tent=dict(e=slack_e, eps=slack_v),
        envelope_certainly_fails_one_step_before=fail_prev,
        entry_envelope_nonincreasing_after_Tent=mono,
        Tbar_ent=dict(lo=Tbar_lo, hi=Tbar_hi),
        H3=dict(test='bar T_ent + dbar <= T_xi', T_xi=float(Q(P['T_xi'])),
                lhs_hi=float(upA(Tbar_hi + float(dq))) if Tbar_hi < INF
                else INF,
                slack_lo=h3_slack_lo, slack_hi=h3_slack_hi,
                holds=h3_holds, certainly_fails=h3_fails),
        C_hnd=dict(test='max_i V_{i-1}(T_xi - dbar) <= V^hnd',
                   grid_index=[kh_lo, kh_hi],
                   Vhnd_lo=VhndL, Vhnd_hi=VhndH, max_hi=max(VhndH),
                   pad=float(Q(P['VHND'])), slack_lo=hnd_slack_lo,
                   holds=bool(hnd_holds), certainly_fails=bool(hnd_fails)),
        chain={key: [pair(x) for x in ch[key]]
               for key in ('taub', 'tact', 'tset', 'E1', 'ebar', 'Lam1',
                           'Lam2', 'fsw', 'tvS2', 'Tk')},
        constants=dict(L1FP=pair(L1FP), KEinf=pair(KEinf),
                       Psi={str(B): pair(PsiB[B]) for B in BSET},
                       kE={str(a): pair(kEs[a]) for a in ASET},
                       kX={str(a): pair(kXs[a]) for a in ASET}),
        V_hi_max_increase=[float(np.max(np.diff(v))) for v in VH[1:]],
        verified=bool(h3_holds and hnd_holds),
    )
    arrays = dict(VH=VH, VL=VL) if keep_arrays else None
    return out, arrays


def evaluate(P, dbar, G, keep_arrays=False):
    t0 = time.time()
    ch = acq_chain(P, dbar)
    out, arrays = entry_layer(P, dbar, G, ch, keep_arrays)
    out['runtime_s'] = round(time.time() - t0, 2)
    return out, arrays


# ============================================================ constants
def _read_json(path, tries=8):
    last = None
    for _ in range(tries):
        try:
            with open(path, encoding='utf-8') as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:  # writer active
            last = exc
            time.sleep(2.0)
    raise RuntimeError('cannot read %s: %s' % (path, last))


def _clean(o):
    """strict JSON: non-finite floats become the strings '+inf'/'-inf'
    (an infinite entry deadline means that no grid point passes)."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return '+inf' if o > 0 else ('-inf' if o < 0 else 'nan')
    return o


def _sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


def _evaluator_constants():
    """Decimal texts of the frozen evaluator's parameter block."""
    src = open(EVAL, encoding='utf-8').read()
    block = src[:src.index('# S2 modal machinery')]
    sc, arr = {}, {}
    for line in block.splitlines():
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*([-+0-9.eE]+)\s*(#.*)?$', line)
        if m:
            sc[m.group(1)] = m.group(2)
            continue
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*np\.array\(\[([^\]]*)\]\)',
                     line)
        if m:
            arr[m.group(1)] = [s.strip() for s in m.group(2).split(',')]
            continue
        m = re.match(r'^([A-Za-z_]\w*)\s*=\s*np\.full\(n,\s*([-+0-9.eE]+)\)',
                     line)
        if m:
            arr[m.group(1)] = [m.group(2)] * int(sc['n'])
    fd = re.search(r'\ndbar = ([0-9.]+)\n', src).group(1)
    return sc, arr, fd


def _finish(P):
    P['lam_'] = A(P['lam'])
    P['beta_'] = A(P['beta'])
    P['gamma_'] = A(P['gamma'])
    # the closed-form modal family assumes critical damping exactly
    assert P['beta'] == P['lam'] ** 2, 'beta != lam^2 in exact arithmetic'
    assert P['gamma'] == 2 * P['lam'], 'gamma != 2 lam in exact arithmetic'
    return P


def params_caseB():
    D = _read_json(B2_JSON)
    des, cert = D['_design'], D['cert']
    B2 = des['B2']
    sc, _, _ = _evaluator_constants()
    n = len(B2['eps0'])
    P = dict(case='B', n=n,
             lam=Q(B2['lam']), beta=Q(B2['beta']), gamma=Q(B2['gamma']),
             alpha=[Q(B2['alpha'])] * n,
             eps_e=Q(B2['eps_e']), eps_v=Q(B2['eps_v']),
             eps_det=Q(sc['eps_det']), phi=Q(sc['phi']), T_c=Q(sc['T_c']),
             T_xi=Q(B2['T_xi']), dbar=Q(B2['dbar']),
             VHND=Q(sc['VHND_PAD']),
             eps0=[Q(x) for x in B2['eps0']],
             e0=[Q(x) for x in des['e0']])
    ref = dict(overrides=dict(B2), e0=[float(x) for x in des['e0']],
               cert_entry=cert['entry'], cert_tset=cert['tset'],
               cert_taub=cert['taub'], cert_E1=cert['E1'],
               ceils=cert['ceils'])
    return _finish(P), ref


def params_caseA():
    sc, arr, fd = _evaluator_constants()
    n = int(sc['n'])
    P = dict(case='A', n=n,
             lam=Q(sc['lam']), beta=Q(sc['beta']), gamma=Q(sc['gamma']),
             alpha=[Q(x) for x in arr['alpha_g']],
             eps_e=Q(sc['eps_e']), eps_v=Q(sc['eps_v']),
             eps_det=Q(sc['eps_det']), phi=Q(sc['phi']), T_c=Q(sc['T_c']),
             T_xi=Q(sc['T_xi']), dbar=Q(fd), VHND=Q(sc['VHND_PAD']),
             eps0=[Q(x) for x in arr['eps0']],
             e0=[Q(x) for x in arr['e0']])
    V30 = _read_json(V30_JSON)
    ref = dict(overrides={}, e0=[float(x) for x in P['e0']],
               cert_entry=dict(Tent_cert=V30['Tent_cert'],
                               Tent_bar=V30['Tent_bar'],
                               T_hold=V30['T_hold'],
                               Vhnd_max=V30['Vhnd_max']),
               cert_tset=V30['tset'], cert_taub=V30['taub'],
               cert_E1=V30['E1'], ceils=V30['ceils'])
    if os.path.exists(WCERT_A):
        ref['witness_core_v7'] = _read_json(WCERT_A)
    return _finish(P), ref


def _pjson(P):
    keys = ('case', 'n', 'lam', 'beta', 'gamma', 'alpha', 'eps_e', 'eps_v',
            'eps_det', 'phi', 'T_c', 'T_xi', 'dbar', 'VHND', 'eps0', 'e0')
    out = {}
    for k in keys:
        v = P[k]
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
        elif isinstance(v, Fr):
            out[k] = str(v)
        else:
            out[k] = v
    return out


# ============================================================ float check
def float_crosscheck(P, ref, res, arrays):
    """Rerun the frozen evaluator's certified_entry in double precision
    (read-only import) and locate it inside the enclosure."""
    sys.path.insert(0, CERTDIR)
    import b2_design as bd          # noqa: E402  (read-only use)
    G = bd._namespace(ref['overrides'])
    G['e0'][:] = np.array(ref['e0'], float)
    dbar = float(P['dbar'])
    R = G['forward_pass'](dbar)
    Tent, Vhnd, V = G['certified_entry'](R, dbar)
    out = dict(Tent_float=[float(x) for x in Tent],
               Tbar_ent_float=float(np.max(R['tset'] + Tent)),
               Vhnd_float=[float(x) for x in Vhnd[1:]],
               Tent_float_equals_verified_upper=[
                   abs(float(a) - b) < 5e-7 for a, b in
                   zip(Tent, res['Tent_verified_upper'])])
    rel = {}
    for key in ('tset', 'tact', 'ebar', 'Lam2', 'fsw'):
        dev = 0.0
        for i, (lo, hi) in enumerate(res['chain'][key]):
            x = float(R[key][i])
            if lo <= x <= hi:
                continue
            scale = max(abs(lo), abs(hi), 1e-300)
            dev = max(dev, min(abs(x - lo), abs(x - hi)) / scale)
        rel[key] = dev
    out['chain_float_outside_enclosure_rel'] = rel
    inside, worst = [], []
    for j in range(1, P['n'] + 1):
        vf = np.asarray(V[j], float)
        lo, hi = arrays['VL'][j], arrays['VH'][j]
        ok = (vf >= lo) & (vf <= hi)
        inside.append(float(ok.mean()))
        scale = np.maximum(np.abs(hi), 1e-300)
        exc = np.maximum(np.maximum(vf - hi, lo - vf), 0.0) / scale
        worst.append(float(exc.max()))
    out['V_float_fraction_inside_enclosure'] = inside
    out['V_float_max_rel_excursion'] = worst
    out['note'] = ('The float evaluator uses Lambda2 + 1e-15 and double '
                   'constants, so excursions of order 1e-12 relative at '
                   'the far tail are rounding, not a discrepancy.')
    del G, R, V
    return out


# ============================================================ brackets
def bracket(P, G, cond, ceil_float, down=4, up=8):
    """Verified feasible point at or below the float boundary estimate
    and verified infeasible point above it (1 ms steps)."""
    key = {'H3': 'H3', 'C_hnd': 'C_hnd'}[cond]
    p0 = Fr(math.floor(Fr(repr(ceil_float)) * 1000), 1000)
    feas = None
    for s in range(down + 1):
        p = p0 - Fr(s, 1000)
        r, _ = evaluate(P, p, G)
        if r[key]['holds']:
            feas = dict(at=float(p), slack_lo=r[key]['slack_lo'],
                        Tbar_ent_hi=r['Tbar_ent']['hi'],
                        Vhnd_max_hi=r['C_hnd']['max_hi'])
            break
    infeas = None
    for s in range(1, up + 1):
        p = p0 + Fr(s, 1000)
        r, _ = evaluate(P, p, G)
        if r[key]['certainly_fails']:
            infeas = dict(at=float(p),
                          slack_hi=(r['H3']['slack_hi'] if key == 'H3'
                                    else float(upA(fhi(A(P['VHND'])) -
                                                   max(r['C_hnd']
                                                       ['Vhnd_lo'])))),
                          Tbar_ent_lo=r['Tbar_ent']['lo'],
                          Vhnd_max_lo=max(r['C_hnd']['Vhnd_lo']))
            break
    return dict(float_boundary_estimate=ceil_float,
                verified_feasible=feas, verified_infeasible=infeas)


# ============================================================ driver
def run_case(P, ref, do_float=True, do_brackets=True, out_path=None,
             extra_ages=()):
    t0 = time.time()
    G = Grid(P['lam'])
    res, arrays = evaluate(P, P['dbar'], G, keep_arrays=do_float)
    doc = dict(
        item='R2-B04 (round-2 audit): interval enclosure of the '
             'persistent-entry layer',
        case=P['case'],
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        script='code/run_witness_caseB_entry.py',
        script_sha256=_sha(os.path.abspath(__file__)),
        evaluator='code/src/certification/certify_v30.py',
        evaluator_sha256=_sha(EVAL),
        inputs=_pjson(P),
        input_sources=(['code/data/new_d19/b2_results.json:_design',
                        'code/src/certification/certify_v30.py '
                        '(eps_det, phi, T_c, VHND_PAD)']
                       if P['case'] == 'B' else
                       ['code/src/certification/certify_v30.py '
                        '(parameter block and featured dbar)']),
        basis=('Arb ball arithmetic (python-flint %s, %d-bit) from the '
               'exact decimal design constants for every scalar; '
               'grid functions as IEEE-double (lo, hi) arrays with one '
               'outward nextafter step per rounded operation on the exact '
               'grid T_k = k/1000 s, k < %d; exponentials on the grid '
               'from Arb-enclosed anchors; undecided branches take the '
               'union; exact integer grid shifts; FT_j lag at the '
               'evaluator floor grid argument; C_hnd at the exact grid '
               'point T_xi - dbar.'
               % (__import__('flint').__version__, ctx.prec, N)),
        lag_sets=dict(A=ASET, B=BSET, grid_step='1/1000', grid_points=N),
        operating_point=res)
    doc['float_reference'] = dict(
        Tent_cert=ref['cert_entry']['Tent_cert'],
        Tent_bar=ref['cert_entry']['Tent_bar'],
        T_hold=ref['cert_entry']['T_hold'],
        Vhnd_max=ref['cert_entry']['Vhnd_max'],
        source=('code/data/new_d19/b2_results.json:cert.entry'
                if P['case'] == 'B' else 'code/data/v30_values.json'))
    if 'witness_core_v7' in ref:
        w = ref['witness_core_v7']
        doc['witness_core_v7_reference'] = dict(
            Tent_hi=w['Tent_hi'], Tbar_ent_hi=w['Tbar_ent_hi'],
            H3=w['H3'], C_hnd=w['C_hnd'],
            source='code/data/witness_certificate.json '
                   '(witness_core_v7.entry_interval via emit_certificate_v7)')
    if do_float:
        try:
            doc['float_crosscheck'] = float_crosscheck(P, ref, res, arrays)
        except Exception as exc:          # report, do not mask
            doc['float_crosscheck'] = dict(error=repr(exc))
    arrays = None
    if do_brackets:
        doc['brackets'] = dict(
            H3=bracket(P, G, 'H3', ref['ceils']['hold']),
            C_hnd=bracket(P, G, 'C_hnd', ref['ceils']['hnd']))
    if extra_ages:
        ex = {}
        for dv in extra_ages:
            r, _ = evaluate(P, Q(dv), G)
            ex[str(float(Q(dv)))] = dict(
                Tent_verified_upper=r['Tent_verified_upper'],
                Tbar_ent_hi=r['Tbar_ent']['hi'],
                H3_slack_lo=r['H3']['slack_lo'], H3_holds=r['H3']['holds'],
                C_hnd_max_hi=r['C_hnd']['max_hi'],
                C_hnd_slack_lo=r['C_hnd']['slack_lo'],
                C_hnd_holds=r['C_hnd']['holds'],
                K_shift=r['K_shift'])
        doc['extra_ages'] = dict(
            note='entry layer at further ages (e.g. the verified-feasible '
                 'end of the H4 bracket), to pair with the closed-form '
                 'witness at the same age',
            results=ex)
    doc['summary'] = dict(
        dbar=float(P['dbar']),
        Tent_verified_upper=res['Tent_verified_upper'],
        Tent_identified_exactly=all(res['Tent_identified_exactly']),
        Tbar_ent_hi=res['Tbar_ent']['hi'],
        H3_lhs_hi=res['H3']['lhs_hi'], T_xi=res['H3']['T_xi'],
        H3_slack_lo=res['H3']['slack_lo'], H3_holds=res['H3']['holds'],
        C_hnd_max_hi=res['C_hnd']['max_hi'],
        C_hnd_slack_lo=res['C_hnd']['slack_lo'],
        C_hnd_holds=res['C_hnd']['holds'],
        entry_layer_interval_verified=res['verified'])
    doc['runtime_s'] = round(time.time() - t0, 1)
    if out_path:
        with open(out_path, 'w', encoding='utf-8') as fh:
            json.dump(_clean(doc), fh, indent=1, allow_nan=False)
        print('[witness] ->', os.path.relpath(out_path, CODE))
    return doc


def _report(doc):
    s = doc['summary']
    print('Case %s at dbar = %.3f s' % (doc['case'], s['dbar']))
    print('  T_ent verified upper :', s['Tent_verified_upper'],
          ' identified exactly:', s['Tent_identified_exactly'])
    print('  bar T_ent upper = %.6f ; H3: %.6f <= %.1f  slack >= %.6f s  [%s]'
          % (s['Tbar_ent_hi'], s['H3_lhs_hi'], s['T_xi'], s['H3_slack_lo'],
             'VERIFIED' if s['H3_holds'] else 'NOT verified'))
    print('  C_hnd: max V upper = %.6e <= 1e-3, slack >= %.6e  [%s]'
          % (s['C_hnd_max_hi'], s['C_hnd_slack_lo'],
             'VERIFIED' if s['C_hnd_holds'] else 'NOT verified'))
    if 'float_crosscheck' in doc:
        fc = doc['float_crosscheck']
        if 'error' in fc:
            print('  float cross-check error:', fc['error'])
        else:
            print('  float T_ent:', fc['Tent_float'], ' equal:',
                  fc['Tent_float_equals_verified_upper'])
            print('  float V inside enclosure (fraction):',
                  [round(x, 6) for x in fc['V_float_fraction_inside_enclosure']],
                  ' max rel excursion:',
                  ['%.1e' % x for x in fc['V_float_max_rel_excursion']])
    if 'brackets' in doc:
        for k, b in doc['brackets'].items():
            f, g = b['verified_feasible'], b['verified_infeasible']
            print('  bracket %-5s: float %.6f | feasible %s | infeasible %s'
                  % (k, b['float_boundary_estimate'],
                     None if f is None else f['at'],
                     None if g is None else g['at']))


if __name__ == '__main__':
    args = set(sys.argv[1:])
    do_float = '--no-float' not in args
    do_br = '--no-brackets' not in args
    PB, refB = params_caseB()
    # extra ages: the 1 ms boundary estimate of H4 and the verified-
    # feasible end of the H4 bracket (data/new_d19/h4_bracket_B.json)
    docB = run_case(PB, refB, do_float, do_br, OUT_B,
                    extra_ages=('0.024', '0.0246'))
    _report(docB)
    if '--no-caseA' not in args:
        PA, refA = params_caseA()
        docA = run_case(PA, refA, do_float, do_br, OUT_A,
                        extra_ages=('0.061',))
        _report(docA)
