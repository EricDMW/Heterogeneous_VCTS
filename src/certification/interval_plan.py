#!/usr/bin/env python3
"""interval_plan.py -- outward-interval enclosure of the closed-form
conditions of certify_plan.evaluate (Case D, Proposition D1).

Method (as interval_profile.py, whose interval class IV, modal bounds
Modal and Arb exponentials are imported read-only): two-sided intervals of
IEEE doubles with one-ulp outward rounding of every elementary operation,
exponentials from python-flint Arb balls (120 bits) stepped one ulp
outward, sup/TV/L1 of the modal family (p + q t) e^{-lam t} by
branch-union upper bounds, every derived constant (Euler's number,
1/(lam e), the kernel TV, the handoff-box peaks and L1 norms) an interval.
The plan quantities (breakpoints, planned states, the minima of g*, h*,
mu* and eps*, the planned speeds at the cell ends, the plan premises) are
exact rationals (certify_plan.Plan, pair_minima) enclosed by the two
neighbouring doubles.

Scope (interval-verified): the acquisition chain, the braking recursion
without the ramp charge, the budgets, thresholds and admission margins
(H4D i-iii), the clearance and barrier floors (H5D), the sign-resolved
authority at the left endpoint of every plan piece and the braking
authority (H1D i-iii), the headroom (H2), the sampled event (H7), the
velocity floor on a rigorous cell partition of [0, T_xi] (H6D; without a
verified P_S1 the cells before a switch deadline also carry the COAST and
S1 terms of P_k, round 5; with P_S1 the enclosure is unchanged), the
premise P_S1, the settling, marker and gap bounds (C_sync, C_al, C_gap)
and T_f_bar; the plan premises P1 to P4 are exact.  Out of scope (grid-evaluated in floating
point, as in the archived Case B and Case C witnesses): H3D and C_hnd
(entry recursion).

API: enclose(cfg, cell=0.05) -> dict with, per condition, the interval
[lo, hi] of the slack and 'verified' = lo > 0 (lo >= 0 for the closed
inequalities); check_enclosure(cfg) adds the floating-point slacks of
certify_plan.evaluate and the check that each lies inside its interval.
"""
import math
import os
import sys
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_plan as cq                                 # noqa: E402
from interval_profile import (IV, Modal, iexp, iabs, imax, imin, ipos,  # noqa: E402
                              isum, isqr, dn, up, INF)


def ivF(x):
    """tightest double interval enclosing the rational x."""
    x = Fr(x)
    f = float(x)
    lo = f if Fr(f) <= x else math.nextafter(f, -INF)
    hi = f if Fr(f) >= x else math.nextafter(f, INF)
    return IV(lo, hi)


def enclose(cfg=None, cell=0.05, verbose=False):
    cfg = cq.config(**(cfg or {})) if (cfg is None or 'n' not in cfg) else cfg
    n = int(cfg['n'])
    P = cq.plan(cfg)
    lam = ivF(cq.F(cfg['lam']))
    beta = lam * lam
    gamma = IV(2.0) * lam
    M = Modal(lam)
    E = iexp(IV(1.0))
    a_b = ivF(cq.F(cfg['a_b'])); kk = ivF(cq.F(cfg['k']))
    v_c1 = ivF(P.v_c1); T_xi = ivF(P.T_xi)
    alpha = [ivF(cq.F(cfg['alpha']))] * n
    A_run = [imax(*alpha[:i + 1]) for i in range(n)]
    v0F = [cq.F(x) for x in cfg['v0']]
    eps0F = [P.v_xi - v0F[0]] + [v0F[i - 1] - v0F[i] for i in range(1, n)]
    eps0 = [ivF(x) for x in eps0F]
    vt0 = [ivF(v0F[i] - P.v_xi) for i in range(n)]
    e0 = [IV(0.0)] * n
    eps_e = ivF(cq.F(cfg['eps_e'])); eps_v = ivF(cq.F(cfg['eps_v']))
    eps_det = ivF(cq.F(cfg['eps_det'])); phi = ivF(cq.F(cfg['phi']))
    T_c = ivF(cq.F(cfg['T_c'])); kappa = IV(1.0)
    s_m = ivF(cq.F(cfg['s_m'])); d = ivF(cq.F(cfg['dbar']))
    VH = ivF(cq.F(cfg['vhnd']))
    bF = [cq.F(a) / cq.F(cq.FIXED['v_c']) for a in cfg['amax']]
    b = [ivF(x) for x in bF]
    Pi = [IV(0.0)] + [ivF(1 / bF[i - 1] - 1 / bF[i]) for i in range(1, n)]
    eta = [IV(0.0)] + [ivF(abs(1 - bF[i] / bF[i - 1])) for i in range(1, n)]
    NUMPAD = ivF(cq.F(cfg['numpad']))
    d_s = [ivF(cq.F(x)) for x in cfg['d_s']]
    ramp = bool(cfg.get('ramp_charge', False))
    GAIN_E = IV(1.0) / (lam * E); GAIN_X = IV(1.0); GAIN_F = gamma
    L1_GE = M.ilam2; L1_GX = IV(2.0) / (lam * E)
    TV_KIMP = IV(2.0) * gamma + IV(2.0) * lam * iexp(-3.0)
    XFREE_PK = eps_v + (lam / E) * eps_e
    EFREE_PK = eps_e + eps_v / (lam * E)

    def neg(x):
        return IV(-x.hi, -x.lo)

    def corner_stats(ebar, xbar):
        best = dict(sup_e=IV(0), sup_x=IV(0), sup_f=IV(0), tv_f=IV(0),
                    int_e=IV(0), int_x=IV(0))
        for se in (+1.0, -1.0):
            for sx in (+1.0, -1.0):
                e_0 = neg(ebar) if se < 0 else ebar
                x_0 = neg(xbar) if sx < 0 else xbar
                c = x_0 + lam * e_0
                spe, _, l1e = M.cf_upper(e_0, c)
                spx, _, l1x = M.cf_upper(x_0, (-lam) * c)
                spf, tvf, _ = M.cf_upper(beta * e_0 + gamma * x_0,
                                         (-(lam * lam)) * c)
                best['sup_e'] = imax(best['sup_e'], spe)
                best['sup_x'] = imax(best['sup_x'], spx)
                best['sup_f'] = imax(best['sup_f'], spf)
                best['tv_f'] = imax(best['tv_f'], tvf)
                best['int_e'] = imax(best['int_e'], l1e)
                best['int_x'] = imax(best['int_x'], l1x)
        return best

    def D_quad(ebar, xbar, bi):
        c1, c2 = (-beta) / bi, IV(1.0) - gamma / bi
        out = IV(0)
        for se in (+1.0, -1.0):
            for sx in (+1.0, -1.0):
                e_0 = neg(ebar) if se < 0 else ebar
                x_0 = neg(xbar) if sx < 0 else xbar
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

    # ---- acquisition chain
    Lam1 = [IV(0)] * n; Lam2 = [IV(0)] * n; taub = [IV(0)] * n
    tact = [IV(0)] * n; tset = [IV(0)] * n; E1 = [IV(0)] * n
    ebar = [IV(0)] * n; fbar = [IV(0)] * n; tv2 = [IV(0)] * n
    x2 = [IV(0)] * n; Tk = [IV(0)] * n; Dq = [IV(0)] * n; DBq = [IV(0)] * n
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
        Lam2[k] = d * isum(tv2[:k])
        x2[k] = st['sup_x'] + GAIN_X * Lam2[k]
        fbar[k] = st['sup_f'] + GAIN_F * Lam2[k]
        tv2[k] = st['tv_f'] + TV_KIMP * Lam2[k]
        Tk[k] = IV(2.0) * alpha[k] + (beta * ebar[k] + gamma * eps_det) + tv2[k]
        Dq[k] = D_quad(ebar[k], eps_det, b[k])
        DBq[k] = DB_quad(b[k])
    Fv = [isum(fbar[:i]) for i in range(n)]
    # ---- braking recursion (Case D: no ramp charge)
    f_xi = beta * eps_e + gamma * eps_v
    W = [IV(0)] * n; epsb = [IV(0)] * n; eb = [IV(0)] * n; fb = [IV(0)] * n
    Th = [IV(0)] * n; IE = [IV(0)] * n; IX = [IV(0)] * n; IF = [IV(0)] * n
    Fcum = [IV(0)] * n
    for k in range(1, n):
        base = VH + a_b + f_xi if ramp else VH + f_xi
        W[k] = d * (base + isum(Th[1:k]) + isum(Fcum[1:k]))
        epsb[k] = XFREE_PK + GAIN_X * W[k]
        eb[k] = EFREE_PK + GAIN_E * W[k]
        fb[k] = beta * eb[k] + gamma * epsb[k]
        Fcum[k] = f_xi + isum(fb[1:k + 1])
        IE[k] = IE_BOX + L1_GE * W[k]
        IX[k] = IX_BOX + L1_GX * W[k]
        IF[k] = beta * IE[k] + gamma * IX[k]
        Th[k] = beta * IX[k] + gamma * (W[k] + IF[k])
    Fb = [f_xi + isum(fb[1:i]) for i in range(n)]
    anet = [a_b - Fb[i] for i in range(1, n)]
    C = {}

    def put(name, lst, strict=True):
        lo = min(x.lo for x in lst); hi = min(x.hi for x in lst)
        C[name] = dict(lo=lo, hi=hi, verified=bool(lo > 0 if strict else lo >= 0))

    put('H2', anet)
    if not all(a.lo > 0 for a in anet):
        return dict(verified=False, conds=C, note='headroom not verified')
    # ---- P_S1 and H7
    put('P_S1', [neg(eps0[k] + Lam1[k]) for k in range(n)], strict=False)
    Leps = imax(*alpha) + isum(Tk)
    put('H7', [(eps_det - phi) - Leps * T_c], strict=False)
    # ---- authority H1D (i), (ii) at the left endpoint of every piece
    corners = []
    for k in range(n):
        eo = IV(-E1[k].hi)                  # outer corner of the one-sided box
        xo = neg(eps_det)
        corners.append([(e, x) for e in (eo, IV(0.0)) for x in (xo, IV(0.0))])
    one_sided = C['P_S1']['verified']
    if not one_sided:
        corners = []
        for k in range(n):
            corners.append([(e, x) for e in (IV(-E1[k].hi), IV(E1[k].hi))
                            for x in (neg(eps_det), eps_det)])
    sig_p = [1.0 if (eps0[k] + Lam1[k]).hi > eps_det.lo else 0.0 for k in range(n)]
    sig_m = [1.0 if (eps0[k] - Lam1[k]).lo < -eps_det.lo else 0.0 for k in range(n)]
    Lam2h = [IV(0.0)] + Lam2[1:]

    def Fpm_hi(i, t):
        tp = IV(0.0); tm = IV(0.0); s1p = 0.0; s1m = 0.0
        for k in range(i + 1):
            Rk = t - IV(float(i - k)) * d - tset[k]
            Rk = IV(max(Rk.lo, 0.0), max(Rk.hi, 0.0))
            bp = 0.0; bm = 0.0
            for (e, x) in corners[k]:
                p = beta * e + gamma * x
                q = (-(lam * lam)) * (x + lam * e)
                bp = max(bp, M.tail_sup_signed_hi(p, q, Rk))
                bm = max(bm, M.tail_sup_signed_hi(neg(p), neg(q), Rk))
            forced = GAIN_F * Lam2h[k]
            tp = tp + IV(0.0, bp) + forced
            tm = tm + IV(0.0, bm) + forced
            chi = 1.0 if t.lo < (tset[k] + IV(float(i - k)) * d).hi else 0.0
            s1p = max(s1p, alpha[k].hi * sig_p[k] * chi)
            s1m = max(s1m, alpha[k].hi * sig_m[k] * chi)
        return up(tp.hi + s1p), up(tm.hi + s1m)

    sl_up = []; sl_lo = []; rows = []
    for i in range(n):
        for (t0, t1, a) in P.acc_pieces(i):
            aP = ivF(-a)
            fp, fm = Fpm_hi(i, ivF(t0))
            su = aP - IV(0.0, fp); sl = kk - aP - IV(0.0, fm)
            sl_up.append(su); sl_lo.append(sl)
            rows.append(dict(unit=i + 1, t0=float(t0), F_plus_hi=fp,
                             F_minus_hi=fm, slack_upper_lo=su.lo,
                             slack_lower_lo=sl.lo))
    put('H1D_i', sl_up, strict=False)
    put('H1D_ii', sl_lo, strict=False)
    put('H1D_iii', [imin(kk - a_b - Fcum[c], a_b - Fcum[c]) for c in range(1, n)],
        strict=False)
    # ---- H6D on the cell partition of [0, T_xi]
    cellF = Fr(cell).limit_denominator(10 ** 6)
    ncell = int(math.ceil(P.T_xi / cellF))
    vfl = INF; vfl_at = None
    for j in range(ncell):
        tjF = j * cellF                      # exact cell ends: the cells
        tj1F = min((j + 1) * cellF, P.T_xi)  # cover [0, T_xi] without gaps
        tj = float(tjF)
        env = 0.0
        Pk = []
        for k in range(n):
            Rl = (ivF(tjF) - tset[k]).lo     # lower end: S is nonincreasing
            R = IV(max(Rl, 0.0))
            best = 0.0
            for (e, x) in corners[k]:
                best = max(best, M.tail_sup_signed_hi(x, (-lam) * (x + lam * e), R))
            pk = up(best + Lam2[k].hi)
            if not one_sided and ivF(tjF).lo < tset[k].hi:
                # before the switch without P_S1 (theory Sec. 5, P_k(t) for
                # t < t^set_k): COAST ledger and S1 envelope M^(1)_k
                M1k = iabs(eps0[k]) + Lam1[k]
                pk = max(pk, ipos(eps0[k] + Lam1[k]).hi,
                         M1k.hi if sig_p[k] else 0.0)
            Pk.append(pk)
        for i in range(n):
            env = 0.0
            for k in range(i + 1):
                env = up(env + Pk[k])
            vlo = ivF(P.state(i, tj1F)[1]).lo
            f = dn(vlo - env)
            if f < vfl:
                vfl = f; vfl_at = (i + 1, tj)
    C['H6D'] = dict(lo=vfl, hi=INF, verified=bool(vfl > 0), at=vfl_at, cell=cell)
    # ---- budgets, admission, floors
    adm = {'H4D_i': [], 'H4D_ii': [], 'H4D_iii': []}
    clr = []; bfl = []; pairs = {}
    kapF = Fr(1); s_mF = cq.F(cfg['s_m'])
    for c in range(1, n):
        V1 = isum([iabs(eps0[k]) + Lam1[k] + x2[k] for k in range(c)])
        H1 = E1[c] + (alpha[c] / b[c]) * taub[c] + V1 * iabs(Pi[c]) + Lam1[c] / b[c]
        V2 = IV(2.0) * isum(x2[:c])
        Ht = Dq[c] + V2 * iabs(Pi[c]) + Lam2[c] * (IV(1.0) / b[c] + DBq[c])
        M1 = iabs(eps0[c]) + Lam1[c]
        hm1 = imax((b[c] * M1 + (Fv[c] + A_run[c - 1]) * eta[c]) / (kappa * b[c]),
                   (alpha[c] + b[c] * M1 + Fv[c] * eta[c]) / (kappa * b[c]))
        hm2 = (fbar[c] + b[c] * x2[c] + Fv[c] * eta[c]) / (kappa * b[c])
        bmin = imin(b[c], b[c - 1])
        an = anet[c - 1]
        Vbrk = v_c1 + IV(float(c)) * eps_v
        Ph2 = IX[c] + epsb[c] / bmin + isqr(epsb[c]) / (IV(2.0) * an) \
            + IF[c] / b[c] + Vbrk * ipos(Pi[c]) + W[c] * (IV(1.0) / b[c] + DBq[c])
        Ubrk = a_b + Fb[c]
        hmb = imax((fb[c] + b[c] * epsb[c] + Ubrk * eta[c]) / (kappa * b[c]),
                   ipos(b[c] * epsb[c] + (b[c] / b[c - 1]) * (a_b + Fb[c]) - a_b)
                   / (kappa * b[c]))
        ht0 = e0[c] + eps0[c] / b[c] + vt0[c - 1] * Pi[c]
        t1F = min(Fr(tset[c].hi), P.T_xi)
        w1 = cq.pair_minima(P, c, Fr(0), t1F, bF, kapF, s_mF)
        w2 = cq.pair_minima(P, c, Fr(0), P.T_xi, bF, kapF, s_mF)
        mu1 = ivF(w1['mu'][0]); mu2 = ivF(w2['mu'][0])
        spT, vpT, _ = P.state(c - 1, P.T_xi); sfT, vfT, _ = P.state(c, P.T_xi)
        hT = ivF(spT - sfT - P.ell - s_mF + vpT / bF[c - 1] - vfT / bF[c])
        s_i = ht0 - H1 + mu1 / kappa - hm1 - NUMPAD
        s_ii = ht0 - H1 - Ht + mu2 / kappa - hm2 - NUMPAD
        s_iii = hT + ht0 - H1 - Ht - Ph2 - hmb - NUMPAD
        adm['H4D_i'].append(s_i); adm['H4D_ii'].append(s_ii)
        adm['H4D_iii'].append(s_iii)
        e2bar = st_k[c]['sup_e'] + GAIN_E * Lam2[c]
        g1 = ivF(w1['g'][0]) + e0[c] - E1[c]
        g2 = ivF(w2['g'][0]) - e2bar
        g3 = (d_s[c] - s_m) - eb[c] - isqr(epsb[c]) / (IV(2.0) * a_b)
        clr.append(imin(g1, g2, g3))
        bfl.append(imin(ivF(w1['h'][0]) + ht0 - H1,
                        ivF(w2['h'][0]) + ht0 - H1 - Ht))
        pairs[c + 1] = dict(slack_i=[s_i.lo, s_i.hi], slack_ii=[s_ii.lo, s_ii.hi],
                            slack_iii=[s_iii.lo, s_iii.hi],
                            gmin=[clr[-1].lo, clr[-1].hi])
    for nm in ('H4D_i', 'H4D_ii', 'H4D_iii'):
        put(nm, adm[nm])
    put('H5D_i', clr)
    put('H5D_ii', bfl, strict=False)
    # ---- terminal
    hops = [epsb[i] / (a_b - Fb[i]) for i in range(1, n)]
    disp = isum(hops)
    gaperr = [EFREE_PK + GAIN_E * W[i] + isqr(epsb[i]) / (IV(2.0) * (a_b - Fb[i]))
              for i in range(1, n)]
    align = eps_e + v_c1 * eps_v / a_b + isqr(eps_v) / (IV(2.0) * a_b) + isum(gaperr)
    put('C_sync', [ivF(cq.F(cfg['sync_tol'])) - disp], strict=False)
    put('C_al', [ivF(cq.F(cfg['align_tol'])) - align], strict=False)
    put('C_gap', [ivF(cq.F(cfg['gap_tol'])) - g for g in gaperr], strict=False)
    epsb_max = imax(*epsb[1:])
    T_f = T_xi + (v_c1 + eps_v) / a_b + IV(float(n - 1)) * epsb_max / a_b
    # ---- plan premises (exact)
    prem = P.premises(cq.F(cfg['k']), v0F[0])
    tcm = prem['t_c_max']
    # one row per premise (P1)-(P4), exact rationals enclosed by doubles
    for nm, ps in (('P1', min(P.v_xi, v0F[0] - P.v_xi)),
                   ('P2', min(prem['a_min'], cq.F(cfg['k']) - prem['a_max'])),
                   ('P3', P.T_xi - tcm), ('P4', P.v_c1)):
        C[nm] = dict(lo=ivF(ps).lo, hi=ivF(ps).hi, verified=bool(prem[nm]),
                     exact=str(ps))
    out = dict(conds=C, cell=cell, one_sided=one_sided,
               values=dict(settling_bound=[disp.lo, disp.hi],
                           align=[align.lo, align.hi],
                           gaperr=[[g.lo, g.hi] for g in gaperr],
                           T_f_bar=[T_f.lo, T_f.hi],
                           T_sw_bar=[imax(*tset).lo, imax(*tset).hi],
                           Fb_units=[[x.lo, x.hi] for x in Fcum],
                           epsb=[[x.lo, x.hi] for x in epsb[1:]],
                           pairs=pairs, authority=rows),
               scope='closed-form conditions verified with outward intervals; '
                     'H6D on a %g s cell partition of [0, T_xi]; H3D and C_hnd '
                     'are grid-evaluated in floating point (certify_plan)' % cell)
    out['verified'] = bool(all(c['verified'] for c in C.values()))
    if verbose:
        for k, c in C.items():
            print('  %-8s slack in [%+.6f, %s]  %s'
                  % (k, c['lo'], ('%+.6f' % c['hi']) if c['hi'] < INF else 'inf',
                     'VERIFIED' if c['verified'] else 'not verified'))
    return out


def check_enclosure(cfg=None, cell=0.05):
    """interval enclosure and the floating-point slacks of
    certify_plan.evaluate, with the check that every fp slack lies in its
    interval (H6D: the fp value on the 1 ms grid must not lie below the
    interval lower bound of the coarser cell partition)."""
    cfg = cq.config(**(cfg or {})) if (cfg is None or 'n' not in cfg) else cfg
    enc = enclose(cfg, cell=cell)
    fp = cq.evaluate(cfg)
    inside = {}
    for k, c in enc['conds'].items():
        s = fp['conds'][k]['slack']
        if s is None or not np.isfinite(s):
            inside[k] = None
        elif k == 'H6D':
            inside[k] = bool(c['lo'] <= s + 1e-12)
        else:
            inside[k] = bool(c['lo'] - 1e-15 <= s <= c['hi'] + 1e-15)
    return dict(enclosure=enc,
                fp_slack={k: fp['conds'][k]['slack'] for k in enc['conds']},
                fp_admitted=fp['admitted'],
                fp_H3D=fp['conds']['H3D'], fp_C_hnd=fp['conds']['C_hnd'],
                inside=inside,
                all_inside=all(v in (True, None) for v in inside.values()),
                verified=enc['verified'] and fp['conds']['H3D']['ok']
                and fp['conds']['C_hnd']['ok'])


if __name__ == '__main__':
    r = check_enclosure()
    for k, c in r['enclosure']['conds'].items():
        print('  %-8s [%+.9f, %s] fp %s inside %s'
              % (k, c['lo'], ('%+.9f' % c['hi']) if c['hi'] < INF else 'inf',
                 r['fp_slack'].get(k), r['inside'].get(k)))
    print('verified:', r['verified'], ' all inside:', r['all_inside'])
