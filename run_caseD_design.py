#!/usr/bin/env python3
"""run_caseD_design.py -- Case D design search, selected design, design
margins and trade-off.

Round 5 (2026-09-27): FAN design (parts fan_search, fan_design,
fan_tradeoff, fan_alt; section "fan family" at the end of this file).  Every
unit brakes at every instant: an approach of t_0 = 10 s at the slow rate
a_c, a common onset t_0 of the main deceleration with unit-specific constant
rates a_i = a_c + G/L_i (the head at a_head, the followers gentler along
the train), joins onto the common crawl line at t^c_i = t_0 + L_i, a crawl
at a_c, a common tail crawl of 5 s, and synchronized braking at the service
rate a_b from T_xi = t^c_n + 5 s.  The closure equations fix
L_i = L_1 + (i - 1) delta with G = 2C/delta.  Run order: fan_search,
fan_design, fan_tradeoff, fan_alt; the parts write design.json,
tradeoff.json and add the alternatives to design.json.

Round 4 (canonical family, parts search, design, lamscan, tradeoff,
condbnd, kept unchanged for the reproduction of the round-4 design; the
round-4 files are archived in data/new_d19/caseD_round4):
design search, selected design, design margins and trade-off (stage 2: main
deceleration at the service rate, jump instants on the 1 ms grid).

Authors' decisions: n = 5, capabilities, k = 1.2, a_b = 1.0, c_i(0) = 50 m,
parking gaps 5 m (6 m head slot), takeover speeds (10.47, 10.55, 10.67,
10.85, 11.10) m/s, communication contract of Case B (T_m = T_c = 1 ms,
19-step transport, age bound 20 ms), and the main deceleration of the plan
a_d = a_b = 1.0 m/s^2 (the service rate; the stage-1 value 1.17 was
rejected).  Frozen: kappa = 1, s_m = 1 m, eps_det = 0.015 m/s,
phi = 0.005 m/s, v_c = 1.2 m/s.

Design choices kept from Cases B and C (comparability): alpha = 0.35,
handoff box (eps_e, eps_v) = (0.15 m, 0.02 m/s), crawl end speed
v_c1 >= 1.0 m/s, planned initial speed v^xi = 10.45 m/s (theory
recommendation: every unit runs a braking S1 stage), canonical plan family
(theory Sec. 2.2; pre-braking rate = crawl rate a_c).

Grid-aligned plan (referee item D6).  Every planned jump instant lies on
the 1 ms grid: head onset t^d_1 = t1 ms, offsets t^d_i - t^d_{i-1} =
delta = p ms, pulse duration D = q ms, T_xi = t^c_n + c ms.  The closure
equations stay exact: Delta_v = C/delta (C = 45 m), the crawl rate solves
D = Delta_v/(a_d - a_c), a_c = a_d - Delta_v/D, and the crawl end speed
is v_c1 = v^xi - Delta_v - a_c T_xi (exact rationals).  The 1 kHz digital
implementation then processes every planned jump at a sampling instant
without event triggering.

Design rule (margins as in Case C, referee item D2): the certificate admits
the design at 20 ms and at 21 ms (1 ms reserve) with the preferred slacks
at 20 ms (admission H4D >= 0.15 m, authority H1D(i), (ii), (iii)
>= 0.02 m/s^2, that is on both sides of [-k, 0]); v_c1 >= 1.0 m/s; the
certified completion bound T_f_bar as small as possible.  No tail-crawl
margin (referee item D1): T_xi = t^c_n + 1 ms, the smallest grid instant
after the tail joins the crawl (premise P3 needs t^c_n < T_xi).  The head
onset t1 is the first 1 ms instant at or after the head's certified S1
detection deadline.

Search: lambda on the 0.01 grid (0.10 to 0.20), V^hnd in {1e-3, 1e-2};
per (lambda, V^hnd) the smallest admissible crawl rate a_c >= F^+_max(0)
+ 0.02 (F^+_i(0) does not depend on the plan) and the integer pair (p, q)
that minimizes T_xi + v_c1 (hence T_f_bar) subject to a_c >= that value
and v_c1 >= 1.0.  Selection: smallest T_f_bar rounded up to 0.01 s; then
the largest certified age boundary; then the smaller pad; then the larger
admission slack.

parts:  search   -> design_search.json (in $CASED_SCRATCH) and the selection
        design   -> data/new_d19/caseD/design.json (certificate, boundaries,
                    interval checks at 20 ms and at the boundary, gap rule,
                    lambda scan, alternatives and the cost of the margins)
        lamscan  -> adds the certified age boundary of every lambda of the
                    search (selected pad) to design.json (lam_boundary_scan)
        tradeoff -> data/new_d19/caseD/tradeoff.json (at most 12 rows)
        condbnd  -> recomputes only the per-condition age boundaries
                    (cond_boundary_ms) of design.json with the switch box
                    of the certificate (see cond_boundaries below) and
                    writes them back; every other field is kept
Run order: search, design, lamscan, tradeoff (search writes the scratch
file design_search.json that the other parts read).

Per-condition boundaries.  The admitted certificate uses the one-sided
switch box, which P_S1 justifies (Lemma D3).  Beyond the age at which P_S1
fails, the box of the certificate is the symmetric one (theory Sec. 2.5,
case_d.tex before eq. (D.13)).  The boundaries of the box-dependent
conditions (H1D)(i), (H1D)(ii) and (H6D) are therefore located with the
box rule one_sided=None (one-sided exactly while P_S1 holds) on a fresh
memo; the other conditions do not use the box and keep the evaluation of
the design configuration.
"""
import json
import math
import os
import sys
import time
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src', 'certification'))
import certify_plan as cq          # noqa: E402
import certify_profile as cp       # noqa: E402
import interval_plan as ip         # noqa: E402

DATA_D = os.path.join(HERE, 'data', 'new_d19', 'caseD')
SCRATCH = os.environ.get('CASED_SCRATCH', DATA_D)
PREF_ADM = 0.15
PREF_AUTH = 0.02
A_D = Fr(1)                        # authors: main deceleration = a_b
V_XI = Fr('10.45')
V_C1_MIN = Fr(1)
MS = Fr(1, 1000)
CRAWL_MIN = Fr(0)                  # off-grid rows: no tail-crawl minimum
T_GRID = MS                        # off-grid rows: T_xi on the 1 ms grid
BASE_PLAN = dict(family='canonical', v_xi='10.45', v_c1='1.0', t_d1='0.1')
LAMS = ['0.10', '0.11', '0.12', '0.13', '0.14', '0.15', '0.16', '0.17',
        '0.18', '0.19', '0.20']


def dstr(x):
    """exact decimal string of a terminating rational (else 'p/q')."""
    x = cq.F(x)
    d = x.denominator
    while d % 2 == 0:
        d //= 2
    while d % 5 == 0:
        d //= 5
    if d != 1:
        return str(x)
    k = 0
    while (x * 10 ** k).denominator != 1:
        k += 1
    s = str(abs(x.numerator * 10 ** k // x.denominator)).rjust(k + 1, '0')
    out = s[:-k] + '.' + s[-k:] if k else s
    return ('-' if x < 0 else '') + out


def closure_sum(cfg):
    n = int(cfg['n'])
    d_s = [cq.F(x) for x in cfg['d_s']]
    c0 = [cq.F(x) for x in cfg['c0']]
    return sum(c0[i - 1] - d_s[i] for i in range(1, n))


def common_closure(cfg):
    n = int(cfg['n'])
    d_s = [cq.F(x) for x in cfg['d_s']]
    c0 = [cq.F(x) for x in cfg['c0']]
    C = [c0[i - 1] - d_s[i] for i in range(1, n)]
    assert len(set(C)) == 1, 'grid-aligned plans need equal closures'
    return C[0]


# ------------------------------------------------------- off-grid plans
def t_cn(v_xi, a_c, a_d, t_d1, T, v_c1, Csum):
    dv = v_xi - v_c1 - a_c * T
    if dv <= 0:
        return None, dv
    return t_d1 + Csum / dv + dv / (a_d - a_c), dv


def min_T_xi(plan, cfg, crawl_min=CRAWL_MIN, grid=T_GRID):
    """off-grid plans (v_c1 fixed): smallest T_xi on the grid with
    t^c_n(T_xi) < T_xi and T_xi - t^c_n(T_xi) >= crawl_min (T - t^c_n(T)
    is increasing in T for the parameters used here)."""
    v_xi = cq.F(plan['v_xi']); a_c = cq.F(plan['a_c']); a_d = cq.F(plan['a_d'])
    t_d1 = cq.F(plan['t_d1']); v_c1 = cq.F(plan['v_c1'])
    Csum = closure_sum(cfg)
    m = max(1, int(math.ceil((t_d1 + crawl_min) / grid)))
    while True:
        T = m * grid
        tc, dv = t_cn(v_xi, a_c, a_d, t_d1, T, v_c1, Csum)
        if tc is None:
            return None
        if T > tc and T - tc >= crawl_min:
            return T
        m += 1
        if m * grid > 400:
            return None


def make_cfg(lam, a_c, a_d, vhnd=1e-3, dbar=0.020, T_xi=None, crawl_min=CRAWL_MIN,
             grid=T_GRID, **kw):
    """off-grid canonical plan with v_c1 fixed (default 1.0 m/s) and T_xi
    minimized on the grid; the jump instants t^d_i, t^c_i are generally
    off the 1 ms grid."""
    plan = dict(BASE_PLAN, a_c=dstr(a_c), a_d=dstr(a_d))
    plan.update(kw.pop('plan', {}))
    cfg = cq.config(lam=lam, vhnd=vhnd, dbar=dbar, **kw)
    if T_xi is None:
        T = min_T_xi(plan, cfg, crawl_min=crawl_min, grid=grid)
        if T is None:
            return None
        T_xi = dstr(T)
    plan['T_xi'] = dstr(T_xi)
    cfg['plan'] = dict(cfg['plan'], **plan)
    return cfg


# --------------------------------------------------- grid-aligned plans
def aligned_cfg(lam, vhnd, t1_ms, p_ms, q_ms, a_d=A_D, crawl_ms=1, dbar=0.020,
                v_xi=V_XI, **kw):
    """canonical plan with every jump instant on the 1 ms grid: t^d_1 =
    t1 ms, offsets p ms, pulse q ms, T_xi = t^c_n + crawl_ms ms; a_c and
    v_c1 exact rationals."""
    cfg = cq.config(lam=lam, vhnd=vhnd, dbar=dbar, **kw)
    n = int(cfg['n'])
    C = common_closure(cfg)
    delta = p_ms * MS; D = q_ms * MS; t1 = t1_ms * MS
    dv = C / delta
    a_c = a_d - dv / D
    T = t1 + (n - 1) * delta + D + crawl_ms * MS
    v_c1 = v_xi - dv - a_c * T
    if not (a_c > 0 and dv > 0 and v_c1 > 0):
        return None
    cfg['plan'] = dict(family='canonical', v_xi=dstr(v_xi), a_c=dstr(a_c),
                       a_d=dstr(a_d), t_d1=dstr(t1), T_xi=dstr(T),
                       v_c1=dstr(v_c1),
                       grid_ms=dict(t_d1=t1_ms, offset=p_ms, D=q_ms,
                                    tail_crawl=crawl_ms))
    return cfg


def best_aligned(a_c_min, t1_ms, a_d=A_D, C=Fr(45), n=5, v_xi=V_XI,
                 v_c1_min=V_C1_MIN, crawl_ms=1, top=40):
    """integer (p, q) minimizing T_xi + v_c1 subject to a_c >= a_c_min and
    v_c1 >= v_c1_min, with T_xi = t1 + (n-1) p + q + crawl_ms (ms).
    Floating-point scan, exact confirmation of the best candidates."""
    a_d_f = float(a_d); C_f = float(C); vx = float(v_xi); vmin = float(v_c1_min)
    L0 = (t1_ms + crawl_ms) / 1000.0
    p_lo = int(math.floor(1000 * C_f / (vx - vmin))) + 1
    cands = []
    for p in range(p_lo, 4 * p_lo):
        delta = p / 1000.0; dv = C_f / delta
        L = L0 + (n - 1) * delta
        q_min = int(math.ceil(1000 * dv / (a_d_f - a_c_min) - 1e-9))
        b = vx - vmin - a_d_f * L
        Dp = (b + math.sqrt(b * b + 4 * a_d_f * dv * L)) / (2 * a_d_f)
        q_max = int(math.floor(1000 * Dp + 1e-9))
        if q_max < q_min:
            continue
        q = np.arange(max(q_min - 1, 1), q_max + 2)
        D = q / 1000.0
        a_c = a_d_f - dv / D
        T = L + D
        vc1 = vx - dv - a_c * T
        ok = (a_c >= a_c_min - 1e-12) & (vc1 >= vmin - 1e-12) & (a_c > 0)
        if not ok.any():
            continue
        obj = np.where(ok, T + vc1, np.inf)
        j = int(np.argmin(obj))
        cands.append((float(obj[j]), p, int(q[j])))
    cands.sort()
    best = None
    for obj, p, q in cands[:top]:
        for qq in (q - 1, q, q + 1):
            delta = p * MS; D = qq * MS
            dv = C / delta; a_c = a_d - dv / D
            T = (t1_ms + crawl_ms) * MS + (n - 1) * delta + D
            vc1 = v_xi - dv - a_c * T
            if a_c < Fr(a_c_min) or vc1 < v_c1_min or a_c <= 0:
                continue
            key = (T + vc1, T, p, qq)
            if best is None or key < best[0]:
                best = (key, dict(p_ms=p, q_ms=qq, a_c=a_c, T_xi=T, v_c1=vc1,
                                  Delta_v=dv, obj=T + vc1))
    return None if best is None else best[1]


def probe(lam, vhnd, a_d=A_D, **kw):
    """plan-independent envelope data: F^+_i(0), the head's certified S1
    detection deadline, the entry deadline (any admissible plan)."""
    cfg = make_cfg(lam, Fr('0.03'), a_d, vhnd=vhnd, **kw)
    r = cq.evaluate(cfg)
    K = r['consts']
    return dict(F_plus_max=max(K['F_plus_0']), tset_head=K['tset'][0],
                T_ent_bar=K['T_ent_bar'])


def head_onset_ms(tset_head):
    """first 1 ms instant at or after the head's certified S1 deadline."""
    return int(math.ceil(tset_head * 1000 - 1e-9))


# ------------------------------------------------------------ rows
def pref_check(r):
    c = r['conds']
    adm = min(c['H4D_i']['slack'], c['H4D_ii']['slack'], c['H4D_iii']['slack'])
    au = min(c['H1D_i']['slack'], c['H1D_ii']['slack'], c['H1D_iii']['slack'])
    return adm, au, bool(r['admitted'] and adm >= PREF_ADM and au >= PREF_AUTH)


def row_of(cfg, r, r21=None):
    K = r.get('consts', {})
    pl = r['plan']
    adm, au, pok = pref_check(r) if r['closed'] and 'consts' in r else (None, None, False)
    g = cfg['plan'].get('grid_ms')
    return dict(
        lam=cfg['lam'], a_c=float(cq.F(cfg['plan']['a_c'])),
        a_c_exact=cfg['plan']['a_c'],
        a_d=float(cq.F(cfg['plan']['a_d'])), vhnd=cfg['vhnd'],
        v_xi=float(cq.F(cfg['plan']['v_xi'])), v_c1=pl['v_c1'],
        v_c1_exact=cfg['plan']['v_c1'],
        t_d1=float(cq.F(cfg['plan']['t_d1'])), T_xi=pl['T_xi'],
        c0=cfg['c0'][0],
        on_grid=g is not None, grid_ms=g,
        t_c_n=pl.get('t_c', [None])[-1] if 't_c' in pl else None,
        tail_crawl=None if 't_c' not in pl else pl['T_xi'] - pl['t_c'][-1],
        Delta_v=pl.get('Delta_v'), D=pl.get('D'),
        offset=None if 'offsets' not in pl else pl['offsets'][0],
        head_join_speed=None if 't_c' not in pl else
        pl['v_c1'] + float(cq.F(cfg['plan']['a_c'])) * (pl['T_xi'] - pl['t_c'][0]),
        tail_join_speed=None if 't_c' not in pl else
        pl['v_c1'] + float(cq.F(cfg['plan']['a_c'])) * (pl['T_xi'] - pl['t_c'][-1]),
        admitted=r['admitted'], failed=r['failed'],
        admitted_21ms=None if r21 is None else r21['admitted'],
        failed_21ms=None if r21 is None else r21['failed'],
        pref_ok=bool(pok and (r21 is None or r21['admitted'])),
        adm_slack=adm, auth_slack=au,
        slack={k: c['slack'] for k, c in r['conds'].items()},
        T_f_bar=K.get('T_f_bar'), T_ent_bar=K.get('T_ent_bar'),
        settling_bound=K.get('settling_bound'), align=K.get('align'),
        gaperr_max=None if not K else max(K['gaperr']),
        F_plus_0_max=None if not K else max(K['F_plus_0']))


def ceil2(x):
    return math.ceil(x * 100 - 1e-9) / 100


def design_for(lam, vhnd, a_d=A_D, pref=True, reserve=True, crawl_ms=1,
               t1_ms=None, a_c_min=None, **kw):
    """grid-aligned design for (lambda, V^hnd): a_c_min from the envelope
    (+ the authority margin if pref), (p, q) by best_aligned; stepping a_c
    up by 1e-4 if the certificate or the preferred slacks fail."""
    pb = probe(lam, vhnd, a_d=a_d, **kw)
    t1 = head_onset_ms(pb['tset_head']) if t1_ms is None else t1_ms
    cfg0 = cq.config(lam=lam, vhnd=vhnd, **kw)
    C = common_closure(cfg0)
    n = int(cfg0['n'])
    # 1e-9: keeps the floating-point slack of H1D(i) at or above the margin
    base = pb['F_plus_max'] + (PREF_AUTH if pref else 0.0) + 1e-9
    if reserve and not pref:
        # without the authority margin the 1 ms reserve binds H1D(i):
        # F^+_i(0) grows with the age bound
        pb21 = probe(lam, vhnd, a_d=a_d, dbar=0.021, **kw)
        base = max(base, pb21['F_plus_max'] + 1e-9)
    ac_min = base if a_c_min is None else a_c_min
    for it in range(30):
        sol = best_aligned(ac_min, t1, a_d=a_d, C=C, n=n, crawl_ms=crawl_ms)
        if sol is None:
            return dict(lam=lam, vhnd=vhnd, feasible=False, note='no aligned plan',
                        a_c_min=ac_min, t1_ms=t1)
        cfg = aligned_cfg(lam, vhnd, t1, sol['p_ms'], sol['q_ms'], a_d=a_d,
                          crawl_ms=crawl_ms, **kw)
        r = cq.evaluate(cfg)
        r21 = cq.evaluate(dict(cfg, dbar=0.021)) if reserve else None
        row = row_of(cfg, r, r21)
        ok = row['pref_ok'] if pref else (r['admitted'] and (r21 is None or r21['admitted']))
        if ok:
            row.update(feasible=True, iterations=it + 1, a_c_min=ac_min, t1_ms=t1,
                       p_ms=sol['p_ms'], q_ms=sol['q_ms'], cfg=cfg,
                       probe=pb)
            return row
        c = r['conds']
        if ((c['H1D_i']['slack'] is not None and
             c['H1D_i']['slack'] < (PREF_AUTH if pref else 0.0)) or
                (r21 is not None and 'H1D_i' in r21['failed'])):
            ac_min += 1e-4
            continue
        row.update(feasible=False,
                   note='fails %s at 20 ms, %s at 21 ms, preferred slacks %s'
                   % (r['failed'], None if r21 is None else r21['failed'],
                      row['pref_ok']),
                   a_c_min=ac_min, t1_ms=t1, cfg=cfg, probe=pb)
        return row
    return dict(lam=lam, vhnd=vhnd, feasible=False, note='no convergence')


def best_over_lam(vhnd, verbose=False, **kw):
    """design_for over the lambda grid at the pad vhnd; the feasible design
    with the smallest T_f_bar (used by the alternatives and the trade-off
    rows in which the plan and lambda are re-optimized)."""
    best = None; scan = []
    for lam_s in LAMS:
        d = design_for(float(lam_s), vhnd, **kw)
        scan.append(dict(lam=float(lam_s), feasible=d.get('feasible'),
                         T_f_bar=d.get('T_f_bar'), note=d.get('note')))
        if d.get('feasible') and (best is None or d['T_f_bar'] < best['T_f_bar']):
            best = d
    if best is None:
        return dict(feasible=False, note='no feasible lambda', lam_scan=scan)
    best = dict(best, lam_scan=scan)
    if verbose:
        print('  best over lambda: %.2f T_f_bar %.4f' % (best['lam'], best['T_f_bar']),
              flush=True)
    return best


def search(verbose=True, pref=True, reserve=True):
    out = []
    t0 = time.time()
    for lam_s in LAMS:
        lam = float(lam_s)
        for vhnd in (1e-3, 1e-2):
            row = design_for(lam, vhnd, pref=pref, reserve=reserve)
            out.append(row)
            if verbose:
                print('lam %.2f vhnd %g: feasible %s a_c %s T_xi %s v_c1 %s T_f_bar %s '
                      'failed %s (%.0f s)'
                      % (lam, vhnd, row.get('feasible'), row.get('a_c'),
                         row.get('T_xi'), row.get('v_c1'), row.get('T_f_bar'),
                         row.get('failed'), time.time() - t0), flush=True)
    return out


def select(rows, verbose=True, pref=True):
    ok = [r for r in rows if r.get('feasible')]
    if not ok:
        return None, []
    tmin = min(ceil2(r['T_f_bar']) for r in ok)
    cands = [r for r in ok if ceil2(r['T_f_bar']) == tmin]
    scored = []
    for r in cands:
        cfg = r['cfg']
        ev = cq.evaluate(cfg)
        ev21 = cq.evaluate(dict(cfg, dbar=0.021))
        bnd = cq.age_boundary_ms(cfg, kmax=200, memo={20: ev, 21: ev21})
        r = dict(r, boundary_ms=bnd['k_ms'], failed_next=bnd.get('failed_next'))
        scored.append(r)
        if verbose:
            print('  candidate lam %.2f vhnd %g: T_f_bar %.4f boundary %d ms (fails %s)'
                  % (r['lam'], r['vhnd'], r['T_f_bar'], bnd['k_ms'],
                     bnd.get('failed_next')), flush=True)
    scored.sort(key=lambda r: (-r['boundary_ms'], r['vhnd'],
                               -min(r['adm_slack'], 10.0), r['T_f_bar']))
    return scored[0], scored


# --------------------------------------------------------- records
def gap_rule(cfg, P):
    """gap rule (8) and braking order in Case D (theory Sec. 2.6)."""
    n = int(cfg['n'])
    vc = cq.F(cq.FIXED['v_c'])
    bF = [cq.F(a) / vc for a in cfg['amax']]
    s_m = cq.F(cfg['s_m'])
    d_s = [cq.F(x) for x in cfg['d_s']]
    rows = []
    for i in range(1, n):
        Pi = 1 / bF[i - 1] - 1 / bF[i]
        tci = P.extra['t_c'][i]
        v_join = P.state(i, tci)[1]
        allow = v_join * max(-Pi, Fr(0))
        rows.append(dict(
            pair=i + 1, Pi=float(Pi), t_c=float(tci), v_join=float(v_join),
            allowance_m=float(allow),
            g_star_implied_m=float(d_s[i] - s_m - allow),
            closure_reduction_max_m=float(P.v_xi * max(-Pi, Fr(0))),
            main_text_allowance_m=float(P.v_xi * max(-Pi, Fr(0))),
            age_term_m=0.0))
    return dict(
        rows=rows,
        v_join_max=max(r['v_join'] for r in rows),
        allowance_max_m=max(r['allowance_m'] for r in rows),
        g_star_implied_min_m=min(r['g_star_implied_m'] for r in rows),
        capability_order=all(r['Pi'] < 0 for r in rows),
        rule='d_s,i = s_m + g* + v_L(t^c_i) (-Pi_i)^+ (age term zero: local ramp)')


def full_record(cfg, label, interval=True, boundary=True):
    r = cq.evaluate(cfg)
    r21 = cq.evaluate(dict(cfg, dbar=0.021))
    memo = {20: r, 21: r21}
    out = dict(label=label, cfg=cfg, row=row_of(cfg, r, r21),
               admitted=r['admitted'], failed=r['failed'],
               admitted_21ms=r21['admitted'],
               slacks={k: c['slack'] for k, c in r['conds'].items()},
               T_f_bar=r['consts']['T_f_bar'] if 'consts' in r else None,
               T_xi=r['plan']['T_xi'], v_c1=r['plan']['v_c1'],
               tail_crawl=r['plan']['T_xi'] - r['plan']['t_c'][-1],
               settling_bound=r['consts'].get('settling_bound'),
               align=r['consts'].get('align'),
               gaperr_max=max(r['consts']['gaperr']),
               T_ent_bar=r['consts'].get('T_ent_bar'))
    if r['admitted'] and boundary:
        bnd = cq.age_boundary_ms(cfg, kmax=200, memo=memo)
        out['boundary_ms'] = bnd['k_ms']
        out['failed_next'] = bnd.get('failed_next')
        if interval:
            enc = ip.check_enclosure(cfg)
            out['interval_at_20ms'] = dict(
                verified=enc['verified'], all_inside=enc['all_inside'],
                lows={k: v['lo'] for k, v in enc['enclosure']['conds'].items()},
                T_f_bar=enc['enclosure']['values']['T_f_bar'])
    return out


def lam_scan(rows, cands):
    bmap = {(c['lam'], c['vhnd']): c.get('boundary_ms') for c in (cands or [])}
    out = []
    for r in rows:
        out.append(dict(lam=r['lam'], vhnd=r['vhnd'], feasible=r.get('feasible'),
                        a_c=r.get('a_c'), T_xi=r.get('T_xi'), v_c1=r.get('v_c1'),
                        T_f_bar=r.get('T_f_bar'), T_ent_bar=r.get('T_ent_bar'),
                        H3D_slack=(r.get('slack') or {}).get('H3D'),
                        F_plus_0_max=r.get('F_plus_0_max'),
                        pref_ok=r.get('pref_ok'), failed=r.get('failed'),
                        note=r.get('note'),
                        boundary_ms=bmap.get((r['lam'], r['vhnd']))))
    return out


def _strip(row):
    return {k: v for k, v in row.items() if k not in ('cfg', 'probe')}


BOX_CONDS = ('H1D_i', 'H1D_ii', 'H6D')     # conditions that use the switch box


def cond_boundaries(cfg, memo=None, kmax=200):
    """Per-condition age boundaries (ms).  The box-dependent conditions
    are located with the box of the certificate, one-sided exactly while
    P_S1 holds (one_sided=None), on a fresh memo; see the module notes."""
    cb = cq.cond_boundaries_ms(cfg, kmax=kmax, memo=memo)
    cb.update(cq.cond_boundaries_ms(dict(cfg, one_sided=None), kmax=kmax,
                                    names=BOX_CONDS))
    for nm in BOX_CONDS:
        cb[nm]['box_rule'] = 'one-sided while P_S1 holds, symmetric otherwise'
    return cb


def main_design(sel, rows=None, cands=None):
    cfg = sel['cfg']
    t0 = time.time()
    r = cq.evaluate(cfg)
    memo = {20: r}
    bnd = cq.age_boundary_ms(cfg, kmax=200, memo=memo)
    cb = cond_boundaries(cfg, memo=memo)
    enc20 = ip.check_enclosure(cfg)
    kb = bnd['k_ms']
    enc_b = ip.check_enclosure(dict(cfg, dbar=kb * 1e-3))
    enc_b1 = ip.check_enclosure(dict(cfg, dbar=(kb + 1) * 1e-3))
    ratios = {}
    for nm in ('H4D_i', 'H4D_ii', 'H4D_iii'):
        ratios[nm] = r['conds'][nm]['slack'] / PREF_ADM
    for nm in ('H1D_i', 'H1D_ii', 'H1D_iii'):
        ratios[nm] = r['conds'][nm]['slack'] / PREF_AUTH
    bind_pref = min(ratios, key=ratios.get)
    P = cq.plan(cfg)
    lam = sel['lam']; vh = sel['vhnd']
    a_c = cq.F(cfg['plan']['a_c'])
    g = cfg['plan']['grid_ms']
    # ---- alternatives
    # (a) off-grid plan: same lambda, pad, head onset and crawl rate,
    #     v_c1 = 1.0 exactly, T_xi smallest on the 1 ms grid
    alt_offgrid = full_record(
        make_cfg(lam, a_c, A_D, vhnd=vh, plan=dict(t_d1=cfg['plan']['t_d1'])),
        'off-grid jump instants: same lambda, pad, head onset and crawl rate, '
        'v_c1 = 1.0 m/s exactly, T_xi the smallest 1 ms instant after t^c_n')
    # (b) the stage-1 main deceleration 1.17 under the same policy (plan and
    #     lambda re-optimized at the selected pad)
    d117 = best_over_lam(vh, a_d=Fr('1.17'))
    alt_117 = full_record(d117['cfg'], 'a_d = 1.17 m/s^2 (stage-1 value, '
                          'rejected by the authors), same policy, plan and '
                          'lambda re-optimized') \
        if d117.get('feasible') else _strip(d117)
    if d117.get('feasible'):
        alt_117['lam_scan'] = d117['lam_scan']
    # (c) margins: authority margin dropped (plan and lambda re-optimized at
    #     the selected pad; admission margin and 1 ms reserve kept)
    d_auth = best_over_lam(vh, pref=False, reserve=True)
    alt_auth = full_record(d_auth['cfg'], 'no authority margin (a_c at its '
                           'certified minimum on the 1 ms plan grid); plan and '
                           'lambda re-optimized at the selected pad; admission '
                           'margin and 1 ms reserve kept') \
        if d_auth.get('feasible') else _strip(d_auth)
    if d_auth.get('feasible'):
        alt_auth['lam_scan'] = d_auth['lam_scan']
    # (d) all margins dropped, lambda and pad re-selected
    rows_nm = search(verbose=False, pref=False, reserve=False)
    ok_nm = [x for x in rows_nm if x.get('feasible')]
    best_nm = min(ok_nm, key=lambda x: (x['T_f_bar'], x['lam'], x['vhnd']))
    alt_nomargin = full_record(best_nm['cfg'], 'no design margins (authority, '
                               'admission, 1 ms reserve), lambda and pad '
                               're-selected for the smallest T_f_bar')
    alt_nomargin['lam_scan'] = [dict(lam=x['lam'], vhnd=x['vhnd'],
                                     feasible=x.get('feasible'),
                                     T_f_bar=x.get('T_f_bar'), a_c=x.get('a_c'),
                                     failed=x.get('failed'), note=x.get('note'))
                                for x in rows_nm]
    design = dict(
        case='D', date='2026-09-27',
        revision='stage 2: a_d = a_b = 1.0 m/s^2 (authors), jump instants on the '
                 '1 ms grid (referee D6), no tail-crawl margin (D1), design '
                 'margins of Case C (D2)',
        selection_rule=('grid-aligned canonical plan (t^d_1, offsets, pulse '
                        'duration and T_xi integer multiples of 1 ms; a_c and '
                        'v_c1 exact rationals); admitted at 20 ms and 21 ms '
                        '(1 ms reserve); preferred slacks at 20 ms (H4D >= '
                        '%.2f m, H1D(i), (ii), (iii) >= %.2f m/s^2); v_c1 >= '
                        '1.0 m/s; T_xi = t^c_n + 1 ms; head onset at the first '
                        '1 ms instant at or after the head S1 detection '
                        'deadline; smallest T_f_bar rounded up to 0.01 s over '
                        'lambda in {0.10, ..., 0.20} and V^hnd in {1e-3, '
                        '1e-2}; then the largest certified age boundary; then '
                        'the smaller pad; then the larger admission slack'
                        % (PREF_ADM, PREF_AUTH)),
        design_margins=dict(authority_m_s2=PREF_AUTH, admission_m=PREF_ADM,
                            age_reserve_ms=1, tail_crawl_s=0.001,
                            note='the tail crawl of 1 ms is the smallest grid '
                                 'step; premise P3 needs t^c_n < T_xi'),
        fixed_choices=dict(alpha=0.35, eps_e=0.15, eps_v=0.02, v_xi=10.45,
                           v_c1_min=1.0, a_d=1.0, family='canonical',
                           grid='1 ms'),
        plan_grid_ms=dict(g, T_xi=int(round(float(P.T_xi) * 1000))),
        plan_exact=dict(
            a_c=cfg['plan']['a_c'], v_c1=cfg['plan']['v_c1'],
            Delta_v=str(P.extra['Delta_v']), D=str(P.extra['D']),
            t_d=[str(x) for x in P.extra['t_d']],
            t_c=[str(x) for x in P.extra['t_c']], T_xi=str(P.T_xi),
            tau_star=str(P.tau_star)),
        jump_instants_on_grid=all(
            (x * 1000).denominator == 1 for x in
            list(P.extra['t_d']) + list(P.extra['t_c']) + [P.T_xi]),
        cfg=cfg, certificate=r,
        slacks={k: c['slack'] for k, c in r['conds'].items()},
        preferred_slack_ratios=ratios, binding_preferred=bind_pref,
        boundary_ms=bnd, binding_age=bnd.get('failed_next'),
        cond_boundary_ms=cb,
        interval_at_20ms=dict(verified=enc20['verified'],
                              all_inside=enc20['all_inside'],
                              conds=enc20['enclosure']['conds'],
                              values={k: v for k, v in enc20['enclosure']['values'].items()
                                      if k != 'authority'},
                              scope=enc20['enclosure']['scope']),
        interval_at_boundary=dict(dbar_ms=kb, verified=enc_b['verified'],
                                  all_inside=enc_b['all_inside'],
                                  conds=enc_b['enclosure']['conds'],
                                  T_f_bar=enc_b['enclosure']['values']['T_f_bar'],
                                  fp_H3D=enc_b['fp_H3D'], fp_C_hnd=enc_b['fp_C_hnd']),
        interval_at_boundary_plus_1=dict(dbar_ms=kb + 1,
                                         verified=enc_b1['verified'],
                                         failed_fp=bnd.get('failed_next'),
                                         conds=enc_b1['enclosure']['conds']),
        gap_rule=gap_rule(cfg, P),
        lam_vhnd_scan=None if rows is None else lam_scan(rows, cands),
        alternatives=dict(off_grid=alt_offgrid, a_d_117=alt_117,
                          no_authority_margin=alt_auth, no_margins=alt_nomargin),
        runtime_s=None)
    design['runtime_s'] = time.time() - t0
    return design


def tradeoff(sel):
    """at most 12 rows around the selected design."""
    lam = sel['lam']; vh = sel['vhnd']
    cfg_sel = sel['cfg']
    a_c = cq.F(cfg_sel['plan']['a_c'])
    t1 = cfg_sel['plan']['grid_ms']['t_d1']
    rows = []

    def add(label, cfg, note=''):
        if cfg is None:
            rows.append(dict(label=label, note='no feasible plan')); return
        r = cq.evaluate(cfg)
        r21 = cq.evaluate(dict(cfg, dbar=0.021))
        row = row_of(cfg, r, r21)
        row.update(label=label, note=note)
        if r['admitted']:
            row['boundary_ms'] = cq.age_boundary_ms(
                cfg, kmax=200, memo={20: r, 21: r21})['k_ms']
        rows.append(row)
        print('  row %-52s a_c %.5f T_xi %.3f v_c1 %.4f T_f_bar %.4f adm %s pref %s '
              'boundary %s' % (label, row['a_c'], row['T_xi'], row['v_c1'],
                               row['T_f_bar'] or float('nan'), row['admitted'],
                               row['pref_ok'], row.get('boundary_ms')), flush=True)

    def aligned(label, note='', **kw):
        if 'lam' in kw:
            d = design_for(kw.pop('lam'), kw.pop('vhnd', vh), **kw)
        else:
            d = best_over_lam(kw.pop('vhnd', vh), **kw)
        if not d.get('feasible'):
            rows.append(dict(label=label, note='no feasible plan: %s' % d.get('note')))
            print('  row %-52s infeasible %s' % (label, d.get('note')), flush=True)
            return
        add(label, d['cfg'], note)

    add('selected (a_d = 1.0, on the 1 ms grid)', cfg_sel)
    aligned('a_d = 1.17 (stage-1 value)', a_d=Fr('1.17'),
            note='plan and lambda re-optimized')
    aligned('a_d = 0.8', a_d=Fr('0.8'), note='plan and lambda re-optimized')
    add('jump instants off the 1 ms grid (v_c1 = 1.0)',
        make_cfg(lam, a_c, A_D, vhnd=vh, plan=dict(t_d1=cfg_sel['plan']['t_d1'])),
        'same lambda, pad, head onset and crawl rate')
    aligned('tail crawl 0.5 s', crawl_ms=500,
            note='the 0.5 s tail-crawl margin of the first round-4 design; '
                 'plan and lambda re-optimized')
    add('stage-1 conventions (t^d_1 = 0.1 s, T_xi on the 0.1 s grid)',
        make_cfg(lam, Fr(math.ceil(float(a_c) * 1000 - 1e-9), 1000), A_D, vhnd=vh,
                 grid=Fr(1, 10)),
        'same lambda and pad; a_c rounded up to the 0.001 grid, v_c1 = 1.0 '
        'exactly, off the 1 ms grid')
    lam_alt = 0.14 if abs(lam - 0.14) > 1e-9 else 0.12
    aligned('lambda = %.2f' % lam_alt, lam=lam_alt,
            note='plan re-optimized (a_c follows F^+_max(0))')
    aligned('lambda = 0.20', lam=0.20, note='plan re-optimized')
    aligned('crawl rate a_c >= 0.05', a_c_min=0.05,
            note='plan and lambda re-optimized with the faster crawl')
    aligned('no authority margin (a_c at its certified minimum)', pref=False,
            note='admission margin and 1 ms reserve kept; plan and lambda '
                 're-optimized')
    add('tail joins the crawl at 1.5 m/s (crawl band 1.5 to 1.0 m/s)',
        make_cfg(lam, a_c, A_D, vhnd=vh, plan=dict(t_d1=cfg_sel['plan']['t_d1']),
                 crawl_min=(Fr(3, 2) - Fr(1)) / a_c),
        'same lambda, pad and crawl rate; off the 1 ms grid; v_c1 = 1.0 exactly')
    aligned('c_i(0) = 48 m', c0=[48.0] * 4, note='plan and lambda re-optimized')
    assert len(rows) <= 12
    return rows


# =================================================== fan family (round 5)
# Authors' decisions (round 5): the fan (common onset, unit-specific constant
# main rates gentler along the train, joins onto a common slowly decelerating
# crawl), an approach of about 10 s at a gentle pre-braking rate, a common
# tail crawl of about 5 s before T_xi, every planned rate in [-k, 0) with a
# braking margin, v_c1 near 1.0 m/s, clearances 50 m, T_m = 1 ms, 19-step
# transport, age bound 20 ms, 1 ms of age reserve, admission slack at least
# 0.15 m, authority slack at least 0.02 m/s^2 on the lower and braking sides
# and an authority margin of 0.005 m/s^2 on the slow pieces (design start:
# an almost flat crawl), all plan instants on the 1 ms grid.
FAN_T0_MS = 10000                  # common onset: approach of 10 s
FAN_TAIL_MS = 5000                 # common tail crawl before T_xi
# Round 6 (2026-09-28, authors' decision): ramped main phases.  The planned
# deceleration of every unit rises over the first 35 % of its main duration
# to its peak, holds the peak and falls back over the last 35 %, as a
# symmetric staircase on the planning grid of 250 ms (certify_plan.fan_plan).
# The speed curves are then curved instead of straight, and the closure
# equations of the fan are unchanged.  The peak rate of the head is
# 0.45 m/s^2; the round-5 design (constant rates, head rate 0.3 m/s^2) is
# the trade-off row 'constant rates' and is archived in caseD_round5.
FAN_A_HEAD = Fr('0.45')            # head peak rate
FAN_RAMP = Fr('0.35')              # ramp fraction rho of every main duration
FAN_TP_MS = 250                    # planning period T_p of the staircase
FAN_MARGIN_UP = 0.005              # authority margin of the slow pieces
FAN_BOX = (0.05, 0.005)            # handoff box (eps_e [m], eps_v [m/s])
FAN_VHND = 1e-3
FAN_LAMS = ['0.05', '0.06', '0.07', '0.08', '0.09', '0.10', '0.11', '0.12',
            '0.13', '0.14', '0.15', '0.16', '0.17', '0.18', '0.19', '0.20']


def fan_cfg(lam, vhnd, t0_ms, L1_ms, delta_ms, a_head, tail_ms, dbar=0.020,
            a_0=None, v_xi=V_XI, box=FAN_BOX, ramp=FAN_RAMP, Tp_ms=FAN_TP_MS,
            **kw):
    """grid-aligned fan plan: onset t0, head main duration L1, join offset
    delta and tail crawl in ms; the head peak rate a_head exact; the crawl
    rate a_c = a_head - G/(L1 - T^r_1) and G = 2C/delta exact rationals;
    a_0 the approach rate (default a_c); ramp the ramp fraction rho of
    every main duration on the planning grid Tp_ms (ramp = 0 or None: the
    fan with constant rates, T^r_1 = 0)."""
    kw = dict(kw)
    if box is not None:
        kw.setdefault('eps_e', box[0]); kw.setdefault('eps_v', box[1])
    cfg = cq.config(lam=lam, vhnd=vhnd, dbar=dbar, **kw)
    n = int(cfg['n'])
    common_closure(cfg)
    T = (t0_ms + L1_ms + (n - 1) * delta_ms + tail_ms) * MS
    pl = dict(family='fan', v_xi=dstr(v_xi), t_0=dstr(t0_ms * MS),
              L_1=dstr(L1_ms * MS), delta=dstr(delta_ms * MS),
              a_head=dstr(a_head), T_xi=dstr(T),
              grid_ms=dict(t_0=t0_ms, L_1=L1_ms, delta=delta_ms,
                           tail_crawl=tail_ms))
    if ramp:
        pl['ramp'] = dstr(ramp)
        pl['T_p'] = dstr(Tp_ms * MS)
        pl['grid_ms']['T_p'] = Tp_ms
    if a_0 is not None:
        pl['a_0'] = dstr(a_0)
    cfg['plan'] = pl
    return cfg


def ramp_time_ms(L_ms, ramp, Tp_ms):
    """ramp time T^r = N T_p in ms of a main duration L (ms): N = floor(rho
    L/T_p + 1/2), at most floor(L/(2 T_p)) (the rule of
    certify_plan.fan_plan); 0 without ramps."""
    if not ramp:
        return 0
    N = min((Fr(ramp) * L_ms / Tp_ms + Fr(1, 2)).__floor__(),
            (Fr(L_ms) / (2 * Tp_ms)).__floor__())
    return N * Tp_ms


def smallest_L1_ms(D, ramp, Tp_ms):
    """smallest integer L1 (ms) with L1 - T^r_1(L1) >= D (D in ms, exact
    rational): for every number N of ramp steps the main durations with
    N = floor(rho L/T_p + 1/2) form the interval [(N - 1/2) T_p/rho,
    (N + 1/2) T_p/rho), on which L1 - N T_p is increasing."""
    D = Fr(D)
    if not ramp:
        return int(-((-D) // 1))
    rho = Fr(ramp)
    N0 = int(rho * D / ((1 - rho) * Tp_ms))
    best = None
    for N in range(max(N0 - 3, 0), N0 + 5):
        lo = max(D + N * Tp_ms, Fr(2 * N - 1, 2) * Tp_ms / rho, Fr(1))
        L = int(-((-lo) // 1))
        if L < Fr(2 * N + 1, 2) * Tp_ms / rho \
                and ramp_time_ms(L, ramp, Tp_ms) == N * Tp_ms:
            best = L if best is None else min(best, L)
    if best is None:                      # rho = 1/2: the cap on N is active
        L = int(-((-D) // 1))
        while L - ramp_time_ms(L, ramp, Tp_ms) < D:
            L += 1
        best = L
    assert best - ramp_time_ms(best, ramp, Tp_ms) >= D
    return best


def fan_best(a_c_min, t0_ms, a_head, tail_ms, C=Fr(45), n=5, v_xi=V_XI,
             v_c1_min=V_C1_MIN, a_0=None, a_b=Fr(1), d_ms=(7000, 16000),
             ramp=FAN_RAMP, Tp_ms=FAN_TP_MS):
    """integer (L1, delta) in ms minimizing T_xi + v_c1/a_b (the plan part
    of T_f_bar) subject to a_c = a_head - G/(L1 - T^r_1) >= a_c_min and
    v_c1 >= v_c1_min; for each delta the smallest admissible L1 (a larger
    L1 raises T_xi and lowers v_c1).  T^r_1 is the ramp time of the head
    (ramp_time_ms; 0 without ramps, where the constraint reads a_c =
    a_head - G/L1).  Exact rationals."""
    a_head = Fr(a_head); a_c_min = Fr(a_c_min)
    if a_head <= a_c_min:
        return None
    t0 = t0_ms * MS
    best = None
    for dms in range(*d_ms):
        delta = dms * MS
        G = 2 * C / delta
        # the effective duration L1 - T^r_1 must reach G/(a_head - a_c_min)
        if ramp:
            cand = (smallest_L1_ms(G / (a_head - a_c_min) * 1000, ramp, Tp_ms),)
        else:
            dmin = int(math.ceil(float(G / (a_head - a_c_min)) * 1000 - 1e-9))
            cand = (dmin - 1, dmin, dmin + 1)
        for l1ms in cand:
            D1 = (l1ms - ramp_time_ms(l1ms, ramp, Tp_ms)) * MS
            if D1 <= 0:
                continue
            L1 = l1ms * MS
            a_c = a_head - G / D1
            if a_c < a_c_min or a_c <= 0:
                continue
            a0 = a_c if a_0 is None else Fr(a_0)
            T = t0 + L1 + (n - 1) * delta + tail_ms * MS
            v_c1 = v_xi - a0 * t0 - G - a_c * (T - t0)
            if v_c1 < v_c1_min:
                continue
            key = (T + v_c1 / a_b, T, dms, l1ms)
            if best is None or key < best[0]:
                best = (key, dict(delta_ms=dms, L1_ms=l1ms, a_c=a_c, G=G,
                                  T_xi=T, v_c1=v_c1))
            break
    return None if best is None else best[1]


def fan_probe(lam, vhnd, dbar=0.020, **kw):
    """plan-independent envelope data (Lemma D1(e)): F^+_i(0), F^+_i(t_0),
    the S1 detection deadlines and the entry deadline, from one admissible
    fan plan."""
    cfg = fan_cfg(lam, vhnd, FAN_T0_MS, 40000, 10000, Fr('0.3'), 1000,
                  dbar=dbar, ramp=None, **kw)
    r = cq.evaluate(cfg)
    K = r['consts']
    t0 = FAN_T0_MS / 1000.0
    return dict(F_plus_max=max(K['F_plus_0']),
                F_plus_t0=max(a['F_plus'] for a in K['authority']
                              if abs(a['t0'] - t0) < 1e-12),
                tset=K['tset'], T_ent_bar=K['T_ent_bar'])


def fan_pref(r, margin_up=FAN_MARGIN_UP):
    c = r['conds']
    adm = min(c['H4D_i']['slack'], c['H4D_ii']['slack'], c['H4D_iii']['slack'])
    up = c['H1D_i']['slack']
    lo = min(c['H1D_ii']['slack'], c['H1D_iii']['slack'])
    ok = bool(r['admitted'] and adm >= PREF_ADM and up >= margin_up
              and lo >= PREF_AUTH)
    return adm, up, lo, ok


def fan_row(cfg, r, r21=None, margin_up=FAN_MARGIN_UP):
    K = r.get('consts', {})
    pl = r['plan']
    P = cq.plan(cfg)
    ex = P.extra
    adm, up, lo, pok = fan_pref(r, margin_up) if 'consts' in r else (None, None, None, False)
    g = cfg['plan'].get('grid_ms')
    return dict(
        lam=cfg['lam'], vhnd=cfg['vhnd'], eps_e=cfg['eps_e'], eps_v=cfg['eps_v'],
        a_head=float(ex['a_head']), a_head_exact=str(ex['a_head']),
        a_c=float(ex['a_c']), a_c_exact=str(ex['a_c']), a_0=float(ex['a_0']),
        t_0=float(ex['t_0']), G=float(ex['G']), rates=[float(x) for x in ex['rates']],
        t_c=[float(x) for x in ex['t_c']], T_xi=pl['T_xi'],
        tail_crawl=pl['T_xi'] - float(ex['t_c'][-1]), v_c1=pl['v_c1'],
        v_c1_exact=str(P.v_c1), v_onset=float(ex['v_onset']),
        head_join_speed=float(ex['v_join'][0]),
        tail_join_speed=float(ex['v_join'][-1]),
        on_grid=g is not None, grid_ms=g, c0=cfg['c0'][0], v0=cfg['v0'],
        a_b=cfg['a_b'], k=cfg['k'],
        ramp=float(ex['ramp']) if 'ramp' in ex else 0.0,
        T_p=float(ex['T_p']) if 'T_p' in ex else None,
        ramp_steps=list(ex['ramp_steps']) if 'ramp_steps' in ex else [0] * P.n,
        T_r=[float(x) for x in ex['T_r']] if 'T_r' in ex else [0.0] * P.n,
        n_pieces=[len(u) for u in P.units],
        admitted=r['admitted'], failed=r['failed'],
        admitted_21ms=None if r21 is None else r21['admitted'],
        failed_21ms=None if r21 is None else r21['failed'],
        pref_ok=bool(pok and (r21 is None or r21['admitted'])),
        adm_slack=adm, auth_slack_upper=up, auth_slack_lower=lo,
        slack={k: c['slack'] for k, c in r['conds'].items()},
        T_f_bar=K.get('T_f_bar'), T_ent_bar=K.get('T_ent_bar'),
        settling_bound=K.get('settling_bound'), align=K.get('align'),
        gaperr_max=None if not K else max(K['gaperr']),
        F_plus_0_max=None if not K else max(K['F_plus_0']))


def fan_design_for(lam, vhnd=FAN_VHND, a_head=FAN_A_HEAD, t0_ms=FAN_T0_MS,
                   tail_ms=FAN_TAIL_MS, margin_up=FAN_MARGIN_UP, pref=True,
                   reserve=True, a_0=None, a0_margin=None, box=FAN_BOX,
                   ramp=FAN_RAMP, Tp_ms=FAN_TP_MS, **kw):
    """grid-aligned fan design at (lambda, V^hnd): the slow rate a_c at
    least F^+_max(0) + margin_up (approach at a_c; with a0_margin the
    approach rate a_0 = F^+_max(0) + a0_margin rounded up to 1e-5 and a_c at
    least F^+_max(t_0) + margin_up), (L1, delta) by fan_best; a_c_min is
    raised by 1e-5 while the upper authority margin fails."""
    pb = fan_probe(lam, vhnd, box=box, **kw)
    pb21 = fan_probe(lam, vhnd, dbar=0.021, box=box, **kw) if reserve else None
    m = margin_up if pref else 0.0
    if a0_margin is not None:
        a_0 = Fr(int(math.ceil((pb['F_plus_max'] + a0_margin) * 1e5 - 1e-9)), 100000)
        base = pb['F_plus_t0'] + m + 1e-9
    else:
        base = pb['F_plus_max'] + m + 1e-9
        if reserve:
            base = max(base, pb21['F_plus_max'] + 1e-9)
    cfg0 = cq.config(lam=lam, vhnd=vhnd, **kw)
    C = common_closure(cfg0)
    n = int(cfg0['n'])
    for it in range(40):
        sol = fan_best(Fr(base).limit_denominator(10 ** 12), t0_ms, a_head, tail_ms,
                       C=C, n=n, a_0=a_0, a_b=cq.F(cfg0['a_b']), ramp=ramp,
                       Tp_ms=Tp_ms)
        if sol is None:
            return dict(lam=lam, vhnd=vhnd, feasible=False, note='no fan plan',
                        a_c_min=base)
        cfg = fan_cfg(lam, vhnd, t0_ms, sol['L1_ms'], sol['delta_ms'], a_head,
                      tail_ms, a_0=a_0, box=box, ramp=ramp, Tp_ms=Tp_ms, **kw)
        r = cq.evaluate(cfg)
        r21 = cq.evaluate(dict(cfg, dbar=0.021)) if reserve else None
        row = fan_row(cfg, r, r21, margin_up=m)
        ok = row['pref_ok'] if pref else (r['admitted'] and (r21 is None or r21['admitted']))
        if ok:
            row.update(feasible=True, iterations=it + 1, a_c_min=base, cfg=cfg,
                       probe=dict(pb, pb21=pb21))
            return row
        c = r['conds']
        if (c['H1D_i']['slack'] is not None and c['H1D_i']['slack'] < m) or \
                (r21 is not None and 'H1D_i' in r21['failed']):
            base += 1e-5
            continue
        row.update(feasible=False, a_c_min=base, cfg=cfg,
                   note='fails %s at 20 ms, %s at 21 ms, preferred slacks %s'
                   % (r['failed'], None if r21 is None else r21['failed'],
                      row['pref_ok']))
        return row
    return dict(lam=lam, vhnd=vhnd, feasible=False, note='no convergence')


def fan_search(verbose=True, **kw):
    """the fan design over the lambda grid, with the certified age boundary
    of every feasible row."""
    out = []
    t0 = time.time()
    for lam_s in FAN_LAMS:
        row = fan_design_for(float(lam_s), **kw)
        if row.get('feasible'):
            b = cq.age_boundary_ms(row['cfg'], kmax=200)
            row.update(boundary_ms=b['k_ms'], failed_next=b.get('failed_next'))
        out.append(row)
        if verbose:
            print('lam %s: feasible %s a_c %s T_xi %s v_c1 %s T_f_bar %s settling '
                  'bound %s boundary %s %s (%.0f s)'
                  % (lam_s, row.get('feasible'), row.get('a_c'), row.get('T_xi'),
                     row.get('v_c1'), row.get('T_f_bar'), row.get('settling_bound'),
                     row.get('boundary_ms'), row.get('failed_next') or row.get('note'),
                     time.time() - t0), flush=True)
    return out


def fan_select(rows):
    """largest certified age boundary; then the smallest T_f_bar rounded up
    to 0.01 s; then the larger hold slack (earlier certified entry)."""
    ok = [r for r in rows if r.get('feasible')]
    if not ok:
        return None
    return sorted(ok, key=lambda r: (-r['boundary_ms'], ceil2(r['T_f_bar']),
                                     -r['slack']['H3D']))[0]


FAN_SELECTION_RULE = (
    'fan plan with ramped main phases and every jump instant on the 1 ms grid '
    '(t_0 = 10 s, L_1, the join offset delta and the tail crawl of 5 s integer '
    'multiples of 1 ms; planning period T_p = 250 ms; ramp time of unit i '
    'T^r_i = T_p floor(rho L_i/T_p + 1/2) with the ramp fraction rho = 0.35; '
    'G = 2C/delta, a_c = a_head - G/(L_1 - T^r_1) and v_c1 exact rationals); '
    'head peak rate 0.45 m/s^2; approach at the crawl rate a_c; a_c at least '
    'the largest '
    'certified positive feedback envelope F^+_i(0) plus 0.005 m/s^2 (and at '
    'least F^+_i(0) at 21 ms); (L_1, delta) minimizing T_xi + v_c1/a_b with '
    'v_c1 >= 1.0 m/s; admitted at 20 ms and 21 ms; preferred slacks at 20 ms '
    '(H4D >= 0.15 m, H1D(ii), (iii) >= 0.02 m/s^2, H1D(i) >= 0.005 m/s^2); '
    'handoff box (0.05 m, 0.005 m/s), V^hnd = 1e-3; lambda on the 0.01 grid '
    'from 0.05 to 0.20: the largest certified age boundary, then the '
    'smallest T_f_bar rounded up to 0.01 s, then the larger hold slack')


def fan_full(cfg, label, interval=False, margin_up=FAN_MARGIN_UP):
    """evaluation record of one fan configuration (20 ms, 21 ms, boundary,
    optional interval check at 20 ms)."""
    r = cq.evaluate(cfg)
    r21 = cq.evaluate(dict(cfg, dbar=0.021))
    out = dict(label=label, cfg=cfg, row=fan_row(cfg, r, r21, margin_up=margin_up))
    if r['admitted']:
        bnd = cq.age_boundary_ms(cfg, kmax=200, memo={20: r, 21: r21})
        out['boundary_ms'] = bnd['k_ms']; out['failed_next'] = bnd.get('failed_next')
        if interval:
            enc = ip.check_enclosure(cfg)
            out['interval_at_20ms'] = dict(
                verified=enc['verified'], all_inside=enc['all_inside'],
                lows={k: v['lo'] for k, v in enc['enclosure']['conds'].items()},
                T_f_bar=enc['enclosure']['values']['T_f_bar'])
    return out


def fan_gap_rule(cfg, P):
    """gap rule (8) in Case D with the fan joins (theory Sec. 2.6)."""
    return gap_rule(cfg, P)


def fan_main_design(sel, rows):
    cfg = sel['cfg']
    t0 = time.time()
    r = cq.evaluate(cfg)
    r21 = cq.evaluate(dict(cfg, dbar=0.021))
    memo = {20: r, 21: r21}
    bnd = cq.age_boundary_ms(cfg, kmax=200, memo=memo)
    cb = cond_boundaries(cfg, memo=memo)
    enc20 = ip.check_enclosure(cfg)
    kb = bnd['k_ms']
    enc_b = ip.check_enclosure(dict(cfg, dbar=kb * 1e-3))
    enc_b1 = ip.check_enclosure(dict(cfg, dbar=(kb + 1) * 1e-3))
    ratios = {}
    for nm in ('H4D_i', 'H4D_ii', 'H4D_iii'):
        ratios[nm] = r['conds'][nm]['slack'] / PREF_ADM
    ratios['H1D_i'] = r['conds']['H1D_i']['slack'] / FAN_MARGIN_UP
    for nm in ('H1D_ii', 'H1D_iii'):
        ratios[nm] = r['conds'][nm]['slack'] / PREF_AUTH
    bind_pref = min(ratios, key=ratios.get)
    P = cq.plan(cfg)
    ex = P.extra
    g = cfg['plan']['grid_ms']
    inst = sorted(set(t for u in P.units for (t, a) in u) | {P.T_xi})
    design = dict(
        case='D', date='2026-09-28', family='fan',
        revision='round 6: fan plan with ramped main phases (common onset '
                 'after a 10 s approach at the crawl rate; the planned '
                 'deceleration of every unit rises over 35 % of its main '
                 'duration to its peak, holds it and falls back over the '
                 'last 35 %, as a symmetric staircase on the planning grid '
                 'of 250 ms; joins onto a common crawl, 5 s tail crawl, '
                 'synchronized braking at a_b from T_xi); head peak rate '
                 '0.45 m/s^2; handoff box (0.05 m, 0.005 m/s); lambda '
                 're-selected',
        selection_rule=FAN_SELECTION_RULE,
        design_margins=dict(authority_upper_m_s2=FAN_MARGIN_UP,
                            authority_lower_m_s2=PREF_AUTH,
                            authority_braking_m_s2=PREF_AUTH,
                            admission_m=PREF_ADM, age_reserve_ms=1,
                            tail_crawl_s=FAN_TAIL_MS / 1000.0,
                            approach_s=FAN_T0_MS / 1000.0,
                            ramp_fraction=float(FAN_RAMP),
                            planning_period_s=FAN_TP_MS / 1000.0,
                            note='the upper authority margin 0.005 m/s^2 of the '
                                 'slow pieces (approach and crawl) replaces the '
                                 'Case C value 0.02 so that the crawl is almost '
                                 'flat; the tail crawl of 5 s shows every unit '
                                 'on the crawl before the synchronized braking'),
        fixed_choices=dict(alpha=float(cfg['alpha']), eps_e=float(cfg['eps_e']),
                           eps_v=float(cfg['eps_v']), vhnd=float(cfg['vhnd']),
                           v_xi=float(P.v_xi), v_c1_min=1.0,
                           a_head=float(ex['a_head']), a_b=float(cfg['a_b']),
                           k=float(cfg['k']), v0=[float(x) for x in cfg['v0']],
                           c0=[float(x) for x in cfg['c0']], family='fan',
                           grid='1 ms'),
        plan_grid_ms=dict(g, T_xi=int(round(float(P.T_xi) * 1000))),
        plan_ramp=dict(
            ramp=str(ex['ramp']) if 'ramp' in ex else '0',
            T_p=str(ex['T_p']) if 'T_p' in ex else None,
            ramp_steps=list(ex.get('ramp_steps', [0] * P.n)),
            T_r=[str(x) for x in ex.get('T_r', [Fr(0)] * P.n)],
            plateau_start=[str(x) for x in ex.get('plateau_start', [ex['t_0']] * P.n)],
            plateau_end=[str(x) for x in ex.get('plateau_end', ex['t_c'])],
            peak_rates=[str(x) for x in ex['rates']],
            n_pieces=[len(u) for u in P.units],
            jerk_max=[float((ex['rates'][i] - ex['a_c']) / ex['T_r'][i])
                      if ex.get('T_r') and ex['T_r'][i] > 0 else None
                      for i in range(P.n)],
            note='rates: peak rates a_c + G/(L_i - T^r_i); ramp step m = 0, ..., '
                 'N_i - 1 has the rate a_c + (a_i - a_c)(2m + 1)/(2 N_i); '
                 'jerk_max: slope of the linear ramp that the staircase samples'),
        plan_exact=dict(
            a_head=str(ex['a_head']), a_c=str(ex['a_c']), a_0=str(ex['a_0']),
            G=str(ex['G']), t_0=str(ex['t_0']),
            rates=[str(x) for x in ex['rates']], L=[str(x) for x in ex['L']],
            t_c=[str(x) for x in ex['t_c']], offsets=[str(x) for x in ex['offsets']],
            v_onset=str(ex['v_onset']), v_join=[str(x) for x in ex['v_join']],
            v_c1=str(P.v_c1), T_xi=str(P.T_xi), tau_star=str(P.tau_star)),
        plan_float=dict(
            a_head=float(ex['a_head']), a_c=float(ex['a_c']), a_0=float(ex['a_0']),
            G=float(ex['G']), t_0=float(ex['t_0']),
            rates=[float(x) for x in ex['rates']], L=[float(x) for x in ex['L']],
            t_c=[float(x) for x in ex['t_c']],
            offsets=[float(x) for x in ex['offsets']],
            v_onset=float(ex['v_onset']), v_join=[float(x) for x in ex['v_join']],
            v_c1=float(P.v_c1), T_xi=float(P.T_xi), tau_star=float(P.tau_star),
            tail_crawl=float(P.T_xi - ex['t_c'][-1])),
        jump_instants_on_grid=all((x * 1000).denominator == 1 for x in inst),
        cfg=cfg, certificate=r,
        slacks={k: c['slack'] for k, c in r['conds'].items()},
        preferred_slack_ratios=ratios, binding_preferred=bind_pref,
        admitted_21ms=r21['admitted'],
        boundary_ms=bnd, binding_age=bnd.get('failed_next'),
        cond_boundary_ms=cb,
        interval_at_20ms=dict(verified=enc20['verified'],
                              all_inside=enc20['all_inside'],
                              conds=enc20['enclosure']['conds'],
                              values={k: v for k, v in enc20['enclosure']['values'].items()
                                      if k != 'authority'},
                              scope=enc20['enclosure']['scope']),
        interval_at_boundary=dict(dbar_ms=kb, verified=enc_b['verified'],
                                  all_inside=enc_b['all_inside'],
                                  conds=enc_b['enclosure']['conds'],
                                  T_f_bar=enc_b['enclosure']['values']['T_f_bar'],
                                  fp_H3D=enc_b['fp_H3D'], fp_C_hnd=enc_b['fp_C_hnd']),
        interval_at_boundary_plus_1=dict(dbar_ms=kb + 1,
                                         verified=enc_b1['verified'],
                                         failed_fp=bnd.get('failed_next'),
                                         conds=enc_b1['enclosure']['conds']),
        gap_rule=fan_gap_rule(cfg, P),
        lam_scan=[dict(lam=x['lam'], feasible=x.get('feasible'), a_c=x.get('a_c'),
                       T_xi=x.get('T_xi'), v_c1=x.get('v_c1'),
                       T_f_bar=x.get('T_f_bar'),
                       settling_bound=x.get('settling_bound'),
                       T_ent_bar=x.get('T_ent_bar'),
                       H3D_slack=(x.get('slack') or {}).get('H3D'),
                       adm_slack=x.get('adm_slack'),
                       F_plus_0_max=x.get('F_plus_0_max'),
                       boundary_ms=x.get('boundary_ms'),
                       failed_next=x.get('failed_next'), note=x.get('note'))
                  for x in rows],
        alternatives={},
        runtime_s=None)
    design['runtime_s'] = time.time() - t0
    return design


def fan_offgrid_cfg(sel_cfg):
    """off-grid variant of the selected plan (outside premise (I2) for a pure
    1 kHz controller): same lambda, box, pad, onset, head peak rate, crawl
    rate and ramp steps; the join offset 0.5 ms longer, L_1 =
    G/(a_head - a_c) + T^r_1 off the grid, T_xi the first 1 ms instant at or
    after t^c_n + 5 s.  The rising ramps start at the onset and stay on the
    grid; the ends of the plateaus, the falling ramps and the joins lie off
    it."""
    P = cq.plan(sel_cfg)
    ex = P.extra
    C = common_closure(sel_cfg)
    n = int(sel_cfg['n'])
    delta = cq.F(sel_cfg['plan']['delta']) + Fr(1, 2000)
    G = 2 * C / delta
    Tr1 = ex['T_r'][0] if 'T_r' in ex else Fr(0)
    L1 = G / (ex['a_head'] - ex['a_c']) + Tr1
    t_cn = ex['t_0'] + L1 + (n - 1) * delta
    T = Fr(int(math.ceil((t_cn + FAN_TAIL_MS * MS) * 1000)), 1000)
    cfg = json.loads(json.dumps(sel_cfg))
    cfg['plan'] = dict(family='fan', v_xi=sel_cfg['plan']['v_xi'],
                       t_0=sel_cfg['plan']['t_0'], L_1=str(L1), delta=str(delta),
                       a_head=sel_cfg['plan']['a_head'], T_xi=dstr(T))
    if 'ramp_steps' in ex:
        cfg['plan'].update(T_p=sel_cfg['plan']['T_p'],
                           ramp_steps=list(ex['ramp_steps']))
    return cfg


def fan_tradeoff(sel):
    """at most 12 rows around the selected fan design."""
    lam = sel['lam']; vh = sel['vhnd']
    rows = []

    def add(label, d, note=''):
        if not d.get('feasible'):
            rows.append(dict(label=label, note='no feasible plan: %s' % d.get('note')))
            print('  row %-44s infeasible %s' % (label, d.get('note')), flush=True)
            return
        cfg = d['cfg']
        r = cq.evaluate(cfg)
        r21 = cq.evaluate(dict(cfg, dbar=0.021))
        row = fan_row(cfg, r, r21, margin_up=d.get('margin_up', FAN_MARGIN_UP))
        row.update(label=label, note=note)
        if r['admitted']:
            # the boundary and the conditions that fail at the next age
            # (round-5 review: archived for every row)
            b = cq.age_boundary_ms(cfg, kmax=200, memo={20: r, 21: r21})
            row['boundary_ms'] = b['k_ms']
            row['failed_next'] = b.get('failed_next')
        rows.append(row)
        print('  row %-44s a_c %.5f t_c^n %.3f T_xi %.3f v_c1 %.4f T_f_bar %.3f '
              'settling %.4f boundary %s pref %s'
              % (label, row['a_c'], row['t_c'][-1], row['T_xi'], row['v_c1'],
                 row['T_f_bar'] or float('nan'), row['settling_bound'] or float('nan'),
                 row.get('boundary_ms'), row['pref_ok']), flush=True)

    add('selected', dict(sel, feasible=True))
    add('constant rates (no ramps), head rate 0.3',
        fan_design_for(lam, vh, a_head=Fr('0.3'), ramp=None),
        'the round-5 design: plan re-optimized, same lambda and box')
    add('ramp fraction 0.2, head peak rate 0.375',
        fan_design_for(lam, vh, a_head=Fr('0.375'), ramp=Fr('0.2')),
        'plan re-optimized, same lambda and box')
    add('ramp fraction 0.5, head peak rate 0.6',
        fan_design_for(lam, vh, a_head=Fr('0.6'), ramp=Fr('0.5')),
        'plan re-optimized, same lambda and box')
    add('head peak rate 0.35', fan_design_for(lam, vh, a_head=Fr('0.35')),
        'plan re-optimized, same lambda, box and ramp fraction')
    add('head peak rate 0.6', fan_design_for(lam, vh, a_head=Fr('0.6')),
        'plan re-optimized, same lambda, box and ramp fraction')
    add('planning period 100 ms', fan_design_for(lam, vh, Tp_ms=100),
        'plan re-optimized, same lambda, box and ramp fraction')
    add('no tail crawl (T_xi = t^c_n + 1 ms)', fan_design_for(lam, vh, tail_ms=1),
        'plan re-optimized')
    add('onset t_0 = 5 s', fan_design_for(lam, vh, t0_ms=5000), 'plan re-optimized')
    add('no approach (t_0 = 0)', fan_design_for(lam, vh, t0_ms=0), 'plan re-optimized')
    add('lambda = 0.11', fan_design_for(0.11, vh), 'plan re-optimized')
    add('lambda = 0.15', fan_design_for(0.15, vh), 'plan re-optimized')
    d = fan_design_for(lam, vh, margin_up=0.02)
    d['margin_up'] = 0.02
    add('crawl margin 0.02', d, 'plan re-optimized')
    add('handoff box (0.15 m, 0.02 m/s)',
        fan_design_for(lam, vh, box=(0.15, 0.02)), 'plan re-optimized')
    d = fan_design_for(lam, vh, pref=False, reserve=False)
    d['margin_up'] = 0.0
    add('no design margins (a_c at F^+_max(0), no age reserve)', d,
        'plan re-optimized')
    add('service rate a_b = 1.1 (k = 1.2)', fan_design_for(lam, vh, a_b=1.1),
        'plan re-optimized')
    assert len(rows) <= 16
    return rows


def fan_alternatives(sel):
    """alternatives kept in design.json: the off-grid variant (nominal check,
    outside the certificate), the Case C head rate and the variant without
    the tail crawl with their full records, and side studies of the
    takeover speeds, the clearances and a separate approach rate."""
    lam = sel['lam']; vh = sel['vhnd']
    alt = {}
    alt['off_grid'] = fan_full(fan_offgrid_cfg(sel['cfg']),
                               'off-grid jump instants: same lambda, box, pad, '
                               'onset, head rate and crawl rate; join offset '
                               '0.5 ms longer; L_1 = G/(a_head - a_c) off the '
                               'grid; T_xi the first 1 ms instant at or after '
                               't^c_n + 5 s')
    d = fan_design_for(lam, vh, a_head=Fr('0.3'), ramp=None)
    alt['constant_rates'] = fan_full(d['cfg'], 'constant rates (no ramps), head '
                                     'rate 0.3 m/s^2 (the round-5 design), plan '
                                     're-optimized', interval=True)
    d = fan_design_for(lam, vh, tail_ms=1)
    alt['no_tail_crawl'] = fan_full(d['cfg'], 'T_xi = t^c_n + 1 ms, plan '
                                    're-optimized', interval=True)
    d = fan_design_for(lam, vh, v0=['10.47', '10.51', '10.57', '10.65', '10.75'])
    alt['compact_takeover_speeds'] = fan_full(
        d['cfg'], 'takeover speeds (10.47, 10.51, 10.57, 10.65, 10.75) m/s, '
        'plan re-optimized') if d.get('feasible') else _strip(d)
    d = fan_design_for(lam, vh, c0=[48.0] * 4)
    alt['clearances_48m'] = fan_full(d['cfg'], 'takeover clearances 48 m, '
                                     'plan re-optimized') if d.get('feasible') else _strip(d)
    d = fan_design_for(lam, vh, a0_margin=0.02)
    alt['approach_margin_0.02'] = fan_full(
        d['cfg'], 'approach rate a_0 = F^+_max(0) + 0.02 rounded up to 1e-5, '
        'crawl rate a_c >= F^+_max(t_0) + 0.005, plan re-optimized') \
        if d.get('feasible') else _strip(d)
    return alt


def _r4dir():
    """output folder of the round-4 (canonical) parts: $CASED_R4_OUT; the
    round-4 files are archived in data/new_d19/caseD_round4."""
    d = os.environ.get('CASED_R4_OUT')
    if not d:
        raise SystemExit('round-4 parts: set CASED_R4_OUT (they would overwrite '
                         'the round-5 design.json)')
    return d


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(cp._py(obj), open(path, 'w'), indent=1)
    print('wrote', path, flush=True)


if __name__ == '__main__':
    parts = sys.argv[1:] or ['fan_search', 'fan_design', 'fan_tradeoff', 'fan_alt']
    rows = None; sel = None; scored = None
    spath = os.path.join(SCRATCH, 'design_search.json')
    fpath = os.path.join(SCRATCH, 'fan_search.json')
    frows = None; fsel = None
    for part in parts:
        if part == 'fan_search':
            frows = fan_search()
            fsel = fan_select(frows)
            print('selected lam %.2f T_f_bar %.4f settling bound %.4f boundary %s'
                  % (fsel['lam'], fsel['T_f_bar'], fsel['settling_bound'],
                     fsel['boundary_ms']), flush=True)
            _dump(dict(rows=frows, selected=fsel), fpath)
        elif part == 'fan_design':
            d = json.load(open(fpath))
            fsel = fsel or d['selected']; frows = frows or d['rows']
            des = fan_main_design(fsel, frows)
            des['search_rows'] = [_strip(x) for x in frows]
            _dump(des, os.path.join(DATA_D, 'design.json'))
        elif part == 'fan_tradeoff':
            if fsel is None:
                fsel = json.load(open(fpath))['selected']
            tr = fan_tradeoff(fsel)
            _dump(dict(note='rows around the selected fan design (ramp fraction '
                            '0.35 on the planning grid of 250 ms, head peak rate '
                            '0.45 m/s^2, approach 10 s at the crawl rate, tail '
                            'crawl 5 s, jump instants on the 1 ms grid, v_c1 >= '
                            '1.0 m/s, handoff box (0.05 m, 0.005 m/s), design '
                            'margins of the selection rule); "plan '
                            're-optimized": the grid-aligned fan is re-solved '
                            'under the same policy; slacks at 20 ms; '
                            'admitted_21ms: the 1 ms reserve; boundary_ms: the '
                            'certified age boundary; T_f_bar: certified '
                            'completion bound; settling_bound: certified bound '
                            'of the settling time (first to last stop)',
                       selection_rule=FAN_SELECTION_RULE, rows=tr),
                  os.path.join(DATA_D, 'tradeoff.json'))
        elif part == 'fan_alt':
            if fsel is None:
                fsel = json.load(open(fpath))['selected']
            dp = os.path.join(DATA_D, 'design.json')
            des = json.load(open(dp))
            assert des.get('family') == 'fan'
            des['alternatives'] = fan_alternatives(fsel)
            _dump(des, dp)
        elif part in ('search', 'design', 'lamscan', 'tradeoff', 'condbnd'):
            # round-4 (canonical) parts: outputs to $CASED_R4_OUT
            DATA_D = _r4dir()
        if part == 'search':
            rows = search()
            sel, scored = select(rows)
            print('selected lam %.2f vhnd %g T_f_bar %.4f boundary %s'
                  % (sel['lam'], sel['vhnd'], sel['T_f_bar'], sel['boundary_ms']),
                  flush=True)
            _dump(dict(rows=rows, candidates=scored, selected=sel), spath)
        elif part == 'design':
            d = json.load(open(spath))
            sel = sel or d['selected']; rows = rows or d['rows']
            scored = scored or d.get('candidates')
            des = main_design(sel, rows, scored)
            des['search_rows'] = [_strip(x) for x in rows]
            des['tie_break_candidates'] = [_strip(x) for x in scored]
            _dump(des, os.path.join(DATA_D, 'design.json'))
        elif part == 'lamscan':
            # certified age boundary of every feasible search row at the
            # selected pad (justification of lambda, referee item D7)
            d = json.load(open(spath))
            sel = sel or d['selected']
            dp = os.path.join(DATA_D, 'design.json')
            des = json.load(open(dp))
            out = []
            for r in d['rows']:
                if not r.get('feasible') or r['vhnd'] != sel['vhnd']:
                    continue
                b = cq.age_boundary_ms(r['cfg'], kmax=200)
                out.append(dict(lam=r['lam'], vhnd=r['vhnd'], a_c=r['a_c'],
                                T_xi=r['T_xi'], v_c1=r['v_c1'],
                                T_f_bar=r['T_f_bar'], H3D_slack=r['slack']['H3D'],
                                boundary_ms=b['k_ms'], failed_next=b.get('failed_next')))
                print('  lam %.2f T_f_bar %.4f boundary %d ms (%s)'
                      % (r['lam'], r['T_f_bar'], b['k_ms'], b.get('failed_next')),
                      flush=True)
            des['lam_boundary_scan'] = out
            _dump(des, dp)
        elif part == 'tradeoff':
            if sel is None:
                d = json.load(open(spath)); sel = d['selected']
            tr = tradeoff(sel)
            _dump(dict(note='rows around the selected design (a_d = 1.0 m/s^2, '
                            'jump instants on the 1 ms grid, T_xi = t^c_n + 1 ms, '
                            'v_c1 >= 1.0 m/s, design margins of Case C); '
                            '"plan re-optimized": the grid-aligned plan is '
                            're-solved under the same policy; slacks at 20 ms; '
                            'admitted_21ms: the 1 ms reserve; boundary_ms: the '
                            'certified age boundary; T_f_bar: certified '
                            'completion bound',
                       rows=tr), os.path.join(DATA_D, 'tradeoff.json'))
        elif part == 'condbnd':
            dp = os.path.join(DATA_D, 'design.json')
            des = json.load(open(dp))
            cfg = des['cfg']
            memo = {20: cq.evaluate(cfg)}
            old = {k: v['k_ms'] for k, v in des['cond_boundary_ms'].items()}
            cb = cond_boundaries(cfg, memo=memo)
            new = {k: v['k_ms'] for k, v in cb.items()}
            for k in sorted(new):
                if new[k] != old.get(k):
                    print('  %-8s %s -> %s ms' % (k, old.get(k), new[k]), flush=True)
            # the conditions without the box keep their archived boundaries
            assert all(new[k] == old[k] for k in new if k not in BOX_CONDS), (old, new)
            des['cond_boundary_ms'] = cb
            _dump(des, dp)
        elif not part.startswith('fan_'):
            raise SystemExit('unknown part ' + part)
