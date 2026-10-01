#!/usr/bin/env python3
r"""run_fb_design.py -- design of the arrival with feedback-only followers
and a shaped head reference ("FB", 2026-09-29).

Setting.  Only the head stores the reference; every follower stores
constants only and regulates its clearance to its parking gap by the COAST,
S1 and S2 laws of the frozen paper from takeover, with the received command
of its predecessor as feedforward.  With the small traction limit U, a
follower closes its gap by braking less than its predecessor, so the head's
reference must decelerate while the S2 feedback of the followers is
positive.  The dispatcher shapes the reference for this (design rule):

  a_r = min(-a_cr, cap_k) on the planning period [k T_p, (k+1) T_p), where
  cap_k is the largest rate on the grid q_a that keeps the upper authority
  test P^+_i + F^+_i <= U - mu of certify_fb on every cell whose window
  [t - (i-1) dbar, t + cell] meets the period, for every unit i; the first
  pieces are held at their minimum until hold_pad after the latest S2
  switch deadline (no reference jump before every pair is in S2, (C0a));
  a_cr >= 0 (crawl rate, grid q_c) makes v_r(T_xi) = v_hand; and T_xi is
  the entry deadline plus (n-1) dbar plus T_pad, on the planning grid.

The envelopes contain the planned pulses of the reference jumps, so the
rule is iterated.  The first two iterations shape freely (the first starts
from a constant deceleration); from the third iteration on, each period's
cap is the minimum of its new and its previous value and T_xi the maximum,
so the caps can only decrease and T_xi can only increase.  The rule stops
when T_xi repeats and every period's rate repeats up to STOP_TOL steps q_c,
or at ITER_MAX iterations (the evaluated iterate is kept); no convergence
property is claimed, and the certificate is evaluated on the final
reference.  Every quantity of the reference is an exact rational.

Parts (all written to data/new_d19/fbref/design_fb.json):
  scan       the rule for every gain of GAINS at the design age bound
             dbar_des = 20 ms; certificate at 20 ms, age boundary on the
             1 ms grid (certify_fb.age_boundary_ms), completion bound;
  selection  the gain with the smallest completion bound among the gains
             certified at 20 ms (ties: the larger age boundary);
  selected   the selected design: exact reference, certificate at 20 ms,
             per-condition boundaries (certify_fb.cond_boundaries_ms), a
             brute-force monotonicity scan on the 1 ms grid, the sequential
             parking gaps of the admission condition (the construction of
             the frozen paper: pair by pair from the front, the smallest gap
             on a 1 cm grid whose admission rows hold with the reserve MU_G,
             the reference fixed), and the regressions of certify_fb.
Usage: python -B run_fb_design.py [--quick]
"""
import copy
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr
from multiprocessing import Pool

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE, os.path.join(CODE, 'src', 'certification')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                                       # noqa: E402
import certify_fb as fb                                  # noqa: E402
import certify_plan as cq                                # noqa: E402

F = cq.F
OUT = os.path.join(fb.DATA_FB, 'design_fb.json')
LOG = os.path.join(CODE, 'logs_ext', 'fbref')

# fixed choices of the design rule
FIXED_CHOICES = OrderedDict(
    v_in='10.45',          # initial reference speed [m/s]
    T_p='1/4',             # planning period [s]
    mu='0.02',             # authority margin of the shaped pieces [m/s^2]
    hold_pad='1/2',        # hold after the latest S2 switch deadline [s]
    v_hand='1',            # handoff speed [m/s]
    T_pad='2',             # T_xi - (entry deadline + (n-1) dbar) [s], at least
    q_a='1/1000',          # grid of the shaped rates [m/s^2]
    q_c='1/1000000',       # grid of the crawl rate [m/s^2]
    dbar_des='0.020',      # design age bound [s]
)
GAINS = (0.05, 0.055, 0.06, 0.065, 0.07)
ITER_MAX = 12
STOP_TOL = 10              # stopping tolerance of the rule, in steps q_c
MU_G = 0.15                # reserve of the sequential parking gaps [m]
KMAX = 120                 # search edge of the age bound [ms]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        h.update(fh.read())
    return h.hexdigest()


# ------------------------------------------------------------ the rule
def _speed_at(pieces, v_in, t):
    v = v_in
    for j, (t0, a) in enumerate(pieces):
        t1 = pieces[j + 1][0] if j + 1 < len(pieces) else None
        if t1 is None or t < t1:
            return v + a * (t - t0)
        v = v + a * (t1 - t0)
    return v


def _pieces(capq, T_p, T_xi, a_cr):
    """a_r = min(-a_cr, capq[k]) on the planning periods (merged)."""
    out = []
    k = 0
    while k * T_p < T_xi:
        c = capq[k] if k < len(capq) else Fr(0)
        a = min(-a_cr, c)
        if not out or out[-1][1] != a:
            out.append((k * T_p, a))
        k += 1
    return out


def _floor_to(x, q):
    return Fr(math.floor(Fr(x) / q)) * q


def _on_periods(pieces, T_p, T_xi):
    """values of the piecewise constant a_r on the planning periods."""
    out = []
    j = 0
    k = 0
    while k * T_p < T_xi:
        t = k * T_p
        while j + 1 < len(pieces) and pieces[j + 1][0] <= t:
            j += 1
        out.append(pieces[j][1])
        k += 1
    return out


def fb_cfg(lam, pieces, T_xi, dbar, extra=None):
    ch = FIXED_CHOICES
    kw = dict(lam=lam, dbar=float(dbar),
              plan=dict(family='shaped', v_xi=ch['v_in'], T_xi=str(T_xi),
                        pieces=[[str(t), str(a)] for t, a in pieces]))
    if extra:
        kw.update(extra)
    return fb.config(**kw)


def shape(lam, extra=None, verbose=False):
    """the design rule for the gain lam (fixed point of the iteration)."""
    ch = FIXED_CHOICES
    v_in = F(ch['v_in']); T_p = Fr(ch['T_p']); mu = float(F(ch['mu']))
    hold_pad = float(Fr(ch['hold_pad'])); v_hand = F(ch['v_hand'])
    T_pad = F(ch['T_pad']); q_a = Fr(ch['q_a']); q_c = Fr(ch['q_c'])
    dbar = F(ch['dbar_des'])
    n = 5
    T_xi = Fr(int(10.5 / lam) + 10)
    a_cr = (v_in - v_hand) / T_xi
    pieces = [(Fr(0), -_floor_to(a_cr, q_c))]
    hist = []
    res = None
    converged = False
    cap_prev = None                 # monotone rule from the second iteration on
    T_prev = None
    for it in range(ITER_MAX):
        cfg = fb_cfg(lam, pieces, T_xi, dbar, extra)
        t0 = time.time()
        res = fb.evaluate(cfg, keep_curves=True)
        cv = res['curves']
        tg = cv['tg']; Fp = cv['Fp']
        Dg = float(cfg['cell_dt'])
        U = float(cfg['U'])
        Tp = float(T_p)
        nK = int(math.ceil(float(T_xi) / Tp)) + 1
        capk = np.full(nK, np.inf)
        for i in range(n):
            val = U - mu - Fp[i]
            k_lo = np.clip(np.floor((tg - i * float(dbar)) / Tp).astype(int), 0, nK - 1)
            k_hi = np.clip(np.floor((tg + Dg - 1e-12) / Tp).astype(int), 0, nK - 1)
            for g in range(len(tg)):
                a, b = k_lo[g], k_hi[g]
                if val[g] < capk[a:b + 1].min():
                    capk[a:b + 1] = np.minimum(capk[a:b + 1], val[g])
        capk = np.minimum(capk, 0.0)
        t_hold = max(res['consts']['tset']) + hold_pad
        k_hold = min(nK, int(math.ceil(t_hold / Tp)))
        capk[:k_hold] = capk[:k_hold].min()
        capq = [_floor_to(Fr(float(c)).limit_denominator(10 ** 12), q_a) for c in capk]
        Tent = res['bounds'].get('entry_deadline')
        if Tent is None or not np.isfinite(Tent):
            Tent = float(T_xi)
        T_new = Fr(math.ceil((Tent + (n - 1) * float(dbar) + float(T_pad)) / Tp)) * T_p
        # monotone rule: once the reference is shaped (from the second
        # iteration on), a period's rate can only decrease and T_xi can only
        # increase; both live on finite grids, so the rule stops
        if cap_prev is not None:
            m = max(len(capq), len(cap_prev))
            capq = [min(capq[k] if k < len(capq) else Fr(0),
                        cap_prev[k] if k < len(cap_prev) else Fr(0)) for k in range(m)]
            T_new = max(T_new, T_prev)
        if it >= 1:
            cap_prev, T_prev = capq, T_new
        lo_a, hi_a = Fr(0), Fr(2)
        for _ in range(80):
            mid = (lo_a + hi_a) / 2
            if _speed_at(_pieces(capq, T_p, T_new, mid), v_in, T_new) > v_hand:
                lo_a = mid
            else:
                hi_a = mid
        a_cr_new = _floor_to(lo_a, q_c)
        pc_new = _pieces(capq, T_p, T_new, a_cr_new)
        hist.append(OrderedDict(
            it=it, admitted=bool(res['admitted']), failed=res['failed'],
            T_xi=float(T_xi), entry=Tent, a_cr=float(a_cr), a_r0=float(pieces[0][1]),
            n_pieces=len(pieces), T_f=res['bounds'].get('T_f_bar'),
            C7a=res['conds']['C7a']['slack'], C7b=res['conds']['C7b']['slack'],
            runtime_s=time.time() - t0))
        if verbose:
            print('   lam %.3f it %d admitted %s failed %s T_xi %.2f entry %.2f a_r0 %.4f'
                  % (lam, it, res['admitted'], res['failed'], float(T_xi), Tent,
                     float(pieces[0][1])), flush=True)
        # stop when T_xi repeats and the pieces repeat up to STOP_TOL grid
        # steps q_c of the rates (the crawl rate can alternate between two
        # neighbouring grid values because the last shaped piece and the
        # crawl piece merge); the current (evaluated) reference is kept
        same = (T_new == T_xi and
                max(abs(x - y) for x, y in zip(_on_periods(pc_new, T_p, T_xi),
                                               _on_periods(pieces, T_p, T_xi)))
                <= STOP_TOL * q_c)
        if same:
            converged = True
            break
        pieces, T_xi, a_cr = pc_new, T_new, a_cr_new
    cfg = fb_cfg(lam, pieces, T_xi, dbar, extra)
    res = fb.evaluate(cfg)
    v_hand_act = _speed_at(pieces, v_in, T_xi)
    return dict(lam=lam, pieces=pieces, T_xi=T_xi, a_cr=a_cr, cfg=cfg, res=res,
                hist=hist, converged=converged, v_hand=v_hand_act)


def plan_record(D):
    return dict(family='shaped', v_xi=FIXED_CHOICES['v_in'], T_xi=str(D['T_xi']),
                pieces=[[str(t), str(a)] for t, a in D['pieces']])


def summary_row(D):
    r = D['res']
    b = r['bounds']
    return OrderedDict(
        lam=D['lam'], converged=D['converged'], iterations=len(D['hist']),
        admitted=bool(r['admitted']), failed=r['failed'],
        T_xi=float(D['T_xi']), T_xi_exact=str(D['T_xi']), a_cr=float(D['a_cr']),
        a_cr_exact=str(D['a_cr']), a_r0=float(D['pieces'][0][1]),
        a_r0_exact=str(D['pieces'][0][1]), n_pieces=len(D['pieces']),
        v_hand=float(D['v_hand']), T_f_bar=b.get('T_f_bar'),
        dispersion=b.get('dispersion'), marker=b.get('marker_max'),
        gap=b.get('gap_max'), entry=b.get('entry_deadline'),
        clearance_floor=b.get('clearance_floor_min'),
        velocity_floor=b.get('velocity_floor'),
        slacks=OrderedDict((k, c['slack']) for k, c in r['conds'].items()))


def scan_job(lam):
    t0 = time.time()
    D = shape(lam)
    row = summary_row(D)
    if row['admitted']:
        B = fb.age_boundary_ms(fb.config(**dict(D['cfg'])), kmax=KMAX)
        row['boundary_ms'] = B['k_ms']
        row['failed_next'] = B.get('failed_next')
        row['coarse_below_all_admitted'] = B.get('coarse_below_all_admitted')
    else:
        row['boundary_ms'] = None
    row['plan'] = plan_record(D)
    row['hist'] = D['hist']
    row['runtime_s'] = time.time() - t0
    return row


# ------------------------------------------------------ selected design
def monotone_scan(cfg, k_hi):
    """brute force on the 1 ms grid: admitted set and per-condition sets."""
    rows = []
    for k in range(1, k_hi + 1):
        r = fb.evaluate(dict(cfg, dbar=k * 1e-3), want_entry=True)
        rows.append(dict(k=k, admitted=bool(r['admitted']),
                         ok={c: bool(v['ok']) for c, v in r['conds'].items()
                             if c in fb.COND_NAMES}))
    adm = [r['admitted'] for r in rows]
    initial = all(adm[:adm.index(False)]) and not any(adm[adm.index(False):]) \
        if False in adm else True
    per = {}
    for c in fb.COND_NAMES:
        s = [r['ok'][c] for r in rows]
        per[c] = (all(s[:s.index(False)]) and not any(s[s.index(False):])) \
            if False in s else True
    return dict(k_max=k_hi, admitted=adm, initial_interval=initial,
                per_condition_initial=per, all_monotone=bool(initial and all(per.values())))


def sequential_gaps(cfg, mu_g=MU_G, step=0.01):
    """the sequential construction of the frozen paper with the reference
    fixed: pair by pair from the front, the smallest parking gap on the grid
    `step` whose admission rows (C8a, C8b, C8c) of that pair hold with the
    reserve mu_g, the gaps ahead fixed at their constructed values."""
    n = int(cfg['n'])
    d_s = [float(x) for x in cfg['d_s']]
    c0 = [float(x) for x in cfg['c0']]
    s_m = float(cfg['s_m'])
    built = list(d_s)
    out = []
    for c in range(1, n):
        def ok(g):
            ds = list(built)
            ds[c] = g
            e0 = ['0'] + [str(F(str(c0[i - 1])) - F(str(ds[i]))) for i in range(1, n)]
            r = fb.evaluate(dict(cfg, d_s=ds, e0=e0), want_entry=False)
            rows = [r['conds'][nm]['per_pair'][c - 1] for nm in ('C8a', 'C8b', 'C8c')]
            return min(rows) >= mu_g, rows
        lo = s_m + step
        hi = c0[c - 1]
        if not ok(hi)[0]:
            out.append(dict(pair=c + 1, gap=None, note='no gap up to c_i(0)'))
            break
        a, b = int(round(lo / step)), int(round(hi / step))
        if ok(a * step)[0]:
            b = a
        while b - a > 1:
            m = (a + b) // 2
            if ok(m * step)[0]:
                b = m
            else:
                a = m
        g = b * step
        built[c] = g
        out.append(dict(pair=c + 1, gap=round(g, 2), rows_at_gap=ok(g)[1]))
    return dict(mu_g=mu_g, step=step, gaps=out)


def main(argv):
    quick = '--quick' in argv
    t0 = time.time()
    os.makedirs(fb.DATA_FB, exist_ok=True)
    os.makedirs(LOG, exist_ok=True)
    gains = GAINS if not quick else (0.065,)
    with Pool(min(len(gains), 10)) as P:
        rows = P.map(scan_job, gains)
    for r in rows:
        print('lam %.3f: admitted %s %s | boundary %s next %s | T_xi %.2f T_f %.2f disp %.4f '
              'marker %.3f gap %.3f gmin %.3f a_r0 %.3f a_cr %.4f (%.0f s)'
              % (r['lam'], r['admitted'], r['failed'], r['boundary_ms'], r.get('failed_next'),
                 r['T_xi'], r['T_f_bar'], r['dispersion'], r['marker'], r['gap'],
                 r['clearance_floor'], r['a_r0'], r['a_cr'], r['runtime_s']), flush=True)
    cand = [r for r in rows if r['admitted']]
    if not cand:
        raise SystemExit('no gain certified at the design age bound')
    sel = min(cand, key=lambda r: (r['T_f_bar'], -(r['boundary_ms'] or 0)))
    print('selected lam %.3f (T_f %.3f, boundary %s ms)' % (sel['lam'], sel['T_f_bar'],
                                                            sel['boundary_ms']), flush=True)
    # re-run the rule for the selected gain (deterministic) and analyze
    D = shape(sel['lam'])
    assert plan_record(D) == sel['plan'], 'the rule is not deterministic'
    cfg = D['cfg']
    r20 = fb.evaluate(dict(cfg, dbar=0.020))
    B = fb.age_boundary_ms(cfg, kmax=KMAX)
    CB = fb.cond_boundaries_ms(cfg, kmax=KMAX)
    mono = monotone_scan(cfg, min(KMAX, B['k_ms'] + 8))
    seq = sequential_gaps(dict(cfg, dbar=0.020))
    reg = fb.regress()
    out = OrderedDict(
        part='design', created=time.strftime('%Y-%m-%d %H:%M:%S'),
        purpose='design of the arrival with feedback-only followers and a shaped '
                'head reference (FB, 2026-09-29)',
        evaluator='src/certification/certify_fb.py',
        code_sha256=dict(run_fb_design=sha256_file(__file__),
                         certify_fb=sha256_file(fb.__file__),
                         certify_plan=sha256_file(cq.__file__)),
        runtime_system=dict(python=platform.python_version(), numpy=np.__version__,
                            system=platform.platform()),
        fixed_choices=FIXED_CHOICES, gains=list(gains), iter_max=ITER_MAX,
        stop_rule='T_xi repeats and every piece repeats up to %d steps q_c; the '
                  'evaluated iterate is kept' % STOP_TOL,
        benchmark=dict(n=int(cfg['n']), amax=cfg['amax'], U=cfg['U'],
                       Uminus=cfg['Uminus'], a_b=cfg['a_b'], alpha=cfg['alpha'],
                       eps_det=cfg['eps_det'], phi=cfg['phi'], T_c=cfg['T_c'],
                       eps_e=cfg['eps_e'], eps_v=cfg['eps_v'], vhnd=cfg['vhnd'],
                       lags=cfg['lags'], s_m=cfg['s_m'], d_s=cfg['d_s'], c0=cfg['c0'],
                       v0=cfg['v0'], e0=cfg['e0'], sync_tol=cfg['sync_tol'],
                       align_tol=cfg['align_tol'], gap_tol=cfg['gap_tol']),
        scan=rows,
        selection=dict(rule='smallest completion bound among the gains certified at '
                            'the design age bound (ties: larger age boundary)',
                       lam=sel['lam']),
        selected=OrderedDict(
            lam=sel['lam'], plan=plan_record(D), T_xi=float(D['T_xi']),
            a_cr=str(D['a_cr']), a_r0=str(D['pieces'][0][1]), v_hand=str(D['v_hand']),
            converged=D['converged'], hist=D['hist'],
            certificate_20ms=OrderedDict(
                admitted=bool(r20['admitted']), failed=r20['failed'],
                conds=r20['conds'], bounds=r20['bounds'],
                consts={k: v for k, v in r20['consts'].items() if k != 'pairs'},
                pairs=r20['consts']['pairs']),
            age_boundary=B, cond_boundaries=CB, monotone=mono,
            sequential_gaps=seq),
        regression=reg,
        runtime_s=time.time() - t0)
    with open(OUT, 'w') as fh:
        json.dump(out, fh, indent=1, default=str)
    print('wrote %s (%.0f s)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main(sys.argv[1:])
