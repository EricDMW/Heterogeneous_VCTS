#!/usr/bin/env python3
"""run_b2_ext.py -- extended benchmark program (root revision, plan of
2026-08-31: items E2-E11, E13, T6 evidence).

All parts merge into data/b2_results_ext.json; nothing here rewrites
b2_results.json / b2_results_fg.json (their digests stay pinned).

  sweepB     Case-B latency sweep: certificate + constant-maximum-age
             simulation + worst of N random FIFO schedules per point
  sweepA     the same sweep on Case A (theorem RULE, applied filter)
  act        Case-B activation / fallback runs (admission violated)
  scale      fleet-size sweep n = 3..10 at the frozen band geometry
  msg        message-period study (delta = T_m + delta_net) and the
             detector-period trade (T_c with eps_det raised), both cases
  tight      tight-hold runs: T_xi = ceil(T_hold) per state, nominal
             + the 300 ensemble states (bound non-vacuity)
  runtime    wall-clock cost of one forward pass, the full ceiling
             bisection, and the interval witness
  plant      out-of-model plant sensitivity table (brake build-up,
             homogeneous / heterogeneous, noise, grade, resistance)
  vthr       standstill-detection threshold rows (ideal model)
  cacc       same-information CACC-style baseline (time headway)
  caseA      Case A under the theorem RULE with applied filter vs the
             archived capability-clipped RULE (E13)
  resid      first-order dispersion law residual R_n, both cases (T6)
  review     pre-submission review follow-ups (2026-09-06): own-S1
             settle durations, message period T_m > T_c under the
             theorem model with channel invariants, filter-off PID
  plant2     every plant row with per-seed joint-contract pass counts,
             plus the compensated rows of review item E5 (grade and
             resistance feedforward, ramp lead)
  gaptest    exhaustive n=2/3 gap-construction tests (review T2/T3)
  mpc        cooperative arrival MPC baseline with a frozen tuning
             protocol, and the feedforward-PID retuned with the same
             budget (review E7)

Usage: python run_b2_ext.py sweepB sweepA ...   ('quick' = the fast set)
"""
import json
import os
import sys
import time

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import run_b2 as rb                     # constants + helpers (no main)
import b2_design as bd
import sim_root2 as sr2
import sim_hifi as sh

OUT = os.path.join(CODE, 'data', 'b2_results_ext.json')
TK = json.load(open(os.path.join(CODE, 'data', 'takeover_v42.json')))
A_DS = np.array(TK['d_s'], float)
A_E0 = np.array(TK['e0'], float)
A_V0 = list(TK['v'])
A_C0 = tuple(float(x) for x in (A_DS[1:] + A_E0[1:]))
A_SIM = dict(sr2.FEATURED)              # Case-A simulator constants
A_REQ = dict(sync=1.0, align=3.0, gap=1.0)
B_REQ = dict(sync=rb.D_REQ, align=rb.ALIGN_TOL, gap=1.0)


PARTDIR = os.path.join(CODE, 'data', 'b2_ext_parts')


def _merge(part, obj):
    """Each part writes its own file (parts run in parallel); 'merge'
    assembles data/b2_results_ext.json from them."""
    os.makedirs(PARTDIR, exist_ok=True)
    p = os.path.join(PARTDIR, part + '.json')
    json.dump(rb._py(obj), open(p, 'w'), indent=1)
    print('[%s] -> %s' % (part, os.path.relpath(p, CODE)))


def part_merge():
    data = {}
    for f in sorted(os.listdir(PARTDIR)):
        if f.endswith('.json') and not f.endswith('_ckpt.json'):
            data[f[:-5]] = json.load(open(os.path.join(PARTDIR, f)))
    data['_design'] = json.load(open(rb.OUT))['_design']
    json.dump(data, open(OUT, 'w'), indent=1)
    print('[merge] %s <- %s' % (os.path.basename(OUT), sorted(data)))


def _case(case):
    """(namespace factory, d_s, c0, v0, e0, sim params, tolerances)."""
    if case == 'B':
        return (lambda ov=None: rb._ns(ov), list(rb.D_S), rb.C0,
                list(rb.V0), list(rb.E0), dict(rb.SIM_P), B_REQ)
    return (lambda ov=None: bd._namespace(ov or {}), list(A_DS),
            A_C0, A_V0, list(A_E0), A_SIM, A_REQ)


def _cert(case, dbar, ov=None, d_s=None, c0=None):
    ns, ds, cc, v0, e0, P, req = _case(case)
    G = ns(ov)
    A = bd.assemble(G, np.array(ds if d_s is None else d_s, float),
                    cc if c0 is None else c0, dbar)
    out = dict(closed=bool(A['closed']))
    if not A['closed']:
        return out, A, None
    E = bd.entry_certify(G, A['R'], dbar)
    conds = dict(adm=bool(min(A['margins']) > 0),
                 clr=bool(A['gmin'] > 0), auth=A['auth_ok'],
                 event=A['event_ok'], ramp=bool(A['ramp_ok']),
                 vel=bool(A['vfloor'] > 0),
                 sync=bool(A['disp'] <= req['sync']),
                 align=bool(A['align'] <= req['align']),
                 gap=bool(np.max(A['gaperr']) <= req['gap']),
                 hold=E['hold_ok'], hnd=E['hnd_ok'])
    out.update(conds=conds, ok=all(conds.values()),
               failed=[k for k, v in conds.items() if not v],
               disp=A['disp'], align=A['align'],
               gaperr=float(np.max(A['gaperr'])),
               margin=float(min(A['margins'])),
               margins=[float(x) for x in A['margins']],
               gmin=A['gmin'], auth_slack=float(min(A['auth_slack'])),
               taub=[float(x) for x in A['R']['taub']],
               T_hold=E['T_hold'], Tent_bar=E['Tent_bar'])
    return out, A, E


# ------------------------------------------------------------ sweeps
SWEEP_B = [0.005, 0.010, 0.015, 0.020, 0.024, 0.030, 0.040, 0.060,
           0.080, 0.100, 0.150, 0.200]
SWEEP_A = [0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080,
           0.100, 0.150, 0.200, 0.300]


def _sweep(case, grid, n_rand, seed):
    ns, ds, cc, v0, e0, P, req = _case(case)
    rng = np.random.default_rng(seed)
    N = int(P['T_END'] / P['dt'])
    rows = []
    t0 = time.time()
    for d in grid:
        c, A, E = _cert(case, d)
        Px = dict(P, dbar=d)
        S = sr2.run(Px, ds, v0, e0, rule='ab', apply_filter=True,
                    terminate_on_contact=True)
        cm = rb._kpis(S)
        rr = []
        dmax = int(round(d / P['dt']))
        for j in range(n_rand):
            Dl = rb._random_schedule(rng, N, dmax)
            Sj = sr2.run(Px, ds, v0, e0, delays=Dl, rule='ab',
                         apply_filter=True, terminate_on_contact=True)
            rr.append(dict(disp=Sj['disp'], align=Sj['align'],
                           gaperr=Sj['gaperr'],
                           gmin=float(np.min(Sj['gmin'])),
                           hits=Sj['hits'], infeas=Sj['infeas'],
                           contact=Sj.get('contact_time'),
                           stopped=Sj['all_stopped']))
        rand = dict(n=n_rand,
                    disp_max=max(r['disp'] for r in rr)
                    if all(r['disp'] is not None for r in rr) else None,
                    align_max=max(r['align'] for r in rr),
                    gaperr_max=max(r['gaperr'] for r in rr),
                    gmin_min=min(r['gmin'] for r in rr),
                    hits=sum(r['hits'] for r in rr),
                    infeas=sum(r['infeas'] for r in rr),
                    contacts=sum(1 for r in rr
                                 if r['contact'] is not None),
                    stopped=sum(1 for r in rr if r['stopped']))
        rows.append(dict(dbar=d, cert=c, constmax=cm, random=rand))
        print('  %s dbar=%.3f closed=%s ok=%s bound=%s | const disp=%s '
              'hits=%d contact=%s | rand worst disp=%s hits=%d (%.0f s)'
              % (case, d, c['closed'], c.get('ok'),
                 ('%.3f' % c['disp']) if c['closed'] else 'n/a',
                 ('%.3f' % cm['disp']) if cm['disp'] is not None
                 else 'n/a', cm['hits'], cm['contact_time'],
                 rand['disp_max'], rand['hits'], time.time() - t0),
              flush=True)
    # summary: first filter action, first bound violation, first contact
    def first(pred):
        for r in rows:
            if pred(r):
                return r['dbar']
        return None
    summ = dict(
        first_filter_action=first(lambda r: r['constmax']['hits'] > 0
                                  or r['random']['hits'] > 0),
        first_bound_violation=first(
            lambda r: r['cert']['closed'] and r['constmax']['disp']
            is not None and (r['constmax']['disp'] > r['cert']['disp']
                             or (r['random']['disp_max'] or 0)
                             > r['cert']['disp'])),
        first_contact=first(lambda r: r['constmax']['contact_time']
                            is not None or r['random']['contacts'] > 0),
        last_admitted=max([r['dbar'] for r in rows
                           if r['cert']['closed'] and r['cert']['ok']]
                          or [None]),
        first_unclosed=first(lambda r: not r['cert']['closed']))
    return dict(grid=grid, n_rand=n_rand, seed=seed, rows=rows,
                summary=summ)


def part_sweepB():
    _merge('sweepB', _sweep('B', SWEEP_B, 20, 301))


def part_sweepA():
    _merge('sweepA', _sweep('A', SWEEP_A, 10, 302))


# ------------------------------------------------- activation / fallback
def part_act():
    D = json.load(open(rb.OUT))
    d_min = D['cert']['d_min']
    out = {}
    for tag, deficit, closing in (('activation', 1.0, 0.5),
                                  ('fallback', 1.5, 0.8)):
        c0 = tuple(float(d_min[i] - deficit) for i in range(1, 5))
        eps0 = [rb.B2['eps0'][0]] + [-closing] * 4
        v0 = rb._v0_of(eps0, rb.B2['v_xi'])
        e0 = [rb.B2['e0'][0]] + [c0[i - 1] - rb.D_S[i]
                                 for i in range(1, 5)]
        G = rb._ns(dict(eps0=eps0))
        A = bd.assemble(G, np.array(rb.D_S, float), c0, rb.B2['dbar'])
        cert = dict(closed=bool(A['closed']),
                    margins=[float(x) for x in A['margins']]
                    if A['closed'] else None,
                    adm=bool(A['closed'] and min(A['margins']) > 0))
        S = sr2.run(rb.SIM_P, rb.D_S, v0, e0, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        k = rb._kpis(S)
        k.update(c0=list(c0), v0=v0, deficit=deficit, closing=closing,
                 cert=cert,
                 sep_above_contact=float(np.min(S['gmin']))
                 + rb.SIM_P['s_m'])
        out[tag] = k
        print('  %-10s adm=%s act_time=%s infeas=%d hits=%d sep=%.2f '
              'stopped=%s contact=%s'
              % (tag, cert['adm'], np.round(k['act_time'], 2),
                 k['infeas'], k['hits'], k['sep_above_contact'],
                 k['all_stopped'], k['contact_time']))
    _merge('act', out)


# ------------------------------------------------------------ fleet size
def part_scale():
    rows = []
    for n in range(3, 11):
        r = (1.21 / 1.60) ** (1.0 / (n - 1))
        am = [round(1.60 * r ** k, 3) for k in range(n)]
        eps0 = [-0.30] + [round(0.25 * (0.7 ** k), 3)
                          for k in range(n - 1)]
        e0 = [-0.5] + [37.0] * (n - 1)
        ds = [rb.HEAD_SLOT] + [5.0] * (n - 1)
        c0 = tuple(np.linspace(42.0, 49.0, n - 1).round(1))
        try:
            G = bd._namespace(dict(rb.B2, n=n, amax=am, eps0=eps0,
                                   e0=e0))
            A = bd.assemble(G, np.array(ds, float), c0, rb.B2['dbar'])
            if A['closed']:
                E = bd.entry_certify(G, A['R'], rb.B2['dbar'])
                conds = rb._condset(A, E)
                ok = all(conds.values())
                rows.append(dict(n=n, ok=bool(ok),
                                 failed=[k for k, v in conds.items()
                                         if not v],
                                 margin=float(min(A['margins'])),
                                 gmin=A['gmin'], disp=A['disp'],
                                 align=A['align'],
                                 auth_slack=float(min(A['auth_slack'])),
                                 T_hold=E['T_hold']))
            else:
                rows.append(dict(n=n, ok=False, failed=['ledger']))
        except Exception as ex:
            rows.append(dict(n=n, ok=False, failed=['error:%s' % ex]))
        print('  n=%d: %s' % (n, rows[-1]))
    nmax = max([r['n'] for r in rows if r['ok']] or [0])
    _merge('scale', dict(rows=rows, n_max=nmax))


# ------------------------------------------------------ message period
def part_msg():
    out = {}
    for case in ('B', 'A'):
        rows = []
        for Tm in (0.005, 0.010, 0.020, 0.050, 0.100):
            for dn in (0.005, 0.010):
                d = round(Tm + dn, 4)
                c, A, E = _cert(case, d)
                rows.append(dict(T_m=Tm, d_net=dn, dbar=d,
                                 closed=c['closed'], ok=c.get('ok'),
                                 failed=c.get('failed'),
                                 margin=c.get('margin'),
                                 disp=c.get('disp')))
                print('  %s T_m=%.3f d_net=%.3f dbar=%.3f ok=%s failed=%s'
                      % (case, Tm, dn, d, c.get('ok'), c.get('failed')))
        det = []
        ns, ds, cc, v0, e0, P, req = _case(case)
        d_op = P['dbar']
        for Tc, ed in ((0.001, 0.015), (0.002, 0.030), (0.005, 0.060),
                       (0.010, 0.120)):
            c, A, E = _cert(case, d_op, ov=dict(T_c=Tc, eps_det=ed))
            S = sr2.run(dict(P, dt=Tc, eps_det=ed), ds, v0, e0,
                        rule='ab', apply_filter=True,
                        terminate_on_contact=True)
            det.append(dict(T_c=Tc, eps_det=ed, dbar=d_op,
                            closed=c['closed'], ok=c.get('ok'),
                            failed=c.get('failed'),
                            margin=c.get('margin'),
                            taub_max=max(c['taub']) if c['closed']
                            else None,
                            disp=c.get('disp'), align=c.get('align'),
                            sim=dict(disp=S['disp'], align=S['align'],
                                     gaperr=S['gaperr'],
                                     gmin=float(np.min(S['gmin'])),
                                     hits=S['hits'],
                                     stopped=S['all_stopped'])))
            print('  %s T_c=%.3f eps_det=%.3f ok=%s failed=%s margin=%s '
                  'taub_max=%s sim disp=%.3f hits=%d'
                  % (case, Tc, ed, c.get('ok'), c.get('failed'),
                     c.get('margin'), det[-1]['taub_max'], S['disp'],
                     S['hits']))
        out[case] = dict(message=rows, detector=det)
    _merge('msg', out)


# -------------------------------------------------------- tight hold
def part_tight(n_states=300):
    D = json.load(open(rb.OUT))
    ens = D['ens']['rows']
    rows = []
    t0 = time.time()
    states = [dict(c0=rb.C0, eps0=rb.B2['eps0'], tag='nominal')] + \
        [dict(c0=tuple(r['c0']), eps0=r['eps0'], tag='ens%d' % j)
         for j, r in enumerate(ens[:n_states])]
    for st in states:
        G = rb._ns(dict(eps0=[round(float(x), 6) for x in st['eps0']]))
        A = bd.assemble(G, np.array(rb.D_S, float), st['c0'],
                        rb.B2['dbar'])
        E = bd.entry_certify(G, A['R'], rb.B2['dbar'])
        T_xi = float(np.ceil(E['T_hold'] * 100.0) / 100.0)
        Gt = rb._ns(dict(eps0=[round(float(x), 6) for x in st['eps0']],
                         T_xi=T_xi))
        At = bd.assemble(Gt, np.array(rb.D_S, float), st['c0'],
                         rb.B2['dbar'])
        Et = bd.entry_certify(Gt, At['R'], rb.B2['dbar'])
        conds = rb._condset(At, Et)
        v0 = rb._v0_of(st['eps0'], rb.B2['v_xi'])
        e0 = [rb.B2['e0'][0]] + [st['c0'][i - 1] - rb.D_S[i]
                                 for i in range(1, 5)]
        P = dict(rb.SIM_P, T_xi=T_xi, T_END=T_xi + 25.0)
        S = sr2.run(P, rb.D_S, v0, e0, rule='ab', apply_filter=True,
                    terminate_on_contact=True)
        rows.append(dict(tag=st['tag'], T_xi=T_xi, T_hold=Et['T_hold'],
                         ok=all(conds.values()),
                         failed=[k for k, v in conds.items() if not v],
                         disp_bound=At['disp'], align_bound=At['align'],
                         disp=S['disp'], align=S['align'],
                         gaperr=S['gaperr'],
                         gmin=float(np.min(S['gmin'])), hits=S['hits'],
                         stopped=S['all_stopped'],
                         eps_at_handoff=None))
        if len(rows) % 25 == 1:
            print('  tight %d/%d: T_xi=%.2f disp=%.4f align=%.3f '
                  'hits=%d (%.0f s)' % (len(rows), len(states), T_xi,
                                        S['disp'], S['align'],
                                        S['hits'], time.time() - t0),
                  flush=True)
    dd = [r['disp'] for r in rows if r['disp'] is not None]
    summ = dict(n=len(rows), admitted=sum(1 for r in rows if r['ok']),
                dormant=sum(1 for r in rows if r['hits'] == 0),
                disp_max=max(dd), disp_nominal=rows[0]['disp'],
                disp_median=float(np.median(dd)),
                align_max=max(r['align'] for r in rows),
                inside=sum(1 for r in rows
                           if r['disp'] <= r['disp_bound']
                           and r['align'] <= r['align_bound']),
                leading_term=4 * rb.B2['eps_v'] / rb.B2['a_b'])
    print('  tight summary:', summ)
    _merge('tight', dict(rows=rows, summary=summ))


# ------------------------------------------------------------ runtime
def part_runtime():
    import platform
    import tempfile
    import witness_root2 as wr2
    out = dict(python=platform.python_version(),
               machine=platform.machine(), system=platform.system())
    for case in ('B', 'A'):
        ns, ds, cc, v0, e0, P, req = _case(case)
        ts = []
        for _ in range(5):
            t0 = time.perf_counter()
            G = ns()
            A = bd.assemble(G, np.array(ds, float), cc, P['dbar'])
            E = bd.entry_certify(G, A['R'], P['dbar'])
            ts.append(time.perf_counter() - t0)
        # full ceiling bisection over the eleven conditions
        t0 = time.perf_counter()
        n_eval = 0
        memo = {}

        def cond(nm, d):
            nonlocal n_eval
            if d not in memo:
                n_eval += 1
                memo[d] = _cert(case, d)[0]
            c = memo[d]
            return c['closed'] and c['conds'][nm]
        ceils = {}
        for nm in ('sync', 'align', 'gap', 'adm', 'auth', 'clr',
                   'event', 'vel', 'ramp', 'hold', 'hnd'):
            lo, hi = 0.002, 0.45
            if not cond(nm, lo):
                ceils[nm] = 0.0; continue
            if cond(nm, hi):
                ceils[nm] = hi; continue
            while hi - lo > 2e-4:
                mid = round(0.5 * (lo + hi), 6)
                if cond(nm, mid):
                    lo = mid
                else:
                    hi = mid
            ceils[nm] = lo
        t_bis = time.perf_counter() - t0
        out[case] = dict(forward_pass_s=float(np.median(ts)),
                         forward_pass_all=ts, bisection_s=t_bis,
                         bisection_evals=n_eval,
                         ceils={k: round(v, 4) for k, v in ceils.items()})
        print('  %s forward pass %.2f s (median of 5); bisection %.1f s '
              '(%d evaluations)' % (case, np.median(ts), t_bis, n_eval))
    t0 = time.perf_counter()
    ov = {k: rb.B2[k] for k in ('v_xi', 'a_b', 'alpha', 'lam', 'beta',
                                'gamma', 's_m', 'T_xi', 'dbar', 'eps_e',
                                'eps0')}
    ov['e0'] = [round(float(x), 4) for x in rb.E0]
    with tempfile.TemporaryDirectory() as td:
        rep = wr2.run(overrides=ov, dbar_op=rb.B2['dbar'], d_s=rb.D_S,
                      sync_tol=rb.D_REQ, align_tol=rb.ALIGN_TOL,
                      workdir=td)
    out['witness_B_s'] = time.perf_counter() - t0
    out['witness_B_ok'] = bool(rep['ok'])
    print('  witness (Case B operating point): %.1f s ok=%s'
          % (out['witness_B_s'], rep['ok']))
    _merge('runtime', out)


# ------------------------------------------------------ plant sensitivity
PLANT_ROWS = [
    ('default', {}),
    ('ideal actuator', dict(tau_act=0.0)),
    ('lag 0.2', dict(tau_act=0.2)),
    ('lag 0.5', dict(tau_act=0.5)),
    ('lag 1.0', dict(tau_act=1.0)),
    ('het lag 0.3 +/-20%', dict(tau_act=[0.36, 0.24, 0.36, 0.24, 0.36])),
    ('het lag 0.5 +/-20%', dict(tau_act=[0.60, 0.40, 0.60, 0.40, 0.60])),
    ('het lag 1.0 +/-20%', dict(tau_act=[1.20, 0.80, 1.20, 0.80, 1.20])),
    ('lag spread 0.2-1.0', dict(tau_act=[0.2, 0.4, 0.6, 0.8, 1.0])),
    ('no sensing noise', dict(noise_gap=0.0, noise_vel=0.0, bias_gap=0.0,
                              quant_vel=1e-9)),
    ('no grade', dict(grade=0.0)),
    ('no resistance', dict(davis=(0.0, 0.0, 0.0))),
    ('no noise, grade, resist.', dict(noise_gap=0.0, noise_vel=0.0,
                                      bias_gap=0.0, quant_vel=1e-9,
                                      grade=0.0, davis=(0.0, 0.0, 0.0))),
]


def part_plant(n_seeds=20):
    rows = []
    t0 = time.time()
    for name, ov in PLANT_ROWS:
        rs = []
        for sd in range(1, n_seeds + 1):
            R = sh.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, hifi=ov, seed=sd)
            rs.append(dict(disp=R['disp'], align=R['align'],
                           gaperr=R['gaperr'], gmin=min(R['gmin']),
                           filt=R['filt_events'],
                           stopped=R['all_stopped'],
                           contact=R['contact_time'],
                           jerk=max(R['jerk_pk'])))
        dd = [r['disp'] for r in rs if r['disp'] is not None]
        row = dict(name=name, overrides=ov, n=n_seeds,
                   disp_med=float(np.median(dd)) if dd else None,
                   disp_max=max(dd) if dd else None,
                   align_max=max(r['align'] for r in rs),
                   gaperr_max=max(r['gaperr'] for r in rs),
                   gmin_min=min(r['gmin'] for r in rs),
                   filt_total=sum(r['filt'] for r in rs),
                   filt_runs=sum(1 for r in rs if r['filt'] > 0),
                   contacts=sum(1 for r in rs if r['contact'] is not None),
                   stopped=sum(1 for r in rs if r['stopped']))
        rows.append(row)
        print('  %-26s disp med/max %s/%s align %.2f gmin %.2f filt %d '
              'runs, contacts %d (%.0f s)'
              % (name, ('%.3f' % row['disp_med']) if dd else 'n/a',
                 ('%.3f' % row['disp_max']) if dd else 'n/a',
                 row['align_max'], row['gmin_min'], row['filt_runs'],
                 row['contacts'], time.time() - t0), flush=True)
    _merge('plant', dict(rows=rows, ideal_bound=json.load(
        open(rb.OUT))['cert']['disp']))


def part_vthr():
    D = json.load(open(rb.OUT))
    C = D['cert']
    a_min = min(C['anet'])
    rows = []
    for vt in (0.015, 0.05, 0.10, 0.20):
        S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                    apply_filter=True, terminate_on_contact=True,
                    v_thr=vt)
        adj = C['disp'] + 4 * vt / rb.B2['a_b']
        adj_gap = max(C['gaperr']) + ((max(C['epsb']) + vt) ** 2
                                      - max(C['epsb']) ** 2) / (2 * a_min)
        rows.append(dict(v_thr=vt, disp=S['disp'], align=S['align'],
                         gaperr=S['gaperr'], gmin=float(np.min(S['gmin'])),
                         hits=S['hits'], stopped=S['all_stopped'],
                         disp_bound_adj=adj, gaperr_bound_adj=adj_gap))
        print('  v_thr=%.3f disp=%.4f (adj bound %.3f) gaperr=%.3f '
              '(adj %.3f) hits=%d' % (vt, S['disp'], adj, S['gaperr'],
                                      adj_gap, S['hits']))
    _merge('vthr', rows)


# ------------------------------------------------------------ baselines
def part_cacc():
    out = {}
    for tag, kw in (('cacc', dict(variant='cacc')),
                    ('cacc_nofilter', dict(variant='cacc',
                                           apply_filter=False))):
        af = kw.pop('apply_filter', True)
        S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                    apply_filter=af, terminate_on_contact=True, **kw)
        out[tag] = rb._kpis(S)
        print('  %-14s disp=%s align=%.3f gaperr=%.3f gmin=%.2f upk=%.3f '
              'hits=%d infeas=%d contact=%s stopped=%s'
              % (tag, ('%.4f' % S['disp']) if S['disp'] is not None
                 else 'n/a', S['align'], S['gaperr'],
                 float(np.min(S['gmin'])), float(np.max(S['upk'])),
                 S['hits'], S['infeas'], S.get('contact_time'),
                 S['all_stopped']))
    eps0h = [2 * x for x in rb.B2['eps0']]
    v0h = rb._v0_of(eps0h, rb.B2['v_xi'])
    S = sr2.run(rb.SIM_P, rb.D_S, v0h, rb.E0, rule='ab',
                apply_filter=True, terminate_on_contact=True,
                variant='cacc')
    out['cacc_hard'] = rb._kpis(S)
    out['h_w'] = sr2.K_CACC['h_w']
    hw_rows = []
    for hw in (0.2, 0.5, 1.0):
        sr2.K_CACC['h_w'] = hw
        S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                    apply_filter=True, terminate_on_contact=True,
                    variant='cacc')
        k = rb._kpis(S); k['h_w'] = hw; hw_rows.append(k)
        print('  cacc h_w=%.1f disp=%s align=%.3f gaperr=%.3f gmin=%.2f '
              'act=%s' % (hw, S['disp'], S['align'], S['gaperr'],
                          float(np.min(S['gmin'])),
                          np.round(S['act_time'], 2)))
    sr2.K_CACC['h_w'] = out['h_w']
    out['h_w_sweep'] = hw_rows
    print('  cacc_hard disp=%s align=%.3f gmin=%.2f hits=%d'
          % (S['disp'], S['align'], float(np.min(S['gmin'])), S['hits']))
    _merge('cacc', out)


# ------------------------------------------------------ Case A, theorem RULE
def part_caseA():
    out = {}
    for tag, rule in (('rule_ab', 'ab'), ('rule_gapclose', 'gapclose')):
        S = sr2.run(A_SIM, A_DS, A_V0, A_E0, rule=rule, apply_filter=True,
                    terminate_on_contact=True)
        out[tag] = rb._kpis(S)
        print('  Case A %-13s disp=%.4f align=%.3f gaperr=%.3f gmin=%s '
              'upk=%s hits=%d' % (tag, S['disp'], S['align'], S['gaperr'],
                                  np.round(S['gmin'], 3),
                                  np.round(S['upk'], 3), S['hits']))
    a, b = out['rule_ab'], out['rule_gapclose']
    out['max_abs_diff'] = dict(
        disp=abs(a['disp'] - b['disp']), align=abs(a['align'] - b['align']),
        gaperr=abs(a['gaperr'] - b['gaperr']),
        gmin=float(np.max(np.abs(np.array(a['gmin']) - np.array(b['gmin'])))),
        upk=float(np.max(np.abs(np.array(a['upk']) - np.array(b['upk'])))),
        tstop=float(np.max(np.abs(np.array(a['tstop'])
                                  - np.array(b['tstop'])))))
    print('  max |difference|:', out['max_abs_diff'])
    _merge('caseA', out)


# ------------------------------------------------------ first-order law
def part_resid():
    out = {}
    for case in ('B', 'A'):
        ns, ds, cc, v0, e0, P, req = _case(case)
        G = ns()
        A = bd.assemble(G, np.array(ds, float), cc, P['dbar'])
        R = A['R']; n = G['n']
        a_b, eps_v, eps_e, d = G['a_b'], G['eps_v'], G['eps_e'], P['dbar']
        lam = G['lam']
        # arrays are unit-indexed (0 = head); pair k = (k-1, k), k = 2..n
        epsb = np.array(R['epsb'][1:n]); Fb = np.array(R['Fb'])
        anet = a_b - Fb[1:n]                  # headroom of pair k = a_b - F^b_{k-1}
        W = np.array(R['W'][1:n])
        q = epsb / anet
        lead = (n - 1) * (eps_v / a_b + d)
        Rn = float(np.sum(q) - lead)
        parts = dict(
            position_tol=float(np.sum(lam * eps_e / np.e / anet)),
            ledger_excess=float(np.sum((W - d * a_b) / anet)),
            headroom_loss=float(np.sum((eps_v + d * a_b)
                                       * (1 / anet - 1 / a_b))))
        # residual at the evaluated synchrony ceiling (the residual
        # grows with the age bound, so the leading-term ceiling
        # sync_ub is an upper bound on the true ceiling)
        dsync = {'B': 0.160, 'A': 0.151}[case]
        As = bd.assemble(ns(), np.array(ds, float), cc, dsync)
        Rs = As['R']
        qs = np.array(Rs['epsb'][1:n]) / (a_b - np.array(Rs['Fb'][1:n]))
        Rn_s = float(np.sum(qs) - (n - 1) * (eps_v / a_b + dsync))
        out[case] = dict(disp=A['disp'], q=[float(x) for x in q],
                         leading=lead, R_n=Rn, R_n_frac=Rn / A['disp'],
                         parts=parts, q_min_check=bool(
                             np.all(q >= eps_v / a_b + d - 1e-12)),
                         sync_ub=float(req['sync'] / (n - 1)
                                       - eps_v / a_b),
                         dsync=dsync, disp_at_dsync=As['disp'],
                         R_n_at_dsync=Rn_s,
                         R_n_frac_at_dsync=Rn_s / As['disp'])
        print('  %s: disp=%.4f leading=%.4f R_n=%.4f (%.1f%%) parts=%s '
              'sync_ub=%.3f; at dsync=%.3f: disp=%.4f R_n=%.4f (%.1f%%)'
              % (case, A['disp'], lead, Rn, 100 * Rn / A['disp'],
                 {k: round(v, 4) for k, v in parts.items()},
                 out[case]['sync_ub'], dsync, As['disp'], Rn_s,
                 100 * Rn_s / As['disp']))
    _merge('resid', out)


# ------------------------------------ pre-submission review follow-ups
def _periodic_schedule(N, m, dn):
    """Per-send delay steps realizing a sender message period of m
    steps with a transport-plus-processing delay of dn steps: every
    send is delivered in the same FIFO batch as the next periodic
    send (index a multiple of m), so the receiver holds the periodic
    sample (zero-order hold) and every intermediate packet is
    superseded in sending order.  Arrival indices are nondecreasing
    by construction and the realized age is at most m + dn - 1 steps
    (strictly below T_m + delta_net)."""
    d = np.empty((5, N), int)
    j = np.arange(N)
    jp = ((j + m - 1) // m) * m
    d[1:] = jp + dn - j
    d[0] = 1
    return d


def _channel_invariants(delays_row, N):
    """Receive-pointer monotonicity and the realized held age of one
    hop under the simulator's FIFO arrival pass."""
    j = np.arange(N)
    arr = np.maximum.accumulate(j + delays_row[:N])
    ks = np.arange(N)
    idx = np.searchsorted(arr, ks, side='right') - 1
    ok = idx >= 0
    age = ks[ok] - idx[ok]
    return dict(pointer_nondecreasing=bool(np.all(np.diff(idx) >= 0)),
                max_age_steps=int(np.max(age)),
                timestamps_transmitted=bool(np.all(idx[ok] <= ks[ok])))


def part_review():
    """(a) settle windows like for like (own S1 duration
    t_set - t_act next to the consecutive-settle difference);
    (b) theorem-model runs with a message period T_m strictly above
    the detector period T_c = 1 ms (zero-order hold of periodic
    samples, FIFO supersession, transport delay), both cases, with
    the channel invariants asserted; (c) the paired filter-off run
    of the tuned feedforward-PID baseline."""
    out = {}
    t0 = time.time()
    S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                apply_filter=True, terminate_on_contact=True)
    own = S['tset'] - S['tact']
    out['settle'] = dict(tact=[float(x) for x in S['tact']],
                         tset=[float(x) for x in S['tset']],
                         own=[float(x) for x in own],
                         diffs=[float(S['tset'][0])] +
                         [float(x) for x in np.diff(S['tset'])],
                         dbar=rb.SIM_P['dbar'],
                         egap_signed=[float(x) for x in S['egap']],
                         marker_signed=[float(x) for x in
                                        (S['s'] - S['marks'])],
                         tstop=[float(x) for x in S['tstop']])
    print('  settle own S1 %s diffs %s' % (np.round(own, 3),
                                          np.round(out['settle']['diffs'],
                                                   3)))
    tm = {}
    for case, combos in (('B', ((0.005, 0.015), (0.010, 0.010))),
                         ('A', ((0.020, 0.040), (0.050, 0.010)))):
        ns, ds, cc, v0, e0, P, req = _case(case)
        N = int(P['T_END'] / P['dt'])
        rows = []
        for Tm, dn in combos:
            d = round(Tm + dn, 4)
            c, A, E = _cert(case, d)
            m = int(round(Tm / P['dt'])); dns = int(round(dn / P['dt']))
            Dl = _periodic_schedule(N, m, dns)
            inv = _channel_invariants(Dl[1], N)
            assert inv['pointer_nondecreasing'] and \
                inv['timestamps_transmitted']
            assert inv['max_age_steps'] * P['dt'] <= d + 1e-12
            Sx = sr2.run(dict(P, dbar=d), ds, v0, e0, delays=Dl,
                         rule='ab', apply_filter=True,
                         terminate_on_contact=True)
            k = rb._kpis(Sx)
            inside = bool(c['closed'] and k['disp'] is not None
                          and k['disp'] <= c['disp'] + 1e-9
                          and k['align'] <= c['align'] + 1e-9
                          and k['gaperr'] <= c['gaperr'] + 1e-9)
            rows.append(dict(T_m=Tm, d_net=dn, dbar=d, T_c=P['dt'],
                             cert=c, sim=k, inside=inside,
                             max_age=inv['max_age_steps'] * P['dt'],
                             channel=inv))
            print('  %s T_m=%.3f d_net=%.3f dbar=%.3f cert ok=%s failed=%s'
                  ' | sim disp=%s align=%.3f gaperr=%.3f hits=%d '
                  'max age=%.3f inside=%s (%.0f s)'
                  % (case, Tm, dn, d, c.get('ok'), c.get('failed'),
                     ('%.3f' % k['disp']) if k['disp'] is not None
                     else 'n/a', k['align'], k['gaperr'], k['hits'],
                     rows[-1]['max_age'], inside, time.time() - t0),
                  flush=True)
        tm[case] = rows
    out['tm'] = tm
    Sp = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                 apply_filter=False, variant='pid',
                 terminate_on_contact=True)
    out['pid_nofilter'] = rb._kpis(Sp)
    print('  pid, filter off: contact=%s gmin=%s disp=%s align=%.2f'
          % (Sp.get('contact_time'), np.round(Sp['gmin'], 2),
             Sp['disp'], Sp['align']))
    _merge('review', out)


# --------------------------------- review follow-ups of 2026-09-07 (E5/T2/E7)
REQ_B = dict(disp=rb.D_REQ, align=rb.ALIGN_TOL, gap=1.0)

PLANT2_ROWS = [
    ('grade feedforward', dict(grade_ff=1.0)),
    ('grade+resistance ff, lead 0.6', dict(grade_ff=1.0, res_ff=1.0,
                                           ramp_lead=0.6)),
    ('grade+resistance ff, lead 0.75', dict(grade_ff=1.0, res_ff=1.0,
                                            ramp_lead=0.75)),
    ('80% ff, lead 0.6', dict(grade_ff=0.8, res_ff=0.8, ramp_lead=0.6)),
]


def _joint(rec):
    return bool(rec['stopped'] and rec['contact'] is None
                and rec['disp'] is not None
                and rec['disp'] <= REQ_B['disp']
                and rec['align'] <= REQ_B['align']
                and rec['gaperr'] <= REQ_B['gap'])


def _ckpt(name):
    """Per-part checkpoint (the long parts are resumable after an OS
    kill): a JSON dict persisted in PARTDIR."""
    os.makedirs(PARTDIR, exist_ok=True)
    p = os.path.join(PARTDIR, name + '_ckpt.json')
    data = json.load(open(p)) if os.path.exists(p) else {}

    def save():
        json.dump(rb._py(data), open(p, 'w'))
    return data, save


def part_plant2(n_seeds=20):
    """Every plant row (the original thirteen plus the compensated
    rows of review item E5) with per-seed records and the joint
    service-contract pass count; the original rows must reproduce the
    'plant' part's maxima (same seeds, same model)."""
    rows = []
    t0 = time.time()
    ck, save = _ckpt('plant2')
    for name, ov in PLANT_ROWS + PLANT2_ROWS:
        if name in ck:
            rows.append(ck[name])
            print('  %-32s (checkpoint)' % name, flush=True)
            continue
        rs = []
        for sd in range(1, n_seeds + 1):
            R = sh.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, hifi=ov, seed=sd)
            rec = dict(seed=sd, disp=R['disp'], align=R['align'],
                       gaperr=R['gaperr'], gmin=min(R['gmin']),
                       filt=R['filt_events'], stopped=R['all_stopped'],
                       contact=R['contact_time'])
            rec['joint'] = _joint(rec)
            rs.append(rec)
        dd = [r['disp'] for r in rs if r['disp'] is not None]
        row = dict(name=name, overrides=ov, n=n_seeds,
                   disp_med=float(np.median(dd)) if dd else None,
                   disp_max=max(dd) if dd else None,
                   align_max=max(r['align'] for r in rs),
                   gaperr_max=max(r['gaperr'] for r in rs),
                   gmin_min=min(r['gmin'] for r in rs),
                   filt_runs=sum(1 for r in rs if r['filt'] > 0),
                   contacts=sum(1 for r in rs if r['contact'] is not None),
                   stopped=sum(1 for r in rs if r['stopped']),
                   pass_disp=sum(1 for r in rs if r['disp'] is not None
                                 and r['disp'] <= REQ_B['disp']),
                   pass_align=sum(1 for r in rs
                                  if r['align'] <= REQ_B['align']),
                   pass_gap=sum(1 for r in rs
                                if r['gaperr'] <= REQ_B['gap']),
                   joint_pass=sum(1 for r in rs if r['joint']),
                   seeds=rs)
        rows.append(row)
        ck[name] = row; save()
        print('  %-32s joint %2d/%d  disp max %.3f align %.2f gap %.2f '
              'gmin %.2f (%.0f s)' % (name, row['joint_pass'], n_seeds,
                                      row['disp_max'] or -1,
                                      row['align_max'], row['gaperr_max'],
                                      row['gmin_min'], time.time() - t0),
              flush=True)
    _merge('plant2', dict(rows=rows, thresholds=REQ_B))


def part_gaptest():
    sys.path.insert(0, os.path.join(CODE, 'src', 'validation'))
    import gap_construction_tests as gct
    _merge('gaptest', gct.main())


def part_delnage(n_chunks=4, n_trials=250, seed_base=100):
    """Readiness review (2026-09-08), item E1: the held information
    age of the 1000 random FIFO delay schedules of part 'deln'
    (run_b2.py; chunks seeded seed_base + chunk) is re-derived from
    the schedules alone -- no simulation -- and checked on EVERY
    step of every hold interval, i.e. up to the instant before the
    next delivery batch, not only at delivery.  Age of the held
    packet j at step k is (k - j) T_m with a_j <= k < a_{j+1};
    since the sender's own command is held over the same period the
    received signal is the sender's signal shifted by exactly this
    age, so the bound is max_j d_j T_m <= dbar."""
    N = int(rb.SIM_P['T_END'] / rb.SIM_P['dt'])
    dmax = int(round(rb.B2['dbar'] / rb.SIM_P['dt']))
    t0 = time.time()
    max_steps = 0; n = 0; ptr_ok = True; ts_ok = True; per_hop = []
    for chunk in range(n_chunks):
        rng = np.random.default_rng(seed_base + chunk)
        for _ in range(n_trials):
            Dl = rb._random_schedule(rng, N, dmax)
            for i in range(1, 5):
                inv = _channel_invariants(Dl[i], N)
                ptr_ok &= inv['pointer_nondecreasing']
                ts_ok &= inv['timestamps_transmitted']
                max_steps = max(max_steps, inv['max_age_steps'])
                per_hop.append(inv['max_age_steps'])
                assert inv['max_age_steps'] <= int(Dl[i].max())
            n += 1
        print('  delnage chunk %d done, max age %d steps (%.0f s)'
              % (chunk, max_steps, time.time() - t0), flush=True)
    out = dict(n_schedules=n, dmax_steps=dmax, T_m=rb.SIM_P['dt'],
               dbar=rb.B2['dbar'], max_age_steps=int(max_steps),
               max_age_s=float(max_steps * rb.SIM_P['dt']),
               all_within_bound=bool(max_steps <= dmax),
               pointer_nondecreasing_all=bool(ptr_ok),
               timestamps_transmitted_all=bool(ts_ok),
               hops_at_bound=int(sum(1 for m in per_hop if m == dmax)),
               hops=len(per_hop))
    print('  delnage: %d schedules, max held age %.3f s (bound %.3f),'
          ' within=%s' % (n, out['max_age_s'], out['dbar'],
                          out['all_within_bound']))
    _merge('delnage', out)


MPC_GRID = [dict(w_e=we, w_v=wv, w_du=wd)
            for we in (0.1, 0.3, 1.0, 3.0) for wv in (0.1, 0.3, 1.0)
            for wd in (0.3, 1.0, 3.0, 10.0)]
PID_GRID = [dict(Kp=kp, Kd=kd, Ki=ki)
            for kp in (0.001, 0.003, 0.01) for kd in (0.05, 0.115, 0.30)
            for ki in (1e-6, 5e-5, 2e-4)]     # 1e-6: integral off


def _score(k):
    """Frozen tuning objective (declared before the search): a run
    with contact or an incomplete stop is rejected; otherwise rank by
    the number of violated contract indices, then by the normalized
    KPI sum disp/0.9 + align/3 + gap/1."""
    if k.get('contact_time') is not None or not k['all_stopped'] \
            or k['disp'] is None:
        return [1e9, 1e9]
    viol = int(k['disp'] > REQ_B['disp']) + int(k['align'] > REQ_B['align']) \
        + int(k['gaperr'] > REQ_B['gap'])
    return [viol, k['disp'] / REQ_B['disp'] + k['align'] / REQ_B['align']
            + k['gaperr'] / REQ_B['gap']]


def _slim(k):
    return {kk: v for kk, v in k.items() if kk not in ('tset',)}


def part_mpc():
    """Cooperative arrival MPC (native-architecture track) with a frozen
    tuning protocol, and the feedforward-PID retuned with the same
    budget on the same training scenario.  Training: the doubled-
    mismatch takeover (ideal model) and the default perturbed plant,
    seed 1.  Held-out: the nominal benchmark (ideal model) and the
    perturbed plant, seeds 2-21."""
    import sim_mpc as sm
    eps0h = [2 * x for x in rb.B2['eps0']]
    v0h = rb._v0_of(eps0h, rb.B2['v_xi'])
    t0 = time.time()
    ck, save = _ckpt('mpc')
    cands = []
    for W in MPC_GRID:
        key = 'cand:' + json.dumps(W, sort_keys=True)
        if key in ck:
            cands.append(ck[key]); continue
        a = sm.run(rb.SIM_P, rb.D_S, v0h, rb.E0, W=W, plant='ideal')
        h1 = sm.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, W=W, plant='hifi',
                    seed=1)
        sa, sh1 = _score(a), _score(h1)
        cands.append(dict(W=W, score=[sa[0] + sh1[0], sa[1] + sh1[1]],
                          train_ideal=_slim(a), train_hifi=_slim(h1)))
        ck[key] = cands[-1]; save()
        print('  mpc %s -> score %s | ideal disp=%s align=%.2f gap=%.2f '
              'hits=%d | hifi disp=%s align=%.2f gap=%.2f contact=%s '
              '(%.0f s)' % (W, cands[-1]['score'], a['disp'], a['align'],
                            a['gaperr'], a['hits'], h1['disp'], h1['align'],
                            h1['gaperr'], h1['contact_time'],
                            time.time() - t0), flush=True)
    best = min(cands, key=lambda c: tuple(c['score']))
    Wb = best['W']
    hk = 'heldout:' + json.dumps(Wb, sort_keys=True)
    if hk not in ck:
        ck[hk] = dict(ideal=None, hifi={})
    if ck[hk]['ideal'] is None:
        ck[hk]['ideal'] = _slim(sm.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0,
                                       W=Wb, plant='ideal')); save()
    ideal = ck[hk]['ideal']
    hifi = []
    for sd in range(2, 22):
        if str(sd) not in ck[hk]['hifi']:
            ck[hk]['hifi'][str(sd)] = _slim(sm.run(
                rb.SIM_P, rb.D_S, rb.V0, rb.E0, W=Wb, plant='hifi',
                seed=sd)); save()
        hifi.append(ck[hk]['hifi'][str(sd)])
    hj = sum(1 for h in hifi if _joint(dict(
        stopped=h['all_stopped'], contact=h['contact_time'], disp=h['disp'],
        align=h['align'], gaperr=h['gaperr'])))
    print('  mpc best %s | held-out ideal disp=%s align=%.2f gap=%.2f hits=%d'
          ' | hifi joint %d/20 (%.0f s)' % (Wb, ideal['disp'], ideal['align'],
                                            ideal['gaperr'], ideal['hits'],
                                            hj, time.time() - t0),
          flush=True)
    # feedforward-PID retuned with the same budget and objective
    K0 = dict(sr2.K_PID)
    pc = []
    for K in PID_GRID:
        key = 'pid:' + json.dumps(K, sort_keys=True)
        if key in ck:
            pc.append(ck[key]); continue
        sr2.K_PID.update(dict(K0, **K))
        S = sr2.run(rb.SIM_P, rb.D_S, v0h, rb.E0, rule='ab',
                    apply_filter=True, terminate_on_contact=True,
                    variant='pid')
        k = rb._kpis(S)
        pc.append(dict(K=K, score=_score(k), train=k))
        ck[key] = pc[-1]; save()
    pbest = min(pc, key=lambda c: tuple(c['score']))
    sr2.K_PID.update(dict(K0, **pbest['K']))
    S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                apply_filter=True, terminate_on_contact=True,
                variant='pid')
    pid_ideal = rb._kpis(S)
    sr2.K_PID.clear(); sr2.K_PID.update(K0)
    print('  pid best %s | held-out ideal disp=%s align=%.2f gap=%.2f '
          'act=%s (%.0f s)' % (pbest['K'], pid_ideal['disp'],
                               pid_ideal['align'], pid_ideal['gaperr'],
                               np.round(pid_ideal['act_time'], 1),
                               time.time() - t0), flush=True)
    _merge('mpc', dict(
        protocol=dict(T_mpc=sm.T_MPC, N=sm.N_H, sig=sm.SIG,
                      grid_size=len(MPC_GRID),
                      training=['ideal, doubled-mismatch takeover',
                                'perturbed plant default, seed 1'],
                      heldout=['ideal nominal benchmark',
                               'perturbed plant default, seeds 2-21'],
                      objective=_score.__doc__),
        grid=MPC_GRID, candidates=cands, best_W=Wb,
        heldout_ideal=_slim(ideal), heldout_hifi=[_slim(h) for h in hifi],
        hifi_joint_pass=hj,
        pid=dict(grid=PID_GRID, candidates=pc, best_K=pbest['K'],
                 original_K=K0, heldout_ideal=pid_ideal),
        req=REQ_B))


# ====================== audit revision of 2026-09-08 (E3, E4, Fig. 1 margin)
def _hop_terms(A, G, dbar):
    """Per-hop terms of the certified dispersion bound (pair k=2..n):
    q_k = epsb_k / a_k with epsb_k = eps_v + lam eps_e/e + W_k."""
    R = A['R']; n = G['n']
    epsb = np.array(R['epsb'][1:n]); Fb = np.array(R['Fb'])
    anet = G['a_b'] - Fb[1:n]
    W = np.array(R['W'][1:n])
    return dict(epsb=epsb, anet=anet, W=W, q=epsb / anet,
                Fb_prev=Fb[1:n], eb=np.array(R['eb'][1:n])
                if 'eb' in R else None)


def part_margin():
    """Benchmark run with the filter inactivity margin recorded (the
    margin panel of the main-text figure) and the E3 diagnostics."""
    S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, rule='ab',
                apply_filter=True, record=True, terminate_on_contact=True,
                extra=True)
    assert S['hits'] == 0 and S['infeas'] == 0 and S['all_stopped']
    assert min(S['mmin']) > 0
    out = dict(mmin=[float(x) for x in S['mmin']],
               m=[[[float(a), float(b)] for a, b in x] for x in S['m']],
               disp=S['disp'], align=S['align'], gaperr=S['gaperr'])
    print('  margin minima [m/s^2]:', np.round(S['mmin'], 3))
    _merge('margin', out)


def part_decomp():
    """E3: certified per-hop dispersion terms against their realized
    counterparts on the benchmark (constant maximum age) and on the
    adversarial schedule of part 'adv'."""
    ns, ds, cc, v0, e0, P, req = _case('B')
    G = ns(); n = G['n']
    A = bd.assemble(G, np.array(ds, float), cc, P['dbar'])
    H = _hop_terms(A, G, P['dbar'])
    lam, eps_e, eps_v, a_b = G['lam'], G['eps_e'], G['eps_v'], G['a_b']
    runs = {}
    adv = json.load(open(rb.OUT))['adv']
    N = int(P['T_END'] / P['dt'])

    def sched(p):
        d = np.empty((5, N), int); d[0] = 1
        for i in range(1, 5):
            e_, l_, ts = p[i - 1]
            kk = int(ts / 1e-3)
            d[i, :kk] = e_; d[i, kk:] = l_
        return d
    for tag, dl in (('constmax', None), ('adversarial',
                                          sched(adv['schedule']))):
        S = sr2.run(P, ds, v0, e0, delays=dl, rule='ab', apply_filter=True,
                    terminate_on_contact=True, extra=True)
        ts = np.array(S['tstop'])
        hop = np.abs(np.diff(ts))                     # |tau_k - tau_{k-1}|
        runs[tag] = dict(
            disp=S['disp'], hop_disp=[float(x) for x in hop],
            e_hnd=[float(abs(x)) for x in S['e_hnd'][1:]],
            eps_hnd=[float(abs(x)) for x in S['eps_hnd'][1:]],
            wint=[float(x) for x in S['wint']],
            epsmax_b=[float(x) for x in S['epsmax_b']],
            fb_real=[float(x) for x in S['fb_real']],
            egap=[float(abs(x)) for x in S['egap'][1:]],
            mmin=[float(x) for x in S['mmin']])
        assert all(hop <= H['q'] + 1e-9), (tag, hop, H['q'])
        assert all(np.array(runs[tag]['wint']) <= H['W'] + 1e-9)
        assert all(np.array(runs[tag]['epsmax_b']) <= H['epsb'] + 1e-9)
        assert all(np.array(runs[tag]['fb_real']) <= H['Fb_prev'] + 1e-9)
    gap_bound = A['gaperr'][1:n] if len(A['gaperr']) == n else A['gaperr']
    out = dict(
        bound=dict(q=[float(x) for x in H['q']],
                   epsb=[float(x) for x in H['epsb']],
                   W=[float(x) for x in H['W']],
                   anet=[float(x) for x in H['anet']],
                   Fb_prev=[float(x) for x in H['Fb_prev']],
                   vel_tol=[eps_v / a for a in H['anet']],
                   pos_tol=[lam * eps_e / np.e / a for a in H['anet']],
                   ledger=[float(w / a) for w, a in zip(H['W'], H['anet'])],
                   gaperr=[float(x) for x in gap_bound],
                   eps_e=eps_e, eps_v=eps_v, disp=A['disp']),
        runs=runs,
        note='realized counterparts of every term of the certified '
             'dispersion bound; ratios use absolute values, no term is '
             'divided by a realized quantity')
    for tag in runs:
        print('  %-11s hop |dtau| %s vs q %s' % (
            tag, np.round(runs[tag]['hop_disp'], 4), np.round(H['q'], 4)))
        print('  %-11s int|w| %s vs W %s' % (
            tag, np.round(runs[tag]['wint'], 4), np.round(H['W'], 4)))
        print('  %-11s |e(T)| %s (eps_e %.2f) |eps(T)| %s (eps_v %.3f)' % (
            tag, np.round(runs[tag]['e_hnd'], 4), eps_e,
            np.round(runs[tag]['eps_hnd'], 5), eps_v))
    _merge('decomp', out)


def part_reach(restarts=4, iters=40, seed=41):
    """E4: handoff-box corner reachability. Random-restart hill climb
    over admitted takeover states (the ensemble box of part 'ens') and
    piecewise-constant FIFO delay schedules, maximizing the largest
    fraction of the handoff box reached at T_xi,
    rho = max_i max{|e_i(T_xi)|/eps_e, |eps_i(T_xi)|/eps_v}.
    A candidate is scored only if the full dispatch test admits it.
    Checkpointed per evaluation (data/b2_ext_parts/reach_ckpt.json)."""
    rng = np.random.default_rng(seed)
    P = dict(rb.SIM_P); N = int(P['T_END'] / P['dt'])
    dmax = int(round(rb.B2['dbar'] / P['dt']))
    eps_e, eps_v = rb.B2['eps_e'], rb.B2['eps_v']
    ck, save = _ckpt('reach')
    evals = ck.setdefault('evals', [])

    def sched(p):
        d = np.empty((5, N), int); d[0] = 1
        for i in range(1, 5):
            e_, l_, ts = p[i - 1]
            kk = int(ts / 1e-3)
            d[i, :kk] = e_; d[i, kk:] = l_
        return d

    def draw_state():
        c0 = [float(x) for x in rng.uniform(40.0, 50.0, 4).round(3)]
        eps0 = [float(rng.uniform(-0.35, 0.0))] + \
            [float(rng.uniform(0.0, 0.30) * f)
             for f in (1.0, 0.8, 0.6, 0.4)]
        return c0, [round(x, 4) for x in eps0]

    def draw_sched():
        return [(int(rng.integers(1, dmax + 1)), int(rng.integers(1, dmax + 1)),
                 float(rng.uniform(rb.B2['T_xi'] - 20, rb.B2['T_xi'])))
                for _ in range(4)]

    def score(c0, eps0, p):
        key = json.dumps([c0, eps0, p])
        for ev in evals:
            if ev['key'] == key:
                return ev
        Gx = bd._namespace(dict(rb.B2, eps0=eps0))
        Ax = bd.assemble(Gx, np.array(rb.D_S, float), tuple(c0),
                         rb.B2['dbar'])
        adm = bool(Ax['closed'] and min(Ax['margins']) > 0 and Ax['auth_ok']
                   and Ax['event_ok'] and Ax['gmin'] > 0)
        ev = dict(key=key, c0=c0, eps0=eps0, p=p, adm=adm)
        if adm:
            v0 = rb._v0_of(eps0, rb.B2['v_xi'])
            e0 = [-0.5] + [c0[i - 1] - rb.D_S[i] for i in range(1, 5)]
            S = sr2.run(P, rb.D_S, v0, e0, delays=sched(p), rule='ab',
                        apply_filter=True, terminate_on_contact=True,
                        extra=True)
            re_ = [abs(x) / eps_e for x in S['e_hnd'][1:]]
            rv = [abs(x) / eps_v for x in S['eps_hnd'][1:]]
            ev.update(rho=float(max(max(re_), max(rv))),
                      rho_e=[float(x) for x in re_],
                      rho_v=[float(x) for x in rv],
                      disp=S['disp'], disp_bound=Ax['disp'],
                      hits=S['hits'], stopped=S['all_stopped'],
                      egap=[float(abs(x)) for x in S['egap'][1:]])
        else:
            ev['rho'] = -1.0
        evals.append(ev); save()
        return ev

    t0 = time.time()
    best = None
    for r in range(restarts):
        c0, eps0 = draw_state(); p = draw_sched()
        cur = score(c0, eps0, p)
        tries = 0
        while not cur['adm'] and tries < 10:
            c0, eps0 = draw_state(); cur = score(c0, eps0, p); tries += 1
        for it in range(iters):
            c1 = [float(np.clip(x + rng.normal(0, 1.0), 40.0, 50.0))
                  for x in c0]
            e1 = [float(np.clip(eps0[0] + rng.normal(0, 0.05), -0.35, 0.0))]
            for j, f in enumerate((1.0, 0.8, 0.6, 0.4)):
                e1.append(float(np.clip(eps0[j + 1] + rng.normal(0, 0.03),
                                        0.0, 0.30 * f)))
            e1 = [round(x, 4) for x in e1]; c1 = [round(x, 3) for x in c1]
            q = [list(x) for x in p]
            i = int(rng.integers(0, 4)); f_ = int(rng.integers(0, 3))
            if f_ < 2:
                q[i][f_] = int(np.clip(q[i][f_] + rng.integers(-6, 7), 1, dmax))
            else:
                q[i][2] = float(np.clip(q[i][2] + rng.uniform(-4, 4),
                                        rb.B2['T_xi'] - 40, rb.B2['T_xi']))
            q = [tuple(x) for x in q]
            cand = score(c1, e1, q)
            if cand['rho'] > cur['rho']:
                cur, c0, eps0, p = cand, c1, e1, q
            print('  reach r%d it%d rho=%.3f best=%.3f (%.0f s)'
                  % (r, it, cand['rho'], cur['rho'], time.time() - t0),
                  flush=True)
        if best is None or cur['rho'] > best['rho']:
            best = cur
    scored = [e for e in evals if e['adm']]
    out = dict(best={k: v for k, v in best.items() if k != 'key'},
               n_evals=len(evals), n_admitted=len(scored),
               rho_max=max(e['rho'] for e in scored),
               rho_e_max=max(max(e['rho_e']) for e in scored),
               rho_v_max=max(max(e['rho_v']) for e in scored),
               disp_max=max(e['disp'] for e in scored),
               all_inside=bool(all(e['disp'] <= e['disp_bound'] + 1e-9
                                   and e['hits'] == 0 and e['stopped']
                                   for e in scored)),
               restarts=restarts, iters=iters, seed=seed,
               label='numerical reachability search over admitted '
                     'takeover states and FIFO schedules; a stress '
                     'estimate, not a proven maximum')
    print('  reach: rho_max=%.3f (e %.3f, v %.3f), disp_max=%.4f, '
          'all inside=%s, %d admitted of %d' % (
              out['rho_max'], out['rho_e_max'], out['rho_v_max'],
              out['disp_max'], out['all_inside'], len(scored),
              len(evals)))
    _merge('reach', out)


def part_age19():
    """Held-signal age check (audit U12): with one packet per 1 ms step
    and a transport delay of d steps, the command in use at an update
    instant is exactly d*T_m old, but the held signal ages to
    (d+1)*T_m just before the next delivery.  Rerun the benchmark with
    the transport delay reduced by one message period (19 steps) so
    that the held-signal age never exceeds the 0.020 s budget, and
    record the same indices as the nominal 20-step run."""
    out = {}
    for d in (19, 20):
        P = dict(rb.SIM_P, dbar=d * 1e-3)
        S = sr2.run(P, rb.D_S, rb.V0, rb.E0, rule='ab', apply_filter=True,
                    terminate_on_contact=True, extra=True)
        out['d%d' % d] = dict(
            transport_steps=d, age_at_update=d * 1e-3,
            held_age_max=(d + 1) * 1e-3,
            disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
            gmin=float(np.min(S['gmin'])), hits=S['hits'],
            infeas=S['infeas'], stopped=S['all_stopped'],
            mmin=[float(x) for x in S['mmin']],
            tstop_max=float(np.nanmax(S['tstop'])))
        print('  transport %d steps: disp=%.4f align=%.3f gap=%.3f '
              'gmin=%.2f hits=%d' % (d, S['disp'], S['align'],
                                     S['gaperr'], np.min(S['gmin']),
                                     S['hits']))
    _merge('age19', out)


PARTS = dict(sweepB=part_sweepB, sweepA=part_sweepA, act=part_act,
             age19=part_age19,
             margin=part_margin, decomp=part_decomp, reach=part_reach,
             scale=part_scale, msg=part_msg, tight=part_tight,
             runtime=part_runtime, plant=part_plant, vthr=part_vthr,
             cacc=part_cacc, caseA=part_caseA, resid=part_resid,
             review=part_review, plant2=part_plant2, gaptest=part_gaptest,
             mpc=part_mpc, delnage=part_delnage, merge=part_merge)

if __name__ == '__main__':
    args = sys.argv[1:] or ['resid']
    if args == ['quick']:
        args = ['resid', 'scale', 'act', 'vthr', 'cacc', 'caseA', 'msg',
                'runtime']
    for a in args:
        print('=== part:', a, '===', flush=True)
        PARTS[a]()
