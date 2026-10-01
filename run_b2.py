#!/usr/bin/env python3
"""run_b2.py -- primary-benchmark experimental program (root revision,
terminal-band design: every pair parks at 5.0 m physical).

Parts (merge-on-write into data/b2_results.json unless stated):
  cert     certificate at the operating point; per-condition latency
           boundaries (bisection, design frozen); certified entry;
           minimal-gap diagnostic (Prop. 1) below the 5.0 m band
  witness  interval witness for the primary design (corrected
           geometry; explicit gap vector) + Case-A corrected rerun
  sim      benchmark simulation (theorem RULE, applied filter),
           step-refinement convergence
  map      redispatch map over (v_xi, a_b) with max returned gap
  fixedmap frozen-design feasibility map over (v_xi, a_b)
  tc       control/detector-period sweep T_c in {1,5,10,20,50} ms
  scale    fleet-size scaling n in {3,5,8,10} (certificate side)
  gaps     uniform initial-gap sweep 40..50 m, frozen design
  ens      300 random takeover states in [40,50]^4 x speed box
  deln     random FIFO delay-realization trials (chunked; merged by
           part 'delnmerge')
  adv      adversarial FIFO delay search (random-restart hill climb)
  base     baselines incl. tuned feedforward-PID; contact-terminated
  het      speed-heterogeneity pricing + rejected wide-spread run
  perm     all 120 capability orderings
  loss     communication-loss / watchdog study on the primary case
  hifi     high-fidelity plant study (outside the theorem)

Usage: python run_b2.py cert sim ...      (or 'all' for fg set)
       python run_b2.py deln <chunk> <n>  (delay-trial chunk)
"""
import json
import os
import sys
import time
import itertools

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import b2_design as bd
import sim_root2 as sr2
import sim_hifi
import witness_root2 as wr2

OUT = os.path.join(CODE, 'data', 'b2_results.json')
OUT_FG = os.path.join(CODE, 'data', 'b2_results_fg.json')

# ------------------------------------------------------------ design
AMAX = [1.60, 1.48, 1.38, 1.29, 1.21]
LAM = 0.055
B2 = dict(v_xi=10.8, a_b=1.0, alpha=0.35, lam=LAM, beta=LAM * LAM,
          gamma=2 * LAM, s_m=1.0, T_xi=165.0, dbar=0.02,
          eps_e=0.15, eps_v=0.02,
          eps0=[-0.30, 0.25, 0.18, 0.12, 0.08],
          e0=[-0.5, 37.0, 37.0, 37.0, 37.0],  # pairs overwritten
          g_star=4.2)
C0 = (42.0, 45.0, 47.0, 49.0)
D_S = [6.0, 5.0, 5.0, 5.0, 5.0]      # prescribed terminal band edge
D_REQ = 0.9
ALIGN_TOL = 3.0
TARGET_MARGIN = 0.15                 # for the Prop.-1 diagnostic only
HEAD_SLOT = 6.0

SIM_P = dict(n=5, dt=1e-3, dbar=B2['dbar'], a_b=B2['a_b'],
             v_xi=B2['v_xi'], T_xi=B2['T_xi'], T_END=B2['T_xi'] + 25.0,
             alpha=B2['alpha'], beta=B2['beta'], gamma=B2['gamma'],
             phi=0.005, eps_det=0.015, ell=25.0, s_m=B2['s_m'],
             kappa=1.0, U=1.6, amax=tuple(AMAX), v_c=1.2)


def _v0_of(eps0, v_xi):
    v = []; vp = v_xi
    for e in eps0:
        v.append(vp - e); vp = v[-1]
    return v


V0 = _v0_of(B2['eps0'], B2['v_xi'])
E0 = [B2['e0'][0]] + [C0[i - 1] - D_S[i] for i in range(1, 5)]


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(x) for x in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return [_py(x) for x in o.tolist()]
    return o


def _merge(part, obj, path=OUT):
    data = {}
    if os.path.exists(path):
        data = json.load(open(path))
    data[part] = _py(obj)
    data['_design'] = _py(dict(B2=B2, c0=list(C0), amax=AMAX, d_s=D_S,
                               d_req=D_REQ, align_tol=ALIGN_TOL,
                               head_slot=HEAD_SLOT, v0=V0, e0=E0))
    json.dump(data, open(path, 'w'), indent=1)
    print('[%s] -> %s' % (part, os.path.basename(path)))


def _ns(ov=None):
    return bd._namespace(dict(B2, **(ov or {})))


def _eval(G, d_s=None, c0=C0, dbar=None):
    d = np.array(D_S if d_s is None else d_s, float)
    return bd.assemble(G, d.copy(), c0,
                       B2['dbar'] if dbar is None else dbar)


def _condset(A, E, align_tol=ALIGN_TOL):
    return dict(adm=bool(min(A['margins']) > 0),
                clr=bool(A['gmin'] > 0), auth=A['auth_ok'],
                event=A['event_ok'], ramp=A['ramp_ok'],
                vel=bool(A['vfloor'] > 0),
                sync=bool(A['disp'] <= D_REQ),
                align=bool(A['align'] <= align_tol),
                gap=bool(np.max(A['gaperr']) <= 1.0),
                hold=E['hold_ok'], hnd=E['hnd_ok'])


def _kpis(S):
    return dict(disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                gmin=[float(x) for x in S['gmin']],
                hmin=[float(x) for x in S['hmin']],
                upk=[float(x) for x in S['upk']],
                hits=S['hits'], infeas=S['infeas'],
                act_time=[float(x) for x in S['act_time']],
                tstop=[None if not np.isfinite(x) else float(x)
                       for x in S['tstop']],
                tset=[None if not np.isfinite(x) else float(x)
                      for x in S['tset']],
                contact_time=S.get('contact_time'),
                all_stopped=S['all_stopped'])


# ================================================================ parts
def part_cert():
    t0 = time.time()
    G = _ns()
    A = _eval(G)
    E = bd.entry_certify(G, A['R'], B2['dbar'])
    conds = _condset(A, E)
    R = A['R']

    # Prop.-1 diagnostic: front-to-back minimal gaps below the band
    d_min, Amin = bd.size_gaps(G, C0, dbar=B2['dbar'],
                               target=TARGET_MARGIN)
    d_min[0] = HEAD_SLOT
    A = _eval(G)     # restore the prescribed-geometry budgets

    memo = {}

    def eval_at(d):
        if d not in memo:
            Ax = _eval(G, dbar=d)
            Ex = bd.entry_certify(G, Ax['R'], d) if Ax['closed'] \
                else None
            memo[d] = (Ax, Ex)
        return memo[d]

    def cond(name, d):
        Ax, Ex = eval_at(d)
        if not Ax['closed']:
            return False
        return dict(sync=Ax['disp'] <= D_REQ,
                    align=Ax['align'] <= ALIGN_TOL,
                    gap=float(np.max(Ax['gaperr'])) <= 1.0,
                    adm=float(min(Ax['margins'])) > 0,
                    auth=Ax['auth_ok'], clr=Ax['gmin'] > 0,
                    event=Ax['event_ok'], vel=Ax['vfloor'] > 0,
                    ramp=Ax['ramp_ok'], hold=Ex['hold_ok'],
                    hnd=Ex['hnd_ok'])[name]

    ceils = {}
    for nm in ('sync', 'align', 'gap', 'adm', 'auth', 'clr', 'event',
               'vel', 'ramp', 'hold', 'hnd'):
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

    out = dict(
        d_s=[float(x) for x in D_S],
        d_min=[float(x) for x in d_min],
        arrest_floor=[float(x) for x in Amin['arrest_floor']],
        v0=V0, c0=list(C0), e0=E0,
        h0=[float(x) for x in A['h0'][1:]],
        margins=[float(x) for x in A['margins']],
        gmin=A['gmin'], auth_slack=A['auth_slack'],
        vfloor=A['vfloor'],
        disp=A['disp'], late=A['late'], sched=A['sched'],
        align=A['align'], gaperr=[float(x) for x in A['gaperr'][1:]],
        T_f=A['T_f'], T_brk=A['T_brk'],
        taub=[float(x) for x in R['taub']],
        tset=[float(x) for x in R['tset']],
        E1=[float(x) for x in R['E1']],
        epsb=[float(x) for x in R['epsb'][1:]],
        anet=[float(x) for x in (B2['a_b'] - R['Fb'][1:5])],
        hops=[float(x) for x in R['hops']],
        Leps=float(R['Leps']),
        entry=E, conds=conds, all_ok=all(conds.values()),
        ceils=ceils, dcert_num=float(min(ceils.values())),
        binding=min(ceils, key=ceils.get),
        runtime_s=round(time.time() - t0, 1))
    _merge('cert', out)
    print('  conds:', conds)
    print('  ceilings:', {k: round(v, 3) for k, v in ceils.items()})
    print('  margins:', np.round(out['margins'], 3),
          ' d_min:', np.round(d_min, 2))


def part_witness():
    import tempfile
    crossings = None
    if os.path.exists(OUT):
        D = json.load(open(OUT))
        if 'cert' in D:
            ce = D['cert']['ceils']
            crossings = {nm: (float(np.floor(ce[nm] * 1000) / 1000),
                              [wn]) for nm, wn in
                         (('adm', 'adm'), ('clr', 'clr'),
                          ('auth', 'auth'), ('sync', 'sync'),
                          ('align', 'align'), ('gap', 'gap'))}
    ov = {k: B2[k] for k in ('v_xi', 'a_b', 'alpha', 'lam', 'beta',
                             'gamma', 's_m', 'T_xi', 'dbar',
                             'eps_e', 'eps0')}
    ov['e0'] = [round(float(x), 4) for x in E0]   # true closures
    with tempfile.TemporaryDirectory() as td:
        rep = wr2.run(overrides=ov,
                      dbar_op=B2['dbar'], d_s=D_S, sync_tol=D_REQ,
                      align_tol=ALIGN_TOL, crossings=crossings,
                      workdir=td)
    with tempfile.TemporaryDirectory() as td:
        repA = wr2.run(workdir=td)          # corrected Case A
    _merge('witness', dict(primary=rep, caseA_corrected=repA))
    mm = {k: round(v, 4) for k, v in rep['margins'].items()
          if not k.startswith(('_', '#'))}
    print('  primary witness ok=%s margins=%s' % (rep['ok'], mm))
    print('  primary disp interval upper = %.6f'
          % rep['margins']['_disp_hi'])
    print('  Case A corrected ok=%s adm=%.4f clr=%.4f'
          % (repA['ok'], repA['margins']['adm'],
             repA['margins']['clr']))


def part_sim():
    runs = {}
    for tag, rule in (('rule_ab', 'ab'), ('rule_gapclose',
                                          'gapclose')):
        S = sr2.run(SIM_P, D_S, V0, E0, rule=rule, apply_filter=True,
                    record=(rule == 'ab'), terminate_on_contact=True)
        runs[tag] = _kpis(S)
        if rule == 'ab':
            np.savez(os.path.join(CODE, 'data', 'b2_traj.npz'),
                     t=S['t'], v=S['v'], e=S['e'], u=S['u'],
                     h=S['h'], g=S['g'], tstop=S['tstop'],
                     tact=S['tact'], tset=S['tset'],
                     marks=S['marks'], s=S['s'], egap=S['egap'],
                     d_s=np.array(D_S), v0=np.array(V0),
                     e0=np.array(E0))
    conv = {}
    for dt in (1e-3, 5e-4, 2.5e-4):
        S = sr2.run(dict(SIM_P, dt=dt), D_S, V0, E0, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        conv['%g' % dt] = dict(disp=S['disp'], align=S['align'],
                               gaperr=S['gaperr'],
                               gmin=float(np.min(S['gmin'])))
    _merge('sim', dict(runs=runs, conv=conv))
    S = runs['rule_ab']
    print('  disp=%.4f align=%.3f gaperr=%.3f gmin=%.2f stopped=%s'
          % (S['disp'], S['align'], S['gaperr'], min(S['gmin']),
             S['all_stopped']))


def _map_rows(fixed_design):
    rows = []
    for v_xi in np.arange(10.0, 15.01, 0.5):
        for a_b in np.arange(0.80, 1.201, 0.05):
            ov = dict(v_xi=float(v_xi), a_b=float(round(a_b, 2)))
            try:
                G = _ns(ov)
                if fixed_design:
                    d_s = np.array(D_S)
                else:
                    d_s, _ = bd.size_gaps(G, C0, dbar=B2['dbar'],
                                          target=TARGET_MARGIN)
                    d_s[0] = HEAD_SLOT
                    d_s[1:] = np.maximum(d_s[1:], 5.0)
                A = bd.assemble(G, np.array(d_s, float).copy(), C0,
                                B2['dbar'])
                E = bd.entry_certify(G, A['R'], B2['dbar']) \
                    if A['closed'] else None
                conds = _condset(A, E) if A['closed'] else {}
                ok = A['closed'] and all(conds.values())
                rows.append(dict(
                    v_xi=float(v_xi), a_b=float(round(a_b, 2)),
                    ok=bool(ok),
                    failed=None if ok else
                    [k for k, v in conds.items() if not v] or
                    ['ledger'],
                    max_ds=float(np.max(np.array(d_s)[1:])),
                    disp=A['disp'] if A['closed'] else None,
                    margin=float(min(A['margins']))
                    if A['closed'] else None))
            except Exception as ex:
                rows.append(dict(v_xi=float(v_xi),
                                 a_b=float(round(a_b, 2)), ok=False,
                                 failed=['error:%s' % ex]))
    return rows


def part_map():
    rows = _map_rows(fixed_design=False)
    _merge('map', dict(rows=rows, kind='redispatch, gaps in '
                                       '[5.0,5.5] band or reject'))
    print('  redispatch map: %d/%d admitted'
          % (sum(1 for r in rows if r['ok']), len(rows)))


def part_fixedmap():
    rows = _map_rows(fixed_design=True)
    _merge('fixedmap', dict(rows=rows, kind='frozen 5.0 m design'))
    print('  fixed-design map: %d/%d admitted'
          % (sum(1 for r in rows if r['ok']), len(rows)))


def part_tc():
    rows = []
    for tc in (0.001, 0.005, 0.010, 0.020, 0.050):
        db = max(B2['dbar'], 2 * tc)     # age bound must exceed T_c
        try:
            G = _ns(dict(T_c=tc, dbar=db))
            A = _eval(G, dbar=db)
            E = bd.entry_certify(G, A['R'], db) if A['closed'] \
                else None
            conds = _condset(A, E) if A['closed'] else {}
            ok = A['closed'] and all(conds.values())
            S = sr2.run(dict(SIM_P, dt=tc, dbar=db), D_S, V0, E0,
                        rule='ab', apply_filter=True,
                        terminate_on_contact=True)
            rows.append(dict(T_c=tc, dbar=db, cert_ok=bool(ok),
                             failed=None if ok else
                             [k for k, v in conds.items()
                              if not v] or ['ledger'],
                             disp_bound=A['disp'] if A['closed']
                             else None,
                             disp=S['disp'], align=S['align'],
                             gmin=float(np.min(S['gmin'])),
                             stopped=S['all_stopped']))
        except Exception as ex:
            rows.append(dict(T_c=tc, dbar=db, cert_ok=False,
                             failed=['error:%s' % ex]))
        print('  T_c=%g: %s' % (tc, rows[-1]))
    _merge('tc', rows, path=OUT_FG)


def part_scale():
    rows = []
    for n in (3, 5, 8, 10):
        r = (1.21 / 1.60) ** (1.0 / (n - 1))
        am = [round(1.60 * r ** k, 3) for k in range(n)]
        eps0 = [-0.30] + [round(0.25 * (0.7 ** k), 3)
                          for k in range(n - 1)]
        e0 = [-0.5] + [37.0] * (n - 1)
        ds = [HEAD_SLOT] + [5.0] * (n - 1)
        c0 = tuple(np.linspace(42.0, 49.0, n - 1).round(1))
        try:
            G = bd._namespace(dict(B2, n=n, amax=am, eps0=eps0,
                                   e0=e0, alpha=B2['alpha']))
            A = bd.assemble(G, np.array(ds, float), c0, B2['dbar'])
            E = bd.entry_certify(G, A['R'], B2['dbar']) \
                if A['closed'] else None
            conds = _condset(A, E) if A['closed'] else {}
            ok = A['closed'] and all(conds.values())
            rows.append(dict(n=n, ok=bool(ok),
                             failed=None if ok else
                             [k for k, v in conds.items()
                              if not v] or ['ledger'],
                             margin=float(min(A['margins']))
                             if A['closed'] else None,
                             disp=A['disp'] if A['closed'] else None,
                             auth_slack=float(min(A['auth_slack']))
                             if A['closed'] else None,
                             LepsTc=float(A['R']['Leps'] * 0.001)
                             if A['closed'] else None))
        except Exception as ex:
            rows.append(dict(n=n, ok=False, failed=['error:%s' % ex]))
        print('  n=%d: %s' % (n, rows[-1]))
    _merge('scale', rows, path=OUT_FG)


def part_gaps():
    G = _ns()
    rows = []
    for c in np.arange(40.0, 50.01, 1.0):
        c0 = (float(c),) * 4
        A = bd.assemble(G, np.array(D_S, float), c0, B2['dbar'])
        e0 = [B2['e0'][0]] + [c - D_S[i] for i in range(1, 5)]
        S = sr2.run(SIM_P, D_S, V0, e0, rule='ab', apply_filter=True,
                    terminate_on_contact=True)
        rows.append(dict(c0=float(c),
                         margin=float(min(A['margins'])),
                         adm=bool(min(A['margins']) > 0),
                         gmin_cert=A['gmin'],
                         disp=S['disp'], align=S['align'],
                         gmin_real=float(np.min(S['gmin'])),
                         hits=S['hits']))
    _merge('gaps', rows, path=OUT_FG)
    print('  gap sweep adm:', [r['adm'] for r in rows])


def part_ens(n_draw=300, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    t0 = time.time()
    for j in range(n_draw):
        c0 = tuple(rng.uniform(40.0, 50.0, 4).round(3))
        eps0 = [float(rng.uniform(-0.35, 0.0))] + \
            [float(rng.uniform(0.0, 0.30) * f)
             for f in (1.0, 0.8, 0.6, 0.4)]
        Gx = bd._namespace(dict(B2, eps0=[round(x, 4) for x in eps0]))
        Ax = bd.assemble(Gx, np.array(D_S, float), c0, B2['dbar'])
        adm = (Ax['closed'] and min(Ax['margins']) > 0 and
               Ax['auth_ok'] and Ax['event_ok'] and Ax['gmin'] > 0)
        rec = dict(c0=list(c0), eps0=eps0, adm=bool(adm))
        if adm:
            rec.update(disp_bound=Ax['disp'], align_bound=Ax['align'],
                       gaperr_bound=float(np.max(Ax['gaperr'])),
                       gmin_cert=Ax['gmin'])
            v0 = _v0_of(eps0, B2['v_xi'])
            e0 = [-0.5] + [c0[i - 1] - D_S[i] for i in range(1, 5)]
            S = sr2.run(SIM_P, D_S, v0, e0, rule='ab',
                        apply_filter=True, terminate_on_contact=True)
            rec.update(disp=S['disp'], align=S['align'],
                       gaperr=S['gaperr'],
                       gmin_real=float(np.min(S['gmin'])),
                       hits=S['hits'], all_stopped=S['all_stopped'])
        rows.append(rec)
        if (j + 1) % 50 == 0:
            print('  ens %d/%d (%.0f s)' % (j + 1, n_draw,
                                            time.time() - t0))
    _merge('ens', dict(rows=rows, n=n_draw,
                       admitted=sum(1 for r in rows if r['adm']),
                       seed=seed))


def _random_schedule(rng, N, dmax):
    """Piecewise-constant random per-hop delays (steps), FIFO via
    the receiver's monotone arrival pass."""
    d = np.empty((5, N), int)
    for i in range(1, 5):
        k = 0
        while k < N:
            seg = int(rng.uniform(0.5, 5.0) / 1e-3)
            d[i, k:k + seg] = rng.integers(1, dmax + 1)
            k += seg
    d[0] = 1
    return d


def part_deln(chunk=0, n_trials=250, seed_base=100):
    rng = np.random.default_rng(seed_base + chunk)
    N = int(SIM_P['T_END'] / SIM_P['dt'])
    dmax = int(round(B2['dbar'] / SIM_P['dt']))
    rows = []
    t0 = time.time()
    for j in range(n_trials):
        Dl = _random_schedule(rng, N, dmax)
        S = sr2.run(SIM_P, D_S, V0, E0, delays=Dl, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        rows.append(dict(disp=S['disp'], align=S['align'],
                         gaperr=S['gaperr'],
                         gmin=float(np.min(S['gmin'])),
                         hits=S['hits'],
                         stopped=S['all_stopped']))
        if (j + 1) % 50 == 0:
            print('  deln c%d %d/%d (%.0f s)'
                  % (chunk, j + 1, n_trials, time.time() - t0))
    json.dump(_py(rows), open(os.path.join(
        CODE, 'data', 'b2_deln_chunk%d.json' % chunk), 'w'))
    print('[deln chunk %d] %d trials done' % (chunk, len(rows)))


def part_delnmerge():
    rows = []
    for f in sorted(os.listdir(os.path.join(CODE, 'data'))):
        if f.startswith('b2_deln_chunk'):
            rows += json.load(open(os.path.join(CODE, 'data', f)))
    disp = sorted(r['disp'] for r in rows)
    aln = sorted(r['align'] for r in rows)

    def pct(a, p):
        return a[min(len(a) - 1, int(p * len(a)))]
    summ = dict(n=len(rows),
                stopped=sum(1 for r in rows if r['stopped']),
                dormant=sum(1 for r in rows if r['hits'] == 0),
                disp_max=max(disp), disp_p95=pct(disp, 0.95),
                disp_p99=pct(disp, 0.99),
                align_max=max(aln), align_p95=pct(aln, 0.95),
                gmin_min=min(r['gmin'] for r in rows))
    _merge('deln', dict(summary=summ, rows=rows))
    print('  deln merged:', summ)


def part_adv(iters=140, seed=23):
    rng = np.random.default_rng(seed)
    N = int(SIM_P['T_END'] / SIM_P['dt'])
    dmax = int(round(B2['dbar'] / SIM_P['dt']))
    # parameterization: per hop, (early delay, late delay, switch
    # time near handoff) -- targets the ramp-propagation window
    def sched(p):
        d = np.empty((5, N), int)
        d[0] = 1
        for i in range(1, 5):
            e_, l_, ts = p[i - 1]
            kk = int(ts / 1e-3)
            d[i, :kk] = e_; d[i, kk:] = l_
        return d

    def cost(p):
        S = sr2.run(SIM_P, D_S, V0, E0, delays=sched(p), rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        return S['disp'], S

    best = None
    for restart in range(4):
        p = [(int(rng.integers(1, dmax + 1)),
              int(rng.integers(1, dmax + 1)),
              float(rng.uniform(B2['T_xi'] - 10, B2['T_xi'] + 5)))
             for _ in range(4)]
        c, S = cost(p)
        for it in range(iters // 4 - 1):
            q = [list(x) for x in p]
            i = int(rng.integers(0, 4))
            f = int(rng.integers(0, 3))
            if f < 2:
                q[i][f] = int(np.clip(q[i][f] + rng.integers(-8, 9),
                                      1, dmax))
            else:
                q[i][2] = float(np.clip(q[i][2] + rng.uniform(-3, 3),
                                        0.0, SIM_P['T_END'] - 1))
            q = [tuple(x) for x in q]
            c2, S2 = cost(q)
            if c2 > c:
                p, c, S = q, c2, S2
        if best is None or c > best[0]:
            best = (c, p, _kpis(S))
    _merge('adv', dict(worst_disp=best[0], schedule=best[1],
                       kpis=best[2], iters=iters,
                       label='adversarial numerical search, not a '
                             'proven worst case'))
    print('  adversarial worst disp = %.4f (bound %.3f)'
          % (best[0], 0.179))


def part_base():
    rows = {}
    for tag, kw in (
            ('proposed', dict(variant='proposed')),
            ('pid', dict(variant='pid')),
            ('single', dict(variant='single')),
            ('noff', dict(variant='noff')),
            ('lqr', dict(variant='lqr')),
            ('common_filter', dict(variant='proposed',
                                   filter_slopes='common')),
            ('proposed_nofilter', dict(variant='proposed',
                                       apply_filter=False))):
        af = kw.pop('apply_filter', True)
        S = sr2.run(SIM_P, D_S, V0, E0, rule='ab', apply_filter=af,
                    terminate_on_contact=True, **kw)
        rows[tag] = _kpis(S)
        print('  %-18s disp=%s align=%.3f gmin=%.2f upk=%.2f '
              'contact=%s stopped=%s'
              % (tag, ('%.4f' % S['disp']) if S['disp'] is not None
                 else 'n/a', S['align'], float(np.min(S['gmin'])),
                 float(np.max(S['upk'])), S.get('contact_time'),
                 S['all_stopped']))
    _merge('base', rows, path=OUT_FG)


def part_hard():
    """S1-stage ablation on a doubled speed-mismatch takeover."""
    eps0h = [2 * x for x in B2['eps0']]
    v0h = _v0_of(eps0h, B2['v_xi'])
    runs = {}
    for tag in ('proposed', 'single'):
        S = sr2.run(SIM_P, D_S, v0h, E0, rule='ab', apply_filter=True,
                    terminate_on_contact=True, variant=tag)
        runs[tag] = _kpis(S)
        print('  hard %-9s disp=%.4f align=%.3f gmin=%.2f upk=%.3f'
              % (tag, S['disp'], S['align'], float(np.min(S['gmin'])),
                 float(np.max(S['upk']))))
    _merge('ablate_hard', dict(eps0=eps0h, v0=v0h, runs=runs),
           path=OUT_FG)


def part_het():
    rows = []
    base = np.array(B2['eps0'])
    for scale in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5):
        eps0 = [round(float(x), 4) for x in base * scale]
        try:
            G = _ns(dict(eps0=eps0))
            d_s, Am = bd.size_gaps(G, C0, dbar=B2['dbar'],
                                   target=TARGET_MARGIN)
            d_s[0] = HEAD_SLOT
            band_ok = bool(np.max(np.maximum(d_s[1:], 5.0)) <= 5.5)
            A = bd.assemble(G, np.maximum(d_s, [0, 5, 5, 5, 5]
                                          ).astype(float), C0,
                            B2['dbar'])
            v0 = _v0_of(eps0, B2['v_xi'])
            ok = (A['closed'] and min(A['margins']) > 0.0
                  and A['auth_ok'] and A['event_ok']
                  and A['gmin'] > 0 and A['align'] <= ALIGN_TOL)
            rows.append(dict(scale=scale, eps0=eps0,
                             spread=float(max(v0) - min(v0)),
                             ok=bool(ok), band_ok=band_ok,
                             d_min=[float(x) for x in d_s[1:]],
                             taub=[float(x) for x in A['R']['taub']],
                             disp=A['disp']))
        except Exception as ex:
            rows.append(dict(scale=scale, ok=False, err=str(ex)))
    eps0w = [-1.2, 1.0, 0.9, 0.8, 0.7]
    v0w = _v0_of(eps0w, 12.0)
    Gw = _ns(dict(v_xi=12.0, eps0=eps0w))
    Aw = bd.assemble(Gw, np.array(D_S, float), C0, B2['dbar'])
    Sw = sr2.run(dict(SIM_P, v_xi=12.0), D_S, v0w, E0, rule='ab',
                 apply_filter=True, terminate_on_contact=True)
    wide = dict(v0=v0w, spread=float(max(v0w) - min(v0w)),
                adm=bool(Aw['closed'] and min(Aw['margins']) > 0),
                **_kpis(Sw))
    _merge('het', dict(rows=rows, wide=wide), path=OUT_FG)
    for r in rows:
        print('  scale %.1f ok=%s band_ok=%s d_min_max=%.2f '
              'taub_max=%.2f'
              % (r['scale'], r['ok'], r.get('band_ok'),
                 max(r.get('d_min', [0])), max(r.get('taub', [0]))))
    print('  wide: adm=%s stopped=%s' % (wide['adm'],
                                         wide['all_stopped']))


def part_perm():
    rows = []
    for j, perm in enumerate(itertools.permutations(AMAX)):
        try:
            G = _ns(dict(amax=list(perm)))
            A = bd.assemble(G, np.array(D_S, float), C0, B2['dbar'])
            ok = (A['closed'] and min(A['margins']) > 0
                  and A['auth_ok'] and A['event_ok'] and A['gmin'] > 0
                  and A['align'] <= ALIGN_TOL)
            rows.append(dict(perm=list(perm), ok=bool(ok),
                             margin=float(min(A['margins']))
                             if A['closed'] else None,
                             gmin=A['gmin'] if A['closed'] else None,
                             auth_slack=float(min(A['auth_slack']))
                             if A['closed'] else None))
        except Exception as ex:
            rows.append(dict(perm=list(perm), ok=False, err=str(ex)))
    okn = sum(1 for r in rows if r['ok'])
    _merge('perm', dict(rows=rows,
                        note='frozen 5.0 m design, gaps NOT resized'))
    print('  perms admitted at the frozen 5.0 m design: %d/120' % okn)


def part_loss():
    out = {}
    for tag, kw in (('no_watchdog', dict(T_wd=None)),
                    ('wd_010', dict(T_wd=0.10)),
                    ('wd_020', dict(T_wd=0.20)),
                    ('wd_025', dict(T_wd=0.25))):
        S = sr2.run(SIM_P, D_S, V0, E0, rule='ab', apply_filter=True,
                    scenario='emergency', t_emergency=25.0,
                    terminate_on_contact=True, **kw)
        out[tag] = _kpis(S)
        print('  %-11s contact=%s gmin=%.2f stopped=%s'
              % (tag, S.get('contact_time'),
                 float(np.min(S['gmin'])), S['all_stopped']))
    _merge('loss', out, path=OUT_FG)


def part_hifi():
    import sim_hifi as sh
    out = {}
    S = sh.run(SIM_P, D_S, V0, E0, seed=1)
    out['nominal_hifi'] = S
    print('  hifi nominal: disp=%s align=%.3f gmin=%.2f contact=%s '
          'stopped=%s jerk_pk=%.2f'
          % (S['disp'], S['align'], min(S['gmin']),
             S['contact_time'], S['all_stopped'], max(S['jerk_pk'])))
    seeds = []
    for sd in range(2, 22):
        R = sh.run(SIM_P, D_S, V0, E0, seed=sd)
        seeds.append(dict(disp=R['disp'], align=R['align'],
                          gmin=min(R['gmin']),
                          stopped=R['all_stopped'],
                          contact=R['contact_time']))
    out['seeds'] = seeds
    out['params'] = sh.DEFAULT
    ok = sum(1 for r in seeds if r['stopped'] and r['contact'] is None)
    print('  hifi seeds: %d/%d stop without contact; disp range '
          '%.3f-%.3f align max %.3f'
          % (ok, len(seeds), min(r['disp'] for r in seeds),
             max(r['disp'] for r in seeds),
             max(r['align'] for r in seeds)))
    _merge('hifi', out, path=OUT_FG)


PARTS = dict(cert=part_cert, witness=part_witness, sim=part_sim,
             map=part_map, fixedmap=part_fixedmap, tc=part_tc,
             scale=part_scale, gaps=part_gaps, ens=part_ens,
             delnmerge=part_delnmerge, adv=part_adv, base=part_base,
             hard=part_hard,
             het=part_het, perm=part_perm, loss=part_loss,
             hifi=part_hifi)

if __name__ == '__main__':
    args = sys.argv[1:] or ['cert', 'sim']
    if args and args[0] == 'deln':
        part_deln(chunk=int(args[1]), n_trials=int(args[2]))
        sys.exit(0)
    if args == ['all']:
        args = ['cert', 'witness', 'sim', 'base', 'tc', 'loss',
                'hifi', 'het', 'gaps', 'scale']
    for a in args:
        print('=== part:', a, '===')
        PARTS[a]()
