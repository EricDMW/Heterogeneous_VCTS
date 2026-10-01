#!/usr/bin/env python3
"""run_new_d19.py -- corrected-contract reruns for the new/ revision
(review of 2026-09-25, items P0-01/P0-02): every simulation presented
as an instance of the theorem must satisfy the continuous-time held-age
contract of its certificate.  With one packet per 1 ms step and a
transport delay of d steps, the held signal ages to (d+1) ms just
before the next delivery, so the 20 ms contract of Case B needs d <= 19
and the 60 ms contract of Case A needs d <= 59.  This driver reruns the
simulation-based parts with

    Case B: constant transport 19 steps (held age <= 20 ms);
            random / adversarial schedules with d_j in {1,...,19};
    Case A: constant transport 59 steps (held age <= 60 ms);
    sweeps: at grid age dbar, transport dbar - 1 ms, random d_j <= that,

and audits, for every schedule, the left-limit held age max_j (a_{j+1}
- j) T_m (age just before the next delivery) in addition to the age at
update instants.  Certificates are untouched (evaluated at the contract
age).  Nothing under data/ is modified: outputs go to data/new_d19/
(seeded with copies of the archived result files so that parts not
rerun -- certificate-only parts, the message-period study, the
perturbed-plant and MPC studies at their documented 20-step delay --
stay available under the same keys).

Usage:  python run_new_d19.py <part> [arg]
  parts: seed | sim | base | cacc | review | act | vthr | loss | caseA |
         emergency | ens <slice 0..2> | ensmerge | deln <chunk 0..3> |
         delnmerge | adv | decomp | reach | sweepB | sweepA | tight |
         merge | lamscan (certificate only) | caseA_conv
"""
import json
import os
import shutil
import sys
import time

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
import run_b2 as rb                      # noqa: E402  (no main on import)
import run_b2_ext as rx                  # noqa: E402
bd = rb.bd
sr2 = rb.sr2

NEW = os.path.join(CODE, 'data', 'new_d19')
OUT_B = os.path.join(NEW, 'b2_results.json')
OUT_FG = os.path.join(NEW, 'b2_results_fg.json')
PARTDIR = os.path.join(NEW, 'b2_ext_parts')
OUT_X = os.path.join(NEW, 'b2_results_ext.json')
TRAJ = os.path.join(NEW, 'b2_traj.npz')
D_B = 19            # Case B transport steps  (contract 20 ms)
D_A = 59            # Case A transport steps  (contract 60 ms)

# ---- contract-consistent simulation parameters (certificates unchanged)
rb.SIM_P['dbar'] = D_B * 1e-3
rx.A_SIM['dbar'] = D_A * 1e-3
rx.PARTDIR = PARTDIR
rb.OUT = OUT_B                       # parts that read ens / adv / cert
rb.OUT_FG = OUT_FG


def _merge_b(part, obj, path=None):
    path = OUT_B if path in (None, ) or path == rb.OUT else path
    if path not in (OUT_B, OUT_FG):
        path = OUT_FG if 'fg' in os.path.basename(path) else OUT_B
    data = json.load(open(path)) if os.path.exists(path) else {}
    data[part] = rb._py(obj)
    data['_contract'] = dict(transport_steps_B=D_B, transport_steps_A=D_A,
                             held_age_max_B=(D_B + 1) * 1e-3,
                             held_age_max_A=(D_A + 1) * 1e-3,
                             note='simulation reruns of 2026-09-25 under '
                                  'the continuous-time held-age contract')
    json.dump(data, open(path, 'w'), indent=1)
    print('[%s] -> %s' % (part, os.path.relpath(path, CODE)), flush=True)


def _merge_x(part, obj):
    os.makedirs(PARTDIR, exist_ok=True)
    p = os.path.join(PARTDIR, part + '.json')
    json.dump(rb._py(obj), open(p, 'w'), indent=1)
    print('[%s] -> %s' % (part, os.path.relpath(p, CODE)), flush=True)


rb._merge = _merge_b
rx._merge = _merge_x


# ------------------------------------------------------------ audit
def held_age_audit(row, N):
    """Left-limit held age of one hop: packets sent at integer steps j
    with transport d_j, FIFO projection a_j = max_{i<=j}(i + d_i); the
    packet held on [a_j, a_{j+1}) ages to (a_{j+1} - j) T_m at the left
    limit of the next delivery.  Returns steps."""
    j = np.arange(N)
    a = np.maximum.accumulate(j + np.asarray(row[:N], int))
    left = a[1:] - j[:-1]
    ks = np.arange(N)
    idx = np.searchsorted(a, ks, side='right') - 1
    ok = idx >= 0
    at_update = ks[ok] - idx[ok]
    return dict(max_left_limit_steps=int(left.max()),
                max_update_steps=int(at_update.max()),
                first_delivery_step=int(a[0]))


def audit_schedule(Dl, N):
    per = [held_age_audit(Dl[i], N) for i in range(1, Dl.shape[0])]
    return dict(max_left_limit_steps=max(p['max_left_limit_steps'] for p in per),
                max_update_steps=max(p['max_update_steps'] for p in per),
                per_hop=per)


# ------------------------------------------------------------- parts
def part_seed():
    os.makedirs(PARTDIR, exist_ok=True)
    for src, dst in ((os.path.join(CODE, 'data', 'b2_results.json'), OUT_B),
                     (os.path.join(CODE, 'data', 'b2_results_fg.json'), OUT_FG)):
        shutil.copy(src, dst)
    src = os.path.join(CODE, 'data', 'b2_ext_parts')
    for f in os.listdir(src):
        if f.endswith('.json') and not f.endswith('_ckpt.json'):
            shutil.copy(os.path.join(src, f), os.path.join(PARTDIR, f))
    print('seeded', NEW)


def part_sim():
    """Canonical Case-B run at 19 steps with the trajectory, the filter
    margins and the step-refinement runs; the transport-delay audit."""
    P = dict(rb.SIM_P)
    N = int(P['T_END'] / P['dt'])
    S = sr2.run(P, rb.D_S, rb.V0, rb.E0, rule='ab', apply_filter=True,
                record=True, terminate_on_contact=True, extra=True)
    assert S['hits'] == 0 and S['infeas'] == 0 and S['all_stopped']
    np.savez(TRAJ, t=S['t'], v=S['v'], e=S['e'], u=S['u'], h=S['h'],
             g=S['g'], tstop=S['tstop'], tact=S['tact'], tset=S['tset'],
             marks=S['marks'], s=S['s'], egap=S['egap'],
             d_s=np.array(rb.D_S), v0=np.array(rb.V0), e0=np.array(rb.E0))
    runs = {'rule_ab': rb._kpis(S)}
    Sg = sr2.run(P, rb.D_S, rb.V0, rb.E0, rule='gapclose', apply_filter=True,
                 terminate_on_contact=True)
    runs['rule_gapclose'] = rb._kpis(Sg)
    conv = {}
    for dt in (1e-3, 5e-4, 2.5e-4):
        Sc = sr2.run(dict(P, dt=dt), rb.D_S, rb.V0, rb.E0, rule='ab',
                     apply_filter=True, terminate_on_contact=True)
        conv['%g' % dt] = dict(disp=Sc['disp'], align=Sc['align'],
                               gaperr=Sc['gaperr'],
                               gmin=float(np.min(Sc['gmin'])))
    const = np.full((5, N), D_B, int); const[0] = 1
    aud = audit_schedule(const, N)
    _merge_b('sim', dict(runs=runs, conv=conv, transport_steps=D_B,
                         age_audit=aud))
    margin = dict(mmin=[float(x) for x in S['mmin']],
                  m=[[[float(a), float(b)] for a, b in x] for x in S['m']],
                  disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                  e_hnd=[float(x) for x in S['e_hnd']],
                  eps_hnd=[float(x) for x in S['eps_hnd']],
                  wint=[float(x) for x in S['wint']],
                  egap_signed=[float(x) for x in S['egap']],
                  marker_signed=[float(x) for x in (S['s'] - S['marks'])],
                  tstop=[float(x) for x in S['tstop']],
                  tset=[float(x) for x in S['tset']],
                  tact=[float(x) for x in S['tact']],
                  upk=[float(x) for x in S['upk']],
                  gmin=[float(x) for x in S['gmin']])
    _merge_x('margin', margin)
    print('  sim: disp=%.4f align=%.3f gaperr=%.3f gmin=%.3f mmin=%.3f '
          'audit=%s' % (S['disp'], S['align'], S['gaperr'],
                        np.min(S['gmin']), min(S['mmin']), aud), flush=True)


def part_ens(slice_k, n_slices=3, n_draw=300, seed=7):
    """Ensemble states are drawn once from the archived seed; each slice
    simulates every third state so that three processes reproduce the
    archived sample exactly."""
    rng = np.random.default_rng(seed)
    states = []
    for j in range(n_draw):
        c0 = tuple(rng.uniform(40.0, 50.0, 4).round(3))
        eps0 = [float(rng.uniform(-0.35, 0.0))] + \
            [float(rng.uniform(0.0, 0.30) * f) for f in (1.0, 0.8, 0.6, 0.4)]
        states.append((j, c0, eps0))
    rows = {}
    t0 = time.time()
    for j, c0, eps0 in states[slice_k::n_slices]:
        Gx = bd._namespace(dict(rb.B2, eps0=[round(x, 4) for x in eps0]))
        Ax = bd.assemble(Gx, np.array(rb.D_S, float), c0, rb.B2['dbar'])
        adm = (Ax['closed'] and min(Ax['margins']) > 0 and Ax['auth_ok']
               and Ax['event_ok'] and Ax['gmin'] > 0)
        rec = dict(c0=list(c0), eps0=eps0, adm=bool(adm))
        if adm:
            rec.update(disp_bound=Ax['disp'], align_bound=Ax['align'],
                       gaperr_bound=float(np.max(Ax['gaperr'])),
                       gmin_cert=Ax['gmin'])
            v0 = rb._v0_of(eps0, rb.B2['v_xi'])
            e0 = [-0.5] + [c0[i - 1] - rb.D_S[i] for i in range(1, 5)]
            S = sr2.run(rb.SIM_P, rb.D_S, v0, e0, rule='ab',
                        apply_filter=True, terminate_on_contact=True)
            rec.update(disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                       gmin_real=float(np.min(S['gmin'])), hits=S['hits'],
                       all_stopped=S['all_stopped'])
        rows[j] = rec
        if len(rows) % 20 == 0:
            print('  ens slice %d: %d done (%.0f s)'
                  % (slice_k, len(rows), time.time() - t0), flush=True)
    json.dump(rb._py(rows), open(os.path.join(NEW, 'ens_slice%d.json'
                                              % slice_k), 'w'))
    print('[ens slice %d] done' % slice_k, flush=True)


def part_ensmerge(n_slices=3, n_draw=300, seed=7):
    rows = {}
    for k in range(n_slices):
        rows.update(json.load(open(os.path.join(NEW, 'ens_slice%d.json' % k))))
    rows = [rows[str(j)] for j in range(n_draw)]
    _merge_b('ens', dict(rows=rows, n=n_draw,
                         admitted=sum(1 for r in rows if r['adm']),
                         seed=seed, transport_steps=D_B))
    dd = [r['disp'] for r in rows if r['adm']]
    print('  ens: admitted %d/%d, dormant %d, disp max %.4f, align max %.3f'
          % (sum(r['adm'] for r in rows), n_draw,
             sum(1 for r in rows if r['adm'] and r['hits'] == 0),
             max(dd), max(r['align'] for r in rows if r['adm'])))


def part_deln(chunk, n_trials=250, seed_base=100):
    rng = np.random.default_rng(seed_base + chunk)
    N = int(rb.SIM_P['T_END'] / rb.SIM_P['dt'])
    rows = []
    t0 = time.time()
    for j in range(n_trials):
        Dl = rb._random_schedule(rng, N, D_B)
        aud = audit_schedule(Dl, N)
        assert aud['max_left_limit_steps'] <= D_B + 1, aud
        S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, delays=Dl, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        rows.append(dict(disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                         gmin=float(np.min(S['gmin'])), hits=S['hits'],
                         stopped=S['all_stopped'],
                         age_left_limit_steps=aud['max_left_limit_steps'],
                         age_update_steps=aud['max_update_steps']))
        if (j + 1) % 25 == 0:
            print('  deln c%d %d/%d (%.0f s)' % (chunk, j + 1, n_trials,
                                                 time.time() - t0), flush=True)
    json.dump(rb._py(rows), open(os.path.join(NEW, 'deln_chunk%d.json'
                                              % chunk), 'w'))
    print('[deln chunk %d] done' % chunk, flush=True)


def part_delnmerge():
    rows = []
    for k in range(4):
        rows += json.load(open(os.path.join(NEW, 'deln_chunk%d.json' % k)))
    disp = sorted(r['disp'] for r in rows)
    aln = sorted(r['align'] for r in rows)

    def pct(a, p):
        return a[min(len(a) - 1, int(p * len(a)))]
    summ = dict(n=len(rows), stopped=sum(1 for r in rows if r['stopped']),
                dormant=sum(1 for r in rows if r['hits'] == 0),
                disp_max=max(disp), disp_p95=pct(disp, 0.95),
                disp_p99=pct(disp, 0.99), align_max=max(aln),
                align_p95=pct(aln, 0.95),
                gmin_min=min(r['gmin'] for r in rows),
                age_left_limit_max_steps=max(r['age_left_limit_steps']
                                             for r in rows),
                age_update_max_steps=max(r['age_update_steps'] for r in rows),
                d_max_steps=D_B)
    _merge_b('deln', dict(summary=summ, rows=rows))
    print('  deln merged:', summ)


def part_adv(iters=140, seed=23):
    rng = np.random.default_rng(seed)
    N = int(rb.SIM_P['T_END'] / rb.SIM_P['dt'])
    dmax = D_B

    def sched(p):
        d = np.empty((5, N), int); d[0] = 1
        for i in range(1, 5):
            e_, l_, ts = p[i - 1]
            kk = int(ts / 1e-3)
            d[i, :kk] = e_; d[i, kk:] = l_
        return d

    def cost(p):
        S = sr2.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, delays=sched(p), rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        return S['disp'], S

    best = None
    t0 = time.time()
    for restart in range(4):
        p = [(int(rng.integers(1, dmax + 1)), int(rng.integers(1, dmax + 1)),
              float(rng.uniform(rb.B2['T_xi'] - 10, rb.B2['T_xi'] + 5)))
             for _ in range(4)]
        c, S = cost(p)
        for it in range(iters // 4 - 1):
            q = [list(x) for x in p]
            i = int(rng.integers(0, 4)); f = int(rng.integers(0, 3))
            if f < 2:
                q[i][f] = int(np.clip(q[i][f] + rng.integers(-8, 9), 1, dmax))
            else:
                q[i][2] = float(np.clip(q[i][2] + rng.uniform(-3, 3), 0.0,
                                        rb.SIM_P['T_END'] - 1))
            q = [tuple(x) for x in q]
            c2, S2 = cost(q)
            if c2 > c:
                p, c, S = q, c2, S2
        print('  adv restart %d: worst %.4f (%.0f s)' % (restart, c,
                                                       time.time() - t0),
              flush=True)
        if best is None or c > best[0]:
            best = (c, p, rb._kpis(S))
    aud = audit_schedule(sched(best[1]), N)
    _merge_b('adv', dict(worst_disp=best[0], schedule=best[1], kpis=best[2],
                         iters=iters, d_max_steps=dmax, age_audit=aud,
                         label='adversarial numerical search, not a '
                               'proven worst case'))
    print('  adversarial worst disp = %.4f' % best[0])


def part_decomp():
    ns, ds, cc, v0, e0, P, req = rx._case('B')
    G = ns(); n = G['n']
    A = bd.assemble(G, np.array(ds, float), cc, rb.B2['dbar'])
    H = rx._hop_terms(A, G, rb.B2['dbar'])
    lam, eps_e, eps_v = G['lam'], G['eps_e'], G['eps_v']
    adv = json.load(open(OUT_B))['adv']
    N = int(P['T_END'] / P['dt'])

    def sched(p):
        d = np.empty((5, N), int); d[0] = 1
        for i in range(1, 5):
            e_, l_, ts = p[i - 1]
            kk = int(ts / 1e-3)
            d[i, :kk] = e_; d[i, kk:] = l_
        return d
    runs = {}
    for tag, dl in (('constmax', None), ('adversarial', sched(adv['schedule']))):
        S = sr2.run(P, ds, v0, e0, delays=dl, rule='ab', apply_filter=True,
                    terminate_on_contact=True, extra=True)
        ts = np.array(S['tstop']); hop = np.abs(np.diff(ts))
        runs[tag] = dict(disp=S['disp'], hop_disp=[float(x) for x in hop],
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
    out = dict(bound=dict(q=[float(x) for x in H['q']],
                          epsb=[float(x) for x in H['epsb']],
                          W=[float(x) for x in H['W']],
                          anet=[float(x) for x in H['anet']],
                          Fb_prev=[float(x) for x in H['Fb_prev']],
                          vel_tol=[eps_v / a for a in H['anet']],
                          pos_tol=[lam * eps_e / np.e / a for a in H['anet']],
                          ledger=[float(w / a) for w, a in zip(H['W'], H['anet'])],
                          gaperr=[float(x) for x in gap_bound],
                          eps_e=eps_e, eps_v=eps_v, disp=A['disp']),
               runs=runs, transport_steps=D_B,
               note='realized counterparts of every term of the certified '
                    'dispersion bound (19-step transport, held age <= 20 ms)')
    _merge_x('decomp', out)
    for tag in runs:
        print('  %-11s hop |dtau| %s vs q %s' % (tag, np.round(runs[tag]['hop_disp'], 4),
                                                 np.round(H['q'], 4)))


def part_reach(restarts=4, iters=40, seed=41):
    rng = np.random.default_rng(seed)
    P = dict(rb.SIM_P); N = int(P['T_END'] / P['dt'])
    dmax = D_B
    eps_e, eps_v = rb.B2['eps_e'], rb.B2['eps_v']
    ck, save = rx._ckpt('reach')
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
            [float(rng.uniform(0.0, 0.30) * f) for f in (1.0, 0.8, 0.6, 0.4)]
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
        Ax = bd.assemble(Gx, np.array(rb.D_S, float), tuple(c0), rb.B2['dbar'])
        adm = bool(Ax['closed'] and min(Ax['margins']) > 0 and Ax['auth_ok']
                   and Ax['event_ok'] and Ax['gmin'] > 0)
        ev = dict(key=key, c0=c0, eps0=eps0, p=p, adm=adm)
        if adm:
            v0 = rb._v0_of(eps0, rb.B2['v_xi'])
            e0 = [-0.5] + [c0[i - 1] - rb.D_S[i] for i in range(1, 5)]
            S = sr2.run(P, rb.D_S, v0, e0, delays=sched(p), rule='ab',
                        apply_filter=True, terminate_on_contact=True, extra=True)
            re_ = [abs(x) / eps_e for x in S['e_hnd'][1:]]
            rv = [abs(x) / eps_v for x in S['eps_hnd'][1:]]
            ev.update(rho=float(max(max(re_), max(rv))),
                      rho_e=[float(x) for x in re_], rho_v=[float(x) for x in rv],
                      disp=S['disp'], disp_bound=Ax['disp'], hits=S['hits'],
                      stopped=S['all_stopped'],
                      egap=[float(abs(x)) for x in S['egap'][1:]])
        else:
            ev['rho'] = -1.0
        evals.append(ev); save()
        return ev

    t0 = time.time(); best = None
    for r in range(restarts):
        c0, eps0 = draw_state(); p = draw_sched()
        cur = score(c0, eps0, p); tries = 0
        while not cur['adm'] and tries < 10:
            c0, eps0 = draw_state(); cur = score(c0, eps0, p); tries += 1
        for it in range(iters):
            c1 = [float(np.clip(x + rng.normal(0, 1.0), 40.0, 50.0)) for x in c0]
            e1 = [float(np.clip(eps0[0] + rng.normal(0, 0.05), -0.35, 0.0))]
            for j, f in enumerate((1.0, 0.8, 0.6, 0.4)):
                e1.append(float(np.clip(eps0[j + 1] + rng.normal(0, 0.03), 0.0, 0.30 * f)))
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
            if it % 10 == 0:
                print('  reach r%d it%d best=%.3f (%.0f s)' % (r, it, cur['rho'],
                                                               time.time() - t0),
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
               restarts=restarts, iters=iters, seed=seed, d_max_steps=dmax,
               label='numerical reachability search over admitted takeover '
                     'states and FIFO schedules; a stress estimate, not a '
                     'proven maximum')
    _merge_x('reach', out)
    print('  reach: rho_max=%.3f (e %.3f, v %.3f), %d admitted of %d'
          % (out['rho_max'], out['rho_e_max'], out['rho_v_max'],
             len(scored), len(evals)))


def _sweep19(case, grid, n_rand, seed):
    """rx._sweep with contract-consistent simulation delays: at grid age
    dbar the constant transport is dbar - T_m and random schedules draw
    d_j <= dbar/T_m - 1, so the held age never exceeds dbar."""
    ns, ds, cc, v0, e0, P, req = rx._case(case)
    rng = np.random.default_rng(seed)
    N = int(P['T_END'] / P['dt'])
    rows = []; t0 = time.time()
    for d in grid:
        c, A, E = rx._cert(case, d)
        steps = int(round(d / P['dt'])) - 1
        Px = dict(P, dbar=steps * P['dt'])
        S = sr2.run(Px, ds, v0, e0, rule='ab', apply_filter=True,
                    terminate_on_contact=True)
        cm = rb._kpis(S)
        rr = []; aud_max = 0
        for j in range(n_rand):
            Dl = rb._random_schedule(rng, N, steps)
            aud_max = max(aud_max, audit_schedule(Dl, N)['max_left_limit_steps'])
            Sj = sr2.run(Px, ds, v0, e0, delays=Dl, rule='ab',
                         apply_filter=True, terminate_on_contact=True)
            rr.append(dict(disp=Sj['disp'], align=Sj['align'],
                           gaperr=Sj['gaperr'], gmin=float(np.min(Sj['gmin'])),
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
                    contacts=sum(1 for r in rr if r['contact'] is not None),
                    stopped=sum(1 for r in rr if r['stopped']),
                    age_left_limit_max_steps=aud_max)
        rows.append(dict(dbar=d, transport_steps=steps, cert=c, constmax=cm,
                         random=rand))
        print('  %s dbar=%.3f (transport %d) closed=%s ok=%s bound=%s | '
              'const disp=%s hits=%d | rand worst disp=%s hits=%d (%.0f s)'
              % (case, d, steps, c['closed'], c.get('ok'),
                 ('%.3f' % c['disp']) if c['closed'] else 'n/a',
                 ('%.3f' % cm['disp']) if cm['disp'] is not None else 'n/a',
                 cm['hits'], rand['disp_max'], rand['hits'],
                 time.time() - t0), flush=True)

    def first(pred):
        for r in rows:
            if pred(r):
                return r['dbar']
        return None
    summ = dict(
        first_filter_action=first(lambda r: r['constmax']['hits'] > 0
                                  or r['random']['hits'] > 0),
        first_bound_violation=first(
            lambda r: r['cert']['closed'] and r['constmax']['disp'] is not None
            and (r['constmax']['disp'] > r['cert']['disp']
                 or (r['random']['disp_max'] or 0) > r['cert']['disp'])),
        first_contact=first(lambda r: r['constmax']['contact_time'] is not None
                            or r['random']['contacts'] > 0),
        last_admitted=max([r['dbar'] for r in rows
                           if r['cert']['closed'] and r['cert']['ok']] or [None]),
        first_unclosed=first(lambda r: not r['cert']['closed']))
    return dict(grid=grid, n_rand=n_rand, seed=seed, rows=rows, summary=summ,
                contract='transport = dbar - T_m; random d_j <= dbar/T_m - 1')


def part_emergency():
    """Case A channel loss with emergency braking (unit 3 at 25 s) at the
    contract-consistent 59-step transport: heterogeneous and common-slope
    filters, and loss watchdogs; buffer crossing (g = 0) and physical
    contact (g = -s_m) located by linear interpolation."""
    TK = json.load(open(os.path.join(CODE, 'data', 'takeover_v42.json')))
    P = dict(sr2.FEATURED, dbar=D_A * 1e-3)
    kw = dict(scenario='emergency', rule='ab', apply_filter=True, record=True,
              emergency_brake=True)
    out = {}; series = {}

    def cross(t, g, level):
        idx = np.nonzero(g < level)[0]
        if len(idx) == 0:
            return None
        k = int(idx[0])
        if k == 0:
            return float(t[0])
        t0, t1, g0, g1 = t[k - 1], t[k], g[k - 1], g[k]
        return float(t0 + (level - g0) / (g1 - g0) * (t1 - t0))
    for tag, extra in (('heterogeneous', dict(filter_slopes='heterogeneous')),
                       ('common', dict(filter_slopes='common'))):
        S = sr2.run(P, TK['d_s'], TK['v'], TK['e0'], **kw, **extra)
        g4 = np.asarray(S['g'][2]); te = np.asarray(S['t'])
        out[tag] = dict(g4_min=float(g4.min()), t_buffer=cross(te, g4, 0.0),
                        t_contact=cross(te, g4, -P['s_m']))
        series[tag] = g4
        series['t'] = te
    for wd in (0.10, 0.20, 0.24, 0.25, 0.30):
        S = sr2.run(P, TK['d_s'], TK['v'], TK['e0'], filter_slopes='heterogeneous',
                    T_wd=wd, **kw)
        g4 = np.asarray(S['g'][2])
        out['watchdog_%.2f' % wd] = dict(g4_min=float(g4.min()),
                                          noncontact=bool(g4.min() > -P['s_m']),
                                          buffer_kept=bool(g4.min() > 0.0),
                                          t_buffer=cross(np.asarray(S['t']), g4, 0.0),
                                          t_contact=cross(np.asarray(S['t']), g4,
                                                          -P['s_m']))
        if wd == 0.20:
            series['watchdog_0.20'] = g4
    out['transport_steps'] = D_A; out['s_m'] = P['s_m']
    np.savez(os.path.join(NEW, 'emergency_caseA.npz'), **series)
    _merge_x('emergency', out)
    print('  emergency:', json.dumps(rb._py(out), indent=None)[:600], flush=True)


def part_caseAtraj():
    """Recorded Case-A canonical run at 59 steps (trajectory for the
    supplement figures and the persistent-entry statistic)."""
    P = dict(rx.A_SIM)
    S = sr2.run(P, rx.A_DS, rx.A_V0, rx.A_E0, rule='ab', apply_filter=True,
                record=True, terminate_on_contact=True, extra=True)
    np.savez(os.path.join(NEW, 'caseA_traj.npz'), t=S['t'], v=S['v'],
             e=S['e'], u=S['u'], h=S['h'], g=S['g'], tstop=S['tstop'],
             tact=S['tact'], tset=S['tset'], marks=S['marks'], s=S['s'],
             egap=S['egap'], d_s=np.array(rx.A_DS), v0=np.array(rx.A_V0),
             e0=np.array(rx.A_E0))
    N = int(P['T_END'] / P['dt'])
    const = np.full((5, N), D_A, int); const[0] = 1
    k = rb._kpis(S)
    k.update(mmin=[float(x) for x in S['mmin']], transport_steps=D_A,
             age_audit=audit_schedule(const, N),
             marker_signed=[float(x) for x in (S['s'] - S['marks'])],
             egap_signed=[float(x) for x in S['egap']])
    _merge_x('caseA_traj', k)
    print('  caseA traj: disp=%.4f align=%.3f gaperr=%.3f gmin=%.3f mmin=%.3f'
          % (S['disp'], S['align'], S['gaperr'], np.min(S['gmin']),
             min(S['mmin'])), flush=True)


def part_merge():
    data = {}
    for f in sorted(os.listdir(PARTDIR)):
        if f.endswith('.json') and not f.endswith('_ckpt.json'):
            data[f[:-5]] = json.load(open(os.path.join(PARTDIR, f)))
    data['_design'] = json.load(open(OUT_B))['_design']
    data['_contract'] = json.load(open(OUT_B))['_contract']
    json.dump(data, open(OUT_X, 'w'), indent=1)
    print('[merge] %s <- %s' % (os.path.basename(OUT_X), sorted(data)))



def part_caseAseeds(n_seeds=20, seed_base=400):
    """Case-A seed study under the corrected contract: 20 random FIFO
    schedules (piecewise-constant per-hop transport delays, d_j <= 59
    steps, left-limit held age audited <= 60 ms), filter applied."""
    P = dict(rx.A_SIM)
    N = int(P['T_END'] / P['dt'])
    rows = []
    t0 = time.time()
    for j in range(n_seeds):
        rng = np.random.default_rng(seed_base + j)
        Dl = rb._random_schedule(rng, N, D_A)
        aud = audit_schedule(Dl, N)
        assert aud['max_left_limit_steps'] <= D_A + 1, aud
        S = sr2.run(P, rx.A_DS, rx.A_V0, rx.A_E0, delays=Dl, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        rows.append(dict(seed=seed_base + j, disp=S['disp'], align=S['align'],
                         gaperr=S['gaperr'], gmin=float(np.min(S['gmin'])),
                         hits=S['hits'], stopped=S['all_stopped'],
                         contact=S.get('contact_time'),
                         age_left_limit_steps=aud['max_left_limit_steps']))
        print('  caseA seed %d: disp=%.4f align=%.3f gaperr=%.3f gmin=%.3f '
              'hits=%d age=%d (%.0f s)' % (seed_base + j, S['disp'], S['align'],
                                          S['gaperr'], np.min(S['gmin']),
                                          S['hits'], aud['max_left_limit_steps'],
                                          time.time() - t0), flush=True)
    summ = dict(n=len(rows), disp_max=max(r['disp'] for r in rows),
                align_max=max(r['align'] for r in rows),
                gaperr_max=max(r['gaperr'] for r in rows),
                gmin_min=min(r['gmin'] for r in rows),
                hits_total=sum(r['hits'] for r in rows),
                stopped=sum(1 for r in rows if r['stopped']),
                age_left_limit_max_steps=max(r['age_left_limit_steps'] for r in rows),
                d_max_steps=D_A, transport_steps_max=D_A)
    _merge_x('caseA_seeds', dict(rows=rows, summary=summ))
    print('  caseA seeds: %s' % summ, flush=True)



# ------------------------------------------------ tight hold, resumable
def _tight_states(n_states=300):
    D = json.load(open(rb.OUT))
    ens = D['ens']['rows']
    return [dict(c0=rb.C0, eps0=rb.B2['eps0'], tag='nominal')] + \
        [dict(c0=tuple(r['c0']), eps0=r['eps0'], tag='ens%d' % j)
         for j, r in enumerate(ens[:n_states])]


def _tight_row(st):
    """One state of run_b2_ext.part_tight, unchanged except that the
    simulator runs at the corrected transport delay (patched SIM_P)."""
    G = rb._ns(dict(eps0=[round(float(x), 6) for x in st['eps0']]))
    A = bd.assemble(G, np.array(rb.D_S, float), st['c0'], rb.B2['dbar'])
    E = bd.entry_certify(G, A['R'], rb.B2['dbar'])
    T_xi = float(np.ceil(E['T_hold'] * 100.0) / 100.0)
    Gt = rb._ns(dict(eps0=[round(float(x), 6) for x in st['eps0']],
                     T_xi=T_xi))
    At = bd.assemble(Gt, np.array(rb.D_S, float), st['c0'], rb.B2['dbar'])
    Et = bd.entry_certify(Gt, At['R'], rb.B2['dbar'])
    conds = rb._condset(At, Et)
    v0 = rb._v0_of(st['eps0'], rb.B2['v_xi'])
    e0 = [rb.B2['e0'][0]] + [st['c0'][i - 1] - rb.D_S[i] for i in range(1, 5)]
    P_ = dict(rb.SIM_P, T_xi=T_xi, T_END=T_xi + 25.0)
    S = sr2.run(P_, rb.D_S, v0, e0, rule='ab', apply_filter=True,
                terminate_on_contact=True)
    return rb._py(dict(tag=st['tag'], T_xi=T_xi, T_hold=Et['T_hold'],
                       ok=all(conds.values()),
                       failed=[k for k, v in conds.items() if not v],
                       disp_bound=At['disp'], align_bound=At['align'],
                       disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                       gmin=float(np.min(S['gmin'])), hits=S['hits'],
                       stopped=S['all_stopped'], eps_at_handoff=None))


def _tight_ckpt(k, K):
    return os.path.join(PARTDIR, 'tight_slice%dof%d_ckpt.json' % (k, K))


def part_tightslice(k, K=2):
    states = _tight_states()
    path = _tight_ckpt(k, K)
    done = json.load(open(path)) if os.path.exists(path) else {}
    t0 = time.time()
    todo = [i for i in range(k, len(states), K) if str(i) not in done]
    print('  tight slice %d/%d: %d done, %d to go' % (k, K, len(done), len(todo)),
          flush=True)
    for n, i in enumerate(todo):
        done[str(i)] = _tight_row(states[i])
        json.dump(done, open(path, 'w'))
        if n % 10 == 0:
            r = done[str(i)]
            print('  tight slice %d: state %d T_xi=%.2f disp=%.4f align=%.3f '
                  'hits=%d (%.0f s)' % (k, i, r['T_xi'], r['disp'], r['align'],
                                         r['hits'], time.time() - t0), flush=True)
    print('  tight slice %d/%d complete (%d rows)' % (k, K, len(done)), flush=True)


def part_tightmerge(K=2):
    states = _tight_states()
    rows = [None] * len(states)
    for k in range(K):
        for key, r in json.load(open(_tight_ckpt(k, K))).items():
            rows[int(key)] = r
    missing = [i for i, r in enumerate(rows) if r is None]
    assert not missing, 'tight rows missing: %s' % missing[:10]
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
    print('  tight summary:', summ, flush=True)
    _merge_x('tight', dict(rows=rows, summary=summ, transport_steps=D_B))


# ------------------------------------- control-period sweep, corrected
def part_tc19():
    """run_b2.part_tc with a contract-correct transport delay: at detector
    and step period T_c the simulator delay is (db - T_c), so the held
    command reaches age db exactly at the left limit of each delivery."""
    rows = []
    for tc in (0.001, 0.005, 0.010, 0.020, 0.050):
        db = max(rb.B2['dbar'], 2 * tc)
        try:
            G = rb._ns(dict(T_c=tc, dbar=db))
            A = rb._eval(G, dbar=db)
            E = bd.entry_certify(G, A['R'], db) if A['closed'] else None
            conds = rb._condset(A, E) if A['closed'] else {}
            ok = A['closed'] and all(conds.values())
            S = sr2.run(dict(rb.SIM_P, dt=tc, dbar=db - tc), rb.D_S, rb.V0,
                        rb.E0, rule='ab', apply_filter=True,
                        terminate_on_contact=True)
            rows.append(dict(T_c=tc, dbar=db, cert_ok=bool(ok),
                             failed=None if ok else
                             [k for k, v in conds.items() if not v] or ['ledger'],
                             disp_bound=A['disp'] if A['closed'] else None,
                             disp=S['disp'], align=S['align'],
                             gmin=float(np.min(S['gmin'])),
                             stopped=S['all_stopped'], hits=S['hits'],
                             transport_steps=int(round((db - tc) / tc)),
                             held_age_max=db))
        except Exception as ex:  # noqa: BLE001
            rows.append(dict(T_c=tc, dbar=db, cert_ok=False,
                             failed=['error:%s' % ex]))
        print('  T_c=%g: %s' % (tc, rows[-1]), flush=True)
    _merge_b('tc', rows, path=OUT_FG)


# --------------------------------------- certificate-only lambda scan
def part_lamscan(lo=0.030, hi=0.090, step=0.0025, tol=2.5e-4):
    """Final revision plan D-16: the critically damped family
    beta_i = lam^2, gamma_i = 2 lam at the Case B geometry and age bound
    (certificate only, no simulation).  A grid records the admitted set;
    bisection then brackets the largest admitted lam above the design
    value 0.055 to tol."""
    def evaluate(L):
        try:
            G = rb._ns(dict(lam=L, beta=L * L, gamma=2 * L))
            A = rb._eval(G)
        except Exception as ex:  # noqa: BLE001
            return dict(lam=L, ok=False, failed=['error:%s' % ex])
        if not A['closed']:
            return dict(lam=L, ok=False, failed=['ledger'])
        E = bd.entry_certify(G, A['R'], rb.B2['dbar'])
        c = rb._condset(A, E)
        slack = [float(x) for x in A['auth_slack']]
        return dict(lam=L, ok=bool(all(c.values())), failed=[k for k, v in c.items() if not v],
                    margin_min=float(min(A['margins'])), auth_slack=slack,
                    auth_unit=int(np.argmin(slack)) + 1, disp=A['disp'])
    t0 = time.time()
    grid = []
    for L in np.arange(lo, hi + 1e-12, step):
        r = evaluate(round(float(L), 6))
        grid.append(r)
        print('  lam=%.4f ok=%s failed=%s (%.0f s)' % (r['lam'], r['ok'], r['failed'],
                                                     time.time() - t0), flush=True)
    base = rb.B2['lam']
    above = [r for r in grid if r['lam'] >= base - 1e-12]
    k = next((j for j, r in enumerate(above) if not r['ok']), None)
    assert above[0]['ok'] and k is not None, 'no rejection above the design lambda on the grid'
    a, b = above[k - 1]['lam'], above[k]['lam']
    while b - a > tol:
        m = round(0.5 * (a + b), 6)
        if evaluate(m)['ok']:
            a = m
        else:
            b = m
    first = evaluate(b)
    _merge_x('lamscan', dict(grid=grid, lam_max=a, lam_first_fail=b,
                             first_fail=first['failed'], first_fail_auth_unit=first.get('auth_unit'),
                             lam_min_grid=min(r['lam'] for r in grid if r['ok']),
                             family='beta_i = lam^2, gamma_i = 2 lam', case='B',
                             dbar=rb.B2['dbar'], tol=tol,
                             note='certificate only; plan D-16'))
    print('  lamscan: largest admitted lam %.4f, first rejected %.4f fails %s (unit %s)'
          % (a, b, first['failed'], first.get('auth_unit')), flush=True)


# ------------------------------------------- Case A step refinement
def part_caseA_conv():
    """Final revision plan D-17: Case A at integration steps of 1, 0.5,
    and 0.25 ms.  The 59 ms transport delay stays fixed in seconds, so the
    held age stays below the 60 ms contract while the message period is
    refined with the step."""
    P = dict(rx.A_SIM)
    rows = {}
    t0 = time.time()
    for dt in (1e-3, 5e-4, 2.5e-4):
        S = sr2.run(dict(P, dt=dt), rx.A_DS, rx.A_V0, rx.A_E0, rule='ab',
                    apply_filter=True, terminate_on_contact=True)
        rows['%g' % dt] = dict(
            dt=dt, transport_steps=int(round(P['dbar'] / dt)),
            disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
            gmin=[float(x) for x in S['gmin']],
            tstop=[None if not np.isfinite(x) else float(x) for x in S['tstop']],
            tset=[None if not np.isfinite(x) else float(x) for x in S['tset']],
            upk=[float(x) for x in S['upk']], hits=S['hits'], infeas=S['infeas'])
        print('  caseA dt=%g: disp=%.4f align=%.4f gaperr=%.4f gmin=%.3f hits=%d '
              'infeas=%d (%.0f s)' % (dt, S['disp'], S['align'], S['gaperr'],
                                      np.min(S['gmin']), S['hits'], S['infeas'],
                                      time.time() - t0), flush=True)
    ref = rows['0.001']
    dev = {k: max(abs(r[k] - ref[k]) for r in rows.values())
           for k in ('disp', 'align', 'gaperr')}
    dev['gmin'] = max(abs(min(r['gmin']) - min(ref['gmin'])) for r in rows.values())
    _merge_x('caseA_conv', dict(rows=rows, max_dev=dev, transport_s=P['dbar'],
                                note='plan D-17; transport delay fixed in seconds'))
    print('  caseA_conv max deviation from 1 ms: %s' % dev, flush=True)


PARTS = dict(seed=part_seed, sim=part_sim, base=rb.part_base,
             cacc=rx.part_cacc, review=rx.part_review, act=rx.part_act,
             vthr=rx.part_vthr, loss=rb.part_loss, caseA=rx.part_caseA,
             emergency=part_emergency, caseAtraj=part_caseAtraj,
             ensmerge=part_ensmerge,
             delnmerge=part_delnmerge, adv=part_adv, decomp=part_decomp,
             reach=part_reach, tight=rx.part_tight, merge=part_merge,
             plant=rx.part_plant, plant2=rx.part_plant2, mpc=rx.part_mpc,
             caseAseeds=part_caseAseeds,
             tightmerge=part_tightmerge, tc=part_tc19,
             lamscan=part_lamscan, caseA_conv=part_caseA_conv,
             gaps=rb.part_gaps, het=rb.part_het, hard=rb.part_hard, hifi=rb.part_hifi,
             sweepB=lambda: _merge_x('sweepB', _sweep19('B', rx.SWEEP_B, 20, 301)),
             sweepA=lambda: _merge_x('sweepA', _sweep19('A', rx.SWEEP_A, 10, 302)))

if __name__ == '__main__':
    part = sys.argv[1]
    t0 = time.time()
    if part == 'ens':
        part_ens(int(sys.argv[2]))
    elif part == 'deln':
        part_deln(int(sys.argv[2]))
    elif part == 'tightslice':
        part_tightslice(int(sys.argv[2]), int(sys.argv[3]))
    elif part == 'tightmerge':
        part_tightmerge(int(sys.argv[2]) if len(sys.argv) > 2 else 2)
    else:
        PARTS[part]()
    print('== %s finished in %.0f s' % (part, time.time() - t0), flush=True)
