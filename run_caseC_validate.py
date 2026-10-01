#!/usr/bin/env python3
"""run_caseC_validate.py -- selection and validation of Case C (crawl
profile of Assumption 1', 2026-09-26).

Certificate: src/certification/certify_profile.py (extended evaluator) and
interval_profile.py.  Simulator: src/simulation/sim_cont.py with the
reference-speed profile, driven through run_caseC.py (build, simulate,
kpis, record, trajectory).  Every output goes to data/new_d19/caseC/.

Parts
  design    the selected design of the late-handoff extension
            (run_caseC_ext.part_select: frontier of scan_compact.json,
            scan_ext.json and scan_ext_low.json, the 1 s handoff
            refinement, the full certificate record with the age
            crossings, the 1 ms boundary and the interval enclosures)
            -> design.json; the first selection at lambda = 0.055 is
            archived in data/new_d19/caseC_prev/
  ref       continuous reference at h = 25 us (T_m = T_c = 1 ms,
            transport 19 ms) with every KPI, the certified bound beside
            each observed value and the phase times
            -> reference.json, traj_caseC.npz
  conv      h in {1e-4, 5e-5, 2.5e-5} (plus 1e-3, 5e-4 diagnostic), same
            event schedule -> convergence.json
  sched     300 random FIFO schedules (seed 500 + j, d_j <= 19 steps),
            'cont' at h = 1 ms, held-age audit -> schedules.json
  sweep     age bounds {5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200,
            250, 300} ms (250 and 300 ms added on 2026-09-27 so that the
            sweep reaches the first failing requirement; the rows of the
            other ages are unchanged): certificate per condition, constant transport dbar - T_m
            at h = 25 us, 10 random schedules at h = 1 ms; per-condition
            1 ms-grid boundaries and the run_b2 bisection crossings
            -> sweep.json
  ens       100 takeover states around the design (clearances in
            [c0 - 3, c0 + 3] m, mismatches as the Case B ensemble):
            certificate and continuous run each -> ensemble.json
  digital   zero-order hold T_u = 1 ms (and 0.5, 0.25, 0.1 ms) against the
            continuous reference -> digital.json
  check     sanity over every file: observed indices inside the certified
            bounds, filter never modified a command in an admitted run
            -> check.json
  all       design ref conv sched sweep ens digital check

Usage: python run_caseC_validate.py [part ...]
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import numpy as np                                      # noqa: E402
import certify_profile as cp                            # noqa: E402
import interval_profile as ip                           # noqa: E402
import run_b2 as rb                                     # noqa: E402
import run_cont as rc                                   # noqa: E402
import run_caseC as rcc                                 # noqa: E402

OUT = rcc.OUT
T_M = rcc.T_M
H_FINE = rcc.H_FINE
H_SCHED = rcc.H_SCHED
DBARS_MS = (5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200, 250, 300)
TU_DIG = (1e-3, 5e-4, 2.5e-4, 1e-4)

# ---------------------------------------------------------------- design
# Case C design (Case B constants except T_xi, t_d, a_d, v_c1, c0)
DESIGN = dict(rcc.PROVISIONAL, name='caseC', T_xi=95.0, t_d=10.0, a_d=0.40,
              v_c1=1.0, lam=0.055, c0=[7.0, 7.0, 7.0, 7.0], T_END=None)

CAND_AD = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60)
CAND_C0 = (7.0, 7.1, 7.2, 7.25, 7.5)


def _py(o):
    return rc._py(o)


def _fin(xs):
    """Finite entries of an iterable (None and NaN dropped)."""
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def _max(xs):
    f = _fin(xs)
    return max(f) if f else None


def _min(xs):
    f = _fin(xs)
    return min(f) if f else None


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(_py(obj), open(p, 'w'), indent=1)
    print('  -> %s' % os.path.relpath(p, CODE), flush=True)
    return p


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def design_cfg():
    d = load('design.json')
    return dict(d['cfg']) if d else dict(DESIGN)


def cert_cfg(cfg, **kw):
    """certify_profile configuration of a run_caseC design.  The design
    keys added on 2026-09-26 (vhnd: handoff pad V^hnd, d_req: dispersion
    requirement, refine: refinements (R1), (R2)) are passed when present;
    their defaults are those of certify_profile.config (1e-3, 0.9 s,
    True), so a design without them is evaluated as before."""
    c = cp.config(n=int(cfg['n']), v_xi=float(cfg['v_xi']),
                  a_b=float(cfg['a_b']), alpha=float(cfg['alpha']),
                  lam=float(cfg['lam']), s_m=float(cfg['s_m']),
                  dbar=float(cfg['dbar']), eps_e=float(cfg['eps_e']),
                  eps_v=float(cfg['eps_v']),
                  eps0=[float(x) for x in cfg['eps0']],
                  d_s=[float(x) for x in cfg['d_s']],
                  amax=[float(x) for x in cfg['amax']],
                  e0_head=float(cfg['e0_head']), T_c=rcc.T_C,
                  eps_det=float(cfg['eps_det']), phi=float(cfg['phi']),
                  c0=tuple(float(x) for x in cfg['c0']),
                  T_xi=float(cfg['T_xi']), t_d=float(cfg['t_d']),
                  a_d=float(cfg['a_d']), v_c1=float(cfg['v_c1']),
                  vhnd=float(cfg.get('vhnd', 1e-3)),
                  d_req=(None if cfg.get('d_req') is None
                         else float(cfg['d_req'])),
                  refine=bool(cfg.get('refine', True)))
    c.update(kw)
    return c


def certificate(cfg, **kw):
    """evaluate() of the design (kw: constant overrides such as dbar)."""
    return cp.evaluate(cert_cfg(cfg, **kw))


# ------------------------------------------------------- bounds table
def bounds_table(ev, k, R=None, rec=None, eps_e=0.15, eps_v=0.02):
    """Certified bound beside each observed value.  ev: evaluate()
    result; k: run_caseC.kpis; R: raw run (per-pair arrays); rec: the
    decimated trajectory (velocity floor)."""
    K = ev['consts']
    n = len(K['dem'])
    rows = {}

    def row(name, bound, obs, kind, unit, note=None):
        b = None if bound is None else float(bound)
        o = None if obs is None else float(obs)
        ok = None
        if b is not None and o is not None and np.isfinite(o):
            ok = bool(o <= b) if kind == 'upper' else bool(o >= b)
        rows[name] = dict(bound=b, observed=o, kind=kind, unit=unit, ok=ok)
        if note:
            rows[name]['note'] = note

    def rowlist(name, bounds, obs, kind, unit, note=None):
        bb = [None if x is None else float(x) for x in bounds]
        oo = [None if x is None else float(x) for x in obs]
        oks = []
        for b, o in zip(bb, oo):
            if b is None or o is None or not np.isfinite(o):
                oks.append(None)
            else:
                oks.append(bool(o <= b) if kind == 'upper' else bool(o >= b))
        rows[name] = dict(bound=bb, observed=oo, kind=kind, unit=unit,
                          ok=all(x in (True, None) for x in oks), ok_each=oks)
        if note:
            rows[name]['note'] = note

    T_xi = ev['case']['T_xi']
    row('dispersion', K['disp'], k['disp'], 'upper', 's')
    row('schedule_error', K['sched'], k['sched'], 'upper', 's',
        'max_i |t_stop,i - tau_r|')
    row('final_standstill', K['T_f_bar'], k['T_f'], 'upper', 's')
    rowlist('stop_times', [K['T_f_bar']] * n, k['tstop'], 'upper', 's')
    row('marker_error', K['align'], k['align'], 'upper', 'm',
        'max_i |s_i(T_f) - marker_i|')
    row('terminal_gap_error', max(K['gaperr']), k['gaperr'], 'upper', 'm')
    rowlist('terminal_gap_error_pairs', K['gaperr'],
            [abs(x) for x in k['egap'][1:]], 'upper', 'm', 'pairs 2..n')
    row('clearance_floor', K['gmin'], k['gmin_min'], 'floor', 'm',
        'min over pairs and time of g_i = c_i - s_m (intersample minimum)')
    fl = [min(x for x in (K['floors'][str(i)]['g_init'],
                          K['floors'][str(i)]['bf'],
                          K['floors'][str(i)]['gspd']) if x is not None)
          for i in range(1, n)]
    if R is not None:
        rowlist('clearance_floor_pairs', fl, list(R['gmin']), 'floor', 'm')
    amax = np.asarray(K['Umin'], float)
    b = amax / cp.FIXED['v_c']
    sig = [cp.FIXED['kappa'] * b[i] * K['pairs'][str(i)]['padded']
           for i in range(1, n)]
    row('inactivity_margin', min(sig), k['mmin_lim_min'], 'floor', 'm/s^2',
        'certified min_i kappa b_i (h_i(0) - thr_i - 0.01) vs observed min '
        'of u_cbf - u_nom incl. event left limits')
    if R is not None:
        rowlist('inactivity_margin_pairs', sig, list(R['mmin_lim']), 'floor',
                'm/s^2')
    row('inactivity_margin_positive', 0.0, k['mmin_lim_min'], 'floor', 'm/s^2')
    row('persistent_entry', K['T_ent_bar'], k['entry_obs'], 'upper', 's',
        'observed entry of all units into the handoff box vs certified '
        'T_ent_bar')
    row('entry_before_deadline', T_xi - ev['case']['dbar'], k['entry_obs'],
        'upper', 's')
    row('handoff_position_error', eps_e, max(abs(x) for x in k['e_hnd']),
        'upper', 'm', 'max_i |e_i(T_xi)| vs eps_e')
    row('handoff_velocity_error', eps_v, max(abs(x) for x in k['eps_hnd']),
        'upper', 'm/s', 'max_i |eps_i(T_xi)| vs eps_v')
    rowlist('braking_mismatch_integral', K['W'], k['wint'], 'upper', 'm/s',
            'int |u*_{i-1} - uhat_{i-1}| over the braking window vs ledger W_i')
    rowlist('braking_speed_error', K['epsb'], k['epsmax_b'], 'upper', 'm/s',
            'max |eps_i| in the braking window vs epsbar_b,i')
    rowlist('post_receipt_envelope', K['Fb'][1:], k['fb_real'], 'upper',
            'm/s^2', 'max |u*_{i-1} + a_b| after receipt vs F^b_i')
    rowlist('peak_command_vs_demand', K['dem'], k['upk'], 'upper', 'm/s^2',
            'max |u_i| vs certified authority demand U_all,i')
    rowlist('peak_command_vs_capability', K['Umin'], k['upk'], 'upper',
            'm/s^2')
    rowlist('s1_switch_deadline', K['tset'], k['tset'], 'upper', 's',
            'observed S2 switch t_set,i vs certified deadline')
    rowlist('activation_deadline', K['t_bar'], k['tact'], 'upper', 's')
    row('segment_after_last_switch', ev['case']['t_d'], max(k['tset']),
        'upper', 's', 't_d >= observed last S1 switch')
    row('segment_after_certified_switch', ev['case']['t_d'], K['T_sw_bar'],
        'upper', 's', 't_d >= T_sw_bar')
    if rec is not None:
        t = np.asarray(rec['t'], float)
        v = np.asarray(rec['v'], float)
        m = t <= T_xi + 1e-9
        vmin = float(np.min(v[:, m]))
        row('velocity_floor', K['vflr'], vmin, 'floor', 'm/s',
            'min over units and the 10 ms record up to T_xi vs the '
            'time-resolved certified floor')
        jx = int(np.argmin(np.abs(t - T_xi)))
        row('handoff_speed_floor', K['vflr_c'], float(np.min(v[:, jx])),
            'floor', 'm/s', 'min_i v_i(T_xi) vs v_c1 - n eps_v')
        # crawl phase: speeds within v_c1 +- envelope (observed range)
        mc = (t >= ev['case']['t_c']) & (t <= T_xi)
        if mc.any():
            rows['crawl_speed_range'] = dict(
                observed=[float(np.min(v[:, mc])), float(np.max(v[:, mc]))],
                v_c1=ev['case']['v_c1'], unit='m/s', kind='info')
    row('filter_modifications', 0, k['n_mod'] + k['n_fallback'] + k['n_sat'],
        'upper', 'count')
    ok = all(r.get('ok') in (True, None) for r in rows.values())
    return dict(rows=rows, all_ok=ok,
                n_checked=sum(1 for r in rows.values() if r.get('ok') is not None),
                n_failed=sum(1 for r in rows.values() if r.get('ok') is False))


def short_bounds(ev, k):
    """The scalar comparisons used for the ensemble, schedule and sweep
    rows (no raw run)."""
    K = ev['consts']
    n = len(K['dem'])
    amax = np.asarray(K['Umin'], float)
    b = amax / cp.FIXED['v_c']
    sig = min(cp.FIXED['kappa'] * b[i] * K['pairs'][str(i)]['padded']
              for i in range(1, n))
    ent = k['entry_obs']
    eps_e = float(k['config']['eps_e']); eps_v = float(k['config']['eps_v'])
    chk = dict(
        dispersion=(K['disp'], k['disp'], 'upper'),
        schedule_error=(K['sched'], k['sched'], 'upper'),
        final_standstill=(K['T_f_bar'], k['T_f'], 'upper'),
        marker_error=(K['align'], k['align'], 'upper'),
        terminal_gap_error=(max(K['gaperr']), k['gaperr'], 'upper'),
        clearance_floor=(K['gmin'], k['gmin_min'], 'floor'),
        inactivity_margin=(sig, k['mmin_lim_min'], 'floor'),
        persistent_entry=(K['T_ent_bar'], ent, 'upper'),
        handoff_position_error=(eps_e, max(abs(x) for x in k['e_hnd']), 'upper'),
        handoff_velocity_error=(eps_v, max(abs(x) for x in k['eps_hnd']), 'upper'),
        braking_mismatch_integral=(min(K['W']), None, 'list'),
        peak_command=(0.0, min(np.asarray(K['dem']) - np.asarray(k['upk'])), 'floor'),
        filter_modifications=(0, k['n_mod'] + k['n_fallback'] + k['n_sat'], 'upper'))
    # list-valued checks
    wint_ok = all(o <= bb for o, bb in zip(k['wint'], K['W']))
    epsb_ok = all(o <= bb for o, bb in zip(k['epsmax_b'], K['epsb']))
    fb_ok = all(o <= bb for o, bb in zip(k['fb_real'], K['Fb'][1:]))
    out = {}
    for name, (bnd, obs, kind) in chk.items():
        if kind == 'list':
            continue
        o = None if obs is None else float(obs)
        ok = None
        if o is not None and np.isfinite(o):
            ok = bool(o <= bnd) if kind == 'upper' else bool(o >= bnd)
        out[name] = dict(bound=float(bnd), observed=o, ok=ok)
    out['braking_mismatch_integral'] = dict(ok=bool(wint_ok), bound=K['W'],
                                            observed=k['wint'])
    out['braking_speed_error'] = dict(ok=bool(epsb_ok), bound=K['epsb'],
                                      observed=k['epsmax_b'])
    out['post_receipt_envelope'] = dict(ok=bool(fb_ok), bound=K['Fb'][1:],
                                        observed=k['fb_real'])
    out['all_stopped'] = dict(ok=bool(k['all_stopped']))
    all_ok = all(r['ok'] in (True, None) for r in out.values())
    return dict(rows=out, all_ok=all_ok)


def cert_summary(ev):
    """Compact certificate record."""
    K = ev['consts']
    return dict(admitted=ev['admitted'], failed=ev['failed'],
                conds=ev['conds'],
                T_ent_bar=K['T_ent_bar'], T_hold=K['T_hold'],
                T_sw_bar=K['T_sw_bar'], T_f_bar=K['T_f_bar'],
                T_brk=K['T_brk'], disp=K['disp'], sched=K['sched'],
                late=K['late'], align=K['align'], gaperr=K['gaperr'],
                gmin=K['gmin'], margins=K['margins'],
                auth_slack=K['auth_slack'], dem=K['dem'],
                vflr=K['vflr'], vflr_time_argmin=K['vflr_time_argmin'],
                vfloor_pre=K['vfloor_pre'], vflr_c=K['vflr_c'],
                Tent=K['Tent'], Vhnd_max=K['Vhnd_max'],
                tset=K['tset'], t_bar=K['t_bar'], tau_bar=K['tau_bar'],
                W=K['W'], epsb=K['epsb'], Fb=K['Fb'],
                Lam1=K['Lam1'], Lam2=K['Lam2'], Lam2_pre=K['Lam2_pre'],
                ff_charges=K['ff_charges'], reference=ev['reference'])


# ------------------------------------------------------ age boundaries
def cond_boundary_ms(cfg, names=None, kmax=450):
    """Per condition: the largest integer k with the condition holding at
    dbar = k ms (bisection on the 1 ms grid, assuming monotonicity; the
    neighbours k and k + 1 are verified and a violation of monotonicity
    at the verification is reported)."""
    names = names or list(cp.COND_NAMES)
    memo = {}

    def ev(k):
        if k not in memo:
            memo[k] = certificate(cfg, dbar=k * 1e-3)
        return memo[k]

    def ok(nm, k):
        r = ev(k)
        return bool(r['closed'] and r['conds'][nm]['ok'])

    out = {}
    for nm in names:
        if not ok(nm, 1):
            out[nm] = dict(k_ms=0, note='fails at 1 ms')
            continue
        if ok(nm, kmax):
            out[nm] = dict(k_ms=kmax, note='holds at kmax')
            continue
        lo, hi = 1, kmax
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(nm, mid):
                lo = mid
            else:
                hi = mid
        out[nm] = dict(k_ms=lo, verified=bool(ok(nm, lo) and not ok(nm, lo + 1)),
                       slack_at_k=ev(lo)['conds'][nm]['slack'],
                       slack_at_k1=ev(lo + 1)['conds'][nm]['slack'])
    return out


# ------------------------------------------------------------- parts
def part_design():
    """Selection of the revised Case C (authors' clarification of
    2026-09-26): run_caseC_ext.part_select writes design.json with the
    run_caseC configuration (keys of PROVISIONAL plus vhnd, d_req and
    refine), the scan row, the 1 s handoff refinement, the runners-up,
    the frontier per clearance and the full certificate record."""
    import run_caseC_ext as rce
    return rce.part_select()


def part_ref():
    cfg = design_cfg()
    B = rcc.build(cfg)
    ev = certificate(cfg)
    print('[ref] h=%g transport=%g s profile=%s' % (H_FINE, B['transport'],
                                                     B['profile']), flush=True)
    R = rcc.simulate(B, h=H_FINE)
    k = rcc.kpis(R, B)
    print('  ' + rcc.summary(k), flush=True)
    rec = rcc.record(B, R)
    T = rcc.trajectory(B, R)
    bt = bounds_table(ev, k, R=R, rec=T, eps_e=B['eps_e'], eps_v=B['eps_v'])
    for name, r in bt['rows'].items():
        if r.get('kind') == 'info':
            continue
        print('  %-32s bound %-22s observed %-22s ok=%s'
              % (name, r['bound'] if not isinstance(r['bound'], list)
                 else '[%s]' % ', '.join('%.4g' % x for x in r['bound']),
                 r['observed'] if not isinstance(r['observed'], list)
                 else '[%s]' % ', '.join('%.4g' % x for x in r['observed']),
                 r['ok']), flush=True)
    print('  all bounds ok:', bt['all_ok'], flush=True)
    rec['certificate'] = cert_summary(ev)
    rec['bounds'] = bt
    rec['phase_times'] = dict(t_d=ev['case']['t_d'], t_c=ev['case']['t_c'],
                              T_xi=ev['case']['T_xi'], tau_r=R['tau_r'],
                              tstop=R['tstop'], T_f=R['T_f'],
                              T_f_bar=ev['consts']['T_f_bar'],
                              tset_obs=R['tset'], tact_obs=R['tact'],
                              trule_obs=R['trule'],
                              tset_cert=ev['consts']['tset'],
                              T_sw_bar=ev['consts']['T_sw_bar'],
                              entry_obs=R['entry_obs'],
                              T_ent_bar=ev['consts']['T_ent_bar'],
                              s0_Txi=ev['reference']['s0_Txi'],
                              s_ref_stop=ev['reference']['s_ref_stop'],
                              markers=ev['reference']['markers'])
    # speed extrema per phase (10 ms record)
    t = np.asarray(T['t']); v = np.asarray(T['v'])
    ph = {}
    for name, (a, bnd) in dict(approach=(0.0, ev['case']['t_d']),
                               deceleration=(ev['case']['t_d'], ev['case']['t_c']),
                               crawl=(ev['case']['t_c'], ev['case']['T_xi']),
                               braking=(ev['case']['T_xi'], R['tau_r'] + 1.0)).items():
        m = (t >= a) & (t <= bnd)
        ph[name] = dict(t=[a, bnd], v_min=float(np.min(v[:, m])),
                        v_max=float(np.max(v[:, m])),
                        eps_max=float(np.max(np.abs(np.asarray(T['eps'])[:, m]))),
                        e_max=float(np.max(np.abs(np.asarray(T['e'])[:, m]))),
                        u_min=float(np.min(np.asarray(T['u'])[:, m])),
                        u_max=float(np.max(np.asarray(T['u'])[:, m])),
                        c_min=float(np.min(np.asarray(T['c'])[:, m])),
                        c_max=float(np.max(np.asarray(T['c'])[:, m])))
    rec['phase_extrema'] = ph
    # follower speeds: the followers close their gaps during the
    # deceleration and the crawl, so they run faster than the head; per
    # unit the largest speed in each phase, the largest speed difference
    # to the head and to the predecessor, and the fraction of the phase in
    # which every follower runs faster than the head (10 ms record)
    fs = {}
    for name in ('deceleration', 'crawl'):
        a, bnd = ph[name]['t']
        m = (t >= a) & (t <= bnd)
        vv = v[:, m]
        dv_head = vv[1:, :] - vv[0:1, :]
        dv_pred = vv[1:, :] - vv[:-1, :]
        jmax = int(np.argmax(vv[1:, :].max(axis=0)))
        fs[name] = dict(
            t=[a, bnd], v_head_max=float(np.max(vv[0])),
            v_head_min=float(np.min(vv[0])),
            v_follower_max=float(np.max(vv[1:, :])),
            v_follower_max_per_unit=vv[1:, :].max(axis=1).tolist(),
            t_follower_max=float(t[m][jmax]),
            dv_to_head_max_per_unit=dv_head.max(axis=1).tolist(),
            dv_to_head_min_per_unit=dv_head.min(axis=1).tolist(),
            dv_to_predecessor_max_per_unit=dv_pred.max(axis=1).tolist(),
            dv_to_predecessor_min_per_unit=dv_pred.min(axis=1).tolist(),
            fraction_all_followers_faster_than_head=float(
                np.mean(np.all(dv_head > 0.0, axis=0))),
            fraction_each_faster_than_head=np.mean(dv_head > 0.0, axis=1).tolist(),
            v_r_range=[float(np.min(np.asarray(T['v_r'])[m])),
                       float(np.max(np.asarray(T['v_r'])[m]))],
            v_c1=ev['case']['v_c1'],
            excess_over_v_c1_max=float(np.max(vv) - ev['case']['v_c1']))
        print('  %-12s head %.4f..%.4f m/s, followers <= %.4f m/s (at %.2f s), '
              'dv to head max %s, faster than head %.3f of the phase'
              % (name, fs[name]['v_head_min'], fs[name]['v_head_max'],
                 fs[name]['v_follower_max'], fs[name]['t_follower_max'],
                 ['%.4f' % x for x in fs[name]['dv_to_head_max_per_unit']],
                 fs[name]['fraction_all_followers_faster_than_head']), flush=True)
    rec['follower_speeds'] = fs
    dump('reference.json', rec)
    rcc.save_traj(os.path.join(OUT, 'traj_caseC.npz'), B, R)
    return rec


def part_conv():
    cfg = design_cfg()
    out = rcc.convergence(cfg, save=False)
    ev = certificate(cfg)
    for row in out['rows']:
        row['bounds'] = short_bounds(ev, row['kpis'])
    out['all_rows_inside_bounds'] = all(r['bounds']['all_ok'] for r in out['rows'])
    out['all_filter_inactive'] = all(r['kpis']['filter_inactive'] for r in out['rows'])
    dump('convergence.json', out)


def part_sched(n=300, seed0=500):
    cfg = design_cfg()
    B = rcc.build(cfg)
    ev = certificate(cfg)
    dmax = B['dmax_steps']
    Nsched = int(round(B['P']['T_END'] / T_M)) + 2
    print('[sched] n=%d seeds %d.. d_j <= %d steps h=%g' % (n, seed0, dmax, H_SCHED),
          flush=True)
    rows = []
    t0 = time.time()
    for j in range(n):
        rng = np.random.default_rng(seed0 + j)
        Dl = rb._random_schedule(rng, Nsched, dmax)
        R = rcc.simulate(B, h=H_SCHED, fifo=Dl)
        k = rcc.kpis(R, B)
        age_ok = R['age']['max_left_limit'] <= B['age_bound'] + 1e-12
        sb = short_bounds(ev, k)
        rows.append(dict(index=j, seed=seed0 + j, kpis=k, age_audit=R['age'],
                         age_ok=bool(age_ok), bounds_ok=sb['all_ok'],
                         bounds_failed=[nm for nm, r in sb['rows'].items()
                                        if r['ok'] is False],
                         schedule_summary=dict(d_min=int(Dl[1:].min()),
                                               d_max=int(Dl[1:].max()),
                                               d_mean=float(Dl[1:].mean()))))
        if (j + 1) % 25 == 0:
            print('  %3d %s age_sup=%.4f bounds_ok=%s (%.0f s)'
                  % (j + 1, rcc.summary(k), R['age']['max_left_limit'],
                     sb['all_ok'], time.time() - t0), flush=True)
    K = [r['kpis'] for r in rows]

    def ext(key, f=max):
        return (_max if f is max else _min)(k[key] for k in K)
    agg = dict(
        n=n, seed0=seed0, dmax_steps=dmax, h=H_SCHED,
        disp=dict(min=ext('disp', min), max=ext('disp'), bound=ev['consts']['disp']),
        marker_error=dict(min=ext('align', min), max=ext('align'),
                          bound=ev['consts']['align']),
        gap_error=dict(min=ext('gaperr', min), max=ext('gaperr'),
                       bound=max(ev['consts']['gaperr'])),
        sched=dict(max=ext('sched'), bound=ev['consts']['sched']),
        gmin=dict(min=ext('gmin_min', min), max=ext('gmin_min'),
                  bound=ev['consts']['gmin']),
        mmin_lim=dict(min=ext('mmin_lim_min', min), max=ext('mmin_lim_min')),
        entry_obs=dict(min=ext('entry_obs', min), max=ext('entry_obs'),
                       bound=ev['consts']['T_ent_bar']),
        T_f=dict(max=ext('T_f'), bound=ev['consts']['T_f_bar']),
        e_hnd_max=max(max(abs(x) for x in k['e_hnd']) for k in K),
        eps_hnd_max=max(max(abs(x) for x in k['eps_hnd']) for k in K),
        n_mod_total=sum(k['n_mod'] for k in K),
        n_fallback_total=sum(k['n_fallback'] for k in K),
        n_sat_total=sum(k['n_sat'] for k in K),
        all_filter_inactive=all(k['filter_inactive'] for k in K),
        all_stopped=all(k['all_stopped'] for k in K),
        all_age_ok=all(r['age_ok'] for r in rows),
        age_sup_max=max(r['age_audit']['max_left_limit'] for r in rows),
        age_sup_min=min(r['age_audit']['max_left_limit'] for r in rows),
        age_bound=B['age_bound'],
        all_bounds_ok=all(r['bounds_ok'] for r in rows),
        n_bounds_failed=sum(1 for r in rows if not r['bounds_ok']))
    print('  aggregate: disp %.4f-%.4f (bound %.4f) align %.4f-%.4f (%.4f) '
          'gaperr %.4f-%.4f (%.4f) gmin >= %.4f (%.4f) mmin >= %.4f entry '
          '%.3f-%.3f (%.3f) T_f <= %.4f (%.4f) n_mod %d age_sup %.4f-%.4f '
          'all_bounds_ok %s'
          % (agg['disp']['min'], agg['disp']['max'], agg['disp']['bound'],
             agg['marker_error']['min'], agg['marker_error']['max'],
             agg['marker_error']['bound'], agg['gap_error']['min'],
             agg['gap_error']['max'], agg['gap_error']['bound'],
             agg['gmin']['min'], agg['gmin']['bound'], agg['mmin_lim']['min'],
             agg['entry_obs']['min'], agg['entry_obs']['max'],
             agg['entry_obs']['bound'], agg['T_f']['max'], agg['T_f']['bound'],
             agg['n_mod_total'], agg['age_sup_min'], agg['age_sup_max'],
             agg['all_bounds_ok']), flush=True)
    out = dict(study=('random FIFO schedules: piecewise-constant per-hop '
                      'transport (segments 0.5-5 s, d_j uniform in 1..%d steps '
                      'of T_m = 1 ms, run_b2._random_schedule with one '
                      'generator per schedule, seed 500 + j), FIFO projection '
                      'at the receiver, held age <= 20 ms audited; continuous '
                      'law at h = %g s (RK4)' % (dmax, H_SCHED)),
               cfg=cfg, certificate=cert_summary(ev), aggregate=agg, rows=rows)
    dump('schedules.json', out)


def part_sweep(dbars_ms=DBARS_MS, n_sched=10, seed0=700):
    cfg = design_cfg()
    B = rcc.build(cfg)
    print('[sweep] age bounds %s ms' % list(dbars_ms), flush=True)
    rows = []
    t0 = time.time()
    for steps in dbars_ms:
        db = steps * 1e-3
        ev = certificate(cfg, dbar=db)
        tr = (steps - 1) * T_M
        R = rcc.simulate(B, h=H_FINE, transport=tr)
        k = rcc.kpis(R, B)
        sb = short_bounds(ev, k) if ev['closed'] else None
        row = dict(age_bound=db, transport=tr, transport_steps=steps - 1,
                   certificate=dict(admitted=ev['admitted'], failed=ev['failed'],
                                    closed=ev['closed'],
                                    conds={nm: dict(ok=c['ok'], slack=c['slack'])
                                           for nm, c in ev['conds'].items()},
                                    summary=cert_summary(ev) if ev['closed'] else None),
                   kpis=k, age_audit=R['age'],
                   bounds=sb, random=[])
        print('  dbar=%3d ms cert admitted=%s failed=%s | const: %s bounds_ok=%s (%.1f s)'
              % (steps, ev['admitted'], ev['failed'], rcc.summary(k),
                 None if sb is None else sb['all_ok'], R['runtime_s']), flush=True)
        if steps - 1 >= 1 and n_sched > 0:
            rng = np.random.default_rng(seed0 + steps)
            Nsched = int(round(B['P']['T_END'] / T_M)) + 2
            for j in range(n_sched):
                Dl = rb._random_schedule(rng, Nsched, steps - 1)
                Rr = rcc.simulate(B, h=H_SCHED, fifo=Dl)
                kr = rcc.kpis(Rr, B)
                sbr = short_bounds(ev, kr) if ev['closed'] else None
                row['random'].append(dict(index=j, kpis=kr, age_audit=Rr['age'],
                                          age_ok=bool(Rr['age']['max_left_limit']
                                                      <= db + 1e-12),
                                          bounds=sbr))
            Kr = [r['kpis'] for r in row['random']]
            row['random_aggregate'] = dict(
                disp_max=_max(x['disp'] for x in Kr),
                align_max=_max(x['align'] for x in Kr),
                gaperr_max=_max(x['gaperr'] for x in Kr),
                gmin_min=_min(x['gmin_min'] for x in Kr),
                mmin_lim_min=_min(x['mmin_lim_min'] for x in Kr),
                entry_max=_max(x['entry_obs'] for x in Kr),
                entry_min=_min(x['entry_obs'] for x in Kr),
                n_no_entry=sum(1 for x in Kr if x['entry_obs'] is None
                               or not np.isfinite(x['entry_obs'])),
                T_f_max=_max(x['T_f'] for x in Kr),
                n_mod_total=sum(x['n_mod'] + x['n_fallback'] + x['n_sat'] for x in Kr),
                all_stopped=all(x['all_stopped'] for x in Kr),
                all_age_ok=all(r['age_ok'] for r in row['random']),
                age_sup_max=max(r['age_audit']['max_left_limit'] for r in row['random']),
                all_bounds_ok=None if sbr is None else
                all(r['bounds']['all_ok'] for r in row['random']))
            ra = row['random_aggregate']
            print('    random x%d: disp <= %.4f align <= %.4f gaperr <= %.4f '
                  'gmin >= %.4f mmin >= %.4f entry %s-%s (no entry: %d) T_f <= '
                  '%.4f n_mod %d age_sup %.4f bounds_ok %s'
                  % (n_sched, ra['disp_max'], ra['align_max'], ra['gaperr_max'],
                     ra['gmin_min'], ra['mmin_lim_min'], ra['entry_min'],
                     ra['entry_max'], ra['n_no_entry'], ra['T_f_max'],
                     ra['n_mod_total'], ra['age_sup_max'], ra['all_bounds_ok']),
                  flush=True)
        rows.append(row)
    # per-condition boundaries
    print('  per-condition 1 ms-grid boundaries ...', flush=True)
    cb = cond_boundary_ms(cfg)
    print('  ', {k: v['k_ms'] for k, v in cb.items()}, flush=True)
    ce = cp.age_ceilings(cert_cfg(cfg))
    print('  run_b2 bisection crossings:', {k: round(v, 4) for k, v in ce.items()},
          flush=True)
    bd = cp.age_boundary_ms(cert_cfg(cfg))
    adm = [r['age_bound'] for r in rows if r['certificate']['admitted']]
    out = dict(study=('age sweep: certificate of the extended evaluator per '
                      'condition at every age bound; constant transport '
                      'dbar - T_m (held age exactly dbar) at h = %g s; %d '
                      'random schedules per age bound with d_j <= dbar/T_m - 1 '
                      'at h = %g s; per-condition boundaries on the 1 ms grid '
                      '(largest k with the condition holding at k ms) and the '
                      'run_b2 bisection crossings' % (H_FINE, n_sched, H_SCHED)),
               cfg=cfg, dbars_ms=list(dbars_ms), rows=rows,
               certified_age_bounds=adm,
               cond_boundary_ms=cb, ceilings_bisection=ce,
               dcert_num=min(ce.values()), binding=min(ce, key=ce.get),
               boundary_ms=bd,
               admitted_rows_inside_bounds=all(
                   r['bounds']['all_ok'] and
                   (r.get('random_aggregate', {}).get('all_bounds_ok') in (True, None))
                   for r in rows if r['certificate']['admitted']),
               admitted_rows_filter_inactive=all(
                   r['kpis']['filter_inactive'] and
                   r.get('random_aggregate', {}).get('n_mod_total', 0) == 0
                   for r in rows if r['certificate']['admitted']),
               all_rows_filter_inactive=all(
                   r['kpis']['filter_inactive'] and
                   r.get('random_aggregate', {}).get('n_mod_total', 0) == 0
                   for r in rows),
               all_rows_stopped=all(r['kpis']['all_stopped'] and
                                    r.get('random_aggregate', {}).get('all_stopped', True)
                                    for r in rows),
               runtime_s=time.time() - t0)
    dump('sweep.json', out)


def part_ens(n_draw=100, seed=7, half_width=3.0):
    cfg = design_cfg()
    ev0 = certificate(cfg)
    c0d = np.asarray(cfg['c0'], float)
    rng = np.random.default_rng(seed)
    print('[ens] %d takeover states, clearances in [%s, %s] m, mismatches as '
          'the Case B ensemble (seed %d)' % (n_draw, (c0d - half_width).tolist(),
                                             (c0d + half_width).tolist(), seed),
          flush=True)
    rows = []
    t0 = time.time()
    for j in range(n_draw):
        c0 = tuple(float(x) for x in
                   (c0d + rng.uniform(-half_width, half_width, 4)).round(3))
        eps0 = [float(rng.uniform(-0.35, 0.0))] + \
            [float(rng.uniform(0.0, 0.30) * f) for f in (1.0, 0.8, 0.6, 0.4)]
        eps0 = [round(x, 4) for x in eps0]
        cfg_j = dict(cfg, c0=list(c0), eps0=eps0, name='ens%03d' % j)
        cp._NS.clear()
        try:
            ev = certificate(cfg_j)
            cert = dict(admitted=ev['admitted'], failed=ev['failed'],
                        closed=ev['closed'],
                        slack={nm: c['slack'] for nm, c in ev['conds'].items()})
            if ev['closed']:
                K = ev['consts']
                cert.update(disp=K['disp'], align=K['align'],
                            gaperr=max(K['gaperr']), gmin=K['gmin'],
                            T_ent_bar=K['T_ent_bar'], T_f_bar=K['T_f_bar'],
                            min_margin=min(K['margins']),
                            min_auth=min(K['auth_slack']), vflr=K['vflr'])
        except Exception as ex:                       # pragma: no cover
            ev = None
            cert = dict(admitted=False, failed=['error'], error=str(ex))
        Bj = rcc.build(cfg_j)
        R = rcc.simulate(Bj, h=H_FINE)
        k = rcc.kpis(R, Bj)
        sb = short_bounds(ev, k) if (ev is not None and ev['closed']) else None
        rows.append(dict(index=j, c0=list(c0), eps0=eps0, e0=Bj['e0'],
                         certificate=cert, kpis=k,
                         bounds_ok=None if sb is None else sb['all_ok'],
                         bounds_failed=None if sb is None else
                         [nm for nm, r in sb['rows'].items() if r['ok'] is False]))
        if (j + 1) % 10 == 0:
            nadm = sum(1 for r in rows if r['certificate']['admitted'])
            print('  ens %d/%d: %d admitted; last: adm=%s failed=%s %s bounds_ok=%s '
                  '(%.0f s)' % (j + 1, n_draw, nadm, cert['admitted'],
                                cert['failed'], rcc.summary(k),
                                rows[-1]['bounds_ok'], time.time() - t0),
                  flush=True)
    adm = [r for r in rows if r['certificate']['admitted']]
    rej = [r for r in rows if not r['certificate']['admitted']]
    fails = {}
    for r in rej:
        for nm in r['certificate']['failed']:
            fails[nm] = fails.get(nm, 0) + 1
    agg = dict(
        n=n_draw, seed=seed, half_width=half_width, n_admitted=len(adm),
        n_rejected=len(rej), failures_among_rejected=fails,
        admitted=dict(
            all_bounds_ok=all(r['bounds_ok'] for r in adm) if adm else None,
            all_filter_inactive=all(r['kpis']['filter_inactive'] for r in adm)
            if adm else None,
            all_stopped=all(r['kpis']['all_stopped'] for r in adm) if adm else None,
            disp_max=_max(r['kpis']['disp'] for r in adm),
            align_max=_max(r['kpis']['align'] for r in adm),
            gaperr_max=_max(r['kpis']['gaperr'] for r in adm),
            gmin_min=_min(r['kpis']['gmin_min'] for r in adm),
            mmin_lim_min=_min(r['kpis']['mmin_lim_min'] for r in adm),
            entry_max=_max(r['kpis']['entry_obs'] for r in adm),
            entry_min=_min(r['kpis']['entry_obs'] for r in adm),
            c0_max=_max(max(r['c0']) for r in adm),
            c0_min=_min(min(r['c0']) for r in adm)),
        rejected=dict(
            n_mod_total=sum(r['kpis']['n_mod'] + r['kpis']['n_fallback']
                            + r['kpis']['n_sat'] for r in rej),
            all_stopped=all(r['kpis']['all_stopped'] for r in rej) if rej else None,
            n_entry_after_deadline=sum(
                1 for r in rej if not r['kpis']['entry_before_deadline']),
            n_handoff_box_violated=sum(
                1 for r in rej
                if max(abs(x) for x in r['kpis']['e_hnd']) > float(cfg['eps_e'])
                or max(abs(x) for x in r['kpis']['eps_hnd']) > float(cfg['eps_v'])),
            entry_max=_max(r['kpis']['entry_obs'] for r in rej),
            c0_min=_min(min(r['c0']) for r in rej),
            c0_max=_max(max(r['c0']) for r in rej)))
    print('  admitted %d of %d; failures among rejected %s; admitted: bounds_ok=%s '
          'filter_inactive=%s; rejected: n_mod=%d entry_after_deadline=%d '
          'handoff_box_violated=%d'
          % (len(adm), n_draw, fails, agg['admitted']['all_bounds_ok'],
             agg['admitted']['all_filter_inactive'], agg['rejected']['n_mod_total'],
             agg['rejected']['n_entry_after_deadline'],
             agg['rejected']['n_handoff_box_violated']), flush=True)
    out = dict(study=('takeover-state ensemble around the Case C design: '
                      'clearances uniform in [c0 - %g, c0 + %g] m per pair, '
                      'mismatches eps0_1 ~ U(-0.35, 0), eps0_i ~ U(0, 0.30) x '
                      '(1, 0.8, 0.6, 0.4) as the Case B ensemble; certificate '
                      '(extended evaluator, 20 ms) and continuous run '
                      '(h = %g s, transport 19 ms) for every state'
                      % (half_width, half_width, H_FINE)),
               cfg=cfg, design_certificate=cert_summary(ev0), aggregate=agg,
               rows=rows, runtime_s=time.time() - t0)
    dump('ensemble.json', out)


def part_digital():
    cfg = design_cfg()
    B = rcc.build(cfg)
    ev = certificate(cfg)
    print('[digital] zoh T_u in %s at h = %g vs continuous reference'
          % (TU_DIG, H_FINE), flush=True)
    Rc = rcc.simulate(B, h=H_FINE)
    ref = rcc.kpis(Rc, B)
    rows = [dict(T_u=H_FINE, h=H_FINE, scheme='cont', kpis=ref,
                 bounds=short_bounds(ev, ref))]
    for tu in TU_DIG:
        R = rcc.simulate(B, h=H_FINE, scheme='zoh', T_u=tu)
        k = rcc.kpis(R, B)
        d = rc.diff(k, ref)
        mx, kx, bad = rc.dmax(d)
        mx2, kx2, _ = rc.dmax({kk: vv for kk, vv in d.items() if kk != 'mmin'})
        rows.append(dict(T_u=tu, h=H_FINE, scheme='zoh', kpis=k,
                         bounds=short_bounds(ev, k), diff_to_continuous=d,
                         max_diff_to_continuous=mx, argmax_kpi=kx,
                         max_diff_excl_eval_margin=mx2, argmax_excl=kx2,
                         nan_mismatch=bad))
        print('  T_u=%-7g %s max|diff|=%.3e (%s); excl. eval-instant margin '
              '%.3e (%s) bounds_ok=%s' % (tu, rcc.summary(k), mx, kx, mx2, kx2,
                                          rows[-1]['bounds']['all_ok']), flush=True)
    # the 1 ms digital run integrated at h = 1 ms (as in the provisional study)
    R = rcc.simulate(B, h=1e-3, scheme='zoh', T_u=1e-3)
    k = rcc.kpis(R, B)
    d = rc.diff(k, ref)
    mx, kx, bad = rc.dmax(d)
    mx2, kx2, _ = rc.dmax({kk: vv for kk, vv in d.items() if kk != 'mmin'})
    rows.append(dict(T_u=1e-3, h=1e-3, scheme='zoh', kpis=k,
                     bounds=short_bounds(ev, k), diff_to_continuous=d,
                     max_diff_to_continuous=mx, argmax_kpi=kx,
                     max_diff_excl_eval_margin=mx2, argmax_excl=kx2,
                     nan_mismatch=bad))
    print('  T_u=1 ms at h=1 ms: max|diff|=%.3e (%s); excl. eval-instant margin '
          '%.3e (%s)' % (mx, kx, mx2, kx2), flush=True)
    out = dict(study=('digital implementation: the local law is sampled every '
                      'T_u and held (zero-order hold) with event-triggered '
                      'updates; plant, message schedule (T_m = 1 ms, transport '
                      '19 ms), detector and gains fixed; compared with the '
                      'continuous reference (T_u = h = %g s).  "mmin" is the '
                      'margin at the controller\'s own evaluation instants, '
                      'which differ by construction; "mmin_lim" (event left '
                      'limits included) is the comparable statistic' % H_FINE),
               cfg=cfg, certificate=cert_summary(ev), rows=rows,
               all_filter_inactive=all(r['kpis']['filter_inactive'] for r in rows),
               all_bounds_ok=all(r['bounds']['all_ok'] for r in rows))
    dump('digital.json', out)


def part_check():
    """Sanity over every file."""
    out = dict(files={}, all_ok=True)

    def add(name, ok, detail=None):
        out['files'][name] = dict(ok=bool(ok), detail=detail)
        if not ok:
            out['all_ok'] = False
        print('  %-18s %s %s' % (name, 'OK' if ok else 'FAIL', detail or ''),
              flush=True)
    d = load('design.json')
    add('design.json', d is not None and d['certificate']['admitted']
        and d['interval_at_20ms']['verified'] and d['interval_at_20ms']['fp_inside'],
        None if d is None else dict(boundary_ms=d['boundary_ms']['k_ms'],
                                    binding=d['binding'], dcert_num=d['dcert_num']))
    r = load('reference.json')
    add('reference.json', r is not None and r['bounds']['all_ok']
        and r['kpis']['filter_inactive'] and r['kpis']['all_stopped'],
        None if r is None else dict(n_checked=r['bounds']['n_checked'],
                                    n_failed=r['bounds']['n_failed'],
                                    n_mod=r['filter_modifications']))
    c = load('convergence.json')
    add('convergence.json', c is not None and c['all_rows_inside_bounds']
        and c['all_filter_inactive'],
        None if c is None else dict(max_diff=c['max_diff_over_h_and_kpis'],
                                    argmax=c['argmax_kpi']))
    s = load('schedules.json')
    add('schedules.json', s is not None and s['aggregate']['all_bounds_ok']
        and s['aggregate']['all_filter_inactive'] and s['aggregate']['all_age_ok']
        and s['aggregate']['all_stopped'],
        None if s is None else dict(n=s['aggregate']['n'],
                                    age_sup_max=s['aggregate']['age_sup_max'],
                                    n_bounds_failed=s['aggregate']['n_bounds_failed']))
    w = load('sweep.json')
    add('sweep.json', w is not None and w['admitted_rows_inside_bounds']
        and w['admitted_rows_filter_inactive'],
        None if w is None else dict(certified_age_bounds=w['certified_age_bounds'],
                                    all_rows_filter_inactive=w['all_rows_filter_inactive'],
                                    all_rows_stopped=w['all_rows_stopped']))
    e = load('ensemble.json')
    if e is not None:
        add('ensemble.json', e['aggregate']['admitted']['all_bounds_ok'] in (True, None)
            and e['aggregate']['admitted']['all_filter_inactive'] in (True, None),
            dict(n_admitted=e['aggregate']['n_admitted'],
                 failures=e['aggregate']['failures_among_rejected'],
                 rejected_n_mod=e['aggregate']['rejected']['n_mod_total']))
    g = load('digital.json')
    add('digital.json', g is not None and g['all_filter_inactive'] and g['all_bounds_ok'],
        None if g is None else dict(max_diff_1ms=[x['max_diff_excl_eval_margin']
                                                  for x in g['rows'] if x['scheme'] == 'zoh'
                                                  and x['T_u'] == 1e-3]))
    dump('check.json', out)
    print('  ALL OK' if out['all_ok'] else '  FAILURES PRESENT', flush=True)


PARTS = dict(design=part_design, ref=part_ref, conv=part_conv, sched=part_sched,
             sweep=part_sweep, ens=part_ens, digital=part_digital, check=part_check)

if __name__ == '__main__':
    todo = sys.argv[1:] or ['all']
    if todo == ['all']:
        todo = ['design', 'ref', 'conv', 'sched', 'sweep', 'ens', 'digital', 'check']
    for p in todo:
        t0 = time.time()
        print('=== part: %s ===' % p, flush=True)
        PARTS[p]()
        print('[%s] done in %.1f s' % (p, time.time() - t0), flush=True)
