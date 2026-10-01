#!/usr/bin/env python3
"""run_cont.py -- canonical continuous-feedback reference runs for the
second-round audit (R2-B02, R2-B03), 2026-09-25.

Simulator: src/simulation/sim_cont.py (see its docstring).  Outputs go
to data/new_d19/cont/ only; nothing else under data/ is touched.

Parts
  regress  sim_cont 'legacy' scheme (h = T_u = T_m = T_c = 1 ms) against
           sim_root2.run for Case B (19-step transport), Case A (59 steps)
           and Case B with exact standstill sensing (v_thr = 0), plus the
           bridge from the 1 kHz implementation to the continuous
           reference (one convention changed per row)  -> regression.json
  conv     theorem system ('cont': continuous local law, RK4 between
           events, localized zero-speed events), T_m = T_c = 1 ms, fixed
           transport, h in {1e-4, 5e-5, 2.5e-5} s (plus two coarser
           diagnostic rows)  -> convergence.json, cont_B.json, cont_A.json,
           traj_cont_B.npz, traj_cont_A.npz
  digital  digital implementation (zero-order hold, update period T_u in
           {1, 0.5, 0.25, 0.1} ms, event-triggered updates) at the finest
           h against the continuous reference  -> digital.json
  xcheck   independent exact (matrix-exponential) propagation of the
           nominal system Sigma^N (src/simulation/sim_expm_nominal.py)
           against the filtered continuous run  -> xcheck.json
  all      regress conv digital xcheck

Usage: python run_cont.py [part ...]
"""
import json
import os
import re
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import numpy as np                                      # noqa: E402
import run_b2 as rb                                     # noqa: E402
import run_b2_ext as rx                                 # noqa: E402
import sim_root2 as sr2                                 # noqa: E402
import sim_cont as sc                                   # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'cont')
NUMBERS = os.path.join(CODE, '..', 'new', 'build', 'numbers.tex')
H_CONV = (1e-4, 5e-5, 2.5e-5)          # convergence study (audit R2-B03)
H_DIAG = (1e-3, 5e-4)                  # coarser diagnostic rows
H_FINE = H_CONV[-1]
TU_DIG = (1e-3, 5e-4, 2.5e-4, 1e-4)    # digital implementation study
T_M = 1e-3
T_C = 1e-3

CASES = dict(
    B=dict(P=dict(rb.SIM_P, dbar=0.019), d_s=list(rb.D_S), v0=list(rb.V0),
           e0=list(rb.E0), eps_e=rb.B2['eps_e'], eps_v=rb.B2['eps_v'],
           transport=0.019, age_bound=0.020, prefix='nB'),
    A=dict(P=dict(rx.A_SIM, dbar=0.059), d_s=[float(x) for x in rx.A_DS],
           v0=list(rx.A_V0), e0=[float(x) for x in rx.A_E0], eps_e=0.20,
           eps_v=0.02, transport=0.059, age_bound=0.060, prefix='nA'))


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(x) for x in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return None if not np.isfinite(x) else x
    return o


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(_py(obj), open(p, 'w'), indent=1)
    print('  -> %s' % os.path.relpath(p, CODE), flush=True)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def run_case(c, **kw):
    C = CASES[c]
    t0 = time.time()
    R = sc.run(C['P'], C['d_s'], C['v0'], C['e0'], eps_e=C['eps_e'],
               eps_v=C['eps_v'], **kw)
    R['runtime_s'] = time.time() - t0
    return R


# KPI vector used for all comparisons
KPI_KEYS = ('disp', 'align', 'gaperr', 'sched', 'T_f', 'tstop', 'tact', 'tset',
            'trule', 'gmin', 'hmin', 'mmin', 'mmin_lim', 'upk', 'e_hnd',
            'eps_hnd', 'wint', 'epsmax_b', 'fb_real', 'marker_signed', 'egap',
            'entry_obs', 'entry_obs_vel_only', 'entry_obs_pos_only')
COUNT_KEYS = ('hits', 'hits_true', 'infeas', 'n_mod', 'n_fallback', 'n_sat',
              'n_localized', 'neg_speed')


def kpis(R):
    k = {key: _py(R[key]) for key in KPI_KEYS}
    k.update({key: int(R[key]) for key in COUNT_KEYS})
    k['gmin_min'] = float(np.min(R['gmin']))
    k['mmin_min'] = float(np.min(R['mmin']))
    k['mmin_lim_min'] = float(np.min(R['mmin_lim']))
    k['all_stopped'] = bool(R['all_stopped'])
    k['contact_time'] = R['contact_time']
    k['filter_inactive'] = bool(R['n_mod'] == 0 and R['n_fallback'] == 0
                                and R['n_sat'] == 0
                                and float(np.min(R['mmin_lim'])) > 0.0)
    k['runtime_s'] = R.get('runtime_s')
    k['config'] = R['config']
    return k


def diff(a, b, keys=KPI_KEYS):
    """Per-KPI max |a - b| (NaN-aware: NaN in both counts as equal; a NaN
    in only one of them makes the entry NaN, reported by dmax)."""
    out = {}
    for key in keys:
        x = np.atleast_1d(np.asarray(a[key], float))
        y = np.atleast_1d(np.asarray(b[key], float))
        both = np.isnan(x) & np.isnan(y)
        d = np.abs(x - y)
        d[both] = 0.0
        out[key] = float(np.max(d)) if not np.any(np.isnan(d)) else float('nan')
    return out


def dmax(d):
    """(max over finite entries, key of the max, keys with a NaN mismatch)."""
    fin = {k: v for k, v in d.items() if np.isfinite(v)}
    bad = sorted(k for k, v in d.items() if not np.isfinite(v))
    if not fin:
        return float('nan'), None, bad
    kmax = max(fin, key=fin.get)
    return fin[kmax], kmax, bad


def macros():
    M = {}
    if os.path.exists(NUMBERS):
        for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{([^}]*)\}', open(NUMBERS).read()):
            M[m.group(1)] = m.group(2)
    return M


def bounds_check(c, k):
    """Observed continuous-run values against the printed certified bounds
    (new/build/numbers.tex, snapshot at run time)."""
    M = macros()
    p = CASES[c]['prefix']

    def num(name):
        try:
            return float(M[p + name])
        except (KeyError, ValueError):
            return None
    rows = dict(
        inactivity_margin=(num('sigmaMin'), k['mmin_lim_min'], 'floor'),
        clearance_margin=(num('gminBound'), k['gmin_min'], 'floor'),
        persistent_entry=(num('entryBound'), k['entry_obs'], 'upper'),
        final_standstill=(num('TfBound'), k['T_f'], 'upper'),
        dispersion=(num('dispBound'), k['disp'], 'upper'),
        schedule_error=(num('schedBound'), k['sched'], 'upper'),
        marker_error=(num('alignBound'), k['align'], 'upper'),
        terminal_gap_error=(num('gapBound'),
                            float(np.max(np.abs(np.asarray(k['egap'][1:])))), 'upper'))
    out = {}
    for name, (bound, obs, kind) in rows.items():
        ok = None if bound is None else (obs >= bound if kind == 'floor' else obs <= bound)
        out[name] = dict(bound=bound, observed=obs, kind=kind, ok=ok)
    out['_source'] = 'new/build/numbers.tex (printed bounds; floors rounded down, bounds up)'
    return out


# ------------------------------------------------------------------ regress
REG_KEYS = ('tstop', 'disp', 'align', 'gaperr', 'egap', 's', 'gmin', 'hmin', 'upk',
            'hits', 'infeas', 'act_time', 'tact', 'tset', 'mmin', 'wint',
            'epsmax_b', 'fb_real', 'e_hnd', 'eps_hnd')


def part_regress():
    print('[regress] sim_cont legacy scheme vs sim_root2.run', flush=True)
    out = dict(tolerance=1e-9, cases={})
    # random FIFO schedule for Case B: i.i.d. transport d_j in {1..19} steps
    NB = int(CASES['B']['P']['T_END'] / 1e-3)
    fifo_B = np.random.default_rng(2026).integers(1, 20, size=(5, NB))
    fifo_B[0] = 1
    for tag, c, v_thr, fifo in (('B', 'B', None, None), ('A', 'A', None, None),
                                ('B_vthr0', 'B', 0.0, None),
                                ('B_fifo', 'B', None, fifo_B)):
        C = CASES[c]
        t0 = time.time()
        S = sr2.run(C['P'], C['d_s'], C['v0'], C['e0'], rule='ab',
                    apply_filter=True, terminate_on_contact=True, extra=True,
                    v_thr=v_thr, delays=fifo)
        t_ref = time.time() - t0
        t0 = time.time()
        R = sc.run(C['P'], C['d_s'], C['v0'], C['e0'], scheme='legacy',
                   v_rule=v_thr, eps_e=C['eps_e'], eps_v=C['eps_v'], fifo=fifo)
        t_new = time.time() - t0
        d = {}
        for key in REG_KEYS:
            a = np.asarray(R[key], float)
            b = np.asarray(S[key], float)
            dd = np.abs(np.nan_to_num(a, nan=-7.0) - np.nan_to_num(b, nan=-7.0))
            d[key] = float(np.max(dd))
        mx = max(d.values())
        out['cases'][tag] = dict(
            call=('sim_root2.run(P, d_s, v0, e0, rule="ab", apply_filter=True, '
                  'terminate_on_contact=True, extra=True, v_thr=%r, delays=%s)'
                  % (v_thr, 'None' if fifo is None else
                     'random FIFO schedule, d_j iid in {1..19}, seed 2026')),
            transport_steps=int(round(C['transport'] / 1e-3)),
            max_abs_diff=d, max_over_keys=mx, pass_=bool(mx <= 1e-9),
            bit_identical=bool(mx == 0.0), runtime_sim_root2_s=t_ref,
            runtime_sim_cont_s=t_new,
            sim_root2=dict(disp=S['disp'], align=S['align'], gaperr=S['gaperr'],
                           tstop=S['tstop'], gmin=S['gmin'], hits=S['hits'],
                           infeas=S['infeas'], mmin=S['mmin']))
        print('  %-8s max diff over %d keys = %.3e (sim_root2 %.1f s, sim_cont %.2f s)'
              % (tag, len(REG_KEYS), mx, t_ref, t_new), flush=True)
    # the same FIFO schedule in the continuous theorem system
    Rf = run_case('B', scheme='cont', h=H_FINE, T_m=T_M, T_c=T_C, fifo=fifo_B,
                  transport=CASES['B']['transport'])
    kf = kpis(Rf)
    out['fifo_cont_B'] = dict(
        what='continuous theorem system (h = %g s) on the random FIFO schedule '
             'of case B_fifo' % H_FINE, kpis=kf,
        age_audit=dict(max_left_limit=Rf['age']['max_left_limit'],
                       max_at_delivery=Rf['age']['max_at_delivery']))
    print('  continuous run on the FIFO schedule: disp=%.6f align=%.6f gmin=%.6f '
          'mmin=%.6f filter_inactive=%s age_sup=%.4f s'
          % (kf['disp'], kf['align'], kf['gmin_min'], kf['mmin_lim_min'],
             kf['filter_inactive'], Rf['age']['max_left_limit']), flush=True)
    out['bridge'] = part_bridge()
    out['note'] = ('The legacy scheme reproduces sim_root2.run operation for '
                   'operation: semi-implicit Euler with the zero clamp, Riemann '
                   'reference, RULE on v_{i-1} <= v_thr (default eps_det = 0.015 '
                   'm/s), LATCH on v_i <= 1e-4 m/s at 1 ms instants, t > T_xi '
                   'guards, head tracking the ramp with feedforward, clip to '
                   '+-a_max_i.  The canonical schemes differ from it only by the '
                   'conventions listed in "bridge" (one change per row).')
    dump('regression.json', out)


def part_bridge():
    """From the 1 kHz implementation (sim_root2 semantics) to the
    continuous reference, one convention per row."""
    rows = {}
    for c in ('B', 'A'):
        steps = [
            ('L0_sim_root2', 'sim_root2 semantics (1 kHz, held commands, '
             'RULE at v_{i-1} <= 0.015 m/s, LATCH at v_i <= 1e-4 m/s on the '
             '1 ms grid, head tracks the ramp, semi-implicit Euler)',
             dict(scheme='legacy')),
            ('L1_exact_standstill', 'L0 with exact standstill sensing for RULE '
             '(v_{i-1} = 0)', dict(scheme='legacy', v_rule=0.0)),
            ('L2_head_brake', 'L1 with the head braking open loop at T_xi '
             '(BRAKE of Sec. III-A)', dict(scheme='legacy', v_rule=0.0,
                                         head='brake')),
            ('Z1_zoh_1ms', 'theorem event semantics with commands held for '
             'T_u = 1 ms: exact double-integrator update, zero-speed events '
             'localized, exact reference (h = 1 ms)',
             dict(scheme='zoh', h=1e-3, T_u=1e-3)),
            ('Z2_zoh_1ms_fine', 'Z1 at h = %g s (held commands are integrated '
             'exactly; checks h-independence)' % H_FINE,
             dict(scheme='zoh', h=H_FINE, T_u=1e-3)),
            ('C_cont', 'continuous reference (T_u = h = %g s, RK4)' % H_FINE,
             dict(scheme='cont', h=H_FINE)),
        ]
        prev = None
        rows[c] = []
        for tag, desc, kw in steps:
            R = run_case(c, T_m=T_M, T_c=T_C, transport=CASES[c]['transport'], **kw)
            k = kpis(R)
            row = dict(tag=tag, what=desc, kpis=k)
            if prev is not None:
                row['change_vs_previous'] = diff(k, prev)
            prev = k
            rows[c].append(row)
            print('    bridge %s %-20s disp=%.6f align=%.6f gaperr=%.6f gmin=%.6f '
                  'mmin=%.6f n_mod=%d (%.2f s)'
                  % (c, tag, k['disp'], k['align'], k['gaperr'], k['gmin_min'],
                     k['mmin_lim_min'], k['n_mod'], R['runtime_s']), flush=True)
        ref = rows[c][-1]['kpis']
        for row in rows[c]:
            row['diff_to_continuous'] = diff(row['kpis'], ref)
    return rows


# --------------------------------------------------------------------- conv
def _traj(c, R, path):
    rec = R['rec']
    C = CASES[c]
    np.savez(path, t=rec['t'], v=rec['v'], e=rec['e'], eps=rec['eps'], u=rec['u'],
             m=rec['m'], g=rec['g'], h=rec['h'], tstop=R['tstop'], tset=R['tset'],
             tact=R['tact'], trule=R['trule'], marks=R['marks'], s=R['s'],
             egap=R['egap'], d_s=np.array(C['d_s']), v0=np.array(C['v0']),
             e0=np.array(C['e0']),
             keys_note=np.array('t [s] every 10 ms; v unit speeds; e pair POSITION '
                                'errors e_i (row 0 = head vs reference); eps pair '
                                'VELOCITY errors; u applied commands (right limits); '
                                'm inactivity margins u_cbf-u_nom (pairs, NaN once '
                                'latched); g clearance margins; h barriers. NOTE: '
                                'in b2_traj.npz / caseA_traj.npz the key "e" holds '
                                'the velocity error eps.'),
             config=np.array(json.dumps(_py(R['config']))))
    print('  -> %s' % os.path.relpath(path, CODE), flush=True)


def _cont_record(c, R):
    """KPIs of the finest continuous run in the field names of
    b2_ext_parts/margin.json (Case B) and caseA_traj.json (Case A)."""
    C = CASES[c]
    rec = R['rec']
    k = kpis(R)
    sel = slice(None, None, 5)                  # (t, m) pairs every 50 ms
    mser = []
    for j in range(rec['m'].shape[0]):
        tt = rec['t'][sel]
        mm = rec['m'][j][sel]
        ok = np.isfinite(mm)
        mser.append([[float(a), float(b)] for a, b in zip(tt[ok], mm[ok])])
    out = dict(
        # margin.json / caseA_traj.json field names
        mmin=R['mmin'], m=mser, disp=R['disp'], align=R['align'],
        gaperr=R['gaperr'], e_hnd=R['e_hnd'], eps_hnd=R['eps_hnd'],
        wint=R['wint'], egap_signed=R['egap'], marker_signed=R['marker_signed'],
        tstop=R['tstop'], tset=R['tset'], tact=R['tact'], upk=R['upk'],
        gmin=R['gmin'], hmin=R['hmin'], hits=R['hits'], infeas=R['infeas'],
        act_time=[0.0] * len(R['tstop']), contact_time=R['contact_time'],
        all_stopped=R['all_stopped'],
        transport_steps=int(round(C['transport'] / T_M)),
        age_audit=dict(max_left_limit_steps=int(round(R['age']['max_left_limit'] / T_M)),
                       max_update_steps=int(round(R['age']['max_at_delivery'] / T_M))),
        # continuous-model extras
        system=('theorem system: continuous local law (T_u = h), RK4 between '
                'events, T_m = T_c = 1 ms, constant transport, exact standstill '
                'sensing, head BRAKE at T_xi, zero-speed events localized'),
        h=R['config']['h'], T_u=R['config']['T_u'], T_m=T_M, T_c=T_C,
        transport=C['transport'], age_bound=C['age_bound'],
        age_audit_s=dict(max_left_limit=R['age']['max_left_limit'],
                         max_at_delivery=R['age']['max_at_delivery']),
        mmin_incl_event_limits=R['mmin_lim'], mmin_min=k['mmin_min'],
        mmin_incl_event_limits_min=k['mmin_lim_min'],
        filter_modifications=dict(min_branch=R['n_mod'], fallback=R['n_fallback'],
                                  saturation=R['n_sat'], evaluations=R['n_eval'],
                                  left_limit_evaluations=R['n_left_limits']),
        filter_inactive=k['filter_inactive'], hits_true=R['hits_true'],
        gmin_grid=R['gmin_grid'], trule=R['trule'], sched=R['sched'], T_f=R['T_f'],
        epsmax_b=R['epsmax_b'], fb_real=R['fb_real'],
        entry_obs=R['entry_obs'], entry_obs_vel_only=R['entry_obs_vel_only'],
        entry_obs_pos_only=R['entry_obs_pos_only'],
        entry_obs_followers=R['entry_obs_followers'],
        entry_definition=('first time after which |e_i| <= eps_e and |eps_i| <= '
                          'eps_v for every i = 1..n (head: reference errors) up '
                          'to T_xi (audit R2-E01); the last exit is localized '
                          'by linear interpolation of the box excess between '
                          'integration steps. entry_obs_vel_only ignores the '
                          'position errors (it reproduces the statistic of '
                          'gen_numbers.last_exit, whose npz key "e" is the '
                          'velocity error)'),
        eps_e=C['eps_e'], eps_v=C['eps_v'],
        n_localized_events=R['n_localized'], root_residual=R['root_resid'],
        neg_speed_events=R['neg_speed'], runtime_s=R['runtime_s'],
        bounds_check=bounds_check(c, k))
    return out


def part_conv():
    print('[conv] continuous reference, h in %s (+ diagnostic %s)' % (H_CONV, H_DIAG),
          flush=True)
    out = dict(study=('numerical convergence (audit R2-B03): same physical '
                      'event schedule (sends every T_m = 1 ms, constant transport, '
                      'detector T_c = 1 ms anchored at activation), same '
                      'continuous law; only the RK4 step h varies'),
               h_values=list(H_CONV), diagnostic_h=list(H_DIAG), cases={})
    for c in ('B', 'A'):
        rows = []
        runs = {}
        for h in H_DIAG + H_CONV:
            R = run_case(c, scheme='cont', h=h, T_m=T_M, T_c=T_C,
                         transport=CASES[c]['transport'])
            runs[h] = R
            rows.append(dict(h=h, diagnostic=h in H_DIAG, kpis=kpis(R)))
            print('  %s h=%-7g disp=%.12f align=%.12f gmin=%.12f mmin=%.9f '
                  'n_mod=%d (%.1f s)' % (c, h, R['disp'], R['align'],
                                         np.min(R['gmin']), np.min(R['mmin_lim']),
                                         R['n_mod'], R['runtime_s']), flush=True)
        ref = rows[-1]['kpis']
        for row in rows:
            row['diff_to_finest'] = diff(row['kpis'], ref)
            mx, kx, bad = dmax(row['diff_to_finest'])
            row['max_diff_to_finest'] = mx
            row['argmax_kpi'] = kx
            row['nan_mismatch'] = bad
        conv_rows = [r for r in rows if not r['diagnostic']]
        worst = {}
        for key in KPI_KEYS:
            worst[key] = max(r['diff_to_finest'][key] for r in conv_rows)
        mx, kx, bad = dmax(worst)
        out['cases'][c] = dict(rows=rows, max_diff_over_h_by_kpi=worst,
                               max_diff_over_h_and_kpis=mx, argmax_kpi=kx,
                               nan_mismatch=bad)
        R = runs[H_FINE]
        dump('cont_%s.json' % c, _cont_record(c, R))
        _traj(c, R, os.path.join(OUT, 'traj_cont_%s.npz' % c))
    dump('convergence.json', out)


# ------------------------------------------------------------------ digital
def part_digital():
    print('[digital] ZOH T_u in %s at h = %g vs continuous reference'
          % (TU_DIG, H_FINE), flush=True)
    out = dict(study=('digital implementation sensitivity (audit R2-B03): '
                      'plant, message/detector schedule and gains fixed; the '
                      'local law is sampled every T_u and held (zero-order '
                      'hold), with event-triggered updates at packet, detector '
                      'and zero-speed events; h = %g s' % H_FINE),
               T_u_values=list(TU_DIG), h=H_FINE, cases={})
    for c in ('B', 'A'):
        Rc = run_case(c, scheme='cont', h=H_FINE, T_m=T_M, T_c=T_C,
                      transport=CASES[c]['transport'])
        ref = kpis(Rc)
        rows = [dict(T_u=H_FINE, scheme='cont', kpis=ref)]
        for tu in TU_DIG:
            R = run_case(c, scheme='zoh', h=H_FINE, T_u=tu, T_m=T_M, T_c=T_C,
                         transport=CASES[c]['transport'])
            k = kpis(R)
            d = diff(k, ref)
            mx, kx, bad = dmax(d)
            # 'mmin' is the minimum over the controller's own evaluation
            # instants, which differ by construction (update instants vs
            # every RK4 stage); 'mmin_lim' adds the event left limits and is
            # the comparable margin statistic
            mx2, kx2, _ = dmax({kk: vv for kk, vv in d.items() if kk != 'mmin'})
            rows.append(dict(T_u=tu, scheme='zoh', kpis=k, diff_to_continuous=d,
                             max_diff_to_continuous=mx, argmax_kpi=kx,
                             max_diff_excl_eval_margin=mx2, argmax_excl=kx2,
                             nan_mismatch=bad))
            print('  %s T_u=%-7g disp=%.9f align=%.9f gmin=%.9f mmin=%.6f n_mod=%d '
                  'max|diff|=%.3e (%s); excl. eval-instant margin %.3e (%s) (%.1f s)'
                  % (c, tu, k['disp'], k['align'], k['gmin_min'], k['mmin_lim_min'],
                     k['n_mod'], mx, kx, mx2, kx2, R['runtime_s']), flush=True)
        # the 1 kHz implementation of the paper (sim_root2 semantics)
        Rl = run_case(c, scheme='legacy')
        kl = kpis(Rl)
        dl = diff(kl, ref)
        mx, kx, bad = dmax(dl)
        rows.append(dict(T_u=1e-3, scheme='legacy (sim_root2 semantics)', kpis=kl,
                         diff_to_continuous=dl, max_diff_to_continuous=mx,
                         argmax_kpi=kx, nan_mismatch=bad))
        print('  %s legacy 1 kHz max|diff| to continuous = %.3e (%s)'
              % (c, mx, kx), flush=True)
        out['cases'][c] = dict(rows=rows)
    dump('digital.json', out)


# ------------------------------------------------------------------- xcheck
def part_xcheck():
    """Independent check of the canonical simulator: exact matrix-
    exponential propagation of the nominal system Sigma^N (filter
    removed; sim_expm_nominal.py shares no code with sim_cont.py)
    against the filtered continuous run at the finest h."""
    import sim_expm_nominal as sx
    print('[xcheck] exact expm propagation of Sigma^N vs sim_cont cont (h = %g)'
          % H_FINE, flush=True)
    out = dict(what=('Sigma^N propagated exactly (expm of the affine mode '
                     'dynamics over each 1 ms interval, Brent-localized zero '
                     'speeds) vs the filtered closed loop of sim_cont (RK4, '
                     'h = %g s). Agreement shows (i) that the canonical '
                     'simulator integrates the stated law correctly and (ii) '
                     'that the executed filtered loop coincides with the '
                     'nominal one (filter inactive).' % H_FINE), cases={})
    pairs = (('tstop', 'tstop'), ('tact', 'tact'), ('tset', 'tset'),
             ('trule', 'trule'), ('disp', 'disp'), ('align', 'align'),
             ('gaperr', 'gaperr'), ('marker_signed', 'marker_signed'),
             ('egap', 'egap'), ('e_hnd', 'e_hnd'), ('eps_hnd', 'eps_hnd'),
             ('gmin_grid', 'gmin_grid'), ('mmin_grid', 'mmin_lim'),
             ('upk_grid', 'upk'))
    for c in ('B', 'A'):
        C = CASES[c]
        t0 = time.time()
        X = sx.run_nominal(C['P'], C['d_s'], C['v0'], C['e0'], C['transport'],
                           T_m=T_M)
        t_x = time.time() - t0
        R = run_case(c, scheme='cont', h=H_FINE, T_m=T_M, T_c=T_C,
                     transport=C['transport'])
        d = {}
        for kx, kc in pairs:
            a = {kx: X[kx]}
            b = {kx: R[kc]}
            d['%s|%s' % (kx, kc)] = diff(a, b, keys=(kx,))[kx]
        mx, kmax, bad = dmax(d)
        out['cases'][c] = dict(max_abs_diff=d, max_over_keys=mx, argmax=kmax,
                               nan_mismatch=bad, runtime_expm_s=t_x,
                               n_s1_unsaturated=X['n_s1_unsaturated'],
                               n_mode_configurations=X['n_mode_configurations'],
                               expm=dict(disp=X['disp'], align=X['align'],
                                         gaperr=X['gaperr'], tstop=X['tstop'],
                                         mmin_grid=X['mmin_grid'],
                                         gmin_grid=X['gmin_grid']),
                               cont_filter_inactive=kpis(R)['filter_inactive'])
        print('  %s max |expm - cont| = %.3e (%s); S1 unsaturated samples %d; '
              'expm %.1f s' % (c, mx, kmax, X['n_s1_unsaturated'], t_x), flush=True)
        for key, val in d.items():
            print('     %-28s %.3e' % (key, val), flush=True)
    dump('xcheck.json', out)


PARTS = dict(regress=part_regress, conv=part_conv, digital=part_digital,
             xcheck=part_xcheck)

if __name__ == '__main__':
    todo = sys.argv[1:] or ['all']
    if todo == ['all']:
        todo = ['regress', 'conv', 'digital', 'xcheck']
    for p in todo:
        t0 = time.time()
        PARTS[p]()
        print('[%s] done in %.1f s' % (p, time.time() - t0), flush=True)
