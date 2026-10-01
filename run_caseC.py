#!/usr/bin/env python3
"""run_caseC.py -- continuous-feedback reference runs of Case C (slow-down
reference of Assumption 1', 2026-09-26).

Case C: the head unit approaches at v_xi, starts to decelerate at t_d at
the constant rate a_d, reaches the crawl speed v_c1 at
t_c = t_d + (v_xi - v_c1)/a_d, crawls until the handoff T_xi and brakes
open loop at a_b; the followers run the unchanged local law (COAST / S1 /
S2 / RULE / LATCH, filter, interlock) on the forwarded command.
Simulator: src/simulation/sim_cont.py with profile=dict(t_d, a_d, v_c1)
(module docstring).  Outputs go to data/new_d19/caseC/ only.

Parts
  ref    one continuous run ('cont', RK4, T_u = h) at the fine step with
         the KPIs and a decimated trajectory every 10 ms
         -> ref_<tag>.json, traj_<tag>.npz
  conv   the same run at h in H_DIAG + H_CONV, differences to the finest
         -> convergence_<tag>.json
  sched  random FIFO schedules with per-send transport d_j <= dmax steps
         (run_b2._random_schedule), held-age audit, 'cont' at h = 1 ms
         -> schedules_<tag>.json
  sweep  age sweep: constant transport dbar - T_m at every age bound in
         DBARS ('cont' at the fine step) plus a few random schedules with
         d_j <= dbar/T_m - 1 at h = 1 ms -> sweep_<tag>.json
  all    ref conv sched sweep

Design: the provisional design PROVISIONAL (c0 = 15 m for every pair,
lambda = 0.08, a_d = 0.2 m/s^2, T_xi = 90 s, t_d = 10 s, v_c1 = 1 m/s) or
a JSON file with the same keys (--cfg path.json); a partial file is
merged over PROVISIONAL.

Usage: python run_caseC.py [part ...] [--cfg FILE] [--tag NAME] [--h H]
                           [--n N] [--seed S] [--dbars a b c ...]
"""
import argparse
import json
import os
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import numpy as np                                      # noqa: E402
import run_b2 as rb                                     # noqa: E402
import run_cont as rc                                   # noqa: E402
import sim_cont as sc                                   # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'caseC')
T_M = 1e-3
T_C = 1e-3
H_FINE = 2.5e-5
H_CONV = (1e-4, 5e-5, 2.5e-5)
H_DIAG = (1e-3, 5e-4)
H_SCHED = 1e-3                          # 'cont' at h = 1 ms on schedules
DBARS = (0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.040, 0.050)

PROVISIONAL = dict(
    name='provisional',
    # reference and handoff
    v_xi=10.8, a_b=1.0, T_xi=90.0, t_d=10.0, a_d=0.2, v_c1=1.0,
    # gains and tolerances (Case B values except lam)
    alpha=0.35, lam=0.08, phi=0.005, eps_det=0.015, kappa=1.0,
    eps_e=0.15, eps_v=0.02,
    # communication contract
    dbar=0.020,                          # held age bound; transport dbar - T_m
    # fleet
    n=5, ell=25.0, s_m=1.0, U=1.6, v_c=1.2,
    amax=[1.60, 1.48, 1.38, 1.29, 1.21],
    d_s=[6.0, 5.0, 5.0, 5.0, 5.0],       # head slot 6 m, gaps 5 m
    c0=[15.0, 15.0, 15.0, 15.0],         # takeover clearances (pairs)
    eps0=[-0.30, 0.25, 0.18, 0.12, 0.08],
    e0_head=-0.5,
    T_END=None,                          # default T_xi + 15 s
)


# --------------------------------------------------------------- helpers
def _py(o):
    return rc._py(o)


def load_cfg(path=None, overrides=None):
    cfg = dict(PROVISIONAL)
    if path:
        cfg.update(json.load(open(path)))
        cfg.setdefault('name', os.path.splitext(os.path.basename(path))[0])
    if overrides:
        cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build(cfg):
    """Simulator inputs (sim_root2 constant format) of a Case C design."""
    lam = float(cfg['lam'])
    n = int(cfg['n'])
    T_xi = float(cfg['T_xi'])
    T_END = cfg.get('T_END') or T_xi + 15.0
    transport = float(cfg['dbar']) - T_M
    P = dict(n=n, dt=1e-3, dbar=transport, a_b=float(cfg['a_b']),
             v_xi=float(cfg['v_xi']), T_xi=T_xi, T_END=float(T_END),
             alpha=float(cfg['alpha']), beta=lam * lam, gamma=2.0 * lam,
             phi=float(cfg['phi']), eps_det=float(cfg['eps_det']),
             ell=float(cfg['ell']), s_m=float(cfg['s_m']),
             kappa=float(cfg['kappa']), U=float(cfg['U']),
             amax=tuple(float(x) for x in cfg['amax']), v_c=float(cfg['v_c']))
    d_s = [float(x) for x in cfg['d_s']]
    c0 = [float(x) for x in cfg['c0']]
    v0 = rb._v0_of(list(cfg['eps0']), P['v_xi'])
    e0 = [float(cfg['e0_head'])] + [c0[i - 1] - d_s[i] for i in range(1, n)]
    profile = dict(t_d=float(cfg['t_d']), a_d=float(cfg['a_d']),
                   v_c1=float(cfg['v_c1']))
    return dict(P=P, d_s=d_s, v0=v0, e0=e0, profile=profile,
                eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
                transport=transport, age_bound=float(cfg['dbar']),
                dmax_steps=int(round(float(cfg['dbar']) / T_M)) - 1, cfg=cfg)


def _fit(fifo, T_END, h, Nm):
    """FIFO rows cover every send of the run (pad by the last value)."""
    if fifo is None:
        return None
    N = int(round(T_END / h))
    Nsend = (N - 1) // Nm + 1
    f = np.asarray(fifo)
    if f.shape[1] >= Nsend:
        return f
    pad = np.repeat(f[:, -1:], Nsend - f.shape[1], axis=1)
    return np.concatenate([f, pad], axis=1)


def simulate(B, h=H_FINE, scheme='cont', fifo=None, transport=None,
             T_u=None, rec_dt=0.01):
    """One run of the built design B; returns the sim_cont output with
    runtime_s."""
    Nm = int(round(T_M / h))
    t0 = time.time()
    R = sc.run(B['P'], B['d_s'], B['v0'], B['e0'], h=h, T_m=T_M, T_c=T_C,
               T_u=T_u, scheme=scheme,
               transport=B['transport'] if transport is None else transport,
               fifo=_fit(fifo, B['P']['T_END'], h, Nm),
               eps_e=B['eps_e'], eps_v=B['eps_v'], rec_dt=rec_dt,
               profile=B['profile'])
    R['runtime_s'] = time.time() - t0
    return R


def kpis(R, B):
    """run_cont KPI vector plus the Case C extras (B: built design)."""
    k = rc.kpis(R)
    k['tau_r'] = R['tau_r']
    k['mark_ref'] = R['mark_ref']
    k['profile'] = R['profile']
    k['n_eval'] = int(R['n_eval'])
    k['n_left_limits'] = int(R['n_left_limits'])
    k['root_resid'] = float(R['root_resid'])
    k['entry_obs_followers'] = R['entry_obs_followers']
    # entry into the handoff box completed before T_xi - dbar
    k['entry_deadline'] = B['P']['T_xi'] - B['age_bound']
    k['entry_before_deadline'] = bool(np.isfinite(R['entry_obs'])
                                      and R['entry_obs'] <= k['entry_deadline'])
    k['age'] = dict(max_left_limit=R['age']['max_left_limit'],
                    max_at_delivery=R['age']['max_at_delivery'])
    # ramp receipts t_rmp,i of the run's delivery schedule; fb_real[i-1]
    # is taken from t_rmp,i-1 to the standstill of unit i-1 (sim_cont,
    # round-4 fix M1)
    k['t_rmp'] = _py(R['t_rmp'])
    return k


def summary(k):
    """Short KPI line for the terminal."""
    return ('disp=%.4f s align=%.4f m gaperr=%.4f m gmin=%.4f m '
            'mmin_lim=%.4f n_mod=%d n_fb=%d n_sat=%d entry_obs=%.3f s '
            'T_f=%.4f s all_stopped=%s'
            % (k['disp'], k['align'], k['gaperr'], k['gmin_min'],
               k['mmin_lim_min'], k['n_mod'], k['n_fallback'], k['n_sat'],
               k['entry_obs'] if k['entry_obs'] is not None else float('nan'),
               k['T_f'], k['all_stopped']))


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(_py(obj), open(p, 'w'), indent=1)
    print('  -> %s' % os.path.relpath(p, CODE), flush=True)
    return p


def trajectory(B, R):
    """Decimated trajectory (every rec_dt) with the reference speed and
    the clearances."""
    rec = R['rec']
    P = B['P']
    t = np.asarray(rec['t'], float)
    ok = np.isfinite(t)
    t = t[ok]
    S0, VR, AR = sc.ref_profile(t, P, B['profile'])
    v = rec['v'][:, ok].copy()
    e = rec['e'][:, ok].copy()
    eps = rec['eps'][:, ok].copy()
    g = rec['g'][:, ok].copy()
    hb = rec['h'][:, ok].copy()
    # note n02 (round 4): sim_cont pads the record after the early exit
    # (every unit latched) with its last sample; the samples from the
    # exit on are recomputed here from the final state (record only;
    # u = 0 and m = NaN there already)
    pad = t >= R['t_end_sim'] - 0.5 * R['config']['h']
    if R['all_stopped'] and pad.any():
        s = np.asarray(R['s'], float)
        vf = np.zeros(len(s))                   # latched units are at rest
        d_s = np.asarray(B['d_s'], float)
        b = np.asarray(P['amax'], float) / P['v_c']
        v[:, pad] = vf[:, None]
        e[0, pad] = S0[pad] - s[0] - P['ell'] - d_s[0]
        eps[0, pad] = VR[pad] - vf[0]
        for i in range(1, len(s)):
            gap = s[i - 1] - s[i] - P['ell']
            e[i, pad] = gap - d_s[i]
            eps[i, pad] = vf[i - 1] - vf[i]
            g[i - 1, pad] = gap - P['s_m']
            hb[i - 1, pad] = gap - P['s_m'] + vf[i - 1] / b[i - 1] - vf[i] / b[i]
    return dict(t=t, v=v, e=e, eps=eps,
                u=rec['u'][:, ok], m=rec['m'][:, ok], g=g, c=g + P['s_m'],
                hbar=hb, v_r=VR, a_r=AR, s_ref=S0,
                tstop=R['tstop'], tset=R['tset'], tact=R['tact'],
                trule=R['trule'], marks=R['marks'], s=R['s'], egap=R['egap'],
                marker_signed=R['marker_signed'], e_hnd=R['e_hnd'],
                eps_hnd=R['eps_hnd'], d_s=np.array(B['d_s']),
                v0=np.array(B['v0']), e0=np.array(B['e0']))


def save_traj(path, B, R):
    T = trajectory(B, R)
    np.savez(path, keys_note=np.array(
        't [s] every rec_dt; v unit speeds; e pair POSITION errors e_i '
        '(row 0 = head vs reference); eps pair VELOCITY errors; u applied '
        'commands (right limits); m inactivity margins u_cbf - u_nom (pairs, '
        'NaN once latched); g clearance margins c_i - s_m; c clearances '
        's_{i-1} - s_i - ell (pairs); hbar barriers; v_r, a_r, s_ref '
        'reference speed, acceleration, position; marks markers; s final '
        'positions; egap signed terminal gap errors.'),
        config=np.array(json.dumps(_py(R['config']))),
        cfg=np.array(json.dumps(_py(B['cfg']))), **T)
    print('  -> %s' % os.path.relpath(path, CODE), flush=True)


def record(B, R):
    """Full JSON record of one run."""
    k = kpis(R, B)
    return dict(
        cfg=B['cfg'], P=B['P'], d_s=B['d_s'], v0=B['v0'], e0=B['e0'],
        profile=R['profile'], transport=R['config']['transport'],
        age_bound=B['age_bound'], kpis=k,
        tstop=R['tstop'], tset=R['tset'], tact=R['tact'], trule=R['trule'],
        e_hnd=R['e_hnd'], eps_hnd=R['eps_hnd'], egap=R['egap'],
        marker_signed=R['marker_signed'], marks=R['marks'], s=R['s'],
        gmin=R['gmin'], gmin_grid=R['gmin_grid'], hmin=R['hmin'],
        mmin=R['mmin'], mmin_lim=R['mmin_lim'], upk=R['upk'], wint=R['wint'],
        epsmax_b=R['epsmax_b'], fb_real=R['fb_real'],
        t_rmp=R['t_rmp'], fb_window_start=R['fb_window_start'],
        fb_window=('fb_real[i-1] = max |u*_{i-1} + a_b| from the ramp '
                   'receipt t_rmp,i-1 of unit i-1 (t_rmp,1 = T_xi; t_rmp,i '
                   'the first delivery to unit i of a packet sent by unit '
                   'i-1 at or after t_rmp,i-1) to the standstill of unit '
                   'i-1, compared with F^b_{i-1}'),
        filter_modifications=dict(min_branch=R['n_mod'],
                                  fallback=R['n_fallback'],
                                  saturation=R['n_sat'],
                                  evaluations=R['n_eval'],
                                  left_limit_evaluations=R['n_left_limits']),
        filter_inactive=k['filter_inactive'], hits_true=R['hits_true'],
        n_localized_events=R['n_localized'], root_residual=R['root_resid'],
        neg_speed_events=R['neg_speed'], contact_time=R['contact_time'],
        all_stopped=R['all_stopped'], entry_obs=R['entry_obs'],
        entry_obs_vel_only=R['entry_obs_vel_only'],
        entry_obs_pos_only=R['entry_obs_pos_only'],
        entry_obs_followers=R['entry_obs_followers'],
        entry_definition=('first time after which |e_i| <= eps_e and '
                          '|eps_i| <= eps_v for every i = 1..n (head: '
                          'reference errors) up to T_xi; the last exit is '
                          'localized by linear interpolation of the box '
                          'excess between integration steps'),
        age_audit=R['age'], runtime_s=R['runtime_s'], config=R['config'],
        system=('theorem system: continuous local law (T_u = h), RK4 '
                'between events, T_m = T_c = 1 ms, exact standstill '
                'sensing, head feedforward a_r(t) before T_xi and BRAKE at '
                'T_xi, zero-speed events localized; reference profile of '
                'Assumption 1\''))


# ------------------------------------------------------------------ parts
def reference(cfg, h=H_FINE, tag=None, save=True, **kw):
    """One continuous run at step h: KPIs and the decimated trajectory."""
    B = build(cfg)
    tag = tag or cfg['name']
    print('[ref] %s h=%g transport=%g s profile=%s' % (tag, h, B['transport'],
                                                       B['profile']), flush=True)
    R = simulate(B, h=h, **kw)
    k = kpis(R, B)
    print('  ' + summary(k), flush=True)
    print('  tstop=%s' % np.array2string(np.asarray(R['tstop']), precision=4))
    print('  tset =%s tact=%s' % (np.array2string(np.asarray(R['tset']), precision=3),
                                  np.array2string(np.asarray(R['tact']), precision=3)))
    print('  e_hnd=%s' % np.array2string(np.asarray(R['e_hnd']), precision=4))
    print('  eps_hnd=%s' % np.array2string(np.asarray(R['eps_hnd']), precision=4))
    print('  marker_signed=%s egap=%s'
          % (np.array2string(np.asarray(R['marker_signed']), precision=4),
             np.array2string(np.asarray(R['egap']), precision=4)))
    print('  entry_obs=%.4f (vel-only %.4f, pos-only %.4f) T_xi - dbar = %.3f'
          % (R['entry_obs'], R['entry_obs_vel_only'], R['entry_obs_pos_only'],
             B['P']['T_xi'] - B['age_bound']))
    print('  age: max held age %.4f s (bound %.3f s); runtime %.1f s'
          % (R['age']['max_left_limit'], B['age_bound'], R['runtime_s']))
    if save:
        dump('ref_%s.json' % tag, record(B, R))
        save_traj(os.path.join(OUT, 'traj_%s.npz' % tag), B, R)
    return R, k


def convergence(cfg, hs=H_DIAG + H_CONV, tag=None, save=True):
    """The reference run at every step in hs; differences to the finest."""
    B = build(cfg)
    tag = tag or cfg['name']
    print('[conv] %s h in %s' % (tag, list(hs)), flush=True)
    rows = []
    for h in hs:
        R = simulate(B, h=h)
        k = kpis(R, B)
        rows.append(dict(h=h, diagnostic=h in H_DIAG, kpis=k))
        print('  h=%-7g %s (%.1f s)' % (h, summary(k), R['runtime_s']), flush=True)
    ref = rows[-1]['kpis']
    for row in rows:
        row['diff_to_finest'] = rc.diff(row['kpis'], ref)
        mx, kx, bad = rc.dmax(row['diff_to_finest'])
        row['max_diff_to_finest'] = mx
        row['argmax_kpi'] = kx
        row['nan_mismatch'] = bad
        print('  h=%-7g max|diff to finest| = %.3e (%s) nan_mismatch=%s'
              % (row['h'], mx, kx, bad), flush=True)
    conv_rows = [r for r in rows if not r['diagnostic']]
    worst = {key: max(r['diff_to_finest'][key] for r in conv_rows)
             for key in rc.KPI_KEYS}
    mx, kx, bad = rc.dmax(worst)
    out = dict(study=('numerical convergence: same event schedule (sends '
                      'every T_m = 1 ms, constant transport, detector T_c = '
                      '1 ms anchored at activation), same continuous law; '
                      'only the RK4 step h varies'),
               cfg=cfg, h_values=list(hs), diagnostic_h=list(H_DIAG),
               rows=rows, max_diff_over_h_by_kpi=worst,
               max_diff_over_h_and_kpis=mx, argmax_kpi=kx, nan_mismatch=bad)
    if save:
        dump('convergence_%s.json' % tag, out)
    return out


def schedules(cfg, n=20, seed=100, tag=None, dmax=None, h=H_SCHED, save=True):
    """n random FIFO schedules (per-send transport d_j <= dmax steps,
    run_b2._random_schedule) with the held-age audit; 'cont' at h."""
    B = build(cfg)
    tag = tag or cfg['name']
    dmax = B['dmax_steps'] if dmax is None else int(dmax)
    rng = np.random.default_rng(seed)
    Nsched = int(round(B['P']['T_END'] / T_M)) + 2
    print('[sched] %s n=%d seed=%d d_j <= %d steps h=%g' % (tag, n, seed, dmax, h),
          flush=True)
    rows = []
    t0 = time.time()
    for j in range(n):
        Dl = rb._random_schedule(rng, Nsched, dmax)
        R = simulate(B, h=h, fifo=Dl)
        k = kpis(R, B)
        age_ok = R['age']['max_left_limit'] <= B['age_bound'] + 1e-12
        rows.append(dict(index=j, kpis=k, age_audit=R['age'],
                         age_ok=bool(age_ok),
                         schedule_summary=dict(d_min=int(Dl[1:].min()),
                                               d_max=int(Dl[1:].max()),
                                               d_mean=float(Dl[1:].mean()))))
        print('  %3d %s age_sup=%.4f s (%.0f s)'
              % (j, summary(k), R['age']['max_left_limit'], time.time() - t0),
              flush=True)
    K = [r['kpis'] for r in rows]
    agg = dict(
        n=n, seed=seed, dmax_steps=dmax, h=h,
        disp_max=max(k['disp'] for k in K),
        align_max=max(k['align'] for k in K),
        gaperr_max=max(k['gaperr'] for k in K),
        sched_max=max(k['sched'] for k in K),
        gmin_min=min(k['gmin_min'] for k in K),
        mmin_lim_min=min(k['mmin_lim_min'] for k in K),
        entry_obs_max=max(k['entry_obs'] for k in K),
        entry_obs_min=min(k['entry_obs'] for k in K),
        T_f_max=max(k['T_f'] for k in K),
        n_mod_total=sum(k['n_mod'] for k in K),
        n_fallback_total=sum(k['n_fallback'] for k in K),
        n_sat_total=sum(k['n_sat'] for k in K),
        all_filter_inactive=all(k['filter_inactive'] for k in K),
        all_stopped=all(k['all_stopped'] for k in K),
        all_age_ok=all(r['age_ok'] for r in rows),
        age_sup_max=max(r['age_audit']['max_left_limit'] for r in rows),
        age_bound=B['age_bound'])
    print('  aggregate: disp_max=%.4f align_max=%.4f gaperr_max=%.4f gmin_min=%.4f '
          'mmin_lim_min=%.4f entry_obs_max=%.3f T_f_max=%.4f n_mod=%d '
          'filter_inactive=%s age_ok=%s'
          % (agg['disp_max'], agg['align_max'], agg['gaperr_max'], agg['gmin_min'],
             agg['mmin_lim_min'], agg['entry_obs_max'], agg['T_f_max'],
             agg['n_mod_total'], agg['all_filter_inactive'], agg['all_age_ok']),
          flush=True)
    out = dict(study=('random FIFO schedules: piecewise-constant per-hop '
                      'transport (segments 0.5-5 s, d_j uniform in 1..dmax '
                      'steps of T_m = 1 ms), FIFO projection at the receiver, '
                      'held age <= (dmax + 1) T_m; continuous law at h = %g s'
                      % h), cfg=cfg, aggregate=agg, rows=rows)
    if save:
        dump('schedules_%s.json' % tag, out)
    return out


def sweep(cfg, dbars=DBARS, n_sched=3, seed=200, tag=None, h=H_FINE, save=True):
    """Age sweep: constant transport dbar - T_m at every age bound (fine
    step) plus n_sched random schedules with d_j <= dbar/T_m - 1 (h = 1 ms)."""
    B = build(cfg)
    tag = tag or cfg['name']
    print('[sweep] %s age bounds %s' % (tag, list(dbars)), flush=True)
    rows = []
    for db in dbars:
        steps = int(round(db / T_M))
        tr = (steps - 1) * T_M
        R = simulate(B, h=h, transport=tr)
        k = kpis(R, B)
        row = dict(age_bound=db, transport=tr, transport_steps=steps - 1,
                   kpis=k, age_audit=R['age'], random=[])
        print('  dbar=%.3f const transport %.3f: %s (%.1f s)'
              % (db, tr, summary(k), R['runtime_s']), flush=True)
        if steps - 1 >= 1 and n_sched > 0:
            rng = np.random.default_rng(seed + steps)
            Nsched = int(round(B['P']['T_END'] / T_M)) + 2
            for j in range(n_sched):
                Dl = rb._random_schedule(rng, Nsched, steps - 1)
                Rr = simulate(B, h=H_SCHED, fifo=Dl)
                kr = kpis(Rr, B)
                row['random'].append(dict(index=j, kpis=kr, age_audit=Rr['age'],
                                          age_ok=bool(Rr['age']['max_left_limit']
                                                      <= db + 1e-12)))
                print('    random %d: %s age_sup=%.4f' % (j, summary(kr),
                                                          Rr['age']['max_left_limit']),
                      flush=True)
        rows.append(row)
    out = dict(study=('age sweep: constant transport dbar - T_m (held age '
                      'exactly dbar) at h = %g s, plus %d random schedules per '
                      'age bound with d_j <= dbar/T_m - 1 at h = %g s'
                      % (h, n_sched, H_SCHED)), cfg=cfg, dbars=list(dbars),
               rows=rows)
    if save:
        dump('sweep_%s.json' % tag, out)
    return out


PARTS = dict(ref=reference, conv=convergence, sched=schedules, sweep=sweep)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('parts', nargs='*', default=['all'])
    ap.add_argument('--cfg', default=None, help='design JSON (merged over PROVISIONAL)')
    ap.add_argument('--tag', default=None, help='output name (default cfg name)')
    ap.add_argument('--h', type=float, default=None, help='step of the ref part')
    ap.add_argument('--n', type=int, default=20, help='schedules of the sched part')
    ap.add_argument('--seed', type=int, default=100)
    ap.add_argument('--dbars', type=float, nargs='*', default=None)
    ap.add_argument('--n-sweep-sched', type=int, default=3)
    # design overrides
    for key in ('lam', 'a_d', 't_d', 'v_c1', 'T_xi', 'alpha', 'dbar'):
        ap.add_argument('--' + key, type=float, default=None)
    ap.add_argument('--c0', type=float, nargs='*', default=None)
    a = ap.parse_args(argv)
    ov = {k: getattr(a, k) for k in ('lam', 'a_d', 't_d', 'v_c1', 'T_xi', 'alpha',
                                      'dbar')}
    if a.c0:
        ov['c0'] = a.c0 if len(a.c0) > 1 else a.c0 * 4
    cfg = load_cfg(a.cfg, ov)
    if a.tag:
        cfg['name'] = a.tag
    todo = a.parts if a.parts != ['all'] else ['ref', 'conv', 'sched', 'sweep']
    for p in todo:
        t0 = time.time()
        if p == 'ref':
            reference(cfg, h=a.h or H_FINE)
        elif p == 'conv':
            convergence(cfg)
        elif p == 'sched':
            schedules(cfg, n=a.n, seed=a.seed)
        elif p == 'sweep':
            sweep(cfg, dbars=tuple(a.dbars) if a.dbars else DBARS,
                  n_sched=a.n_sweep_sched)
        else:
            raise SystemExit('unknown part %r' % p)
        print('[%s] done in %.1f s' % (p, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
