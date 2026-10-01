#!/usr/bin/env python3
"""run_caseD_validate.py -- validation of the selected Case D design
(deceleration-only arrival by planned closure; round 4 and round 5,
2026-09-27).

Design: data/new_d19/caseD/design.json (run_caseD_design.py).  Round 5:
the fan plan (family 'fan' of certify_plan.plan: approach of t_0 = 10 s at
the crawl rate a_c, common onset t_0 of the main deceleration with
unit-specific constant rates a_i = a_c + G/L_i, joins onto the common crawl
line at t^c_i = t_0 + L_i, tail crawl of 5 s, synchronized braking at a_b
from T_xi; every jump instant on the 1 ms grid).  Round 4: the canonical
plan (staggered onsets at a_d = a_b); the parts handle both families
(plan_events, ens_plan, part_edge, _ref_plan).
Certificate: src/certification/certify_plan.py (Proposition D1 of
data/new_d19/caseD/theory_caseD.md).  Simulator: src/simulation/sim_cont.py
with plan= (the Case D laws of Sec. 2.4 of the theory note: pair errors
relative to the plan, local planned feedforward, deviation message formed
at the send instant, filter estimate a*_{i-1}(t) + received deviation,
input set [-k, 0] with fallback -k, braking in every plan at T_xi, BRAKE,
RULE and LATCH).  Independent simulators: data/new_d19/caseD/
design_checks/sim_plan.py (design check) and ref_sim.py (referee).
Every output goes to data/new_d19/caseD/.

Parts
  regress  default behaviour of sim_cont.py (plan=None) after the plan
           option: the Case A and Case B continuous references, the legacy
           scheme against sim_root2 and the Case C reference re-run into a
           temporary directory and compared leaf by leaf with the archived
           files (only run times may differ) -> sim_cont_regression.json
  exact    exact event handling of sim_cont.run(plan=...) at every planned
           jump (round 5): from a takeover state on the plan the law
           applies u_i = a*_i, so the simulated speeds must equal the
           planned ones to rounding; the selected design, its off-grid
           alternative and a fan with unequal closures, 'cont' and 'zoh'
           at h = 25 us and 1 ms; control with every jump one step late
           -> plan_exactness.json
  ref      continuous reference ('cont': continuous local law, RK4 at
           h = 25 us, T_m = T_c = 1 ms, constant transport 19 ms, held age
           20 ms): every observation beside its certified bound, phase
           times and extrema -> reference.json, traj_caseD.npz (10 ms)
  conv     step halving h in {1e-4, 5e-5, 2.5e-5, 1.25e-5} s (plus 1e-3
           and 5e-4 s as diagnostic rows), differences to h = 12.5 us
           -> convergence.json
  digital  zero-order hold with T_u in {1, 0.5, 0.25, 0.1} ms at h = 25 us
           (event-triggered updates at the unit's own events) and the
           pure 1 kHz run at h = 1 ms, against the reference
           -> digital.json
  sched    300 random FIFO schedules (seed 500 + j; piecewise-constant
           per-hop transport, segments 0.5 to 5 s, d_j uniform in 1..19
           steps of T_m, as run_b2._random_schedule), 'cont' at h = 1 ms,
           held-age audit -> schedules.json
  adv      adversarial schedule families (maximal transport around every
           planned jump instant, around the activations and S2 switches,
           around T_xi, minimal elsewhere; bursts; sawtooth batches;
           per-hop patterns), 56 schedules, 'cont' at h = 1 ms
           -> adversarial.json
  sweep    age bounds 5 ms to 5000 ms (the certified age boundary of
           design.json and the next age included): certificate per age,
           constant transport (age - 1 ms) at h = 25 us and 10 random
           schedules per age at h = 1 ms; per age the settling time,
           marker and gap errors, command range against [-k, 0], filter
           modifications, entry, and the first failing certificate
           condition, certified bound and requirement -> sweep.json
  ens      100 takeover states around the nominal state (clearances,
           head speed, speed increments within stated ranges), plan by
           the design rule (ens_plan; the fan rule of
           run_caseD_design.fan_design_for with the drawn clearances),
           certificate and continuous run for every state (rejected
           states simulated too) -> ensemble.json
  edge     the input-set guarantee at its edge: the certified design
           without the design margins, the slow rate lowered through
           F^+_5(0) ((H1D)(i) fails) and the main rate of the head (fan)
           or the main deceleration a_d (canonical) raised toward k
           ((H1D)(ii) fails), T_xi kept, h = 25 us -> edge.json
  ps1      takeover states against the premise P_S1 (pair-3 speed
           increment from +0.03 to -0.05 m/s, plans kept) under constant
           transport and three FIFO schedules; the certificate with the
           one-sided box of the design and, for comparison, with the box
           rule of the certificate (symmetric box once P_S1 fails)
           -> ps1_states.json
  indep    the reference against sim_plan.py and ref_sim.py on the same
           delays: stop instants, settling time, marker and gap errors,
           command extrema, switch instants, entry -> independent.json
  check    every observation against its certified bound, every bounded
           row compared (a missing observation fails), every command in
           [-k, 0] before LATCH and exactly 0 after, zero filter
           modifications at certified ages; ALL OK required -> check.json
  all      regress exact ref conv digital sched adv sweep ens edge ps1 indep
           check

The reference part also records the run every 1 ms and compares
Proposition D1(b) pointwise (the deviation u_i - a*_i against
[-F^-_i(t), F^+_i(t)] at every record instant before T_xi), the layer
velocity envelope of (H6D), and the S1 stages of Lemma D3 (eps_k <= 0 on
[0, t^set_k], eps_k <= -phi on [t^act_k, t^set_k]).  Round-4 review
(2026-09-27): a missing observation where a bound exists now fails
(bounds_rows), every run stores n_checked and n_rows, and the check part
requires n_checked == n_rows for every run at a certified age.  Round 5:
the reference part also checks that every unit's speed is nonincreasing
on the 1 ms record (row speed_nonincreasing).  Round-5 review: the
reference part also stores the speed order of every pair on the 1 ms
record (speed_order: the record instants on the main piece at which a
follower was not faster than its predecessor, which lie only right after
the common onset and right before the join, their windows and the largest
deficit, and the clearances against the parking gap).

Terminology.  "Settling time" is the interval from the first to the last
stop, Delta_tau = max_i tau_i - min_i tau_i (the main text calls it the
stopping-time dispersion).  The S1 detection instant t^set_i, the matching
flag and the detection deadline are the main-text "settling" quantities of
the S1 speed-matching detector.

Usage: python run_caseD_validate.py [part ...]
"""
import json
import os
import sys
import time
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import numpy as np                                      # noqa: E402
import certify_plan as cq                               # noqa: E402
import sim_cont as sc                                   # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'caseD')
CHECKS = os.path.join(OUT, 'design_checks')
T_M = 1e-3
T_C = 1e-3
H_FINE = 2.5e-5
H_SCHED = 1e-3
H_CONV = (1e-4, 5e-5, 2.5e-5, 1.25e-5)
H_DIAG = (1e-3, 5e-4)
TU_DIG = (1e-3, 5e-4, 2.5e-4, 1e-4)
T_TAIL = 8.0                          # T_END = T_xi + 8 s
# age grid of the sweep; part_sweep adds the certified age boundary of
# design.json and the next age (round 4: 23 and 24 ms; round 5: 72, 73 ms)
AGES_MS = (5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 100, 150, 200,
           250, 300, 400, 500, 750, 1000, 1500, 2000, 3000, 5000)
KPI_KEYS = ('tstop', 'settling', 'T_f', 'marker_signed', 'gap_signed',
            'gmin', 'hmin', 'mmin_lim', 'u_min_unit', 'u_max_unit',
            'u_max_pre_unit', 'entry_obs', 'e_hnd', 'eps_hnd', 'e_sw',
            'eps_sw', 'fb_unit', 'wint', 'epsmax_b', 'ebmax_b', 'tset', 'tact')


# ------------------------------------------------------------ helpers
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
    if isinstance(o, Fr):
        return float(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return None if not np.isfinite(x) else x
    return o


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    json.dump(_py(obj), open(p, 'w'), indent=1)
    print('  -> data/new_d19/caseD/%s' % name, flush=True)
    return p


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def _fin(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def _max(xs):
    f = _fin(xs)
    return max(f) if f else None


def _min(xs):
    f = _fin(xs)
    return min(f) if f else None


def arg_ext(rows, key, fn=max):
    """(extreme value, name of the run) over rows[name][key]."""
    vals = [(r[key], nm) for nm, r in rows if r.get(key) is not None
            and np.isfinite(r[key])]
    if not vals:
        return None, None
    v = fn(vals, key=lambda z: z[0])
    return v[0], v[1]


def random_schedule(rng, N, dmax, n=5):
    """Piecewise-constant random per-hop transport (steps of T_m): segments
    of U(0.5, 5) s, d uniform in 1..dmax per segment; the generator of
    run_b2._random_schedule, with the FIFO projection at the receiver."""
    d = np.empty((n, N), int)
    for i in range(1, n):
        k = 0
        while k < N:
            seg = int(rng.uniform(0.5, 5.0) / 1e-3)
            d[i, k:k + seg] = rng.integers(1, dmax + 1)
            k += seg
    d[0] = 1
    return d


def fifo_project(d):
    """Projected transport d'_j = max_{l <= j}(l + d_l) - j (monotone
    arrivals, the FIFO projection of sim_cont.deliveries)."""
    j = np.arange(d.shape[1])
    out = d.copy()
    for i in range(1, d.shape[0]):
        out[i] = np.maximum.accumulate(j + d[i]) - j
    return out


# ------------------------------------------------------------- design
def load_design():
    return json.load(open(os.path.join(OUT, 'design.json')))


def design_cfg():
    return dict(load_design()['cfg'])


def certificate(cfg, **kw):
    return cq.evaluate(dict(cfg, **kw))


def plan_events(Pl):
    """Onsets t^d_i of the main deceleration and crawl joins t^c_i (floats)
    of a certify_plan.Plan: canonical family, staggered onsets; fan family,
    the common onset t_0 for every unit."""
    tc = [float(x) for x in Pl.extra['t_c']]
    if Pl.family == 'fan':
        td = [float(Pl.extra['t_0'])] * Pl.n
    else:
        td = [float(x) for x in Pl.extra['t_d']]
    return td, tc


def build(cfg):
    """Simulator inputs of a Case D configuration (certify_plan format)."""
    Pl = cq.plan(cfg)
    lam = float(cfg['lam'])
    n = int(cfg['n'])
    T_xi = float(Pl.T_xi)
    dbar = float(cfg['dbar'])
    P = dict(n=n, dt=1e-3, dbar=dbar - T_M, a_b=float(cfg['a_b']),
             v_xi=float(Pl.v_xi), T_xi=T_xi, T_END=T_xi + T_TAIL,
             alpha=float(cfg['alpha']), beta=lam * lam, gamma=2.0 * lam,
             phi=float(cfg['phi']), eps_det=float(cfg['eps_det']),
             ell=cq.FIXED['ell'], s_m=float(cfg['s_m']),
             kappa=cq.FIXED['kappa'], U=cq.FIXED['U'],
             amax=[float(x) for x in cfg['amax']], v_c=cq.FIXED['v_c'])
    d_s = [float(x) for x in cfg['d_s']]
    c0 = [float(x) for x in cfg['c0']]
    e0 = [0.0] + [c0[i - 1] - d_s[i] for i in range(1, n)]
    v0 = [float(cq.F(x)) for x in cfg['v0']]
    return dict(P=P, d_s=d_s, v0=v0, e0=e0, plan=sc.plan_from(Pl, cfg['k']),
                Pl=Pl, eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
                transport=dbar - T_M, age_bound=dbar,
                dmax_steps=int(round(dbar / T_M)) - 1, cfg=cfg,
                k=float(cfg['k']))


def _fit(fifo, T_END, h, Nm):
    if fifo is None:
        return None
    N = int(round(T_END / h))
    Nsend = (N - 1) // Nm + 1
    f = np.asarray(fifo)
    if f.shape[1] >= Nsend:
        return f
    pad = np.repeat(f[:, -1:], Nsend - f.shape[1], axis=1)
    return np.concatenate([f, pad], axis=1)


def simulate(B, h=H_FINE, scheme='cont', fifo=None, transport=None, T_u=None,
             rec_dt=0.01):
    Nm = int(round(T_M / h))
    t0 = time.time()
    R = sc.run(B['P'], B['d_s'], B['v0'], B['e0'], h=h, T_m=T_M, T_c=T_C,
               T_u=T_u, scheme=scheme,
               transport=B['transport'] if transport is None else transport,
               fifo=_fit(fifo, B['P']['T_END'], h, Nm), eps_e=B['eps_e'],
               eps_v=B['eps_v'], rec_dt=rec_dt, plan=B['plan'])
    R['runtime_s'] = time.time() - t0
    return R


def kpis(R, B):
    """Observations of one run (errors relative to the plan)."""
    n = len(R['tstop'])
    k = B['k']
    tstop = np.asarray(R['tstop'], float)
    order = R['stop_order']
    umin = float(np.nanmin(R['u_min']))
    umax = float(np.nanmax(R['u_max']))
    out = dict(
        tstop=tstop, settling=R['disp'], T_f=R['T_f'],
        first_stop=float(np.nanmin(tstop)) if np.any(np.isfinite(tstop)) else None,
        stop_order=order,
        tail_first_head_last=bool(order is not None and order[0] == n
                                  and order[-1] == 1),
        stop_offsets=tstop - R['tau_r'], sched=R['sched'],
        stop_increments=np.abs(np.diff(tstop)),
        marker_signed=R['marker_signed'], marker_abs_max=R['align'],
        gap_signed=np.asarray(R['egap'])[1:], gap_abs_max=R['gaperr'],
        head_slot_err=R['head_slot_err'],
        u_min_unit=R['u_min'], u_max_unit=R['u_max'], u_min=umin, u_max=umax,
        u_min_pre_unit=R['u_min_pre'], u_max_pre_unit=R['u_max_pre'],
        u_min_pre=float(np.nanmin(R['u_min_pre'])),
        u_max_pre=float(np.nanmax(R['u_max_pre'])),
        u_abs_after_latch=float(np.max(R['u_abs_latched'])),
        commands_in_set=bool(umin >= -k and umax <= 0.0),
        zero_after_latch=bool(float(np.max(R['u_abs_latched'])) == 0.0),
        n_mod=int(R['n_mod']), n_fallback=int(R['n_fallback']),
        n_sat=int(R['n_sat']),
        filter_modifications=int(R['n_mod'] + R['n_fallback'] + R['n_sat']),
        mmin_lim=R['mmin_lim'], mmin_lim_min=float(np.min(R['mmin_lim'])),
        mmin_phase=R['mmin_phase'],
        gmin=R['gmin'], gmin_min=float(np.min(R['gmin'])),
        hmin=R['hmin'], hmin_min=float(np.min(R['hmin'])),
        htil_min=R['htil_min'], h_brk_min=R['h_brk_min'],
        entry_obs=R['entry_obs'], entry_obs_followers=R['entry_obs_followers'],
        entry_obs_vel_only=R['entry_obs_vel_only'],
        entry_obs_pos_only=R['entry_obs_pos_only'],
        e_hnd=R['e_hnd'], eps_hnd=R['eps_hnd'],
        tact=R['tact'], tset=R['tset'], trule=R['trule'],
        rule_entered=R['rule_entered'], e_sw=R['e_sw'], eps_sw=R['eps_sw'],
        fb_unit=R['fb_unit'], fb_real=R['fb_real'], t_rmp=R['t_rmp'],
        fb_window=R['fb_window'], wint=R['wint'], epsmax_b=R['epsmax_b'],
        ebmax_b=R['ebmax_b'], vmin_pre=R['vmin_pre'],
        vmin_pre_min=float(np.min(R['vmin_pre'])),
        dev_pos=R['dev_pos'], dev_neg=R['dev_neg'],
        all_stopped=R['all_stopped'], contact_time=R['contact_time'],
        age=dict(max_left_limit=R['age']['max_left_limit'],
                 max_at_delivery=R['age']['max_at_delivery']),
        hits_true=R['hits_true'], n_eval=R['n_eval'],
        n_left_limits=R['n_left_limits'], n_localized=R['n_localized'],
        root_resid=R['root_resid'], neg_speed=R['neg_speed'],
        runtime_s=R.get('runtime_s'), config=R['config'])
    out['filter_inactive'] = bool(out['filter_modifications'] == 0
                                  and out['mmin_lim_min'] > 0.0)
    return out


def summary(k):
    return ('settling=%.5f s T_f=%.4f s marker<=%.4f m gap<=%.4f m u in '
            '[%.4f, %.4f] (pre max %.4f) mods=%d entry=%.3f s order=%s rule=%s'
            % (k['settling'], k['T_f'], k['marker_abs_max'], k['gap_abs_max'],
               k['u_min'], k['u_max'], k['u_max_pre'], k['filter_modifications'],
               k['entry_obs'] if k['entry_obs'] is not None else float('nan'),
               k['stop_order'], k['rule_entered']))


def requirements(cfg, k):
    """The authors' requirements and the structural requirements."""
    r = dict(
        settling=bool(k['all_stopped'] and k['settling'] <= float(cfg['sync_tol'])),
        marker=bool(k['all_stopped'] and k['marker_abs_max'] <= float(cfg['align_tol'])),
        gap=bool(k['all_stopped'] and k['gap_abs_max'] <= float(cfg['gap_tol'])),
        input_set_unmodified=bool(k['n_sat'] == 0),
        filter_inactive=bool(k['filter_modifications'] == 0),
        no_contact=k['contact_time'] is None,
        all_stopped=bool(k['all_stopped']))
    failed = [nm for nm, v in r.items() if v is False]
    r['all_ok'] = not failed
    r['failed'] = failed
    return r


# ------------------------------------------------------- bounds table
def marker_bounds(K, cfg):
    a_b = float(cfg['a_b'])
    eps_e = float(cfg['eps_e'])
    eps_v = float(cfg['eps_v'])
    head = eps_e + K['v_c1'] * eps_v / a_b + eps_v ** 2 / (2 * a_b)
    n = len(K['tset'])
    return [head + sum(K['gaperr'][:i]) for i in range(n)]


def _piece_index(pieces_i, t0):
    for j, p in enumerate(pieces_i):
        if abs(p['t0'] - t0) < 1e-9:
            return j
    return None


def bounds_rows(K, cfg, k, pieces, extra=None):
    """Each observation of kpis k against its certified bound (K: consts
    of certify_plan.evaluate; pieces: the simulator's plan pieces)."""
    n = len(K['tset'])
    b = [float(a) / cq.FIXED['v_c'] for a in cfg['amax']]
    NP = float(K['NUMPAD'])
    kk = float(cfg['k'])
    eps_det = float(cfg['eps_det'])
    rows = {}

    def put(name, bound, obs, kind, unit, note=None, optional=False):
        """A missing observation (None or NaN) where a bound exists is a
        failure, unless the row is optional (round-4 review, finding
        validation_logic-04); a missing bound leaves the entry unchecked."""
        def cmp(x, y):
            if x is None:
                return None
            if y is None or not np.isfinite(y):
                return None if optional else False
            return bool(y <= x) if kind == 'upper' else bool(y >= x)
        if isinstance(bound, (list, tuple, np.ndarray)):
            bb = [None if x is None else float(x) for x in bound]
            oo = [None if x is None else float(x) for x in obs]
            assert len(bb) == len(oo), name
            oks = [cmp(x, y) for x, y in zip(bb, oo)]
            chk = [z for z in oks if z is not None]
            rows[name] = dict(bound=bb, observed=oo, kind=kind, unit=unit,
                              ok=None if not chk else all(chk), ok_each=oks)
        else:
            x = None if bound is None else float(bound)
            y = None if obs is None else float(obs)
            rows[name] = dict(bound=x, observed=y, kind=kind, unit=unit, ok=cmp(x, y))
        if note:
            rows[name]['note'] = note

    put('settling_time', K['settling_bound'], k['settling'], 'upper', 's',
        'first-to-last stop interval (stopping-time dispersion) vs sum_k q_k')
    put('stop_increments', K['q'], k['stop_increments'], 'upper', 's',
        '|tau_i - tau_{i-1}| vs q_i')
    put('final_standstill', K['T_f_bar'], k['T_f'], 'upper', 's')
    put('stop_offsets', K['sched'], k['sched'], 'upper', 's',
        'max_i |tau_i - tau*|')
    put('marker_error', K['align'], k['marker_abs_max'], 'upper', 'm')
    put('marker_error_units', marker_bounds(K, cfg),
        np.abs(k['marker_signed']), 'upper', 'm',
        'per unit: head term + sum of the gap bounds ahead')
    put('terminal_gap_error_pairs', K['gaperr'], np.abs(k['gap_signed']),
        'upper', 'm', 'pairs 2..n')
    # pair keys are ints in a fresh evaluation and strings after JSON
    pr = {str(key): val for key, val in K['pairs'].items()}
    put('clearance_floor_pairs',
        [min(pr[str(i + 1)]['gmin_acq'], pr[str(i + 1)]['gmin_s2'],
             pr[str(i + 1)]['gmin_brk']) for i in range(1, n)],
        k['gmin'], 'floor', 'm', 'g_i = c_i - s_m, intersample minimum')
    put('barrier_nonnegative', [0.0] * (n - 1), k['hmin'], 'floor', 'm')
    put('barrier_deviation_acquisition',
        [pr[str(i + 1)]['h_tilde0'] - pr[str(i + 1)]['H1'] for i in range(1, n)],
        k['htil_min'][0], 'floor', 'm',
        'min of h_i - h*_i in COAST and S1 vs h~_i(0) - H1_i')
    put('barrier_deviation_s2',
        [pr[str(i + 1)]['h_tilde0'] - pr[str(i + 1)]['H1'] - pr[str(i + 1)]['Ht']
         for i in range(1, n)],
        k['htil_min'][1], 'floor', 'm',
        'min of h_i - h*_i in S2 before T_xi vs h~_i(0) - H1_i - H_tail,i')
    put('barrier_braking', [pr[str(i + 1)]['h_b'] for i in range(1, n)],
        k['h_brk_min'], 'floor', 'm', 'min of h_i from T_xi vs h_b,i')
    for ph, key in ((0, 'slack_i'), (1, 'slack_ii'), (2, 'slack_iii')):
        put('inactivity_margin_phase%d' % (ph + 1),
            [b[i] * (pr[str(i + 1)][key] + NP) for i in range(1, n)],
            k['mmin_phase'][ph], 'floor', 'm/s^2',
            'u_cbf - u_nom (event left limits included) vs b_i (H4D slack '
            '+ pad)')
    put('inactivity_margin', min(pr[str(i + 1)]['sigma'] for i in range(1, n)),
        k['mmin_lim_min'], 'floor', 'm/s^2')
    put('command_lower', -kk, k['u_min'], 'floor', 'm/s^2',
        'every command before LATCH >= -k')
    put('command_upper', 0.0, k['u_max'], 'upper', 'm/s^2',
        'every command before LATCH <= 0')
    put('command_after_latch', 0.0, k['u_abs_after_latch'], 'upper', 'm/s^2',
        '|u| after LATCH (exactly 0)')
    fp = []; fo = []; mp = []; mo = []
    for row in K['authority']:
        i = row['unit'] - 1
        j = _piece_index(pieces[i], row['t0'])
        dp = None if j is None else k['dev_pos'][i][j]
        dm = None if j is None else k['dev_neg'][i][j]
        fp.append(row['F_plus']); fo.append(dp)
        mp.append(row['F_minus']); mo.append(dm)
    put('deviation_upper_pieces', fp, fo, 'upper', 'm/s^2',
        'max (u_i - a*_i) on each plan piece before T_xi vs F^+_i (H1D(i))')
    put('deviation_lower_pieces', mp, mo, 'upper', 'm/s^2',
        'max (a*_i - u_i) on each plan piece before T_xi vs F^-_i (H1D(ii))')
    put('braking_envelope', K['Fb_units'], k['fb_unit'], 'upper', 'm/s^2',
        'max |u_i + a_b| from the ramp receipt T_xi to the standstill vs F^b_i')
    put('braking_mismatch_integral', K['W'], k['wint'], 'upper', 'm/s',
        'int |u~_{i-1}(t) - received deviation| over the braking window vs W_i')
    put('braking_speed_band', K['epsb'], k['epsmax_b'], 'upper', 'm/s')
    put('braking_position_band', K['eb'], k['ebmax_b'], 'upper', 'm')
    put('detection_deadlines', K['tset'], k['tset'], 'upper', 's',
        'S1 detection instants t^set_i vs the detection deadlines')
    put('activation_deadlines', K['tact'], k['tact'], 'upper', 's')
    put('switch_position_lower', [-x for x in K['E1']], k['e_sw'], 'floor', 'm',
        'one-sided switch box: e^sw_i >= -E1_i')
    put('switch_position_upper', [0.0] * n, k['e_sw'], 'upper', 'm')
    put('switch_speed_lower', [-eps_det] * n, k['eps_sw'], 'floor', 'm/s')
    put('switch_speed_upper', [0.0] * n, k['eps_sw'], 'upper', 'm/s')
    put('persistent_entry', K['T_ent_bar'], k['entry_obs'], 'upper', 's')
    put('handoff_position_error', float(cfg['eps_e']),
        max(abs(x) for x in k['e_hnd']), 'upper', 'm')
    put('handoff_velocity_error', float(cfg['eps_v']),
        max(abs(x) for x in k['eps_hnd']), 'upper', 'm/s')
    put('velocity_floor', K['vfloor_min'], k['vmin_pre_min'], 'floor', 'm/s')
    put('filter_modifications', 0, k['filter_modifications'], 'upper', 'count')
    rows['all_stopped'] = dict(ok=bool(k['all_stopped']), kind='bool')
    rows['no_contact'] = dict(ok=k['contact_time'] is None, kind='bool')
    if extra:
        rows.update(extra)
    ok = all(r.get('ok') in (True, None) for r in rows.values())
    return dict(rows=rows, all_ok=bool(ok),
                n_checked=sum(1 for r in rows.values() if r.get('ok') is not None),
                n_rows=len(rows),
                unchecked=[nm for nm, r in rows.items() if r.get('ok') is None],
                failed=[nm for nm, r in rows.items() if r.get('ok') is False])


def authority_envelopes(cfg, K):
    """F^+_i(t) and F^-_i(t) of eq. (D.14) of case_d.tex at arbitrary
    instants (the certificate evaluates them only at the left ends of the
    plan pieces); returns Fpm(i, t) -> (F^+_i(t), F^-_i(t)), i 0-based.
    Checked against the piece values of the certificate (K['authority'])."""
    G = cq.namespace(cfg)
    lam = G['lam']; gamma = G['gamma']; alpha_g = G['alpha_g']
    dbar = float(cfg['dbar']); eps_det = float(cfg['eps_det'])
    n = int(cfg['n'])
    tset = np.asarray(K['tset'], float)
    E1 = np.asarray(K['E1'], float)
    Lam1 = np.asarray(K['Lam1'], float)
    Lam2h = np.asarray(K['Lam2'], float).copy()
    Lam2h[0] = 0.0                                   # Lambda^(2)_1 := 0
    eps0 = np.asarray(K['eps0'], float)
    corners = [cq.box_corners(0.0, E1[k], eps_det, bool(K['one_sided'])) for k in range(n)]
    sig_p = [1.0 if eps0[k] + Lam1[k] > eps_det else 0.0 for k in range(n)]
    sig_m = [1.0 if eps0[k] - Lam1[k] < -eps_det else 0.0 for k in range(n)]

    def Fpm(i, t):
        t = np.atleast_1d(np.asarray(t, float))
        fp = np.zeros_like(t); fm = np.zeros_like(t)
        s1p = np.zeros_like(t); s1m = np.zeros_like(t)
        for k in range(i + 1):
            Rk = np.maximum(t - (i - k) * dbar - tset[k], 0.0)
            chi = (t < tset[k] + (i - k) * dbar).astype(float)
            fp = fp + cq.Tf(lam, corners[k], Rk, +1) + gamma * Lam2h[k]
            fm = fm + cq.Tf(lam, corners[k], Rk, -1) + gamma * Lam2h[k]
            s1p = np.maximum(s1p, alpha_g[k] * sig_p[k] * chi)
            s1m = np.maximum(s1m, alpha_g[k] * sig_m[k] * chi)
        return fp + s1p, fm + s1m
    worst = 0.0
    for row in K['authority']:
        fp, fm = Fpm(row['unit'] - 1, [row['t0']])
        worst = max(worst, abs(fp[0] - row['F_plus']), abs(fm[0] - row['F_minus']))
    assert worst <= 1e-12, 'F^+- reimplementation differs from the certificate: %g' % worst
    return Fpm, worst


def pointwise_rows(K, cfg, T1, k):
    """Pointwise comparisons on the 1 ms record T1 (round-4 review,
    findings validation_logic-07 and -11): Proposition D1(b) at every
    record instant before T_xi, a*_i - F^-_i(t) <= u_i(t) <= a*_i + F^+_i(t);
    Proposition D1(d) in its layer form eps_k(t) <= P_k(t) (H6D); Lemma D3
    and Proposition D1(c): eps_k <= 0 on [0, t^set_k], and the S1
    saturation engaged, eps_k <= -phi on [t^act_k, t^set_k] (so that
    f_k = -alpha_k)."""
    n = int(cfg['n'])
    Fpm, reimpl = authority_envelopes(cfg, K)
    t = np.asarray(T1['t'], float)
    m = t < T1['T_xi'] - 1e-12
    tm = t[m]
    dev = np.asarray(T1['dev'])[:, m]
    eps = np.asarray(T1['eps'])
    up_obs, lo_obs, up_at, lo_at = [], [], [], []
    for i in range(n):
        fp, fm = Fpm(i, tm)
        gu = dev[i] - fp                     # <= 0 required
        gl = -dev[i] - fm                    # <= 0 required
        ju, jl = int(np.argmax(gu)), int(np.argmax(gl))
        up_obs.append(float(gu[ju])); up_at.append(float(tm[ju]))
        lo_obs.append(float(gl[jl])); lo_at.append(float(tm[jl]))
    worst, at = layer_envelope(K, cfg, T1)
    phi = float(cfg['phi'])
    s1s, s1b, s1sat = [], [], []
    for q in range(n):
        ts_, ta_ = float(k['tset'][q]), float(k['tact'][q])
        w0 = t <= ts_ + 1e-9
        s1s.append(float(np.max(eps[q, w0])))
        w1 = (t >= ta_ - 1e-9) & (t <= ts_ + 1e-9)
        if ts_ > ta_ + 1e-9 and w1.any():
            s1b.append(-phi); s1sat.append(float(np.max(eps[q, w1])))
        else:                                # instantaneous switch: no S1 stage
            s1b.append(None); s1sat.append(None)
    rows = {}
    rows['deviation_upper_pointwise'] = dict(
        bound=[0.0] * n, observed=up_obs, at=up_at, kind='upper', unit='m/s^2',
        ok=bool(max(up_obs) <= 0.0), ok_each=[bool(x <= 0.0) for x in up_obs],
        note='max over the 1 ms record before T_xi of (u_i - a*_i) - F^+_i(t)')
    rows['deviation_lower_pointwise'] = dict(
        bound=[0.0] * n, observed=lo_obs, at=lo_at, kind='upper', unit='m/s^2',
        ok=bool(max(lo_obs) <= 0.0), ok_each=[bool(x <= 0.0) for x in lo_obs],
        note='max over the 1 ms record before T_xi of (a*_i - u_i) - F^-_i(t)')
    rows['layer_speed_envelope'] = dict(
        bound=0.0, observed=worst, kind='upper', unit='m/s', at=at,
        ok=bool(worst <= 0.0),
        note='max over the 1 ms record before T_xi and the layers of '
             'eps_k(t) - P_k(t) (velocity-floor envelope of H6D)')
    rows['S1_sign'] = dict(
        bound=[0.0] * n, observed=s1s, kind='upper', unit='m/s',
        ok=bool(max(s1s) <= 0.0), ok_each=[bool(x <= 0.0) for x in s1s],
        note='max of eps_k on the 1 ms record over [0, t^set_k] (Lemma D3)')
    chk = [(b, o) for b, o in zip(s1b, s1sat) if b is not None]
    rows['S1_saturated'] = dict(
        bound=s1b, observed=s1sat, kind='upper', unit='m/s',
        ok=None if not chk else all(o <= b for b, o in chk),
        ok_each=[None if b is None else bool(o <= b) for b, o in zip(s1b, s1sat)],
        note='max of eps_k on the 1 ms record over [t^act_k, t^set_k] vs '
             '-phi: the S1 feedback saturated at -alpha_k (Prop. D1(c))')
    summary = dict(F_pm_reimplementation_max_diff=reimpl,
                   record_dt=float(t[1] - t[0]), last_sample_before_T_xi=float(tm[-1]),
                   upper_gap_min=-max(up_obs), upper_gap_unit=int(np.argmax(up_obs)) + 1,
                   upper_gap_at=up_at[int(np.argmax(up_obs))],
                   lower_gap_min=-max(lo_obs), lower_gap_unit=int(np.argmax(lo_obs)) + 1,
                   lower_gap_at=lo_at[int(np.argmax(lo_obs))],
                   layer_max=worst, layer_at=at)
    return rows, summary


def layer_envelope(K, cfg, T):
    """max over the record T before T_xi and the layers k of
    eps_k(t) - P_k(t) (P_k the velocity-floor envelope of H6D); <= 0 means
    every layer stayed below its envelope."""
    lam = float(cfg['lam'])
    t = np.asarray(T['t'], float)
    m = t < T['T_xi'] - 1e-9
    worst = -np.inf; at = None
    for q in range(len(K['tset'])):
        corners = cq.box_corners(0.0, K['E1'][q], float(cfg['eps_det']),
                                 bool(K['one_sided']))
        R = np.maximum(t[m] - K['tset'][q], 0.0)
        Pk = cq.EnvX(lam, corners, R) + K['Lam2'][q]
        d = np.asarray(T['eps'])[q, m] - Pk
        j = int(np.argmax(d))
        if d[j] > worst:
            worst = float(d[j]); at = (q + 1, float(t[m][j]))
    return worst, at


def speed_rows(T1, k):
    """Every unit's speed on the 1 ms record T1 (round 5): the largest
    increment v_i(t_{j+1}) - v_i(t_j) over the whole record (<= 0: the
    speed never increases; after the stop the record repeats the final
    state) and over the increments that end before the unit's stop
    (< 0: the speed decreases strictly while the unit moves)."""
    v = np.asarray(T1['v'], float)
    t = np.asarray(T1['t'], float)
    n = v.shape[0]
    dv = np.diff(v, axis=1)
    inc, inc_at, mov, mov_at = [], [], [], []
    for i in range(n):
        j = int(np.argmax(dv[i]))
        inc.append(float(dv[i, j])); inc_at.append(float(t[j + 1]))
        m = t[1:] < float(k['tstop'][i]) - 1e-12
        jm = int(np.argmax(np.where(m, dv[i], -np.inf)))
        mov.append(float(dv[i, jm])); mov_at.append(float(t[jm + 1]))
    rows = dict(
        speed_nonincreasing=dict(
            bound=[0.0] * n, observed=inc, at=inc_at, kind='upper', unit='m/s',
            ok=bool(max(inc) <= 0.0), ok_each=[bool(x <= 0.0) for x in inc],
            note='largest increment v_i(t_{j+1}) - v_i(t_j) on the 1 ms record'),
        speed_decreasing_while_moving=dict(
            bound=[0.0] * n, observed=mov, at=mov_at, kind='upper (strict)',
            unit='m/s', ok=bool(max(mov) < 0.0), ok_each=[bool(x < 0.0) for x in mov],
            note='largest increment on the 1 ms record before the unit\'s stop'))
    mono = dict(record_dt=float(t[1] - t[0]), n_samples=int(t.size),
                largest_increment=max(inc), largest_increment_unit=inc,
                largest_increment_while_moving=max(mov),
                largest_increment_while_moving_unit=mov,
                nonincreasing_all=bool(max(inc) <= 0.0),
                strictly_decreasing_while_moving=bool(max(mov) < 0.0))
    return rows, mono


def _blocks(idx):
    """Maximal runs of consecutive record indices, as (first, last) pairs."""
    out = []
    for j in idx:
        if out and j == out[-1][1] + 1:
            out[-1][1] = int(j)
        else:
            out.append([int(j), int(j)])
    return out


def speed_order(T1, Pl, d_s):
    """Speed order and clearances of every pair on the 1 ms record T1
    (round-5 review).  For pair i >= 2 on the follower's main piece
    (t_0, t^c_i): the record instants at which the follower was not faster
    than its predecessor (v_i <= v_{i-1}), grouped into blocks; the block
    that starts at the first record instant after the onset and the block
    that ends at the last record instant before the join; the windows that
    contain them, measured conservatively from the last record instant at
    which the follower was faster (so that they also cover the crossing
    between two record instants); the largest deficit v_{i-1} - v_i.  On
    the approach [0, t_0]: the record instants with v_i < v_{i-1}.  Per
    pair also the first record instant below the parking gap, the smallest
    clearance and its instant, and the clearance at the join and at the
    end of the record."""
    t = np.asarray(T1['t'], float)
    v = np.asarray(T1['v'], float)
    c = np.asarray(T1['c'], float)
    dt = float(t[1] - t[0])
    td, tc = plan_events(Pl)
    tol = 1e-9
    pairs = {}
    onset_w, join_w, deficit, other = [], [], [], []
    for i in range(1, v.shape[0]):
        t0 = td[i]
        rel = v[i] - v[i - 1]                      # > 0: the follower is faster
        main = np.where((t > t0 + tol) & (t < tc[i] - tol))[0]
        slow = main[rel[main] <= 0.0]
        blk = _blocks(slow)
        first_main, last_main = int(main[0]), int(main[-1])
        ob = [b for b in blk if b[0] == first_main]
        jb = [b for b in blk if b[1] == last_main]
        rest = [b for b in blk if b not in ob and b not in jb]
        # conservative windows: from the onset to the first faster record
        # instant after the onset block; from the last faster record
        # instant before the join block to the join
        w_on = float(t[ob[0][1] + 1] - t0) if ob else 0.0
        w_jn = float(tc[i] - t[jb[0][0] - 1]) if jb else 0.0
        dfc = float((-rel[slow]).max()) if slow.size else 0.0
        app = np.where(t <= t0 + tol)[0]
        aslow = app[rel[app] < 0.0]
        below = np.where(c[i - 1] < d_s[i])[0]
        jc = int(np.argmin(np.abs(t - tc[i])))
        km = int(np.argmin(c[i - 1]))
        pairs[str(i + 1)] = dict(
            main_piece=[t0, tc[i]], n_main=int(main.size), n_not_faster=int(slow.size),
            blocks=[[float(t[a]), float(t[b]), b - a + 1] for a, b in blk],
            onset_block=[float(t[ob[0][0]]), float(t[ob[0][1]])] if ob else None,
            join_block=[float(t[jb[0][0]]), float(t[jb[0][1]])] if jb else None,
            other_blocks=[[float(t[a]), float(t[b])] for a, b in rest],
            onset_window=w_on, join_window=w_jn, max_deficit=dfc,
            max_deficit_at=float(t[slow[np.argmax(-rel[slow])]]) if slow.size else None,
            approach_slower=[float(t[aslow[0]]), float(t[aslow[-1]])] if aslow.size else None,
            approach_max_deficit=float((-rel[aslow]).max()) if aslow.size else 0.0,
            min_rel_outside_blocks=float(np.delete(rel[main], np.searchsorted(main, slow)).min()),
            parking_gap=float(d_s[i]),
            first_below_parking=float(t[below[0]]) if below.size else None,
            c_min=float(c[i - 1][km]), c_min_at=float(t[km]),
            c_at_join=float(c[i - 1][jc]), c_end=float(c[i - 1][-1]))
        onset_w.append(w_on); join_w.append(w_jn); deficit.append(dfc)
        other += rest
    return dict(
        record_dt=dt, n_samples=int(t.size),
        note=('follower not faster than its predecessor (v_i <= v_{i-1}) on '
              'the main piece (t_0, t^c_i) of the 1 ms record; windows '
              'measured from the last faster record instant'),
        pairs=pairs, onset_window_max=max(onset_w), join_window_max=max(join_w),
        max_deficit=max(deficit), only_onset_and_join_blocks=not other,
        undershoot_max=max(p['parking_gap'] - p['c_min'] for p in pairs.values()))


def cert_summary(ev):
    K = ev['consts']
    return dict(admitted=ev['admitted'], failed=ev['failed'], dbar=ev['dbar'],
                conds={nm: dict(ok=c['ok'], slack=c['slack'])
                       for nm, c in ev['conds'].items()},
                settling_bound=K['settling_bound'], q=K['q'],
                T_f_bar=K['T_f_bar'], sched=K['sched'], align=K['align'],
                gaperr=K['gaperr'], T_ent_bar=K['T_ent_bar'],
                T_hold=K['T_hold'], tset=K['tset'], tact=K['tact'],
                E1=K['E1'], W=K['W'], epsb=K['epsb'], eb=K['eb'],
                Fb_units=K['Fb_units'], vfloor_min=K['vfloor_min'],
                v_c1=K['v_c1'], tau_star=K['tau_star'])


def diff(a, b, keys=KPI_KEYS):
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
    fin = {k: v for k, v in d.items() if np.isfinite(v)}
    bad = sorted(k for k, v in d.items() if not np.isfinite(v))
    if not fin:
        return float('nan'), None, bad
    kx = max(fin, key=fin.get)
    return fin[kx], kx, bad


def compact(k, cfg, bt=None):
    """Row of a schedule, sweep or ensemble table."""
    r = dict(settling=k['settling'], T_f=k['T_f'], sched=k['sched'],
             marker_abs_max=k['marker_abs_max'], gap_abs_max=k['gap_abs_max'],
             u_min=k['u_min'], u_max=k['u_max'], u_max_pre=k['u_max_pre'],
             u_abs_after_latch=k['u_abs_after_latch'],
             n_mod=k['n_mod'], n_fallback=k['n_fallback'], n_sat=k['n_sat'],
             filter_modifications=k['filter_modifications'],
             mmin_lim_min=k['mmin_lim_min'], gmin_min=k['gmin_min'],
             hmin_min=k['hmin_min'], entry_obs=k['entry_obs'],
             e_hnd_max=max(abs(x) for x in k['e_hnd']),
             eps_hnd_max=max(abs(x) for x in k['eps_hnd']),
             tset=k['tset'], stop_order=k['stop_order'],
             tail_first_head_last=k['tail_first_head_last'],
             rule_entered=k['rule_entered'], all_stopped=k['all_stopped'],
             contact_time=k['contact_time'], age=k['age'],
             requirements=requirements(cfg, k))
    if bt is not None:
        r['bounds_ok'] = bt['all_ok']
        r['bounds_failed'] = bt['failed']
        r['n_checked'] = bt['n_checked']
        r['n_rows'] = bt['n_rows']
    return r


def aggregate(rows, K=None):
    """Extremes over named compact rows with the run that attains them."""
    it = list(rows.items())
    agg = dict(n=len(it))
    for key, fn in (('settling', max), ('T_f', max), ('sched', max),
                    ('marker_abs_max', max), ('gap_abs_max', max),
                    ('u_min', min), ('u_max', max), ('u_max_pre', max),
                    ('mmin_lim_min', min), ('gmin_min', min),
                    ('entry_obs', max), ('e_hnd_max', max),
                    ('eps_hnd_max', max)):
        v, nm = arg_ext(it, key, fn)
        agg[key] = dict(value=v, run=nm)
    v, nm = arg_ext(it, 'settling', min)
    agg['settling_min'] = dict(value=v, run=nm)
    agg['filter_modifications_total'] = sum(r['filter_modifications'] for _, r in it)
    agg['u_abs_after_latch_max'] = max(r['u_abs_after_latch'] for _, r in it)
    agg['all_stopped'] = all(r['all_stopped'] for _, r in it)
    agg['no_contact'] = all(r['contact_time'] is None for _, r in it)
    agg['tail_first_head_last_all'] = all(r['tail_first_head_last'] for _, r in it)
    agg['rule_entered_any'] = any(r['rule_entered'] for _, r in it)
    agg['age_sup_max'] = max(r['age']['max_left_limit'] for _, r in it)
    agg['requirements_all_ok'] = all(r['requirements']['all_ok'] for _, r in it)
    if all('bounds_ok' in r for _, r in it):
        agg['bounds_all_ok'] = all(r['bounds_ok'] for _, r in it)
        agg['n_bounds_failed'] = sum(1 for _, r in it if not r['bounds_ok'])
        # every bounded row compared in every run (no missing observation)
        agg['all_rows_checked'] = all(r['n_checked'] == r['n_rows'] for _, r in it)
        agg['n_checked_min'] = min(r['n_checked'] for _, r in it)
    if K is not None:
        agg['certified'] = dict(settling_bound=K['settling_bound'],
                                T_f_bar=K['T_f_bar'], sched=K['sched'],
                                align=K['align'], gaperr_max=max(K['gaperr']),
                                T_ent_bar=K['T_ent_bar'])
    return agg


# ------------------------------------------------------------- parts
def part_ref():
    des = load_design()
    cfg = dict(des['cfg'])
    ev = certificate(cfg)
    K = ev['consts']
    cons = dict(admitted=ev['admitted'],
                same_as_design=all(
                    np.allclose(np.asarray(K[key], float),
                                np.asarray(des['certificate']['consts'][key], float),
                                rtol=0, atol=0)
                    for key in ('settling_bound', 'T_f_bar', 'align', 'gaperr',
                                'T_ent_bar', 'tset', 'Fb_units', 'q')))
    B = build(cfg)
    print('[ref] h=%g transport=%g s T_xi=%.3f s tau*=%.6f s certificate '
          'admitted=%s (same as design.json: %s)'
          % (H_FINE, B['transport'], B['P']['T_xi'], K['tau_star'],
             ev['admitted'], cons['same_as_design']), flush=True)
    R = simulate(B, h=H_FINE)
    k = kpis(R, B)
    print('  ' + summary(k), '(%.1f s)' % R['runtime_s'], flush=True)
    T = trajectory(B, R)
    # the same run recorded every 1 ms for the pointwise comparisons (the
    # record interval does not enter the integration: asserted)
    R1 = simulate(B, h=H_FINE, rec_dt=1e-3)
    k1 = kpis(R1, B)
    for key in ('tstop', 'settling', 'marker_signed', 'gap_signed', 'tset', 'tact',
                'e_sw', 'eps_sw', 'e_hnd', 'eps_hnd', 'u_min', 'u_max', 'mmin_lim_min',
                'gmin_min', 'vmin_pre_min', 'entry_obs'):
        assert np.array_equal(np.asarray(k1[key], float), np.asarray(k[key], float)), key
    T1 = trajectory(B, R1)
    extra, pw = pointwise_rows(K, cfg, T1, k)
    print('  pointwise (1 ms record): %s' % json.dumps(_py(pw)), flush=True)
    srows, mono = speed_rows(T1, k)
    extra.update(srows)
    print('  speeds (1 ms record): %s' % json.dumps(_py(mono)), flush=True)
    order = speed_order(T1, B['Pl'], B['d_s'])
    print('  speed order (1 ms record): onset window %.4f s, join window %.4f s, '
          'deficit %.3e m/s, only onset and join blocks %s, undershoot %.4f m'
          % (order['onset_window_max'], order['join_window_max'], order['max_deficit'],
             order['only_onset_and_join_blocks'], order['undershoot_max']), flush=True)
    bt = bounds_rows(K, cfg, k, R['plan']['pieces'], extra=extra)
    for name, r in bt['rows'].items():
        if r.get('kind') == 'bool':
            print('  %-34s ok=%s' % (name, r['ok']), flush=True)
            continue
        fmt = (lambda x: '[%s]' % ', '.join('%.4g' % z if z is not None else '-'
                                            for z in x)
               if isinstance(x, list) else '%.6g' % x if x is not None else '-')
        print('  %-34s bound %-40s observed %-40s ok=%s'
              % (name, fmt(r['bound']), fmt(r['observed']), r['ok']), flush=True)
    print('  all bounds ok:', bt['all_ok'], bt['failed'], flush=True)
    req = requirements(cfg, k)
    Pl = B['Pl']
    td, tc = plan_events(Pl)
    T_xi = float(Pl.T_xi)
    # phase extrema on the 10 ms record: before the head onset (fan: the
    # approach), the main deceleration (t^d_1 to t^c_n; canonical: the
    # staggered pulses), the tail crawl t^c_n to T_xi, braking
    t = T['t']
    ph = {}
    main = 'main_deceleration' if Pl.family == 'fan' else 'pulses'
    for name, (a, bnd) in {'pre_onset': (0.0, td[0]),
                           main: (td[0], tc[-1]),
                           'tail_crawl': (tc[-1], T_xi),
                           'braking': (T_xi, float(np.nanmax(k['tstop'])))}.items():
        m = (t >= a - 1e-9) & (t <= bnd + 1e-9)
        if not m.any():
            ph[name] = dict(t=[a, bnd], samples=0)
            continue
        ph[name] = dict(t=[a, bnd], samples=int(m.sum()),
                        v_min=float(np.min(T['v'][:, m])),
                        v_max=float(np.max(T['v'][:, m])),
                        u_min=float(np.min(T['u'][:, m])),
                        u_max=float(np.max(T['u'][:, m])),
                        dev_min=float(np.min(T['dev'][:, m])),
                        dev_max=float(np.max(T['dev'][:, m])),
                        e_abs_max=float(np.max(np.abs(T['e'][:, m]))),
                        eps_abs_max=float(np.max(np.abs(T['eps'][:, m]))),
                        c_min=float(np.min(T['c'][:, m])),
                        c_max=float(np.max(T['c'][:, m])))
    headline = dict(
        settling_time=dict(observed=k['settling'], bound=K['settling_bound'],
                           requirement=float(cfg['sync_tol'])),
        marker_error=dict(observed=k['marker_abs_max'], bound=K['align'],
                          requirement=float(cfg['align_tol'])),
        terminal_gap_error=dict(observed=k['gap_abs_max'], bound=max(K['gaperr']),
                                requirement=float(cfg['gap_tol'])),
        completion=dict(observed=k['T_f'], bound=K['T_f_bar'],
                        planned_stop=K['tau_star']),
        command_range=dict(observed=[k['u_min'], k['u_max']],
                           before_T_xi=[k['u_min_pre'], k['u_max_pre']],
                           input_set=[-float(cfg['k']), 0.0],
                           after_latch_abs_max=k['u_abs_after_latch']),
        entry=dict(observed=k['entry_obs'], bound=K['T_ent_bar'],
                   T_xi=T_xi),
        stop_order=k['stop_order'], rule_entered=k['rule_entered'],
        filter_modifications=k['filter_modifications'],
        speed_nonincreasing=mono['nonincreasing_all'],
        speed_largest_increment=mono['largest_increment'])
    rec = dict(
        cfg=cfg, P=B['P'], d_s=B['d_s'], v0=B['v0'], e0=B['e0'],
        family=Pl.family, plan=R['plan'], plan_exact=des.get('plan_exact'),
        certificate=cert_summary(ev), certificate_consistency=cons,
        transport=R['config']['transport'], age_bound=B['age_bound'],
        kpis=k, bounds=bt, requirements=req, headline=headline,
        pointwise=pw, speed_monotonicity=mono, speed_order=order,
        phase_times=dict(t_d=td, t_c=tc, T_xi=T_xi, tau_star=K['tau_star'],
                         tstop=k['tstop'], T_f=k['T_f'], T_f_bar=K['T_f_bar'],
                         tset_obs=k['tset'], tact_obs=k['tact'],
                         trule_obs=k['trule'], tset_cert=K['tset'],
                         tact_cert=K['tact'], entry_obs=k['entry_obs'],
                         T_ent_bar=K['T_ent_bar'], t_rmp=k['t_rmp'],
                         fb_window=k['fb_window']),
        phase_extrema=ph,
        filter_modifications=dict(min_branch=R['n_mod'], fallback=R['n_fallback'],
                                  saturation=R['n_sat'], evaluations=R['n_eval'],
                                  left_limit_evaluations=R['n_left_limits']),
        age_audit=R['age'], runtime_s=R['runtime_s'], config=R['config'],
        system=('Case D closed loop: continuous local law (T_u = h), RK4 '
                'between events with the steps split at the planned jumps, '
                'T_m = T_c = 1 ms, constant transport 19 ms, exact standstill '
                'sensing, head BRAKE and every plan at -a_b from T_xi, '
                'zero-speed events localized; errors relative to the plan'),
        terminology=('settling time = first-to-last stop interval (the main '
                     'text calls it the stopping-time dispersion); tset are '
                     'the S1 detection instants, compared with the detection '
                     'deadlines'))
    dump('reference.json', rec)
    save_traj(os.path.join(OUT, 'traj_caseD.npz'), B, R, T)
    print('  headline: settling %.4f ms (bound %.4f s), marker %.4f m (%.4f), '
          'gap %.4f m (%.4f), T_f %.4f s (%.4f), u in [%.5f, %.5f], entry %.3f s '
          '(%.3f)' % (1e3 * k['settling'], K['settling_bound'], k['marker_abs_max'],
                       K['align'], k['gap_abs_max'], max(K['gaperr']), k['T_f'],
                       K['T_f_bar'], k['u_min'], k['u_max'], k['entry_obs'],
                       K['T_ent_bar']), flush=True)
    return rec


def trajectory(B, R):
    rec = R['rec']
    t = np.asarray(rec['t'], float)
    ok = np.isfinite(t)
    P = B['P']
    return dict(t=t[ok], v=rec['v'][:, ok], e=rec['e'][:, ok],
                eps=rec['eps'][:, ok], u=rec['u'][:, ok], m=rec['m'][:, ok],
                g=rec['g'][:, ok], c=rec['g'][:, ok] + P['s_m'],
                hbar=rec['h'][:, ok], ps=rec['ps'][:, ok], pv=rec['pv'][:, ok],
                pa=rec['pa'][:, ok], dev=rec['dev'][:, ok], T_xi=P['T_xi'])


def save_traj(path, B, R, T):
    np.savez(path, keys_note=np.array(
        't [s] every 10 ms; v unit speeds; e pair POSITION errors relative to '
        'the plan (row 0: head, s*_1 - s_1); eps pair VELOCITY errors relative '
        'to the plan; u applied commands (right limits); m inactivity margins '
        'u_cbf - u_nom (pairs, NaN once latched); g clearance margins c_i - '
        's_m; c clearances (pairs); hbar barriers; ps, pv, pa planned '
        'position, speed, acceleration of every unit; dev = u - pa; tstop '
        'stop instants; tset S1 detection instants; tact activations; marks '
        'markers; s final positions; egap signed terminal gap errors (entry 0: '
        'head slot).  After the last stop the record repeats the final '
        'state.'),
        config=np.array(json.dumps(_py(R['config']))),
        cfg=np.array(json.dumps(_py(B['cfg']))),
        plan=np.array(json.dumps(_py(R['plan']))),
        t=T['t'], v=T['v'], e=T['e'], eps=T['eps'], u=T['u'], m=T['m'],
        g=T['g'], c=T['c'], hbar=T['hbar'], ps=T['ps'], pv=T['pv'],
        pa=T['pa'], dev=T['dev'], tstop=R['tstop'], tset=R['tset'],
        tact=R['tact'], trule=R['trule'], marks=R['marks'], s=R['s'],
        egap=R['egap'], marker_signed=R['marker_signed'], e_hnd=R['e_hnd'],
        eps_hnd=R['eps_hnd'], d_s=np.array(B['d_s']), v0=np.array(B['v0']),
        e0=np.array(B['e0']), T_xi=B['P']['T_xi'], tau_star=R['tau_r'])
    print('  -> data/new_d19/caseD/%s' % os.path.basename(path), flush=True)


def part_conv():
    cfg = design_cfg()
    ev = certificate(cfg)
    K = ev['consts']
    B = build(cfg)
    print('[conv] h in %s (+ diagnostic %s)' % (H_CONV, H_DIAG), flush=True)
    rows = []
    for h in H_DIAG + H_CONV:
        R = simulate(B, h=h)
        k = kpis(R, B)
        bt = bounds_rows(K, cfg, k, R['plan']['pieces'])
        rows.append(dict(h=h, diagnostic=h in H_DIAG, kpis=k,
                         bounds_ok=bt['all_ok'], bounds_failed=bt['failed'],
                         n_checked=bt['n_checked'], n_rows=bt['n_rows']))
        print('  h=%-8g %s bounds_ok=%s (%.1f s)' % (h, summary(k), bt['all_ok'],
                                                    R['runtime_s']), flush=True)
    ref = rows[-1]['kpis']
    for r in rows:
        d = diff(r['kpis'], ref)
        mx, kx, bad = dmax(d)
        r['diff_to_finest'] = d
        r['max_diff_to_finest'] = mx
        r['argmax_kpi'] = kx
        r['nan_mismatch'] = bad
        print('  h=%-8g max |diff| to h=%g: %.3e (%s)' % (r['h'], H_CONV[-1], mx, kx),
              flush=True)
    conv = [r for r in rows if not r['diagnostic'] and r['h'] != H_CONV[-1]]
    worst = {key: max(r['diff_to_finest'][key] for r in conv) for key in KPI_KEYS}
    mx, kx, bad = dmax(worst)
    tsd = max(r['diff_to_finest']['tstop'] for r in conv)
    out = dict(study=('numerical convergence: the same event schedule (sends '
                      'every T_m = 1 ms, constant transport 19 ms, detector '
                      'T_c = 1 ms anchored at activation) and the same '
                      'continuous law; only the RK4 step h varies; differences '
                      'to h = %g s' % H_CONV[-1]),
               h_values=list(H_CONV), diagnostic_h=list(H_DIAG), rows=rows,
               max_diff_over_h_by_kpi=worst, max_diff_over_h_and_kpis=mx,
               argmax_kpi=kx, nan_mismatch=bad, max_stop_diff=tsd,
               all_rows_inside_bounds=all(r['bounds_ok'] for r in rows),
               all_filter_inactive=all(r['kpis']['filter_inactive'] for r in rows))
    dump('convergence.json', out)


def part_digital():
    cfg = design_cfg()
    ev = certificate(cfg)
    K = ev['consts']
    B = build(cfg)
    print('[digital] zoh T_u in %s at h=%g, pure 1 kHz at h=1 ms' % (TU_DIG, H_FINE),
          flush=True)
    Rc = simulate(B, h=H_FINE)
    ref = kpis(Rc, B)
    bt = bounds_rows(K, cfg, ref, Rc['plan']['pieces'])
    rows = [dict(T_u=H_FINE, h=H_FINE, scheme='cont', kpis=ref,
                 bounds_ok=bt['all_ok'], bounds_failed=bt['failed'],
                 n_checked=bt['n_checked'], n_rows=bt['n_rows'])]
    for tu, h in [(x, H_FINE) for x in TU_DIG] + [(1e-3, 1e-3)]:
        R = simulate(B, h=h, scheme='zoh', T_u=tu)
        k = kpis(R, B)
        bt = bounds_rows(K, cfg, k, R['plan']['pieces'])
        d = diff(k, ref)
        mx, kx, bad = dmax(d)
        rows.append(dict(T_u=tu, h=h, scheme='zoh', kpis=k, bounds_ok=bt['all_ok'],
                         bounds_failed=bt['failed'], n_checked=bt['n_checked'],
                         n_rows=bt['n_rows'], diff_to_continuous=d,
                         max_diff_to_continuous=mx, argmax_kpi=kx,
                         nan_mismatch=bad,
                         max_stop_diff=d['tstop'],
                         settling_diff=d['settling'],
                         max_marker_diff=d['marker_signed'],
                         jumps_on_update_grid=bool(R['plan']['jumps_on_Tm_grid'])))
        print('  T_u=%-7g h=%-7g %s max|diff|=%.3e (%s) bounds_ok=%s'
              % (tu, h, summary(k), mx, kx, bt['all_ok']), flush=True)
    out = dict(study=('digital implementation: the local law is sampled every '
                      'T_u and held (zero-order hold); each unit also updates '
                      'at its own events (packet batch, detector sample, own '
                      'planned jump, zero-speed instants); every planned jump '
                      'of the selected plan lies on the 1 ms grid, so the '
                      'pure 1 kHz run (T_u = h = 1 ms) processes every jump at '
                      'its exact instant (premise (I2)); plant, message '
                      'schedule (T_m = 1 ms, transport 19 ms), detector and '
                      'gains fixed; compared with the continuous reference '
                      '(h = %g s)' % H_FINE),
               rows=rows,
               all_bounds_ok=all(r['bounds_ok'] for r in rows),
               all_filter_inactive=all(r['kpis']['filter_inactive'] for r in rows),
               max_stop_diff_1kHz=rows[-1]['max_stop_diff'],
               max_marker_diff_1kHz=rows[-1]['max_marker_diff'])
    dump('digital.json', out)


def _run_rows(B, K, cfg, specs, h=H_SCHED, label=''):
    """Run the named FIFO schedules; compact rows and the held-age audit."""
    rows = {}
    t0 = time.time()
    for j, (name, D) in enumerate(specs):
        R = simulate(B, h=h, fifo=D)
        k = kpis(R, B)
        bt = bounds_rows(K, cfg, k, R['plan']['pieces'])
        r = compact(k, cfg, bt)
        r['age_ok'] = bool(R['age']['max_left_limit'] <= B['age_bound'] + 1e-12)
        r['schedule'] = dict(d_min=int(D[1:].min()), d_max=int(D[1:].max()),
                             d_mean=float(D[1:].mean()))
        rows[name] = r
        if (j + 1) % 25 == 0 or j + 1 == len(specs):
            print('  %s %3d/%d last %s: %s bounds_ok=%s age=%.4f (%.0f s)'
                  % (label, j + 1, len(specs), name, summary(k), bt['all_ok'],
                     R['age']['max_left_limit'], time.time() - t0), flush=True)
    return rows


def part_sched(n_sched=300, seed0=500):
    cfg = design_cfg()
    ev = certificate(cfg)
    K = ev['consts']
    B = build(cfg)
    N = int(round(B['P']['T_END'] / T_M)) + 2
    print('[sched] %d random FIFO schedules, d_j in 1..%d steps, h=%g'
          % (n_sched, B['dmax_steps'], H_SCHED), flush=True)
    specs = []
    for j in range(n_sched):
        rng = np.random.default_rng(seed0 + j)
        specs.append(('seed%d' % (seed0 + j),
                      random_schedule(rng, N, B['dmax_steps'])))
    rows = _run_rows(B, K, cfg, specs, label='sched')
    agg = aggregate(rows, K)
    agg['all_age_ok'] = all(r['age_ok'] for r in rows.values())
    agg['age_sup_min'] = min(r['age']['max_left_limit'] for r in rows.values())
    print('  aggregate: settling <= %.5f s (%s), marker <= %.4f m, gap <= %.4f m, '
          'u in [%.5f, %.5f], mods %d, entry <= %.3f s, bounds_ok %s, age_ok %s'
          % (agg['settling']['value'], agg['settling']['run'],
             agg['marker_abs_max']['value'], agg['gap_abs_max']['value'],
             agg['u_min']['value'], agg['u_max']['value'],
             agg['filter_modifications_total'], agg['entry_obs']['value'],
             agg.get('bounds_all_ok'), agg['all_age_ok']), flush=True)
    out = dict(study=('random FIFO schedules: piecewise-constant per-hop '
                      'transport (segments 0.5 to 5 s, d_j uniform in 1..%d '
                      'steps of T_m = 1 ms, one generator per schedule, seed '
                      '%d + j), FIFO projection at the receiver, held age <= '
                      '20 ms audited; Case D continuous law at h = %g s (RK4)'
                      % (B['dmax_steps'], seed0, H_SCHED)),
               certificate=cert_summary(ev), aggregate=agg, rows=rows)
    dump('schedules.json', out)


def _windows(events, W, shape):
    if shape == 'around':
        return [(te - W, te + W) for te in events]
    if shape == 'after':
        return [(te, te + W) for te in events]
    return [(te - W, te) for te in events]


def _sched_windows(Nsend, win_by_hop, n=5, dmax=19, dmin=1):
    d = np.full((n, Nsend), dmin, np.int64)
    for i in range(1, n):
        for (a, b) in win_by_hop[i]:
            ja = max(0, int(np.floor(a / T_M - 1e-9)))
            jb = min(Nsend - 1, int(np.ceil(b / T_M + 1e-9)))
            if jb >= ja:
                d[i, ja:jb + 1] = dmax
    d[0] = 1
    return d


def adv_specs(B, Rnom, dmax=19):
    """Adversarial schedule families (transport in steps of T_m)."""
    Pl = B['Pl']
    n = int(B['P']['n'])
    T_xi = float(Pl.T_xi)
    Nsend = int(round(B['P']['T_END'] / T_M)) + 2
    td, tc = plan_events(Pl)
    tact = [float(x) for x in Rnom['tact']]
    tset = [float(x) for x in Rnom['tset']]
    # every planned jump instant before T_xi once (the fan's common onset
    # t_0 is one instant for all units)
    ev_sets = dict(jumps=sorted(set(td + tc)), switch=sorted(tact[1:] + tset),
                   txi=[T_xi])
    ev_sets['all'] = sorted(ev_sets['jumps'] + ev_sets['switch'] + [T_xi])
    specs = []
    meta = {}
    for es, evs in ev_sets.items():
        for shape in ('around', 'after', 'before'):
            for W in (0.02, 0.1, 0.5):
                nm = 'max_%s_%s_W%gms' % (shape, es, W * 1e3)
                win = {i: _windows(evs, W, shape) for i in range(1, n)}
                specs.append((nm, _sched_windows(Nsend, win, n, dmax)))
                meta[nm] = dict(family='windows', events=es, shape=shape, W=W,
                                n_events=len(evs))
    # per hop: around the events of the sender and the receiver only
    win = {i: _windows(sorted(set([td[i - 1], tc[i - 1], td[i], tc[i]])), 0.1,
                       'around')
           for i in range(1, n)}
    specs.append(('max_around_hop_jumps_W100ms', _sched_windows(Nsend, win, n, dmax)))
    meta['max_around_hop_jumps_W100ms'] = dict(family='per-hop windows',
                                               events='jumps of units i-1 and i')
    win = {i: _windows([tset[i - 1], tact[i], tset[i]], 0.1, 'around')
           for i in range(1, n)}
    specs.append(('max_around_hop_switch_W100ms', _sched_windows(Nsend, win, n, dmax)))
    meta['max_around_hop_switch_W100ms'] = dict(
        family='per-hop windows', events='t^set_{i-1}, t^act_i, t^set_i')
    # bursts: alternating blocks of dmax and 1 of random length 5..200 sends
    for seed in range(1, 11):
        rng = np.random.default_rng(900 + seed)
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            j = 0; val = dmax if rng.random() < 0.5 else 1
            while j < Nsend:
                L = int(rng.integers(5, 201))
                d[i, j:j + L] = val
                val = 1 if val == dmax else dmax
                j += L
        nm = 'burst_s%d' % seed
        specs.append((nm, d))
        meta[nm] = dict(family='bursts', seed=900 + seed)
    # sawtooth: FIFO batches of dmax packets, held age through 1..dmax+1
    for ph in (0, 4, 8, 12):
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            p = (ph + 3 * i) % dmax
            d[i] = dmax - ((np.arange(Nsend) + p) % dmax)
        nm = 'sawtooth_ph%d' % ph
        specs.append((nm, d))
        meta[nm] = dict(family='sawtooth batches', phase=ph)
    # hop patterns
    for par in (0, 1):
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            d[i] = dmax if i % 2 == par else 1
        nm = 'hops_%s_max' % ('even' if par == 0 else 'odd')
        specs.append((nm, d))
        meta[nm] = dict(family='per-hop constant')
    for nm, val in (('const_max', dmax), ('const_min', 1)):
        d = np.full((n, Nsend), val, np.int64)
        d[0] = 1
        specs.append((nm, d))
        meta[nm] = dict(family='constant', d=val)
    return specs, meta, ev_sets


def part_adv():
    cfg = design_cfg()
    ev = certificate(cfg)
    K = ev['consts']
    B = build(cfg)
    Rnom = simulate(B, h=H_SCHED)
    specs, meta, ev_sets = adv_specs(B, Rnom, B['dmax_steps'])
    print('[adv] %d adversarial schedules at h=%g' % (len(specs), H_SCHED), flush=True)
    rows = _run_rows(B, K, cfg, specs, label='adv')
    for nm in rows:
        rows[nm]['family'] = meta[nm]
    agg = aggregate(rows, K)
    agg['all_age_ok'] = all(r['age_ok'] for r in rows.values())
    fam = {}
    for nm, r in rows.items():
        f = meta[nm]['family']
        g = fam.setdefault(f, dict(n=0, settling_max=0.0, marker_max=0.0,
                                   gap_max=0.0, u_max_pre=-np.inf,
                                   bounds_ok=True))
        g['n'] += 1
        g['settling_max'] = max(g['settling_max'], r['settling'])
        g['marker_max'] = max(g['marker_max'], r['marker_abs_max'])
        g['gap_max'] = max(g['gap_max'], r['gap_abs_max'])
        g['u_max_pre'] = max(g['u_max_pre'], r['u_max_pre'])
        g['bounds_ok'] = g['bounds_ok'] and r['bounds_ok']
    print('  aggregate: settling <= %.5f s (%s), marker <= %.4f m, gap <= %.4f m, '
          'u in [%.5f, %.5f], mods %d, bounds_ok %s, age_ok %s'
          % (agg['settling']['value'], agg['settling']['run'],
             agg['marker_abs_max']['value'], agg['gap_abs_max']['value'],
             agg['u_min']['value'], agg['u_max']['value'],
             agg['filter_modifications_total'], agg.get('bounds_all_ok'),
             agg['all_age_ok']), flush=True)
    out = dict(study=('adversarial FIFO schedules (transport 1..%d steps of '
                      'T_m = 1 ms, held age <= 20 ms): maximal transport in '
                      'windows around, after or before every planned jump '
                      'instant (onsets and crawl joins of all units), every '
                      'activation and S1 detection instant of the nominal run, '
                      'T_xi, or all of them (W = 20, 100, 500 ms), minimal '
                      'elsewhere; per-hop windows; bursts; sawtooth FIFO '
                      'batches; per-hop constant patterns; constant maximal '
                      'and minimal transport; Case D continuous law at h = %g '
                      's' % (B['dmax_steps'], H_SCHED)),
               event_sets=ev_sets, certificate=cert_summary(ev), aggregate=agg,
               families=fam, rows=rows)
    dump('adversarial.json', out)


def part_sweep(ages_ms=AGES_MS, n_sched=10, seed0=700):
    des = load_design()
    cfg = dict(des['cfg'])
    kb = int(des['boundary_ms']['k_ms'])
    # the certified age boundary of the design and the next age
    ages_ms = tuple(sorted(set(ages_ms) | {kb, kb + 1}))
    ev20 = certificate(cfg)
    K20 = ev20['consts']
    print('[sweep] age bounds %s ms' % list(ages_ms), flush=True)
    rows = []
    t0 = time.time()
    for A in ages_ms:
        db = A * 1e-3
        try:
            ev = certificate(cfg, dbar=db)
        except Exception as ex:                      # pragma: no cover
            ev = dict(admitted=False, failed=['error: %s' % ex], closed=False,
                      conds={})
        cfgA = dict(cfg, dbar=db)
        B = build(cfgA)
        R = simulate(B, h=H_FINE)
        k = kpis(R, B)
        bt_age = bounds_rows(ev['consts'], cfg, k, R['plan']['pieces']) \
            if ev['closed'] else None
        bt_des = bounds_rows(K20, cfg, k, R['plan']['pieces'])
        const = compact(k, cfg)
        const['bounds_at_age_ok'] = None if bt_age is None else bt_age['all_ok']
        const['bounds_at_age_failed'] = None if bt_age is None else bt_age['failed']
        const['n_checked_at_age'] = None if bt_age is None else bt_age['n_checked']
        const['n_rows_at_age'] = None if bt_age is None else bt_age['n_rows']
        const['design_bounds_ok'] = bt_des['all_ok']
        const['design_bounds_failed'] = bt_des['failed']
        const['age_ok'] = bool(R['age']['max_left_limit'] <= db + 1e-12)
        rand = []
        N = int(round(B['P']['T_END'] / T_M)) + 2
        rng = np.random.default_rng(seed0 + A)
        for j in range(n_sched):
            D = random_schedule(rng, N, A - 1)
            Rr = simulate(B, h=H_SCHED, fifo=D)
            kr = kpis(Rr, B)
            ba = bounds_rows(ev['consts'], cfg, kr, Rr['plan']['pieces']) \
                if ev['closed'] else None
            bd = bounds_rows(K20, cfg, kr, Rr['plan']['pieces'])
            r = compact(kr, cfg)
            r['bounds_at_age_ok'] = None if ba is None else ba['all_ok']
            r['bounds_at_age_failed'] = None if ba is None else ba['failed']
            r['n_checked_at_age'] = None if ba is None else ba['n_checked']
            r['n_rows_at_age'] = None if ba is None else ba['n_rows']
            r['design_bounds_ok'] = bd['all_ok']
            r['design_bounds_failed'] = bd['failed']
            r['age_ok'] = bool(Rr['age']['max_left_limit'] <= db + 1e-12)
            rand.append(r)
        allr = [const] + rand

        def ext(key, fn):
            return (_max if fn is max else _min)(r[key] for r in allr)
        row = dict(
            age_ms=A, age_bound=db, transport_steps=A - 1,
            certificate=dict(admitted=ev['admitted'], failed=ev['failed'],
                             closed=ev['closed'],
                             conds={nm: dict(ok=c['ok'], slack=c['slack'])
                                    for nm, c in ev['conds'].items()},
                             bounds=None if not ev['closed'] else dict(
                                 settling_bound=ev['consts']['settling_bound'],
                                 T_f_bar=ev['consts']['T_f_bar'],
                                 align=ev['consts']['align'],
                                 gaperr_max=max(ev['consts']['gaperr']),
                                 T_ent_bar=ev['consts']['T_ent_bar'])),
            constant=const, random=rand,
            summary=dict(
                settling_max=ext('settling', max), settling_const=const['settling'],
                marker_max=ext('marker_abs_max', max),
                gap_max=ext('gap_abs_max', max),
                u_min=ext('u_min', min), u_max=ext('u_max', max),
                u_max_pre=ext('u_max_pre', max),
                commands_in_set=bool(ext('u_min', min) >= -float(cfg['k'])
                                     and ext('u_max', max) <= 0.0),
                filter_modifications=sum(r['filter_modifications'] for r in allr),
                saturations=sum(r['n_sat'] for r in allr),
                entry_max=_max(r['entry_obs'] for r in allr),
                entry_const=const['entry_obs'],
                all_stopped=all(r['all_stopped'] for r in allr),
                no_contact=all(r['contact_time'] is None for r in allr),
                tail_first_head_last=all(r['tail_first_head_last'] for r in allr),
                rule_entered_any=any(r['rule_entered'] for r in allr),
                requirements_ok=all(r['requirements']['all_ok'] for r in allr),
                requirements_failed=sorted(set(x for r in allr
                                               for x in r['requirements']['failed'])),
                bounds_at_age_ok=None if not ev['closed'] else all(
                    r['bounds_at_age_ok'] for r in allr),
                bounds_at_age_failed=None if not ev['closed'] else sorted(set(
                    x for r in allr for x in r['bounds_at_age_failed'])),
                design_bounds_ok=all(r['design_bounds_ok'] for r in allr),
                design_bounds_failed=sorted(set(x for r in allr
                                                for x in r['design_bounds_failed'])),
                age_ok=all(r['age_ok'] for r in allr)))
        rows.append(row)
        s = row['summary']
        print('  %4d ms cert %s %-26s | settling <= %.5f s marker <= %.4f m gap <= '
              '%.4f m u in [%.4f, %.4f] mods %d entry <= %s req %s bounds@age %s '
              'design %s (%.0f s)'
              % (A, 'ADM' if ev['admitted'] else ('cl ' if ev['closed'] else '---'),
                 ','.join(ev['failed'])[:26], s['settling_max'], s['marker_max'],
                 s['gap_max'], s['u_min'], s['u_max'], s['filter_modifications'],
                 '%.3f' % s['entry_max'] if s['entry_max'] is not None else 'none',
                 s['requirements_failed'] or 'ok',
                 s['bounds_at_age_failed'] if s['bounds_at_age_ok'] is False
                 else s['bounds_at_age_ok'],
                 s['design_bounds_failed'] or 'ok', time.time() - t0), flush=True)
    # first failures along the age axis
    first = {}
    for row in rows:
        A = row['age_ms']
        s = row['summary']
        if 'certificate' not in first and not row['certificate']['admitted']:
            first['certificate'] = dict(age_ms=A, failed=row['certificate']['failed'])
        if 'bound_at_age' not in first and s['bounds_at_age_ok'] is False:
            first['bound_at_age'] = dict(age_ms=A, failed=s['bounds_at_age_failed'])
        if 'design_bound' not in first and not s['design_bounds_ok']:
            first['design_bound'] = dict(age_ms=A, failed=s['design_bounds_failed'])
        if 'requirement' not in first and not s['requirements_ok']:
            first['requirement'] = dict(age_ms=A, failed=s['requirements_failed'])
        if 'closure' not in first and not row['certificate']['closed']:
            first['closure'] = dict(age_ms=A)
    # per requirement: the first age with a failing run and the last age
    # before it
    first_req = {}
    for row in rows:
        for nm in row['summary']['requirements_failed']:
            if nm not in first_req:
                prev = [r['age_ms'] for r in rows if r['age_ms'] < row['age_ms']]
                first_req[nm] = dict(age_ms=row['age_ms'],
                                     last_age_holding_ms=max(prev) if prev else None)
    first['requirement_each'] = first_req
    bnd = cq.age_boundary_ms(cfg, kmax=400)
    # per-condition boundaries with the switch box of the certificate
    # (run_caseD_design.cond_boundaries)
    cb = cq.cond_boundaries_ms(cfg, kmax=400)
    cb.update(cq.cond_boundaries_ms(dict(cfg, one_sided=None), kmax=400,
                                    names=('H1D_i', 'H1D_ii', 'H6D')))
    adm = [r['age_ms'] for r in rows if r['certificate']['admitted']]
    print('  boundary %s ms, next fails %s; first failures %s'
          % (bnd['k_ms'], bnd.get('failed_next'), first), flush=True)
    out = dict(study=('age sweep: certificate (certify_plan) at every age '
                      'bound; constant transport age - T_m (held age exactly '
                      'the age bound) at h = %g s; %d random schedules per age '
                      'with d_j in 1..age/T_m - 1 at h = %g s; observations '
                      'against the certificate evaluated at that age (when its '
                      'recursion closes) and against the design certificate at '
                      '20 ms; requirements: settling time <= 1.0 s, marker '
                      'error <= 3 m, terminal gap error <= 1 m, no filter '
                      'modification or saturation, no contact, all stopped'
                      % (H_FINE, n_sched, H_SCHED)),
               ages_ms=list(ages_ms), rows=rows, certified_ages_ms=adm,
               boundary_ms=bnd, cond_boundary_ms=cb, first_failures=first,
               admitted_rows_inside_bounds=all(
                   r['summary']['bounds_at_age_ok'] for r in rows
                   if r['certificate']['admitted']),
               admitted_rows_all_checked=all(
                   x['n_checked_at_age'] == x['n_rows_at_age']
                   for r in rows if r['certificate']['admitted']
                   for x in [r['constant']] + r['random']),
               admitted_rows_filter_inactive=all(
                   r['summary']['filter_modifications'] == 0 for r in rows
                   if r['certificate']['admitted']),
               admitted_rows_requirements_ok=all(
                   r['summary']['requirements_ok'] for r in rows
                   if r['certificate']['admitted']),
               all_rows_commands_in_set=all(r['summary']['commands_in_set']
                                            for r in rows),
               runtime_s=time.time() - t0)
    dump('sweep.json', out)


def ens_plan_fan(cfg, c0, v0, margin_up=None):
    """Plan of a takeover state by the fan design rule
    (run_caseD_design.fan_design_for) with the drawn clearances: planned
    initial speed v^xi = v_1(0) - (v_1(0) - v^xi of the design), so the head
    runs a braking S1 stage; onset t_0, head main rate a_head, approach at
    the crawl rate and tail crawl T_xi - t^c_n of the design; the crawl rate
    a_c at least the largest certified positive envelope F^+_i(0) of this
    state plus the design margin (0.005 m/s^2) and at least F^+_i(0) at
    21 ms (F^+_i(0) does not depend on the plan, Lemma D1(e); it is read
    from the certificate of a probe plan: the design's G and L_1 with the
    drawn closures); the head main duration L_1 and the mean join offset
    delta_bar on the 1 ms grid by run_caseD_design.fan_best with the mean
    closure C_bar = sum_i C_i/(n - 1) and G = 2 C_bar/delta_bar, which
    minimizes T_xi + v_c1/a_b subject to v_c1 >= 1.0 m/s; the closure
    equations with the drawn closures give L_i = L_{i-1} + 2 C_i/G, so
    that L_n = L_1 + (n - 1) delta_bar: t_0, t^c_1, t^c_n and T_xi lie on
    the 1 ms grid and the other joins in general off it (processed at their
    exact instants by the continuous law); as in fan_design_for, a_c_min
    rises by 1e-5 while (H1D)(i) misses the margin.  For the nominal state
    the rule returns the plan of the design (asserted in part_ens).
    Returns (plan dict, tail crawl in s, head S1 deadline in s, info)."""
    import run_caseD_design as rd
    Pl0 = cq.plan(cfg)
    if margin_up is None:
        margin_up = float(load_design()['design_margins']['authority_upper_m_s2'])
    n = int(cfg['n'])
    ms = Fr(1, 1000)
    t0_ms = int(round(float(Pl0.extra['t_0']) * 1000))
    tail = Pl0.T_xi - Pl0.extra['t_c'][-1]
    tail_ms = int(tail / ms)
    assert tail_ms * ms == tail, 'tail crawl off the 1 ms grid'
    a_head = Pl0.extra['a_head']
    v_xi = cq.F(v0[0]) - (cq.F(cfg['v0'][0]) - Pl0.v_xi)
    d_s = [cq.F(x) for x in cfg['d_s']]
    C = [cq.F(c0[i - 1]) - d_s[i] for i in range(1, n)]
    C_bar = sum(C) / (n - 1)

    ramp = cfg['plan'].get('ramp')
    Tp_ms = None if cfg['plan'].get('T_p') is None else \
        int(cq.F(cfg['plan']['T_p']) / ms)

    def mk(L1, G, T):
        pl = dict(family='fan', v_xi=str(v_xi), t_0=str(t0_ms * ms), L_1=str(L1),
                  G=str(G), a_head=str(a_head), T_xi=str(T))
        if ramp is not None:
            pl.update(ramp=ramp, T_p=cfg['plan']['T_p'])
        return pl
    G0 = Pl0.extra['G']
    L10 = Pl0.extra['L'][0]
    Tp = t0_ms * ms + L10 + 2 * sum(C) / G0 + tail
    Tp = -((-Tp) // ms) * ms                           # up to the 1 ms grid
    cfgp = dict(cfg, c0=c0, v0=v0, plan=mk(L10, G0, Tp))
    info = dict(margin_up=margin_up, C_bar=float(C_bar))
    t_head = None
    base = float(Pl0.extra['a_c'])
    try:
        e20 = cq.evaluate(cfgp, want_entry=False)
        e21 = cq.evaluate(dict(cfgp, dbar=0.021), want_entry=False)
        if e20['closed'] and e21['closed']:
            fp20 = max(e20['consts']['F_plus_0'])
            fp21 = max(e21['consts']['F_plus_0'])
            base = max(fp20 + margin_up + 1e-9, fp21 + 1e-9)
            t_head = float(e20['consts']['tset'][0])
            info.update(F_plus_max_20ms=fp20, F_plus_max_21ms=fp21)
        else:
            info['probe_not_closed'] = True
    except Exception as ex:                          # pragma: no cover
        info['probe_error'] = str(ex)
    for it in range(40):
        sol = rd.fan_best(Fr(base).limit_denominator(10 ** 12), t0_ms, a_head,
                          tail_ms, C=C_bar, n=n, v_xi=v_xi, a_b=cq.F(cfg['a_b']),
                          ramp=None if ramp is None else cq.F(ramp), Tp_ms=Tp_ms)
        if sol is None:
            info.update(feasible=False, a_c_min=base, iterations=it + 1)
            return cfgp['plan'], float(tail), t_head, info
        pl = mk(sol['L1_ms'] * ms, sol['G'], sol['T_xi'])
        info.update(feasible=True, a_c_min=base, iterations=it + 1,
                    L1_ms=sol['L1_ms'], delta_bar_ms=sol['delta_ms'],
                    a_c=float(sol['a_c']), v_c1=float(sol['v_c1']))
        try:
            ev = cq.evaluate(dict(cfg, c0=c0, v0=v0, plan=pl), want_entry=False)
            sl = ev['conds']['H1D_i']['slack']
        except Exception:                            # pragma: no cover
            sl = None
        if sl is not None and sl < margin_up:
            base += 1e-5
            continue
        return pl, float(tail), t_head, info
    info.update(feasible=True, note='margin iteration did not converge')
    return pl, float(tail), t_head, info


def ens_plan(cfg, c0, v0):
    """Plan of a takeover state by the dispatch rule of the design
    (run_caseD_design.py; round 5, fan family: ens_plan_fan; round 4,
    canonical family: this function): planned initial speed v^xi = v_1(0) - 0.02 m/s
    (the design's head mismatch, so the head runs a braking S1 stage); the
    speed drop Delta_v, pulse duration D and rates a_c, a_d of the design;
    head onset t^d_1 at the first 1 ms instant at or after the head's
    certified S1 detection deadline of this state; offsets C_i / Delta_v
    from the drawn closures; T_xi the first 1 ms instant after the tail
    joins the crawl (T_xi - t^c_n in (0, 1] ms); v_c1 = v^xi - Delta_v -
    a_c T_xi.  Returns (plan dict, tail crawl in s, head deadline in s,
    info)."""
    if cfg['plan'].get('family') == 'fan':
        return ens_plan_fan(cfg, c0, v0)
    Pl0 = cq.plan(cfg)
    dv = Pl0.extra['Delta_v']
    D = Pl0.extra['D']
    a_c = Pl0.params['a_c']
    v_xi = cq.F(v0[0]) - (cq.F(cfg['v0'][0]) - Pl0.params['v_xi'])
    d_s = [cq.F(x) for x in cfg['d_s']]
    C = [cq.F(c0[i - 1]) - d_s[i] for i in range(1, len(d_s))]
    ms = Fr(1, 1000)

    def build_plan(t_d1):
        tcn = t_d1 + sum(C) / dv + D
        T_xi = (tcn // ms + 1) * ms
        v_c1 = v_xi - dv - a_c * T_xi
        return dict(cfg['plan'], v_xi=str(v_xi), t_d1=str(t_d1), T_xi=str(T_xi),
                    v_c1=str(v_c1), grid_ms=None), float(T_xi - tcn)
    # the S1 detection deadlines do not depend on the plan (Lemma D1(e))
    pl, _ = build_plan(Pl0.params['t_d1'])
    t_head = None
    try:
        ev = cq.evaluate(dict(cfg, c0=c0, v0=v0, plan=pl), want_entry=False)
        if ev['closed']:
            t_head = float(ev['consts']['tset'][0])
    except Exception:                                # pragma: no cover
        t_head = None
    t_d1 = Pl0.params['t_d1'] if t_head is None else \
        Fr(int(np.ceil(t_head / 1e-3 - 1e-9)), 1000)
    pl, tail = build_plan(t_d1)
    return pl, tail, t_head, None


def part_ens(n_draw=100, seed=7):
    cfg = design_cfg()
    rng = np.random.default_rng(seed)
    v0n = [float(cq.F(x)) for x in cfg['v0']]
    inc0 = [round(v0n[i] - v0n[i - 1], 4) for i in range(1, len(v0n))]
    ranges = dict(c0=[47.0, 53.0], v1=[10.42, 10.52],
                  increments=[[round(x - 0.05, 4), round(x + 0.05, 4)] for x in inc0])
    print('[ens] %d takeover states: c_i(0) in %s m, v_1(0) in %s m/s, '
          'v_i(0) - v_{i-1}(0) in %s m/s (seed %d)'
          % (n_draw, ranges['c0'], ranges['v1'], ranges['increments'], seed),
          flush=True)
    fan = cfg['plan'].get('family') == 'fan'
    rule_nominal = None
    if fan:
        # the rule reproduces the plan of the design at the nominal state
        pl_nom, _, _, info_nom = ens_plan(cfg, cfg['c0'], cfg['v0'])
        P_nom = cq.plan(dict(cfg, plan=pl_nom))
        P_des = cq.plan(cfg)
        same = (P_nom.T_xi == P_des.T_xi and P_nom.v_c1 == P_des.v_c1
                and all(P_nom.units[i] == P_des.units[i] for i in range(P_des.n)))
        rule_nominal = dict(plan=pl_nom, info=info_nom, reproduces_design=bool(same))
        print('  rule at the nominal state reproduces the design plan: %s (%s)'
              % (same, json.dumps(_py(info_nom))), flush=True)
        assert same, 'ensemble rule does not reproduce the design plan'
    rows = []
    t0 = time.time()
    for j in range(n_draw):
        c0 = [round(float(x), 3) for x in rng.uniform(*ranges['c0'], 4)]
        v1 = round(float(rng.uniform(*ranges['v1'])), 4)
        inc = [round(float(rng.uniform(*r)), 4) for r in ranges['increments']]
        v0 = [v1]
        for x in inc:
            v0.append(round(v0[-1] + x, 4))
        cq._NS.clear()        # one namespace per state; keeps memory flat
        plan_j, tail, t_head, info_j = ens_plan(cfg, c0, v0)
        cfg_j = dict(cfg, c0=c0, v0=v0, plan=plan_j)
        Pl_j = cq.plan(cfg_j)
        td_j, tc_j = plan_events(Pl_j)
        try:
            ev = certificate(cfg_j)
            cert = dict(admitted=ev['admitted'], failed=ev['failed'],
                        closed=ev['closed'],
                        slack={nm: c['slack'] for nm, c in ev['conds'].items()})
            K = ev['consts'] if ev['closed'] else None
            if K is not None:
                cert.update(settling_bound=K['settling_bound'], T_f_bar=K['T_f_bar'],
                            align=K['align'], gaperr_max=max(K['gaperr']),
                            T_ent_bar=K['T_ent_bar'])
        except Exception as ex:                      # pragma: no cover
            ev = None; K = None
            cert = dict(admitted=False, failed=['error'], error=str(ex))
        B = build(cfg_j)
        R = simulate(B, h=H_FINE)
        k = kpis(R, B)
        bt = bounds_rows(K, cfg_j, k, R['plan']['pieces']) if K is not None else None
        r = compact(k, cfg_j, bt)
        r.update(index=j, c0=c0, v0=v0, eps0=cq.eps0_of(cfg_j),
                 v_xi=float(Pl_j.v_xi),
                 t_d1=td_j[0], head_deadline=t_head,
                 T_xi=float(Pl_j.T_xi),
                 v_c1=float(Pl_j.v_c1), tail_crawl=tail,
                 a_c=float(Pl_j.params['a_c']), t_c=tc_j,
                 plan=plan_j, rule=info_j,
                 certificate=cert)
        rows.append(r)
        if (j + 1) % 10 == 0:
            na = sum(1 for x in rows if x['certificate']['admitted'])
            print('  %3d/%d: %d admitted; last adm=%s failed=%s %s (%.0f s)'
                  % (j + 1, n_draw, na, cert['admitted'], cert['failed'],
                     summary(k), time.time() - t0), flush=True)
    adm = [r for r in rows if r['certificate']['admitted']]
    rej = [r for r in rows if not r['certificate']['admitted']]
    fails = {}
    for r in rej:
        for nm in r['certificate']['failed']:
            fails[nm] = fails.get(nm, 0) + 1

    def grp(rs):
        if not rs:
            return None
        return dict(
            n=len(rs),
            bounds_all_ok=all(r.get('bounds_ok') for r in rs)
            if all('bounds_ok' in r for r in rs) else None,
            n_bounds_failed=sum(1 for r in rs if r.get('bounds_ok') is False),
            bounds_failed=sorted(set(x for r in rs for x in r.get('bounds_failed') or [])),
            requirements_all_ok=all(r['requirements']['all_ok'] for r in rs),
            n_requirements_failed=sum(1 for r in rs if not r['requirements']['all_ok']),
            requirements_failed=sorted(set(x for r in rs
                                           for x in r['requirements']['failed'])),
            filter_modifications_total=sum(r['filter_modifications'] for r in rs),
            saturations_total=sum(r['n_sat'] for r in rs),
            settling_max=_max(r['settling'] for r in rs),
            marker_max=_max(r['marker_abs_max'] for r in rs),
            gap_max=_max(r['gap_abs_max'] for r in rs),
            u_min=_min(r['u_min'] for r in rs), u_max=_max(r['u_max'] for r in rs),
            u_max_pre=_max(r['u_max_pre'] for r in rs),
            entry_max=_max(r['entry_obs'] for r in rs),
            all_stopped=all(r['all_stopped'] for r in rs),
            no_contact=all(r['contact_time'] is None for r in rs),
            n_handoff_box_violated=sum(1 for r in rs if r['e_hnd_max'] > float(cfg['eps_e'])
                                       or r['eps_hnd_max'] > float(cfg['eps_v'])),
            T_xi_range=[min(r['T_xi'] for r in rs), max(r['T_xi'] for r in rs)],
            v_c1_range=[min(r['v_c1'] for r in rs), max(r['v_c1'] for r in rs)],
            a_c_range=[min(r['a_c'] for r in rs), max(r['a_c'] for r in rs)],
            T_f_bar_max=_max(r['certificate'].get('T_f_bar') for r in rs),
            settling_bound_max=_max(r['certificate'].get('settling_bound') for r in rs),
            tail_first_head_last_all=all(r['tail_first_head_last'] for r in rs),
            rule_entered_any=any(r['rule_entered'] for r in rs))
    agg = dict(n=n_draw, seed=seed, ranges=ranges, n_admitted=len(adm),
               n_rejected=len(rej), failures_among_rejected=fails,
               admitted=grp(adm), rejected=grp(rej))
    print('  admitted %d of %d; failures among rejected %s' % (len(adm), n_draw, fails),
          flush=True)
    print('  admitted: %s' % json.dumps(_py(agg['admitted']))[:600], flush=True)
    print('  rejected: %s' % json.dumps(_py(agg['rejected']))[:600], flush=True)
    rule_fan = ('fan design rule (ens_plan_fan: v^xi = v_1(0) - 0.02 m/s; '
                'onset t_0 = 10 s, head main rate, approach at the crawl rate '
                'and tail crawl of 5 s of the design; crawl rate a_c at least '
                'the largest certified F^+_i(0) of the state plus 0.005 '
                'm/s^2 and at least F^+_i(0) at 21 ms; L_1 and the mean join '
                'offset on the 1 ms grid minimizing T_xi + v_c1/a_b with v_c1 '
                '>= 1.0 m/s; L_i from the closure equations with the drawn '
                'clearances, so that t_0, t^c_1, t^c_n and T_xi lie on the '
                '1 ms grid and the other joins in general off it, processed '
                'at their exact instants by the continuous law; at the '
                'nominal state the rule returns the plan of the design)')
    rule_can = ('dispatch rule of the design (ens_plan: v^xi = v_1(0) - '
                '0.02 m/s; Delta_v, D, a_c and a_d of the design; head '
                'onset at the first 1 ms instant at or after the head\'s '
                'certified S1 detection deadline; offsets C_i/Delta_v; '
                'T_xi the first 1 ms instant after t^c_n; v_c1 from the '
                'crawl line; jump instants in general off the 1 ms '
                'grid, processed at their exact instants by the '
                'continuous law)')
    out = dict(study=('takeover-state ensemble around the nominal state: '
                      'clearances c_i(0) uniform in [47, 53] m (3 decimals), '
                      'head speed v_1(0) uniform in [10.42, 10.52] m/s, speed '
                      'increments v_i(0) - v_{i-1}(0) uniform within +-0.05 '
                      'm/s of the nominal increments (0.08, 0.12, 0.18, 0.25) '
                      'm/s (4 decimals), seed %d; plan of every state by the '
                      '%s; certificate at 20 ms and continuous '
                      'run (h = %g s, transport 19 ms) for every state, the '
                      'rejected ones included'
                      % (seed, rule_fan if fan else rule_can, H_FINE)),
               design_plan=cfg['plan'], rule_at_nominal_state=rule_nominal,
               aggregate=agg, rows=rows, runtime_s=time.time() - t0)
    dump('ensemble.json', out)


def _plan_fixed_Txi(cfg, a_c, a_d, T_xi, v_c1=Fr(1)):
    """Canonical plan with the crawl rate a_c and the main deceleration a_d
    changed, T_xi and the head onset of the design kept and the crawl end
    speed v_c1 = 1 m/s: Delta_v = v^xi - v_c1 - a_c T_xi, D = Delta_v /
    (a_d - a_c), offsets C_i / Delta_v (jump instants in general off the
    1 ms grid, processed at their exact instants by the continuous law)."""
    v_xi = cq.F(cfg['plan']['v_xi'])
    t_d1 = cq.F(cfg['plan']['t_d1'])
    d_s = [cq.F(x) for x in cfg['d_s']]
    C = [cq.F(cfg['c0'][i - 1]) - d_s[i] for i in range(1, len(d_s))]
    dv = v_xi - v_c1 - a_c * T_xi
    tcn = t_d1 + sum(C) / dv + dv / (a_d - a_c)
    assert tcn < T_xi, (float(tcn), float(T_xi))
    return dict(family='canonical', v_xi=str(v_xi), a_c=str(a_c), a_d=str(a_d),
                t_d1=str(t_d1), T_xi=str(T_xi), v_c1=str(v_c1), grid_ms=None)


def _fan_plan_fixed(cfg, a_c=None, a_head=None):
    """Fan plan with the onset t_0, the drop G and T_xi of the design kept:
    with a_c given, the approach and crawl rate a_c and L_1 of the design
    (a_head = a_c + G/L_1 follows); with a_head given, the crawl rate of
    the design and L_1 = G/(a_head - a_c) (the joins move earlier; jump
    instants in general off the 1 ms grid, processed at their exact
    instants by the continuous law)."""
    Pl0 = cq.plan(cfg)
    p = cfg['plan']
    G = Pl0.extra['G']
    pl = dict(family='fan', v_xi=p['v_xi'], t_0=p['t_0'], G=str(G),
              T_xi=p['T_xi'], grid_ms=None)
    ramped = 'ramp_steps' in Pl0.extra
    if a_c is not None:
        # L_1 and the ramp steps of the design; the peak rates follow
        pl.update(L_1=p['L_1'], a_c=str(a_c))
        if ramped:
            pl.update(T_p=p['T_p'], ramp_steps=list(Pl0.extra['ramp_steps']))
    else:
        a_c0 = Pl0.extra['a_c']
        D = G / (Fr(a_head) - a_c0)          # effective duration L_1 - T^r_1
        if ramped:
            # ramp fraction of the design: L_1 = D + N T_p with
            # N = floor(rho L_1/T_p + 1/2) (the smallest consistent N)
            rho = Pl0.extra['ramp']; T_p = Pl0.extra['T_p']
            N = 0
            while (rho * (D + N * T_p) / T_p + Fr(1, 2)).__floor__() != N:
                N += 1
                assert N < 10 ** 5
            pl.update(L_1=str(D + N * T_p), a_c=str(a_c0), ramp=p['ramp'],
                      T_p=p['T_p'])
        else:
            pl.update(L_1=str(D), a_c=str(a_c0))
    return pl


def _main_rate(Pl):
    """largest planned main rate: the head rate a_head (fan) or a_d
    (canonical)."""
    return float(Pl.extra['a_head']) if Pl.family == 'fan' else float(Pl.params['a_d'])


def _edge_row(label, cfg, axis):
    ev = certificate(cfg)
    K = ev['consts'] if ev['closed'] else None
    B = build(cfg)
    R = simulate(B, h=H_FINE)
    k = kpis(R, B)
    bt = bounds_rows(K, cfg, k, R['plan']['pieces']) if K is not None else None
    Pl = B['Pl']
    row = dict(label=label, axis=axis, lam=float(cfg['lam']), family=Pl.family,
               a_c=float(Pl.params['a_c']), a_d=_main_rate(Pl),
               T_xi=float(Pl.T_xi), v_c1=float(Pl.v_c1),
               admitted=ev['admitted'], failed=ev['failed'],
               slack={nm: ev['conds'][nm]['slack'] for nm in ('H1D_i', 'H1D_ii', 'H3D', 'P_S1')},
               u_min=k['u_min'], u_max=k['u_max'], u_max_pre=k['u_max_pre'],
               n_sat=k['n_sat'], n_mod=k['n_mod'], n_fallback=k['n_fallback'],
               filter_modifications=k['filter_modifications'],
               settling=k['settling'], marker_abs_max=k['marker_abs_max'],
               gap_abs_max=k['gap_abs_max'], stop_order=k['stop_order'],
               rule_entered=k['rule_entered'], all_stopped=k['all_stopped'],
               requirements=requirements(cfg, k))
    if K is not None:
        up_ = bt['rows']['deviation_upper_pieces']
        lo_ = bt['rows']['deviation_lower_pieces']
        row.update(
            bounds_ok=bt['all_ok'], bounds_failed=bt['failed'],
            n_checked=bt['n_checked'], n_rows=bt['n_rows'],
            F_plus_tail_first=[a['F_plus'] for a in K['authority']
                               if a['unit'] == int(cfg['n']) and a['t0'] == 0.0][0],
            dev_pos_tail_first=k['dev_pos'][int(cfg['n']) - 1][0],
            # smallest distance of a certified piece envelope from the
            # observed deviation on that piece (upper: F^+ - max(u - a*),
            # lower: F^- - max(a* - u))
            upper_env_gap=min(b - o for b, o in zip(up_['bound'], up_['observed'])),
            lower_env_gap=min(b - o for b, o in zip(lo_['bound'], lo_['observed'])))
    print('  %-34s adm=%-5s failed=%-14s H1D_i=%+.6f H1D_ii=%+.6f | u in [%.5f, %.6f] '
          'sat=%d mods=%d rule=%s req=%s'
          % (label, ev['admitted'], ','.join(ev['failed'])[:14], row['slack']['H1D_i'],
             row['slack']['H1D_ii'], k['u_min'], k['u_max_pre'], k['n_sat'],
             k['filter_modifications'], k['rule_entered'], row['requirements']['failed']),
          flush=True)
    return row


def part_edge():
    """The input-set guarantee at its edge (round-4 review, finding
    validation_logic-02): the certified design without the authority
    margin (Table D.III), the crawl rate a_c lowered through the positive
    envelope F^+_5(0) so that (H1D)(i) fails, and the main deceleration a_d
    raised toward k so that (H1D)(ii) becomes tight and fails; lambda,
    pad, head onset and T_xi of the selected design kept, v_c1 = 1 m/s;
    continuous reference law at h = 25 us, constant transport 19 ms.

    Round 5, fan family: the certified fan without the design margins
    (run_caseD_design.fan_design_for with pref=False, reserve=False: a_c at
    F^+_max(0), plan re-optimized, the trade-off row 'no design margins'),
    the approach and crawl rate a_c lowered through F^+_5(0) with t_0, L_1,
    G and T_xi kept, and the head main rate a_head raised toward k with
    t_0, a_c, G and T_xi kept (L_1 = G/(a_head - a_c)); _fan_plan_fixed."""
    des = load_design()
    base = dict(des['cfg'])
    T_sel = cq.F(base['plan']['T_xi'])
    fan = base['plan'].get('family') == 'fan'
    rows = [_edge_row('selected design', base, 'selected')]
    if fan:
        import run_caseD_design as rd
        print('[edge] authority edge along a_c and a_head (fan; t_0, G and '
              'T_xi = %s s kept)' % float(T_sel), flush=True)
        nm_row = rd.fan_design_for(float(base['lam']), float(base['vhnd']),
                                   pref=False, reserve=False,
                                   box=(float(base['eps_e']), float(base['eps_v'])))
        assert nm_row.get('feasible'), nm_row.get('note')
        rows.append(_edge_row('no design margins (trade-off row)', dict(nm_row['cfg']),
                              'no margin'))
        for a_c in ('0.003', '0.0015', '0.0009', '0.00085', '0.0005', '0.0002',
                    '0.00012', '0.0001', '0.00005'):
            cfg = dict(base, plan=_fan_plan_fixed(base, a_c=Fr(a_c)))
            rows.append(_edge_row('a_c = %s' % a_c, cfg, 'a_c'))
        for a_h in ('1.19', '1.199', '1.1993', '1.1994', '1.1996', '1.2'):
            cfg = dict(base, plan=_fan_plan_fixed(base, a_head=Fr(a_h)))
            rows.append(_edge_row('a_head = %s' % a_h, cfg, 'a_d'))
    else:
        a_c0 = cq.F(base['plan']['a_c'])
        print('[edge] authority edge along a_c and a_d (T_xi = %s s kept)'
              % float(T_sel), flush=True)
        na = des['alternatives']['no_authority_margin']
        rows.append(_edge_row('no authority margin (Table D.III)', dict(na['cfg']),
                              'no margin'))
        for a_c in (Fr(21, 10000), Fr(15, 10000), Fr(11, 10000), Fr(8, 10000),
                    Fr(5, 10000)):
            cfg = dict(base, plan=_plan_fixed_Txi(base, a_c, Fr(1), T_sel))
            rows.append(_edge_row('a_c = %s' % float(a_c), cfg, 'a_c'))
        for a_d in (Fr(119, 100), Fr(1195, 1000), Fr(1197, 1000), Fr(1199, 1000)):
            cfg = dict(base, plan=_plan_fixed_Txi(base, a_c0, a_d, T_sel))
            rows.append(_edge_row('a_d = %s' % float(a_d), cfg, 'a_d'))
    adm = [r for r in rows if r['admitted']]
    rej = [r for r in rows if not r['admitted']]
    clip_a_c = [r['a_c'] for r in rows if r['axis'] == 'a_c' and r['n_sat'] > 0]
    noclip_rej_a_c = [r['a_c'] for r in rows if r['axis'] == 'a_c' and not r['admitted']
                      and r['n_sat'] == 0]
    clip_a_d = [r['a_d'] for r in rows if r['axis'] == 'a_d' and r['n_sat'] > 0]
    agg = dict(
        n=len(rows), n_admitted=len(adm),
        admitted_filter_inactive=all(r['filter_modifications'] == 0 for r in adm),
        admitted_inside_bounds=all(r['bounds_ok'] for r in adm),
        admitted_all_checked=all(r['n_checked'] == r['n_rows'] for r in adm),
        admitted_requirements_ok=all(r['requirements']['all_ok'] for r in adm),
        rejected_failures=sorted(set(x for r in rej for x in r['failed'])),
        clip_first_a_c=max(clip_a_c) if clip_a_c else None,
        smallest_unclipped_rejected_a_c=min(noclip_rej_a_c) if noclip_rej_a_c else None,
        clip_first_a_d=min(clip_a_d) if clip_a_d else None,
        rule_entered_any=any(r['rule_entered'] for r in rows),
        requirements_ok_all=all(r['requirements']['all_ok'] or r['n_sat'] > 0 for r in rows))
    if fan:
        study = ('authority edge of the Case D certificate (fan): the selected '
                 'design, the certified fan without the design margins '
                 '(run_caseD_design.fan_design_for, pref=False, reserve=False; '
                 'a_c at F^+_max(0), plan re-optimized), the approach and '
                 'crawl rate a_c lowered through F^+_5(0) (H1D(i)) with t_0, '
                 'L_1, G and T_xi of the design kept, and the head main rate '
                 'a_head (column a_d) raised toward k = 1.2 m/s^2 (H1D(ii)) '
                 'with t_0, a_c, G and T_xi kept and L_1 = G/(a_head - a_c) '
                 '(_fan_plan_fixed); lambda, box and pad of the design; '
                 'continuous reference law at h = %g s, constant transport '
                 '19 ms; certificate at 20 ms' % H_FINE)
    else:
        study = ('authority edge of the Case D certificate: the selected '
                 'design, the certified design without the authority '
                 'margin of Table D.III, the crawl rate a_c lowered '
                 'through F^+_5(0) (H1D(i)) and the main deceleration '
                 'a_d raised toward k = 1.2 m/s^2 (H1D(ii)); lambda, '
                 'pad, head onset and T_xi of the selected design kept, '
                 'v_c1 = 1 m/s, the other plan quantities re-solved '
                 '(_plan_fixed_Txi); continuous reference law at h = %g '
                 's, constant transport 19 ms; certificate at 20 ms'
                 % H_FINE)
    out = dict(study=study, rows=rows, aggregate=agg)
    print('  aggregate: %s' % json.dumps(_py(agg)), flush=True)
    dump('edge.json', out)


def part_ps1():
    """Takeover states that violate the Case D premise P_S1 (round-4
    review, finding validation_logic-03): the speed increment of pair 3,
    v_3(0) - v_2(0), set from +0.03 down to -0.05 m/s (nominal 0.12), the
    later increments and the plans of the design kept; constant transport
    19 ms (h = 25 us) and the FIFO schedules const_max, const_min and
    max_before_all_W20ms of the adversarial part (h = 1 ms, windows from
    the nominal run of each state)."""
    base = design_cfg()
    v0n = [float(cq.F(x)) for x in base['v0']]
    inc = [round(v0n[i] - v0n[i - 1], 4) for i in range(1, len(v0n))]
    print('[ps1] pair-3 increment against P_S1 (nominal %.2f m/s)' % inc[1], flush=True)
    rows = []
    for inc3 in (0.03, 0.02, 0.01, 0.0, -0.01, -0.03, -0.05):
        d = list(inc)
        d[1] = inc3
        v0 = [v0n[0]]
        for x in d:
            v0.append(round(v0[-1] + x, 4))
        cfg = dict(base, v0=v0)
        ev = certificate(cfg)
        # round-5 review: the same state under the box rule of the certificate
        # (one-sided box only while P_S1 holds, the symmetric box otherwise);
        # the table keeps the one-sided box of the design and says so
        ev_sym = certificate(cfg, one_sided=None)
        K = ev['consts'] if ev['closed'] else None
        B = build(cfg)
        Rnom = simulate(B, h=H_SCHED)
        specs, meta, _ = adv_specs(B, Rnom, B['dmax_steps'])
        sp = dict(specs)
        runs = [('constant', None, H_FINE)] + [(nm, sp[nm], H_SCHED) for nm in
                                               ('const_max', 'const_min', 'max_before_all_W20ms')]
        res = {}
        for nm, D, h in runs:
            R = simulate(B, h=h, fifo=D)
            k = kpis(R, B)
            bt = bounds_rows(K, cfg, k, R['plan']['pieces']) if K is not None else None
            res[nm] = dict(h=h, n_sat=k['n_sat'], n_mod=k['n_mod'], n_fallback=k['n_fallback'],
                           filter_modifications=k['filter_modifications'],
                           u_min=k['u_min'], u_max_pre=k['u_max_pre'],
                           rule_entered=k['rule_entered'], stop_order=k['stop_order'],
                           tail_first_head_last=k['tail_first_head_last'],
                           settling=k['settling'], marker_abs_max=k['marker_abs_max'],
                           gap_abs_max=k['gap_abs_max'], gmin_min=k['gmin_min'],
                           eps_sw=k['eps_sw'], e_sw=k['e_sw'], tset=k['tset'],
                           eps_hnd=k['eps_hnd'], all_stopped=k['all_stopped'],
                           contact_time=k['contact_time'],
                           requirements=requirements(cfg, k),
                           bounds_ok=None if bt is None else bt['all_ok'],
                           bounds_failed=None if bt is None else bt['failed'],
                           n_checked=None if bt is None else bt['n_checked'],
                           n_rows=None if bt is None else bt['n_rows'])
        row = dict(inc3=inc3, v0=v0, eps0=K['eps0'] if K is not None else None,
                   admitted=ev['admitted'], failed=ev['failed'],
                   admitted_symmetric_box=ev_sym['admitted'],
                   failed_symmetric_box=ev_sym['failed'],
                   P_S1_slack=ev['conds']['P_S1']['slack'], runs=res,
                   settling_max=max(r['settling'] for r in res.values() if r['settling'] is not None),
                   n_sat_max=max(r['n_sat'] for r in res.values()),
                   n_rule=sum(1 for r in res.values() if r['rule_entered']),
                   requirements_ok_all=all(r['requirements']['all_ok'] for r in res.values()),
                   requirements_failed=sorted(set(x for r in res.values()
                                                  for x in r['requirements']['failed'])),
                   orders=sorted(set(tuple(r['stop_order'] or ()) for r in res.values())))
        rows.append(row)
        print('  inc3=%+.2f adm=%s failed=%s P_S1=%+.5f | sat max %d, RULE in %d of %d, '
              'settling <= %.4f s, orders %s, requirements failed %s'
              % (inc3, ev['admitted'], ev['failed'], row['P_S1_slack'], row['n_sat_max'],
                 row['n_rule'], len(res), row['settling_max'], row['orders'],
                 row['requirements_failed']), flush=True)
    adm = [r for r in rows if r['admitted']]
    agg = dict(n_states=len(rows), n_runs=sum(len(r['runs']) for r in rows),
               n_admitted=len(adm),
               admitted_inside_bounds=all(x['bounds_ok'] for r in adm for x in r['runs'].values()),
               admitted_all_checked=all(x['n_checked'] == x['n_rows'] for r in adm
                                        for x in r['runs'].values()),
               admitted_filter_inactive=all(x['filter_modifications'] == 0 for r in adm
                                            for x in r['runs'].values()),
               rejected_failures=sorted(set(x for r in rows if not r['admitted'] for x in r['failed'])),
               rejected_failures_symmetric_box=sorted(set(
                   x for r in rows if not r['admitted_symmetric_box']
                   for x in r['failed_symmetric_box'])),
               rule_states=[r['inc3'] for r in rows if r['n_rule'] > 0],
               clip_states=[r['inc3'] for r in rows if r['n_sat_max'] > 0],
               requirements_ok_all=all(r['requirements_ok_all'] for r in rows),
               settling_max=max(r['settling_max'] for r in rows))
    out = dict(study=('takeover states against the premise P_S1: the speed '
                      'increment of pair 3 from +0.03 to -0.05 m/s (nominal '
                      '0.12 m/s), the later increments and the plans of the '
                      'design kept; certificate at 20 ms; constant transport '
                      '19 ms (continuous law, h = %g s) and the FIFO schedules '
                      'const_max, const_min and max_before_all_W20ms of the '
                      'adversarial part (h = %g s)' % (H_FINE, H_SCHED)),
               rows=rows, aggregate=agg)
    print('  aggregate: %s' % json.dumps(_py(agg)), flush=True)
    dump('ps1_states.json', out)


def _ref_plan(rs, pl):
    """ref_sim plan with every jump on the double m T_m (run_indep.snap_to_grid).
    Round 5: a fan plan is built by design_checks/ref_fan.FanPlan from its
    defining equations (own exact arithmetic, no code shared with
    certify_plan.py); ref_sim.Plan builds the canonical plan."""
    if pl.get('family') == 'fan':
        import ref_fan                                 # noqa: E402
        p = ref_fan.FanPlan(v_xi=pl['v_xi'], t_0=pl['t_0'], L_1=pl['L_1'],
                            delta=pl['delta'], a_head=pl['a_head'],
                            T_xi=pl['T_xi'], a_0=pl.get('a_0'),
                            ramp=pl.get('ramp'), T_p=pl.get('T_p'),
                            ramp_steps=pl.get('ramp_steps'))
    else:
        p = rs.Plan(v_xi=pl['v_xi'], a_c=pl['a_c'], a_d=pl['a_d'], t_d1=pl['t_d1'],
                    T_xi=pl['T_xi'], v_c1=pl['v_c1'])
    jumps = sorted(set(x for i in range(len(p.T0)) for x in p.T0[i][1:]))
    ms = [int(round(x / rs.TM)) for x in jumps]
    aligned = all(abs(m * rs.TM - x) < 1e-12 for m, x in zip(ms, jumps))
    if aligned:
        mp = dict(zip(jumps, [m * rs.TM for m in ms]))
        for i in range(len(p.T0)):
            p.T0[i] = [p.T0[i][0]] + [mp[x] for x in p.T0[i][1:]]
        p.jumps = sorted(set(x for i in range(len(p.T0)) for x in p.T0[i][1:]))
        p.T_xi = mp[p.T_xi]
    return p, aligned


def part_indep():
    sys.path.insert(0, CHECKS)
    import sim_plan as sp                              # noqa: E402
    import ref_sim as rs                               # noqa: E402
    cfg = design_cfg()
    ev = certificate(cfg)
    K = ev['consts']
    B = build(cfg)
    Pl = B['Pl']
    n = int(cfg['n'])
    lam = float(cfg['lam'])
    assert all(float(x) == 50.0 for x in cfg['c0']), 'ref_sim.Plan assumes 50 m'
    T_xi = float(Pl.T_xi)
    N = int((T_xi + 10.0) / T_M)
    # delays: constant 19 steps; a random schedule (iid 1..19) projected to
    # monotone arrivals so that the three FIFO conventions coincide
    rng = np.random.default_rng(20260927)
    draw = rng.integers(1, 20, size=(n, N + 64))
    draw[0] = 1
    rand = fifo_project(draw)
    const = np.full((n, N + 64), 19, np.int64)
    const[0] = 1
    cases = [('const19_cont', 'cont', const), ('const19_zoh', 'zoh', const),
             ('random_cont', 'cont', rand)]
    cert = dict(tset=K['tset'], E1=K['E1'], Lam2=K['Lam2'])
    runs = {}
    print('[indep] sim_cont vs sim_plan.py vs ref_sim.py', flush=True)
    for name, scheme, D in cases:
        res = {}
        # sim_cont: h = 25 us ('cont') and h = 1 ms (same step as the others)
        for h in ((H_FINE, 1e-3) if scheme == 'cont' else (1e-3,)):
            R = simulate(B, h=h, scheme=scheme, fifo=D, T_u=1e-3 if scheme == 'zoh' else None)
            k = kpis(R, B)
            bt = bounds_rows(K, cfg, k, R['plan']['pieces'])
            res['sim_cont_h%g' % h] = dict(
                tau=list(k['tstop']), settling=k['settling'],
                marker_err=list(k['marker_signed']), gap_err=list(k['gap_signed']),
                u_min=k['u_min'], u_max=k['u_max'], u_max_pre=k['u_max_pre'],
                u_min_pre=k['u_min_pre'], tset=list(k['tset']), tact=list(k['tact']),
                entry=k['entry_obs'], stop_order=k['stop_order'],
                rule_entered=k['rule_entered'], filter_modifications=k['filter_modifications'],
                bounds_ok=bt['all_ok'], runtime_s=R['runtime_s'])
        # sim_plan.py (design check), 1 ms RK4 or exact zoh
        t0 = time.time()
        S = sp.Sim(Pl, cfg, D, scheme=scheme)
        r = S.run()
        res['sim_plan'] = dict(
            tau=r['tau'], settling=r['settling_time'], marker_err=r['marker_err'],
            gap_err=r['gap_err'], u_min=min(r['u_min']), u_max=max(r['u_max']),
            u_max_pre=max(r['u_max_pre']), u_min_pre=min(r['u_min_pre']),
            tset=r['t_set'], tact=r['t_act'], entry=r['entry_obs'],
            stop_order=[int(x) + 1 for x in np.argsort(r['tau'])],
            rule_entered=any(x is not None for x in r['t_rule']),
            filter_modifications=sum(r['n_filter']) + sum(r['n_sat']),
            runtime_s=time.time() - t0)
        # ref_sim.py (referee), 1 ms RK4 or pure 1 kHz zoh
        t0 = time.time()
        p, aligned = _ref_plan(rs, cfg['plan'])
        dl = [None] + [np.asarray(D[i]) for i in range(1, n)]
        sim = rs.Sim(p, dl, lam=lam, v0=tuple(B['v0']), scheme=scheme, nsub=1,
                     certified=cert, record_errors=True)
        q = sim.run()
        dg = q['diag']
        # entry into the handoff box of the design (ref_sim's own diagnostic
        # last_out_box uses the box (0.15 m, 0.02 m/s) of round 4): the 1 ms
        # instant after the last recorded instant before T_xi with a pair
        # error outside the box (run_indep.box_entry)
        ee, ev_ = float(cfg['eps_e']), float(cfg['eps_v'])
        last = -1
        for m, row in enumerate(sim.errs):
            if any(abs(row[i][0]) > ee or abs(row[i][1]) > ev_ for i in range(n)):
                last = m
        entry_ref = 0.0 if last < 0 else (last + 1) * rs.TM
        sim.errs = []
        res['ref_sim'] = dict(
            tau=q['tau'], settling=q['settling'], marker_err=q['marker_err'],
            gap_err=q['gap_err'], u_min=min(dg['umin']), u_max=max(dg['umax']),
            u_max_pre=max(dg['umax_pre']), u_min_pre=min(dg['umin_pre']),
            tset=q['tset'], tact=list(sim.tact), entry=entry_ref,
            entry_round4_box=dg['last_out_box'] + T_M,
            stop_order=[int(x) + 1 for x in np.argsort(q['tau'])],
            rule_entered=any(x is not None for x in q['rule']),
            filter_modifications=dg['filter_active'] + dg['clip_active'] + dg['fallback'],
            plan_grid_aligned=aligned, runtime_s=time.time() - t0)
        # pairwise differences
        base = 'sim_cont_h0.001'
        cmpd = {}
        for other in [x for x in res if x != base]:
            a = res[base]; b = res[other]
            cmpd['%s_vs_%s' % (base, other)] = dict(
                max_stop_diff=max(abs(x - y) for x, y in zip(a['tau'], b['tau'])),
                settling_diff=abs(a['settling'] - b['settling']),
                max_marker_diff=max(abs(x - y) for x, y in zip(a['marker_err'],
                                                              b['marker_err'])),
                max_gap_diff=max(abs(x - y) for x, y in zip(a['gap_err'], b['gap_err'])),
                u_min_diff=abs(a['u_min'] - b['u_min']),
                u_max_pre_diff=abs(a['u_max_pre'] - b['u_max_pre']),
                tset_max_diff=max(abs(x - y) for x, y in zip(a['tset'], b['tset'])),
                tact_max_diff=max(abs(x - y) for x, y in zip(a['tact'], b['tact'])),
                entry_diff=abs(a['entry'] - b['entry']),
                stop_order_equal=a['stop_order'] == b['stop_order'],
                rule_equal=a['rule_entered'] == b['rule_entered'])
        runs[name] = dict(scheme=scheme, results=res, compare=cmpd)
        for nm, c in cmpd.items():
            print('  %-14s %-34s stops %.2e s settling %.2e s markers %.2e m gaps '
                  '%.2e m u_min %.1e entry %.4f s order %s'
                  % (name, nm, c['max_stop_diff'], c['settling_diff'],
                     c['max_marker_diff'], c['max_gap_diff'], c['u_min_diff'],
                     c['entry_diff'], c['stop_order_equal']), flush=True)
    same_step = [c for r in runs.values() for nm, c in r['compare'].items()
                 if 'sim_plan' in nm or 'ref_sim' in nm]
    fine = [c for r in runs.values() for nm, c in r['compare'].items()
            if 'sim_cont_h2.5e-05' in nm]
    agg = dict(
        same_step_max_stop_diff=max(c['max_stop_diff'] for c in same_step),
        same_step_max_settling_diff=max(c['settling_diff'] for c in same_step),
        same_step_max_marker_diff=max(c['max_marker_diff'] for c in same_step),
        same_step_max_gap_diff=max(c['max_gap_diff'] for c in same_step),
        same_step_max_u_min_diff=max(c['u_min_diff'] for c in same_step),
        same_step_max_tset_diff=max(c['tset_max_diff'] for c in same_step),
        same_step_max_entry_diff=max(c['entry_diff'] for c in same_step),
        same_step_stop_order_equal=all(c['stop_order_equal'] for c in same_step),
        fine_vs_1ms_max_stop_diff=max(c['max_stop_diff'] for c in fine),
        fine_vs_1ms_max_marker_diff=max(c['max_marker_diff'] for c in fine),
        all_sim_cont_bounds_ok=all(v['bounds_ok'] for r in runs.values()
                                   for nm, v in r['results'].items()
                                   if nm.startswith('sim_cont')))
    print('  aggregate: %s' % json.dumps(_py(agg)), flush=True)
    out = dict(study=('independent-code agreement: sim_cont.py (plan=) against '
                      'the design-check simulator sim_plan.py and the referee '
                      'simulator ref_sim.py (data/new_d19/caseD/design_checks/, '
                      'no shared code) on identical delays: constant 19-step '
                      'transport (continuous law and pure 1 kHz zero-order '
                      'hold) and one random schedule (iid 1..19 steps, seed '
                      '20260927) projected to monotone arrivals, so that the '
                      'FIFO projection of sim_cont and sim_plan and the '
                      'discard rule of ref_sim coincide.  sim_plan and ref_sim '
                      'integrate with RK4 at h = 1 ms; sim_cont is compared at '
                      'h = 1 ms (same step) and at h = 25 us.  Entry: sim_cont '
                      'interpolates the last exit from the handoff box between '
                      'steps, sim_plan and ref_sim report the grid instant '
                      'after it (differences up to 1 ms)'),
               runs=runs, aggregate=agg, certificate=cert_summary(ev))
    dump('independent.json', out)


def _shifted_plan(Pl, k, dt):
    """plan dictionary of Pl with every jump strictly between 0 and T_xi
    moved by dt and the pieces chained consistently (a different plan whose
    jumps come dt late); markers and planned stop of Pl."""
    segs = []
    for i in range(Pl.n):
        m = len(Pl.segs[i])
        pcs = [(float(t0) + (dt if 0 < j < m - 1 else 0.0), float(a))
               for j, (t0, t1, a, s, v) in enumerate(Pl.segs[i])]
        s, v = float(Pl.segs[i][0][3]), float(Pl.segs[i][0][4])
        out = []
        for j, (t0, a) in enumerate(pcs):
            out.append((t0, a, s, v))
            if j + 1 < len(pcs):
                L = pcs[j + 1][0] - t0
                s = s + v * L + 0.5 * a * L * L
                v = v + a * L
        segs.append(out)
    return dict(segs=segs, markers=[float(x) for x in Pl.markers],
                tau_star=float(Pl.tau_star), s_ref_stop=float(Pl.s_ref_stop),
                k=float(k))


def _tracking_run(cfg, label, h, scheme, T_u=None, late=False):
    """sim_cont.run(plan=...) from a takeover state on the plan (v_i(0) =
    v^xi, zero pair errors): the law applies u_i = a*_i, so the simulated
    speeds equal the planned ones to rounding when every planned jump is
    processed at its exact instant; late=True moves every jump of the
    simulated plan one step h late (sensitivity control)."""
    cfg = dict(cfg, v0=[cfg['plan']['v_xi']] * int(cfg['n']))
    B = build(cfg)
    Pl = B['Pl']
    if late:
        B = dict(B, plan=_shifted_plan(Pl, B['k'], h))
    R = simulate(B, h=h, scheme=scheme, T_u=T_u, rec_dt=1e-3)
    rec = R['rec']
    t = np.asarray(rec['t'], float)
    ok = np.isfinite(t)
    m = t[ok] < float(Pl.tau_star) - 1e-3
    tt = t[ok][m]
    dv = 0.0
    for i in range(Pl.n):
        dv = max(dv, float(np.max(np.abs(rec['v'][i, ok][m] - Pl.state_f(i, tt)[1]))))
    ex = sorted(set(float(t0) for i in range(Pl.n) for (t0, a) in Pl.units[i][1:])
                | {float(Pl.T_xi)})
    sj = R['plan']['jumps']
    match = None if late else bool(len(ex) == len(sj) and max(
        abs(x - y) for x, y in zip(ex, sj)) <= 1e-12)
    row = dict(plan=label, scheme=scheme, h=h, T_u=T_u, late_by_one_step=late,
               n_jumps=len(sj), jumps_match_exact=match,
               jumps_on_h_grid=R['plan']['jumps_on_h_grid'],
               max_speed_diff_to_plan=dv,
               max_abs_pair_pos_err=float(np.max(np.abs(rec['e'][:, ok][:, m]))),
               max_abs_pair_vel_err=float(np.max(np.abs(rec['eps'][:, ok][:, m]))),
               max_abs_deviation=float(max(np.nanmax(R['dev_pos']), np.nanmax(R['dev_neg']))),
               max_stop_diff_to_tau_star=float(np.max(np.abs(
                   np.asarray(R['tstop'], float) - float(Pl.tau_star)))),
               stop_order=R['stop_order'], rule_entered=R['rule_entered'],
               filter_modifications=int(R['n_mod'] + R['n_fallback'] + R['n_sat']))
    print('  %-24s %-4s h=%-7g T_u=%-6s late=%-5s jumps %d match %s | max|v - v*| %.2e '
          '|e| %.2e |eps| %.2e |u - a*| %.2e |tau - tau*| %.2e mods %d'
          % (label, scheme, h, T_u, late, len(sj), match, dv, row['max_abs_pair_pos_err'],
             row['max_abs_pair_vel_err'], row['max_abs_deviation'],
             row['max_stop_diff_to_tau_star'], row['filter_modifications']), flush=True)
    return row


def part_exact():
    """Exact event handling of sim_cont.run(plan=...) at every planned jump
    (round 5, fan plans; sim_cont.py unchanged since round 4).  Plans: the
    selected design (every jump on the 1 ms grid), the off-grid alternative
    of design.json (joins 0.5 ms off the grid) and, for the fan, a plan with
    unequal closures (clearances 47.3, 52.1, 49.0, 50.7 m; the design's G,
    L_1, a_head and t_0; intermediate joins off the grid).  Each from a
    takeover state on the plan, with 'cont' at h = 25 us and 1 ms and
    'zoh' (T_u = 1 ms) at h = 25 us and 1 ms; the control moves every jump
    one step late.  -> plan_exactness.json"""
    import hashlib
    des = load_design()
    base = dict(des['cfg'])
    plans = [('selected design', base)]
    og = des.get('alternatives', {}).get('off_grid')
    if og is not None:
        plans.append(('off-grid alternative', dict(og['cfg'])))
    if base['plan'].get('family') == 'fan':
        G = cq.plan(base).extra['G']
        pl = dict(family='fan', v_xi=base['plan']['v_xi'], t_0=base['plan']['t_0'],
                  L_1=base['plan']['L_1'], G=str(G), a_head=base['plan']['a_head'],
                  T_xi='200')
        if base['plan'].get('ramp') is not None:
            pl.update(ramp=base['plan']['ramp'], T_p=base['plan']['T_p'])
        cu = dict(base, c0=[47.3, 52.1, 49.0, 50.7], plan=pl)
        ms = Fr(1, 1000)
        T = cq.plan(cu).extra['t_c'][-1] + 5
        pl['T_xi'] = str(-((-T) // ms) * ms)
        plans.append(('unequal closures', dict(cu, plan=pl)))
    print('[exact] planned jumps processed at their exact instants', flush=True)
    rows = []
    for label, cfg in plans:
        for h, scheme, T_u in ((H_FINE, 'cont', None), (1e-3, 'cont', None),
                               (H_FINE, 'zoh', 1e-3), (1e-3, 'zoh', 1e-3)):
            rows.append(_tracking_run(cfg, label, h, scheme, T_u))
    for label, cfg in plans[:2]:
        for h in (H_FINE, 1e-3):
            rows.append(_tracking_run(cfg, label, h, 'cont', None, late=True))
    ex = [r for r in rows if not r['late_by_one_step']]
    lt = [r for r in rows if r['late_by_one_step']]
    agg = dict(
        n_runs=len(ex), all_jumps_match=all(r['jumps_match_exact'] for r in ex),
        max_speed_diff_to_plan=max(r['max_speed_diff_to_plan'] for r in ex),
        max_abs_pair_vel_err=max(r['max_abs_pair_vel_err'] for r in ex),
        max_abs_pair_pos_err=max(r['max_abs_pair_pos_err'] for r in ex),
        max_abs_deviation=max(r['max_abs_deviation'] for r in ex),
        max_stop_diff_to_tau_star=max(r['max_stop_diff_to_tau_star'] for r in ex),
        filter_modifications=sum(r['filter_modifications'] for r in ex),
        late_min_speed_diff_to_plan=min(r['max_speed_diff_to_plan'] for r in lt) if lt else None)
    sha = hashlib.sha256(open(os.path.join(CODE, 'src', 'simulation', 'sim_cont.py'),
                              'rb').read()).hexdigest()
    out = dict(study=part_exact.__doc__.strip(), rows=rows, aggregate=agg,
               sim_cont_sha256=sha)
    print('  aggregate: %s' % json.dumps(_py(agg)), flush=True)
    dump('plan_exactness.json', out)


def _walk(a, b, path, out):
    """Leaves that differ between two loaded JSON or npz records (exact
    equality; NaN equals NaN)."""
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a or key not in b:
                out.append((path + '/' + str(key), key in a, key in b))
            else:
                _walk(a[key], b[key], path + '/' + str(key), out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, 'len %d' % len(a), 'len %d' % len(b)))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, path + '[%d]' % i, out)
        return
    if isinstance(a, float) and isinstance(b, float):
        if a == b or (a != a and b != b):
            return
    elif a == b:
        return
    out.append((path, a, b))


def _load_any(p):
    if p.endswith('.npz'):
        z = np.load(p, allow_pickle=False)
        d = {}
        for key in z.files:
            x = z[key]
            d[key] = (x.astype(float).tolist() if x.ndim else float(x)) \
                if x.dtype.kind in 'fiub' else str(x)
        return d
    return json.load(open(p))


def part_regress():
    """Default behaviour of sim_cont.py (plan=None) after the plan option:
    the Case A and Case B continuous references (run_cont), the legacy
    scheme against sim_root2 (run_cont.part_regress) and the Case C
    reference (run_caseC_validate.part_ref) are re-run into a temporary
    directory and compared leaf by leaf with the archived files.  Only run
    times may differ."""
    import hashlib
    import re
    import shutil
    import tempfile
    import run_cont as rc
    import run_caseC as rcc
    import run_caseC_validate as rcv
    data = os.path.join(CODE, 'data', 'new_d19')
    tmp = tempfile.mkdtemp(prefix='caseD_regress_')
    for m in (rc, rcc, rcv):
        m.CODE = tmp                 # used only to print relative paths
    rc.OUT = tmp
    rcc.OUT = tmp
    rcv.OUT = tmp
    meta = re.compile(r'runtime')
    cmp = {}

    def compare(name, old, new):
        diffs = []
        _walk(_load_any(old), _load_any(new), '', diffs)
        other = [d for d in diffs if not meta.search(d[0])]
        cmp[name] = dict(archived=os.path.relpath(old, CODE).replace('\\', '/'),
                         n_differing_leaves=len(diffs),
                         n_runtime_leaves=len(diffs) - len(other),
                         n_other=len(other),
                         other=[[str(x) for x in d] for d in other[:20]],
                         identical_except_runtime=not other)
        print('  %-26s %5d differing leaves, %d outside run times'
              % (name, len(diffs), len(other)), flush=True)
    print('[regress] sim_cont default path (plan=None)', flush=True)
    try:
        for c in ('B', 'A'):
            R = rc.run_case(c, scheme='cont', h=rc.H_FINE, T_m=rc.T_M, T_c=rc.T_C,
                            transport=rc.CASES[c]['transport'])
            rc.dump('cont_%s.json' % c, rc._cont_record(c, R))
            rc._traj(c, R, os.path.join(tmp, 'traj_cont_%s.npz' % c))
            compare('cont/cont_%s.json' % c, os.path.join(data, 'cont', 'cont_%s.json' % c),
                    os.path.join(tmp, 'cont_%s.json' % c))
            compare('cont/traj_cont_%s.npz' % c,
                    os.path.join(data, 'cont', 'traj_cont_%s.npz' % c),
                    os.path.join(tmp, 'traj_cont_%s.npz' % c))
        rc.part_regress()
        compare('cont/regression.json', os.path.join(data, 'cont', 'regression.json'),
                os.path.join(tmp, 'regression.json'))
        reg = json.load(open(os.path.join(tmp, 'regression.json')))
        legacy = {key: bool(v['bit_identical']) for key, v in reg['cases'].items()}
        shutil.copy(os.path.join(data, 'caseC', 'design.json'),
                    os.path.join(tmp, 'design.json'))
        rcv.part_ref()
        compare('caseC/reference.json', os.path.join(data, 'caseC', 'reference.json'),
                os.path.join(tmp, 'reference.json'))
        compare('caseC/traj_caseC.npz', os.path.join(data, 'caseC', 'traj_caseC.npz'),
                os.path.join(tmp, 'traj_caseC.npz'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    out = dict(
        study=('bit-identity of the default behaviour of sim_cont.py after '
               'the plan option (round 4, stage 2; sim_cont.py unchanged in '
               'round 5, see sim_cont_sha256): Case A and Case B '
               'continuous references, legacy scheme against sim_root2 (four '
               'cases) and the Case C reference (profile=) re-run and '
               'compared leaf by leaf with the archived files; only run '
               'times may differ'),
        sim_cont_sha256=hashlib.sha256(open(os.path.join(
            CODE, 'src', 'simulation', 'sim_cont.py'), 'rb').read()).hexdigest(),
        compared=cmp, legacy_bit_identical_to_sim_root2=legacy,
        all_identical_except_runtime=all(v['identical_except_runtime']
                                         for v in cmp.values()),
        all_legacy_bit_identical=all(legacy.values()))
    print('  identical except run times: %s; legacy bit-identical: %s'
          % (out['all_identical_except_runtime'], out['all_legacy_bit_identical']),
          flush=True)
    dump('sim_cont_regression.json', out)


def part_check():
    out = dict(files={}, all_ok=True)

    def add(name, ok, detail=None):
        out['files'][name] = dict(ok=bool(ok), detail=detail)
        if not ok:
            out['all_ok'] = False
        print('  %-20s %s %s' % (name, 'OK' if ok else 'FAIL',
                                 json.dumps(_py(detail))[:300] if detail else ''),
              flush=True)
    d = load('design.json')
    add('design.json', d is not None and d['certificate']['admitted']
        and d['interval_at_20ms']['verified'],
        None if d is None else dict(boundary_ms=d['boundary_ms']['k_ms'],
                                    T_f_bar=d['certificate']['consts']['T_f_bar']))
    r = load('reference.json')
    add('reference.json', r is not None and r['bounds']['all_ok']
        and r['bounds']['n_checked'] == r['bounds']['n_rows'] == len(r['bounds']['rows'])
        and r['pointwise']['F_pm_reimplementation_max_diff'] <= 1e-12
        and r['certificate_consistency']['same_as_design']
        and r['kpis']['filter_modifications'] == 0 and r['kpis']['commands_in_set']
        and r['kpis']['zero_after_latch'] and r['kpis']['all_stopped']
        and r['requirements']['all_ok']
        and (r.get('speed_monotonicity') is None
             or r['speed_monotonicity']['nonincreasing_all'])
        and (r.get('speed_order') is None
             or r['speed_order']['only_onset_and_join_blocks']),
        None if r is None else dict(n_checked=r['bounds']['n_checked'],
                                    failed=r['bounds']['failed'],
                                    settling=r['kpis']['settling'],
                                    u=[r['kpis']['u_min'], r['kpis']['u_max']],
                                    speed_largest_increment=None
                                    if r.get('speed_monotonicity') is None
                                    else r['speed_monotonicity']['largest_increment'],
                                    order_windows_s=None if r.get('speed_order') is None
                                    else [r['speed_order']['onset_window_max'],
                                          r['speed_order']['join_window_max']]))
    c = load('convergence.json')
    add('convergence.json', c is not None and c['all_rows_inside_bounds']
        and all(x['n_checked'] == x['n_rows'] for x in c['rows'])
        and c['all_filter_inactive'] and c['max_stop_diff'] < 1e-8,
        None if c is None else dict(max_diff=c['max_diff_over_h_and_kpis'],
                                    argmax=c['argmax_kpi'],
                                    max_stop_diff=c['max_stop_diff']))
    g = load('digital.json')
    add('digital.json', g is not None and g['all_bounds_ok'] and g['all_filter_inactive']
        and all(x['n_checked'] == x['n_rows'] for x in g['rows']),
        None if g is None else dict(max_stop_diff_1kHz=g['max_stop_diff_1kHz'],
                                    max_marker_diff_1kHz=g['max_marker_diff_1kHz']))
    for nm in ('schedules.json', 'adversarial.json'):
        s = load(nm)
        a = None if s is None else s['aggregate']
        add(nm, s is not None and a['bounds_all_ok'] and a['all_age_ok']
            and a['all_rows_checked']
            and a['all_stopped'] and a['no_contact']
            and a['filter_modifications_total'] == 0
            and a['u_abs_after_latch_max'] == 0.0
            and a['u_min']['value'] >= -1.2 and a['u_max']['value'] <= 0.0,
            None if s is None else dict(n=a['n'], settling_max=a['settling']['value'],
                                        age_sup_max=a['age_sup_max'],
                                        n_bounds_failed=a['n_bounds_failed']))
    w = load('sweep.json')
    add('sweep.json', w is not None and w['admitted_rows_inside_bounds']
        and w['admitted_rows_all_checked']
        and w['admitted_rows_filter_inactive'] and w['admitted_rows_requirements_ok']
        and w['all_rows_commands_in_set'],
        None if w is None else dict(certified_ages_ms=w['certified_ages_ms'],
                                    boundary_ms=w['boundary_ms']['k_ms'],
                                    first_failures=w['first_failures']))
    e = load('ensemble.json')
    if e is not None:
        a = e['aggregate']['admitted']
        rn = e.get('rule_at_nominal_state')
        add('ensemble.json', (a is None or (a['bounds_all_ok'] and
                                            a['filter_modifications_total'] == 0
                                            and a['requirements_all_ok']
                                            and all(x['n_checked'] == x['n_rows']
                                                    for x in e['rows']
                                                    if x['certificate']['admitted'])))
            and (rn is None or rn['reproduces_design']),
            dict(n_admitted=e['aggregate']['n_admitted'],
                 failures=e['aggregate']['failures_among_rejected'],
                 rule_reproduces_design=None if rn is None else rn['reproduces_design']))
    x = load('edge.json')
    add('edge.json', x is not None and x['aggregate']['admitted_filter_inactive']
        and x['aggregate']['admitted_inside_bounds'] and x['aggregate']['admitted_all_checked']
        and x['aggregate']['admitted_requirements_ok'],
        None if x is None else x['aggregate'])
    x = load('ps1_states.json')
    add('ps1_states.json', x is not None and x['aggregate']['admitted_inside_bounds']
        and x['aggregate']['admitted_all_checked'] and x['aggregate']['admitted_filter_inactive'],
        None if x is None else x['aggregate'])
    g = load('sim_cont_regression.json')
    add('sim_cont_regression.json', g is not None
        and g['all_identical_except_runtime'] and g['all_legacy_bit_identical'],
        None if g is None else dict(sha256=g['sim_cont_sha256'][:16],
                                    legacy=g['legacy_bit_identical_to_sim_root2']))
    x = load('plan_exactness.json')
    if x is not None or load('design.json').get('family') == 'fan':
        a = None if x is None else x['aggregate']
        add('plan_exactness.json', a is not None and a['all_jumps_match']
            and a['max_speed_diff_to_plan'] <= 1e-8 and a['max_abs_deviation'] <= 1e-8
            and a['max_stop_diff_to_tau_star'] <= 1e-8 and a['filter_modifications'] == 0
            and (a['late_min_speed_diff_to_plan'] is None
                 or a['late_min_speed_diff_to_plan'] >= 100 * a['max_speed_diff_to_plan']),
            a)
    i = load('independent.json')
    add('independent.json', i is not None
        and i['aggregate']['same_step_max_stop_diff'] < 1e-9
        and i['aggregate']['same_step_stop_order_equal']
        and i['aggregate']['all_sim_cont_bounds_ok'],
        None if i is None else dict(
            same_step_max_stop_diff=i['aggregate']['same_step_max_stop_diff'],
            fine_vs_1ms_max_stop_diff=i['aggregate']['fine_vs_1ms_max_stop_diff']))
    dump('check.json', out)
    print('  ALL OK' if out['all_ok'] else '  FAILURES PRESENT', flush=True)


PARTS = dict(regress=part_regress, exact=part_exact, ref=part_ref, conv=part_conv,
             digital=part_digital, sched=part_sched, adv=part_adv,
             sweep=part_sweep, ens=part_ens, edge=part_edge, ps1=part_ps1,
             indep=part_indep, check=part_check)

if __name__ == '__main__':
    todo = sys.argv[1:] or ['all']
    if todo == ['all']:
        todo = ['regress', 'exact', 'ref', 'conv', 'digital', 'sched', 'adv',
                'sweep', 'ens', 'edge', 'ps1', 'indep', 'check']
    for p in todo:
        t0 = time.time()
        print('=== part: %s ===' % p, flush=True)
        PARTS[p]()
        print('[%s] done in %.1f s' % (p, time.time() - t0), flush=True)
