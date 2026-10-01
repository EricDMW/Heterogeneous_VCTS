#!/usr/bin/env python3
r"""run_fb_validate.py -- validation suite of the dispatch certificate of the
arrival with feedback-only followers and a shaped head reference (FB
design), 2026-09-29.  Adapted from run_hr_validate.py (the schedule version,
which stays unchanged with its records in data/new_d19/headref/).

Design (data/new_d19/fbref/design_fb.json, selected by run_fb_design.py)
  Only the head stores a reference: the shaped reference of the design
  record (exact pieces of a_r on [0, T_xi): -0.728 m/s^2 on [0, 3.25 s),
  then steps every 0.25 s up to the crawl rate -0.029953 m/s^2 from 15.75 s,
  T_xi = 148.75 s, handoff speed v_r(T_xi) = 1.000001 m/s), tracked by the
  head with S1/S2 and feedforward a_r, braking at a_b from T_xi.  Every
  follower stores constants only and regulates its clearance to its parking
  gap from takeover (COAST, S1, S2 of the frozen paper on e_i = c_i - d_s,i,
  eps_i = v_{i-1} - v_i, feedforward the received command); messages carry
  the applied command and the matching flag.  lam = 0.065 (beta = lam^2,
  gamma = 2 lam), U = 0.02, U^- = 1.2 m/s^2, handoff box (0.05 m, 0.005 m/s),
  parking gaps 6.5 m (head slot 6 m), closures c_i(0) - d_s,i = 43.5 m,
  certificate certify_fb.config() (V^hnd = 1e-2, lags (41, 41), one-sided
  switch box with the earliest S2 switch instants), age bound 20 ms with
  T_m = T_c = 1 ms and a constant transport of 19 ms for the reference run.
  (Admitted at 20 ms; age boundary 24 ms, C7a fails at 25 ms.)

FB law in the simulators.  Every run of the proposed controller uses
sim_hr.run(..., schedules=False, nosched='park') on sim_hr.hr_setup of the
shaped configuration: no schedule pieces (the setup has none), desired
clearance d_s,i from t = 0.  sim_hr still carries the CLOSE bit of the
schedule version in its packets and records the instants tcl; with no
schedule pieces (NS_i = 0) the bit starts no schedule clock and changes no
command.  The reference part checks on the record that e_i = c_i - d_s,i and
eps_i = v_{i-1} - v_i at every record instant (bit for bit) and that no
schedule jump exists; every other run checks the same (bool row fb_law).

Every part compares every observed quantity with its certified bound from
certify_fb.evaluate at the run's age bound (bounds_rows: one row per
quantity, per unit or pair where the certificate is per unit or pair, and a
pass/fail flag per row; a missing observation where a bound exists fails;
comparisons use an absolute tolerance TOL = 1e-12).  Rows of the schedule
version that concern only the schedules or the closure onsets are dropped;
rows added for FB: the one-sided switch box around e_i(0) (e^sw_i in
[e_i(0) - E^(1)_i, e_i(0)]), the first-window excursion |e_i - e_i(0)| <=
E^(1)_i, the earliest S2 switch instants tlow_i <= t^sw_i (FB refinement),
the activation after the predecessor's switch, and the FB-law row.

Parts (python run_fb_validate.py [part ...]; default: all in this order)
  ref          reference run 'cont' (RK4 between events) at h = 1e-4 s,
               constant transport 19 ms, filter on; every observation
               against its bound; records for the figures every 10 ms and
               1 ms (traj_fb.npz, traj_fb_1ms.npz) from this run; fine run at
               h = 2.5e-5 s (convergence and every step on [0, 4 s] and on
               [T_xi - 0.05 s, T_f + 0.005 s]: traj_fb_stop.npz)
               -> reference.json
  conv         'cont' at h = 1e-4, 5e-5, 2.5e-5 s (+ diagnostic 1e-3, 5e-4 s)
               -> convergence.json
  digital      'zoh' 1 kHz against 'cont' (h = 1e-4 s) on the constant
               transport and 5 random schedules -> digital.json
  sched        1000 random FIFO schedules (seed 500 + j, random_schedule of
               run_caseD_validate: per-hop delays 1..19 ms, segments 0.5 to
               5 s), 'zoh' 1 kHz -> schedules.json
  adv          constructed schedules: short ages while the planned
               deceleration falls (around the 51 reference steps, windows
               [t_J - dbar, t_J + (i-1) dbar + 2 ms] on hop i) and long ages
               while it is constant, and the reverse, per hop and globally;
               constant 1 and 19 ms; bursts; windows of maximal transport
               around the steps, the switches, the takeover and T_xi;
               sawtooth batches; alternating; per-hop constant -> adversarial.json
  sweep        age bounds 5 ms .. 1000 ms: certificate of the fixed design at
               every age, constant transport (age - 1 ms) and 20 random
               schedules per age ('zoh' 1 kHz) -> sweep.json
  ensemble     takeover states inside C2 (clearances c_i(0) in [47, 53] m on
               the 1 mm grid, head speed in [10.45, 10.55] m/s, speed
               increments as in the schedule version); for every state the
               dispatcher re-runs the design rule at the design gain
               (run_fb_design.shape(0.065) with the state's c0, v0, e0), then
               the certificate at 20 ms, then two runs ('zoh': constant 19 ms
               and one random schedule) -> ensemble.json
  outside      takeover states outside C2 (one follower slower than its
               predecessor by 0.01..0.10 m/s, one pair at a time), design
               reference, no certificate (C2 fails), runs ('zoh' constant 19
               and 1 ms, one random schedule; 'cont' constant 19 ms) with RULE
               entries, positive and clipped commands, filter action, errors
               -> outside.json
  independent  sim_hr ('zoh', 'cont') against sim_bench_hr (variants
               'proposed' and 'feedback', ideal plant, own kernel) on the
               reference case and 3 random schedules -> independent.json
  age_audit    held information age of every schedule of the suite,
               recomputed from the schedules alone and compared with the
               simulator's audit of every run -> age_audit.json
  check        every pass/fail flag of the files above -> check.json

Evidence classes: certificate (floating point, certify_fb), simulation
(sim_hr, sim_bench_hr), sampled search (random and constructed schedules,
sampled takeover states).  Every record states its class, the code SHA-256,
seeds, options and runtimes.

Outputs: data/new_d19/fbref/*.json, *.npz; log logs_ext/fbref/fb_validate.log.
No existing file is modified.
"""
import copy
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(CODE, 'src', 'simulation'),
           os.path.join(CODE, 'src', 'certification'), CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                                        # noqa: E402
import certify_fb as cf                                   # noqa: E402
import certify_plan as cq                                 # noqa: E402
import sim_hr as hr                                       # noqa: E402
import run_caseD_validate as rcv                          # noqa: E402
import run_fb_design as rfd                               # noqa: E402

OUT = cf.DATA_FB
LOGDIR = os.path.join(CODE, 'logs_ext', 'fbref')
LOG = os.path.join(LOGDIR, 'fb_validate.log')
DESIGN_FB = cf.DESIGN_FB

# ---- final design (design record)
_DREC = json.load(open(DESIGN_FB, encoding='utf-8'))
LAM = float(_DREC['selected']['lam'])
DBAR = float(_DREC['fixed_choices']['dbar_des'])
AGE_BOUNDARY_MS = int(_DREC['selected']['age_boundary']['k_ms'])
U_TR = 0.02                  # traction limit U
U_BR = 1.2                   # braking limit U_i^-
VHND = 0.01
LAGS = [41, 41]
# ---- channel and schemes
T_M = 1e-3
T_C = 1e-3
TRANSPORT = 0.019
T_TAIL = 8.0                 # simulated horizon T_END = T_xi + T_TAIL (sim_hr)
H_REF = 1e-4
H_FINE = 2.5e-5
H_CONV = (1e-4, 5e-5, 2.5e-5)
H_DIAG = (1e-3, 5e-4)
H_ZOH = 1e-3
REC_DT = 1e-3
TOL = 1e-12
SIM_LAW = dict(schedules=False, nosched='park')     # the FB law in sim_hr
START_WINDOW = (0.0, 4.0)    # fine record of the start (every step, h = 2.5e-5 s)
# ---- studies
N_SCHED = 1000
SEED_SCHED = 500
N_DIGITAL = 5
SEED_DIGITAL = 600
AGES_MS = (5, 10, 15, 20, AGE_BOUNDARY_MS, AGE_BOUNDARY_MS + 1, 30, 40, 50, 70, 100,
           150, 200, 300, 500, 1000)
N_SWEEP = 20
SEED_SWEEP = 700
N_ENS = int(os.environ.get('FBV_NENS', 300))
SEED_ENS = 7
SEED_ENS_SCHED = 800
SEED_OUT_SCHED = 900
INDEP_SEEDS = (500, 501, 502)
ENS_RANGES = OrderedDict(
    c0_mm=[47000, 53000],                  # c_i(0) on the 1 mm grid [mm]
    v1=[10.45, 10.55],
    increments=[[0.02, 0.13], [0.035, 0.17], [0.05, 0.23], [0.065, 0.30]])
NPROC = max(1, min(10, int(os.environ.get('FBV_NPROC', (os.cpu_count() or 2) - 2))))

KPI_KEYS = ('tstop', 'settling', 'T_f', 'marker_signed', 'gap_signed', 'gmin',
            'hmin', 'mmin_lim', 'u_min_unit', 'u_max_unit', 'u_max_pre_unit',
            'u_min_pre_unit', 'entry_obs', 'e_hnd', 'eps_hnd', 'e_sw', 'eps_sw',
            'wint', 'epsmax_b', 'fb_unit', 'tset', 'tact', 't_rmp')

EVIDENCE = dict(
    reference='simulation (sim_hr, continuous local law, nominal plant) against the '
              'floating-point certificate of certify_fb at 20 ms',
    convergence='simulation (numerical convergence of sim_hr in the RK4 step)',
    digital='simulation (sim_hr 1 kHz digital law against the continuous law)',
    schedules='sampled search (1000 random FIFO schedules) + simulation, against the '
              'certificate at 20 ms',
    adversarial='sampled search (constructed FIFO schedules) + simulation, against the '
                'certificate at 20 ms',
    sweep='certificate (floating point, every age bound) + simulation (constant and random '
          'schedules per age)',
    ensemble='sampled search (random takeover states inside C2) + design rule + certificate '
             '(floating point) + simulation',
    outside='simulation (takeover states outside C2; no certificate applies)',
    independent='simulation (two independent implementations of the FB closed loop)',
    age_audit='recomputation from the schedules (integer arithmetic) against the '
              "simulator's audit",
    check='summary of the pass/fail flags of the records')


# ============================================================== helpers
def log(msg=''):
    print(msg, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def _py(o):
    if isinstance(o, dict):
        return {str(k): _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(x) for x in o]
    if isinstance(o, np.ndarray):
        return _py(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, Fr):
        return float(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        return x if np.isfinite(x) else None
    return o


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    tmp = p + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(_py(obj), fh, indent=1)
    os.replace(tmp, p)
    log('  -> data/new_d19/fbref/%s' % name)
    return p


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def code_hashes():
    import sim_cont as sc
    files = OrderedDict(
        run_fb_validate=os.path.abspath(__file__), certify_fb=cf.__file__,
        certify_plan=cq.__file__, certify_profile=os.path.join(
            CODE, 'src', 'certification', 'certify_profile.py'),
        sim_hr=hr.__file__, sim_cont=sc.__file__,
        sim_bench_hr=os.path.join(CODE, 'src', 'simulation', 'sim_bench_hr.py'),
        run_fb_design=rfd.__file__, run_caseD_validate=rcv.__file__,
        design_fb_json=DESIGN_FB)
    return OrderedDict((k, sha256(p)) for k, p in files.items())


def meta(part, **kw):
    """provenance of a record: evidence class, code hashes, versions,
    options."""
    try:
        import numba
        nb = numba.__version__
    except Exception:                                        # pragma: no cover
        nb = None
    H = code_hashes()
    D = _DREC['code_sha256']
    out = OrderedDict(
        part=part, evidence_class=EVIDENCE.get(part), generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        script='code/run_fb_validate.py', template='code/run_hr_validate.py (not modified)',
        code_sha256=H,
        evaluator_matches_design_record=bool(H['certify_fb'] == D['certify_fb']
                                             and H['certify_plan'] == D['certify_plan']),
        design_rule_matches_design_record=bool(H['run_fb_design'] == D['run_fb_design']),
        python=sys.version.split()[0], numpy=np.__version__, numba=nb,
        platform=platform.platform(), processes=NPROC,
        design=OrderedDict(lam=LAM, U=U_TR, Uminus=U_BR, vhnd=VHND, lags=LAGS, dbar=DBAR,
                           age_boundary_ms=AGE_BOUNDARY_MS,
                           T_xi=_DREC['selected']['T_xi'],
                           plan_T_xi_exact=_DREC['selected']['plan']['T_xi'],
                           n_pieces=len(_DREC['selected']['plan']['pieces'])),
        sim_law=dict(SIM_LAW, T_m=T_M, T_c=T_C))
    out.update(kw)
    return out


def _f(x):
    """float, or None for None / NaN / inf / non-numbers."""
    if x is None:
        return None
    try:
        y = float(x)
    except (TypeError, ValueError):
        return None
    return y if np.isfinite(y) else None


def _arr(x):
    """float array, NaN for None."""
    if x is None:
        return np.array([np.nan])
    return np.array([np.nan if v is None else float(v)
                     for v in np.ravel(np.asarray(x, dtype=object))], float)


def _fin(xs):
    return [float(x) for x in xs if x is not None and np.isfinite(x)]


def _max(xs):
    f = _fin(xs)
    return max(f) if f else None


def _min(xs):
    f = _fin(xs)
    return min(f) if f else None


def _rng(xs):
    f = _fin(xs)
    return [min(f), max(f)] if f else None


def _mx(a, w):
    return float(np.max(a[w])) if np.any(w) else None


def _mn(a, w):
    return float(np.min(a[w])) if np.any(w) else None


def _nanmn(a, w):
    if not np.any(w):
        return None
    x = a[w]
    x = x[np.isfinite(x)]
    return float(np.min(x)) if x.size else None


def arg_ext(rows, key, fn=max):
    vals = [(r[key], nm) for nm, r in rows if r.get(key) is not None
            and np.isfinite(r[key])]
    if not vals:
        return None, None
    v = fn(vals, key=lambda z: z[0])
    return v[0], v[1]


# ============================================================== design
_CACHE = {}


def cert_cfg(dbar=DBAR, **kw):
    """certificate configuration of the FB design (certify_fb.config: the
    selected design of design_fb.json) at the age bound dbar; entries of kw
    are passed to config (plan entries merged, e0 recomputed from c0 unless
    given)."""
    return cf.config(**dict(kw, dbar=float(dbar)))


def sim_cfg(**kw):
    """configuration of the simulator: the same dictionary (sim_hr.hr_setup
    reads the shaped plan, c0, v0, d_s, gains and detector)."""
    return cert_cfg(**kw)


def setup_of(cfg):
    return hr.hr_setup(cfg, U=U_TR, Uminus=U_BR)


def design_setup():
    if 'S' not in _CACHE:
        _CACHE['S'] = setup_of(sim_cfg())
    return _CACHE['S']


def check_design_constants(cfg):
    """the configuration of certify_fb.config against the constants of this
    driver and the design record."""
    sel = _DREC['selected']
    out = OrderedDict(
        lam=cfg['lam'] == LAM, U=float(cfg['U']) == U_TR, Uminus=float(cfg['Uminus']) == U_BR,
        vhnd=float(cfg['vhnd']) == VHND, lags=list(cfg['lags']) == LAGS,
        plan=cfg['plan'] == sel['plan'], one_sided=bool(cfg['one_sided']),
        switch_lower=bool(cfg['switch_lower']),
        d_s=[float(x) for x in cfg['d_s']] == [6.0, 6.5, 6.5, 6.5, 6.5])
    out['all'] = all(out.values())
    return out


def n_gen(S):
    """length of a generated schedule (as run_caseD_validate: T_END/T_m + 2)."""
    return int(round(S.T_END / T_M)) + 2


def simulate(S, scheme='zoh', h=None, transport=TRANSPORT, fifo=None,
             rec_dt=REC_DT, T_end=None, **kw):
    """one sim_hr run of the FB law: 'zoh' is the pure 1 kHz implementation
    (T_u = h = T_m = T_c = 1 ms), 'cont' the continuous local law at step h;
    no schedules, desired clearance d_s from takeover (nosched 'park')."""
    if h is None:
        h = H_ZOH if scheme == 'zoh' else H_REF
    T_u = T_M if scheme == 'zoh' else None
    t0 = time.time()
    R = hr.run(S, scheme=scheme, h=h, T_m=T_M, T_c=T_C, T_u=T_u,
               transport=transport, fifo=fifo, rec_dt=rec_dt, T_end=T_end,
               **dict(SIM_LAW, **kw))
    R['runtime_s'] = time.time() - t0
    return R


def ref_jumps(S):
    """jumps (t_J, h_J) of the head reference a_r on (0, T_xi) (exact
    pieces of the shaped plan, as floats)."""
    out = []
    H = S.head_exact
    for j in range(1, len(H)):
        if H[j][0] >= S.T_xi_exact:
            break
        h_ = H[j][1] - H[j - 1][1]
        if h_ != 0:
            out.append((float(H[j][0]), float(h_)))
    return out


# ======================================================== trajectories
def mode_codes(t, R, n):
    """mode of every unit at the record instants (post-event values), from
    the event instants of the run: COAST, S1 from t^act, S2 from t^sw, BRAKE
    (head) from T_xi, RULE from t^rule, LATCH from the stop."""
    T_xi = float(R['T_xi'])
    M = np.zeros((n, t.size), np.int8)
    for i in range(n):
        m = np.full(t.size, hr.COAST, np.int8)
        ta, tw, tr, tz = (float(R[key][i]) for key in ('tact', 'tset', 'trule', 'tstop'))
        if np.isfinite(ta):
            m[t >= ta] = hr.S1
        if np.isfinite(tw):
            m[t >= tw] = hr.S2
        if i == 0:
            m[t >= T_xi] = hr.BRAKE
        if np.isfinite(tr):
            m[t >= tr] = hr.RULE
        if np.isfinite(tz):
            m[t >= tz] = hr.LATCH
        M[i] = m
    return M


def traj(R, S, sel=None):
    """record of a run (every record instant, or the indices sel): states,
    commands, pair errors (FB: e_i = c_i - d_s,i, eps_i = v_{i-1} - v_i; row
    0 the head against its reference), clearances, barriers, margins, modes,
    the head reference, the desired clearances d_i = d_s,i (constant), dd_i =
    rho_i = 0, and the realization plan of the shaped plan (every unit plans
    the head's accelerations with the parking gaps: pa_i = a_r, pv_i = v_r,
    ps_i = ps_{i-1} - ell - d_s,i)."""
    rec = R['rec']
    tt = np.asarray(rec['t'], float)
    if sel is None:
        sel = np.nonzero(np.isfinite(tt))[0]
    t = tt[sel]
    n = S.n
    T = OrderedDict(t=t)
    for key in ('s', 'v', 'u', 'e', 'eps'):
        T[key] = np.asarray(rec[key])[:, sel]
    T['g'] = np.asarray(rec['g'])[:, sel]
    T['c'] = T['g'] + S.s_m
    T['h'] = np.asarray(rec['h'])[:, sel]
    T['m'] = np.asarray(rec['m'])[:, sel]
    T['modes'] = mode_codes(t, R, n)
    s0r, vr, ar = S.ref(t)
    T['s0'], T['vr'], T['ar'] = s0r, vr, ar
    T['d'] = np.repeat(np.asarray(S.d_s[1:], float)[:, None], t.size, axis=1)
    T['dd'] = np.zeros((n - 1, t.size))
    T['rho'] = np.zeros((n - 1, t.size))
    pa = np.empty((n, t.size))
    pv = np.empty_like(pa)
    ps = np.empty_like(pa)
    pa[0] = ar
    pv[0] = vr
    ps[0] = s0r - (S.d_s[0] + S.ell)
    for i in range(1, n):
        pa[i] = ar
        pv[i] = vr
        ps[i] = ps[i - 1] - S.ell - S.d_s[i]
    T['ps'], T['pv'], T['pa'] = ps, pv, pa
    T['dev'] = T['u'] - pa
    T['T_xi'] = float(R['T_xi'])
    T['rec_dt'] = float(t[1] - t[0]) if t.size > 1 else None
    # recorded errors against the parking gaps, on the samples before the
    # early exit of the kernel (after it sim_hr continues the record with the
    # final positions and the last recorded errors)
    gen = t < float(R['t_end_sim']) - (T['rec_dt'] or 0.0) - 1e-9
    T['gen'] = gen
    ec = np.empty((n, t.size))
    xc = np.empty((n, t.size))
    ec[0] = s0r - T['s'][0] - S.ell - S.d_s[0]
    xc[0] = vr - T['v'][0]
    for i in range(1, n):
        ec[i] = T['s'][i - 1] - T['s'][i] - S.ell - S.d_s[i]
        xc[i] = T['v'][i - 1] - T['v'][i]
    T['e_consistency'] = float(np.nanmax(np.abs(ec - T['e'])[:, gen])) if np.any(gen) else None
    T['e_consistency_followers'] = float(np.nanmax(np.abs(ec - T['e'])[1:, gen])) \
        if np.any(gen) else None
    T['eps_consistency_followers'] = float(np.nanmax(np.abs(xc - T['eps'])[1:, gen])) \
        if np.any(gen) else None
    return T


def fb_law_check(R, S, T):
    """the FB law in a run: no schedule pieces in the setup, schedules off
    and nosched 'park' in the run, no schedule jump, desired clearance d_s,i
    from t = 0: e_i = s_{i-1} - s_i - ell - d_s,i and eps_i = v_{i-1} - v_i
    bit for bit at every record instant up to the early exit (followers),
    and e_i(0) = c_i(0) - d_s,i."""
    cfgr = R['config']
    e0_rec = [float(x) for x in np.asarray(T['e'])[:, 0]]
    e0_exp = [0.0] + [float(S.c0[i - 1] - S.d_s[i]) for i in range(1, S.n)]
    out = OrderedDict(
        setup_has_schedules=S.sched is not None,
        run_schedules=bool(cfgr['schedules']), run_nosched=cfgr['nosched'],
        schedule_jumps=sum(len(v) for v in R['schedule_jumps_abs'].values()),
        d_pre_parking=bool(np.array_equal(np.asarray(S.d_pre[1:], float),
                                          np.asarray(S.d_s[1:], float))),
        e_minus_parking_gap_max_abs=T['e_consistency_followers'],
        eps_minus_raw_max_abs=T['eps_consistency_followers'],
        head_error_consistency=T['e_consistency'],
        e_at_0=e0_rec, e0_expected=e0_exp,
        e_at_0_ok=bool(max(abs(a - b) for a, b in zip(e0_rec, e0_exp)) <= TOL),
        closure_bit=('CLOSE relayed by the simulator (instants tcl recorded), inert: no '
                     'schedule pieces, so it starts no schedule clock'))
    out['ok'] = bool(not out['setup_has_schedules'] and not out['run_schedules']
                     and out['run_nosched'] == 'park' and out['schedule_jumps'] == 0
                     and out['d_pre_parking'] and out['e_minus_parking_gap_max_abs'] == 0.0
                     and out['eps_minus_raw_max_abs'] == 0.0 and out['e_at_0_ok'])
    return out


def record_obs(T, R, S):
    """observations on the record T (1 ms): phase windows by the run's event
    instants."""
    t = T['t']
    n = S.n
    T_xi = float(R['T_xi'])
    a_b = float(S.a_b)
    tol = 1e-9
    ts = _arr(R['tset'])
    tz = _arr(R['tstop'])
    tr = _arr(R['t_rmp'])
    tz_i = np.where(np.isfinite(tz), tz, np.inf)
    tr_i = np.where(np.isfinite(tr), tr, np.inf)
    pre = t < T_xi - tol
    V, E, X, Uc, Hb, M = T['v'], T['e'], T['eps'], T['u'], T['h'], T['m']
    o = OrderedDict()
    o['rec_dt'] = T['rec_dt']
    o['vmin_pre'] = [_mn(V[i], pre) for i in range(n)]
    o['vmin_pre_min'] = _min(o['vmin_pre'])
    jx = np.nonzero(np.abs(t - T_xi) < tol)[0]
    o['v_at_T_xi'] = [float(x) for x in V[:, jx[0]]] if jx.size else [None] * n
    w1 = [(t <= ts[i] + tol) if np.isfinite(ts[i]) else pre for i in range(n)]
    w2 = [((t >= ts[i] - tol) & pre) if np.isfinite(ts[i]) else np.zeros(t.size, bool)
          for i in range(n)]
    e_start = E[:, 0]
    o['e_start'] = [float(x) for x in e_start]
    o['e_first_dev_max'] = [_mx(np.abs(E[i] - e_start[i]), w1[i]) for i in range(n)]
    o['e_first_max'] = [_mx(E[i], w1[i]) for i in range(n)]
    o['e_first_min'] = [_mn(E[i], w1[i]) for i in range(n)]
    o['e_first_abs_max'] = [_mx(np.abs(E[i]), w1[i]) for i in range(n)]
    o['eps_first_abs_max'] = [_mx(np.abs(X[i]), w1[i]) for i in range(n)]
    o['eps_first_max'] = [_mx(X[i], w1[i]) for i in range(n)]
    o['e_s2_abs_max'] = [_mx(np.abs(E[i]), w2[i]) for i in range(n)]
    o['eps_s2_abs_max'] = [_mx(np.abs(X[i]), w2[i]) for i in range(n)]
    # signed extremes of the pair errors before T_xi (the window of the
    # planned pulses of the reference steps)
    wc = pre
    o['e_closure_hi'] = [_mx(E[i], wc) for i in range(n)]
    o['e_closure_lo'] = [_mn(E[i], wc) for i in range(n)]
    o['eps_closure_hi'] = [_mx(X[i], wc) for i in range(n)]
    o['eps_closure_lo'] = [_mn(X[i], wc) for i in range(n)]
    wb = t >= T_xi - tol
    o['h_acq_min'] = [_mn(Hb[c - 1], w1[c]) for c in range(1, n)]
    o['h_s2_min'] = [_mn(Hb[c - 1], w2[c]) for c in range(1, n)]
    o['h_brk_min'] = [_mn(Hb[c - 1], wb) for c in range(1, n)]
    o['m_acq_min'] = [_nanmn(M[c - 1], w1[c]) for c in range(1, n)]
    o['m_s2_min'] = [_nanmn(M[c - 1], w2[c]) for c in range(1, n)]
    o['m_brk_min'] = [_nanmn(M[c - 1], wb) for c in range(1, n)]
    pmx, pmn, fbr, umxa, umna = [None], [None], [], [], []
    for i in range(n):
        wq = (t >= tr_i[i] - tol) & (t < tz_i[i] - tol)
        fbr.append(_mx(np.abs(Uc[i] + a_b), wq))
        wa = (t >= T_xi - tol) & (t < tz_i[i] - tol)
        umxa.append(_mx(Uc[i], wa))
        umna.append(_mn(Uc[i], wa))
        if i >= 1:
            wp = (t >= T_xi - tol) & (t < tr_i[i] - tol) & (t < tz_i[i] - tol)
            pmx.append(_mx(Uc[i], wp))
            pmn.append(_mn(Uc[i], wp))
    o['u_prerec_max'] = pmx
    o['u_prerec_min'] = pmn
    o['fb_post_rec'] = fbr
    o['u_max_post_unit'] = umxa
    o['u_min_post_unit'] = umna
    eb, xb = [], []
    for c in range(1, n):
        w = (t >= T_xi - tol) & (t < min(tz_i[c - 1], tz_i[c]) - tol)
        eb.append(_mx(np.abs(E[c]), w))
        xb.append(_mx(np.abs(X[c]), w))
    o['ebmax_b'] = eb
    o['epsbmax_rec'] = xb
    ent = []
    for i in range(n):
        out = ((np.abs(E[i]) > S.eps_e) | (np.abs(X[i]) > S.eps_v)) & pre
        idx = np.nonzero(out)[0]
        if idx.size == 0:
            ent.append(0.0)
        else:
            j = int(idx[-1])
            ent.append(float(t[j + 1]) if (j + 1 < t.size and pre[j + 1]) else None)
    o['entry_units'] = ent
    dv = np.diff(V, axis=1)
    inc, incm = [], []
    for i in range(n):
        inc.append(float(np.max(dv[i])))
        mv = t[1:] < tz_i[i] - tol
        incm.append(float(np.max(dv[i][mv])) if np.any(mv) else None)
    o['speed_largest_increment_unit'] = inc
    o['speed_largest_increment_while_moving_unit'] = incm
    npos, upos = [], []
    for i in range(n):
        wl = t < tz_i[i] - tol
        x = Uc[i][wl]
        npos.append(int(np.sum(x > 0.0)))
        upos.append(float(np.max(x)) if np.any(x > 0.0) else None)
    o['n_pos_rec_unit'] = npos
    o['n_pos_rec'] = int(sum(npos))
    o['u_pos_max_rec'] = _max(upos)
    o['e_consistency'] = T['e_consistency']
    o['fb_law'] = fb_law_check(R, S, T)
    return o


# ================================================================ kpis
def kpis(R, S, T=None):
    """observations of one sim_hr run (key names of run_hr_validate.kpis;
    settling = stopping-time dispersion)."""
    n = S.n
    tstop = _arr(R['tstop'])
    all_st = bool(R['all_stopped'])
    order = R['stop_order']
    tau_r = float(R['tau_r'])
    k = OrderedDict()
    k['tstop'] = tstop
    k['settling'] = _f(R['disp']) if all_st else None
    k['T_f'] = _f(R['T_f']) if all_st else None
    k['first_stop'] = _f(np.nanmin(tstop)) if np.any(np.isfinite(tstop)) else None
    k['stop_order'] = order
    k['head_first_tail_last'] = bool(order is not None and list(order) == list(range(1, n + 1)))
    k['tail_first_head_last'] = bool(order is not None and order[0] == n and order[-1] == 1)
    k['stop_increments'] = np.abs(np.diff(tstop))
    k['stop_offsets'] = tstop - tau_r
    k['sched'] = _f(np.max(np.abs(tstop - tau_r))) if all_st else None
    k['marker_signed'] = np.asarray(R['marker_signed'], float) if all_st else np.full(n, np.nan)
    k['marker_abs_max'] = _f(R['align']) if all_st else None
    k['gap_signed'] = np.asarray(R['gap_signed'], float) if all_st else np.full(n - 1, np.nan)
    k['gap_abs_max'] = _f(R['gaperr']) if all_st else None
    k['head_slot_err'] = _f(R['head_slot_err']) if all_st else None
    k['gmin'] = np.asarray(R['gmin'], float)
    k['gmin_min'] = float(np.min(R['gmin']))
    k['gmin_grid'] = np.asarray(R['gmin_grid'], float)
    k['hmin'] = np.asarray(R['hmin'], float)
    k['hmin_min'] = float(np.min(R['hmin']))
    k['mmin'] = np.asarray(R['mmin'], float)
    k['mmin_lim'] = np.asarray(R['mmin_lim'], float)
    k['mmin_lim_min'] = float(np.min(R['mmin_lim']))
    umn = _arr(R['u_min'])
    umx = _arr(R['u_max'])
    k['u_min_unit'] = umn
    k['u_max_unit'] = umx
    k['u_min'] = _f(np.nanmin(umn))
    k['u_max'] = _f(np.nanmax(umx))
    k['u_min_pre_unit'] = _arr(R['u_min_pre'])
    k['u_max_pre_unit'] = _arr(R['u_max_pre'])
    k['u_min_pre'] = _f(np.nanmin(k['u_min_pre_unit']))
    k['u_max_pre'] = _f(np.nanmax(k['u_max_pre_unit']))
    k['u_max_pre_informed_unit'] = _arr(R['u_max_pre_informed'])
    k['u_max_pre_informed'] = _f(R['u_max_pre_informed_all'])
    k['unom_max_unit'] = _arr(R['unom_max'])
    k['unom_max_pre_unit'] = _arr(R['unom_max_pre'])
    k['u_abs_after_latch'] = float(np.max(R['u_abs_latched']))
    Umin = np.asarray(S.Uminus, float)
    k['commands_in_set'] = bool(np.all(np.nan_to_num(umn, nan=0.0) >= -Umin - TOL)
                                and np.all(np.nan_to_num(umx, nan=0.0) <= S.U + TOL))
    k['zero_after_latch'] = bool(k['u_abs_after_latch'] == 0.0)
    k['n_pos_nom'] = int(R['n_pos_nom'])
    k['t_pos_nom'] = float(R['t_pos_nom'])
    for key in ('n_mod', 'n_fallback', 'n_sat', 'n_sat_hi', 'n_sat_lo'):
        k[key] = int(R[key])
    for key in ('t_mod', 't_fallback', 't_sat'):
        k[key] = float(R[key])
    k['filter_modifications'] = int(R['filter_modifications'])
    k['filter_inactive'] = bool(k['filter_modifications'] == 0 and k['mmin_lim_min'] > 0.0)
    k['counts'] = R['counts']
    k['time_spent'] = R['time_spent']
    for key in ('tact', 'tset', 'tcl', 'trule', 't_rmp', 'e_sw', 'eps_sw',
                'e_hnd', 'eps_hnd'):
        k[key] = _arr(R[key])
    k['rule_entered'] = bool(R['rule_entered'])
    k['rule_units'] = [i + 1 for i in range(n) if np.isfinite(k['trule'][i])]
    k['rule_before_T_xi'] = [i + 1 for i in range(n) if np.isfinite(k['trule'][i])
                             and k['trule'][i] < float(R['T_xi'])]
    k['wint'] = np.asarray(R['wint'], float)
    k['epsmax_b'] = np.asarray(R['epsmax_b'], float)
    k['fb_real'] = np.asarray(R['fb_real'], float)
    for key in ('entry_obs', 'entry_obs_followers', 'entry_obs_vel_only',
                'entry_obs_pos_only'):
        k[key] = _f(R[key])
    k['box_last_out'] = R['box_last_out']
    age = R['age']
    k['age'] = dict(max_left_limit=float(age['max_left_limit']),
                    max_at_delivery=float(age['max_at_delivery']),
                    max_left_limit_with_notional=float(age['max_left_limit_with_notional']),
                    first_delivery_max=max(float(o['first_delivery']) for o in age['per_hop']))
    for key in ('hits_true', 'n_eval', 'n_left_limits', 'n_localized', 'neg_speed'):
        k[key] = int(R[key])
    k['root_resid'] = float(R['root_resid'])
    k['all_stopped'] = all_st
    k['contact_time'] = R['contact_time']
    k['runtime_s'] = R.get('runtime_s')
    k['config'] = R['config']
    fbk = list(k['fb_real']) + [np.nan]
    if T is not None:
        k.update(record_obs(T, R, S))
        k['fb_unit'] = [_max([fbk[i], k['fb_post_rec'][i]]) for i in range(n)]
    else:
        k['fb_unit'] = [_f(x) for x in fbk]
    return k


def kpi_identical(a, b, keys=('tstop', 'settling', 'marker_signed', 'gap_signed',
                              'tset', 'tact', 't_rmp', 'e_sw', 'eps_sw',
                              'e_hnd', 'eps_hnd', 'u_min_unit', 'u_max_unit',
                              'mmin_lim', 'gmin', 'hmin', 'wint', 'entry_obs')):
    bad = []
    for key in keys:
        x = np.atleast_1d(_arr(a[key]))
        y = np.atleast_1d(_arr(b[key]))
        if x.shape != y.shape or not np.array_equal(x, y, equal_nan=True):
            bad.append(key)
    return dict(identical=not bad, differing=bad, keys=list(keys))


# ========================================================= certificate
PAIR_KEYS = ('sigma', 'slack_a', 'slack_b', 'slack_c', 'hfloor_acq', 'hfloor_s2',
             'h_b', 'gmin', 'gmin_direct', 'gmin_barrier', 'h0', 'h_tilde0',
             'H1', 'H2', 'Hb', 'hm1', 'hm2', 'hmb', 'mu1', 'mu2', 'e2bar')
CONST_KEYS = ('a_crawl', 'v_xi', 'v_in', 't_0', 'T_xi', 'a_r0', 'f_box', 'U',
              'Umin', 'eps0', 'e0', 'Lam1', 'Lam2', 'A1', 'A2', 'P_e', 'P_eps', 'P_f',
              'taub', 'tact', 'tset', 'E1', 'ebar', 'x2tot', 'e2bar', 'fbar', 'F', 'W',
              'Wphi', 'epsb', 'eb', 'fb', 'Fb_units', 'anet', 'q', 'T_brk', 'Qm', 'P_pre',
              'Q_plus', 'Q_minus', 'Tent_abs', 'T_ent_bar', 'Vhnd', 'Vhnd_max',
              'F_plus_0', 'F_minus_0', 'tlow_sw', 'switch_lower', 'NUMPAD', 'VHND_PAD',
              'cell_dt')


def cert_pack(ev, cfg, curves=True):
    """compact certificate of one evaluation (conditions, bounds, the
    constants used by the comparisons and, with curves, the cell envelopes
    of the commands and of the velocity floor before T_xi)."""
    out = OrderedDict(
        admitted=bool(ev['admitted']), failed=list(ev['failed']),
        closed=bool(ev.get('closed', False)), dbar=float(ev['dbar']),
        conds=OrderedDict((nm, dict(ok=bool(c['ok']), slack=_f(c['slack'])))
                          for nm, c in ev['conds'].items()),
        eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
        eps_det=float(cfg['eps_det']), a_b=float(cfg['a_b']),
        sync_tol=float(cfg['sync_tol']), align_tol=float(cfg['align_tol']),
        gap_tol=float(cfg['gap_tol']), U=float(cfg['U']),
        Uminus=float(cfg['Uminus']), lam=float(cfg['lam']),
        vhnd=float(cfg['vhnd']), lags=list(cfg.get('lags', [15, 20])),
        one_sided=bool(cfg.get('one_sided', True)),
        switch_lower=bool(cfg.get('switch_lower', False)))
    if ev.get('bounds') is None:
        out['bounds'] = None
        out['consts'] = None
        return out
    K = ev['consts']
    out['bounds'] = _py(ev['bounds'])
    cs = OrderedDict((key, _py(K[key])) for key in CONST_KEYS if key in K)
    cs['pairs'] = OrderedDict((str(c), OrderedDict((key, _f(p[key])) for key in PAIR_KEYS))
                              for c, p in K['pairs'].items())
    out['consts'] = cs
    if curves and ev.get('curves') is not None:
        cv = ev['curves']
        Ek = np.asarray(cv['Ek'], float)
        flr = np.asarray(cv['vrmin'], float)[None, :] - np.cumsum(Ek, axis=0)
        out['curves'] = dict(tg=np.asarray(cv['tg'], float),
                             up=np.asarray(cv['up'], float),
                             lo=np.asarray(cv['lo'], float), flr=flr,
                             Dg=float(cfg['cell_dt']))
    return out


def cert_summary(cp):
    """certificate summary (key names of run_hr_validate: settling_bound =
    dispersion bound, align = marker bound, gaperr = gap bounds, T_ent_bar =
    entry deadline, tset / tact = switch / activation deadlines) plus the FB
    earliest switch instants tlow_sw."""
    out = OrderedDict(admitted=cp['admitted'], failed=cp['failed'],
                      closed=cp['closed'], dbar=cp['dbar'],
                      conds=cp['conds'])
    B = cp.get('bounds')
    K = cp.get('consts')
    if B is None:
        return out
    out.update(settling_bound=B['dispersion'], q=B['q'], T_f_bar=B['T_f_bar'],
               sched=B['schedule_error'], align=B['marker_max'],
               marker_units=B['marker'], gaperr=B['gap'],
               gaperr_max=B['gap_max'], T_ent_bar=B['entry_deadline'],
               T_hold=B['T_hold'], tset=B['switch_deadline'],
               tact=B['activation_deadline'], tlow_sw=K.get('tlow_sw'),
               E1=K['E1'], e0=K.get('e0'), W=K['W'],
               epsb=B['epsb'], eb=B['eb'], Fb_units=K['Fb_units'],
               vfloor_min=B['velocity_floor'], v_c1=K['v_xi'],
               tau_star=K['T_xi'] + K['v_xi'] / cp['a_b'],
               clearance_floor=B['clearance_floor'],
               clearance_floor_min=B['clearance_floor_min'],
               sigma=B['admission_slack_sigma'],
               u_max_before_T_xi=B['u_max_before_T_xi'],
               u_min_before_T_xi=B['u_min_before_T_xi'],
               u_max_pre_receipt=B['u_max_pre_receipt'],
               u_min_pre_receipt=B['u_min_pre_receipt'],
               u_band_post_receipt=B['u_band_post_receipt'],
               ramp_receipt_bound=B['ramp_receipt_bound'],
               handoff_floor=B['handoff_floor'], Tent_abs=K['Tent_abs'],
               Vhnd_max=K['Vhnd_max'], T_xi=K['T_xi'], a_r0=K['a_r0'],
               a_crawl=K['a_crawl'])
    return out


def disp_expression(cp, n):
    """dispersion bound sum_k q_k with q_k = eps-bar_{b,k}/(a_b - F^b_{k-1})
    and its explicit parts: (n-1) eps_v/a_b (box) and (n-1) dbar (a_b -
    a_crawl)/a_b (age)."""
    B = cp.get('bounds')
    if B is None:
        return None
    K = cp['consts']
    a_b = cp['a_b']
    age = (n - 1) * cp['dbar'] * (a_b - K['a_crawl']) / a_b
    box = (n - 1) * cp['eps_v'] / a_b
    return OrderedDict(
        expression='sum_k q_k, q_k = eps-bar_{b,k}/(a_b - F^b_{k-1}) >= '
                   'eps_v/a_b + dbar (a_b - a_crawl)/a_b',
        bound=B['dispersion'], q=B['q'], box_term=box, age_term=age,
        feedback_remainder=B['dispersion'] - box - age,
        age_term_per_hop=cp['dbar'] * (a_b - K['a_crawl']) / a_b)


# ========================================================== bound rows
def _putter(rows):
    def one(x, y, kind, optional):
        if x is None:
            return None
        if y is None:
            return None if optional else False
        return bool(y <= x + TOL) if kind == 'upper' else bool(y >= x - TOL)

    def put(name, bound, obs, kind, unit, note=None, optional=False):
        if isinstance(bound, (list, tuple, np.ndarray)):
            bb = [_f(x) for x in bound]
            oo = [None] * len(bb) if obs is None else [_f(x) for x in obs]
            assert len(bb) == len(oo), name
            oks = [one(x, y, kind, optional) for x, y in zip(bb, oo)]
            chk = [z for z in oks if z is not None]
            rows[name] = dict(bound=bb, observed=oo, kind=kind, unit=unit,
                              ok=None if not chk else all(chk), ok_each=oks)
        else:
            x = _f(bound)
            y = _f(obs)
            rows[name] = dict(bound=x, observed=y, kind=kind, unit=unit,
                              ok=one(x, y, kind, optional))
        if note:
            rows[name]['note'] = note
    return put


def pointwise(cp, T):
    """largest excess of the record over the cell envelopes before T_xi:
    u_i(t) - (P^+_i + F^+_i)(cell), (P^-_i - F^-_i)(cell) - u_i(t), and the
    velocity floor flr_i(cell) - v_i(t) (<= 0 required)."""
    cv = cp['curves']
    n = len(cv['up'])
    t = T['t']
    pre = t < cp['consts']['T_xi'] - 1e-9
    tp = t[pre]
    ng = len(cv['tg'])
    g = np.clip(np.floor(tp / cv['Dg'] + 1e-9).astype(np.int64), 0, ng - 1)
    Uc = T['u'][:, pre]
    V = T['v'][:, pre]
    out = dict(up=[], up_at=[], lo=[], lo_at=[], vf=[], vf_at=[])
    for i in range(n):
        for key, a in (('up', Uc[i] - cv['up'][i][g]), ('lo', cv['lo'][i][g] - Uc[i]),
                       ('vf', cv['flr'][i][g] - V[i])):
            j = int(np.argmax(a))
            out[key].append(float(a[j]))
            out[key + '_at'].append(float(tp[j]))
    out['record_dt'] = T['rec_dt']
    out['last_sample_before_T_xi'] = float(tp[-1])
    return out


def box_hold(T, T_ent, T_xi):
    t = T['t']
    w = (t >= T_ent - 1e-9) & (t < T_xi - 1e-9)
    if not np.any(w):
        return dict(e=None, eps=None)
    return dict(e=float(np.max(np.abs(T['e'][:, w]))),
                eps=float(np.max(np.abs(T['eps'][:, w]))))


def bounds_rows(cp, k, T=None):
    """every observation of the run (kpis k, record T) against its certified
    bound (certificate cp at the run's age bound)."""
    B = cp['bounds']
    K = cp['consts']
    n = len(K['tset'])
    rows = OrderedDict()
    put = _putter(rows)
    P = [K['pairs'][str(c + 1)] for c in range(1, n)]
    dbar = cp['dbar']
    T_xi = K['T_xi']
    eps_e, eps_v, eps_det = cp['eps_e'], cp['eps_v'], cp['eps_det']
    e0 = [float(x) for x in K['e0']]
    # ---- stops and terminal errors (C10-C12)
    put('settling_time', B['dispersion'], k['settling'], 'upper', 's',
        'first-to-last stop interval (stopping-time dispersion) vs sum_k q_k')
    put('stop_increments', B['q'], k['stop_increments'], 'upper', 's',
        '|tau_i - tau_{i-1}| vs q_i')
    put('final_standstill', B['T_f_bar'], k['T_f'], 'upper', 's')
    put('stop_offsets', B['schedule_error'], k['sched'], 'upper', 's',
        'max_i |tau_i - tau_r|, tau_r = T_xi + v^xi/a_b, vs eps_v/a_b + sum_k q_k')
    put('stops_after_T_xi', [T_xi] * n, k['tstop'], 'floor', 's',
        'no unit stops before the handoff (velocity floor, C4)')
    put('marker_error', B['marker_max'], k['marker_abs_max'], 'upper', 'm')
    put('marker_error_units', B['marker'], np.abs(k['marker_signed']), 'upper', 'm',
        'per unit: head term + sum of the gap bounds ahead')
    put('terminal_gap_error', B['gap_max'], k['gap_abs_max'], 'upper', 'm')
    put('terminal_gap_error_pairs', B['gap'], np.abs(k['gap_signed']), 'upper', 'm',
        'pairs 2..n')
    # ---- clearance, barrier, margins (C8, C9)
    put('clearance_floor_pairs', B['clearance_floor'], k['gmin'], 'floor', 'm',
        'g_i = c_i - s_m (intersample minimum) vs g_min,i (C9a)')
    put('barrier_nonnegative', [0.0] * (n - 1), k['hmin'], 'floor', 'm')
    if T is not None:
        put('barrier_floor_acquisition', [p['hfloor_acq'] for p in P], k['h_acq_min'],
            'floor', 'm', 'min h_i on [0, t^sw_i] (1 ms record) vs h*^(1)_min + '
            'h~_i(0) - H^(1)_i (C9b)')
        put('barrier_floor_s2', [p['hfloor_s2'] for p in P], k['h_s2_min'], 'floor', 'm',
            'min h_i on [t^sw_i, T_xi) (1 ms record) vs h*^(2)_min + h~_i(0) - H^(1)_i '
            '- H^(2)_i (C9b)')
        put('barrier_braking', [p['h_b'] for p in P], k['h_brk_min'], 'floor', 'm',
            'min h_i from T_xi (1 ms record) vs h*_i(T_xi) + h~_i(0) - H^(1) - H^(2) - H^b')
    put('inactivity_margin_pairs', B['admission_slack_sigma'], k['mmin_lim'], 'floor',
        'm/s^2', 'u_cbf - u_nom (event left limits included) vs sigma_i (C8)')
    put('inactivity_margin', _min(B['admission_slack_sigma']), k['mmin_lim_min'], 'floor',
        'm/s^2')
    # ---- commands (C7)
    put('command_lower', [-x for x in K['Umin']], k['u_min_unit'], 'floor', 'm/s^2',
        'every applied command before LATCH >= -U_i^-')
    put('command_upper', [K['U']] * n, k['u_max_unit'], 'upper', 'm/s^2',
        'every applied command before LATCH <= U')
    put('nominal_command_upper', [K['U']] * n, k['unom_max_unit'], 'upper', 'm/s^2',
        'every nominal command before LATCH <= U (no clipping)')
    put('command_after_latch', 0.0, k['u_abs_after_latch'], 'upper', 'm/s^2',
        '|u| after LATCH (exactly 0)')
    put('command_upper_before_T_xi', B['u_max_before_T_xi'], k['u_max_pre_unit'], 'upper',
        'm/s^2', 'max_{t < T_xi} u_i vs max over the cells of P^+_i + F^+_i (C7a)')
    put('command_lower_before_T_xi', B['u_min_before_T_xi'], k['u_min_pre_unit'], 'floor',
        'm/s^2', 'min_{t < T_xi} u_i vs min over the cells of P^-_i - F^-_i (C7b)')
    if T is not None:
        put('command_upper_pre_receipt', B['u_max_pre_receipt'], k['u_prerec_max'],
            'upper', 'm/s^2', 'max u_i on [T_xi, t^rmp_i) (1 ms record) vs -a_crawl + '
            'Q^+_i (C7c_pre_up)')
        put('command_lower_pre_receipt', B['u_min_pre_receipt'], k['u_prerec_min'],
            'floor', 'm/s^2', 'min u_i on [T_xi, t^rmp_i) (1 ms record) vs -a_crawl - Q^-_i')
    put('braking_envelope', K['Fb_units'], k['fb_unit'], 'upper', 'm/s^2',
        'max |u_i + a_b| from the ramp receipt t^rmp_i to the standstill vs F^b_i '
        '(kernel for units 1..n-1, 1 ms record for every unit)')
    if T is not None and cp.get('curves') is not None:
        pw = pointwise(cp, T)
        rows_pw = dict(
            command_upper_pointwise=('up', 'm/s^2', 'max over the 1 ms record before '
                                     'T_xi of u_i(t) - (P^+_i + F^+_i)(cell of t)'),
            command_lower_pointwise=('lo', 'm/s^2', 'max over the 1 ms record before '
                                     'T_xi of (P^-_i - F^-_i)(cell of t) - u_i(t)'),
            velocity_floor_pointwise=('vf', 'm/s', 'max over the 1 ms record before '
                                      'T_xi of v_floor_i(cell of t) - v_i(t) (C4 cells)'))
        for nm, (key, unit, note) in rows_pw.items():
            put(nm, [0.0] * n, pw[key], 'upper', unit, note)
            rows[nm]['at'] = pw[key + '_at']
    if T is not None:
        put('velocity_floor', B['velocity_floor'], k['vmin_pre_min'], 'floor', 'm/s',
            'min speed before T_xi (1 ms record) vs the velocity floor (C4)')
        put('handoff_speed_floor', B['handoff_floor'], _min(k['v_at_T_xi']), 'floor',
            'm/s', 'min_i v_i(T_xi) vs v^xi - n eps_v (H8)')
    # ---- acquisition (C0-C3) with the FB takeover errors e_i(0) = c_i(0) - d_s,i
    put('activation_deadlines', B['activation_deadline'], k['tact'], 'upper', 's')
    put('detection_deadlines', B['switch_deadline'], k['tset'], 'upper', 's',
        'S2 switch (detection) instants t^sw_i vs the switch deadlines')
    put('earliest_switch_instants', K['tlow_sw'], k['tset'], 'floor', 's',
        'FB refinement: t^sw_i >= tlow_i (earliest S2 switch instants of the certificate)')
    put('activation_after_earliest_predecessor_switch',
        [0.0] + list(K['tlow_sw'][:-1]), k['tact'], 'floor', 's',
        't^act_i >= t^sw_{i-1} >= tlow_{i-1} (lemma of the FB refinement; head: 0)')
    tsw_prev = [0.0] + [_f(x) for x in list(k['tset'])[:-1]]
    put('activation_after_predecessor_switch', tsw_prev, k['tact'], 'floor', 's',
        'premise of the FB refinement on the run: t^act_i >= t^sw_{i-1} (observed; head: 0)')
    put('switch_position_lower', [a - b for a, b in zip(e0, K['E1'])], k['e_sw'], 'floor', 'm',
        'one-sided switch box: e^sw_i >= e_i(0) - E^(1)_i')
    put('switch_position_upper', e0, k['e_sw'], 'upper', 'm',
        'one-sided switch box: e^sw_i <= e_i(0)')
    put('switch_speed_lower', [-eps_det] * n, k['eps_sw'], 'floor', 'm/s')
    put('switch_speed_upper', [0.0] * n, k['eps_sw'], 'upper', 'm/s')
    if T is not None:
        put('first_window_position_excursion', K['E1'], k['e_first_dev_max'], 'upper', 'm',
            'max |e_i - e_i(0)| on [0, t^sw_i] (1 ms record) vs E^(1)_i')
        put('first_window_position_nonincreasing', e0, k['e_first_max'], 'upper', 'm',
            'max e_i on [0, t^sw_i] (1 ms record) vs e_i(0) (eps_i <= 0 there, C2)')
        put('first_window_velocity_error', [abs(a) + b for a, b in zip(K['eps0'], K['Lam1'])],
            k['eps_first_abs_max'], 'upper', 'm/s',
            'max |eps_i| on [0, t^sw_i] (1 ms record) vs |eps_i(0)| + Lambda^(1)_i')
        put('S1_sign', [0.0] * n, k['eps_first_max'], 'upper', 'm/s',
            'max eps_i on [0, t^sw_i] (1 ms record): every S1 stage brakes (C2)')
        put('s2_position_error', K['e2bar'], k['e_s2_abs_max'], 'upper', 'm',
            'max |e_i| on [t^sw_i, T_xi) (1 ms record) vs e-bar^(2)_i')
        put('s2_velocity_error', K['x2tot'], k['eps_s2_abs_max'], 'upper', 'm/s',
            'max |eps_i| on [t^sw_i, T_xi) (1 ms record) vs eps-bar^(2)_i')
    put('ramp_receipts', B['ramp_receipt_bound'], k['t_rmp'], 'upper', 's',
        't^rmp_i <= T_xi + (i-1) dbar (Lemma 6.3a)')
    drm = np.diff(k['t_rmp'])
    put('ramp_relay_steps', [dbar] * (n - 1), drm, 'upper', 's')
    put('ramp_relay_order', [0.0] * (n - 1), drm, 'floor', 's')
    # ---- handoff (C3, C5)
    put('persistent_entry', B['entry_deadline'], k['entry_obs'], 'upper', 's',
        'persistent entry of every pair error into the handoff box (last exit, '
        'every step) vs the entry deadline (C3)')
    if T is not None:
        put('persistent_entry_pairs', K['Tent_abs'], k['entry_units'], 'upper', 's',
            'per pair (1 ms record) vs Tent_i')
        if B['entry_deadline'] is not None:
            hold = box_hold(T, B['entry_deadline'], T_xi)
        else:
            hold = dict(e=None, eps=None)
        put('box_hold_position', eps_e if B['entry_deadline'] is not None else None,
            hold['e'], 'upper', 'm', 'max_i |e_i| on [T_ent_bar, T_xi) (1 ms record)')
        put('box_hold_velocity', eps_v if B['entry_deadline'] is not None else None,
            hold['eps'], 'upper', 'm/s', 'max_i |eps_i| on [T_ent_bar, T_xi) (1 ms record)')
    put('handoff_position_error', [eps_e] * n, np.abs(k['e_hnd']), 'upper', 'm',
        'pair position errors at T_xi')
    put('handoff_velocity_error', [eps_v] * n, np.abs(k['eps_hnd']), 'upper', 'm/s',
        'pair velocity errors at T_xi')
    # ---- relayed braking
    put('braking_mismatch_integral', K['W'], k['wint'], 'upper', 'm/s',
        'int |u_{i-1} - uhat_{i-1}| on [T_xi, min(tau_{i-1}, tau_i)) vs W_i')
    put('braking_speed_band', B['epsb'], k['epsmax_b'], 'upper', 'm/s',
        'max |v_{i-1} - v_i| on the braking window vs eps-bar_b,i')
    if T is not None:
        put('braking_position_band', B['eb'], k['ebmax_b'], 'upper', 'm',
            'max |e_i| on the braking window (1 ms record) vs e-bar_b,i')
    # ---- filter, age, completion, FB law
    put('filter_modifications', 0, k['filter_modifications'], 'upper', 'count',
        'min branch, fallback and saturation events of the filter')
    put('information_age', dbar, k['age']['max_left_limit_with_notional'], 'upper', 's',
        'held age, the zero command before the first delivery included')
    rows['all_stopped'] = dict(ok=bool(k['all_stopped']), kind='bool')
    rows['no_contact'] = dict(ok=k['contact_time'] is None, kind='bool')
    rows['no_rule_before_T_xi'] = dict(ok=not k['rule_before_T_xi'], kind='bool',
                                       units=k['rule_before_T_xi'])
    if T is not None:
        rows['fb_law'] = dict(ok=bool(k['fb_law']['ok']), kind='bool',
                              detail=dict((a, b) for a, b in k['fb_law'].items()
                                          if a in ('e_minus_parking_gap_max_abs',
                                                   'eps_minus_raw_max_abs', 'schedule_jumps',
                                                   'run_nosched', 'e_at_0_ok')))
    ok = all(r.get('ok') in (True, None) for r in rows.values())
    return OrderedDict(rows=rows, all_ok=bool(ok),
                       n_checked=sum(1 for r in rows.values() if r.get('ok') is not None),
                       n_rows=len(rows),
                       unchecked=[nm for nm, r in rows.items() if r.get('ok') is None],
                       failed=[nm for nm, r in rows.items() if r.get('ok') is False])


def row_slacks(bt):
    """smallest slack of every bounded row (bound - observed for upper rows,
    observed - bound for floors; None where no entry is compared)."""
    out = OrderedDict()
    for nm, r in bt['rows'].items():
        if r.get('kind') not in ('upper', 'floor'):
            continue
        bb = r['bound'] if isinstance(r['bound'], list) else [r['bound']]
        oo = r['observed'] if isinstance(r['observed'], list) else [r['observed']]
        sl = []
        for x, y in zip(bb, oo):
            if x is None or y is None:
                continue
            sl.append(x - y if r['kind'] == 'upper' else y - x)
        out[nm] = min(sl) if sl else None
    return out


def requirements(cp, k):
    r = OrderedDict(
        settling=bool(k['all_stopped'] and k['settling'] is not None
                      and k['settling'] <= cp['sync_tol']),
        marker=bool(k['all_stopped'] and k['marker_abs_max'] is not None
                    and k['marker_abs_max'] <= cp['align_tol']),
        gap=bool(k['all_stopped'] and k['gap_abs_max'] is not None
                 and k['gap_abs_max'] <= cp['gap_tol']),
        input_set_unmodified=bool(k['n_sat'] == 0),
        filter_inactive=bool(k['filter_modifications'] == 0),
        no_contact=k['contact_time'] is None,
        all_stopped=bool(k['all_stopped']))
    failed = [nm for nm, v in r.items() if v is False]
    r['all_ok'] = not failed
    r['failed'] = failed
    return r


def _ratio(obs, bound):
    if obs is None or bound is None or bound == 0:
        return None
    return float(obs) / float(bound)


def compact(k, bt=None, cp=None):
    """row of a schedule, sweep or ensemble table."""
    r = OrderedDict(
        settling=k['settling'], T_f=k['T_f'], sched=k['sched'],
        first_stop=k['first_stop'], marker_abs_max=k['marker_abs_max'],
        gap_abs_max=k['gap_abs_max'], marker_signed=k['marker_signed'],
        gap_signed=k['gap_signed'], u_min=k['u_min'], u_max=k['u_max'],
        u_max_pre=k['u_max_pre'], u_min_pre=k['u_min_pre'],
        u_max_pre_unit=k['u_max_pre_unit'],
        u_max_pre_informed=k['u_max_pre_informed'],
        unom_max=_max(k['unom_max_unit']), u_abs_after_latch=k['u_abs_after_latch'],
        n_pos_nom=k['n_pos_nom'], t_pos_nom=k['t_pos_nom'],
        n_pos_rec=k.get('n_pos_rec'), u_pos_max_rec=k.get('u_pos_max_rec'),
        n_mod=k['n_mod'], n_fallback=k['n_fallback'], n_sat=k['n_sat'],
        n_sat_hi=k['n_sat_hi'], n_sat_lo=k['n_sat_lo'], t_sat=k['t_sat'],
        filter_modifications=k['filter_modifications'],
        mmin_lim_min=k['mmin_lim_min'], gmin_min=k['gmin_min'], hmin_min=k['hmin_min'],
        gmin=k['gmin'], entry_obs=k['entry_obs'], entry_units=k.get('entry_units'),
        e_hnd_max=_max(np.abs(k['e_hnd'])), eps_hnd_max=_max(np.abs(k['eps_hnd'])),
        vmin_pre_min=k.get('vmin_pre_min'),
        e_s2_abs_max=k.get('e_s2_abs_max'), eps_s2_abs_max=k.get('eps_s2_abs_max'),
        e_closure_hi=k.get('e_closure_hi'), e_closure_lo=k.get('e_closure_lo'),
        eps_closure_hi=k.get('eps_closure_hi'), eps_closure_lo=k.get('eps_closure_lo'),
        tstop=k['tstop'], tset=k['tset'],
        tact=k['tact'], t_rmp=k['t_rmp'], trule=k['trule'],
        stop_order=k['stop_order'], head_first_tail_last=k['head_first_tail_last'],
        tail_first_head_last=k['tail_first_head_last'],
        rule_entered=k['rule_entered'], rule_units=k['rule_units'],
        rule_before_T_xi=k['rule_before_T_xi'],
        all_stopped=k['all_stopped'], contact_time=k['contact_time'],
        wint_max=_max(k['wint']), epsmax_b_max=_max(k['epsmax_b']),
        fb_unit_max=_max(k['fb_unit']), speed_largest_increment=_max(
            k.get('speed_largest_increment_unit') or []),
        fb_law_ok=(k.get('fb_law') or {}).get('ok'),
        age=k['age'], runtime_s=k.get('runtime_s'))
    if cp is not None:
        r['requirements'] = requirements(cp, k)
    if bt is not None:
        B = cp['bounds']
        r['bounds_ok'] = bt['all_ok']
        r['bounds_failed'] = bt['failed']
        r['n_checked'] = bt['n_checked']
        r['n_rows'] = bt['n_rows']
        r['slack'] = row_slacks(bt)
        r['ratio'] = OrderedDict(
            settling=_ratio(k['settling'], B['dispersion']),
            marker=_ratio(k['marker_abs_max'], B['marker_max']),
            gap=_ratio(k['gap_abs_max'], B['gap_max']),
            entry=_ratio(k['entry_obs'], B['entry_deadline']))
    return r


def diff(a, b, keys=KPI_KEYS):
    out = OrderedDict()
    for key in keys:
        x = np.atleast_1d(_arr(a.get(key)))
        y = np.atleast_1d(_arr(b.get(key)))
        if x.shape != y.shape:
            out[key] = float('nan')
            continue
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


def hist(values, width, lo=None):
    v = np.asarray(_fin(values), float)
    if v.size == 0:
        return None
    lo = math.floor(v.min() / width) * width if lo is None else lo
    hi = (math.floor(v.max() / width) + 1) * width
    nb = max(1, int(round((hi - lo) / width)))
    edges = lo + width * np.arange(nb + 1)
    cnt, _ = np.histogram(v, bins=edges)
    return dict(bin_width=width, edges=edges.tolist(), counts=cnt.tolist())


def quant(values):
    v = np.asarray(_fin(values), float)
    if v.size == 0:
        return None
    q = np.quantile(v, [0.0, 0.05, 0.5, 0.95, 1.0])
    return dict(min=float(q[0]), p05=float(q[1]), median=float(q[2]),
                p95=float(q[3]), max=float(q[4]), mean=float(v.mean()), n=int(v.size))


def aggregate(rows, cp=None):
    """extremes over named compact rows with the run that attains them, the
    worst slack of every bounded row, and the distributions."""
    it = list(rows.items())
    agg = OrderedDict(n=len(it))
    for key, fn in (('settling', max), ('T_f', max), ('sched', max),
                    ('marker_abs_max', max), ('gap_abs_max', max),
                    ('u_min', min), ('u_max', max), ('u_max_pre', max),
                    ('u_min_pre', min), ('unom_max', max), ('u_max_pre_informed', max),
                    ('mmin_lim_min', min), ('gmin_min', min), ('hmin_min', min),
                    ('entry_obs', max), ('e_hnd_max', max), ('eps_hnd_max', max),
                    ('vmin_pre_min', min), ('wint_max', max), ('epsmax_b_max', max),
                    ('fb_unit_max', max), ('speed_largest_increment', max)):
        v, nm = arg_ext(it, key, fn)
        agg[key] = dict(value=v, run=nm)
    v, nm = arg_ext(it, 'settling', min)
    agg['settling_min'] = dict(value=v, run=nm)
    agg['filter_modifications_total'] = sum(r['filter_modifications'] for _, r in it)
    agg['min_branch_total'] = sum(r['n_mod'] for _, r in it)
    agg['fallback_total'] = sum(r['n_fallback'] for _, r in it)
    agg['saturations_total'] = sum(r['n_sat'] for _, r in it)
    agg['saturations_upper_total'] = sum(r['n_sat_hi'] for _, r in it)
    agg['positive_nominal_total'] = sum(r['n_pos_nom'] for _, r in it)
    agg['runs_with_positive_nominal'] = sum(1 for _, r in it if r['n_pos_nom'] > 0)
    agg['positive_applied_record_total'] = sum((r.get('n_pos_rec') or 0) for _, r in it)
    agg['runs_with_positive_applied_record'] = sum(1 for _, r in it
                                                   if (r.get('n_pos_rec') or 0) > 0)
    agg['u_abs_after_latch_max'] = max(r['u_abs_after_latch'] for _, r in it)
    agg['all_stopped'] = all(r['all_stopped'] for _, r in it)
    agg['no_contact'] = all(r['contact_time'] is None for _, r in it)
    agg['head_first_tail_last_all'] = all(r['head_first_tail_last'] for _, r in it)
    agg['tail_first_head_last_all'] = all(r['tail_first_head_last'] for _, r in it)
    agg['stop_orders'] = sorted(set(tuple(r['stop_order'] or ()) for _, r in it))
    agg['rule_entered_any'] = any(r['rule_entered'] for _, r in it)
    agg['runs_with_rule_before_T_xi'] = sum(1 for _, r in it if r['rule_before_T_xi'])
    agg['rule_entered_all_followers_all'] = all(
        r['rule_units'] == list(range(2, len(r['tstop']) + 1)) for _, r in it)
    agg['fb_law_all'] = all(r.get('fb_law_ok') is True for _, r in it)
    agg['age_sup_max'] = max(r['age']['max_left_limit'] for _, r in it)
    agg['age_sup_min'] = min(r['age']['max_left_limit'] for _, r in it)
    agg['age_sup_with_notional_max'] = max(r['age']['max_left_limit_with_notional']
                                           for _, r in it)
    if all('requirements' in r for _, r in it):
        agg['requirements_all_ok'] = all(r['requirements']['all_ok'] for _, r in it)
    if all('bounds_ok' in r for _, r in it):
        agg['bounds_all_ok'] = all(r['bounds_ok'] for _, r in it)
        agg['n_bounds_failed'] = sum(1 for _, r in it if not r['bounds_ok'])
        agg['bounds_failed_rows'] = sorted(set(x for _, r in it for x in r['bounds_failed']))
        agg['all_rows_checked'] = all(r['n_checked'] == r['n_rows'] for _, r in it)
        agg['n_checked_min'] = min(r['n_checked'] for _, r in it)
        ws = OrderedDict()
        for nm in it[0][1]['slack']:
            vals = [(r['slack'][nm], rn) for rn, r in it if r['slack'].get(nm) is not None]
            if vals:
                v, rn = min(vals, key=lambda z: z[0])
                ws[nm] = dict(slack=v, run=rn)
        agg['worst_slack'] = ws
        for key in ('settling', 'marker', 'gap', 'entry'):
            v, nm = arg_ext([(rn, r['ratio']) for rn, r in it], key, max)
            agg['ratio_' + key + '_max'] = dict(value=v, run=nm)
    if cp is not None and cp.get('bounds') is not None:
        B = cp['bounds']
        agg['certified'] = dict(settling_bound=B['dispersion'], T_f_bar=B['T_f_bar'],
                                sched=B['schedule_error'], align=B['marker_max'],
                                gaperr_max=B['gap_max'], T_ent_bar=B['entry_deadline'],
                                clearance_floor_min=B['clearance_floor_min'],
                                sigma_min=min(B['admission_slack_sigma']),
                                u_max_before_T_xi=B['u_max_before_T_xi'])
    agg['distributions'] = OrderedDict(
        settling_ms=dict(hist=hist([None if r['settling'] is None else 1e3 * r['settling']
                                    for _, r in it], 1.0, 0.0),
                         quantiles=quant([None if r['settling'] is None else 1e3 * r['settling']
                                          for _, r in it])),
        marker_abs_max_mm=dict(hist=hist([None if r['marker_abs_max'] is None
                                          else 1e3 * r['marker_abs_max'] for _, r in it], 1.0, 0.0),
                               quantiles=quant([None if r['marker_abs_max'] is None
                                                else 1e3 * r['marker_abs_max'] for _, r in it])),
        gap_abs_max_mm=dict(hist=hist([None if r['gap_abs_max'] is None
                                       else 1e3 * r['gap_abs_max'] for _, r in it], 1.0, 0.0),
                            quantiles=quant([None if r['gap_abs_max'] is None
                                             else 1e3 * r['gap_abs_max'] for _, r in it])),
        entry_obs_s=dict(hist=hist([r['entry_obs'] for _, r in it], 1.0),
                         quantiles=quant([r['entry_obs'] for _, r in it])),
        T_f_s=dict(quantiles=quant([r['T_f'] for _, r in it])),
        gmin_min_m=dict(quantiles=quant([r['gmin_min'] for _, r in it])),
        mmin_lim_min=dict(quantiles=quant([r['mmin_lim_min'] for _, r in it])),
        u_max_pre=dict(quantiles=quant([r['u_max_pre'] for _, r in it])),
        u_max_pre_informed=dict(quantiles=quant([r['u_max_pre_informed'] for _, r in it])),
        e_hnd_max_m=dict(quantiles=quant([r['e_hnd_max'] for _, r in it])),
        eps_hnd_max=dict(quantiles=quant([r['eps_hnd_max'] for _, r in it])))
    return agg


def age_ok(k, bound):
    return bool(k['age']['max_left_limit_with_notional'] <= bound + TOL)


# ====================================================== worker plumbing
_W = {}


def _winit(payload):
    _W.clear()
    _W.update(payload or {})


def _wsetup():
    if 'S' not in _W:
        _W['S'] = setup_of(_W['cfg']) if 'cfg' in _W else design_setup()
    return _W['S']


def pool_map(func, jobs, payload=None, label='', every=0, chunksize=1):
    """ordered map over jobs in NPROC spawned processes (serial if NPROC = 1)."""
    t0 = time.time()
    out = []
    if NPROC <= 1 or len(jobs) <= 1:
        _winit(payload)
        for q, j in enumerate(jobs):
            out.append(func(j))
            if every and (q + 1) % every == 0:
                log('  %s %d/%d (%.0f s)' % (label, q + 1, len(jobs), time.time() - t0))
        return out
    ctx = mp.get_context('spawn')
    with ctx.Pool(min(NPROC, len(jobs)), initializer=_winit, initargs=(payload,)) as pool:
        for q, r in enumerate(pool.imap(func, jobs, chunksize=chunksize)):
            out.append(r)
            if every and (q + 1) % every == 0:
                log('  %s %d/%d (%.0f s)' % (label, q + 1, len(jobs), time.time() - t0))
    return out


def run_rows(S, cp, scheme, D=None, transport=TRANSPORT, h=None, T_end=None):
    """one run with its record, observations and bound rows."""
    R = simulate(S, scheme, h=h, transport=transport, fifo=D, T_end=T_end)
    T = traj(R, S)
    k = kpis(R, S, T)
    bt = bounds_rows(cp, k, T) if (cp is not None and cp.get('bounds') is not None) else None
    return R, T, k, bt


# ================================================================ ref
def summary(k):
    return ('disp=%.5f s T_f=%.4f s marker<=%.4f m gap<=%.4f m u in [%.5f, %.5f] '
            '(pre max %.5f, informed %.5f) mods=%d npos=%d entry=%s order=%s'
            % (k['settling'] if k['settling'] is not None else float('nan'),
               k['T_f'] if k['T_f'] is not None else float('nan'),
               k['marker_abs_max'] if k['marker_abs_max'] is not None else float('nan'),
               k['gap_abs_max'] if k['gap_abs_max'] is not None else float('nan'),
               k['u_min'], k['u_max'], k['u_max_pre'],
               k['u_max_pre_informed'] if k['u_max_pre_informed'] is not None else float('nan'),
               k['filter_modifications'],
               k['n_pos_nom'], '%.3f s' % k['entry_obs'] if k['entry_obs'] is not None
               else 'none', k['stop_order']))


def plan_table(S):
    """the head reference of a setup (floats and exact) and the takeover
    state; the followers store no schedule."""
    return OrderedDict(
        family='shaped',
        head=[dict(t0=p[0], a=p[1], s0=p[2], v0=p[3]) for p in S.head],
        head_exact=[dict(t0=str(p[0]), a=str(p[1]), s0=str(p[2]), v0=str(p[3]))
                    for p in S.head_exact],
        reference_steps=[dict(t=t, h=h_) for (t, h_) in ref_jumps(S)],
        schedules=None, t_0=S.t_0, T_xi=S.T_xi, tau_r=S.tau_r, tau_star=S.tau_r,
        s_ref_stop=S.s_ref_stop, markers=S.markers, v_in=float(S.plan.v_xi),
        v_xi=S.v_c1, d_s=S.d_s, c0=S.c0, v0=S.v0,
        e0=[0.0] + [float(S.c0[i - 1] - S.d_s[i]) for i in range(1, S.n)])


def certificate_consistency(cp):
    """bounds, condition slacks and constants of this evaluation against the
    certificate stored in design_fb.json (selected, 20 ms): bit-identical
    expected (same evaluator, same configuration)."""
    C = _DREC['selected']['certificate_20ms']
    same_b = OrderedDict()
    for key, x in C['bounds'].items():
        a = _arr(x)
        b = _arr(cp['bounds'].get(key))
        same_b[key] = bool(a.shape == b.shape and np.array_equal(a, b, equal_nan=True))
    same_c = OrderedDict()
    for nm, c in C['conds'].items():
        a = _f(c.get('slack'))
        b = cp['conds'].get(nm, {}).get('slack')
        same_c[nm] = bool((a is None and b is None) or (a is not None and b is not None
                                                       and a == b))
    diff_k = [k_ for k_ in ('tlow_sw', 'E1', 'e0', 'Lam1', 'tset', 'tact', 'W', 'Fb_units',
                            'Tent_abs', 'T_ent_bar')
              if k_ in C['consts'] and not np.array_equal(_arr(C['consts'][k_]),
                                                          _arr(cp['consts'].get(k_)),
                                                          equal_nan=True)]
    return OrderedDict(bounds_identical=all(same_b.values()),
                       bounds_differing=[k_ for k_, v in same_b.items() if not v],
                       condition_slacks_identical=all(same_c.values()),
                       conditions_differing=[k_ for k_, v in same_c.items() if not v],
                       constants_checked=['tlow_sw', 'E1', 'e0', 'Lam1', 'tset', 'tact', 'W',
                                          'Fb_units', 'Tent_abs', 'T_ent_bar'],
                       constants_differing=diff_k,
                       same_as_design=bool(all(same_b.values()) and all(same_c.values())
                                           and not diff_k))


def gap_closure(T1, R, S, cp):
    """closure of every follower gap under the FB law on the 1 ms record:
    per pair the takeover error e_i(0), the S2 switch instant, the first
    instants with e_i <= e_i(0)/2 and e_i <= e_i(0)/10, the first instant in
    the position band |e_i| <= eps_e, the persistent entry into the handoff
    box, the largest speed excess of the follower over its predecessor, the
    smallest clearance (undershoot below the parking gap), and the clearance
    at T_xi."""
    t = np.asarray(T1['t'], float)
    e = np.asarray(T1['e'], float)
    v = np.asarray(T1['v'], float)
    c = np.asarray(T1['c'], float)
    T_xi = float(R['T_xi'])
    pre = t < T_xi - 1e-9
    pairs = OrderedDict()
    for i in range(1, S.n):
        e0 = float(e[i, 0])
        ei = e[i]

        def first(mask):
            idx = np.nonzero(mask & pre)[0]
            return float(t[idx[0]]) if idx.size else None
        rel = v[i] - v[i - 1]
        jx = int(np.argmax(np.where(pre, rel, -np.inf)))
        km = int(np.argmin(np.where(pre, c[i - 1], np.inf)))
        kx = np.nonzero(np.abs(t - T_xi) < 1e-9)[0]
        below = np.nonzero((c[i - 1] < S.d_s[i]) & pre)[0]
        pairs[str(i + 1)] = OrderedDict(
            e0=e0, c0=float(c[i - 1, 0]), parking_gap=float(S.d_s[i]),
            t_sw=float(R['tset'][i]),
            t_half=first(ei <= 0.5 * e0), t_tenth=first(ei <= 0.1 * e0),
            t_position_band=first(np.abs(ei) <= S.eps_e),
            t_entry=None if T1.get('entry_units') is None else T1['entry_units'][i],
            max_speed_excess=float(rel[jx]), max_speed_excess_at=float(t[jx]),
            c_min_before_T_xi=float(c[i - 1, km]), c_min_at=float(t[km]),
            undershoot_below_parking=float(max(S.d_s[i] - c[i - 1, km], 0.0)),
            first_below_parking=float(t[below[0]]) if below.size else None,
            c_at_T_xi=float(c[i - 1, kx[0]]) if kx.size else None,
            e_at_T_xi=float(ei[kx[0]]) if kx.size else None)
    return OrderedDict(
        record_dt=float(t[1] - t[0]), pairs=pairs,
        t_half_max=_max(p['t_half'] for p in pairs.values()),
        t_tenth_max=_max(p['t_tenth'] for p in pairs.values()),
        t_position_band_max=_max(p['t_position_band'] for p in pairs.values()),
        max_speed_excess=_max(p['max_speed_excess'] for p in pairs.values()),
        undershoot_max=_max(p['undershoot_below_parking'] for p in pairs.values()),
        note='FB law: e_i = c_i - d_s,i from takeover (e_i(0) = c_i(0) - d_s,i); the S2 '
             'feedback closes the gaps; t_position_band: first instant with |e_i| <= eps_e')


def speed_monotonicity(T1, R):
    v = np.asarray(T1['v'], float)
    t = np.asarray(T1['t'], float)
    tz = _arr(R['tstop'])
    dv = np.diff(v, axis=1)
    inc, inc_at, mov, mov_at = [], [], [], []
    for i in range(v.shape[0]):
        j = int(np.argmax(dv[i]))
        inc.append(float(dv[i, j]))
        inc_at.append(float(t[j + 1]))
        m = t[1:] < (tz[i] if np.isfinite(tz[i]) else np.inf) - 1e-12
        jm = int(np.argmax(np.where(m, dv[i], -np.inf)))
        mov.append(float(dv[i, jm]))
        mov_at.append(float(t[jm + 1]))
    return dict(record_dt=float(t[1] - t[0]), n_samples=int(t.size),
                largest_increment=max(inc), largest_increment_unit=inc,
                largest_increment_at=inc_at,
                largest_increment_while_moving=max(mov),
                largest_increment_while_moving_unit=mov,
                largest_increment_while_moving_at=mov_at,
                nonincreasing_all=bool(max(inc) <= 0.0),
                strictly_decreasing_while_moving=bool(max(mov) < 0.0),
                note='informational: the certificate does not certify nonpositive '
                     'commands (NONPOS is report-only)')


def phase_extrema(T, R, S):
    t = T['t']
    T_xi = float(R['T_xi'])
    J = ref_jumps(S)
    t_first = J[0][0] if J else T_xi
    t_crawl = J[-1][0] if J else T_xi
    tz = _arr(R['tstop'])
    tsw = float(np.nanmax(_arr(R['tset'])))
    ph = OrderedDict()
    for name, (a, b) in (('acquisition', (0.0, tsw)),
                         ('hardest_deceleration', (0.0, t_first)),
                         ('reference_steps', (t_first, t_crawl)),
                         ('crawl', (t_crawl, T_xi)),
                         ('braking', (T_xi, float(np.nanmax(tz))))):
        m = (t >= a - 1e-9) & (t <= b + 1e-9)
        if not m.any():
            ph[name] = dict(t=[a, b], samples=0)
            continue
        ph[name] = dict(t=[a, b], samples=int(m.sum()),
                        v_min=float(np.min(T['v'][:, m])), v_max=float(np.max(T['v'][:, m])),
                        u_min=float(np.min(T['u'][:, m])), u_max=float(np.max(T['u'][:, m])),
                        dev_min=float(np.min(T['dev'][:, m])),
                        dev_max=float(np.max(T['dev'][:, m])),
                        e_abs_max=float(np.max(np.abs(T['e'][:, m]))),
                        e_follower_min=float(np.min(T['e'][1:, m])),
                        e_follower_max=float(np.max(T['e'][1:, m])),
                        eps_abs_max=float(np.max(np.abs(T['eps'][:, m]))),
                        c_min=float(np.min(T['c'][:, m])), c_max=float(np.max(T['c'][:, m])))
    ph['note'] = ('phases of the FB run: acquisition [0, last S2 switch]; hardest_deceleration '
                  '= first reference piece [0, first step); reference_steps [first step, crawl '
                  'onset]; crawl up to T_xi; braking from T_xi to the last stop; dev = u - a_r')
    return ph, dict(t_first_step=t_first, t_crawl=t_crawl, last_switch=tsw)


TRAJ_NOTE = ('t [s]; s positions; v unit speeds; u applied commands (right limits); '
             'e, eps pair errors of the FB law (row 0: head against its reference, e_1 = '
             's_0 - s_1 - (d_s,1 + ell), eps_1 = v_r - v_1; rows i >= 1: e_i = c_i - d_s,i, '
             'eps_i = v_{i-1} - v_i, from takeover); g clearance margins c_i - s_m; c '
             'clearances; h barriers h_i (hbar: same); m inactivity margins u_cbf - u_nom '
             'at the last command update (NaN once latched); modes (0 COAST, 1 S1, 2 S2, '
             '3 RULE, 4 LATCH, 5 BRAKE; post-event values); s0, vr, ar head reference '
             '(virtual leader); d desired clearances (= d_s,i, constant), dd, rho zero (no '
             'schedules); ps, pv, pa realization plan of the shaped plan (pa_i = a_r, pv_i = '
             'v_r, ps_i = ps_{i-1} - ell - d_s,i); dev = u - pa; tstop stop instants; tset S2 '
             'switch instants; tact activations; tcl instants at which the inert CLOSE bit '
             'of the simulator arrived (not used by the FB law); trule RULE entries; t_rmp '
             'ramp receipts; marks markers; s_final final positions; egap signed terminal '
             'gap errors (entry 0: head slot); tlow_sw earliest S2 switch instants of the '
             'certificate.  After the last stop the record repeats the final state.')

TRAJ_KEYS = ('t', 's', 'v', 'u', 'e', 'eps', 'g', 'c', 'h', 'm', 'modes', 's0', 'vr',
             'ar', 'd', 'dd', 'rho', 'ps', 'pv', 'pa', 'dev')
WIN_KEYS = ('t', 's', 'v', 'u', 'e', 'eps', 'g', 'c', 'h', 'm', 'modes', 'vr', 'ar',
            'pv', 'pa')


def save_traj(path, T, R, S, extra=None, keys=TRAJ_KEYS, note=TRAJ_NOTE, cp=None):
    out = dict(keys_note=np.array(note),
               config=np.array(json.dumps(_py(R['config']))),
               cfg=np.array(json.dumps(_py(S.cfg))),
               plan=np.array(json.dumps(_py(plan_table(S)))),
               mode_names=np.array(json.dumps(list(hr.MODE_NAMES))),
               record_dt=T['rec_dt'])
    for key in keys:
        out[key] = T[key]
    if 'h' in keys:
        out['hbar'] = T['h']
    for key in ('tstop', 'tset', 'tact', 'tcl', 'trule', 't_rmp', 'marker_signed',
                'e_hnd', 'eps_hnd', 'e_sw', 'eps_sw'):
        out[key] = np.asarray(R[key], float)
    out['marks'] = np.asarray(R['marks'], float)
    out['s_final'] = np.asarray(R['s'], float)
    out['egap'] = np.asarray(R['egap'], float)
    out['d_s'] = np.asarray(S.d_s, float)
    out['c0'] = np.asarray(S.c0, float)
    out['v0'] = np.asarray(S.v0, float)
    out['e0'] = np.array([0.0] + [float(S.c0[i - 1] - S.d_s[i]) for i in range(1, S.n)])
    out['T_xi'] = float(R['T_xi'])
    out['t_0'] = float(R['t_0'])
    out['tau_star'] = float(R['tau_r'])
    if cp is not None and cp.get('consts') is not None:
        out['tlow_sw'] = np.asarray(cp['consts']['tlow_sw'], float)
        out['switch_deadline'] = np.asarray(cp['bounds']['switch_deadline'], float)
        out['entry_deadline'] = float(cp['bounds']['entry_deadline'])
    if extra:
        out.update(extra)
    np.savez_compressed(path, **out)
    log('  -> data/new_d19/fbref/%s (%s samples, record %.3g s)'
        % (os.path.basename(path), T['t'].size if 't' in keys else 'windowed', out['record_dt']))


def part_ref():
    t_part = time.time()
    cfgc = cert_cfg()
    dc = check_design_constants(cfgc)
    t0 = time.time()
    ev = cf.evaluate(cfgc, keep_curves=True)
    cp = cert_pack(ev, cfgc)
    log('[ref] certificate lam=%g vhnd=%g lags=%s dbar=%g ms: admitted=%s failed=%s '
        '(%.1f s); design constants %s'
        % (LAM, VHND, LAGS, 1e3 * DBAR, cp['admitted'], cp['failed'], time.time() - t0,
           dc['all']))
    cons = certificate_consistency(cp)
    log('  bounds and slacks identical to design_fb.json (selected, 20 ms): %s (differing %s %s %s)'
        % (cons['same_as_design'], cons['bounds_differing'], cons['conditions_differing'],
           cons['constants_differing']))
    S = design_setup()
    # ---- reference run: 'cont' at h = 1e-4, records every 10 ms and every 1 ms
    R10 = simulate(S, 'cont', h=H_REF, rec_dt=0.01)
    R1 = simulate(S, 'cont', h=H_REF, rec_dt=1e-3)
    T1 = traj(R1, S)
    k = kpis(R1, S, T1)
    T10 = traj(R10, S)
    k10 = kpis(R10, S, T10)
    same_rec = kpi_identical(k10, k)
    bt = bounds_rows(cp, k, T1)
    law = k['fb_law']
    log('  reference h=%g: %s (%.1f s); records 10 ms / 1 ms identical: %s'
        % (H_REF, summary(k), R1['runtime_s'], same_rec['identical']))
    log('  FB law on the reference run: ok %s (setup schedules %s, run schedules %s, nosched %s, '
        'schedule jumps %d, max |e_i - (c_i - d_s,i)| %s, max |eps_i - (v_{i-1} - v_i)| %s, '
        'e(0) %s)' % (law['ok'], law['setup_has_schedules'], law['run_schedules'],
                      law['run_nosched'], law['schedule_jumps'],
                      law['e_minus_parking_gap_max_abs'], law['eps_minus_raw_max_abs'],
                      law['e_at_0']))
    # ---- fine: h = 2.5e-5 (convergence of the reference run)
    Rf1 = simulate(S, 'cont', h=H_FINE, rec_dt=1e-3)
    Tf1 = traj(Rf1, S)
    kf = kpis(Rf1, S, Tf1)
    btf = bounds_rows(cp, kf, Tf1)
    dfine = diff(k, kf)
    mxd, kxd, badd = dmax(dfine)
    log('  fine h=%g: %s (%.1f s); max |reference - fine| %.2e (%s)'
        % (H_FINE, summary(kf), Rf1['runtime_s'], mxd, kxd))
    for name, r in bt['rows'].items():
        if r.get('kind') == 'bool':
            log('  %-44s ok=%s' % (name, r['ok']))
            continue
        fmt = (lambda x: '[%s]' % ', '.join('%.5g' % z if z is not None else '-' for z in x)
               if isinstance(x, list) else ('%.6g' % x if x is not None else '-'))
        log('  %-44s bound %-48s observed %-48s ok=%s'
            % (name, fmt(r['bound']), fmt(r['observed']), r['ok']))
    log('  all bounds ok (reference): %s failed %s; checked %d of %d rows; fine: %s %s'
        % (bt['all_ok'], bt['failed'], bt['n_checked'], bt['n_rows'], btf['all_ok'],
           btf['failed']))
    # ---- records for the figures (reference run, h = 1e-4 s)
    save_traj(os.path.join(OUT, 'traj_fb.npz'), T10, R10, S, cp=cp)
    save_traj(os.path.join(OUT, 'traj_fb_1ms.npz'), T1, R1, S, cp=cp)
    # every integration step of h = 2.5e-5 s on two windows
    T_f = float(np.nanmax(Rf1['tstop']))
    Rs = simulate(S, 'cont', h=H_FINE, rec_dt=H_FINE, T_end=round(T_f + 0.02, 3))
    ks = kpis(Rs, S)
    same_stop = kpi_identical(ks, kf)
    tt = np.asarray(Rs['rec']['t'], float)
    wins = OrderedDict(start=START_WINDOW, stop=(S.T_xi - 0.05, T_f + 0.005))
    ex = dict(record_dt=H_FINE, window_start=np.array(wins['start']),
              window_stop=np.array(wins['stop']))
    res_stop = []
    Tw = None
    for nm, (a, b) in wins.items():
        sel = np.nonzero(np.isfinite(tt) & (tt >= a - 1e-12) & (tt <= b + 1e-12))[0]
        Tw = traj(Rs, S, sel)
        for key in WIN_KEYS:
            ex[key + '_' + nm] = Tw[key]
        if nm == 'stop':
            # stop instants resolved on the record: extrapolation of the last
            # two positive speeds of every unit to zero
            for i in range(S.n):
                vi = Tw['v'][i]
                pos = np.nonzero(vi > 0.0)[0]
                j = int(pos[-1]) if pos.size else None
                if j is None or j < 1:
                    res_stop.append(None)
                    continue
                slope = (vi[j] - vi[j - 1]) / (Tw['t'][j] - Tw['t'][j - 1])
                tzx = float(Tw['t'][j] - vi[j] / slope)
                res_stop.append(tzx)
        log('  window %-5s [%.4f, %.4f] s: %d samples' % (nm, a, b, sel.size))
    tz = np.asarray(Rs['tstop'], float)
    stop_res = [None if x is None else abs(x - z) for x, z in zip(res_stop, tz)]
    ex['tstop_from_record'] = np.array([np.nan if x is None else x for x in res_stop])
    save_traj(os.path.join(OUT, 'traj_fb_stop.npz'), Tw, Rs, S, extra=ex, keys=(), cp=cp,
              note=TRAJ_NOTE + '  Every integration step (h = 2.5e-5 s) on two windows: '
              'suffix _start: [0, 4 s]; suffix _stop: [T_xi - 0.05 s, T_f + 0.005 s]; '
              'tstop_from_record: zero of the linear extrapolation of the last two positive '
              'speeds of the stop window (resolution check of the stop instants).')
    del Rs
    gc.collect()
    # ---- informational observations
    mono = speed_monotonicity(T1, R1)
    closure = gap_closure(dict(T1, entry_units=k['entry_units']), R1, S, cp)
    ph, ph_t = phase_extrema(T10, R10, S)
    req = requirements(cp, k)
    B = cp['bounds']
    K = cp['consts']
    nonpos = ev['conds']['NONPOS']
    headline = OrderedDict(
        settling_time=dict(observed=k['settling'], bound=B['dispersion'],
                           requirement=cp['sync_tol']),
        marker_error=dict(observed=k['marker_abs_max'], bound=B['marker_max'],
                          requirement=cp['align_tol']),
        terminal_gap_error=dict(observed=k['gap_abs_max'], bound=B['gap_max'],
                                requirement=cp['gap_tol']),
        completion=dict(observed=k['T_f'], bound=B['T_f_bar'], planned_stop=S.tau_r),
        command_range=dict(observed=[k['u_min'], k['u_max']],
                           before_T_xi=[k['u_min_pre'], k['u_max_pre']],
                           before_T_xi_informed_max=k['u_max_pre_informed'],
                           before_T_xi_certified=[min(B['u_min_before_T_xi']),
                                                  max(B['u_max_before_T_xi'])],
                           before_T_xi_certified_per_unit=dict(
                               upper=B['u_max_before_T_xi'], lower=B['u_min_before_T_xi']),
                           after_T_xi=[_min(k['u_min_post_unit']), _max(k['u_max_post_unit'])],
                           input_set=[-U_BR, U_TR],
                           after_latch_abs_max=k['u_abs_after_latch']),
        positive_commands=dict(n_positive_nominal=k['n_pos_nom'],
                               t_positive_nominal=k['t_pos_nom'],
                               n_positive_applied_record=k['n_pos_rec'],
                               u_max_before_T_xi_informed=k['u_max_pre_informed'],
                               certificate_NONPOS=dict(ok=nonpos['ok'], slack=nonpos['slack'],
                                                       before_T_xi=nonpos.get('before_T_xi'),
                                                       at=nonpos.get('before_T_xi_at'),
                                                       note='report-only: the certificate does '
                                                            'not certify nonpositive commands')),
        clearance=dict(observed=k['gmin_min'], bound=B['clearance_floor_min']),
        margin=dict(observed=k['mmin_lim_min'], bound=min(B['admission_slack_sigma'])),
        entry=dict(observed=k['entry_obs'], bound=B['entry_deadline'], T_xi=S.T_xi),
        switches=dict(observed=k['tset'], earliest=K['tlow_sw'],
                      deadline=B['switch_deadline']),
        age=dict(observed=k['age']['max_left_limit_with_notional'], bound=DBAR),
        stop_order=k['stop_order'], rule_entered=k['rule_entered'],
        rule_before_T_xi=k['rule_before_T_xi'],
        filter_modifications=k['filter_modifications'],
        speed_nonincreasing=mono['nonincreasing_all'],
        speed_largest_increment=mono['largest_increment'],
        gap_closure=dict(t_half_max=closure['t_half_max'], t_tenth_max=closure['t_tenth_max'],
                         t_position_band_max=closure['t_position_band_max'],
                         undershoot_max=closure['undershoot_max']))
    P = dict(n=S.n, dt=1e-3, dbar=DBAR, transport=TRANSPORT, a_b=S.a_b, v_xi=S.v_c1,
             v_in=float(S.plan.v_xi), T_xi=S.T_xi, T_END=S.T_END, alpha=S.alpha,
             lam=S.lam, beta=S.beta, gamma=S.gamma, phi=S.phi, eps_det=S.eps_det,
             ell=S.ell, s_m=S.s_m, kappa=S.kappa, U=S.U, Uminus=S.Uminus,
             amax=S.amax, v_c=S.v_c, b=S.b, eps_e=S.eps_e, eps_v=S.eps_v)
    rec = OrderedDict(
        meta=meta('reference', options=dict(scheme='cont', h=H_REF, h_fine=H_FINE,
                                            transport=TRANSPORT, rec_dt=[0.01, 1e-3],
                                            start_window=START_WINDOW, filter=True)),
        design='FB (feedback-only followers, shaped head reference); lam = %g, U = %g, '
               'U^- = %g, V^hnd = %g, lags %s, dbar = %g ms' % (LAM, U_TR, U_BR, VHND, LAGS,
                                                                1e3 * DBAR),
        design_constants=dc,
        cfg=S.cfg, cert_cfg=cfgc, P=P, family='shaped (head reference only; no schedules)',
        plan=plan_table(S), certificate=cert_summary(cp),
        certificate_full=dict(conds=cp['conds'], bounds=cp['bounds'], consts=cp['consts']),
        certificate_nonpos=dict((a, _py(b)) for a, b in nonpos.items()),
        certificate_consistency=cons,
        fb_law_check=law,
        transport=TRANSPORT, age_bound=DBAR, kpis=k, bounds=bt,
        requirements=req, headline=headline,
        record_identical_10ms_1ms=same_rec,
        fine=OrderedDict(h=H_FINE, kpis=kf, bounds=btf,
                         diff_reference_fine=dfine, max_diff=mxd, argmax=kxd,
                         nan_mismatch=badd, max_stop_diff=dfine['tstop'],
                         stop_window_run_identical=same_stop),
        stop_record=OrderedDict(
            record_dt=H_FINE, windows=wins, tstop=tz, tstop_from_record=res_stop,
            max_abs_diff=_max(stop_res),
            resolves_1us=bool(_max(stop_res) is not None and _max(stop_res) < 1e-6),
            note='stop instants from the root finder of sim_hr (RK4 step localized); the '
                 'record every 25 us resolves them: linear extrapolation of the last two '
                 'positive speeds'),
        figure_records=dict(traj_fb='every 10 ms, reference run (cont, h = 1e-4 s)',
                            traj_fb_1ms='every 1 ms, reference run (cont, h = 1e-4 s)',
                            traj_fb_stop='every step of the fine run (h = 2.5e-5 s) on [0, 4 s] '
                                         'and [T_xi - 0.05 s, T_f + 0.005 s]'),
        pointwise=dict(
            command_upper=bt['rows']['command_upper_pointwise'],
            command_lower=bt['rows']['command_lower_pointwise'],
            velocity_floor=bt['rows']['velocity_floor_pointwise'],
            record_dt=REC_DT),
        speed_monotonicity=mono, gap_closure=closure,
        phase_times=OrderedDict(
            t_0=S.t_0, t_first_step=ph_t['t_first_step'], t_crawl=ph_t['t_crawl'],
            T_xi=S.T_xi, tau_star=S.tau_r,
            tstop=k['tstop'], T_f=k['T_f'], T_f_bar=B['T_f_bar'],
            tset_obs=k['tset'], tact_obs=k['tact'], tlow_sw=K['tlow_sw'],
            trule_obs=k['trule'], t_rmp=k['t_rmp'], tcl_inert=k['tcl'],
            tset_cert=B['switch_deadline'], tact_cert=B['activation_deadline'],
            ramp_receipt_bound=B['ramp_receipt_bound'],
            entry_obs=k['entry_obs'], T_ent_bar=B['entry_deadline'],
            entry_units=k['entry_units'], Tent_abs=K['Tent_abs']),
        phase_extrema=ph,
        filter_modifications=dict(min_branch=k['n_mod'], fallback=k['n_fallback'],
                                  saturation=k['n_sat'], evaluations=k['n_eval'],
                                  left_limit_evaluations=k['n_left_limits']),
        age_audit=R1['age'], runtime_s=R1['runtime_s'], config=R1['config'],
        system=('FB closed loop (sim_hr.py, schedules=False, nosched=park): continuous local '
                'law (T_u = h), RK4 between events, T_m = T_c = 1 ms, constant transport '
                '19 ms, shaped head reference with its steps as head clock events, followers '
                'regulating c_i - d_s,i from takeover (COAST, S1, S2 with the received '
                'command as feedforward), relayed braking, exact standstill sensing, RULE and '
                'LATCH, safety filter on'),
        terminology=('settling time = first-to-last stop interval (stopping-time '
                     'dispersion); tset are the S2 switch (detection) instants, compared with '
                     'the switch deadlines and the earliest switch instants tlow_sw of the '
                     'certificate'),
        part_runtime_s=time.time() - t_part)
    dump('reference.json', rec)
    log('  headline: dispersion %.3f ms (bound %.4f s), marker %.4f m (%.4f), gap %.4f m '
        '(%.4f), T_f %.4f s (%.4f), u in [%.5f, %.5f], entry %.3f s (%.3f), age %.3f ms'
        % (1e3 * k['settling'], B['dispersion'], k['marker_abs_max'], B['marker_max'],
           k['gap_abs_max'], B['gap_max'], k['T_f'], B['T_f_bar'], k['u_min'], k['u_max'],
           k['entry_obs'], B['entry_deadline'], 1e3 * k['age']['max_left_limit_with_notional']))
    return rec


# ================================================================ conv
def part_conv():
    t_part = time.time()
    cfgc = cert_cfg()
    cp = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    S = design_setup()
    log('[conv] cont h in %s (+ diagnostic %s), constant transport 19 ms' % (H_CONV, H_DIAG))
    rows = []
    for h in H_DIAG + H_CONV:
        R, T, k, bt = run_rows(S, cp, 'cont', h=h)
        rows.append(dict(h=h, diagnostic=h in H_DIAG, kpis=k, bounds_ok=bt['all_ok'],
                         bounds_failed=bt['failed'], n_checked=bt['n_checked'],
                         n_rows=bt['n_rows'], slack=row_slacks(bt)))
        log('  h=%-8g %s bounds_ok=%s (%.1f s)' % (h, summary(k), bt['all_ok'], R['runtime_s']))
    ref = rows[-1]['kpis']
    for r in rows:
        d = diff(r['kpis'], ref)
        mx, kx, bad = dmax(d)
        r['diff_to_finest'] = d
        r['max_diff_to_finest'] = mx
        r['argmax_kpi'] = kx
        r['nan_mismatch'] = bad
    conv = [r for r in rows if not r['diagnostic']]
    succ = OrderedDict()
    for a, b in zip(conv[:-1], conv[1:]):
        d = diff(a['kpis'], b['kpis'])
        succ['%g->%g' % (a['h'], b['h'])] = OrderedDict(
            tstop=d['tstop'], settling=d['settling'], marker=d['marker_signed'],
            gap=d['gap_signed'], gmin=d['gmin'], hmin=d['hmin'], mmin_lim=d['mmin_lim'],
            entry_obs=d['entry_obs'], e_hnd=d['e_hnd'], eps_hnd=d['eps_hnd'],
            max=dmax(d)[0], argmax=dmax(d)[1])
        log('  %-14s |dtau| %.2e s |ddisp| %.2e s |dmarker| %.2e m |dgap| %.2e m '
            '|dgmin| %.2e m |de_hnd| %.2e m' % ('%g->%g' % (a['h'], b['h']), d['tstop'],
                                               d['settling'], d['marker_signed'],
                                               d['gap_signed'], d['gmin'], d['e_hnd']))
    worst = {key: max(r['diff_to_finest'][key] for r in conv[:-1]) for key in KPI_KEYS}
    mx, kx, bad = dmax(worst)
    out = OrderedDict(
        meta=meta('convergence', options=dict(h=list(H_CONV), diagnostic_h=list(H_DIAG),
                                              transport=TRANSPORT)),
        study=('numerical convergence: the same event schedule (sends every T_m = 1 ms, '
               'constant transport 19 ms, detector T_c = 1 ms anchored at activation) '
               'and the same continuous FB law (sim_hr scheme cont, nosched park); only the '
               'RK4 step h varies; differences to h = %g s and between successive steps'
               % H_CONV[-1]),
        h_values=list(H_CONV), diagnostic_h=list(H_DIAG), rows=rows,
        successive=succ, max_diff_over_h_by_kpi=worst, max_diff_over_h_and_kpis=mx,
        argmax_kpi=kx, nan_mismatch=bad,
        max_stop_diff=max(r['diff_to_finest']['tstop'] for r in conv[:-1]),
        all_rows_inside_bounds=all(r['bounds_ok'] for r in rows),
        all_rows_checked=all(r['n_checked'] == r['n_rows'] for r in rows),
        all_filter_inactive=all(r['kpis']['filter_inactive'] for r in rows),
        all_age_ok=all(age_ok(r['kpis'], DBAR) for r in rows),
        all_fb_law=all(r['kpis']['fb_law']['ok'] for r in rows),
        runtime_s=time.time() - t_part)
    log('  max |diff| to h=%g over the refinements: %.2e (%s); max stop diff %.2e s'
        % (H_CONV[-1], mx, kx, out['max_stop_diff']))
    dump('convergence.json', out)
    return out


# ============================================================= digital
def part_digital():
    t_part = time.time()
    cfgc = cert_cfg()
    cp = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    S = design_setup()
    N = n_gen(S)
    cases = [('const19', None)]
    for j in range(N_DIGITAL):
        seed = SEED_DIGITAL + j
        cases.append(('seed%d' % seed, rcv.random_schedule(np.random.default_rng(seed), N, 19)))
    log('[digital] zoh 1 kHz (T_u = h = 1 ms) vs cont (h = %g s): constant 19 ms and %d '
        'random schedules' % (H_REF, N_DIGITAL))
    rows = []
    for name, D in cases:
        Rc, Tc, kc, btc = run_rows(S, cp, 'cont', D=D, h=H_REF)
        Rz, Tz, kz, btz = run_rows(S, cp, 'zoh', D=D, h=H_ZOH)
        d = diff(kz, kc)
        mx, kx, bad = dmax(d)
        common = dict(setting=name, schedule=None if D is None else dict(
            d_min=int(D[1:].min()), d_max=int(D[1:].max()), d_mean=float(D[1:].mean())))
        rows.append(dict(common, scheme='cont', T_u=H_REF, h=H_REF, kpis=kc,
                         bounds_ok=btc['all_ok'], bounds_failed=btc['failed'],
                         n_checked=btc['n_checked'], n_rows=btc['n_rows'],
                         slack=row_slacks(btc), age_ok=age_ok(kc, DBAR)))
        rows.append(dict(common, scheme='zoh', T_u=T_M, h=H_ZOH, kpis=kz,
                         bounds_ok=btz['all_ok'], bounds_failed=btz['failed'],
                         n_checked=btz['n_checked'], n_rows=btz['n_rows'],
                         slack=row_slacks(btz), age_ok=age_ok(kz, DBAR),
                         diff_to_continuous=d, max_diff_to_continuous=mx, argmax_kpi=kx,
                         nan_mismatch=bad, max_stop_diff=d['tstop'],
                         settling_diff=d['settling'], max_marker_diff=d['marker_signed'],
                         max_gap_diff=d['gap_signed'], entry_diff=d['entry_obs'],
                         jumps_on_update_grid=True))
        log('  %-9s cont: %s bounds_ok=%s' % (name, summary(kc), btc['all_ok']))
        log('  %-9s zoh : %s bounds_ok=%s | |dtau| %.2e s |ddisp| %.2e s |dmarker| %.2e m '
            '|dentry| %.2e s' % (name, summary(kz), btz['all_ok'], d['tstop'], d['settling'],
                                 d['marker_signed'], d['entry_obs']))
    zr = [r for r in rows if r['scheme'] == 'zoh']
    out = OrderedDict(
        meta=meta('digital', seeds=[SEED_DIGITAL + j for j in range(N_DIGITAL)],
                  options=dict(h_cont=H_REF, h_zoh=H_ZOH, T_u=T_M, dmax_steps=19)),
        study=('digital implementation: the FB law sampled every T_u = 1 ms and held '
               '(zero-order hold, pure 1 kHz: h = T_u = T_m = T_c = 1 ms), each unit also '
               'updates at its own events (packet batch, detector sample, own reference '
               'step at the head, zero-speed instants); every step of the head reference '
               'lies on the 1 ms grid; compared with the continuous law (h = %g s) on the '
               'constant transport 19 ms and %d random schedules (seed %d + j, '
               'random_schedule of run_caseD_validate)' % (H_REF, N_DIGITAL, SEED_DIGITAL)),
        rows=rows,
        all_bounds_ok=all(r['bounds_ok'] for r in rows),
        all_rows_checked=all(r['n_checked'] == r['n_rows'] for r in rows),
        all_filter_inactive=all(r['kpis']['filter_inactive'] for r in rows),
        all_age_ok=all(r['age_ok'] for r in rows),
        all_fb_law=all(r['kpis']['fb_law']['ok'] for r in rows),
        max_stop_diff_1kHz=max(r['max_stop_diff'] for r in zr),
        max_settling_diff_1kHz=max(r['settling_diff'] for r in zr),
        max_marker_diff_1kHz=max(r['max_marker_diff'] for r in zr),
        max_gap_diff_1kHz=max(r['max_gap_diff'] for r in zr),
        max_entry_diff_1kHz=max(r['entry_diff'] for r in zr),
        runtime_s=time.time() - t_part)
    log('  zoh vs cont: max |dtau| %.2e s, |ddisp| %.2e s, |dmarker| %.2e m, |dentry| %.2e s; '
        'all bounds ok %s' % (out['max_stop_diff_1kHz'], out['max_settling_diff_1kHz'],
                              out['max_marker_diff_1kHz'], out['max_entry_diff_1kHz'],
                              out['all_bounds_ok']))
    dump('digital.json', out)
    return out


# ============================================================ schedules
def _job_sched(job):
    name, kind, arg, meta_ = job
    S = _wsetup()
    cp = _W['cp']
    if kind == 'random':
        D = rcv.random_schedule(np.random.default_rng(arg), n_gen(S), 19)
    else:
        D = np.asarray(arg, np.int64)
    R, T, k, bt = run_rows(S, cp, 'zoh', D=D)
    r = compact(k, bt, cp)
    r['age_ok'] = age_ok(k, cp['dbar'])
    r['schedule'] = dict(d_min=int(D[1:].min()), d_max=int(D[1:].max()),
                         d_mean=float(D[1:].mean()))
    if meta_ is not None:
        r['family'] = meta_
    if _W.get('e_ref') is not None:
        # pair errors minus those of the constant 19 ms run before T_xi (both
        # zoh 1 kHz, same record instants)
        t = T['t']
        w = t < S.T_xi - 1e-9
        de = T['e'][:, w] - _W['e_ref'][:, w]
        dx = T['eps'][:, w] - _W['eps_ref'][:, w]
        r['de_closure_hi'] = [float(x) for x in de.max(axis=1)]
        r['de_closure_lo'] = [float(x) for x in de.min(axis=1)]
        r['deps_closure_hi'] = [float(x) for x in dx.max(axis=1)]
        r['deps_closure_lo'] = [float(x) for x in dx.min(axis=1)]
    return name, _py(r)


def _sched_payload():
    cfgc = cert_cfg()
    cp = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    return dict(cfg=sim_cfg(), cp=cp), cp


def part_sched(n_sched=N_SCHED, seed0=SEED_SCHED):
    payload, cp = _sched_payload()
    log('[sched] %d random FIFO schedules (seed %d + j), d_j in 1..19 steps, zoh 1 kHz, '
        '%d processes' % (n_sched, seed0, NPROC))
    jobs = [('seed%d' % (seed0 + j), 'random', seed0 + j, None) for j in range(n_sched)]
    t0 = time.time()
    res = pool_map(_job_sched, jobs, payload, label='sched', every=100)
    rows = OrderedDict(res)
    agg = aggregate(rows, cp)
    agg['all_age_ok'] = all(r['age_ok'] for r in rows.values())
    worst = OrderedDict()
    for key, fn in (('settling', max), ('marker_abs_max', max), ('gap_abs_max', max),
                    ('entry_obs', max), ('gmin_min', min), ('mmin_lim_min', min),
                    ('u_max_pre', max), ('u_max_pre_informed', max)):
        v, nm = arg_ext(list(rows.items()), key, fn)
        if nm is not None:
            r = rows[nm]
            worst[key] = dict(run=nm, value=v, settling=r['settling'],
                              marker_abs_max=r['marker_abs_max'],
                              gap_abs_max=r['gap_abs_max'], entry_obs=r['entry_obs'],
                              bounds_ok=r['bounds_ok'], schedule=r['schedule'])
    agg['worst_cases'] = worst
    log('  aggregate: dispersion %.3f..%.3f ms (bound %.2f ms), marker <= %.4f m (%.4f), gap '
        '<= %.4f m (%.4f), entry <= %.2f s (%.2f), u_max_pre %.5f (informed %.5f), mods %d, '
        'sat %d, npos %d, bounds_ok %s (%d failed %s), age_ok %s, fb_law %s, stop orders %s '
        '(%.0f s)'
        % (1e3 * agg['settling_min']['value'], 1e3 * agg['settling']['value'],
           1e3 * agg['certified']['settling_bound'], agg['marker_abs_max']['value'],
           agg['certified']['align'], agg['gap_abs_max']['value'],
           agg['certified']['gaperr_max'], agg['entry_obs']['value'],
           agg['certified']['T_ent_bar'], agg['u_max_pre']['value'],
           agg['u_max_pre_informed']['value'],
           agg['filter_modifications_total'], agg['saturations_total'],
           agg['positive_nominal_total'], agg['bounds_all_ok'], agg['n_bounds_failed'],
           agg['bounds_failed_rows'], agg['all_age_ok'], agg['fb_law_all'],
           agg['stop_orders'], time.time() - t0))
    out = OrderedDict(
        meta=meta('schedules', seeds=[seed0, seed0 + n_sched - 1],
                  options=dict(n=n_sched, dmax_steps=19, segments_s=[0.5, 5.0], scheme='zoh',
                               h=H_ZOH)),
        study=('random FIFO schedules: piecewise-constant per-hop transport (segments '
               '0.5 to 5 s, d_j uniform in 1..19 steps of T_m = 1 ms, one generator per '
               'schedule, seed %d + j; random_schedule of run_caseD_validate), FIFO '
               'projection at the receiver, held age <= 20 ms audited; FB digital law '
               "(sim_hr scheme 'zoh', T_u = 1 ms, nosched park) at h = 0.001 s" % seed0),
        certificate=cert_summary(cp), aggregate=agg, rows=rows,
        runtime_s=time.time() - t0)
    dump('schedules.json', out)
    return out


# ---- constructed schedules
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


def step_table(S, Nsend, mode, hops, others_val, dmax=19, dbar_ms=20):
    """adversarial pulse pattern of the reference steps (Lemma 7.1): the
    step J of the head reference at t_J (h_J > 0: the planned deceleration
    falls) reaches hop i (sender unit i-1, receiver i) through the relayed
    commands within [t_J, t_J + (i-1) dbar] and enters w^p_i as a pulse whose
    length is the age of the hop at that time.  mode 'short': short ages (1)
    in the windows [t_J - dbar, t_J + (i-1) dbar + 2 ms] (the leading dbar
    makes the short ages FIFO-feasible) and long ages (dmax) elsewhere, i.e.
    while the planned deceleration is constant (including the hardest
    deceleration right after takeover); mode 'long': the reverse.  The
    patterned hops get the pattern, the other hops others_val."""
    n = S.n
    D = np.full((n, Nsend), others_val, np.int64)
    D[0] = 1
    for i in hops:
        w = np.zeros(Nsend, bool)
        for (tJ, hJ) in ref_jumps(S):
            a = int(round(tJ * 1000)) - dbar_ms
            b = int(round(tJ * 1000)) + (i - 1) * dbar_ms + 2
            w[max(a, 0):min(b, Nsend - 1) + 1] = True
        if mode == 'short':
            D[i] = np.where(w, 1, dmax)
        else:
            D[i] = np.where(w, dmax, 1)
    return D


def adv_specs(S, Rnom, dmax=19):
    """constructed schedule families (transport in steps of T_m)."""
    n = S.n
    T_xi = S.T_xi
    Nsend = n_gen(S)
    tact = [float(x) for x in Rnom['tact']]
    tset = [float(x) for x in Rnom['tset']]
    J = ref_jumps(S)
    assert J and all(h_ > 0 for (_, h_) in J), 'every step of the shaped reference rises'
    jumps = sorted(set(t for (t, h_) in J))
    ev_sets = OrderedDict(jumps=jumps, switch=sorted(tact[1:] + tset), txi=[T_xi],
                          takeover=[0.0])
    ev_sets['all'] = sorted(set(ev_sets['jumps'] + ev_sets['switch'] + [T_xi, 0.0]))
    specs = []
    meta_ = OrderedDict()

    def add(nm, D, m):
        specs.append((nm, D))
        meta_[nm] = m
    # adversarial pulse patterns of the reference steps: globally and per hop
    for mode, rule in (('short', 'short ages (1 ms) while the planned deceleration falls '
                                 '(windows around the reference steps), long ages (19 ms) '
                                 'while it is constant'),
                       ('long', 'long ages (19 ms) while the planned deceleration falls, '
                                'short ages (1 ms) while it is constant')):
        add('steps_%s_global' % mode, step_table(S, Nsend, mode, range(1, n), dmax, dmax),
            dict(family='step patterns', scope='global', mode=mode, rule=rule))
        for hop in range(1, n):
            for ov in (1, dmax):
                add('steps_%s_hop%d_others%d' % (mode, hop, ov),
                    step_table(S, Nsend, mode, [hop], ov, dmax),
                    dict(family='step patterns', scope='hop %d' % hop, mode=mode,
                         others=ov, rule=rule))
    for nm, val in (('const_max', dmax), ('const_min', 1)):
        d = np.full((n, Nsend), val, np.int64)
        d[0] = 1
        add(nm, d, dict(family='constant', d=val))
    # bursts: alternating blocks of dmax and 1 of random length 5..200 sends
    for seed in range(1, 11):
        rng = np.random.default_rng(900 + seed)
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            j = 0
            val = dmax if rng.random() < 0.5 else 1
            while j < Nsend:
                L = int(rng.integers(5, 201))
                d[i, j:j + L] = val
                val = 1 if val == dmax else dmax
                j += L
        add('burst_s%d' % seed, d, dict(family='bursts', seed=900 + seed))
    # windows: maximal transport around, after or before the events
    for es, evs in ev_sets.items():
        for shape in ('around', 'after', 'before'):
            for W in (0.02, 0.1, 0.5):
                win = {i: _windows(evs, W, shape) for i in range(1, n)}
                add('max_%s_%s_W%gms' % (shape, es, W * 1e3), _sched_windows(Nsend, win, n, dmax),
                    dict(family='windows', events=es, shape=shape, W=W, n_events=len(evs)))
    win = {i: _windows([tset[i - 1], tact[i], tset[i]], 0.1, 'around') for i in range(1, n)}
    add('max_around_hop_switch_W100ms', _sched_windows(Nsend, win, n, dmax),
        dict(family='per-hop windows', events='t^sw_{i-1}, t^act_i, t^sw_i'))
    # sawtooth: FIFO batches of dmax packets, held age through 1..dmax+1
    for ph in (0, 4, 8, 12):
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            p = (ph + 3 * i) % dmax
            d[i] = dmax - ((np.arange(Nsend) + p) % dmax)
        add('sawtooth_ph%d' % ph, d, dict(family='sawtooth batches', phase=ph))
    # alternating 1 / dmax per send
    for nm, shift in (('alternating', False), ('alternating_shift', True)):
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            par = (np.arange(Nsend) + (i if shift else 0)) % 2
            d[i] = np.where(par == 0, 1, dmax)
        add(nm, d, dict(family='alternating', per_hop_phase=shift))
    # hop patterns
    for par in (0, 1):
        d = np.ones((n, Nsend), np.int64)
        for i in range(1, n):
            d[i] = dmax if i % 2 == par else 1
        add('hops_%s_max' % ('even' if par == 0 else 'odd'), d, dict(family='per-hop constant'))
    return specs, meta_, ev_sets


def part_adv():
    payload, cp = _sched_payload()
    S = design_setup()
    Rnom = simulate(S, 'zoh')
    Tnom = traj(Rnom, S)
    payload['e_ref'] = Tnom['e']
    payload['eps_ref'] = Tnom['eps']
    specs, meta_, ev_sets = adv_specs(S, Rnom, 19)
    log('[adv] %d constructed schedules, zoh 1 kHz, %d processes' % (len(specs), NPROC))
    jobs = [(nm, 'table', D.astype(np.int8), meta_[nm]) for nm, D in specs]
    t0 = time.time()
    res = pool_map(_job_sched, jobs, payload, label='adv', every=25)
    rows = OrderedDict(res)
    agg = aggregate(rows, cp)
    agg['all_age_ok'] = all(r['age_ok'] for r in rows.values())
    fam = OrderedDict()
    for nm, r in rows.items():
        f = meta_[nm]['family']
        g = fam.setdefault(f, dict(n=0, settling_max=0.0, marker_max=0.0, gap_max=0.0,
                                   entry_max=0.0, u_max_pre=-np.inf, u_max_pre_informed=-np.inf,
                                   gmin_min=np.inf, mmin_lim_min=np.inf, bounds_ok=True,
                                   n_pos_nom=0, filter_modifications=0, rule_before_T_xi=0,
                                   worst_settling_run=None))
        g['n'] += 1
        if r['settling'] is not None and r['settling'] > g['settling_max']:
            g['settling_max'] = r['settling']
            g['worst_settling_run'] = nm
        g['marker_max'] = max(g['marker_max'], r['marker_abs_max'] or 0.0)
        g['gap_max'] = max(g['gap_max'], r['gap_abs_max'] or 0.0)
        g['entry_max'] = max(g['entry_max'], r['entry_obs'] or 0.0)
        g['u_max_pre'] = max(g['u_max_pre'], r['u_max_pre'])
        if r['u_max_pre_informed'] is not None:
            g['u_max_pre_informed'] = max(g['u_max_pre_informed'], r['u_max_pre_informed'])
        g['gmin_min'] = min(g['gmin_min'], r['gmin_min'])
        g['mmin_lim_min'] = min(g['mmin_lim_min'], r['mmin_lim_min'])
        g['bounds_ok'] = g['bounds_ok'] and r['bounds_ok']
        g['n_pos_nom'] += r['n_pos_nom']
        g['filter_modifications'] += r['filter_modifications']
        g['rule_before_T_xi'] += int(bool(r['rule_before_T_xi']))
    for f, g in fam.items():
        log('  %-18s n=%3d disp <= %.3f ms marker <= %.4f m gap <= %.4f m entry <= %.2f s '
            'u_max_pre %.5f (informed %.5f) gmin %.3f m mmin %.3f npos %d mods %d bounds_ok %s'
            % (f, g['n'], 1e3 * g['settling_max'], g['marker_max'], g['gap_max'],
               g['entry_max'], g['u_max_pre'], g['u_max_pre_informed'], g['gmin_min'],
               g['mmin_lim_min'], g['n_pos_nom'], g['filter_modifications'], g['bounds_ok']))
    pulse = OrderedDict((nm, dict(settling=r['settling'], marker_abs_max=r['marker_abs_max'],
                                  gap_abs_max=r['gap_abs_max'], entry_obs=r['entry_obs'],
                                  e_hnd_max=r['e_hnd_max'], u_max_pre=r['u_max_pre'],
                                  u_max_pre_informed=r['u_max_pre_informed'],
                                  e_closure_hi=r['e_closure_hi'], e_closure_lo=r['e_closure_lo'],
                                  eps_closure_hi=r['eps_closure_hi'],
                                  eps_closure_lo=r['eps_closure_lo'],
                                  e_s2_abs_max=r['e_s2_abs_max'],
                                  de_closure_hi=r['de_closure_hi'],
                                  de_closure_lo=r['de_closure_lo'],
                                  deps_closure_hi=r['deps_closure_hi'],
                                  deps_closure_lo=r['deps_closure_lo'],
                                  slack_s2_position=r['slack'].get('s2_position_error'),
                                  slack_command_upper=r['slack'].get('command_upper_before_T_xi'),
                                  bounds_ok=r['bounds_ok']))
                        for nm, r in rows.items() if meta_[nm]['family'] == 'step patterns')
    rc = rows['const_max']
    pulse['const_max (reference)'] = dict(e_closure_hi=rc['e_closure_hi'],
                                          e_closure_lo=rc['e_closure_lo'],
                                          eps_closure_hi=rc['eps_closure_hi'],
                                          eps_closure_lo=rc['eps_closure_lo'],
                                          e_s2_abs_max=rc['e_s2_abs_max'],
                                          de_closure_hi=rc['de_closure_hi'],
                                          de_closure_lo=rc['de_closure_lo'])
    for nm, p in pulse.items():
        log('  %-30s e - e(const 19 ms) before T_xi: hi %s lo %s' % (
            nm, ['%+.4f' % x if x is not None else '-' for x in p['de_closure_hi']],
            ['%+.4f' % x if x is not None else '-' for x in p['de_closure_lo']]))
    pulse_bound = dict(P_e=cp['consts']['P_e'], P_eps=cp['consts']['P_eps'],
                       note='sup-norms of the certified planned forced response of every '
                            'pair (Lemma 7.1, cell form) at 20 ms')
    log('  aggregate: dispersion <= %.3f ms (%s), marker <= %.4f m, gap <= %.4f m, entry <= '
        '%.2f s, u_max_pre %.5f, mods %d, npos %d, bounds_ok %s (%d failed %s), age_ok %s '
        '(%.0f s)'
        % (1e3 * agg['settling']['value'], agg['settling']['run'], agg['marker_abs_max']['value'],
           agg['gap_abs_max']['value'], agg['entry_obs']['value'], agg['u_max_pre']['value'],
           agg['filter_modifications_total'], agg['positive_nominal_total'],
           agg['bounds_all_ok'], agg['n_bounds_failed'],
           agg['bounds_failed_rows'], agg['all_age_ok'], time.time() - t0))
    out = OrderedDict(
        meta=meta('adversarial', options=dict(dmax_steps=19, scheme='zoh', h=H_ZOH,
                                              n_schedules=len(specs))),
        study=('constructed FIFO schedules (transport 1..19 steps of T_m = 1 ms, held age '
               '<= 20 ms): the shaped reference decelerates hardest right after takeover '
               '(-0.728 m/s^2 on [0, 3.25 s)) and its rate then falls in 51 steps of '
               '0.25 s to the crawl rate (15.75 s); step patterns (Lemma 7.1): short ages '
               'while the planned deceleration falls (windows [t_J - dbar, t_J + (i-1) dbar '
               '+ 2 ms] around every step on hop i) and long ages while it is constant, and '
               'the reverse, globally and per hop (the other hops at 1 or 19 ms); constant '
               '1 and 19 ms; bursts; maximal transport in windows around, after or before '
               'the steps, the activations and S2 switches of the nominal run, the takeover '
               'and T_xi, or all of them (W = 20, 100, 500 ms), minimal elsewhere; per-hop '
               'switch windows; sawtooth FIFO batches; alternating 1/19 ms per send; per-hop '
               "constant patterns; FB digital law (sim_hr scheme 'zoh', T_u = 1 ms) at "
               'h = 0.001 s'),
        event_sets=ev_sets, certificate=cert_summary(cp), aggregate=agg, families=fam,
        step_patterns=pulse, planned_response_bound=pulse_bound, rows=rows,
        runtime_s=time.time() - t0)
    dump('adversarial.json', out)
    return out


# =============================================================== sweep
def _job_sweep(A):
    S = _wsetup()
    cp20 = _W['cp']
    db = A * 1e-3
    cfgc = cert_cfg(dbar=db)
    t0 = time.time()
    ev = cf.evaluate(cfgc, keep_curves=True)
    cp = cert_pack(ev, cfgc)
    t_cert = time.time() - t0
    T_end = round(S.T_xi + T_TAIL + 2 * (S.n - 1) * db, 3)
    N = int(round(T_end / T_M)) + 2
    runs = []
    specs = [('constant', None)]
    rng = np.random.default_rng(SEED_SWEEP + A)
    for j in range(N_SWEEP):
        specs.append(('random%d' % j, rcv.random_schedule(rng, N, A - 1)))
    for nm, D in specs:
        R, T, k, bt = run_rows(S, cp, 'zoh', D=D, transport=(A - 1) * 1e-3, T_end=T_end)
        bd = bounds_rows(cp20, k, T)
        r = compact(k, bt, cp if cp.get('bounds') is not None else cp20)
        r['name'] = nm
        r['bounds_at_age_ok'] = None if bt is None else bt['all_ok']
        r['bounds_at_age_failed'] = None if bt is None else bt['failed']
        r['n_checked_at_age'] = None if bt is None else bt['n_checked']
        r['n_rows_at_age'] = None if bt is None else bt['n_rows']
        r['design_bounds_ok'] = bd['all_ok']
        r['design_bounds_failed'] = bd['failed']
        r['age_ok'] = age_ok(k, db)
        r['schedule'] = None if D is None else dict(d_min=int(D[1:].min()),
                                                    d_max=int(D[1:].max()),
                                                    d_mean=float(D[1:].mean()))
        r['T_end'] = T_end
        runs.append(r)
    nonpos = ev['conds'].get('NONPOS', {})
    return A, _py(dict(cp=dict((k_, v) for k_, v in cp.items() if k_ != 'curves'),
                       runs=runs, t_cert=t_cert,
                       nonpos=dict(ok=nonpos.get('ok'), slack=nonpos.get('slack'))))


def _job_boundary(what):
    cfgc = cert_cfg()
    memo = {}
    if what == 'age':
        return what, _py(cf.age_boundary_ms(cfgc, kmax=200, memo=memo))
    return what, _py(cf.cond_boundaries_ms(cfgc, kmax=200, memo=memo))


def part_sweep(ages_ms=AGES_MS):
    t_part = time.time()
    cfgc = cert_cfg()
    cp20 = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    payload = dict(cfg=sim_cfg(), cp=cp20)
    log('[sweep] age bounds %s ms: certificate of the fixed design, constant transport and '
        '%d random schedules per age (zoh 1 kHz), %d processes' % (list(ages_ms), N_SWEEP, NPROC))
    t0 = time.time()
    res = pool_map(_job_sweep, list(ages_ms), payload, label='sweep', every=1)
    bres = pool_map(_job_boundary, ['age', 'cond'], None)
    bnd = dict(bres)['age']
    cb = dict(bres)['cond']
    S = design_setup()
    n = S.n
    rows = []
    for A, rr in res:
        cp = rr['cp']
        runs = rr['runs']
        const = runs[0]
        rand = runs[1:]
        allr = runs

        def ext(key, fn):
            return (_max if fn is max else _min)(r[key] for r in allr)
        B = cp.get('bounds')
        cert = OrderedDict(
            admitted=cp['admitted'], failed=cp['failed'], closed=cp['closed'],
            conds=cp['conds'], nonpos=rr['nonpos'],
            bounds=None if B is None else OrderedDict(
                settling_bound=B['dispersion'], T_f_bar=B['T_f_bar'], align=B['marker_max'],
                marker=B['marker'], gaperr_max=B['gap_max'], gap=B['gap'],
                T_ent_bar=B['entry_deadline'], clearance_floor_min=B['clearance_floor_min'],
                sigma_min=_min(B['admission_slack_sigma']),
                u_max_before_T_xi=B['u_max_before_T_xi'], schedule_error=B['schedule_error']),
            tlow_sw=None if cp.get('consts') is None else cp['consts'].get('tlow_sw'),
            dispersion_expression=disp_expression(cp, n), runtime_s=rr['t_cert'])
        summ = OrderedDict(
            settling_max=ext('settling', max), settling_min=ext('settling', min),
            settling_const=const['settling'],
            marker_max=ext('marker_abs_max', max), gap_max=ext('gap_abs_max', max),
            gmin_min=ext('gmin_min', min), hmin_min=ext('hmin_min', min),
            mmin_lim_min=ext('mmin_lim_min', min),
            e_hnd_max=ext('e_hnd_max', max), eps_hnd_max=ext('eps_hnd_max', max),
            u_min=ext('u_min', min), u_max=ext('u_max', max), u_max_pre=ext('u_max_pre', max),
            u_max_pre_informed=ext('u_max_pre_informed', max),
            unom_max=ext('unom_max', max),
            commands_in_set=bool(all(r['u_min'] >= -U_BR - TOL and r['u_max'] <= U_TR + TOL
                                     for r in allr)),
            filter_modifications=sum(r['filter_modifications'] for r in allr),
            min_branch=sum(r['n_mod'] for r in allr),
            fallback=sum(r['n_fallback'] for r in allr),
            saturations=sum(r['n_sat'] for r in allr),
            saturations_upper=sum(r['n_sat_hi'] for r in allr),
            saturations_lower=sum(r['n_sat_lo'] for r in allr),
            positive_nominal=sum(r['n_pos_nom'] for r in allr),
            runs_with_positive_nominal=sum(1 for r in allr if r['n_pos_nom'] > 0),
            runs_with_rule_before_T_xi=sum(1 for r in allr if r['rule_before_T_xi']),
            entry_max=_max(r['entry_obs'] for r in allr), entry_const=const['entry_obs'],
            all_stopped=all(r['all_stopped'] for r in allr),
            no_contact=all(r['contact_time'] is None for r in allr),
            n_contact=sum(1 for r in allr if r['contact_time'] is not None),
            head_first_tail_last=all(r['head_first_tail_last'] for r in allr),
            tail_first_head_last=all(r['tail_first_head_last'] for r in allr),
            stop_orders=sorted(set(tuple(r['stop_order'] or ()) for r in allr)),
            rule_entered_any=any(r['rule_entered'] for r in allr),
            requirements_ok=all(r['requirements']['all_ok'] for r in allr),
            requirements_failed=sorted(set(x for r in allr for x in r['requirements']['failed'])),
            bounds_at_age_ok=None if B is None else all(r['bounds_at_age_ok'] for r in allr),
            bounds_at_age_failed=None if B is None else sorted(set(
                x for r in allr for x in (r['bounds_at_age_failed'] or []))),
            design_bounds_ok=all(r['design_bounds_ok'] for r in allr),
            design_bounds_failed=sorted(set(x for r in allr for x in r['design_bounds_failed'])),
            fb_law_all=all(r['fb_law_ok'] is True for r in allr),
            age_ok=all(r['age_ok'] for r in allr))
        row = OrderedDict(age_ms=A, age_bound=A * 1e-3, transport_steps=A - 1,
                          certificate=cert, constant=const, random=rand, summary=summ)
        rows.append(row)
        s = summ
        log('  %4d ms cert %s %-30s | disp %.1f..%.1f ms (bound %s) marker <= %.4f m gap <= '
            '%.4f m gmin %.3f m u_max_pre %.5f npos %d sat %d mods %d entry <= %s contact %d '
            'bounds@age %s design %s'
            % (A, 'ADM' if cp['admitted'] else ('cl ' if cp['closed'] else '---'),
               ','.join(cp['failed'])[:30], 1e3 * (s['settling_min'] or float('nan')),
               1e3 * (s['settling_max'] or float('nan')),
               '%.1f ms' % (1e3 * B['dispersion']) if B else '-',
               s['marker_max'] or float('nan'), s['gap_max'] or float('nan'),
               s['gmin_min'], s['u_max_pre'], s['positive_nominal'], s['saturations'],
               s['filter_modifications'],
               '%.2f s' % s['entry_max'] if s['entry_max'] is not None else 'none',
               s['n_contact'], s['bounds_at_age_failed'] if s['bounds_at_age_ok'] is False
               else s['bounds_at_age_ok'], s['design_bounds_failed'] or 'ok'))
    first = OrderedDict()
    for row in rows:
        A = row['age_ms']
        s = row['summary']
        if 'certificate' not in first and not row['certificate']['admitted']:
            first['certificate'] = dict(age_ms=A, failed=row['certificate']['failed'])
        if 'bound_at_age' not in first and s['bounds_at_age_ok'] is False:
            first['bound_at_age'] = dict(age_ms=A, failed=s['bounds_at_age_failed'],
                                         admitted=row['certificate']['admitted'])
        if 'design_bound' not in first and not s['design_bounds_ok']:
            first['design_bound'] = dict(age_ms=A, failed=s['design_bounds_failed'])
        if 'requirement' not in first and not s['requirements_ok']:
            first['requirement'] = dict(age_ms=A, failed=s['requirements_failed'])
        if 'closure' not in first and not row['certificate']['closed']:
            first['closure'] = dict(age_ms=A)
        if 'clipped_command' not in first and s['saturations'] > 0:
            first['clipped_command'] = dict(age_ms=A, saturations=s['saturations'],
                                            upper=s['saturations_upper'],
                                            lower=s['saturations_lower'])
        if 'filter_action' not in first and (s['min_branch'] + s['fallback']) > 0:
            first['filter_action'] = dict(age_ms=A, min_branch=s['min_branch'],
                                          fallback=s['fallback'])
        if 'clip_or_filter' not in first and s['filter_modifications'] > 0:
            first['clip_or_filter'] = dict(age_ms=A, count=s['filter_modifications'])
        if 'positive_nominal' not in first and s['positive_nominal'] > 0:
            first['positive_nominal'] = dict(age_ms=A, runs=s['runs_with_positive_nominal'])
        if 'rule_before_T_xi' not in first and s['runs_with_rule_before_T_xi'] > 0:
            first['rule_before_T_xi'] = dict(age_ms=A, runs=s['runs_with_rule_before_T_xi'])
        if 'contact' not in first and s['n_contact'] > 0:
            first['contact'] = dict(age_ms=A, runs=s['n_contact'])
    first_req = OrderedDict()
    for row in rows:
        for nm in row['summary']['requirements_failed']:
            if nm not in first_req:
                prev = [r['age_ms'] for r in rows if r['age_ms'] < row['age_ms']]
                first_req[nm] = dict(age_ms=row['age_ms'],
                                     last_age_holding_ms=max(prev) if prev else None)
    first['requirement_each'] = first_req
    adm = [r['age_ms'] for r in rows if r['certificate']['admitted']]
    kb = bnd['k_ms']
    out = OrderedDict(
        meta=meta('sweep', seeds='%d + age' % SEED_SWEEP,
                  options=dict(ages_ms=list(ages_ms), n_random=N_SWEEP, scheme='zoh',
                               h=H_ZOH, horizon='T_xi + %g s + 2 (n-1) dbar' % T_TAIL,
                               boundary_kmax_ms=200)),
        study=('age sweep: certificate of the fixed FB design (certify_fb, lam %g, V^hnd 1e-2, '
               'lags 41, 41; reference and gain fixed at the design) at every age bound; '
               'constant transport age - T_m (held age exactly the age bound) and %d random '
               'schedules per age with d_j in 1..age/T_m - 1 (seed %d + age); FB digital law '
               '(zoh, T_u = 1 ms) at h = 0.001 s; horizon T_xi + %g s + 2 (n-1) dbar; '
               'observations against the certificate at that age (when its recursion '
               'closes; certified only where admitted) and against the design certificate '
               'at 20 ms; requirements: dispersion <= 1.0 s, marker error <= 3 m, terminal '
               'gap error <= 1 m, no filter modification or saturation, no contact, all '
               'stopped' % (LAM, N_SWEEP, SEED_SWEEP, T_TAIL)),
        ages_ms=list(ages_ms), rows=rows, certified_ages_ms=adm,
        boundary_ms=bnd, cond_boundary_ms=cb, first_failures=first,
        boundary_equals_design_record=bool(kb == AGE_BOUNDARY_MS),
        admitted_grid_consistent_with_boundary=bool(
            adm == [A for A in ages_ms if A <= kb]),
        admitted_rows_inside_bounds=all(r['summary']['bounds_at_age_ok'] for r in rows
                                        if r['certificate']['admitted']),
        admitted_rows_all_checked=all(x['n_checked_at_age'] == x['n_rows_at_age']
                                      for r in rows if r['certificate']['admitted']
                                      for x in [r['constant']] + r['random']),
        admitted_rows_filter_inactive=all(r['summary']['filter_modifications'] == 0
                                          for r in rows if r['certificate']['admitted']),
        admitted_rows_requirements_ok=all(r['summary']['requirements_ok'] for r in rows
                                          if r['certificate']['admitted']),
        admitted_rows_commands_in_set=all(r['summary']['commands_in_set'] for r in rows
                                          if r['certificate']['admitted']),
        all_rows_commands_in_set=all(r['summary']['commands_in_set'] for r in rows),
        all_rows_fb_law=all(r['summary']['fb_law_all'] for r in rows),
        all_age_ok=all(r['summary']['age_ok'] for r in rows),
        runtime_s=time.time() - t_part)
    log('  boundary %s ms (next fails %s); admitted ages on the grid %s; first failures %s'
        % (kb, bnd.get('failed_next'), adm, json.dumps(_py({k_: v for k_, v in first.items()
                                                            if k_ != 'requirement_each'}))))
    dump('sweep.json', out)
    return out


# ============================================================ ensemble
def ens_draw(rng, ranges=ENS_RANGES):
    """one takeover state: clearances c_i(0) on the 1 mm grid (decimal
    strings), head speed and speed increments (4 decimals)."""
    c0 = ['%.3f' % (int(rng.integers(ranges['c0_mm'][0], ranges['c0_mm'][1] + 1)) / 1000.0)
          for _ in range(4)]
    v1 = round(float(rng.uniform(*ranges['v1'])), 4)
    inc = [round(float(rng.uniform(*r)), 4) for r in ranges['increments']]
    v0 = [v1]
    for x in inc:
        v0.append(round(v0[-1] + x, 4))
    return c0, ['%.4f' % x for x in v0], inc


def ens_e0(c0):
    d_s = cf.FB_KEYS['d_s']
    return ['0'] + [str(cq.F(c0[i - 1]) - cq.F(d_s[i])) for i in range(1, 5)]


def _job_ens(job):
    idx, c0, v0, inc = job
    e0 = ens_e0(c0)
    t0 = time.time()
    try:
        D = rfd.shape(LAM, extra=dict(c0=list(c0), v0=list(v0), e0=e0))
    except Exception as ex:                                  # pragma: no cover
        return _py(dict(index=idx, c0=c0, v0=v0, error='design rule: %r' % ex, C2_ok=False))
    t_rule = time.time() - t0
    cfg = D['cfg']
    t1 = time.time()
    try:
        ev = cf.evaluate(cfg, keep_curves=True)
    except Exception as ex:                                  # pragma: no cover
        return _py(dict(index=idx, c0=c0, v0=v0, error='certificate: %r' % ex, C2_ok=False))
    cp = cert_pack(ev, cfg)
    t_cert = time.time() - t1
    rule = OrderedDict(converged=bool(D['converged']), iterations=len(D['hist']),
                       T_xi=float(D['T_xi']), T_xi_exact=str(D['T_xi']),
                       a_r0=float(D['pieces'][0][1]), a_cr=float(D['a_cr']),
                       n_pieces=len(D['pieces']), v_hand=float(D['v_hand']),
                       plan=rfd.plan_record(D), runtime_s=t_rule)
    res = OrderedDict(index=idx, c0=c0, v0=v0, increments=inc, e0=e0,
                      e0_float=[float(cq.F(x)) for x in e0], rule=rule,
                      eps0=cp['consts']['eps0'] if cp.get('consts') else None,
                      C2_ok=bool(cp['conds']['C2']['ok']),
                      C2_slack=cp['conds']['C2']['slack'],
                      certificate=cert_summary(cp),
                      nonpos=dict(ok=ev['conds']['NONPOS']['ok'],
                                  slack=_f(ev['conds']['NONPOS']['slack'])),
                      t_cert=t_cert)
    if not res['C2_ok']:
        return _py(res)
    S = setup_of(cfg)
    res['T_xi'] = S.T_xi
    res['v_c1'] = S.v_c1
    runs = OrderedDict()
    N = n_gen(S)
    specs = [('constant', None),
             ('random', rcv.random_schedule(np.random.default_rng(SEED_ENS_SCHED + idx), N, 19))]
    for nm, Dq in specs:
        R, T, k, bt = run_rows(S, cp, 'zoh', D=Dq)
        r = compact(k, bt, cp)
        r['age_ok'] = age_ok(k, DBAR)
        r['seed'] = None if Dq is None else SEED_ENS_SCHED + idx
        runs[nm] = r
    res['runs'] = runs
    res['runtime_s'] = time.time() - t0
    return _py(res)


def _worst_slack_runs(rs):
    ws = OrderedDict()
    for r in rs:
        for rn, x in r['runs'].items():
            for nm, v in (x.get('slack') or {}).items():
                if v is None:
                    continue
                if nm not in ws or v < ws[nm]['slack']:
                    ws[nm] = dict(slack=v, state=r['index'], run=rn)
    return ws


def ens_group(rs, eps_e, eps_v):
    """aggregate of a group of ensemble states: counts, observation ranges,
    certified bound ranges, ratios, worst slacks."""
    if not rs:
        return None
    runs = [x for r in rs for x in r['runs'].values()]
    C = [r['certificate'] for r in rs]
    return OrderedDict(
        n=len(rs), n_runs=len(runs),
        bounds_all_ok=all(x['bounds_ok'] for x in runs),
        n_bounds_failed=sum(1 for x in runs if not x['bounds_ok']),
        bounds_failed=sorted(set(y for x in runs for y in x['bounds_failed'])),
        all_rows_checked=all(x['n_checked'] == x['n_rows'] for x in runs),
        requirements_all_ok=all(x['requirements']['all_ok'] for x in runs),
        requirements_failed=sorted(set(y for x in runs for y in x['requirements']['failed'])),
        filter_modifications_total=sum(x['filter_modifications'] for x in runs),
        runs_with_filter_action=sum(1 for x in runs if (x['n_mod'] + x['n_fallback']) > 0),
        saturations_total=sum(x['n_sat'] for x in runs),
        positive_nominal_total=sum(x['n_pos_nom'] for x in runs),
        runs_with_positive_nominal=sum(1 for x in runs if x['n_pos_nom'] > 0),
        runs_with_rule_before_T_xi=sum(1 for x in runs if x['rule_before_T_xi']),
        fb_law_all=all(x['fb_law_ok'] is True for x in runs),
        all_stopped=all(x['all_stopped'] for x in runs),
        no_contact=all(x['contact_time'] is None for x in runs),
        age_ok=all(x['age_ok'] for x in runs),
        n_handoff_box_violated=sum(1 for x in runs if x['e_hnd_max'] > eps_e
                                   or x['eps_hnd_max'] > eps_v),
        observed=OrderedDict(
            settling=_rng(x['settling'] for x in runs),
            T_f=_rng(x['T_f'] for x in runs),
            marker=_rng(x['marker_abs_max'] for x in runs),
            gap=_rng(x['gap_abs_max'] for x in runs),
            entry=_rng(x['entry_obs'] for x in runs),
            gmin_min=_rng(x['gmin_min'] for x in runs),
            mmin_lim_min=_rng(x['mmin_lim_min'] for x in runs),
            u_min=_rng(x['u_min'] for x in runs),
            u_max_pre=_rng(x['u_max_pre'] for x in runs),
            u_max_pre_informed=_rng(x['u_max_pre_informed'] for x in runs),
            e_hnd_max=_rng(x['e_hnd_max'] for x in runs),
            eps_hnd_max=_rng(x['eps_hnd_max'] for x in runs)),
        certified=OrderedDict(
            T_xi=_rng(c.get('T_xi') for c in C),
            a_r0=_rng(c.get('a_r0') for c in C),
            a_crawl=_rng(c.get('a_crawl') for c in C),
            T_f_bar=_rng(c.get('T_f_bar') for c in C),
            settling_bound=_rng(c.get('settling_bound') for c in C),
            align=_rng(c.get('align') for c in C),
            gaperr_max=_rng(c.get('gaperr_max') for c in C),
            T_ent_bar=_rng(c.get('T_ent_bar') for c in C),
            clearance_floor_min=_rng(c.get('clearance_floor_min') for c in C),
            sigma_min=_rng(_min(c.get('sigma') or []) for c in C),
            u_max_before_T_xi=_rng(_max(c.get('u_max_before_T_xi') or []) for c in C),
            vfloor_min=_rng(c.get('vfloor_min') for c in C)),
        ratio_max=OrderedDict(
            (key, _max(x['ratio'][key] for x in runs if x.get('ratio')))
            for key in ('settling', 'marker', 'gap', 'entry')),
        rule=OrderedDict(converged=sum(1 for r in rs if r['rule']['converged']),
                         not_converged=[r['index'] for r in rs if not r['rule']['converged']],
                         iterations=_rng(r['rule']['iterations'] for r in rs),
                         runtime_s=_rng(r['rule']['runtime_s'] for r in rs)),
        c0_range=[min(float(x) for r in rs for x in r['c0']),
                  max(float(x) for r in rs for x in r['c0'])],
        e0_range=[min(x for r in rs for x in r['e0_float'][1:]),
                  max(x for r in rs for x in r['e0_float'][1:])],
        v1_range=[min(float(r['v0'][0]) for r in rs), max(float(r['v0'][0]) for r in rs)],
        nonpos_certified=sum(1 for r in rs if r['nonpos']['ok']),
        head_first_tail_last_all=all(x['head_first_tail_last'] for x in runs),
        stop_orders=sorted(set(tuple(x['stop_order'] or ()) for x in runs)),
        rule_entered_any=any(x['rule_entered'] for x in runs),
        worst_slack=_worst_slack_runs(rs))


def part_ens(n_draw=N_ENS, seed=SEED_ENS):
    t_part = time.time()
    cfg0 = cert_cfg()
    # the rule at the nominal state reproduces the design reference
    t0 = time.time()
    c0n = ['%.3f' % float(x) for x in cfg0['c0']]
    v0n = ['%.4f' % float(x) for x in cfg0['v0']]
    Dn = rfd.shape(LAM, extra=dict(c0=c0n, v0=v0n, e0=ens_e0(c0n)))
    same = bool(rfd.plan_record(Dn) == _DREC['selected']['plan'])
    log('[ens] %d takeover states inside C2: c_i(0) in [47, 53] m (1 mm grid), v_1(0) in %s '
        'm/s, increments in %s m/s (seed %d); the design rule is re-run at lam = %g for every '
        'state; rule at the nominal state reproduces the design reference: %s (%.1f s)'
        % (n_draw, ENS_RANGES['v1'], ENS_RANGES['increments'], seed, LAM, same, time.time() - t0))
    assert same, 'the design rule does not reproduce the design reference'
    rng = np.random.default_rng(seed)
    t0 = time.time()
    rows = []
    rejected_c2 = []
    batch = 0
    while len(rows) < n_draw:
        m = int(1.25 * (n_draw - len(rows))) + 8
        jobs = []
        for _ in range(m):
            c0, v0, inc = ens_draw(rng)
            jobs.append((batch, c0, v0, inc))
            batch += 1
        res = pool_map(_job_ens, jobs, dict(), label='ens', every=50)
        for r in res:
            if len(rows) >= n_draw:
                break
            if r.get('C2_ok'):
                rows.append(r)
            else:
                rejected_c2.append(dict(index=r['index'], c0=r.get('c0'), v0=r.get('v0'),
                                        C2_slack=r.get('C2_slack'), error=r.get('error'),
                                        rule_converged=(r.get('rule') or {}).get('converged')))
    adm = [r for r in rows if r['certificate']['admitted']]
    rej = [r for r in rows if not r['certificate']['admitted']]
    fails = {}
    for r in rej:
        for nm in r['certificate']['failed']:
            fails[nm] = fails.get(nm, 0) + 1
    eps_e, eps_v = float(cfg0['eps_e']), float(cfg0['eps_v'])
    # admitted fraction against the takeover closure e_i(0) (largest pair)
    bins = [40.5, 41.5, 42.5, 43.5, 44.5, 45.5, 46.5]
    by_e0 = []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = [r for r in rows if a <= max(r['e0_float'][1:]) < b
               or (b == bins[-1] and max(r['e0_float'][1:]) == b)]
        by_e0.append(dict(max_e0_range=[a, b], n=len(sel),
                          admitted=sum(1 for r in sel if r['certificate']['admitted'])))
    agg = OrderedDict(n=n_draw, seed=seed, ranges=ENS_RANGES, n_admitted=len(adm),
                      n_rejected=len(rej), failures_among_rejected=fails,
                      n_draws=len(rows) + len(rejected_c2), n_rejected_by_C2=len(rejected_c2),
                      admitted_by_largest_takeover_error=by_e0,
                      admitted=ens_group(adm, eps_e, eps_v), rejected=ens_group(rej, eps_e, eps_v))
    log('  %d states (C2 rejected %d draws): admitted %d, rejected %d %s (%.0f s)'
        % (len(rows), len(rejected_c2), len(adm), len(rej), fails, time.time() - t0))
    log('  admitted by largest takeover error: %s' % by_e0)
    log('  admitted: %s' % json.dumps(_py({k_: v for k_, v in (agg['admitted'] or {}).items()
                                            if k_ != 'worst_slack'}))[:2500])
    if agg['rejected']:
        log('  rejected: %s' % json.dumps(_py({k_: v for k_, v in agg['rejected'].items()
                                                if k_ != 'worst_slack'}))[:2500])
    out = OrderedDict(
        meta=meta('ensemble', seeds=dict(states=seed, schedules='%d + index' % SEED_ENS_SCHED),
                  options=dict(n_states=n_draw, ranges=ENS_RANGES, runs_per_state=2,
                               scheme='zoh', h=H_ZOH, design_gain=LAM,
                               design_rule='run_fb_design.shape(lam, extra=dict(c0, v0, e0))')),
        study=('takeover-state ensemble inside C2: clearances c_i(0) uniform on the 1 mm grid '
               'in [47, 53] m (takeover errors e_i(0) = c_i(0) - 6.5 m in [40.5, 46.5] m), '
               'head speed v_1(0) uniform in [10.45, 10.55] m/s (>= v^in = 10.45 m/s), speed '
               'increments v_i(0) - v_{i-1}(0) uniform in %s m/s (4 decimals), seed %d; for '
               'every state the dispatcher re-runs the design rule of run_fb_design at the '
               'design gain %g (the fixed choices of the design record: T_p 0.25 s, mu 0.02, '
               'hold pad 0.5 s, handoff speed 1 m/s, T_pad 2 s, grids q_a, q_c, age bound '
               '20 ms) with the state\'s c0, v0 and e0, then certifies the shaped reference '
               'at 20 ms; draws whose certificate violates C2 (eps_k(0) + Lambda^(1)_k <= 0) '
               'are redrawn; two runs per state (zoh 1 kHz: constant 19 ms and random '
               'schedule seed %d + index), compared with the state\'s certificate (certified '
               'only where admitted)' % (ENS_RANGES['increments'], seed, LAM, SEED_ENS_SCHED)),
        design_plan=_DREC['selected']['plan'],
        rule_at_nominal_state=dict(reproduces_design=same, c0=c0n, v0=v0n),
        aggregate=agg, rows=rows, rejected_by_C2=rejected_c2, runtime_s=time.time() - t_part)
    dump('ensemble.json', out)
    return out


# ============================================================= outside
def _job_out(job):
    pair, delta = job
    base = cert_cfg()
    v0n = [float(cq.F(x)) for x in base['v0']]
    inc = [round(v0n[i] - v0n[i - 1], 4) for i in range(1, len(v0n))]
    inc[pair - 2] = -round(delta, 4)
    v0 = [v0n[0]]
    for x in inc:
        v0.append(round(v0[-1] + x, 4))
    v0s = ['%.4f' % x for x in v0]
    cs = sim_cfg(v0=v0s)
    cc = cert_cfg(v0=v0s)
    ev = cf.evaluate(cc, keep_curves=False)
    cp = cert_pack(ev, cc, curves=False)
    ev_s = cf.evaluate(cert_cfg(v0=v0s, one_sided=False), want_entry=True)
    S = setup_of(cs)
    N = n_gen(S)
    specs = [('zoh_const19', 'zoh', None, TRANSPORT, None),
             ('zoh_const1', 'zoh', None, 1e-3, None),
             ('zoh_random', 'zoh', rcv.random_schedule(
                 np.random.default_rng(SEED_OUT_SCHED + 10 * pair + int(round(delta * 100))),
                 N, 19), None, None),
             ('cont_const19', 'cont', None, TRANSPORT, H_REF)]
    runs = OrderedDict()
    cp20 = _W['cp']
    for nm, scheme, D, tr, h in specs:
        R, T, k, _ = run_rows(S, None, scheme, D=D, transport=tr, h=h)
        bd = bounds_rows(cp20, k, T)
        r = compact(k, None, cp20)
        r['eps_sw'] = k['eps_sw']
        r['e_sw'] = k['e_sw']
        r['sat_counts'] = dict(upper=k['counts']['sat_upper'], lower=k['counts']['sat_lower'])
        r['time_saturated'] = k['time_spent']['saturation']
        r['time_positive_nominal'] = k['time_spent']['positive_nominal']
        r['min_branch_counts'] = k['counts']['min_branch']
        r['within_design_bounds'] = bd['all_ok']
        r['design_bounds_failed'] = bd['failed']
        r['age_ok'] = age_ok(k, DBAR)
        runs[nm] = r
    return _py(OrderedDict(
        pair=pair, delta=delta, increments=inc, v0=v0s, eps0=cp['consts']['eps0']
        if cp.get('consts') else None,
        certificate=dict(admitted=cp['admitted'], failed=cp['failed'],
                         C2_ok=cp['conds']['C2']['ok'], C2_slack=cp['conds']['C2']['slack'],
                         applicable=bool(cp['conds']['C2']['ok'])),
        certificate_symmetric_box=dict(admitted=bool(ev_s['admitted']), failed=ev_s['failed']),
        runs=runs))


def part_outside():
    t_part = time.time()
    cfgc = cert_cfg()
    cp20 = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    deltas = [round(0.01 * j, 2) for j in range(1, 11)]
    jobs = [(p, d) for p in range(2, 6) for d in deltas]
    log('[outside] %d takeover states outside C2 (pair p in 2..5 slower than its predecessor '
        'by %s m/s), design reference, 4 runs each, %d processes' % (len(jobs), deltas, NPROC))
    t0 = time.time()
    rows = pool_map(_job_out, jobs, dict(cfg=sim_cfg(), cp=cp20), label='outside', every=10)
    for r in rows:
        rr = r['runs']
        log('  pair %d delta %.2f: C2 %s (slack %+.4f), sym. box adm %s %s | ' % (
            r['pair'], r['delta'], r['certificate']['C2_ok'], r['certificate']['C2_slack'],
            r['certificate_symmetric_box']['admitted'],
            ','.join(r['certificate_symmetric_box']['failed'])[:40])
            + '; '.join('%s: disp %.1f ms sat %d/%d npos %d mods %d rule %s (before T_xi %s) '
                        'marker %.3f contact %s'
                        % (nm, 1e3 * x['settling'] if x['settling'] is not None else float('nan'),
                           x['n_sat_hi'], x['n_sat_lo'], x['n_pos_nom'], x['filter_modifications'],
                           x['rule_units'], x['rule_before_T_xi'],
                           x['marker_abs_max'] if x['marker_abs_max'] is not None
                           else float('nan'), x['contact_time'])
                        for nm, x in rr.items() if nm in ('zoh_const19', 'zoh_random')))
    runs = [x for r in rows for x in r['runs'].values()]
    agg = OrderedDict(
        n_states=len(rows), n_runs=len(runs),
        certificate_not_applicable_all=all(not r['certificate']['C2_ok'] for r in rows),
        admitted_any=any(r['certificate']['admitted'] for r in rows),
        symmetric_box_admitted=[[r['pair'], r['delta']] for r in rows
                                if r['certificate_symmetric_box']['admitted']],
        symmetric_box_failures=sorted(set(x for r in rows
                                          for x in r['certificate_symmetric_box']['failed'])),
        runs_with_clipping=sum(1 for x in runs if x['n_sat'] > 0),
        runs_with_upper_clipping=sum(1 for x in runs if x['n_sat_hi'] > 0),
        runs_with_positive_nominal=sum(1 for x in runs if x['n_pos_nom'] > 0),
        runs_with_filter_action=sum(1 for x in runs if (x['n_mod'] + x['n_fallback']) > 0),
        runs_with_rule_before_T_xi=sum(1 for x in runs if x['rule_before_T_xi']),
        runs_all_followers_rule=sum(1 for x in runs if x['rule_units'] == [2, 3, 4, 5]),
        runs_within_design_bounds=sum(1 for x in runs if x['within_design_bounds']),
        runs_not_stopped=sum(1 for x in runs if not x['all_stopped']),
        runs_with_contact=sum(1 for x in runs if x['contact_time'] is not None),
        runs_fb_law=sum(1 for x in runs if x['fb_law_ok'] is True),
        settling_max=_max(x['settling'] for x in runs),
        marker_max=_max(x['marker_abs_max'] for x in runs),
        gap_max=_max(x['gap_abs_max'] for x in runs),
        u_max=_max(x['u_max'] for x in runs), unom_max=_max(x['unom_max'] for x in runs),
        u_max_pre_informed=_max(x['u_max_pre_informed'] for x in runs),
        e_hnd_max=_max(x['e_hnd_max'] for x in runs),
        eps_hnd_max=_max(x['eps_hnd_max'] for x in runs),
        gmin_min=_min(x['gmin_min'] for x in runs),
        mmin_lim_min=_min(x['mmin_lim_min'] for x in runs),
        requirements_failed=sorted(set(y for x in runs for y in x['requirements']['failed'])),
        design_bounds_failed=sorted(set(y for x in runs for y in x['design_bounds_failed'])),
        stop_orders=sorted(set(tuple(x['stop_order'] or ()) for x in runs)),
        age_ok=all(x['age_ok'] for x in runs),
        first_clip_delta_by_pair={str(p): _min(r['delta'] for r in rows if r['pair'] == p
                                               and any(x['n_sat'] > 0 for x in r['runs'].values()))
                                  for p in range(2, 6)},
        first_positive_delta_by_pair={str(p): _min(r['delta'] for r in rows if r['pair'] == p
                                                   and any(x['n_pos_nom'] > 0
                                                           for x in r['runs'].values()))
                                      for p in range(2, 6)},
        first_filter_delta_by_pair={str(p): _min(r['delta'] for r in rows if r['pair'] == p
                                                 and any((x['n_mod'] + x['n_fallback']) > 0
                                                         for x in r['runs'].values()))
                                    for p in range(2, 6)},
        first_design_bound_failure_delta_by_pair={
            str(p): _min(r['delta'] for r in rows if r['pair'] == p
                         and any(not x['within_design_bounds'] for x in r['runs'].values()))
            for p in range(2, 6)})
    log('  aggregate: %s' % json.dumps(_py(agg))[:2000])
    out = OrderedDict(
        meta=meta('outside', seeds='%d + 10 p + 100 delta' % SEED_OUT_SCHED,
                  options=dict(deltas=deltas, pairs=[2, 3, 4, 5],
                               runs=['zoh_const19', 'zoh_const1', 'zoh_random', 'cont_const19'])),
        study=('takeover states outside C2: the speed increment v_p(0) - v_{p-1}(0) of one '
               'pair p in 2..5 set to -delta, delta in 0.01..0.10 m/s (the follower slower '
               'than its predecessor), the head speed and the other increments of the design '
               'kept, design reference (no dispatch: the certificate is not applicable, C2 '
               'fails; also reported: the certificate with the symmetric switch box); runs: '
               "zoh 1 kHz with constant 19 ms, constant 1 ms and one random schedule (seed %d "
               "+ 10 p + 100 delta), and 'cont' (h = %g s) with constant 19 ms; reported: "
               'RULE entries, positive and clipped commands, filter action, errors, and '
               'whether the run stays within the bounds of the design certificate '
               '(informational)' % (SEED_OUT_SCHED, H_REF)),
        aggregate=agg, rows=rows, runtime_s=time.time() - t_part)
    dump('outside.json', out)
    return out


# =========================================================== age audit
def hop_audit(d, Nsend):
    """held age of one hop on the grid of T_m (port of run_caseD_audit)."""
    d = np.asarray(d[:Nsend], np.int64)
    j = np.arange(Nsend, dtype=np.int64)
    raw = j + d
    arr = np.maximum.accumulate(raw)
    assert np.all(np.diff(arr) >= 0)
    K = int(arr[-1])
    k = np.arange(K + 1, dtype=np.int64)
    held = np.searchsorted(arr, k, side='right') - 1
    assert held[-1] == Nsend - 1 and np.all(np.diff(held) >= 0)
    assert np.all(held >= -1) and np.all(held <= k)
    age = k - held
    left = k[1:] - held[:-1]
    deliv = np.nonzero(np.diff(np.r_[-1, held]) > 0)[0]
    first = int(deliv[0])
    assert first == int(arr[0])
    later = deliv[deliv > first]
    left_later = later - held[later - 1]
    at_deliv = deliv - held[deliv]
    batch = np.diff(np.r_[-1, held[deliv]])
    return dict(
        max_left=int(left_later.max()) if len(left_later) else None,
        max_left_with_startup=int(max(left.max(), first + 1)),
        max_at_delivery=int(at_deliv.max()),
        max_age_on_grid=int(age[first:].max()),
        first_delivery=first, n_packets=int(Nsend), n_deliveries=int(len(deliv)),
        n_batches_multi=int(np.sum(batch > 1)), n_held_back=int(np.sum(arr > raw)),
        d_min=int(d.min()), d_max=int(d.max()))


def schedule_audit(D, Nsend, bound):
    hops = [hop_audit(D[i], Nsend) for i in range(1, D.shape[0])]
    out = dict(
        max_left=max(h['max_left'] for h in hops),
        max_left_with_startup=max(h['max_left_with_startup'] for h in hops),
        max_at_delivery=max(h['max_at_delivery'] for h in hops),
        hops_max_left=[h['max_left'] for h in hops],
        hops_reaching_bound=sum(1 for h in hops if h['max_left_with_startup'] == bound),
        first_delivery_max=max(h['first_delivery'] for h in hops),
        n_batches_multi=sum(h['n_batches_multi'] for h in hops),
        n_held_back=sum(h['n_held_back'] for h in hops),
        n_packets=sum(h['n_packets'] for h in hops))
    out['within_bound'] = bool(out['max_left_with_startup'] <= bound)
    return out


def matches(a, archived):
    """recomputed maxima [ms] against the simulator's audit of the run [s]."""
    return bool(abs(a['max_left'] * 1e-3 - archived['max_left_limit']) < 1e-12
                and abs(a['max_at_delivery'] * 1e-3 - archived['max_at_delivery']) < 1e-12
                and abs(a['max_left_with_startup'] * 1e-3
                        - archived['max_left_limit_with_notional']) < 1e-12)


def window(D_row, Nsend, s_, pre, post, kind):
    """packets and held age of one hop around send step s_ (port of
    run_caseD_audit.window)."""
    d = np.asarray(D_row[:Nsend], np.int64)
    j = np.arange(Nsend, dtype=np.int64)
    raw = j + d
    arr = np.maximum.accumulate(raw)
    lo, hi = s_ - pre, s_ + post
    k = np.arange(lo - 30, hi + 31, dtype=np.int64)
    held = np.searchsorted(arr, k, side='right') - 1
    sel = (arr >= lo - 1) & (j <= hi + 1)
    deliv = np.nonzero(np.diff(held) > 0)[0] + 1
    td = k[deliv]
    inw = (td >= lo) & (td <= hi)
    td = td[inw]
    h_after = held[deliv][inw]
    h_before = held[deliv - 1][inw]
    tt = []
    aa = []
    for t_, hb, ha in zip(td, h_before, h_after):
        tt += [int(t_), int(t_)]
        aa += [int(t_ - hb), int(t_ - ha)]
    h_lo = int(held[np.nonzero(k == lo)[0][0]])
    h_hi = int(held[np.nonzero(k == hi)[0][0]])
    tt = [int(lo)] + tt + [int(hi)]
    aa = [int(lo - h_lo)] + aa + [int(hi - h_hi)]
    return dict(
        kind=kind, send_step=int(s_), lo_ms=int(lo), hi_ms=int(hi),
        d_before_ms=int(d[s_ - 1]), d_after_ms=int(d[s_]),
        packets=dict(t_send_ms=[int(x) for x in j[sel]],
                     t_raw_arrival_ms=[int(x) for x in raw[sel]],
                     t_delivery_ms=[int(x) for x in arr[sel]],
                     held_back=[bool(x) for x in (arr[sel] > raw[sel])]),
        deliveries=dict(t_ms=[int(x) for x in td],
                        age_left_limit_ms=[int(x) for x in (td - h_before)],
                        age_after_ms=[int(x) for x in (td - h_after)],
                        batch_size=[int(x) for x in (h_after - h_before)]),
        age_polyline=dict(t_ms=tt, age_ms=aa),
        max_left_limit_ms=int(max(td - h_before)), max_age_after_ms=int(max(td - h_after)))


def _audit_group(items, bound_ms):
    """items: list of (name, D, Nsend, archived age dict or None)."""
    rows = OrderedDict()
    mism = []
    for nm, D, Nsend, arch in items:
        a = schedule_audit(D, Nsend, bound_ms)
        a['matches_run'] = None if arch is None else matches(a, arch)
        if arch is not None and not a['matches_run']:
            mism.append(nm)
        rows[nm] = a
    mls = [r['max_left_with_startup'] for r in rows.values()]
    return rows, dict(
        n_schedules=len(rows), max_left_limit_ms=max(mls) if mls else None,
        min_over_schedules_of_max_left_limit_ms=min(mls) if mls else None,
        schedules_reaching_bound=sum(1 for x in mls if x == bound_ms),
        max_at_delivery_ms=max(r['max_at_delivery'] for r in rows.values()) if rows else None,
        hops_reaching_bound=sum(r['hops_reaching_bound'] for r in rows.values()),
        min_over_hops_of_max_left_limit_ms=min(h for r in rows.values()
                                               for h in r['hops_max_left']) if rows else None,
        first_delivery_max_ms=max(r['first_delivery_max'] for r in rows.values()) if rows else None,
        batches_with_several_packets=sum(r['n_batches_multi'] for r in rows.values()),
        packets_held_back=sum(r['n_held_back'] for r in rows.values()),
        packets=sum(r['n_packets'] for r in rows.values()),
        all_within_bound=all(r['within_bound'] for r in rows.values()),
        mismatches_vs_runs=len(mism), mismatched=mism)


def _nsend(T_end, h=H_ZOH):
    N = int(round(T_end / h))
    Nm = int(round(T_M / h))
    return (N - 1) // Nm + 1


def part_age_audit():
    t0 = time.time()
    S = design_setup()
    n = S.n
    bound = int(round(DBAR * 1000))
    N = n_gen(S)
    NsendR = _nsend(S.T_END)
    REF = load('reference.json')
    CONV = load('convergence.json')
    DIG = load('digital.json')
    SCH = load('schedules.json')
    ADV = load('adversarial.json')
    SWP = load('sweep.json')
    ENS = load('ensemble.json')
    OUTS = load('outside.json')
    IND = load('independent.json')
    doc = OrderedDict(
        meta=meta('age_audit'),
        item='held information age of every delay schedule of the FB validation suite, '
             'recomputed from the schedules in the time domain and compared with the '
             "simulator's audit of every run (sim_hr: sim_cont.age_audit plus the notional "
             'packet before the first delivery)',
        T_m_ms=1, age_bound_ms=bound, sends_per_hop=NsendR, hops=n - 1,
        conventions=dict(
            fifo='a_j = max_{l <= j}(l + d_l); packets with the same a_j form one batch',
            startup='before the first delivery the receiver holds the zero command, a '
                    'packet of send index -1',
            age='after the deliveries at k: k - J(k); left limit at k: k - J(k - 1)'))
    totals = dict(schedules=0, runs_compared=0, mismatches=0, all_within_bound=True)

    def acc(summ):
        totals['schedules'] += summ['n_schedules']
        totals['mismatches'] += summ['mismatches_vs_runs']
        totals['all_within_bound'] = totals['all_within_bound'] and summ['all_within_bound']
    # ---- constant transport: the reference, convergence, digital, and every swept age
    const_items = []
    if REF is not None:
        const_items.append(('reference_h1e-4', np.full((n, NsendR), 19, np.int64), NsendR,
                            REF['kpis']['age']))
        const_items.append(('reference_fine_h2.5e-5', np.full((n, NsendR), 19, np.int64), NsendR,
                            REF['fine']['kpis']['age']))
    if CONV is not None:
        for r in CONV['rows']:
            const_items.append(('conv_h%g' % r['h'], np.full((n, NsendR), 19, np.int64),
                                NsendR, r['kpis']['age']))
    if DIG is not None:
        for r in DIG['rows']:
            if r['setting'] == 'const19':
                const_items.append(('digital_%s_%s' % (r['setting'], r['scheme']),
                                    np.full((n, NsendR), 19, np.int64), NsendR, r['kpis']['age']))
    crows, csum = _audit_group(const_items, bound)
    acc(csum)
    swc = OrderedDict()
    if SWP is not None:
        for row in SWP['rows']:
            A = int(row['age_ms'])
            T_end = row['constant']['T_end']
            Ns = _nsend(T_end)
            D = np.full((n, Ns), A - 1, np.int64)
            a = schedule_audit(D, Ns, A)
            ok = matches(a, row['constant']['age'])
            swc[str(A)] = dict(max_left_limit_ms=a['max_left_with_startup'],
                               max_at_delivery_ms=a['max_at_delivery'], matches_run=ok,
                               within_bound=a['within_bound'])
            totals['schedules'] += 1
            totals['mismatches'] += int(not ok)
            totals['all_within_bound'] = totals['all_within_bound'] and a['within_bound']
    doc['constant'] = dict(runs=crows, summary=csum,
                           ages_ms=[int(a) for a in swc], rows=swc,
                           note='the held age alternates between the transport delay and '
                                'the age bound')
    log('[age_audit] constant transport: %d runs + %d swept ages, within bound %s, '
        'mismatches %d' % (len(crows), len(swc), csum['all_within_bound'],
                           csum['mismatches_vs_runs'] + sum(1 for v in swc.values()
                                                            if not v['matches_run'])))
    # ---- random schedules at 20 ms
    keep = None
    items = []
    if SCH is not None:
        for nm, r in SCH['rows'].items():
            seed = int(nm[4:])
            D = rcv.random_schedule(np.random.default_rng(seed), N, 19)
            if seed == SEED_SCHED:
                keep = D.copy()
            sc_ = r['schedule']
            assert sc_['d_min'] == int(D[1:].min()) and sc_['d_max'] == int(D[1:].max())
            items.append((nm, D, NsendR, r['age']))
    rrows, rsum = _audit_group(items, bound)
    acc(rsum)
    doc['random'] = rsum
    log('  random: %d schedules, sup of the age %s ms, %d hops reach the bound, mismatches %d'
        % (rsum['n_schedules'], rsum['max_left_limit_ms'], rsum['hops_reaching_bound'],
           rsum['mismatches_vs_runs']))
    # ---- constructed schedules
    if ADV is not None:
        Rnom = simulate(S, 'zoh')
        specs, meta_, ev2 = adv_specs(S, Rnom, 19)
        assert [nm for nm, _ in specs] == list(ADV['rows'].keys())
        for key in ('jumps', 'switch', 'txi', 'all'):
            assert np.allclose(ev2[key], ADV['event_sets'][key], rtol=0, atol=1e-9), key
        items = [(nm, D, NsendR, ADV['rows'][nm]['age']) for nm, D in specs]
        arows, asum = _audit_group(items, bound)
        asum['families'] = sorted(set(m['family'] for m in meta_.values()))
        acc(asum)
        doc['searched'] = asum
        log('  constructed: %d schedules, sup of the age %s ms, mismatches %d'
            % (asum['n_schedules'], asum['max_left_limit_ms'], asum['mismatches_vs_runs']))
    # ---- sweep random schedules
    if SWP is not None:
        srows = OrderedDict()
        nr = 0
        mis = 0
        allw = True
        for row in SWP['rows']:
            A = int(row['age_ms'])
            T_end = row['constant']['T_end']
            Ng = int(round(T_end / T_M)) + 2
            Ns = _nsend(T_end)
            rng = np.random.default_rng(SEED_SWEEP + A)
            worst = 0
            reach = 0
            for jj in range(len(row['random'])):
                D = rcv.random_schedule(rng, Ng, A - 1)
                a = schedule_audit(D, Ns, A)
                ok = matches(a, row['random'][jj]['age'])
                mis += int(not ok)
                allw = allw and a['within_bound']
                worst = max(worst, a['max_left_with_startup'])
                reach += int(a['max_left_with_startup'] == A)
                nr += 1
            srows[str(A)] = dict(n=len(row['random']), max_left_limit_ms=worst,
                                 schedules_reaching_bound=reach)
        doc['sweep'] = dict(ages_ms=[int(r['age_ms']) for r in SWP['rows']], rows=srows,
                            n_schedules=nr, mismatches_vs_runs=mis, all_within_bound=allw)
        totals['schedules'] += nr
        totals['mismatches'] += mis
        totals['all_within_bound'] = totals['all_within_bound'] and allw
        log('  sweep: %d random schedules at %d ages, within bounds %s, mismatches %d'
            % (nr, len(srows), allw, mis))
    # ---- digital random, ensemble, outside, independent
    other = OrderedDict()
    if DIG is not None:
        items = []
        for r in DIG['rows']:
            if r['setting'] != 'const19':
                seed = int(r['setting'][4:])
                D = rcv.random_schedule(np.random.default_rng(seed), N, 19)
                items.append(('digital_%s_%s' % (r['setting'], r['scheme']), D, NsendR,
                              r['kpis']['age']))
        g, s_ = _audit_group(items, bound)
        other['digital_random'] = s_
        acc(s_)
    if ENS is not None:
        items = []
        for r in ENS['rows']:
            T_end = r['T_xi'] + T_TAIL
            Ns = _nsend(T_end)
            Ng = int(round(T_end / T_M)) + 2
            items.append(('ens%d_const' % r['index'], np.full((n, Ns), 19, np.int64), Ns,
                          r['runs']['constant']['age']))
            D = rcv.random_schedule(np.random.default_rng(SEED_ENS_SCHED + r['index']), Ng, 19)
            items.append(('ens%d_random' % r['index'], D, Ns, r['runs']['random']['age']))
        g, s_ = _audit_group(items, bound)
        other['ensemble'] = s_
        acc(s_)
    if OUTS is not None:
        items = []
        for r in OUTS['rows']:
            p, dl = r['pair'], r['delta']
            items.append(('out_p%d_d%.2f_const19' % (p, dl), np.full((n, NsendR), 19, np.int64),
                          NsendR, r['runs']['zoh_const19']['age']))
            items.append(('out_p%d_d%.2f_const1' % (p, dl), np.full((n, NsendR), 1, np.int64),
                          NsendR, r['runs']['zoh_const1']['age']))
            D = rcv.random_schedule(np.random.default_rng(SEED_OUT_SCHED + 10 * p
                                                          + int(round(dl * 100))), N, 19)
            items.append(('out_p%d_d%.2f_random' % (p, dl), D, NsendR,
                          r['runs']['zoh_random']['age']))
            items.append(('out_p%d_d%.2f_cont19' % (p, dl), np.full((n, NsendR), 19, np.int64),
                          NsendR, r['runs']['cont_const19']['age']))
        g, s_ = _audit_group(items, bound)
        other['outside'] = s_
        acc(s_)
    if IND is not None:
        items = []
        for nm, run in IND['runs'].items():
            for key in ('sim_hr_zoh_h0.001', 'sim_hr_cont_h0.0001'):
                a = run['results'].get(key)
                if a is None:
                    continue
                if run['seed'] is None:
                    D = np.full((n, NsendR), 19, np.int64)
                else:
                    D = rcv.random_schedule(np.random.default_rng(run['seed']), N, 19)
                items.append(('indep_%s_%s' % (nm, key), D, NsendR, a['age']))
        g, s_ = _audit_group(items, bound)
        other['independent'] = s_
        acc(s_)
    doc['other'] = other
    # ---- two windows of one hop of the first random schedule
    if keep is not None:
        hop = 4
        d = keep[hop]
        chg = np.nonzero(np.diff(d[:NsendR]))[0] + 1
        rise = [int(c) for c in chg if d[c] == 19 and d[c - 1] <= 5]
        fall = [int(c) for c in chg if d[c - 1] == 19 and d[c] <= 5]
        if rise and fall:
            doc['windows'] = dict(
                schedule='seed%d' % SEED_SCHED, hop_row=hop, sender_unit=hop,
                receiver_unit=hop + 1,
                rise=window(d, NsendR, rise[0], 12, 38, 'transport delay rises'),
                fall=window(d, NsendR, fall[0], 11, 39, 'transport delay falls'))
    doc['totals'] = dict(schedules=totals['schedules'], all_within_bound=totals['all_within_bound'],
                         mismatches_vs_runs=totals['mismatches'])
    doc['all_ok'] = bool(totals['all_within_bound'] and totals['mismatches'] == 0)
    doc['runtime_s'] = round(time.time() - t0, 1)
    log('  totals: %d schedules, all within bound %s, mismatches with the runs %d (%.0f s)'
        % (totals['schedules'], totals['all_within_bound'], totals['mismatches'],
           time.time() - t0))
    dump('age_audit.json', doc)
    return doc


# ========================================================= independent
def _job_indep(job):
    variant, name, seed = job
    import sim_bench_hr as sbh
    S = design_setup()
    D = None if seed is None else rcv.random_schedule(np.random.default_rng(seed), n_gen(S), 19)
    t0 = time.time()
    B = sbh.setup(sim_cfg(), lam=LAM, U=U_TR, Uminus=U_BR, eps_e=S.eps_e, eps_v=S.eps_v,
                  dbar=DBAR, vhnd=VHND)
    Rb = sbh.run(B, variant, fifo=D)
    Rb['runtime_s_total'] = time.time() - t0
    Rb['no_sched'] = bool(B['no_sched'])
    Rb['NS'] = [int(x) for x in B['NS']]
    Rb['DPRE'] = [float(x) for x in B['DPRE']]
    return variant, name, _py(Rb)


def _cmp(a, b):
    """differences between two result dicts in the common layout."""
    def mxd(x, y):
        x = _arr(x)
        y = _arr(y)
        if x.shape != y.shape:
            return None
        both = np.isnan(x) & np.isnan(y)
        d = np.abs(x - y)
        d[both] = 0.0
        return float(np.max(d)) if not np.any(np.isnan(d)) else None
    out = OrderedDict()
    for key in ('tau', 'marker_err', 'gap_err', 'tact', 'tset', 'trule', 'e_hnd',
                'eps_hnd', 'gmin', 'u_min_unit', 'u_max_unit', 'u_min_pre_unit',
                'u_max_pre_unit'):
        if a.get(key) is not None and b.get(key) is not None:
            out['max_%s_diff' % key] = mxd(a[key], b[key])
    for key in ('settling', 'entry', 'u_min', 'u_max'):
        if a.get(key) is not None and b.get(key) is not None:
            out['%s_diff' % key] = abs(float(a[key]) - float(b[key]))
    out['stop_order_equal'] = a.get('stop_order') == b.get('stop_order')
    if a.get('rule_units') is not None and b.get('rule_units') is not None:
        out['rule_equal'] = a['rule_units'] == b['rule_units']
    return out


def _std_hr(k):
    return OrderedDict(
        tau=k['tstop'], settling=k['settling'], marker_err=k['marker_signed'],
        gap_err=k['gap_signed'], tact=k['tact'], tset=k['tset'],
        trule=k['trule'], e_hnd=k['e_hnd'], eps_hnd=k['eps_hnd'], gmin=k['gmin'],
        u_min_unit=k['u_min_unit'], u_max_unit=k['u_max_unit'],
        u_min_pre_unit=k['u_min_pre_unit'], u_max_pre_unit=k['u_max_pre_unit'],
        u_min=k['u_min'], u_max=k['u_max'], entry=k['entry_obs'],
        stop_order=k['stop_order'], rule_units=k['rule_units'], n_pos=k['n_pos_nom'],
        n_clip=k['n_sat'], filter_modifications=k['filter_modifications'],
        box_last_out=k['box_last_out'][0], age=k['age'], fb_law=k.get('fb_law'),
        runtime_s=k.get('runtime_s'))


def _std_bench(o, n):
    return OrderedDict(
        tau=o['tstop'], settling=o['disp'], marker_err=o['marker_signed'],
        gap_err=o['gap_signed'], tact=o['tact'], tset=o['tset'],
        trule=o['trule'], e_hnd=o['e_hnd'], eps_hnd=o['eps_hnd'], gmin=o['gmin'],
        u_min_unit=o['u_min_unit'], u_max_unit=o['u_max_unit'],
        u_min_pre_unit=o['u_min_pre_unit'], u_max_pre_unit=o['u_max_pre_unit'],
        u_min=o['u_min'], u_max=o['u_max'], stop_order=o['stop_order'],
        rule_units=o['rule_units'], t_clipU=o.get('t_clipU'),
        t_nom_pos=o.get('t_nom_pos'), n_latch_before_S2=o.get('n_latch_before_S2'),
        schedule_mode=o.get('schedule_mode'), kernel_law=o.get('kernel_law'),
        no_sched=o.get('no_sched'), NS=o.get('NS'), DPRE=o.get('DPRE'),
        contact_time=o.get('contact_time'), buffer_lost=o.get('buffer_lost'),
        runtime_s=o.get('runtime_s'))


def part_indep():
    t0 = time.time()
    cfgc = cert_cfg()
    cp = cert_pack(cf.evaluate(cfgc, keep_curves=True), cfgc)
    S = design_setup()
    cases = [('const19', None)] + [('seed%d' % s, s) for s in INDEP_SEEDS]
    log("[indep] sim_hr (zoh 1 ms, cont 1e-4 s; nosched park) vs sim_bench_hr ('proposed', "
        "'feedback'; ideal plant, own kernel) on %s" % [c[0] for c in cases])
    jobs = []
    for name, seed in cases:
        jobs.append(('proposed', name, seed))
        jobs.append(('feedback', name, seed))
    ext = pool_map(_job_indep, jobs, None)
    extd = {(v, nm): o for (v, nm, o) in ext}
    runs = OrderedDict()
    for name, seed in cases:
        D = None if seed is None else rcv.random_schedule(np.random.default_rng(seed), n_gen(S), 19)
        res = OrderedDict()
        R_z, T_z, k_z, bt_z = run_rows(S, cp, 'zoh', D=D)
        R_c, T_c, k_c, bt_c = run_rows(S, cp, 'cont', D=D, h=H_REF)
        res['sim_hr_zoh_h0.001'] = dict(_std_hr(k_z), bounds_ok=bt_z['all_ok'],
                                        bounds_failed=bt_z['failed'])
        res['sim_hr_cont_h0.0001'] = dict(_std_hr(k_c), bounds_ok=bt_c['all_ok'],
                                          bounds_failed=bt_c['failed'])
        res['sim_bench_hr_proposed'] = _std_bench(extd[('proposed', name)], S.n)
        res['sim_bench_hr_feedback'] = _std_bench(extd[('feedback', name)], S.n)
        base = 'sim_hr_zoh_h0.001'
        cmpd = OrderedDict()
        for other in [x for x in res if x != base]:
            cmpd['%s_vs_%s' % (base, other)] = _cmp(res[base], res[other])
        cmpd['sim_bench_hr_proposed_vs_feedback'] = _cmp(res['sim_bench_hr_proposed'],
                                                         res['sim_bench_hr_feedback'])
        runs[name] = OrderedDict(seed=seed, results=res, compare=cmpd)
        for nm, c in cmpd.items():
            log('  %-9s %-58s stops %s s markers %s m gaps %s m act %s sw %s order %s'
                % (name, nm, '%.2e' % c['max_tau_diff'] if c.get('max_tau_diff') is not None
                   else '-', '%.2e' % c['max_marker_err_diff'] if c.get('max_marker_err_diff')
                   is not None else '-', '%.2e' % c['max_gap_err_diff']
                   if c.get('max_gap_err_diff') is not None else '-',
                   '%.1e' % c['max_tact_diff'] if c.get('max_tact_diff') is not None else '-',
                   '%.1e' % c['max_tset_diff'] if c.get('max_tset_diff') is not None else '-',
                   c['stop_order_equal']))

    def agg_of(key):
        cs = [r['compare'][key] for r in runs.values()]
        return OrderedDict(
            max_stop_diff=_max(c.get('max_tau_diff') for c in cs),
            max_settling_diff=_max(c.get('settling_diff') for c in cs),
            max_marker_diff=_max(c.get('max_marker_err_diff') for c in cs),
            max_gap_diff=_max(c.get('max_gap_err_diff') for c in cs),
            max_tact_diff=_max(c.get('max_tact_diff') for c in cs),
            max_tset_diff=_max(c.get('max_tset_diff') for c in cs),
            max_trule_diff=_max(c.get('max_trule_diff') for c in cs),
            max_e_hnd_diff=_max(c.get('max_e_hnd_diff') for c in cs),
            max_gmin_diff=_max(c.get('max_gmin_diff') for c in cs),
            max_u_min_unit_diff=_max(c.get('max_u_min_unit_diff') for c in cs),
            max_u_max_unit_diff=_max(c.get('max_u_max_unit_diff') for c in cs),
            max_entry_diff=_max(c.get('entry_diff') for c in cs),
            stop_order_equal=all(c['stop_order_equal'] for c in cs),
            rule_equal=all(c.get('rule_equal', True) for c in cs))
    b = 'sim_hr_zoh_h0.001_vs_'
    agg = OrderedDict(
        sim_bench_hr_proposed=agg_of(b + 'sim_bench_hr_proposed'),
        sim_bench_hr_feedback=agg_of(b + 'sim_bench_hr_feedback'),
        cont_vs_zoh=agg_of(b + 'sim_hr_cont_h0.0001'),
        bench_proposed_vs_feedback=agg_of('sim_bench_hr_proposed_vs_feedback'),
        bench_no_schedule=all(r['results']['sim_bench_hr_proposed']['no_sched']
                              for r in runs.values()),
        sim_hr_positive_nominal=sum(r['results']['sim_hr_zoh_h0.001']['n_pos']
                                    for r in runs.values()),
        all_sim_hr_bounds_ok=all(v['bounds_ok'] for r in runs.values()
                                 for nm, v in r['results'].items() if nm.startswith('sim_hr')),
        all_sim_hr_fb_law=all(v['fb_law']['ok'] for r in runs.values()
                              for nm, v in r['results'].items() if nm.startswith('sim_hr')))
    tol = dict(sim_bench_hr_stop_s=1e-9, sim_bench_hr_marker_m=1e-9)
    agg['tolerances'] = tol
    for key in ('sim_bench_hr_proposed', 'sim_bench_hr_feedback'):
        sb_ = agg[key]
        agg[key + '_agrees'] = bool(sb_['max_stop_diff'] is not None
                                    and sb_['max_stop_diff'] <= tol['sim_bench_hr_stop_s']
                                    and sb_['max_marker_diff'] <= tol['sim_bench_hr_marker_m']
                                    and sb_['max_gap_diff'] <= tol['sim_bench_hr_marker_m']
                                    and sb_['stop_order_equal'] and sb_['rule_equal'])
    pf = agg['bench_proposed_vs_feedback']
    agg['bench_proposed_identical_to_feedback'] = bool(
        pf['max_stop_diff'] == 0.0 and pf['max_marker_diff'] == 0.0
        and pf['max_gap_diff'] == 0.0 and pf['stop_order_equal'])
    log('  aggregate: %s' % json.dumps(_py(agg))[:2000])
    out = OrderedDict(
        meta=meta('independent', seeds=list(INDEP_SEEDS),
                  options=dict(sim_hr=['zoh h = 1 ms', 'cont h = 1e-4 s'],
                               sim_bench_hr=['proposed', 'feedback'], plant='ideal')),
        study=('independent-code agreement on identical delays (constant 19 ms and random '
               'schedules seed %s of the schedules part, per-hop delays 1..19 ms): '
               "sim_hr.py (scheme 'zoh', pure 1 kHz, the base; and 'cont' at h = 1e-4 s; "
               "schedules=False, nosched='park') and src/simulation/sim_bench_hr.py "
               "(variants 'proposed' and 'feedback' on the shaped setup, ideal plant, 1 ms; "
               'own kernel, shares hr_setup and sim_cont.deliveries with sim_hr).  On a '
               'shaped setup sim_bench_hr has no schedule pieces, so its proposed law '
               'regulates to the parking gaps from takeover (the FB law) and equals its '
               "'feedback' variant.  The prototype of the schedule version is not used"
               % list(INDEP_SEEDS)),
        runs=runs, aggregate=agg, certificate=cert_summary(cp), runtime_s=time.time() - t0)
    dump('independent.json', out)
    return out


# =============================================================== check
def part_check():
    out = OrderedDict(meta=meta('check'), files=OrderedDict(), all_ok=True)

    def add(name, ok, detail=None, flags=None):
        out['files'][name] = dict(ok=bool(ok), flags=flags, detail=detail)
        if not ok:
            out['all_ok'] = False
        log('  %-18s %s %s' % (name, 'OK' if ok else 'FAIL',
                               json.dumps(_py(detail))[:600] if detail else ''))

    def fl(**kw):
        return OrderedDict((k_, bool(v)) for k_, v in kw.items())

    r = load('reference.json')
    if r is not None:
        f = fl(certificate_admitted=r['certificate']['admitted'],
               design_constants=r['design_constants']['all'],
               bounds_all_ok=r['bounds']['all_ok'],
               all_rows_checked=r['bounds']['n_checked'] == r['bounds']['n_rows'],
               fine_bounds_all_ok=r['fine']['bounds']['all_ok'],
               records_identical=r['record_identical_10ms_1ms']['identical']
               and r['fine']['stop_window_run_identical']['identical'],
               fine_vs_reference_stops=r['fine']['max_stop_diff'] < 1e-8,
               stop_record_resolves_1us=r['stop_record']['resolves_1us'],
               certificate_consistent_with_design_record=r['certificate_consistency'][
                   'same_as_design'],
               evaluator_matches_design_record=r['meta']['evaluator_matches_design_record'],
               fb_law=r['fb_law_check']['ok'],
               filter_inactive=r['kpis']['filter_inactive'],
               commands_in_set=r['kpis']['commands_in_set'],
               zero_after_latch=r['kpis']['zero_after_latch'],
               all_stopped=r['kpis']['all_stopped'],
               requirements=r['requirements']['all_ok'])
        add('reference.json', all(f.values()), dict(
            n_checked=r['bounds']['n_checked'], failed=r['bounds']['failed'],
            dispersion=r['kpis']['settling'], u=[r['kpis']['u_min'], r['kpis']['u_max']],
            positive_nominal=r['kpis']['n_pos_nom']), f)
    else:
        add('reference.json', False, dict(missing=True))
    c = load('convergence.json')
    if c is not None:
        f = fl(all_rows_inside_bounds=c['all_rows_inside_bounds'],
               all_rows_checked=c['all_rows_checked'],
               all_filter_inactive=c['all_filter_inactive'],
               stops_converged=c['max_stop_diff'] < 1e-8, age=c['all_age_ok'],
               fb_law=c['all_fb_law'])
        add('convergence.json', all(f.values()), dict(max_diff=c['max_diff_over_h_and_kpis'],
                                                      argmax=c['argmax_kpi'],
                                                      max_stop_diff=c['max_stop_diff']), f)
    else:
        add('convergence.json', False, dict(missing=True))
    g = load('digital.json')
    if g is not None:
        f = fl(all_bounds_ok=g['all_bounds_ok'], all_rows_checked=g['all_rows_checked'],
               all_filter_inactive=g['all_filter_inactive'], age=g['all_age_ok'],
               fb_law=g['all_fb_law'])
        add('digital.json', all(f.values()), dict(max_stop_diff_1kHz=g['max_stop_diff_1kHz'],
                                                  max_marker_diff_1kHz=g['max_marker_diff_1kHz']),
            f)
    else:
        add('digital.json', False, dict(missing=True))
    for nm in ('schedules.json', 'adversarial.json'):
        s = load(nm)
        if s is None:
            add(nm, False, dict(missing=True))
            continue
        a = s['aggregate']
        f = fl(bounds_all_ok=a['bounds_all_ok'], all_rows_checked=a['all_rows_checked'],
               age=a['all_age_ok'], all_stopped=a['all_stopped'], no_contact=a['no_contact'],
               filter_inactive=a['filter_modifications_total'] == 0,
               zero_after_latch=a['u_abs_after_latch_max'] == 0.0,
               commands_in_set=a['u_min']['value'] >= -U_BR - TOL
               and a['u_max']['value'] <= U_TR + TOL,
               no_rule_before_T_xi=a['runs_with_rule_before_T_xi'] == 0,
               fb_law=a['fb_law_all'], requirements=a['requirements_all_ok'])
        add(nm, all(f.values()), dict(n=a['n'], settling_max=a['settling']['value'],
                                      n_bounds_failed=a['n_bounds_failed'],
                                      failed_rows=a['bounds_failed_rows'],
                                      positive_nominal=a['positive_nominal_total']), f)
    w = load('sweep.json')
    if w is not None:
        f = fl(admitted_rows_inside_bounds=w['admitted_rows_inside_bounds'],
               admitted_rows_all_checked=w['admitted_rows_all_checked'],
               admitted_rows_filter_inactive=w['admitted_rows_filter_inactive'],
               admitted_rows_requirements_ok=w['admitted_rows_requirements_ok'],
               admitted_rows_commands_in_set=w['admitted_rows_commands_in_set'],
               admitted_grid_consistent_with_boundary=w['admitted_grid_consistent_with_boundary'],
               boundary_equals_design_record=w['boundary_equals_design_record'],
               fb_law=w['all_rows_fb_law'], age=w['all_age_ok'])
        add('sweep.json', all(f.values()), dict(certified_ages_ms=w['certified_ages_ms'],
                                                boundary_ms=w['boundary_ms']['k_ms'],
                                                first_failures={k_: v for k_, v in
                                                                w['first_failures'].items()
                                                                if k_ != 'requirement_each'}), f)
    else:
        add('sweep.json', False, dict(missing=True))
    e = load('ensemble.json')
    if e is not None:
        a = e['aggregate']['admitted']
        f = fl(rule_reproduces_design=e['rule_at_nominal_state']['reproduces_design'],
               admitted_inside_bounds=a is None or a['bounds_all_ok'],
               admitted_all_checked=a is None or a['all_rows_checked'],
               admitted_filter_inactive=a is None or a['filter_modifications_total'] == 0,
               admitted_requirements=a is None or a['requirements_all_ok'],
               admitted_age=a is None or a['age_ok'],
               admitted_fb_law=a is None or a['fb_law_all'],
               all_states_inside_C2=all(x['C2_ok'] for x in e['rows']))
        add('ensemble.json', all(f.values()), dict(
            n=e['aggregate']['n'], n_admitted=e['aggregate']['n_admitted'],
            failures=e['aggregate']['failures_among_rejected'],
            n_rejected_by_C2=e['aggregate']['n_rejected_by_C2']), f)
    else:
        add('ensemble.json', False, dict(missing=True))
    o = load('outside.json')
    if o is not None:
        a = o['aggregate']
        f = fl(certificate_not_applicable_all=a['certificate_not_applicable_all'],
               not_admitted=not a['admitted_any'], age=a['age_ok'],
               fb_law=a['runs_fb_law'] == a['n_runs'])
        add('outside.json', all(f.values()), dict(
            runs=a['n_runs'], clipping=a['runs_with_clipping'],
            positive=a['runs_with_positive_nominal'], filter=a['runs_with_filter_action'],
            rule_before_T_xi=a['runs_with_rule_before_T_xi'],
            within_design_bounds=a['runs_within_design_bounds'],
            not_stopped=a['runs_not_stopped'], contact=a['runs_with_contact']), f)
    else:
        add('outside.json', False, dict(missing=True))
    x = load('age_audit.json')
    if x is not None:
        f = fl(all_within_bound=x['totals']['all_within_bound'],
               matches_runs=x['totals']['mismatches_vs_runs'] == 0)
        add('age_audit.json', all(f.values()), x['totals'], f)
    else:
        add('age_audit.json', False, dict(missing=True))
    i = load('independent.json')
    if i is not None:
        a = i['aggregate']
        f = fl(sim_bench_hr_proposed_agrees=a['sim_bench_hr_proposed_agrees'],
               sim_bench_hr_feedback_agrees=a['sim_bench_hr_feedback_agrees'],
               bench_proposed_identical_to_feedback=a['bench_proposed_identical_to_feedback'],
               bench_no_schedule=a['bench_no_schedule'],
               all_sim_hr_bounds_ok=a['all_sim_hr_bounds_ok'],
               all_sim_hr_fb_law=a['all_sim_hr_fb_law'])
        add('independent.json', all(f.values()), dict(
            sim_bench_hr_proposed=a['sim_bench_hr_proposed']), f)
    else:
        add('independent.json', False, dict(missing=True))
    out['n_flags'] = sum(len(v['flags'] or {}) for v in out['files'].values())
    out['failed_flags'] = ['%s:%s' % (fn, k_) for fn, v in out['files'].items()
                           for k_, ok in (v['flags'] or {}).items() if not ok]
    dump('check.json', out)
    log('  ALL OK' if out['all_ok'] else '  FAILURES PRESENT: %s' % out['failed_flags'])
    return out


# ================================================================ main
PARTS = OrderedDict(ref=part_ref, conv=part_conv, digital=part_digital, sched=part_sched,
                    adv=part_adv, sweep=part_sweep, ensemble=part_ens, outside=part_outside,
                    independent=part_indep, age_audit=part_age_audit, check=part_check)
ALIASES = dict(schedules=['sched', 'adv'], ens=['ensemble'], indep=['independent'],
               audit=['age_audit'])
ORDER = ['ref', 'conv', 'digital', 'sched', 'adv', 'sweep', 'ensemble', 'outside',
         'independent', 'age_audit', 'check']


def main():
    todo = sys.argv[1:] or ['all']
    if todo == ['all']:
        todo = list(ORDER)
    parts = []
    for p in todo:
        parts += ALIASES.get(p, [p])
    for p in parts:
        if p not in PARTS:
            raise SystemExit('unknown part %s (parts: %s, aliases %s)'
                             % (p, list(PARTS), list(ALIASES)))
    log('==== run_fb_validate.py %s parts %s (python %s, numpy %s, %s, %d processes) ===='
        % (time.strftime('%Y-%m-%d %H:%M:%S'), parts, sys.version.split()[0],
           np.__version__, platform.platform(), NPROC))
    for p in parts:
        t0 = time.time()
        log('=== part: %s ===' % p)
        PARTS[p]()
        log('[%s] done in %.1f s' % (p, time.time() - t0))


if __name__ == '__main__':
    mp.freeze_support()
    main()
