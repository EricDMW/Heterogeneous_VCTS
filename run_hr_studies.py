#!/usr/bin/env python3
r"""run_hr_studies.py -- studies of the head-only reference arrival (HR
design, data/new_d19/headref/sim_spec.md), 2026-09-28.

Simulator: src/simulation/sim_bench_hr.py.  Outputs go to
data/new_d19/headref/studies/<part>.json and the log of a part to
logs_ext/headref/studies_<part>.log.  Nothing else is written.

Design parameters (command-line options, recorded in every output, so that
a rerun at another design is one command; defaults = the final design of
2026-09-28): the configuration cfg of data/new_d19/caseD/design.json
(round-6 fan; reference = plan of unit 1, schedules rho_i = a*_i -
a*_{i-1}) with
  --lam 0.15      common gain, beta = lam^2, gamma = 2 lam for every unit
  --U 0.02        traction limit [m/s^2]
  --Uminus 1.2    braking limit [m/s^2]
  --eps_e 0.05 --eps_v 0.005   handoff box [m, m/s]
  --dbar 0.020    age bound [s]; T_m = T_c = 1 ms, constant transport
                  dbar - T_m (19 ms)
  --vhnd 0.01     handoff pad of the certificate (recorded; used by T5 only)
  --lags 41,41    entry-curve lags of the certificate (T5 only)
  --design PATH   design.json (default data/new_d19/caseD/design.json)
  --workers N     processes for the MPC runs (default 8)
  --tag NAME      suffix of the output and log names (<part>_NAME.json,
                  studies_<part>_NAME.log), for runs at other parameters
                  that must not overwrite the final files (e.g. lam010)

Parts
  check   (T1) 'proposed' on the ideal plant against
          sim_hr.run(scheme='zoh', h = T_u = T_m = T_c = 1 ms) with the
          constant transport and with the FIFO schedule of seed 1
          (run_caseD_validate.random_schedule, as run_hr_simcheck.py);
          tolerances 1e-6 s (stop, switch, activation, closure and RULE
          instants) and 1e-6 m (marker and gap errors, clearance minima).
          (T2) 'feedback_step' against sim_hr(schedules=False,
          nosched='step').  (T3) 'nofilter' against 'proposed' field by
          field on both takeovers.  (T4) port checks of 'feedback' and
          'cacc' against sim_bench with the same input set: perturbed plant
          (seeds 1-3, without and with compensation, tolerance 1e-9) and
          ideal plant, nominal and loss scenarios (records before the first
          RULE entry, stop or contact, tolerance 1e-9; afterwards the RULE
          convention differs: sim_bench enters RULE at the next grid
          instant).  (T5) observations of 'proposed' against the bounds of
          the certificate, evaluated here by
          certify_headref.evaluate(config(lam, vhnd, lags, U, Uminus, dbar,
          eps_e, eps_v)) (read-only import; informational).
  base    every variant on the benchmark takeover and on the takeover with
          doubled speed increments: proposed, nofilter, single, nofeedback,
          feedback, feedback_step, cacc (h_w 0.2, 0.5, 1.0), pid
          (hand-tuned gains of sim_bench), plans, mpc (weights selected by
          the extended protocol of data/new_d19/caseD/studies/mpc.json), and
          the comparators with the filter off.
  retune  the fbtune protocol of run_bench_studies.py: one common gain
          lam in FB_LAMBDAS (0.05, ..., 0.20, 0.25, 0.30, 0.40, 0.50) for
          'feedback', 'feedback_step' and 'cacc' (h_w 0.2, 0.5, 1.0), and
          PID_GRID (27 triples) for 'pid'; training on the doubled-increment
          takeover, held-out on the benchmark takeover; objective
          run_bench_studies.score; selection: lexicographic minimum of the
          training score, first in grid order on ties.  'proposed' over
          FB_LAMBDAS is added for information (its gain comes from the
          certificate, not from this protocol).
  loss    t_loss in (15, 25, 35, 50, 70, 84) s; 'proposed', 'feedback' at
          the design gain and at its retuned gain (retune.json), and
          'plans' (comparator), each without and with emergency braking of
          unit 3 and with the watchdog T_wd in (none, 0.10, 0.25, 0.50,
          1.00) s; the state of the nominal run at the loss instants.
  plant   perturbed plant, seeds 1-20, default and with compensation of
          grade and resistance (grade_ff = res_ff = 1): 'proposed',
          'feedback' (design and retuned gain), 'feedback_step' (retuned),
          'cacc' (retuned, every h_w), 'pid' (retuned), 'plans'; 'mpc' on
          the default plant only (sim_bench's MPC has no compensation); the
          variations of the previous study (actuator lag 0, 0.5, 1.0 s, no
          grade, no resistance) for 'proposed', without and with
          compensation.
  all     check base retune loss plant

Usage: python -B run_hr_studies.py <part> ... [options]
"""
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE, os.path.join(CODE, 'src', 'simulation'),
           os.path.join(CODE, 'src', 'certification')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                                       # noqa: E402
import sim_bench_hr as sbh                               # noqa: E402
import sim_bench as sb                                   # noqa: E402
import sim_hr as hr                                      # noqa: E402
import run_bench_studies as rbs                          # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'headref', 'studies')
LOGDIR = os.path.join(CODE, 'logs_ext', 'headref')
DESIGN = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')

# ---- protocol of the previous study (run_bench_studies.py, read-only)
REQ = dict(rbs.REQ)
FB_LAMBDAS = tuple(rbs.FB_LAMBDAS)
PID_GRID = list(rbs.PID_GRID)
T_LOSS = tuple(rbs.T_LOSS)
SEEDS = tuple(rbs.SEEDS)
score = rbs.score
joint = rbs.joint
# ---- this study
T_WD = (None, 0.10, 0.25, 0.50, 1.00)
HW_LIST = (0.2, 0.5, 1.0)
EM_UNIT = 3
COMP = dict(grade_ff=1.0, res_ff=1.0)
PLANT_VAR_ROWS = [
    ('ideal actuator', dict(tau_act=0.0)),
    ('lag 0.5', dict(tau_act=0.5)),
    ('lag 1.0', dict(tau_act=1.0)),
    ('no grade', dict(grade=0.0)),
    ('no resistance', dict(davis=(0.0, 0.0, 0.0))),
]
TOL_CHECK = dict(time=1e-6, position=1e-6)
TOL_PORT = 1e-9
DEFAULT_OPTS = dict(lam=0.15, U=0.02, Uminus=1.2, eps_e=0.05, eps_v=0.005,
                    dbar=0.020, vhnd=0.01, lags=[41, 41], design=DESIGN,
                    workers=8, tag='')
TAG = dict(value='')


def pname(part):
    """File name of a part with the optional tag."""
    return part + ('_' + TAG['value'] if TAG['value'] else '')


# ================================================================ helpers
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


class Log:
    def __init__(self, part):
        os.makedirs(LOGDIR, exist_ok=True)
        self.path = os.path.join(LOGDIR, 'studies_%s.log' % pname(part))
        self.f = open(self.path, 'w')
        self.t0 = time.time()

    def __call__(self, msg=''):
        line = str(msg)
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def rel(p):
    return os.path.relpath(p, CODE).replace('\\', '/')


def sha256_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()
                          ).hexdigest()


def fmt(x, f='%.4f', na='n.d.'):
    return na if x is None else f % x


def slim(R):
    return {k: v for k, v in R.items() if k != 'rec'}


def setup_params(opts):
    return dict(lam=float(opts['lam']), U=float(opts['U']),
                Uminus=float(opts['Uminus']), eps_e=float(opts['eps_e']),
                eps_v=float(opts['eps_v']), dbar=float(opts['dbar']),
                vhnd=float(opts['vhnd']))


def context(opts):
    D = json.load(open(opts['design']))
    B = sbh.setup(D['cfg'], **setup_params(opts))
    return D, B


def header(part, D, B, opts):
    files = dict(
        sim_bench_hr=os.path.join(CODE, 'src', 'simulation',
                                  'sim_bench_hr.py'),
        run_hr_studies=os.path.abspath(__file__),
        sim_hr=os.path.join(CODE, 'src', 'simulation', 'sim_hr.py'),
        sim_bench=os.path.join(CODE, 'src', 'simulation', 'sim_bench.py'),
        sim_cont=os.path.join(CODE, 'src', 'simulation', 'sim_cont.py'),
        certify_plan=os.path.join(CODE, 'src', 'certification',
                                  'certify_plan.py'),
        run_bench_studies=os.path.join(CODE, 'run_bench_studies.py'),
        design=opts['design'])
    return dict(
        part=part, simulator='src/simulation/sim_bench_hr.py',
        driver='run_hr_studies.py',
        command='python -B run_hr_studies.py ' + ' '.join(sys.argv[1:]),
        options={k: (rel(v) if k == 'design' else v)
                 for k, v in opts.items()},
        design=dict(
            file=rel(opts['design']), revision=D.get('revision'),
            cfg=D['cfg'], cfg_sha256=cfg_hash(D['cfg']),
            params=dict(B['params']), cfg_used=B['cfg'],
            cfg_used_sha256=cfg_hash(B['cfg']),
            reference='head: a_r = a*_1 of certify_plan.plan(cfg) on [0, '
                      'T_xi), -a_b until v_r = 0; followers: rho_i = a*_i - '
                      'a*_{i-1} in local time from the closure onset '
                      '(sim_hr.hr_setup)'),
        requirements=dict(REQ),
        constants=dict(
            n=B['n'], dt=B['dt'], T_m=B['T_m'], T_c=B['T_c'],
            transport=B['transport'], dbar=B['dbar'], U=B['U'],
            Uminus=B['Uminus'], lam=B['lam'], beta=B['beta'],
            gamma=B['gamma'], alpha=B['alpha'], phi=B['phi'],
            eps_det=B['eps_det'], eps_e=B['eps_e'], eps_v=B['eps_v'],
            a_b=B['a_b'], kappa=B['kappa'], s_m=B['s_m'], ell=B['ell'],
            v_c=B['v_c'], amax=B['amax'], d_s=B['d_s'], c0=B['c0'],
            s0=B['s0'], v0=B['v0'], t_0=B['t_0'], T_xi=B['T_xi'],
            tau_r=B['tau_r'], v_c1=B['v_c1'], s_ref_stop=B['s_ref_stop'],
            markers=B['markers'], t_join=B['t_join'],
            v0_doubled=sbh.doubled_mismatch(B),
            pid_hand_tuned=dict(sb.K_PID), hifi_default=dict(sb.HIFI_DEFAULT),
            grade_signs=list(sb.GRADE_SIGNS), actuator_upper=sb.U_ACT),
        code_sha256={k: sha256_file(p) for k, p in files.items()},
        runtime_date=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_system=dict(python=platform.python_version(),
                            numpy=np.__version__,
                            numba=__import__('numba').__version__,
                            machine=platform.machine(),
                            system=platform.platform(),
                            cpus=os.cpu_count()))


def dump(part, obj, log):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, pname(part) + '.json')
    json.dump(_py(obj), open(p, 'w'), indent=1)
    log('  -> %s (%.1f kB)' % (rel(p), os.path.getsize(p) / 1e3))
    return p


def finish(part, obj, opts, log, t0):
    D2 = json.load(open(opts['design']))
    obj['design']['cfg_unchanged_at_end'] = bool(
        cfg_hash(D2['cfg']) == obj['design']['cfg_sha256'])
    obj['runtime_s'] = time.time() - t0
    log('  design cfg unchanged at the end: %s; part run time %.1f s'
        % (obj['design']['cfg_unchanged_at_end'], obj['runtime_s']))
    dump(part, obj, log)
    log.close()


def load_part(part, B):
    """A part file written by this driver for the same design and
    parameters (retune.json is read by loss and plant; with a tag the
    tagged file)."""
    p = os.path.join(OUT, pname(part) + '.json')
    if not os.path.exists(p):
        raise SystemExit('%s not found: run part %s first (same options)'
                         % (rel(p), part))
    J = json.load(open(p))
    if J['design']['cfg_used_sha256'] != cfg_hash(B['cfg']) or \
            J['design']['params'] != _py(B['params']):
        raise SystemExit('%s belongs to other design parameters: rerun part '
                         '%s' % (rel(p), part))
    return J


def _vals(a):
    return np.array([np.nan if x is None else x for x in a], float)


def maxdiff(a, b):
    """Largest absolute difference of two lists (None/NaN must coincide;
    None is returned otherwise)."""
    a = _vals(a)
    b = _vals(b)
    if a.shape != b.shape or np.any(np.isnan(a) != np.isnan(b)):
        return None
    if np.all(np.isnan(a)):
        return 0.0
    return float(np.nanmax(np.abs(a - b)))


def row_of(name, R, **extra):
    """Summary row of one run (base, retune)."""
    done = bool(R['all_stopped'] and R['contact_time'] is None)
    nn = R.get('t_nonneg')
    tb = R['t_barrier']
    r = dict(
        name=name, variant=R['variant'], plant=R['plant'],
        apply_filter=R['apply_filter'], lam=R.get('lam'),
        h_w=R.get('h_w'), gains=R.get('gains'), v0=R.get('v0'),
        all_stopped=R['all_stopped'], contact=R['contact_time'],
        contact_pair=R['contact_pair'],
        disp=R['disp'] if done else None, T_f=R['T_f'],
        marker=R['marker_abs_max'] if done else None,
        gap=R['gap_abs_max'] if done else None,
        state_at_end=dict(marker=R['marker_abs_max'], gap=R['gap_abs_max'],
                          t=R['t_end_sim']),
        gmin=R['gmin_min'], gmin_pair=R['gmin_pair'], hmin=R['hmin_min'],
        u_abs_max=R['u_abs_max'], u_min=R['u_min'], u_max=R['u_max'],
        atU=max(R['t_atU']), atU_units=R['t_atU'],
        nonneg=max(nn) if nn is not None else None, nonneg_units=nn,
        nom_above_U=max(R['t_nom_above_U']), clipU=max(R['t_clipU']),
        barrier=max(tb),
        barrier_unit=(int(np.argmax(tb)) + 1) if max(tb) > 0 else None,
        barrier_raw=max(R['t_barrier_raw']),
        fallback=float(sum(R['t_infeas'])),
        rule_units=R['rule_units'], stop_order=R['stop_order'],
        buffer_lost=R['buffer_lost'],
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], met=R['pass_all'])
    r.update(extra)
    return r


def row_line(r):
    return ('  %-28s disp=%s T_f=%s marker=%s gap=%s gmin=%.3f max|u|=%s '
            'atU=%.2f nonneg=%s barrier=%.2f fallback=%.2f RULE=%s order=%s '
            'contact=%s met=%s'
            % (r['name'], fmt(r['disp']), fmt(r['T_f'], '%.3f'),
               fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'), r['gmin'],
               fmt(r['u_abs_max'], '%.3f'), r['atU'],
               fmt(r['nonneg'], '%.2f', 'n.a.'), r['barrier'], r['fallback'],
               r['rule_units'], r['stop_order'],
               fmt(r['contact'], '%.3f', '-'), r['met']))


def loss_row(R, **extra):
    k = int(np.argmin(_vals(R['gmin'])))
    tb = [x for x in R['t_buffer_lost'] if x is not None]
    aborted = list(R.get('override_units') or [])
    completed = bool(R['all_stopped'] and not aborted)
    r = dict(
        variant=R['variant'], lam=R.get('lam'), t_loss=R['t_loss'],
        emergency_unit=R['emergency_unit'], T_wd=R['T_wd'],
        contact=R['contact_time'] is not None, t_contact=R['contact_time'],
        contact_pair=R['contact_pair'],
        buffer_lost=R['buffer_lost'], t_buffer_lost=min(tb) if tb else None,
        buffer_lost_pairs=[i + 2 for i, x in enumerate(R['t_buffer_lost'])
                           if x is not None],
        gmin=R['gmin_min'], gmin_pair=k + 2, t_gmin=R['t_gmin'][k],
        gmin_pairs=R['gmin'],
        all_stopped=R['all_stopped'], n_stopped=R['n_stopped'],
        override_units=aborted, watchdog_units=R.get('watchdog_units'),
        t_watchdog=R.get('t_watchdog'),
        arrival_completed=completed,
        disp=R['disp'] if completed else None,
        marker=R['marker_abs_max'] if completed else None,
        gap=R['gap_abs_max'] if completed else None,
        pass_all=bool(completed and R['pass_all']),
        tstop=R['tstop'], T_f=R['T_f'], stop_order=R['stop_order'],
        rule_units=R['rule_units'], barrier=max(R['t_barrier']),
        fallback=float(sum(R['t_infeas'])), atU=max(R['t_atU']),
        u_min=R['u_min'],
        state_at_end=dict(t=R['t_end_sim'], marker_signed=R['marker_signed'],
                          gap_signed=R['gap_signed'], v=R['v_final'],
                          mode=R.get('mode_final')))
    r.update(extra)
    return r


def loss_line(tag, r):
    return ('  %-18s t_loss=%4.0f em=%-4s T_wd=%-4s contact=%s buffer lost=%s '
            '(%s) gmin=%.3f (pair %d, t=%s) stopped=%d completed=%s disp=%s '
            'marker=%s gap=%s met=%s barrier=%.2f fallback=%.2f'
            % (tag, r['t_loss'], r['emergency_unit'], r['T_wd'],
               fmt(r['t_contact'], '%.3f', '-'), r['buffer_lost'],
               fmt(r['t_buffer_lost'], '%.3f', '-'), r['gmin'],
               r['gmin_pair'], fmt(r['t_gmin'], '%.2f', '-'),
               r['n_stopped'], r['arrival_completed'], fmt(r['disp']),
               fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'),
               r['pass_all'], r['barrier'], r['fallback']))


def plant_rec(R, **extra):
    nn = R.get('t_nonneg')
    rec = dict(
        seed=R['seed'], disp=R['disp'], marker=R['marker_abs_max'],
        gap=R['gap_abs_max'], marker_signed=R['marker_signed'],
        gap_signed=R['gap_signed'], tstop=R['tstop'], T_f=R['T_f'],
        stop_order=R['stop_order'], gmin=R['gmin_min'],
        gmin_pair=R['gmin_pair'], gmin_pairs=R['gmin'], hmin=R['hmin_min'],
        barrier=max(R['t_barrier']), fallback=float(sum(R['t_infeas'])),
        atU=max(R['t_atU']), atU_units=R['t_atU'],
        nonneg=max(nn) if nn is not None else None,
        clipU=max(R['t_clipU']), nom_above_U=max(R['t_nom_above_U']),
        clipk=max(R['t_clipk']), u_min=R['u_min'], u_max=R['u_max'],
        u_realized_abs_max=max(R['u_realized_abs_max']),
        jerk_pk=max(R['jerk_pk']), stopped=R['all_stopped'],
        contact=R['contact_time'], buffer_lost=R['buffer_lost'],
        tset=R['tset'], rule_units=R['rule_units'],
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], joint=joint(R), runtime_s=R['runtime_s'])
    rec.update(extra)
    return rec


def plant_row(name, variant, ov, recs, **cfg):
    dd = [r['disp'] for r in recs if r['disp'] is not None]
    fin = [r for r in recs if r['stopped']]
    at = [r['atU'] for r in recs]
    return dict(
        name=name, variant=variant, overrides=ov, n=len(recs),
        disp_med=float(np.median(dd)) if dd else None,
        disp_min=min(dd) if dd else None, disp_max=max(dd) if dd else None,
        marker_max=max([r['marker'] for r in fin] or [None]),
        marker_med=float(np.median([r['marker'] for r in fin]))
        if fin else None,
        gap_max=max([r['gap'] for r in fin] or [None]),
        gap_med=float(np.median([r['gap'] for r in fin])) if fin else None,
        gmin_min=min(r['gmin'] for r in recs),
        atU_med=float(np.median(at)), atU_max=max(at),
        barrier_runs=sum(1 for r in recs if r['barrier'] > 0),
        barrier_max=max(r['barrier'] for r in recs),
        fallback_runs=sum(1 for r in recs if r['fallback'] > 0),
        T_f_max=max([r['T_f'] for r in recs if r['T_f'] is not None]
                    or [None]),
        contacts=sum(1 for r in recs if r['contact'] is not None),
        buffer_lost_runs=sum(1 for r in recs if r['buffer_lost']),
        stopped=sum(1 for r in recs if r['stopped']),
        pass_disp=sum(1 for r in recs if r['pass_disp']),
        pass_marker=sum(1 for r in recs if r['pass_marker']),
        pass_gap=sum(1 for r in recs if r['pass_gap']),
        joint_pass=sum(1 for r in recs if r['joint']),
        config=cfg, seeds=recs)


def plant_line(r):
    return ('  %-44s joint %2d/%d disp med/max %s/%s marker %s gap %s gmin '
            '%.2f atU med %.1f s barrier %d contacts %d stopped %d '
            '(pass d/m/g %d/%d/%d)'
            % (r['name'], r['joint_pass'], r['n'],
               fmt(r['disp_med'], '%.3f'), fmt(r['disp_max'], '%.3f'),
               fmt(r['marker_max'], '%.2f'), fmt(r['gap_max'], '%.2f'),
               r['gmin_min'], r['atU_med'], r['barrier_runs'], r['contacts'],
               r['stopped'], r['pass_disp'], r['pass_marker'],
               r['pass_gap']))


# ============================================================ MPC workers
_WB = {}


def _mpc_job(job):
    """One MPC run in a worker process (the setup is rebuilt from the design
    file and the parameters)."""
    key, design, params, kw = job
    h = json.dumps([design, params], sort_keys=True)
    if h not in _WB:
        _WB[h] = sbh.setup(json.load(open(design))['cfg'], **params)
    R = sbh.run(_WB[h], 'mpc', **kw)
    return key, slim(R)


def pool_map(jobs, workers, log, label):
    res = {}
    t0 = time.time()
    if workers <= 1:
        for j, job in enumerate(jobs):
            k, R = _mpc_job(job)
            res[k] = R
        log('    %s: %d runs in %.0f s' % (label, len(jobs), time.time() - t0))
        return res
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, R in ex.map(_mpc_job, jobs):
            res[k] = R
    log('    %s: %d runs on %d workers in %.0f s'
        % (label, len(jobs), workers, time.time() - t0))
    return res


def mpc_kw(sel):
    return dict(W=dict(sel['W']), mpc_planner=dict(sel['planner']),
                mpc_stop='detector')


# ================================================================== check
def _cmp_hr(R, H):
    """Differences of a sim_bench_hr run against a sim_hr run."""
    d = dict(
        stop_instants=maxdiff(R['tstop'], H['tstop']),
        marker_errors=maxdiff(R['marker_signed'], H['marker_signed']),
        gap_errors=maxdiff(R['gap_signed'], H['gap_signed']),
        switch_instants=maxdiff(R['tset'], H['tset']),
        activation_instants=maxdiff(R['tact'], H['tact']),
        closure_onsets=maxdiff(R['tcl'], H['tcl']),
        rule_entries=maxdiff(R['trule'], H['trule']),
        clearance_margin_minima=maxdiff(R['gmin'], H['gmin']),
        barrier_minima=maxdiff(R['hmin'], H['hmin']),
        command_minima=maxdiff(R['u_min_unit'], H['u_min']),
        command_maxima=maxdiff(R['u_max_unit'], H['u_max']),
        command_maxima_before_T_xi=maxdiff(R['u_max_pre_unit'],
                                           H['u_max_pre']),
        handoff_position_errors=maxdiff(R['e_hnd'], H['e_hnd']),
        handoff_velocity_errors=maxdiff(R['eps_hnd'], H['eps_hnd']),
        dispersion=abs(R['disp'] - H['disp']))
    tt = TOL_CHECK['time']
    tp = TOL_CHECK['position']
    ok = bool(
        all(d[k] is not None for k in d)
        and d['stop_instants'] <= tt and d['switch_instants'] <= tt
        and d['activation_instants'] <= tt and d['closure_onsets'] <= tt
        and d['rule_entries'] <= tt and d['marker_errors'] <= tp
        and d['gap_errors'] <= tp and d['clearance_margin_minima'] <= tp
        and R['stop_order'] == [int(x) for x in H['stop_order']])
    return d, ok


def _hr_summary(H):
    return dict(tstop=H['tstop'], disp=H['disp'], T_f=H['T_f'],
                marker_signed=H['marker_signed'], gap_signed=H['gap_signed'],
                tset=H['tset'], tact=H['tact'], tcl=H['tcl'],
                trule=H['trule'], gmin=H['gmin'], stop_order=H['stop_order'],
                neg_speed=H['neg_speed'], n_localized=H['n_localized'],
                config=H['config'])


def _first_event(*runs):
    """Earliest RULE entry, stop or contact instant of the runs."""
    ts = []
    for R in runs:
        for key in ('trule', 'tstop'):
            ts += [x for x in R[key] if x is not None]
        if R['contact_time'] is not None:
            ts.append(R['contact_time'])
    return min(ts) if ts else np.inf


def _rec_diff(A, C, t_first):
    """Largest |ds|, |dv| of the records strictly before t_first."""
    ta = np.asarray(A['rec']['t'])
    tc = np.asarray(C['rec']['t'])
    m = min(len(ta), len(tc))
    sel = ta[:m] < t_first - 1e-9
    if not np.allclose(ta[:m][sel], tc[:m][sel], rtol=0, atol=1e-9):
        return None, None, 0
    ds = np.abs(np.asarray(A['rec']['s'])[:, :m][:, sel]
                - np.asarray(C['rec']['s'])[:, :m][:, sel])
    dv = np.abs(np.asarray(A['rec']['v'])[:, :m][:, sel]
                - np.asarray(C['rec']['v'])[:, :m][:, sel])
    return (float(ds.max()) if ds.size else 0.0,
            float(dv.max()) if dv.size else 0.0, int(sel.sum()))


def _certified(B, R, opts, log):
    """(T5) observations of 'proposed' against the bounds of the certificate
    evaluated by certify_headref.evaluate (read-only import)."""
    try:
        import certify_headref as ch
    except Exception as ex:                           # pragma: no cover
        return dict(checked=False, note='certify_headref not importable: %s'
                    % ex)
    cc = ch.config(lam=float(opts['lam']), vhnd=float(opts['vhnd']),
                   lags=list(opts['lags']), U=float(opts['U']),
                   Uminus=float(opts['Uminus']), dbar=float(opts['dbar']),
                   eps_e=float(opts['eps_e']), eps_v=float(opts['eps_v']))
    ta = time.time()
    ev = ch.evaluate(cc)
    te = time.time() - ta
    bd = ev['bounds']
    log('  T5 certificate %s: admitted %s, failed %s (%.1f s)'
        % ({k: cc[k] for k in ('lam', 'vhnd', 'lags', 'U', 'Uminus', 'dbar',
                                'eps_e', 'eps_v')}, ev['admitted'],
           ev['failed'], te))
    rows = []

    def chk(name, obs, bound, sense='<='):
        ok = bool(obs is not None and bound is not None
                  and (obs <= bound if sense == '<=' else obs >= bound))
        rows.append(dict(observation=name, observed=obs, bound=bound,
                         sense=sense, inside=ok))
        log('  T5 %-40s observed %-13.6g %s %-13.6g inside %s'
            % (name, obs, sense, bound, ok))
    chk('dispersion [s]', R['disp'], bd['dispersion'])
    chk('completion T_f [s]', R['T_f'], bd['T_f_bar'])
    for i in range(B['n']):
        chk('|marker error| unit %d [m]' % (i + 1),
            abs(R['marker_signed'][i]), bd['marker'][i])
    for i in range(B['n'] - 1):
        chk('|terminal gap error| pair %d [m]' % (i + 2),
            abs(R['gap_signed'][i]), bd['gap'][i])
        chk('smallest clearance margin pair %d [m]' % (i + 2),
            R['gmin'][i], bd['clearance_floor'][i], '>=')
    for i in range(B['n']):
        chk('activation of unit %d [s]' % (i + 1), R['tact'][i],
            bd['activation_deadline'][i])
        chk('S2 switch of unit %d [s]' % (i + 1), R['tset'][i],
            bd['switch_deadline'][i])
        chk('closure offset of unit %d [s]' % (i + 1),
            R['tcl'][i] - B['t_0'], bd['closure_offset_bound'][i])
        chk('largest command before T_xi, unit %d' % (i + 1),
            R['u_max_pre_unit'][i], bd['u_max_before_T_xi'][i])
        chk('smallest command before T_xi, unit %d' % (i + 1),
            R['u_min_pre_unit'][i], bd['u_min_before_T_xi'][i], '>=')
    return dict(checked=True, evaluator='src/certification/'
                'certify_headref.py', config={k: cc[k] for k in (
                    'lam', 'vhnd', 'lags', 'U', 'Uminus', 'dbar', 'eps_e',
                    'eps_v', 'cell_dt')},
                admitted=ev['admitted'], failed=ev['failed'],
                bounds=bd, evaluate_runtime_s=te, rows=rows,
                n_inside=sum(1 for r in rows if r['inside']),
                n_rows=len(rows),
                all_inside=bool(all(r['inside'] for r in rows)),
                handoff_errors_in_box_at_T_xi=R['in_box_at_Txi'],
                e_hnd=R['e_hnd'], eps_hnd=R['eps_hnd'])


def part_check(opts):
    log = Log('check')
    t0 = time.time()
    log('=== part check ===')
    D, B = context(opts)
    out = header('check', D, B, opts)
    S = B['S']
    sbh.run(B, 'proposed')                          # compile / load cache
    # ---------------------------------------------------------------- T1
    import run_caseD_validate as rcv
    Nsch = int(round(S.T_END / B['T_m'])) + 2
    fifo = rcv.random_schedule(np.random.default_rng(1), Nsch, 19)
    t1 = dict(reference='sim_hr.run(scheme="zoh", h = T_u = T_m = T_c = '
                        '1 ms), same setup (sim_hr.hr_setup)',
              tolerance=dict(TOL_CHECK), cases={})
    for tag, kh, ks in (
            ('constant transport', dict(transport=B['transport']), {}),
            ('FIFO schedule seed 1', dict(fifo=fifo), dict(fifo=fifo))):
        ta = time.time()
        H = hr.run(S, scheme='zoh', h=B['dt'], T_m=B['T_m'], T_c=B['T_c'],
                   T_u=B['dt'], **kh)
        th = time.time() - ta
        R = sbh.run(B, 'proposed', **ks)
        d, ok = _cmp_hr(R, H)
        t1['cases'][tag] = dict(
            max_abs_difference=d, stop_order_equal=bool(
                R['stop_order'] == [int(x) for x in H['stop_order']]),
            n_latch_before_S2=R['n_latch_before_S2'],
            sim_hr_neg_speed=int(H['neg_speed']), passed=ok,
            sim_hr=_hr_summary(H), sim_bench_hr=slim(R),
            runtime_s=dict(sim_hr=th, sim_bench_hr=R['runtime_s']),
            schedule=None if not ks else dict(
                generator='run_caseD_validate.random_schedule('
                          'np.random.default_rng(1), N, 19)',
                N=Nsch, d_min_ms=int(fifo[1:].min()),
                d_max_ms=int(fifo[1:].max())))
        log('  T1 %-22s max |diff|: stop %.1e s, marker %.1e m, gap %.1e m, '
            'switch %.1e s, activation %.1e s, closure %.1e s, RULE %.1e s, '
            'g_min %.1e m, u_min %.1e, u_max %.1e -> %s (disp %.4f ms, '
            'sim_hr %.3f s, sim_bench_hr %.3f s)'
            % (tag, d['stop_instants'], d['marker_errors'], d['gap_errors'],
               d['switch_instants'], d['activation_instants'],
               d['closure_onsets'], d['rule_entries'],
               d['clearance_margin_minima'], d['command_minima'],
               d['command_maxima'], 'PASS' if ok else 'FAIL',
               1e3 * R['disp'], th, R['runtime_s']))
    t1['pass'] = bool(all(c['passed'] for c in t1['cases'].values()))
    # ---------------------------------------------------------------- T2
    H = hr.run(S, scheme='zoh', h=B['dt'], T_m=B['T_m'], T_c=B['T_c'],
               T_u=B['dt'], transport=B['transport'], schedules=False,
               nosched='step')
    R = sbh.run(B, 'feedback_step')
    d, ok = _cmp_hr(R, H)
    t2 = dict(reference='sim_hr.run(zoh, schedules=False, nosched="step")',
              max_abs_difference=d, passed=ok, sim_hr=_hr_summary(H),
              sim_bench_hr=slim(R))
    t2['pass'] = ok
    log('  T2 feedback_step vs sim_hr nosched=step: stop %.1e s, marker %.1e '
        'm, gap %.1e m, switch %.1e s -> %s'
        % (d['stop_instants'], d['marker_errors'], d['gap_errors'],
           d['switch_instants'], 'PASS' if ok else 'FAIL'))
    # ---------------------------------------------------------------- T3
    t3 = dict(cases={})
    skip = ('variant', 'apply_filter', 'runtime_s')
    for tag, v0 in (('benchmark takeover', None),
                    ('doubled increments', sbh.doubled_mismatch(B))):
        Rp = sbh.run(B, 'proposed', v0=v0)
        Rn = sbh.run(B, 'nofilter', v0=v0)
        diffs = []
        nf = 0
        for key in sorted(Rp):
            if key in skip:
                continue
            nf += 1
            if json.dumps(_py(Rp[key]), sort_keys=True) != \
                    json.dumps(_py(Rn.get(key)), sort_keys=True):
                diffs.append(key)
        t3['cases'][tag] = dict(fields_compared=nf, fields_different=diffs,
                                identical=not diffs,
                                barrier_time=max(Rp['t_barrier']),
                                fallback_time=sum(Rp['t_infeas']),
                                smallest_inactivity_margin=min(
                                    x for x in Rp['mmin'] if x is not None))
        log('  T3 nofilter vs proposed (%s): %d fields, %d different; '
            'smallest inactivity margin %.4f m/s^2'
            % (tag, nf, len(diffs),
               t3['cases'][tag]['smallest_inactivity_margin']))
    t3['pass'] = bool(all(c['identical'] for c in t3['cases'].values()))
    # ---------------------------------------------------------------- T4
    bounds = ([float(-x) for x in B['Uminus']], float(B['U']))
    t4 = dict(reference='sim_bench.run(variant, bounds=([-U_i^-], U)) with '
                        'the same gain (sim_bench.setup of the same cfg)',
              tolerance=TOL_PORT, hifi=[], ideal=[], loss=[])
    ok4 = True
    for var, kw in (('feedback', {}), ('cacc', dict(h_w=0.5))):
        for ovn, ov in (('default', {}), ('compensated', dict(COMP))):
            for sd in (1, 2, 3):
                A = sbh.run(B, var, plant='hifi', hifi=ov, seed=sd, **kw)
                C = sb.run(B['sb'], var, plant='hifi', hifi=ov, seed=sd,
                           bounds=bounds, **kw)
                d = dict(stop_instants=maxdiff(A['tstop'], C['tstop']),
                         marker_errors=maxdiff(A['marker_signed'],
                                               C['marker_signed']),
                         gap_errors=maxdiff(A['gap_signed'],
                                            C['gap_signed']),
                         clearance_margin_minima=maxdiff(A['gmin'],
                                                         C['gmin']),
                         switch_instants=maxdiff(A['tset'], C['tset']),
                         time_at_U=maxdiff(A['t_atU'], C['t_at0']),
                         barrier_time=maxdiff(A['t_barrier'],
                                              C['t_barrier']))
                ok = bool(all(x is not None and x <= TOL_PORT
                              for x in d.values()))
                ok4 = ok4 and ok
                t4['hifi'].append(dict(variant=var, h_w=kw.get('h_w'),
                                       plant=ovn, seed=sd,
                                       max_abs_difference=d, passed=ok))
            log('  T4 perturbed plant %-8s %-11s seeds 1-3: max stop %.1e s, '
                'marker %.1e m, gap %.1e m -> %s'
                % (var, ovn, max(r['max_abs_difference']['stop_instants']
                                 for r in t4['hifi'][-3:]),
                   max(r['max_abs_difference']['marker_errors']
                       for r in t4['hifi'][-3:]),
                   max(r['max_abs_difference']['gap_errors']
                       for r in t4['hifi'][-3:]),
                   all(r['passed'] for r in t4['hifi'][-3:])))
        A = sbh.run(B, var, record=True, **kw)
        C = sb.run(B['sb'], var, bounds=bounds, record=True, **kw)
        tf = _first_event(A, C)
        ds, dv, nr = _rec_diff(A, C, tf)
        ok = bool(ds is not None and ds <= TOL_PORT and dv <= TOL_PORT)
        ok4 = ok4 and ok
        t4['ideal'].append(dict(
            variant=var, h_w=kw.get('h_w'), window_end=tf, n_samples=nr,
            max_abs_ds=ds, max_abs_dv=dv, passed=ok,
            terminal=dict(stop_instants=maxdiff(A['tstop'], C['tstop']),
                          marker_errors=maxdiff(A['marker_signed'],
                                                C['marker_signed']),
                          gap_errors=maxdiff(A['gap_signed'],
                                             C['gap_signed']),
                          rule_units=[A['rule_units'], C['rule_units']])))
        log('  T4 ideal %-8s before the first RULE/stop (%.3f s): |ds| %.1e m '
            '|dv| %.1e m/s -> %s; terminal (RULE convention differs): stop '
            '%.1e s, marker %.1e m'
            % (var, tf, ds, dv, ok, t4['ideal'][-1]['terminal']
               ['stop_instants'], t4['ideal'][-1]['terminal']
               ['marker_errors']))
        for tl in T_LOSS:
            for em in (None, EM_UNIT):
                for wd in (None, 0.25):
                    A = sbh.run(B, var, scenario='loss', t_loss=tl,
                                emergency_unit=em, T_wd=wd, record=True, **kw)
                    C = sb.run(B['sb'], var, scenario='loss', t_loss=tl,
                               emergency_unit=em, T_wd=wd, bounds=bounds,
                               record=True, **kw)
                    tf = _first_event(A, C)
                    ds, dv, nr = _rec_diff(A, C, tf)
                    same_c = (A['contact_time'] is None) == \
                        (C['contact_time'] is None)
                    ok = bool(ds is not None and ds <= TOL_PORT
                              and dv <= TOL_PORT and same_c)
                    ok4 = ok4 and ok
                    t4['loss'].append(dict(
                        variant=var, h_w=kw.get('h_w'), t_loss=tl,
                        emergency_unit=em, T_wd=wd, window_end=tf,
                        n_samples=nr, max_abs_ds=ds, max_abs_dv=dv,
                        contact=[A['contact_time'], C['contact_time']],
                        same_contact_outcome=same_c,
                        gmin=[A['gmin_min'], C['gmin_min']], passed=ok))
        rl = [r for r in t4['loss'] if r['variant'] == var]
        log('  T4 loss %-8s %d runs: max |ds| %.1e m |dv| %.1e m/s before the '
            'first RULE/stop/contact; contact outcome equal in %d; max '
            '|contact instant diff| %.1e s -> %s'
            % (var, len(rl), max(r['max_abs_ds'] for r in rl),
               max(r['max_abs_dv'] for r in rl),
               sum(1 for r in rl if r['same_contact_outcome']),
               max([abs(r['contact'][0] - r['contact'][1]) for r in rl
                    if r['contact'][0] is not None
                    and r['contact'][1] is not None] or [0.0]),
               all(r['passed'] for r in rl)))
    t4['pass'] = bool(ok4)
    t4['note'] = ('after the first RULE entry sim_bench and sim_bench_hr '
                  'differ by convention: sim_bench enters RULE at the next '
                  'grid instant after the zero-speed instant of the '
                  'predecessor, sim_bench_hr (as sim_hr and the '
                  'specification) at that instant with an immediate command '
                  'update; the perturbed plant detects standstill at control '
                  'instants in both, so it agrees to the end of the run')
    # ---------------------------------------------------------------- T5
    Rp = sbh.run(B, 'proposed')
    t5 = _certified(B, Rp, opts, log)
    out.update(T1=t1, T2=t2, T3=t3, T4=t4, T5=t5,
               all_pass=bool(t1['pass'] and t2['pass'] and t3['pass']
                             and t4['pass']))
    log('  check: T1 %s, T2 %s, T3 %s, T4 %s; T5 %s'
        % (t1['pass'], t2['pass'], t3['pass'], t4['pass'],
           ('%d of %d inside' % (t5['n_inside'], t5['n_rows']))
           if t5.get('checked') else t5.get('note')))
    finish('check', out, opts, log, t0)
    return out['all_pass']


# =================================================================== base
def base_specs(sel):
    return [
        ('proposed', 'proposed', {}),
        ('nofilter', 'nofilter', {}),
        ('single', 'single', {}),
        ('nofeedback', 'nofeedback', {}),
        ('feedback', 'feedback', {}),
        ('feedback_step', 'feedback_step', {}),
        ('cacc h_w=0.2', 'cacc', dict(h_w=0.2)),
        ('cacc h_w=0.5', 'cacc', dict(h_w=0.5)),
        ('cacc h_w=1.0', 'cacc', dict(h_w=1.0)),
        ('pid hand-tuned', 'pid', dict(gains=dict(sb.K_PID))),
        ('plans', 'plans', {}),
        ('mpc', 'mpc', mpc_kw(sel)),
        ('nofeedback, filter off', 'nofeedback', dict(apply_filter=False)),
        ('feedback, filter off', 'feedback', dict(apply_filter=False)),
        ('feedback_step, filter off', 'feedback_step',
         dict(apply_filter=False)),
        ('cacc h_w=0.5, filter off', 'cacc',
         dict(h_w=0.5, apply_filter=False)),
        ('pid hand-tuned, filter off', 'pid',
         dict(gains=dict(sb.K_PID), apply_filter=False)),
    ]


def part_base(opts):
    log = Log('base')
    t0 = time.time()
    log('=== part base ===')
    D, B = context(opts)
    out = header('base', D, B, opts)
    sel = sbh.mpc_selection()
    out['mpc_selection'] = sel
    log('  MPC weights %s planner %s from %s (tuned with the input set %s; '
        'not retuned here)' % (sel['W'], sel['planner'], sel['source'],
                               sel['input_set_of_tuning']))
    v0h = sbh.doubled_mismatch(B)
    blocks = {}
    for tag, v0 in (('benchmark', None), ('doubled', v0h)):
        rows = []
        runs = {}
        log('  -- takeover: %s %s' % (tag, list(B['v0']) if v0 is None
                                      else v0))
        for name, var, kw in base_specs(sel):
            R = sbh.run(B, var, v0=v0, **kw)
            r = row_of(name, R, runtime_s=R['runtime_s'])
            rows.append(r)
            runs[name] = slim(R)
            log(row_line(r))
        blocks[tag] = dict(v0=list(B['v0']) if v0 is None else v0,
                           rows=rows, runs=runs)
    out['benchmark'] = blocks['benchmark']
    out['doubled'] = blocks['doubled']
    out['columns'] = dict(
        disp='stopping-time dispersion, last stop minus first stop [s] '
             '(None: not every unit stopped, or contact)',
        T_f='last stop instant [s]', marker='largest |marker error| [m]',
        gap='largest |terminal gap error| [m]',
        gmin='smallest clearance margin g_i = c_i - s_m over the run and the '
             'pairs [m] (intersample minimum)',
        u_abs_max='largest |applied net command| before the latch '
                  '[m/s^2] (override braking included)',
        atU='time with the applied command at the traction limit U, worst '
            'unit [s]',
        nonneg='time with the applied command at 0 or above, worst unit [s] '
               '(includes the startup of follower i, (i-1) x 19 ms with a '
               'zero received command)',
        nom_above_U='time with u_nom > U, worst unit [s]',
        clipU='time in which the input set clipped the filter output at U, '
              'worst unit [s]',
        barrier='time in which the filter lowered the applied command '
                '(fallback included), worst unit [s]',
        fallback='time with u_cbf < -U_i^- (fallback -U_i^-), summed over '
                 'the followers [s]',
        rule_units='units that entered RULE', stop_order='units in the '
        'order of their stops', met='every unit stopped, no contact, '
        'dispersion <= 1 s, markers <= 3 m, gaps <= 1 m')
    finish('base', out, opts, log, t0)


# ================================================================= retune
FB_CONTROLLERS = (('feedback', 'feedback', None),
                  ('feedback_step', 'feedback_step', None),
                  ('cacc h_w=0.2', 'cacc', 0.2),
                  ('cacc h_w=0.5', 'cacc', 0.5),
                  ('cacc h_w=1.0', 'cacc', 1.0))


def part_retune(opts):
    log = Log('retune')
    t0 = time.time()
    log('=== part retune ===')
    D, B = context(opts)
    out = header('retune', D, B, opts)
    v0h = sbh.doubled_mismatch(B)
    lam0 = float(B['lam'])
    ctrl = {}
    for name, var, hw in FB_CONTROLLERS + (('proposed (information)',
                                            'proposed', None),):
        kw = {} if hw is None else dict(h_w=hw)
        cands = []
        for lam in FB_LAMBDAS:
            Bg = sbh.with_gain(B, lam)
            S = sbh.run(Bg, var, v0=v0h, **kw)
            Hh = sbh.run(Bg, var, **kw)
            cands.append(dict(
                lam=lam, beta=Bg['beta'], gamma=Bg['gamma'], score=score(S),
                train=row_of(name + ' train', S), heldout=row_of(
                    name + ' held-out', Hh),
                train_run=slim(S), heldout_run=slim(Hh)))
            c = cands[-1]
            log('  %-22s lam=%.2f score %-26s | train disp=%s marker=%.3f '
                'gap=%.3f contact=%s | held-out disp=%s marker=%.3f gap=%.3f '
                'gmin=%.3f atU=%.2f barrier=%.2f contact=%s met=%s'
                % (name, lam, c['score'], fmt(S['disp']),
                   S['marker_abs_max'], S['gap_abs_max'],
                   fmt(S['contact_time'], '%.3f', '-'), fmt(Hh['disp']),
                   Hh['marker_abs_max'], Hh['gap_abs_max'], Hh['gmin_min'],
                   max(Hh['t_atU']), max(Hh['t_barrier']),
                   fmt(Hh['contact_time'], '%.3f', '-'), Hh['pass_all']))
        best = min(cands, key=lambda c: tuple(c['score']))
        des = [c for c in cands if c['lam'] == lam0]
        ctrl[name] = dict(
            variant=var, h_w=hw, candidates=cands, best_lam=best['lam'],
            best_score=best['score'], best_train=best['train'],
            best_heldout=best['heldout'],
            design_lam=lam0,
            design_score=des[0]['score'] if des else None,
            design_train=des[0]['train'] if des else None,
            design_heldout=des[0]['heldout'] if des else None,
            lams_without_violation_train=[c['lam'] for c in cands
                                          if c['score'][0] == 0],
            lams_pass_heldout=[c['lam'] for c in cands
                               if c['heldout']['met']],
            lams_rejected=[c['lam'] for c in cands if c['score'][0] >= 1e9],
            informational=var == 'proposed')
        log('  %-22s selected lam=%.2f score %s; held-out disp=%s marker=%s '
            'gap=%s atU=%.2f met=%s; design lam=%.2f score %s; training '
            'without violation: %s; held-out met: %s'
            % (name, best['lam'], best['score'],
               fmt(best['heldout']['disp']),
               fmt(best['heldout']['marker'], '%.3f'),
               fmt(best['heldout']['gap'], '%.3f'), best['heldout']['atU'],
               best['heldout']['met'], lam0,
               des[0]['score'] if des else None,
               ctrl[name]['lams_without_violation_train'],
               ctrl[name]['lams_pass_heldout']))
    # ---- PID over PID_GRID
    cands = []
    for K in PID_GRID:
        g = dict(sb.K_PID, **K)
        S = sbh.run(B, 'pid', gains=g, v0=v0h)
        Hh = sbh.run(B, 'pid', gains=g)
        cands.append(dict(K=K, gains=g, score=score(S),
                          train=row_of('pid train', S),
                          heldout=row_of('pid held-out', Hh),
                          train_run=slim(S), heldout_run=slim(Hh)))
        log('  pid %-40s score %-26s | train disp=%s marker=%.3f gap=%.3f | '
            'held-out disp=%s marker=%.3f gap=%.3f gmin=%.3f atU=%.2f met=%s'
            % (K, cands[-1]['score'], fmt(S['disp']), S['marker_abs_max'],
               S['gap_abs_max'], fmt(Hh['disp']), Hh['marker_abs_max'],
               Hh['gap_abs_max'], Hh['gmin_min'], max(Hh['t_atU']),
               Hh['pass_all']))
    best = min(cands, key=lambda c: tuple(c['score']))
    S0 = sbh.run(B, 'pid', gains=dict(sb.K_PID), v0=v0h)
    H0 = sbh.run(B, 'pid', gains=dict(sb.K_PID))
    ctrl['pid'] = dict(
        variant='pid', grid=PID_GRID, candidates=cands,
        best_K=best['K'], best_gains=best['gains'], best_score=best['score'],
        best_train=best['train'], best_heldout=best['heldout'],
        hand_tuned=dict(sb.K_PID), hand_tuned_score=score(S0),
        hand_tuned_train=row_of('pid hand-tuned train', S0),
        hand_tuned_heldout=row_of('pid hand-tuned held-out', H0),
        K_without_violation_train=[c['K'] for c in cands
                                   if c['score'][0] == 0],
        K_pass_heldout=[c['K'] for c in cands if c['heldout']['met']],
        critically_damped_Kd_of_best=2.0 * np.sqrt(best['K']['Kp']))
    log('  pid selected %s score %s; held-out disp=%s marker=%s gap=%s '
        'met=%s; hand-tuned score %s held-out met=%s'
        % (best['K'], best['score'], fmt(best['heldout']['disp']),
           fmt(best['heldout']['marker'], '%.3f'),
           fmt(best['heldout']['gap'], '%.3f'), best['heldout']['met'],
           score(S0), H0['pass_all']))
    out['controllers'] = ctrl
    out['protocol'] = dict(
        source='run_bench_studies.py part fbtune (FB_LAMBDAS, score) and '
               'part base (PID_GRID); constants imported read-only',
        family='critically damped S2 feedback with one common gain lam for '
               'every unit, the head included: beta = lam^2, gamma = 2 lam; '
               'every other constant unchanged (alpha, detector, filter, '
               'input set, channel)',
        grid=list(FB_LAMBDAS), pid_grid=PID_GRID,
        pid_Imax=sb.K_PID['Imax'],
        training='ideal plant, doubled-increment takeover (increments '
                 'doubled, head speed unchanged), filter on',
        heldout='ideal plant, benchmark takeover, filter on',
        objective=score.__doc__,
        selection='lexicographic minimum of the training score, first in '
                  'grid order on ties')
    out['v0_training'] = v0h
    out['selected'] = dict(
        feedback=ctrl['feedback']['best_lam'],
        feedback_step=ctrl['feedback_step']['best_lam'],
        cacc={str(hw): ctrl['cacc h_w=%.1f' % hw]['best_lam']
              for hw in HW_LIST},
        pid=ctrl['pid']['best_gains'])
    # ---- consistency with base.json (design-gain rows)
    checks = {}
    pb = os.path.join(OUT, 'base.json')
    if os.path.exists(pb):
        BJ = json.load(open(pb))
        if BJ['design']['cfg_used_sha256'] == cfg_hash(B['cfg']):
            rb = {r['name']: r for r in BJ['benchmark']['rows']}
            for name, var, hw in FB_CONTROLLERS:
                a = dict(ctrl[name]['design_heldout'] or {})
                b = dict(rb.get(name if hw is None else
                                'cacc h_w=%.1f' % hw, {}))
                for k_ in ('name', 'runtime_s'):
                    a.pop(k_, None)
                    b.pop(k_, None)
                checks['base.json row %s' % name] = \
                    json.loads(json.dumps(_py(a))) == \
                    json.loads(json.dumps(_py(b)))
    out['checks'] = checks
    log('  checks against base.json: %s' % checks)
    finish('retune', out, opts, log, t0)


# =================================================================== loss
def part_loss(opts):
    log = Log('loss')
    t0 = time.time()
    log('=== part loss ===')
    D, B = context(opts)
    out = header('loss', D, B, opts)
    RT = load_part('retune', B)
    lam_fb = float(RT['selected']['feedback'])
    lam0 = float(B['lam'])
    ctrls = [('proposed', 'proposed', lam0, 'design'),
             ('feedback design', 'feedback', lam0, 'design'),
             ('feedback retuned', 'feedback', lam_fb, 'retuned'),
             ('plans', 'plans', lam0, 'design')]
    rows = []
    runs = []
    summ = []
    for tag, var, lam, gtag in ctrls:
        Bg = sbh.with_gain(B, lam)
        for em in (None, EM_UNIT):
            for wd in T_WD:
                grp = []
                for tl in T_LOSS:
                    R = sbh.run(Bg, var, scenario='loss', t_loss=tl,
                                emergency_unit=em, T_wd=wd)
                    r = loss_row(R, controller=tag, gain=gtag)
                    rows.append(r)
                    grp.append(r)
                    runs.append(dict(slim(R), controller=tag, gain=gtag))
                    log(loss_line(tag, r))
                comp = [r for r in grp if r['arrival_completed']]
                summ.append(dict(
                    controller=tag, variant=var, lam=lam, gain=gtag,
                    emergency_unit=em, T_wd=wd, n=len(grp),
                    n_contact=sum(1 for r in grp if r['contact']),
                    t_loss_contact=[r['t_loss'] for r in grp if r['contact']],
                    n_buffer_lost=sum(1 for r in grp if r['buffer_lost']),
                    n_completed=len(comp),
                    n_met=sum(1 for r in grp if r['pass_all']),
                    n_aborted=sum(1 for r in grp if r['override_units']),
                    gmin_min=min(r['gmin'] for r in grp),
                    disp_max_completed=max([r['disp'] for r in comp]
                                           or [None]),
                    marker_max_completed=max([r['marker'] for r in comp]
                                             or [None]),
                    gap_max_completed=max([r['gap'] for r in comp]
                                          or [None])))
                s_ = summ[-1]
                log('  == %-18s em=%-4s T_wd=%-4s contact %d/%d %s, buffer '
                    'lost %d, completed %d, met %d, aborted %d, gmin min '
                    '%.3f, disp max (completed) %s'
                    % (tag, em, wd, s_['n_contact'], s_['n'],
                       s_['t_loss_contact'], s_['n_buffer_lost'],
                       s_['n_completed'], s_['n_met'], s_['n_aborted'],
                       s_['gmin_min'], fmt(s_['disp_max_completed'])))
    Rn = sbh.run(B, 'proposed', record=True, rec_dt=1.0)
    st = []
    for tl in T_LOSS:
        j = int(round(tl))
        s = Rn['rec']['s'][:, j]
        v = Rn['rec']['v'][:, j]
        u = Rn['rec']['u'][:, j]
        st.append(dict(t=float(Rn['rec']['t'][j]), v=[float(x) for x in v],
                       u=[float(x) for x in u],
                       clearance=[float(s[i - 1] - s[i] - B['ell'])
                                  for i in range(1, B['n'])]))
        log('  nominal state at %4.0f s: v %s, clearances %s, u %s'
            % (tl, np.round(v, 3).tolist(),
               [round(x, 2) for x in st[-1]['clearance']],
               np.round(u, 3).tolist()))
    out.update(rows=rows, summary=summ, runs=runs,
               nominal_state_at_loss=st, t_loss=list(T_LOSS),
               T_wd=list(T_WD), emergency_unit=EM_UNIT,
               gains=dict(design=lam0, feedback_retuned=lam_fb,
                          retune_source='studies/retune.json'),
               scenario=dict(
                   loss='every link fails at the first grid instant at or '
                        'after t_loss; every receiver sets its held command '
                        'to zero and keeps its flags and mode; no message or '
                        'flag is delivered afterwards',
                   emergency='unit 3 applies -amax_3 = -1.38 m/s^2 from the '
                             'loss instant until it rests (laws, filter and '
                             'input set bypassed)',
                   watchdog='a follower that has received no packet for '
                            'T_wd applies -amax_i until it rests (arrival '
                            'aborted)',
                   completion='every unit stopped, no contact, no override '
                              '(no watchdog braking, no emergency braking)'))
    finish('loss', out, opts, log, t0)


# ================================================================== plant
def part_plant(opts):
    log = Log('plant')
    t0 = time.time()
    log('=== part plant ===')
    D, B = context(opts)
    out = header('plant', D, B, opts)
    RT = load_part('retune', B)
    selr = RT['selected']
    lam0 = float(B['lam'])
    lam_fb = float(selr['feedback'])
    lam_fs = float(selr['feedback_step'])
    g_pid = dict(selr['pid'])
    sel = sbh.mpc_selection()
    out['mpc_selection'] = sel
    out['retuned'] = dict(selr, source='studies/retune.json')
    specs = []
    for cn, ov in (('default', {}), ('compensated', dict(COMP))):
        specs += [
            ('proposed, %s' % cn, 'proposed', lam0, ov, {}),
            ('feedback design lam=%.2f, %s' % (lam0, cn), 'feedback', lam0,
             ov, {}),
            ('feedback retuned lam=%.2f, %s' % (lam_fb, cn), 'feedback',
             lam_fb, ov, {}),
            ('feedback_step retuned lam=%.2f, %s' % (lam_fs, cn),
             'feedback_step', lam_fs, ov, {})]
        for hw in HW_LIST:
            lh = float(selr['cacc'][str(hw)])
            specs.append(('cacc h_w=%.1f retuned lam=%.2f, %s' % (hw, lh, cn),
                          'cacc', lh, ov, dict(h_w=hw)))
        specs += [
            ('pid retuned, %s' % cn, 'pid', lam0, ov, dict(gains=g_pid)),
            ('plans, %s' % cn, 'plans', lam0, ov, {})]
    for nm, ov in PLANT_VAR_ROWS:
        for cn, co in (('default', {}), ('compensated', dict(COMP))):
            specs.append(('proposed, %s, %s' % (nm, cn), 'proposed', lam0,
                          dict(ov, **co), {}))
    rows = []
    for name, var, lam, ov, kw in specs:
        Bg = sbh.with_gain(B, lam)
        recs = []
        for sd in SEEDS:
            R = sbh.run(Bg, var, plant='hifi', hifi=ov, seed=sd, **kw)
            recs.append(plant_rec(R))
        row = plant_row(name, var, ov, recs, lam=lam,
                        gains=kw.get('gains'), h_w=kw.get('h_w'),
                        hifi=dict(sb.HIFI_DEFAULT, **ov))
        rows.append(row)
        log(plant_line(row))
    # ---- MPC (default plant; sim_bench's MPC has no compensation)
    params = setup_params(opts)
    jobs = [(sd, opts['design'], params,
             dict(plant='hifi', seed=sd, T_end=B['T_xi'] + 120.0,
                  **mpc_kw(sel))) for sd in SEEDS]
    res = pool_map(jobs, int(opts['workers']), log, 'mpc perturbed plant')
    recs = [plant_rec(res[sd]) for sd in SEEDS]
    row = plant_row('mpc (selected weights), default', 'mpc', {}, recs,
                    W=sel['W'], planner=sel['planner'],
                    hifi=dict(sb.HIFI_DEFAULT),
                    note='sim_bench MPC: no compensation option; the '
                         'compensated row is not run')
    row['runtime_solve_ms_mean'] = [res[sd]['runtime_solve_ms_mean']
                                    for sd in SEEDS]
    row['n_solver_not_converged'] = [res[sd]['n_solver_not_converged']
                                     for sd in SEEDS]
    rows.append(row)
    log(plant_line(row))
    # ---- check of the port: the rows of 'proposed' with the variations
    #      reuse sim_bench's plant constants
    out.update(rows=rows, seeds=list(SEEDS),
               variation_rows=[dict(name=n, overrides=o)
                               for n, o in PLANT_VAR_ROWS],
               variation_rows_equal_previous=bool(all(
                   any(n == pn and json.dumps(_py(o), sort_keys=True)
                       == json.dumps(_py(po), sort_keys=True)
                       for pn, po in rbs.PLANT_ROWS)
                   for n, o in PLANT_VAR_ROWS)),
               compensation=dict(COMP),
               note='net command in [-U_i^-, U]; the compensation is added '
                    'after the input set (actuator command may exceed U); '
                    'standstill detection without the t > T_xi gate of '
                    'sim_bench (sim_bench_hr docstring); plans and mpc run '
                    'in sim_bench (with its gate)')
    finish('plant', out, opts, log, t0)


# =================================================================== main
PARTS = dict(check=part_check, base=part_base, retune=part_retune,
             loss=part_loss, plant=part_plant)
ALL = ['check', 'base', 'retune', 'loss', 'plant']


def parse(argv):
    opts = dict(DEFAULT_OPTS)
    parts = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith('--'):
            key = a[2:]
            if key not in opts or i + 1 >= len(argv):
                raise SystemExit('unknown or incomplete option %s' % a)
            val = argv[i + 1]
            if key == 'design':
                opts[key] = os.path.abspath(val)
            elif key == 'workers':
                opts[key] = int(val)
            elif key == 'lags':
                opts[key] = [int(x) for x in val.split(',')]
            elif key == 'tag':
                opts[key] = str(val)
            else:
                opts[key] = float(val)
            i += 2
        else:
            parts.append(a)
            i += 1
    if parts == ['all']:
        parts = list(ALL)
    for p in parts:
        if p not in PARTS:
            raise SystemExit('unknown part %s' % p)
    return parts, opts


if __name__ == '__main__':
    parts_, opts_ = parse(sys.argv[1:])
    if not parts_:
        print(__doc__)
        sys.exit(0)
    TAG['value'] = opts_['tag']
    t_all = time.time()
    for p_ in parts_:
        ok_ = PARTS[p_](opts_)
        if p_ == 'check' and ok_ is False:
            raise SystemExit('check: acceptance tests failed; the studies '
                             'are not run')
    print('total %.1f s' % (time.time() - t_all))
