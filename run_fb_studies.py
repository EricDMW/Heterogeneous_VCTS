#!/usr/bin/env python3
r"""run_fb_studies.py -- studies of the feedback-only arrival (FB design,
data/new_d19/fbref/design_fb.json and notes_fb.md), 2026-09-29.

Adapted from run_hr_studies.py (studies of the head-only reference design of
2026-09-28, which stays unchanged with its records in
data/new_d19/headref/studies/).

Simulator: src/simulation/sim_bench_hr.py on the FB configuration (the cfg of
certify_fb.config() with the selected shaped reference of the design record).
A shaped setup has no schedule pieces, so every sim_bench_hr variant
regulates to the parking gap from takeover; 'proposed' is the FB law, run for
run identical to 'feedback' and 'feedback_step' on this setup (part check,
T3).  Outputs go to data/new_d19/fbref/studies/<part>.json and the log of a
part to logs_ext/fbref/studies_<part>.log.  Nothing else is written.

Design parameters (command-line options, recorded in every output; defaults =
the selected design of design_fb.json):
  --lam 0.065     common gain, beta = lam^2, gamma = 2 lam for every unit
                  (default: design_fb.json['selected']['lam'])
  --U 0.02        traction limit [m/s^2]
  --Uminus 1.2    braking limit [m/s^2]
  --eps_e 0.05 --eps_v 0.005   handoff box [m, m/s]
  --dbar 0.020    age bound [s]; T_m = T_c = 1 ms, constant transport
                  dbar - T_m (19 ms)
  --vhnd 0.01     handoff pad of the certificate
  --lags 41,41    entry-curve lags of the certificate
  --design PATH   design record (default data/new_d19/fbref/design_fb.json)
  --workers N     processes for the MPC runs and the certificate scans
                  (default 10)
  --tag NAME      suffix of the output and log names (<part>_NAME.json,
                  studies_<part>_NAME.log)

Comparators (every one with the same takeover state, input set [-U_i^-, U],
channel, detector, filter, RULE and LATCH):
  proposed     the FB law: followers COAST, S1, S2 on e_i = c_i - d_s,i,
               eps_i = v_{i-1} - v_i with the received command as
               feedforward; head S1/S2 on its reference errors with the
               feedforward a_r of the shaped reference, BRAKE at -a_b from
               T_xi; messages: applied command and matching flag.
  nofilter     'proposed' with the filter removed (input set kept).
  nofeedback   feedforward only: every follower applies the received command,
               the head a_r, from takeover (mode FF); RULE, LATCH and the
               filter kept.
  unshaped     the same followers, gains and head law with an unshaped head
               reference: one constant deceleration from takeover to T_xi,
               chosen so that v_r(T_xi) equals v_r(T_xi) of the design at the
               same T_xi (exact rationals; built as a shaped plan with a
               single piece).  Its markers are the planned stop positions of
               its own reference.
  cacc         time headway: spacing target d_s,i + h_w v_i, single stage.
  pid          PID on the errors to the parking gaps, single stage (hand-tuned
               gains of sim_bench; PID_GRID in part retune).
  mpc          cooperative arrival MPC of sim_bench (weights and planner of
               data/new_d19/caseD/studies/mpc.json, tuned there with the
               input set [-1.2, 0]; not retuned) through sim_bench_hr's
               external variant.  sim_bench.setup places the units at the
               planned positions of the shaped plan, i.e. at the parking
               gaps; the driver substitutes the takeover positions of
               sim_bench_hr.setup (clearances c0) into the sim_bench inputs.
               Nothing else is changed (port check T4 uses the same
               substitution).
  stored-schedule design of the previous round (reference only, outside the
               information model: every follower stores a gap-closing
               schedule): numbers copied from
               data/new_d19/headref/studies/base.json, never rerun here.

Parts
  check   (T3) identity of 'nofilter', 'feedback' and 'feedback_step' with
          'proposed', field by field, on both takeovers.  (T4) port checks of
          'feedback' and 'cacc' (h_w 0.5) against sim_bench on the same
          configuration with the substituted takeover positions: perturbed
          plant (seeds 1-3, default and compensated, tolerance 1e-9), ideal
          plant nominal and loss scenarios (records before the first RULE
          entry, stop or contact; afterwards the RULE conventions differ).
          (T5) observations of 'proposed' (ideal plant, benchmark takeover,
          record at 1 ms) against the bounds of the certificate
          certify_fb.evaluate(cfg): dispersion, completion, markers, terminal
          gaps, clearance floors, activation and switch deadlines, command
          bounds before T_xi, handoff box at T_xi, entry instants, velocity
          floor, pre-receipt and post-receipt command bounds after T_xi.
  base    every comparator on the benchmark takeover and on the takeover with
          doubled speed increments, filter-off versions of the comparators;
          time with a positive applied command from a 1 ms record;
          certificates (informative) of the design reference and of the
          unshaped reference at both takeovers and of the unshaped reference
          over the gain grid; the rows of the previous round (reference).
  retune  fbtune protocol of run_bench_studies.py: one common gain lam in
          FB_LAMBDAS (0.05, ..., 0.20, 0.25, 0.30, 0.40, 0.50) for 'feedback'
          (the proposed law with a retuned gain), 'unshaped' and 'cacc'
          (h_w 0.2, 0.5, 1.0), PID_GRID (27 triples) for 'pid'; training on
          the doubled-increment takeover, held-out on the benchmark
          takeover; objective run_bench_studies.score; selection:
          lexicographic minimum of the training score, first in grid order
          on ties.  The design gain (not on the grid) is run for
          information; 'proposed' over the grid is compared with 'feedback'.
  loss    t_loss in T_LOSS = (15, 30, 60, 90, 120, 140, 145) s (the instants
          of the previous study, (15, 25, 35, 50, 70, 84) s with T_xi =
          86.142 s, scaled to T_xi = 148.75 s: 145 s is 84 s scaled);
          'proposed' and 'feedback' at its retuned gain (retune.json), each
          without and with emergency braking of unit 3 and with the watchdog
          T_wd in (none, 0.10, 0.25, 0.50, 1.00) s; the state of the nominal
          run at the loss instants.
  plant   perturbed plant, seeds 1-20, default and with compensation of grade
          and resistance (grade_ff = res_ff = 1): 'proposed' and the retuned
          alternatives ('feedback', 'unshaped', 'cacc' for every h_w, 'pid');
          'mpc' on the default plant only (sim_bench's MPC has no
          compensation); the variations of the previous study (actuator lag
          0, 0.5, 1.0 s, no grade, no resistance) for 'proposed', without and
          with compensation.
  all     check base retune loss plant

Usage: python -B run_fb_studies.py <part> ... [options]
"""
import copy
import hashlib
import json
import os
import platform
import sys
import time
from fractions import Fraction as Fr

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
for _p in (CODE, os.path.join(CODE, 'src', 'simulation'),
           os.path.join(CODE, 'src', 'certification')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import numpy as np                                       # noqa: E402
import sim_bench_hr as sbh                               # noqa: E402
import sim_bench as sb                                   # noqa: E402
import certify_fb as cf                                  # noqa: E402
import run_bench_studies as rbs                          # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'fbref', 'studies')
LOGDIR = os.path.join(CODE, 'logs_ext', 'fbref')
DESIGN = os.path.join(CODE, 'data', 'new_d19', 'fbref', 'design_fb.json')
HR_BASE = os.path.join(CODE, 'data', 'new_d19', 'headref', 'studies',
                       'base.json')
MPC_SEL = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'studies',
                       'mpc.json')

# ---- protocol of the previous studies (run_bench_studies.py, read-only)
REQ = dict(rbs.REQ)
FB_LAMBDAS = tuple(rbs.FB_LAMBDAS)
PID_GRID = list(rbs.PID_GRID)
SEEDS = tuple(rbs.SEEDS)
score = rbs.score
joint = rbs.joint
T_LOSS_PREV = tuple(rbs.T_LOSS)          # (15, 25, 35, 50, 70, 84) s
T_XI_PREV = 86.142                       # T_xi of the previous study [s]
# ---- this study
T_LOSS = (15.0, 30.0, 60.0, 90.0, 120.0, 140.0, 145.0)
T_WD = (None, 0.10, 0.25, 0.50, 1.00)
T_WD_PORT = (None, 0.25)
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
TOL_PORT = 1e-9
DEFAULT_OPTS = dict(lam=None, U=0.02, Uminus=1.2, eps_e=0.05, eps_v=0.005,
                    dbar=0.020, vhnd=0.01, lags=[41, 41], design=DESIGN,
                    workers=10, tag='')
TAG = dict(value='')
EVIDENCE = dict(
    check='simulation against certificate: T3 and T4 compare simulations '
          'field by field (simulation); T5 compares observations of one '
          'simulation (sim_bench_hr, ideal plant, 1 kHz) with the bounds of '
          'the float evaluator certify_fb.evaluate (certificate)',
    base='simulation (sim_bench_hr ideal plant, 1 kHz; mpc: sim_bench); the '
         'certificate rows are evaluations of the float evaluator '
         'certify_fb.evaluate (certificate, informative); the rows of the '
         'previous round are copied from its record (simulation, not rerun)',
    retune='simulation; the selection is a search over the finite grids '
           'FB_LAMBDAS and PID_GRID (sampled search)',
    loss='simulation (sim_bench_hr ideal plant, 1 kHz)',
    plant='simulation (perturbed plant of sim_bench, 20 noise seeds: '
          'sampled); mpc: sim_bench')


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
    if isinstance(o, Fr):
        return str(o)
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
    return hashlib.sha256(json.dumps(_py(cfg), sort_keys=True).encode()
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


def cert_params(opts):
    return dict(setup_params(opts), lags=[int(x) for x in opts['lags']])


def fb_cfg(opts, plan, **extra):
    """FB configuration (certify_fb.config) with the reference plan and the
    parameters of opts; extra entries (e.g. v0) are merged."""
    kw = cert_params(opts)
    kw.update(extra)
    return cf.config(plan=copy.deepcopy(plan), **kw)


def exact_v_end(plan):
    """v_r(T_xi) of a shaped plan (exact rational)."""
    v = Fr(str(plan['v_xi']))
    T = Fr(str(plan['T_xi']))
    pcs = [(Fr(str(t)), Fr(str(a))) for t, a in plan['pieces']]
    for j, (t, a) in enumerate(pcs):
        t1 = pcs[j + 1][0] if j + 1 < len(pcs) else T
        v += a * (t1 - t)
    return v


def exact_distance(plan):
    """Planned distance of the reference on [0, T_xi] (exact rational)."""
    v = Fr(str(plan['v_xi']))
    T = Fr(str(plan['T_xi']))
    x = Fr(0)
    pcs = [(Fr(str(t)), Fr(str(a))) for t, a in plan['pieces']]
    for j, (t, a) in enumerate(pcs):
        t1 = pcs[j + 1][0] if j + 1 < len(pcs) else T
        L = t1 - t
        x += v * L + a * L * L / 2
        v += a * L
    return x


def unshaped_plan(plan):
    """Head reference with one constant deceleration from takeover to T_xi
    that reaches v_r(T_xi) of plan at the same T_xi (a shaped plan with a
    single piece) and its data."""
    v_in = Fr(str(plan['v_xi']))
    T = Fr(str(plan['T_xi']))
    v_x = exact_v_end(plan)
    a = (v_in - v_x) / T
    up = dict(family='shaped', v_xi=str(plan['v_xi']), T_xi=str(plan['T_xi']),
              pieces=[['0', str(-a)]])
    info = dict(
        rate=str(a), rate_float=float(a), v_in=str(v_in), T_xi=str(T),
        v_r_T_xi_design=str(v_x), v_r_T_xi_design_float=float(v_x),
        v_r_T_xi_unshaped=str(exact_v_end(up)),
        equal_handoff_speed=bool(exact_v_end(up) == v_x),
        distance_to_T_xi_design=float(exact_distance(plan)),
        distance_to_T_xi_unshaped=float(exact_distance(up)),
        design_first_rate=str(plan['pieces'][0][1]),
        design_last_rate=str(plan['pieces'][-1][1]),
        design_n_pieces=len(plan['pieces']))
    return up, info


def context(opts):
    """Design record, configurations and simulator inputs of the design and
    of the unshaped comparator."""
    D = json.load(open(opts['design']))
    plan = D['selected']['plan']
    cfg = fb_cfg(opts, plan)
    B = sbh.setup(cfg, **setup_params(opts))
    up, uinfo = unshaped_plan(plan)
    cfg_u = fb_cfg(opts, up)
    Bu = sbh.setup(cfg_u, **setup_params(opts))
    return dict(D=D, plan=plan, cfg=cfg, B=B, uplan=up, uinfo=uinfo,
                cfg_u=cfg_u, Bu=Bu)


def mpc_inputs(B):
    """B with the takeover positions of sim_bench_hr substituted into the
    sim_bench inputs (sim_bench.setup places the units at the planned
    positions of the shaped plan, i.e. at the parking gaps)."""
    Bs = dict(B['sb'], s0=np.array(B['s0'], float).copy())
    return dict(B, sb=Bs)


def mpc_port_note(B):
    n = B['n']
    ell = B['ell']
    sa = np.asarray(B['sb']['s0'], float)
    su = np.asarray(B['s0'], float)
    return dict(
        substituted='s0 of the sim_bench inputs := s0 of sim_bench_hr.setup '
                    '(takeover positions); every other input unchanged',
        initial_clearances_sim_bench_setup=[float(sa[i - 1] - sa[i] - ell)
                                            for i in range(1, n)],
        initial_clearances_used=[float(su[i - 1] - su[i] - ell)
                                 for i in range(1, n)],
        head_position_equal=bool(sa[0] == su[0]),
        reason='certify_plan.shaped_plan keeps the parking gaps in the '
               'planned positions (s0 = takeover_positions(d_s, d_s[1:])); '
               'sim_bench.setup uses the planned positions as the initial '
               'state; sim_bench_hr.setup (sim_hr.hr_setup) places the units '
               'at the clearances c0 with the head at its slot')


def header(part, X, opts):
    D = X['D']
    B = X['B']
    Bu = X['Bu']
    files = dict(
        sim_bench_hr=os.path.join(CODE, 'src', 'simulation',
                                  'sim_bench_hr.py'),
        run_fb_studies=os.path.abspath(__file__),
        sim_hr=os.path.join(CODE, 'src', 'simulation', 'sim_hr.py'),
        sim_bench=os.path.join(CODE, 'src', 'simulation', 'sim_bench.py'),
        sim_cont=os.path.join(CODE, 'src', 'simulation', 'sim_cont.py'),
        certify_plan=os.path.join(CODE, 'src', 'certification',
                                  'certify_plan.py'),
        certify_fb=os.path.join(CODE, 'src', 'certification',
                                'certify_fb.py'),
        certify_profile=os.path.join(CODE, 'src', 'certification',
                                     'certify_profile.py'),
        run_bench_studies=os.path.join(CODE, 'run_bench_studies.py'),
        design=opts['design'], caseD_design=cf.DESIGN_D,
        mpc_selection=MPC_SEL)
    sel = D['selected']
    return dict(
        part=part, evidence_class=EVIDENCE[part],
        simulator='src/simulation/sim_bench_hr.py',
        driver='run_fb_studies.py',
        command='python -B run_fb_studies.py ' + ' '.join(sys.argv[1:]),
        options={k: (rel(v) if k == 'design' else v)
                 for k, v in opts.items()},
        design=dict(
            file=rel(opts['design']), created=D.get('created'),
            selected_lam=sel.get('lam'), T_xi=sel.get('T_xi'),
            a_r0=sel.get('a_r0'), a_cr=sel.get('a_cr'),
            v_hand=sel.get('v_hand'),
            n_pieces=len(sel['plan']['pieces']),
            certificate_20ms=dict(
                admitted=sel.get('certificate_20ms', {}).get('admitted'),
                failed=sel.get('certificate_20ms', {}).get('failed')),
            cfg=X['cfg'], cfg_sha256=cfg_hash(X['cfg']),
            params=dict(B['params']), cfg_used=B['cfg'],
            cfg_used_sha256=cfg_hash(B['cfg']),
            reference='head: shaped reference a_r of design_fb.json on '
                      '[0, T_xi), -a_b until v_r = 0; followers: constants '
                      'only, desired clearance d_s,i from takeover '
                      '(sim_bench_hr on a shaped setup)'),
        unshaped=dict(plan=X['uplan'], info=X['uinfo'],
                      cfg_used_sha256=cfg_hash(Bu['cfg']),
                      markers=Bu['markers'], s_ref_stop=Bu['s_ref_stop'],
                      tau_r=Bu['tau_r'], v_c1=Bu['v_c1']),
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
            markers=B['markers'], no_schedules=B['no_sched'],
            desired_clearance=B['DPRE'][1:],
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
    obj['design']['record_unchanged_at_end'] = bool(
        sha256_file(opts['design']) == obj['code_sha256']['design'])
    obj['runtime_s'] = time.time() - t0
    log('  design record unchanged at the end: %s; part run time %.1f s'
        % (obj['design']['record_unchanged_at_end'], obj['runtime_s']))
    dump(part, obj, log)
    log.close()


def load_part(part, B):
    """A part file written by this driver for the same design and parameters
    (retune.json is read by loss and plant; with a tag the tagged file)."""
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


def pos_time(R, latch):
    """Time with an applied command above zero per unit, before the latch,
    from a record at every step (ideal plant); None without a record."""
    rc = R.get('rec')
    if rc is None or len(rc['t']) < 2:
        return None
    u = np.asarray(rc['u'])
    md = np.asarray(rc['mode'])
    rdt = float(rc['t'][1] - rc['t'][0])
    return [float(np.sum((md[i] != latch) & (u[i] > 0.0)) * rdt)
            for i in range(u.shape[0])]


def row_of(name, R, pos=None, **extra):
    """Summary row of one run (base, retune)."""
    done = bool(R['all_stopped'] and R['contact_time'] is None)
    nn = R.get('t_nonneg')
    tb = R['t_barrier']
    e_h = R.get('e_hnd')
    x_h = R.get('eps_hnd')
    up = R.get('u_max_pre_unit')
    r = dict(
        name=name, variant=R['variant'], plant=R['plant'],
        apply_filter=R['apply_filter'], lam=R.get('lam'),
        h_w=R.get('h_w'), gains=R.get('gains'), v0=R.get('v0'),
        all_stopped=R['all_stopped'], contact=R['contact_time'],
        contact_pair=R['contact_pair'],
        disp=R['disp'] if done else None, T_f=R['T_f'],
        marker=R['marker_abs_max'] if done else None,
        gap=R['gap_abs_max'] if done else None,
        marker_signed=R['marker_signed'], gap_signed=R['gap_signed'],
        state_at_end=dict(marker=R['marker_abs_max'], gap=R['gap_abs_max'],
                          t=R['t_end_sim']),
        gmin=R['gmin_min'], gmin_pair=R['gmin_pair'], gmin_pairs=R['gmin'],
        hmin=R['hmin_min'],
        u_abs_max=R['u_abs_max'], u_min=R['u_min'], u_max=R['u_max'],
        u_max_before_T_xi=max(x for x in up if x is not None)
        if up else None,
        atU=max(R['t_atU']), atU_units=R['t_atU'],
        nonneg=max(nn) if nn is not None else None, nonneg_units=nn,
        pos=max(pos) if pos is not None else None, pos_units=pos,
        nom_pos=max(R['t_nom_pos']) if 't_nom_pos' in R else None,
        nom_above_U=max(R['t_nom_above_U']), clipU=max(R['t_clipU']),
        barrier=max(tb),
        barrier_unit=(int(np.argmax(tb)) + 1) if max(tb) > 0 else None,
        barrier_raw=max(R['t_barrier_raw']),
        fallback=float(sum(R['t_infeas'])),
        tset=R.get('tset'),
        in_box_at_Txi=R.get('in_box_at_Txi'),
        e_hnd_abs_max=max(abs(x) for x in e_h) if e_h and None not in e_h
        else None,
        eps_hnd_abs_max=max(abs(x) for x in x_h) if x_h and None not in x_h
        else None,
        rule_units=R['rule_units'], stop_order=R['stop_order'],
        buffer_lost=R['buffer_lost'],
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], met=R['pass_all'])
    r.update(extra)
    return r


def row_line(r):
    return ('  %-31s disp=%s T_f=%s marker=%s gap=%s gmin=%.3f max|u|=%s '
            'u_max=%s atU=%.2f pos=%s barrier=%.2f fallback=%.2f inbox=%s '
            'contact=%s met=%s'
            % (r['name'], fmt(r['disp']), fmt(r['T_f'], '%.3f'),
               fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'), r['gmin'],
               fmt(r['u_abs_max'], '%.3f'), fmt(r['u_max'], '%.4f'),
               r['atU'], fmt(r['pos'], '%.2f', 'n.a.'), r['barrier'],
               r['fallback'], r['in_box_at_Txi'],
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
        u_min=R['u_min'], u_max=R['u_max'],
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
        u_max_max=max(r['u_max'] for r in recs),
        config=cfg, seeds=recs)


def plant_line(r):
    return ('  %-44s joint %2d/%d disp med/max %s/%s marker %s gap %s gmin '
            '%.2f atU med/max %.1f/%.1f s barrier %d contacts %d stopped %d '
            '(pass d/m/g %d/%d/%d)'
            % (r['name'], r['joint_pass'], r['n'],
               fmt(r['disp_med'], '%.3f'), fmt(r['disp_max'], '%.3f'),
               fmt(r['marker_max'], '%.2f'), fmt(r['gap_max'], '%.2f'),
               r['gmin_min'], r['atU_med'], r['atU_max'], r['barrier_runs'],
               r['contacts'], r['stopped'], r['pass_disp'],
               r['pass_marker'], r['pass_gap']))


# ================================================== certificate summaries
def cert_summary(ev, runtime_s=None):
    """Admission, failed conditions (with their rows) and the main bounds of
    one evaluation of certify_fb.evaluate."""
    conds = ev['conds']
    bd = ev.get('bounds') or {}
    failed = {k: {kk: conds[k].get(kk) for kk in
                  ('slack', 'value', 'threshold', 'sense', 'unit', 'at',
                   'per_unit_max', 'per_unit_min', 'per_pair', 'per_unit')
                  if kk in conds[k]} for k in ev['failed']}
    return dict(
        admitted=bool(ev['admitted']), failed=list(ev['failed']),
        failed_detail=failed,
        slacks={k: c.get('slack') for k, c in conds.items()},
        NONPOS=dict(ok=conds['NONPOS']['ok'],
                    slack=conds['NONPOS'].get('slack'),
                    before_T_xi=conds['NONPOS'].get('before_T_xi'),
                    before_T_xi_at=conds['NONPOS'].get('before_T_xi_at')),
        bounds=dict((k, bd.get(k)) for k in (
            'dispersion', 'T_f_bar', 'marker_max', 'gap_max',
            'entry_deadline', 'clearance_floor_min', 'velocity_floor',
            'u_max_before_T_xi', 'u_min_before_T_xi')),
        T_xi=ev.get('T_xi'), lam=ev.get('lam'), dbar=ev.get('dbar'),
        runtime_s=runtime_s)


def cert_line(tag, c):
    fd = c['failed_detail']
    det = '; '.join('%s slack %s at %s' % (k, fmt(v.get('slack'), '%.4g'),
                                            v.get('at')) for k, v in fd.items())
    return ('  certificate %-44s admitted %s, failed %s %s (C7a slack %s, '
            'NONPOS slack %s)'
            % (tag, c['admitted'], c['failed'], ('[' + det + ']') if det
               else '', fmt(c['slacks'].get('C7a'), '%.4g'),
               fmt(c['NONPOS']['slack'], '%.4g')))


# ============================================================ worker jobs
_WB = {}


def _setups_for(design, params):
    """(B, Bu) of a worker process, rebuilt from the design file and the
    parameters (cached per process)."""
    h = json.dumps([design, params], sort_keys=True)
    if h not in _WB:
        opts = dict(DEFAULT_OPTS, design=design, **params)
        X = context(opts)
        _WB[h] = X
    return _WB[h]


def _job(job):
    """One job of a worker process: ('mpc', key, design, params, kw) runs
    the MPC with the substituted takeover positions; ('cert', key, design,
    params, which, extra) evaluates the certificate of the design ('design')
    or of the unshaped reference ('unshaped')."""
    kind = job[0]
    if kind == 'mpc':
        _, key, design, params, kw = job
        X = _setups_for(design, params)
        Bm = mpc_inputs(X['B'])
        R = sbh.run(Bm, 'mpc', **kw)
        pos = pos_time(R, sb.LATCH) if R.get('rec') is not None else None
        out = slim(R)
        out['t_pos'] = pos
        return key, out
    _, key, design, params, which, extra = job
    X = _setups_for(design, params)
    plan = X['plan'] if which == 'design' else X['uplan']
    opts = dict(DEFAULT_OPTS, design=design, **params)
    ta = time.time()
    ev = cf.evaluate(fb_cfg(opts, plan, **extra))
    return key, cert_summary(ev, time.time() - ta)


def pool_map(jobs, workers, log, label):
    res = {}
    t0 = time.time()
    if workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            k, R = _job(job)
            res[k] = R
        log('    %s: %d jobs in %.0f s' % (label, len(jobs), time.time() - t0))
        return res
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
        for k, R in ex.map(_job, jobs):
            res[k] = R
    log('    %s: %d jobs on %d workers in %.0f s'
        % (label, len(jobs), min(workers, len(jobs)), time.time() - t0))
    return res


def worker_params(opts):
    return dict(cert_params(opts))


def mpc_kw(sel):
    return dict(W=dict(sel['W']), mpc_planner=dict(sel['planner']),
                mpc_stop='detector')


def v0_strings(v0):
    """Decimal strings of takeover speeds for the certificate (exact)."""
    return [repr(round(float(x), 9)) for x in v0]


# ================================================================== check
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


def _observed(B, R):
    """Observations of a run with a record at every step (ideal plant):
    entry instants into the handoff box before T_xi, smallest speed before
    T_xi, commands after T_xi before and after the receipt of the braking
    onset (receipt of unit i at T_xi + (i-1) x transport with the constant
    transport, checked against the command drop)."""
    rc = R['rec']
    t = np.asarray(rc['t'])
    s = np.asarray(rc['s'])
    v = np.asarray(rc['v'])
    u = np.asarray(rc['u'])
    md = np.asarray(rc['mode'])
    n = B['n']
    dt = B['dt']
    k_xi = int(B['k_xi'])
    pre = np.arange(len(t)) < k_xi
    assert abs(t[k_xi] - B['T_xi']) < 1e-9
    s0r, vr, _ = B['S'].ref(t[pre])
    ent = []
    last_bad = []
    for i in range(n):
        if i == 0:
            e = s0r - s[0, pre] - B['ell'] - B['d_s'][0]
            x = vr - v[0, pre]
        else:
            e = s[i - 1, pre] - s[i, pre] - B['ell'] - B['d_s'][i]
            x = v[i - 1, pre] - v[i, pre]
        bad = (np.abs(e) > B['eps_e']) | (np.abs(x) > B['eps_v'])
        jb = np.nonzero(bad)[0]
        if jb.size == 0:
            ent.append(float(t[0]))
            last_bad.append(None)
        elif jb[-1] + 1 < k_xi:
            ent.append(float(t[jb[-1] + 1]))
            last_bad.append(float(t[jb[-1]]))
        else:
            ent.append(None)
            last_bad.append(float(t[jb[-1]]))
    vmin_pre = float(np.min(v[:, pre]))
    vmin_unit = int(np.argmin(np.min(v[:, pre], axis=1))) + 1
    after = []
    for i in range(n):
        live = md[i] != sbh.LATCH
        k_rx = k_xi + i * int(B['Ntr'])
        idx = np.arange(len(t))
        pre_w = (idx >= k_xi) & (idx < k_rx) & live
        post_w = (idx >= k_rx) & live
        drop = np.nonzero((idx >= k_xi) & live & (u[i] < -0.5 * B['a_b']))[0]
        after.append(dict(
            unit=i + 1, receipt_instant=float(t[k_rx]),
            first_command_below_half_a_b=float(t[drop[0]]) if drop.size
            else None,
            receipt_consistent=bool(drop.size and drop[0] == k_rx),
            pre_min=float(np.min(u[i, pre_w])) if pre_w.any() else None,
            pre_max=float(np.max(u[i, pre_w])) if pre_w.any() else None,
            post_min=float(np.min(u[i, post_w])) if post_w.any() else None,
            post_max=float(np.max(u[i, post_w])) if post_w.any() else None,
            n_pre=int(pre_w.sum()), n_post=int(post_w.sum())))
    return dict(entry=ent, last_outside_box=last_bad, vmin_pre=vmin_pre,
                vmin_unit=vmin_unit, after=after, record_dt=dt)


def _certified(X, R, log):
    """(T5) observations of 'proposed' against the bounds of the certificate
    of the same configuration (certify_fb.evaluate, read-only import)."""
    B = X['B']
    cfg = X['cfg']
    ta = time.time()
    ev = cf.evaluate(cfg)
    te = time.time() - ta
    bd = ev['bounds']
    cs = ev['consts']
    log('  T5 certificate (lam %.4g, dbar %.4g, U %.4g): admitted %s, failed '
        '%s (%.1f s)' % (cfg['lam'], cfg['dbar'], cfg['U'], ev['admitted'],
                         ev['failed'], te))
    ob = _observed(B, R)
    rows = []

    def chk(name, obs, bound, sense='<=', group=''):
        ok = bool(obs is not None and bound is not None
                  and (obs <= bound if sense == '<=' else obs >= bound))
        rows.append(dict(observation=name, group=group, observed=obs,
                         bound=bound, sense=sense, inside=ok,
                         margin=None if (obs is None or bound is None)
                         else (bound - obs if sense == '<=' else obs - bound)))
        log('  T5 %-50s observed %-13s %s %-13s inside %s'
            % (name, fmt(obs, '%.6g'), sense, fmt(bound, '%.6g'), ok))
    n = B['n']
    chk('dispersion [s]', R['disp'], bd['dispersion'], group='terminal')
    chk('completion T_f [s]', R['T_f'], bd['T_f_bar'], group='terminal')
    for i in range(n):
        chk('|marker error| unit %d [m]' % (i + 1),
            abs(R['marker_signed'][i]), bd['marker'][i], group='terminal')
    for i in range(n - 1):
        chk('|terminal gap error| pair %d [m]' % (i + 2),
            abs(R['gap_signed'][i]), bd['gap'][i], group='terminal')
        chk('smallest clearance margin pair %d [m]' % (i + 2),
            R['gmin'][i], bd['clearance_floor'][i], '>=', group='clearance')
    for i in range(n):
        chk('activation of unit %d [s]' % (i + 1), R['tact'][i],
            bd['activation_deadline'][i], group='acquisition')
        chk('S2 switch of unit %d [s]' % (i + 1), R['tset'][i],
            bd['switch_deadline'][i], group='acquisition')
        chk('largest command before T_xi, unit %d' % (i + 1),
            R['u_max_pre_unit'][i], bd['u_max_before_T_xi'][i],
            group='authority before T_xi')
        chk('smallest command before T_xi, unit %d' % (i + 1),
            R['u_min_pre_unit'][i], bd['u_min_before_T_xi'][i], '>=',
            group='authority before T_xi')
    for i in range(n):
        chk('|e_i(T_xi)| unit %d [m] (handoff box)' % (i + 1),
            abs(R['e_hnd'][i]), float(B['eps_e']), group='handoff')
        chk('|eps_i(T_xi)| unit %d [m/s] (handoff box)' % (i + 1),
            abs(R['eps_hnd'][i]), float(B['eps_v']), group='handoff')
        chk('entry into the handoff box, unit %d [s]' % (i + 1),
            ob['entry'][i], cs['Tent_abs'][i], group='handoff')
    chk('smallest speed before T_xi (unit %d) [m/s]' % ob['vmin_unit'],
        ob['vmin_pre'], bd['velocity_floor'], '>=', group='velocity')
    for a in ob['after']:
        i = a['unit'] - 1
        if i >= 1:
            chk('largest command pre-receipt, unit %d' % (i + 1), a['pre_max'],
                bd['u_max_pre_receipt'][i], group='after T_xi')
            chk('smallest command pre-receipt, unit %d' % (i + 1),
                a['pre_min'], bd['u_min_pre_receipt'][i], '>=',
                group='after T_xi')
        lo, hi = bd['u_band_post_receipt'][i]
        chk('largest command post-receipt, unit %d' % (i + 1), a['post_max'],
            hi, group='after T_xi')
        chk('smallest command post-receipt, unit %d' % (i + 1),
            a['post_min'], lo, '>=', group='after T_xi')
    return dict(checked=True, evaluator='src/certification/certify_fb.py',
                config={k: cfg[k] for k in (
                    'lam', 'vhnd', 'lags', 'U', 'Uminus', 'dbar', 'eps_e',
                    'eps_v', 'cell_dt', 'one_sided', 'switch_lower')},
                admitted=ev['admitted'], failed=ev['failed'],
                certificate=cert_summary(ev, te),
                bounds=bd, Tent_abs=cs['Tent_abs'], tlow_sw=cs['tlow_sw'],
                evaluate_runtime_s=te, rows=rows,
                n_inside=sum(1 for r in rows if r['inside']),
                n_rows=len(rows),
                all_inside=bool(all(r['inside'] for r in rows)),
                smallest_margin_per_group={
                    g: min(r['margin'] for r in rows
                           if r['group'] == g and r['margin'] is not None)
                    for g in sorted(set(r['group'] for r in rows))},
                observed=ob,
                handoff_errors_in_box_at_T_xi=R['in_box_at_Txi'],
                e_hnd=R['e_hnd'], eps_hnd=R['eps_hnd'],
                notes=['closure-offset rows of run_hr_studies.py are dropped: '
                       'no FB law uses the closure flag (the simulator still '
                       'relays it)',
                       'entry instant: first record instant from which both '
                       'handoff errors stay in the box (|e| <= eps_e, |eps| '
                       '<= eps_v) up to T_xi, record at every 1 ms step; '
                       'bound: the entry instant Tent_abs of the certificate',
                       'receipt of the braking onset by unit i: T_xi + (i-1) '
                       'x transport (constant transport); receipt_consistent '
                       'states that the first command below -a_b/2 after '
                       'T_xi occurs there'])


def _identical(Ra, Rb, skip):
    diffs = []
    nf = 0
    for key in sorted(set(Ra) | set(Rb)):
        if key in skip or key == 'rec':
            continue
        nf += 1
        if json.dumps(_py(Ra.get(key)), sort_keys=True) != \
                json.dumps(_py(Rb.get(key)), sort_keys=True):
            diffs.append(key)
    return nf, diffs


def part_check(opts):
    log = Log('check')
    t0 = time.time()
    log('=== part check ===')
    X = context(opts)
    B = X['B']
    out = header('check', X, opts)
    sbh.run(B, 'proposed')                          # compile / load cache
    v0h = sbh.doubled_mismatch(B)
    # ---------------------------------------------------------------- T3
    t3 = dict(cases={}, skipped_fields=['variant', 'apply_filter',
                                        'kernel_law', 'schedule_mode',
                                        'runtime_s'])
    skip = tuple(t3['skipped_fields'])
    for tag, v0 in (('benchmark takeover', None), ('doubled increments', v0h)):
        Rp = sbh.run(B, 'proposed', v0=v0)
        c = dict()
        for var in ('nofilter', 'feedback', 'feedback_step'):
            Rv = sbh.run(B, var, v0=v0)
            nf, diffs = _identical(Rp, Rv, skip)
            c[var] = dict(fields_compared=nf, fields_different=diffs,
                          identical=not diffs)
            log('  T3 %-13s vs proposed (%s): %d fields, %d different %s'
                % (var, tag, nf, len(diffs), diffs))
        c['barrier_time'] = max(Rp['t_barrier'])
        c['fallback_time'] = sum(Rp['t_infeas'])
        c['smallest_inactivity_margin'] = min(x for x in Rp['mmin']
                                              if x is not None)
        log('  T3 proposed (%s): barrier %.3f s, fallback %.3f s, smallest '
            'inactivity margin u_cbf - u_nom %.4f m/s^2'
            % (tag, c['barrier_time'], c['fallback_time'],
               c['smallest_inactivity_margin']))
        t3['cases'][tag] = c
    t3['pass'] = bool(all(t3['cases'][tg][v]['identical']
                          for tg in t3['cases']
                          for v in ('nofilter', 'feedback', 'feedback_step')))
    # ---------------------------------------------------------------- T4
    Bm = mpc_inputs(B)
    Bs = Bm['sb']
    bounds = ([float(-x) for x in B['Uminus']], float(B['U']))
    t4 = dict(reference='sim_bench.run(variant, bounds=([-U_i^-], U)) with '
                        'the same gain on sim_bench.setup of the same cfg '
                        'with the takeover positions of sim_bench_hr '
                        'substituted (as for the MPC)',
              substitution=mpc_port_note(B),
              tolerance=TOL_PORT, hifi=[], ideal=[], loss=[])
    ok4 = True
    for var, kw in (('feedback', {}), ('cacc', dict(h_w=0.5))):
        for ovn, ov in (('default', {}), ('compensated', dict(COMP))):
            for sd in (1, 2, 3):
                A = sbh.run(B, var, plant='hifi', hifi=ov, seed=sd,
                            hold_brake=True, **kw)
                C = sb.run(Bs, var, plant='hifi', hifi=ov, seed=sd,
                           bounds=bounds, hold_brake=True, **kw)
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
            last = t4['hifi'][-3:]
            log('  T4 perturbed plant %-8s %-11s seeds 1-3: max stop %s s, '
                'marker %s m, gap %s m -> %s'
                % (var, ovn,
                   fmt(max(_vals([r['max_abs_difference']['stop_instants']
                                  for r in last])), '%.1e'),
                   fmt(max(_vals([r['max_abs_difference']['marker_errors']
                                  for r in last])), '%.1e'),
                   fmt(max(_vals([r['max_abs_difference']['gap_errors']
                                  for r in last])), '%.1e'),
                   all(r['passed'] for r in last)))
        A = sbh.run(B, var, record=True, **kw)
        C = sb.run(Bs, var, bounds=bounds, record=True, **kw)
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
        log('  T4 ideal %-8s before the first RULE/stop (%.3f s): |ds| %s m '
            '|dv| %s m/s -> %s; terminal (RULE convention differs): stop %s '
            's, marker %s m'
            % (var, tf, fmt(ds, '%.1e'), fmt(dv, '%.1e'), ok,
               fmt(t4['ideal'][-1]['terminal']['stop_instants'], '%.1e'),
               fmt(t4['ideal'][-1]['terminal']['marker_errors'], '%.1e')))
        for tl in T_LOSS:
            for em in (None, EM_UNIT):
                for wd in T_WD_PORT:
                    A = sbh.run(B, var, scenario='loss', t_loss=tl,
                                emergency_unit=em, T_wd=wd, record=True, **kw)
                    C = sb.run(Bs, var, scenario='loss', t_loss=tl,
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
        log('  T4 loss %-8s %d runs: max |ds| %s m |dv| %s m/s before the '
            'first RULE/stop/contact; contact outcome equal in %d; max '
            '|contact instant diff| %.1e s -> %s'
            % (var, len(rl), fmt(max(_vals([r['max_abs_ds'] for r in rl])),
                                 '%.1e'),
               fmt(max(_vals([r['max_abs_dv'] for r in rl])), '%.1e'),
               sum(1 for r in rl if r['same_contact_outcome']),
               max([abs(r['contact'][0] - r['contact'][1]) for r in rl
                    if r['contact'][0] is not None
                    and r['contact'][1] is not None] or [0.0]),
               all(r['passed'] for r in rl)))
    t4['pass'] = bool(ok4)
    t4['note'] = ('after the first RULE entry sim_bench and sim_bench_hr '
                  'differ by convention: sim_bench enters RULE at the next '
                  'grid instant after the zero-speed instant of the '
                  'predecessor, sim_bench_hr at that instant with an '
                  'immediate command update; the perturbed plant detects '
                  'standstill at control instants in both')
    # ---------------------------------------------------------------- T5
    Rp = sbh.run(B, 'proposed', record=True, rec_dt=B['dt'])
    t5 = _certified(X, Rp, log)
    t5['run'] = slim(Rp)
    out.update(T3=t3, T4=t4, T5=t5,
               all_pass=bool(t3['pass'] and t4['pass']))
    log('  check: T3 %s, T4 %s; T5 %d of %d inside (certificate admitted %s)'
        % (t3['pass'], t4['pass'], t5['n_inside'], t5['n_rows'],
           t5['admitted']))
    finish('check', out, opts, log, t0)
    return out['all_pass']


# =================================================================== base
def base_specs(sel):
    """(name, setup, variant, kwargs): setup 'B' design reference, 'Bu'
    unshaped reference, 'Bm' design reference with the substituted takeover
    positions in the sim_bench inputs (MPC)."""
    return [
        ('proposed', 'B', 'proposed', {}),
        ('proposed, filter off', 'B', 'nofilter', {}),
        ('without feedback', 'B', 'nofeedback', {}),
        ('unshaped reference', 'Bu', 'proposed', {}),
        ('cacc h_w=0.2', 'B', 'cacc', dict(h_w=0.2)),
        ('cacc h_w=0.5', 'B', 'cacc', dict(h_w=0.5)),
        ('cacc h_w=1.0', 'B', 'cacc', dict(h_w=1.0)),
        ('pid hand-tuned', 'B', 'pid', dict(gains=dict(sb.K_PID))),
        ('mpc', 'Bm', 'mpc', mpc_kw(sel)),
        ('without feedback, filter off', 'B', 'nofeedback',
         dict(apply_filter=False)),
        ('unshaped reference, filter off', 'Bu', 'nofilter', {}),
        ('cacc h_w=0.5, filter off', 'B', 'cacc',
         dict(h_w=0.5, apply_filter=False)),
        ('pid hand-tuned, filter off', 'B', 'pid',
         dict(gains=dict(sb.K_PID), apply_filter=False)),
    ]


def previous_round_rows():
    """'proposed' rows of the stored-schedule design of the previous round
    (data/new_d19/headref/studies/base.json, read-only)."""
    J = json.load(open(HR_BASE))
    c = J['constants']
    rows = {}
    for blk in ('benchmark', 'doubled'):
        rows[blk] = [dict(r) for r in J[blk]['rows'] if r['name'] == 'proposed']
    return dict(
        label='stored-schedule design of the previous round (followers store '
              'gap-closing schedules started by a relayed closure flag): '
              'outside the information model of the FB design; reference '
              'only; numbers copied from its record, not rerun',
        file=rel(HR_BASE), sha256=sha256_file(HR_BASE),
        runtime_date=J.get('runtime_date'), driver=J.get('driver'),
        driver_sha256=J['code_sha256'].get('run_hr_studies'),
        design=dict(file=J['design']['file'], lam=c['lam'], T_xi=c['T_xi'],
                    t_0=c['t_0'], d_s=c['d_s'], c0=c['c0'],
                    params=J['design']['params']),
        not_comparable='other design: gain 0.15, T_xi 86.142 s, parking gaps '
                       '5 m (head slot 6 m); T_f and clearance margins are '
                       'not comparable with the FB rows',
        rows=rows)


def part_base(opts):
    log = Log('base')
    t0 = time.time()
    log('=== part base ===')
    X = context(opts)
    B = X['B']
    Bu = X['Bu']
    out = header('base', X, opts)
    sel = sbh.mpc_selection()
    out['mpc_selection'] = sel
    out['mpc_port'] = mpc_port_note(B)
    log('  MPC weights %s planner %s from %s (tuned with the input set %s; '
        'not retuned); takeover positions substituted: initial clearances '
        'of sim_bench.setup %s -> %s'
        % (sel['W'], sel['planner'], sel['source'],
           sel['input_set_of_tuning'],
           out['mpc_port']['initial_clearances_sim_bench_setup'],
           out['mpc_port']['initial_clearances_used']))
    log('  unshaped reference: rate %s = %.9f m/s^2, v_r(T_xi) %s (design %s,'
        ' equal %s); planned distance to T_xi %.3f m (design %.3f m); '
        'reference stop %.3f m (design %.3f m)'
        % (X['uinfo']['rate'], X['uinfo']['rate_float'],
           X['uinfo']['v_r_T_xi_unshaped'], X['uinfo']['v_r_T_xi_design'],
           X['uinfo']['equal_handoff_speed'],
           X['uinfo']['distance_to_T_xi_unshaped'],
           X['uinfo']['distance_to_T_xi_design'], Bu['s_ref_stop'],
           B['s_ref_stop']))
    v0h = sbh.doubled_mismatch(B)
    setups = dict(B=B, Bu=Bu)
    params = worker_params(opts)
    # ---- MPC runs (sim_bench) in worker processes
    jobs = []
    for tag, v0 in (('benchmark', None), ('doubled', v0h)):
        for name, key, var, kw in base_specs(sel):
            if var == 'mpc':
                jobs.append(('mpc', (tag, name), opts['design'], params,
                             dict(kw, v0=v0, record=True)))
    mres = pool_map(jobs, int(opts['workers']), log, 'mpc')
    blocks = {}
    for tag, v0 in (('benchmark', None), ('doubled', v0h)):
        rows = []
        runs = {}
        log('  -- takeover: %s %s' % (tag, list(B['v0']) if v0 is None
                                      else v0))
        for name, key, var, kw in base_specs(sel):
            if var == 'mpc':
                R = mres[(tag, name)]
                pos = R.get('t_pos')
            else:
                R = sbh.run(setups[key], var, v0=v0, record=True,
                            rec_dt=B['dt'], **kw)
                pos = pos_time(R, sbh.LATCH)
            r = row_of(name, R, pos=pos, runtime_s=R['runtime_s'],
                       reference='unshaped' if key == 'Bu' else 'design')
            rows.append(r)
            runs[name] = slim(R)
            log(row_line(r))
        blocks[tag] = dict(v0=list(B['v0']) if v0 is None else v0,
                           rows=rows, runs=runs)
    out['benchmark'] = blocks['benchmark']
    out['doubled'] = blocks['doubled']
    # ---- certificates (informative)
    lam0 = float(opts['lam'])
    cj = [('cert', ('design', 'benchmark', lam0), opts['design'], params,
           'design', {}),
          ('cert', ('design', 'doubled', lam0), opts['design'], params,
           'design', dict(v0=v0_strings(v0h))),
          ('cert', ('unshaped', 'benchmark', lam0), opts['design'], params,
           'unshaped', {}),
          ('cert', ('unshaped', 'doubled', lam0), opts['design'], params,
           'unshaped', dict(v0=v0_strings(v0h)))]
    grid = sorted(set(FB_LAMBDAS) | {lam0})
    for g in grid:
        if g == lam0:
            continue
        cj.append(('cert', ('unshaped', 'benchmark', g), opts['design'],
                   dict(params, lam=float(g)), 'unshaped', {}))
    cres = pool_map(cj, int(opts['workers']), log, 'certificates')
    certs = dict(
        design_benchmark=cres[('design', 'benchmark', lam0)],
        design_reference_doubled_takeover=cres[('design', 'doubled', lam0)],
        unshaped_benchmark=cres[('unshaped', 'benchmark', lam0)],
        unshaped_doubled_takeover=cres[('unshaped', 'doubled', lam0)],
        unshaped_gain_scan=[dict(cres[('unshaped', 'benchmark', g)],
                                 lam_requested=g) for g in grid],
        doubled_takeover_v0=v0_strings(v0h),
        note='float evaluator certify_fb.evaluate at the design age bound; '
             'informative (the dispatcher of the FB design re-runs the '
             'design rule for another takeover state; here the reference is '
             'kept fixed)')
    for k in ('design_benchmark', 'design_reference_doubled_takeover',
              'unshaped_benchmark', 'unshaped_doubled_takeover'):
        log(cert_line(k, certs[k]))
    for c in certs['unshaped_gain_scan']:
        log(cert_line('unshaped, benchmark, lam=%.3f' % c['lam'], c))
    out['certificates'] = certs
    out['previous_round'] = previous_round_rows()
    for blk, rr in out['previous_round']['rows'].items():
        for r in rr:
            log('  previous round (%s, stored schedules, reference only): '
                'disp=%s T_f=%s marker=%s gap=%s gmin=%.3f atU=%.2f '
                'barrier=%.2f met=%s'
                % (blk, fmt(r['disp']), fmt(r['T_f'], '%.3f'),
                   fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'),
                   r['gmin'], r['atU'], r['barrier'], r['met']))
    out['columns'] = dict(
        disp='stopping-time dispersion, last stop minus first stop [s] '
             '(None: not every unit stopped, or contact)',
        T_f='last stop instant [s]', marker='largest |marker error| [m] '
        '(markers: planned stop positions of the run\'s own reference)',
        gap='largest |terminal gap error| [m] (to the parking gaps)',
        gmin='smallest clearance margin g_i = c_i - s_m over the run and the '
             'pairs [m] (intersample minimum)',
        u_abs_max='largest |applied net command| before the latch [m/s^2]',
        u_max='largest applied net command [m/s^2] (0: the zero command '
              'held by follower i before its first delivery, (i-1) x 19 ms)',
        u_max_before_T_xi='largest applied command before T_xi [m/s^2]',
        atU='time with the applied command at the traction limit U, worst '
            'unit [s]',
        nonneg='time with the applied command at 0 or above, worst unit [s] '
               '(includes the startup of follower i, (i-1) x 19 ms)',
        pos='time with the applied command above 0, worst unit [s] (record '
            'at every 1 ms step, ideal plant)',
        nom_pos='time with u_nom > 0, worst unit [s]',
        nom_above_U='time with u_nom > U, worst unit [s]',
        clipU='time in which the input set clipped the filter output at U, '
              'worst unit [s]',
        barrier='time in which the filter lowered the applied command '
                '(fallback included), worst unit [s]',
        fallback='time with u_cbf < -U_i^- (fallback -U_i^-), summed over '
                 'the followers [s]',
        in_box_at_Txi='every pair error in the handoff box (eps_e, eps_v) at '
                      'T_xi',
        rule_units='units that entered RULE', stop_order='units in the '
        'order of their stops', met='every unit stopped, no contact, '
        'dispersion <= 1 s, markers <= 3 m, gaps <= 1 m')
    finish('base', out, opts, log, t0)


# ================================================================= retune
FB_CONTROLLERS = (('feedback', 'B', 'feedback', None),
                  ('unshaped reference', 'Bu', 'proposed', None),
                  ('cacc h_w=0.2', 'B', 'cacc', 0.2),
                  ('cacc h_w=0.5', 'B', 'cacc', 0.5),
                  ('cacc h_w=1.0', 'B', 'cacc', 1.0))
CMP_KEYS = ('disp', 'T_f', 'marker', 'gap', 'gmin', 'atU', 'barrier',
            'fallback', 'u_min', 'u_max', 'contact', 'met')


def part_retune(opts):
    log = Log('retune')
    t0 = time.time()
    log('=== part retune ===')
    X = context(opts)
    B = X['B']
    out = header('retune', X, opts)
    setups = dict(B=B, Bu=X['Bu'])
    v0h = sbh.doubled_mismatch(B)
    lam0 = float(B['lam'])
    ctrl = {}
    prop_info = []
    for name, key, var, hw in FB_CONTROLLERS:
        kw = {} if hw is None else dict(h_w=hw)
        cands = []
        for lam in FB_LAMBDAS:
            Bg = sbh.with_gain(setups[key], lam)
            S = sbh.run(Bg, var, v0=v0h, **kw)
            Hh = sbh.run(Bg, var, **kw)
            cands.append(dict(
                lam=lam, beta=Bg['beta'], gamma=Bg['gamma'], score=score(S),
                train=row_of(name + ' train', S), heldout=row_of(
                    name + ' held-out', Hh),
                train_run=slim(S), heldout_run=slim(Hh)))
            c = cands[-1]
            log('  %-20s lam=%.2f score %-26s | train disp=%s marker=%.3f '
                'gap=%.3f contact=%s | held-out disp=%s marker=%.3f gap=%.3f '
                'gmin=%.3f atU=%.2f u_max=%.4f barrier=%.2f contact=%s met=%s'
                % (name, lam, c['score'], fmt(S['disp']),
                   S['marker_abs_max'], S['gap_abs_max'],
                   fmt(S['contact_time'], '%.3f', '-'), fmt(Hh['disp']),
                   Hh['marker_abs_max'], Hh['gap_abs_max'], Hh['gmin_min'],
                   max(Hh['t_atU']), Hh['u_max'], max(Hh['t_barrier']),
                   fmt(Hh['contact_time'], '%.3f', '-'), Hh['pass_all']))
            if var == 'feedback':
                # the proposed law over the same grid (information): run for
                # run identical to 'feedback' on the shaped setup
                Sp = sbh.run(Bg, 'proposed', v0=v0h)
                Hp = sbh.run(Bg, 'proposed')
                skip = ('variant', 'apply_filter', 'kernel_law',
                        'schedule_mode', 'runtime_s')
                n1, d1 = _identical(S, Sp, skip)
                n2, d2 = _identical(Hh, Hp, skip)
                prop_info.append(dict(lam=lam, identical_train=not d1,
                                      identical_heldout=not d2,
                                      fields_different=sorted(set(d1 + d2))))
        # design gain (not on the grid): information only
        Bd = sbh.with_gain(setups[key], lam0)
        Sd = sbh.run(Bd, var, v0=v0h, **kw)
        Hd = sbh.run(Bd, var, **kw)
        best = min(cands, key=lambda c: tuple(c['score']))
        ctrl[name] = dict(
            variant=var, reference='unshaped' if key == 'Bu' else 'design',
            h_w=hw, candidates=cands, best_lam=best['lam'],
            best_score=best['score'], best_train=best['train'],
            best_heldout=best['heldout'],
            design_lam=lam0, design_on_grid=lam0 in FB_LAMBDAS,
            design_score=score(Sd), design_train=row_of(name + ' train', Sd),
            design_heldout=row_of(name + ' held-out', Hd),
            lams_without_violation_train=[c['lam'] for c in cands
                                          if c['score'][0] == 0],
            lams_pass_heldout=[c['lam'] for c in cands
                               if c['heldout']['met']],
            lams_rejected=[c['lam'] for c in cands if c['score'][0] >= 1e9],
            lams_heldout_atU=[c['lam'] for c in cands
                              if c['heldout']['atU'] > 0],
            lams_heldout_barrier=[c['lam'] for c in cands
                                  if c['heldout']['barrier'] > 0])
        log('  %-20s selected lam=%.2f score %s; held-out disp=%s marker=%s '
            'gap=%s atU=%.2f barrier=%.2f met=%s; design lam=%.3f score %s '
            'held-out met %s; training without violation: %s; held-out met: '
            '%s'
            % (name, best['lam'], best['score'],
               fmt(best['heldout']['disp']),
               fmt(best['heldout']['marker'], '%.3f'),
               fmt(best['heldout']['gap'], '%.3f'), best['heldout']['atU'],
               best['heldout']['barrier'], best['heldout']['met'], lam0,
               ctrl[name]['design_score'], Hd['pass_all'],
               ctrl[name]['lams_without_violation_train'],
               ctrl[name]['lams_pass_heldout']))
    log('  proposed vs feedback over the grid: identical at %d of %d gains'
        % (sum(1 for p in prop_info if p['identical_train']
               and p['identical_heldout']), len(prop_info)))
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
            'held-out disp=%s marker=%.3f gap=%.3f gmin=%.3f atU=%.2f '
            'barrier=%.2f met=%s'
            % (K, cands[-1]['score'], fmt(S['disp']), S['marker_abs_max'],
               S['gap_abs_max'], fmt(Hh['disp']), Hh['marker_abs_max'],
               Hh['gap_abs_max'], Hh['gmin_min'], max(Hh['t_atU']),
               max(Hh['t_barrier']), Hh['pass_all']))
    best = min(cands, key=lambda c: tuple(c['score']))
    S0 = sbh.run(B, 'pid', gains=dict(sb.K_PID), v0=v0h)
    H0 = sbh.run(B, 'pid', gains=dict(sb.K_PID))
    ctrl['pid'] = dict(
        variant='pid', reference='design', grid=PID_GRID, candidates=cands,
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
        'gmin=%.3f barrier=%.2f met=%s; hand-tuned score %s held-out met=%s'
        % (best['K'], best['score'], fmt(best['heldout']['disp']),
           fmt(best['heldout']['marker'], '%.3f'),
           fmt(best['heldout']['gap'], '%.3f'), best['heldout']['gmin'],
           best['heldout']['barrier'], best['heldout']['met'],
           score(S0), H0['pass_all']))
    out['controllers'] = ctrl
    out['proposed_vs_feedback_over_grid'] = prop_info
    out['protocol'] = dict(
        source='run_bench_studies.py part fbtune (FB_LAMBDAS, score) and '
               'part base (PID_GRID); constants imported read-only; as '
               'run_hr_studies.py part retune',
        family='critically damped S2 feedback with one common gain lam for '
               'every unit, the head included: beta = lam^2, gamma = 2 lam; '
               'every other constant unchanged (alpha, detector, filter, '
               'input set, channel); the reference is not redesigned for the '
               'gain (the shaped reference of the design gain, or the '
               'unshaped reference)',
        grid=list(FB_LAMBDAS), pid_grid=PID_GRID,
        pid_Imax=sb.K_PID['Imax'],
        training='ideal plant, doubled-increment takeover (increments '
                 'doubled, head speed unchanged), filter on',
        heldout='ideal plant, benchmark takeover, filter on',
        objective=score.__doc__,
        selection='lexicographic minimum of the training score, first in '
                  'grid order on ties',
        feedback_note="'feedback' is the proposed law ('proposed' and "
                      "'feedback' give identical runs on the shaped setup, "
                      "proposed_vs_feedback_over_grid); its retuned gain is "
                      "not certified")
    out['v0_training'] = v0h
    out['selected'] = dict(
        feedback=ctrl['feedback']['best_lam'],
        unshaped=ctrl['unshaped reference']['best_lam'],
        cacc={str(hw): ctrl['cacc h_w=%.1f' % hw]['best_lam']
              for hw in HW_LIST},
        pid=ctrl['pid']['best_gains'])
    # ---- consistency with base.json (design-gain held-out rows)
    checks = {}
    pb = os.path.join(OUT, pname('base') + '.json')
    if os.path.exists(pb):
        BJ = json.load(open(pb))
        if BJ['design']['cfg_used_sha256'] == cfg_hash(B['cfg']):
            rb = {r['name']: r for r in BJ['benchmark']['rows']}
            for name, key, var, hw in FB_CONTROLLERS:
                a = ctrl[name]['design_heldout']
                bname = 'proposed' if var == 'feedback' else name
                b = rb.get(bname, {})
                checks['base.json row %s vs %s at the design gain'
                       % (bname, name)] = bool(all(
                           json.dumps(_py(a.get(k))) == json.dumps(
                               _py(b.get(k))) for k in CMP_KEYS))
            a = ctrl['pid']['hand_tuned_heldout']
            b = rb.get('pid hand-tuned', {})
            checks['base.json row pid hand-tuned'] = bool(all(
                json.dumps(_py(a.get(k))) == json.dumps(_py(b.get(k)))
                for k in CMP_KEYS))
    out['checks'] = checks
    log('  checks against base.json: %s' % checks)
    finish('retune', out, opts, log, t0)


# =================================================================== loss
def part_loss(opts):
    log = Log('loss')
    t0 = time.time()
    log('=== part loss ===')
    X = context(opts)
    B = X['B']
    out = header('loss', X, opts)
    RT = load_part('retune', B)
    lam_fb = float(RT['selected']['feedback'])
    lam0 = float(B['lam'])
    ctrls = [('proposed', 'proposed', lam0, 'design'),
             ('feedback retuned', 'feedback', lam_fb, 'retuned')]
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
                    t_loss_buffer_lost=[r['t_loss'] for r in grp
                                        if r['buffer_lost']],
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
                    'lost %d %s, completed %d, met %d, aborted %d, gmin min '
                    '%.3f, disp max (completed) %s'
                    % (tag, em, wd, s_['n_contact'], s_['n'],
                       s_['t_loss_contact'], s_['n_buffer_lost'],
                       s_['t_loss_buffer_lost'], s_['n_completed'],
                       s_['n_met'], s_['n_aborted'], s_['gmin_min'],
                       fmt(s_['disp_max_completed'])))
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
                                  for i in range(1, B['n'])],
                       gap_error=[float(s[i - 1] - s[i] - B['ell']
                                        - B['d_s'][i])
                                  for i in range(1, B['n'])]))
        log('  nominal state at %4.0f s: v %s, clearances %s, u %s'
            % (tl, np.round(v, 3).tolist(),
               [round(x, 2) for x in st[-1]['clearance']],
               np.round(u, 3).tolist()))
    out.update(rows=rows, summary=summ, runs=runs,
               nominal_state_at_loss=st, t_loss=list(T_LOSS),
               t_loss_previous=list(T_LOSS_PREV),
               t_loss_scaling=dict(
                   T_xi_previous=T_XI_PREV, T_xi=B['T_xi'],
                   factor=B['T_xi'] / T_XI_PREV,
                   scaled_previous=[round(x * B['T_xi'] / T_XI_PREV, 2)
                                    for x in T_LOSS_PREV],
                   choice='15, 30, 60, 90, 120, 140 s (brief) and 145 s = '
                          '84 s x T_xi / T_xi,previous, the instant shortly '
                          'before T_xi'),
               T_wd=list(T_WD), emergency_unit=EM_UNIT,
               gains=dict(design=lam0, feedback_retuned=lam_fb,
                          retune_source='studies/%s.json' % pname('retune')),
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
    X = context(opts)
    B = X['B']
    Bu = X['Bu']
    out = header('plant', X, opts)
    RT = load_part('retune', B)
    selr = RT['selected']
    lam0 = float(B['lam'])
    lam_fb = float(selr['feedback'])
    lam_u = float(selr['unshaped'])
    g_pid = dict(selr['pid'])
    sel = sbh.mpc_selection()
    out['mpc_selection'] = sel
    out['mpc_port'] = mpc_port_note(B)
    out['retuned'] = dict(selr, source='studies/%s.json' % pname('retune'))
    specs = []
    for cn, ov in (('default', {}), ('compensated', dict(COMP))):
        specs += [
            ('proposed, %s' % cn, B, 'proposed', lam0, ov, {}),
            ('feedback retuned lam=%.2f, %s' % (lam_fb, cn), B, 'feedback',
             lam_fb, ov, {}),
            ('unshaped reference retuned lam=%.2f, %s' % (lam_u, cn), Bu,
             'proposed', lam_u, ov, {})]
        for hw in HW_LIST:
            lh = float(selr['cacc'][str(hw)])
            specs.append(('cacc h_w=%.1f retuned lam=%.2f, %s' % (hw, lh, cn),
                          B, 'cacc', lh, ov, dict(h_w=hw)))
        specs.append(('pid retuned, %s' % cn, B, 'pid', lam0, ov,
                      dict(gains=g_pid)))
    for nm, ov in PLANT_VAR_ROWS:
        for cn, co in (('default', {}), ('compensated', dict(COMP))):
            specs.append(('proposed, %s, %s' % (nm, cn), B, 'proposed', lam0,
                          dict(ov, **co), {}))
    # ---- MPC (default plant; sim_bench's MPC has no compensation) in
    #      worker processes, started first
    params = worker_params(opts)
    jobs = [('mpc', sd, opts['design'], params,
             dict(plant='hifi', seed=sd, T_end=B['T_xi'] + 120.0,
                  **mpc_kw(sel))) for sd in SEEDS]
    res = pool_map(jobs, int(opts['workers']), log, 'mpc perturbed plant')
    rows = []
    for name, Bx, var, lam, ov, kw in specs:
        Bg = sbh.with_gain(Bx, lam)
        recs = []
        for sd in SEEDS:
            R = sbh.run(Bg, var, plant='hifi', hifi=ov, seed=sd, **kw)
            recs.append(plant_rec(R))
        row = plant_row(name, var, ov, recs, lam=lam,
                        reference='unshaped' if Bx is Bu else 'design',
                        gains=kw.get('gains'), h_w=kw.get('h_w'),
                        hifi=dict(sb.HIFI_DEFAULT, **ov))
        rows.append(row)
        log(plant_line(row))
    recs = [plant_rec(res[sd]) for sd in SEEDS]
    row = plant_row('mpc (selected weights), default', 'mpc', {}, recs,
                    W=sel['W'], planner=sel['planner'],
                    hifi=dict(sb.HIFI_DEFAULT),
                    note='sim_bench MPC with the substituted takeover '
                         'positions; no compensation option; the '
                         'compensated row is not run')
    row['runtime_solve_ms_mean'] = [res[sd]['runtime_solve_ms_mean']
                                    for sd in SEEDS]
    row['n_solver_not_converged'] = [res[sd]['n_solver_not_converged']
                                     for sd in SEEDS]
    rows.append(row)
    log(plant_line(row))
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
                    'sim_bench (sim_bench_hr docstring); mpc runs in '
                    'sim_bench (with its gate)')
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
    if opts['lam'] is None:
        opts['lam'] = float(json.load(open(opts['design']))['selected']['lam'])
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
