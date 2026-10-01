#!/usr/bin/env python3
"""run_bench_studies.py -- comparison and stress studies of the planned,
deceleration-only arrival (Case D benchmark, design of
data/new_d19/caseD/design.json), 2026-09-28.

Simulator: src/simulation/sim_bench.py.  Every output goes to
data/new_d19/caseD/studies/<part>.json and the log of a part to
logs_ext/caseD_round6/studies_<part>.log.  Nothing else is written.

Parts
  bridge  acceptance tests of the simulator: (T1) the 'proposed' run of
          the ideal plant against run_caseD_validate.simulate(B, h = 1 ms,
          scheme 'zoh', T_u = 1 ms) and against the continuous reference
          (h = 25 us); (T2) 'nofilter' against 'proposed' field by field;
          (T3) the observations of 'proposed' against the certified bounds
          of design.json (settling_bound, T_f_bar, align, gaperr).
          Port checks (additional): (T4) the 'feedback' variant with the
          constant plan and the input set of Case B against
          sim_cont.run(scheme 'zoh', h = 1 ms) on Case B; (T5) the MPC
          with the input set of Case B against sim_mpc.run on Case B;
          (T6) the kernel of the perturbed plant against a plain-Python
          implementation of the same closed loop.
  base    every controller variant on the ideal plant with the filter, the
          PID and headway laws also without it; retuning of the PID law on
          PID_GRID (training: doubled-mismatch takeover) -> one summary
          row per run with the columns of the old Table III plus the
          clip-at-0 time.
  plant   perturbed plant: the rows PLANT_ROWS and PLANT2_ROWS of
          run_b2_ext.py, seeds 1 to 20 each, per-seed records and the
          joint pass count.
  loss    communication loss at t_loss in {15, 25, 35, 50, 70, 84} s for
          the proposed law, without and with emergency braking of unit 3,
          watchdog T_wd in {none, 0.10, 0.20, 0.25, 0.30} s; the runs
          without emergency braking repeated for 'feedback' at 25 and 50 s.
  clock   clock offsets of the units for the proposed law: bound in
          {1, 2, 5, 10, 20, 50, 100} ms; alternating, head late, tail
          late, and 20 random draws per bound (seeds 1 to 20).
  mpc     cooperative arrival MPC with the tuning protocol of
          run_b2_ext.part_mpc (grid MPC_GRID; training: doubled-mismatch
          takeover on the ideal plant and the default perturbed plant,
          seed 1; held-out: nominal takeover on the ideal plant and the
          perturbed plant, seeds 2 to 21).
  all     bridge mpc base plant loss clock
  fbtune  (added 2026-09-28, not part of 'all') retuning of the two
          alternative controllers without follower plans, 'feedback' and
          'cacc' with h_w in {0.2, 0.5, 1.0} s, over the common gain lambda
          of the critically damped family (beta = lambda^2, gamma =
          2 lambda, FB_LAMBDAS) with the protocol of the PID retuning
          (training: doubled-mismatch takeover on the ideal plant;
          held-out: nominal takeover); for 'feedback' with the design gain
          and with its retuned gain: loss of every link at the six
          instants T_LOSS without watchdog and without emergency braking,
          and the perturbed plant without and with compensation of grade
          and resistance, seeds 1 to 20.  The part changes no other part
          and no existing record; it checks the two archived loss runs of
          'feedback', the archived rows of base.json, and the default row
          of plant.json bit for bit.

Usage: python -B run_bench_studies.py <part> ... [--workers N]
"""
import hashlib
import json
import os
import platform
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, os.path.join(CODE, 'src', 'simulation'))
import numpy as np                                       # noqa: E402
import sim_bench as sb                                   # noqa: E402

OUT = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'studies')
LOGDIR = os.path.join(CODE, 'logs_ext', 'caseD_round6')
DESIGN = os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')

REQ = dict(disp=1.0, marker=3.0, gap=1.0)

# ---- constants of the old studies (run_b2_ext.py), verified against the
# originals by _verify_constants()
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
PLANT2_ROWS = [
    ('grade feedforward', dict(grade_ff=1.0)),
    ('grade+resistance ff, lead 0.6', dict(grade_ff=1.0, res_ff=1.0,
                                           ramp_lead=0.6)),
    ('grade+resistance ff, lead 0.75', dict(grade_ff=1.0, res_ff=1.0,
                                            ramp_lead=0.75)),
    ('80% ff, lead 0.6', dict(grade_ff=0.8, res_ff=0.8, ramp_lead=0.6)),
]
# additional rows of the plant part (not in run_b2_ext.py)
PLANT_EXTRA_ROWS = [
    ('resistance feedforward', dict(res_ff=1.0)),
    ('grade+resistance ff, no lead', dict(grade_ff=1.0, res_ff=1.0)),
    ('lead 0.6, no feedforward', dict(ramp_lead=0.6)),
]
MPC_GRID = [dict(w_e=we, w_v=wv, w_du=wd)
            for we in (0.1, 0.3, 1.0, 3.0) for wv in (0.1, 0.3, 1.0)
            for wd in (0.3, 1.0, 3.0, 10.0)]
PID_GRID = [dict(Kp=kp, Kd=kd, Ki=ki)
            for kp in (0.001, 0.003, 0.01) for kd in (0.05, 0.115, 0.30)
            for ki in (1e-6, 5e-5, 2e-4)]     # 1e-6: integral off
# closing-speed limits of the planner layer of the MPC: 0.5 m/s is the
# value of sim_mpc.py (the literal protocol); 1.0 and 2.0 m/s extend the
# grid (README, deviations)
MPC_DV = (0.5, 1.0, 2.0)
T_END_MPC = 120.0            # horizon of the MPC runs after T_xi [s]

T_LOSS = (15.0, 25.0, 35.0, 50.0, 70.0, 84.0)
T_WD = (None, 0.10, 0.20, 0.25, 0.30)
CLOCK_MS = (1, 2, 5, 10, 20, 50, 100)
SEEDS = tuple(range(1, 21))


# ---------------------------------------------------------------- helpers
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
        self.path = os.path.join(LOGDIR, 'studies_%s.log' % part)
        self.f = open(self.path, 'w')
        self.t0 = time.time()

    def __call__(self, msg=''):
        line = str(msg)
        print(line, flush=True)
        self.f.write(line + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def dump(part, obj, log):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, part + '.json')
    json.dump(_py(obj), open(p, 'w'), indent=1)
    log('  -> data/new_d19/caseD/studies/%s.json (%.1f kB)'
        % (part, os.path.getsize(p) / 1e3))
    return p


def cfg_hash(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()
                          ).hexdigest()


def load_design(log=None, wait_s=3600.0, poll_s=120.0):
    """design.json; waits until the round-6 design is in place (revision
    'round 6...', plan keys 'ramp' and 'T_p')."""
    t0 = time.time()
    while True:
        try:
            D = json.load(open(DESIGN))
            pl = D['cfg'].get('plan', {})
            ok = str(D.get('revision', '')).startswith('round 6') \
                and 'ramp' in pl and 'T_p' in pl
        except Exception as ex:                       # being rewritten
            ok = False
            D = None
            if log:
                log('  design.json not readable (%s)' % ex)
        if ok:
            return D
        if time.time() - t0 > wait_s:
            raise RuntimeError('design.json is not the round-6 design')
        if log:
            log('  waiting for the round-6 design.json ...')
        time.sleep(poll_s)


def header(part, D, B, t0):
    cons = D['certificate']['consts']
    return dict(
        part=part, simulator='src/simulation/sim_bench.py',
        driver='run_bench_studies.py',
        design=dict(file='data/new_d19/caseD/design.json',
                    revision=D['revision'], cfg=D['cfg'],
                    cfg_sha256=cfg_hash(D['cfg']),
                    certificate_admitted=D['certificate'].get('admitted'),
                    certified=dict(settling_bound=cons['settling_bound'],
                                   T_f_bar=cons['T_f_bar'],
                                   align=cons['align'],
                                   gaperr=cons['gaperr'])),
        requirements=dict(REQ),
        constants=dict(dt=B['dt'], T_m=B['T_m'], T_c=B['T_c'],
                       transport=B['transport'], age_bound=B['dbar'],
                       k=B['k'], a_b=B['a_b'], alpha=B['alpha'],
                       lam=B['lam'], beta=B['beta'], gamma=B['gamma'],
                       phi=B['phi'], eps_det=B['eps_det'],
                       kappa=B['kappa'], s_m=B['s_m'], ell=B['ell'],
                       v_c=B['v_c'], amax=B['amax'], d_s=B['d_s'],
                       v0=B['v0'], s0=B['s0'], T_xi=B['T_xi'],
                       tau_star=B['tau_star'], v_c1=B['v_c1'],
                       v_xi=B['v_xi'], markers=B['markers'],
                       s_ref_stop=B['s_ref_stop']),
        runtime_date=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_system=dict(python=platform.python_version(),
                            numpy=np.__version__,
                            machine=platform.machine(),
                            system=platform.system()))


def finish(part, obj, D, log, t0):
    D2 = load_design()
    obj['design']['cfg_unchanged_at_end'] = bool(
        cfg_hash(D2['cfg']) == obj['design']['cfg_sha256'])
    obj['runtime_s'] = time.time() - t0
    log('  design cfg unchanged at the end: %s; run time %.1f s'
        % (obj['design']['cfg_unchanged_at_end'], obj['runtime_s']))
    dump(part, obj, log)
    log.close()


def slim(R):
    return {k: v for k, v in R.items() if k != 'rec'}


def fmt(x, f='%.4f', na='n.d.'):
    return na if x is None else f % x


def row_of(name, R, **extra):
    """Summary row of one run: the columns of the old Table III
    (dispersion, marker error, gap error, smallest clearance margin,
    largest |u|, filter time of the worst pair with fallback steps,
    infeasible time summed over the followers, contact time) and the
    clip-at-0 time."""
    done = R['terminal_defined']
    r = dict(
        name=name, variant=R['variant'], plant=R['plant'],
        apply_filter=R['apply_filter'],
        all_stopped=R['all_stopped'],
        disp=R['disp'] if done else None,
        marker=R['marker_abs_max'] if done else None,
        gap=R['gap_abs_max'] if done else None,
        T_f=R['T_f'],
        gmin=R['gmin_min'], gmin_pair=R['gmin_pair'],
        t_gmin=R['t_gmin'][R['gmin_pair'] - 2],
        hmin=R['hmin_min'],
        u_abs_max=R['u_abs_max'], u_min=R['u_min'], u_max=R['u_max'],
        filt=max(R['t_barrier']),
        filt_unit=(int(np.argmax(R['t_barrier'])) + 1)
        if max(R['t_barrier']) > 0 else None,
        filt_raw=max(R['t_barrier_raw']),
        infeas=float(sum(R['t_infeas'])),
        contact=R['contact_time'], contact_pair=R['contact_pair'],
        clip0=max(R['t_clip0']),
        clip0_unit=(int(np.argmax(R['t_clip0'])) + 1)
        if max(R['t_clip0']) > 0 else None,
        clip0_sum=float(sum(R['t_clip0'])), clip0_units=R['t_clip0'],
        nom_pos=max(R['t_nom_pos']),
        clipk=max(R['t_clipk']), at0=max(R['t_at0']),
        at0_units=R['t_at0'],
        buffer_lost=R['buffer_lost'],
        rule_units=R['rule_units'], stop_order=R['stop_order'],
        state_at_end=dict(marker=R['marker_abs_max'], gap=R['gap_abs_max'],
                          t=R['t_end_sim']),
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], pass_all=R['pass_all'])
    r.update(extra)
    return r


def row_line(r):
    return ('  %-26s disp=%s marker=%s gap=%s gmin=%.3f max|u|=%.3f '
            'filt=%.2f infeas=%.2f contact=%s clip0=%.2f stopped=%s pass=%s'
            % (r['name'], fmt(r['disp']), fmt(r['marker'], '%.3f'),
               fmt(r['gap'], '%.3f'), r['gmin'], r['u_abs_max'], r['filt'],
               r['infeas'], fmt(r['contact'], '%.3f', '-'), r['clip0'],
               r['all_stopped'], r['pass_all']))


def score(k):
    """Tuning objective of run_b2_ext._score with the requirements of the
    benchmark: a run with contact or an incomplete stop is rejected
    ([1e9, 1e9]); otherwise the number of violated requirements, then
    disp/1.0 + marker/3 + gap/1."""
    if k.get('contact_time') is not None or not k['all_stopped'] \
            or k['disp'] is None:
        return [1e9, 1e9]
    viol = int(k['disp'] > REQ['disp']) \
        + int(k['marker_abs_max'] > REQ['marker']) \
        + int(k['gap_abs_max'] > REQ['gap'])
    return [viol, k['disp'] / REQ['disp'] + k['marker_abs_max'] / REQ['marker']
            + k['gap_abs_max'] / REQ['gap']]


def joint(R):
    """run_b2_ext._joint with the requirements of the benchmark."""
    return bool(R['all_stopped'] and R['contact_time'] is None
                and R['disp'] is not None and R['disp'] <= REQ['disp']
                and R['marker_abs_max'] <= REQ['marker']
                and R['gap_abs_max'] <= REQ['gap'])


def _verify_constants(log):
    """The copies above against run_b2_ext.py, sim_root2.py, sim_hifi.py
    and sim_mpc.py (read-only imports)."""
    out = {}
    try:
        import run_b2_ext as rx
        import sim_root2 as sr2
        import sim_hifi as sh
        import sim_mpc as sm

        def same(a, b):
            return json.dumps(_py(a), sort_keys=True) == \
                json.dumps(_py(b), sort_keys=True)
        out = dict(
            PLANT_ROWS=same(PLANT_ROWS, rx.PLANT_ROWS),
            PLANT2_ROWS=same(PLANT2_ROWS, rx.PLANT2_ROWS),
            MPC_GRID=same(MPC_GRID, rx.MPC_GRID),
            PID_GRID=same(PID_GRID, rx.PID_GRID),
            K_PID=same(sb.K_PID, sr2.K_PID),
            K_CACC=same(sb.K_CACC, sr2.K_CACC),
            HIFI_DEFAULT=same(sb.HIFI_DEFAULT, sh.DEFAULT),
            MPC=same([sb.MPC_DEFAULT_W, sb.MPC_SIG, sb.T_MPC, sb.N_H,
                      sb.V_STOP_MPC, sb.TAU_C, sb.DV_MAX],
                     [sm.DEFAULT_W, sm.SIG, sm.T_MPC, sm.N_H,
                      sm.V_STOP_IDEAL, sm.TAU_C, sm.DV_MAX]))
        out['all'] = all(out.values())
    except Exception as ex:                           # pragma: no cover
        out = dict(all=False, error=str(ex))
    log('  constants of the old studies verified: %s' % out)
    return out


# ================================================================= bridge
def _maxdiff(a, b):
    a = np.array([np.nan if x is None else x for x in a], float)
    b = np.array([np.nan if x is None else x for x in b], float)
    if a.shape != b.shape or np.any(np.isnan(a) != np.isnan(b)):
        return None
    if np.all(np.isnan(a)):
        return 0.0
    return float(np.nanmax(np.abs(a - b)))


def _port_checks(log):
    """(T4) and (T5): the 'feedback' variant and the MPC on Case B (the
    benchmark of the main text: run_b2.py) against the simulators of the
    old studies.  The plan of the head is the constant plan of
    certify_plan (a* = 0, v* = v_xi until T_xi), so the 'feedback'
    variant is the law of the main text, and the input set is that of
    Case B, [-min(U, amax_i), U]."""
    import run_b2 as rb
    import sim_cont as sc
    import sim_mpc as sm
    cfgB = dict(n=5, amax=list(rb.AMAX), k=1.6, a_b=rb.B2['a_b'],
                alpha=rb.B2['alpha'], lam=rb.B2['lam'], s_m=rb.B2['s_m'],
                eps_det=0.015, phi=0.005, T_c=0.001, dbar=0.02,
                d_s=list(rb.D_S), c0=list(rb.C0),
                v0=[repr(x) for x in rb.V0],
                plan=dict(family='constant', v_xi=repr(rb.B2['v_xi']),
                          T_xi=repr(rb.B2['T_xi'])))
    bounds = ([-min(1.6, a) for a in rb.AMAX], 1.6)
    T_end = rb.SIM_P['T_END']

    def case_b(dbar):
        Bb = sb.setup(dict(cfgB, dbar=dbar))
        Bb['s0'] = -np.cumsum(Bb['ell'] + Bb['d_s']
                              + np.array(rb.E0, float))
        Bb['v0'] = np.array(rb.V0, float)
        return Bb
    # ---- T4
    Bb = case_b(0.02)
    R = sb.run(Bb, 'feedback', bounds=bounds, T_end=T_end)
    C = sc.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, h=1e-3, T_m=1e-3, T_c=1e-3,
               T_u=1e-3, scheme='zoh', transport=0.019)
    d4 = dict(
        stop_instants=_maxdiff(R['tstop'], C['tstop']),
        s1_detection_instants=_maxdiff(R['tset'], C['tset']),
        activation_instants=_maxdiff(R['tact'], C['tact']),
        rule_entries=_maxdiff(R['trule'][1:], C['trule'][1:]),
        marker_errors=_maxdiff(R['marker_signed'], C['marker_signed']),
        gap_errors=_maxdiff(R['gap_signed'], np.asarray(C['egap'])[1:]),
        clearance_margin_minima=_maxdiff(R['gmin'], C['gmin']))
    t4 = dict(
        case='Case B (run_b2.py), transport 19 ms',
        reference='sim_cont.run(scheme zoh, h = T_u = 1 ms)',
        max_abs_difference=d4,
        tolerance=dict(time=1e-5, position=1e-6),
        note='sim_cont updates the command of a follower at the zero-speed '
             'instant of its predecessor inside the step (RULE); sim_bench '
             'enters RULE at the next grid instant',
        sim_bench=dict(disp=R['disp'], marker=R['marker_abs_max'],
                       gap=R['gap_abs_max'], tstop=R['tstop'],
                       gmin=R['gmin_min'], rule_units=R['rule_units']),
        sim_cont=dict(disp=C['disp'], marker=C['align'], gap=C['gaperr'],
                      tstop=C['tstop'], gmin=float(np.min(C['gmin']))))
    t4['pass'] = bool(d4['stop_instants'] < 1e-5
                      and d4['s1_detection_instants'] < 1e-9
                      and d4['activation_instants'] < 1e-9
                      and d4['marker_errors'] < 1e-6
                      and d4['gap_errors'] < 1e-6)
    log('  T4 feedback law on Case B against sim_cont (zoh, 1 ms): max '
        '|difference| stop %.3e s, S1 detection %.3e s, RULE entry %.3e s, '
        'marker %.3e m, gap %.3e m, pass %s'
        % (d4['stop_instants'], d4['s1_detection_instants'],
           d4['rule_entries'], d4['marker_errors'], d4['gap_errors'],
           t4['pass']))
    # ---- T5
    W = dict(w_e=1.0, w_v=1.0, w_du=10.0)
    Bm = case_b(0.021)                  # sim_mpc: delay of 20 steps
    M = sb.run(Bm, 'mpc', W=W, bounds=bounds, T_end=T_end,
               mpc_stop='detector')
    O = sm.run(rb.SIM_P, rb.D_S, rb.V0, rb.E0, W=W, plant='ideal')
    d5 = dict(
        stop_instants=_maxdiff(M['tstop'], O['tstop']),
        dispersion=abs(M['disp'] - O['disp']),
        marker_error=abs(M['marker_abs_max'] - O['align']),
        gap_errors=_maxdiff(M['gap_signed'], O['egap'][1:]),
        clearance_margin_minima=_maxdiff(M['gmin'], O['gmin']))
    t5 = dict(
        case='Case B (run_b2.py), weights of the old study, delay of 20 '
             'steps',
        reference='sim_mpc.run(plant ideal)', W=W,
        max_abs_difference=d5,
        tolerance=dict(time=2e-3, position=1e-3),
        note='sim_mpc integrates with v <- max(0, v + u dt), s += v dt and '
             'clips the command to [-amax_i, amax_i]; sim_bench integrates '
             'exactly and uses [-amax_i, U]',
        sim_bench=dict(disp=M['disp'], marker=M['marker_abs_max'],
                       gap=M['gap_abs_max'], tstop=M['tstop'],
                       gmin=M['gmin_min']),
        sim_mpc=dict(disp=O['disp'], marker=O['align'], gap=O['gaperr'],
                     tstop=O['tstop'], gmin=min(O['gmin'])))
    t5['pass'] = bool(d5['stop_instants'] <= 2e-3
                      and d5['marker_error'] <= 1e-3
                      and d5['gap_errors'] <= 1e-3)
    log('  T5 MPC on Case B against sim_mpc: max |difference| stop %.3e s, '
        'dispersion %.3e s, marker %.3e m, gap %.3e m, pass %s'
        % (d5['stop_instants'], d5['dispersion'], d5['marker_error'],
           d5['gap_errors'], t5['pass']))
    return t4, t5


def _hifi_reference(B, ov, seed, T_end):
    """Plain-Python implementation of the proposed law on the perturbed
    plant (no code of the kernel; plan states from the Plan object)."""
    H = dict(sb.HIFI_DEFAULT, **ov)
    Pl = B['Pl']
    n = B['n']
    dt = B['dt']
    N = int(round(T_end / dt))
    hold = int(round(H['T_s'] / dt))
    Z = np.random.default_rng(seed).standard_normal((N // hold + 2, n, 4))
    kk, a_b, alpha = B['k'], B['a_b'], B['alpha']
    beta, gamma, phi = B['beta'], B['gamma'], B['phi']
    eps_det, kappa, ell, s_m = B['eps_det'], B['kappa'], B['ell'], B['s_m']
    b, amax, T_xi = B['bsl'], B['amax'], B['T_xi']
    lead = H['ramp_lead']
    tau_act = np.broadcast_to(np.asarray(H['tau_act'], float),
                              (n,)).astype(float)
    grade = H['grade'] * np.array(sb.GRADE_SIGNS[:n])
    a0, a1, a2 = H['davis']
    q = H['quant_vel']

    def plan(i, t):
        ps, pv, pa = Pl.state_f(i, t)
        return float(ps), float(pv), float(pa)
    s = B['s0'].copy()
    v = B['v0'].copy()
    mode = ['S1'] + ['COAST'] * (n - 1)
    flag = [False] * n
    rflag = [False] * n
    uh = np.zeros(n)
    uc = np.zeros(n)
    ua = np.zeros(n)
    comp = np.zeros(n)
    stopped = [False] * n
    tstop = [None] * n
    tset = [None] * n
    sent = np.zeros((n, N + 2))
    sentf = np.zeros((n, N + 2), bool)
    gmin = np.full(n - 1, np.inf)
    clip0 = np.zeros(n)
    for k in range(N + 1):
        t = k * dt
        tl = round((t + lead) * 1000) / 1000.0
        if k % hold == 0:
            jz = k // hold
            vim = [np.round((v[i] + H['noise_vel'] * Z[jz, i, 1]) / q) * q
                   for i in range(n)]
            vpm = [None] + [np.round((v[i - 1] + H['noise_vel']
                                      * Z[jz, i, 2]) / q) * q
                            for i in range(1, n)]
            sg = [None] + [(s[i - 1] - s[i]) + H['bias_gap']
                           + H['noise_gap'] * Z[jz, i, 0]
                           for i in range(1, n)]
            pe0 = H['bias_gap'] + H['noise_gap'] * Z[jz, 0, 0]
            for i in range(n):
                if not stopped[i] and mode[i] in ('S2', 'RULE', 'BRAKE') \
                        and vim[i] <= H['v_stop'] and t > T_xi:
                    stopped[i] = True
                    tstop[i] = t
                    v[i] = 0.0
                    mode[i] = 'LATCH'
                    flag[i] = True
            for i in range(1, n):
                if mode[i] == 'S2' and vpm[i] <= eps_det and t > T_xi:
                    mode[i] = 'RULE'
            j = k - B['Ntr']
            if j >= 0:
                for i in range(1, n):
                    uh[i] = sent[i - 1, j]
                    if sentf[i - 1, j]:
                        rflag[i] = True
            for i in range(1, n):
                if mode[i] == 'COAST' and rflag[i]:
                    mode[i] = 'S1'
            if mode[0] in ('S1', 'S2') and t + lead >= T_xi - 1e-9:
                mode[0] = 'BRAKE'
                flag[0] = True
            for i in range(n):
                if mode[i] == 'S1':
                    po = plan(i, t)
                    if i == 0:
                        x = po[1] - vim[0]
                    else:
                        pp = plan(i - 1, t)
                        x = (vpm[i] - vim[i]) - (pp[1] - po[1])
                    if abs(x) <= eps_det:
                        mode[i] = 'S2'
                        flag[i] = True
                        tset[i] = t
            for i in range(n):
                dumax = H['jerk_max'] * H['T_s']
                if stopped[i]:
                    comp[i] = 0.0
                    uc[i] = float(np.clip(0.0, uc[i] - dumax, uc[i] + dumax))
                    continue
                po = plan(i, t)
                pol = plan(i, tl)
                if i == 0:
                    e = (po[0] - s[0]) + pe0
                    x = po[1] - vim[0]
                    ff = pol[2]
                else:
                    pp = plan(i - 1, t)
                    ppl = plan(i - 1, tl)
                    e = sg[i] - (pp[0] - po[0])
                    x = (vpm[i] - vim[i]) - (pp[1] - po[1])
                    ff = pol[2] + uh[i]
                    ub = ppl[2] + uh[i]
                m = mode[i]
                if m in ('RULE', 'BRAKE'):
                    un = -a_b
                elif m == 'S2':
                    un = ff + beta * e + gamma * x
                elif m == 'S1':
                    un = ff + alpha * float(np.clip(x / phi, -1, 1))
                else:
                    un = ff
                uq = un
                if i >= 1:
                    g = sg[i] - ell - s_m
                    hb = g + vpm[i] / b[i - 1] - vim[i] / b[i]
                    ucbf = b[i] * ((vpm[i] - vim[i]) + ub / b[i - 1]
                                   + kappa * hb)
                    if ucbf < -kk - 1e-12:
                        uq = -kk
                    elif ucbf < un:
                        uq = ucbf
                if uq > 0:
                    clip0[i] += H['T_s']
                u = min(0.0, max(-kk, uq))
                c = H['grade_ff'] * grade[i] + H['res_ff'] * (
                    a0 + a1 * abs(vim[i]) + a2 * vim[i] ** 2)
                comp[i] = c
                x = float(np.clip(u + c, uc[i] - dumax, uc[i] + dumax))
                uc[i] = float(np.clip(x, -H['derate'] * amax[i], sb.U_ACT))
        for i in range(n):
            sent[i, k] = (uc[i] - comp[i]) - plan(i, tl)[2]
            sentf[i, k] = flag[i]
        if all(stopped) or k == N:
            break
        for i in range(n):
            if tau_act[i] > 0:
                ua[i] += dt / tau_act[i] * (uc[i] - ua[i])
            else:
                ua[i] = uc[i]
            if stopped[i]:
                continue
            acc = ua[i] - np.sign(v[i]) * (a0 + a1 * v[i]
                                           + a2 * v[i] ** 2) - grade[i]
            v[i] = max(0.0, v[i] + acc * dt)
            s[i] += v[i] * dt
        for i in range(1, n):
            gmin[i - 1] = min(gmin[i - 1], s[i - 1] - s[i] - ell - s_m)
    return dict(tstop=tstop, tset=tset, marker=s - B['markers'],
                gap=(s[:-1] - s[1:] - ell) - B['d_s'][1:], gmin=gmin,
                clip0=clip0)


def _hifi_check(B, log):
    """(T6): kernel of the perturbed plant against _hifi_reference."""
    rows = []
    for name, ov, sd in (
            ('default', {}, 1),
            ('grade+resistance ff, lead 0.6',
             dict(grade_ff=1.0, res_ff=1.0, ramp_lead=0.6), 3),
            ('het lag 1.0 +/-20%',
             dict(tau_act=[1.20, 0.80, 1.20, 0.80, 1.20]), 5)):
        P = _hifi_reference(B, ov, sd, B['T_xi'] + 60.0)
        R = sb.run(B, 'proposed', plant='hifi', hifi=ov, seed=sd)
        d = dict(stop_instants=_maxdiff(R['tstop'], P['tstop']),
                 s1_detection_instants=_maxdiff(R['tset'], P['tset']),
                 marker_errors=_maxdiff(R['marker_signed'], P['marker']),
                 gap_errors=_maxdiff(R['gap_signed'], P['gap']),
                 clearance_margin_minima=_maxdiff(R['gmin'], P['gmin']),
                 clip0_time=_maxdiff(R['t_clip0'], P['clip0']))
        ok = bool(all(x is not None and x < 1e-9 for x in d.values()))
        rows.append(dict(row=name, seed=sd, max_abs_difference=d, equal=ok))
        log('  T6 perturbed plant %-30s seed %d: max |difference| stop '
            '%.1e s, marker %.1e m, gap %.1e m, gmin %.1e m, equal %s'
            % (name, sd, d['stop_instants'], d['marker_errors'],
               d['gap_errors'], d['clearance_margin_minima'], ok))
    return dict(rows=rows, tolerance=1e-9,
                reference='_hifi_reference of run_bench_studies.py',
                **{'pass': bool(all(r['equal'] for r in rows))})


def part_bridge(args):
    log = Log('bridge')
    t0 = time.time()
    log('=== part bridge ===')
    D = load_design(log)
    cfg = D['cfg']
    B = sb.setup(cfg)
    out = header('bridge', D, B, t0)
    R = sb.run(B, 'proposed')
    R = sb.run(B, 'proposed')                 # second call: compiled kernel
    # ---------------------------------------------------------------- T1
    import run_caseD_validate as rv
    Bp = rv.build(cfg)
    Z = rv.simulate(Bp, h=1e-3, scheme='zoh', T_u=1e-3)
    kz = rv.kpis(Z, Bp)
    C = rv.simulate(Bp, h=2.5e-5)
    kc = rv.kpis(C, Bp)
    t1 = {}
    for tag, K in (('project_zoh_1ms', kz), ('continuous_reference', kc)):
        d = dict(
            stop_instants=_maxdiff(R['tstop'], K['tstop']),
            settling=abs(R['disp'] - float(K['settling'])),
            marker_errors=_maxdiff(R['marker_signed'], K['marker_signed']),
            gap_errors=_maxdiff(R['gap_signed'], K['gap_signed']),
            s1_detection_instants=_maxdiff(R['tset'], K['tset']),
            activation_instants=_maxdiff(R['tact'], K['tact']),
            clearance_margin_minima=_maxdiff(R['gmin'], K['gmin']),
            barrier_minima=_maxdiff(R['hmin'], K['hmin']),
            command_minima=_maxdiff(R['u_min_unit'], K['u_min_unit']),
            command_maxima=_maxdiff(R['u_max_unit'], K['u_max_unit']),
            inactivity_margin_minima=_maxdiff(R['mmin'], K['mmin_lim']))
        t1[tag] = dict(
            max_abs_difference=d,
            stop_order_equal=bool(R['stop_order'] == K['stop_order']),
            rule_entered=[R['rule_entered'], bool(K['rule_entered'])],
            filter_modifications_project=int(K['filter_modifications']),
            project=dict(
                tstop=K['tstop'], settling=K['settling'], T_f=K['T_f'],
                marker_signed=K['marker_signed'],
                gap_signed=K['gap_signed'], tset=K['tset'], tact=K['tact'],
                gmin=K['gmin'], hmin=K['hmin'],
                u_min_unit=K['u_min_unit'], u_max_unit=K['u_max_unit'],
                stop_order=K['stop_order'], config=K['config'],
                runtime_s=K['runtime_s']))
        log('  T1 %-22s max |difference|: stop %.3e s, marker %.3e m, gap '
            '%.3e m, S1 detection %.3e s, settling %.3e s'
            % (tag, d['stop_instants'], d['marker_errors'], d['gap_errors'],
               d['s1_detection_instants'], d['settling']))
    dz = t1['project_zoh_1ms']['max_abs_difference']
    t1['tolerance'] = dict(time=1e-6, position=1e-6)
    t1['pass'] = bool(
        dz['stop_instants'] is not None and dz['stop_instants'] < 1e-6
        and dz['s1_detection_instants'] < 1e-6
        and dz['marker_errors'] < 1e-6 and dz['gap_errors'] < 1e-6
        and t1['project_zoh_1ms']['stop_order_equal'])
    log('  T1 pass (1 kHz run against the zoh run of the project, 1e-6 s '
        'and 1e-6 m): %s' % t1['pass'])
    # ---------------------------------------------------------------- T2
    Rn = sb.run(B, 'nofilter')
    skip = ('variant', 'apply_filter', 'runtime_s')
    diffs = []
    nfield = 0
    for key in sorted(R):
        if key in skip:
            continue
        nfield += 1
        a = json.dumps(_py(R[key]), sort_keys=True)
        b = json.dumps(_py(Rn.get(key)), sort_keys=True)
        if a != b:
            diffs.append(dict(field=key, proposed=R[key],
                              nofilter=Rn.get(key)))
    t2 = dict(fields_compared=nfield, fields_excluded=list(skip),
              fields_different=diffs, identical=bool(not diffs),
              filter_time_proposed=max(R['t_barrier']),
              infeasible_time_proposed=float(sum(R['t_infeas'])),
              smallest_inactivity_margin=min(R['mmin']))
    t2['pass'] = t2['identical']
    log('  T2 nofilter against proposed: %d fields compared, %d different, '
        'pass %s' % (nfield, len(diffs), t2['pass']))
    # ---------------------------------------------------------------- T3
    cons = D['certificate']['consts']
    rows = []

    def chk(name, obs, bound):
        ok = bool(obs is not None and obs <= bound)
        rows.append(dict(observation=name, observed=obs, bound=bound,
                         inside=ok))
        log('  T3 %-28s observed %.6g  bound %.6g  inside %s'
            % (name, obs, bound, ok))
    chk('settling time [s]', R['disp'], float(cons['settling_bound']))
    chk('completion T_f [s]', R['T_f'], float(cons['T_f_bar']))
    al = cons['align']
    for i in range(B['n']):
        bnd = float(al[i]) if isinstance(al, list) else float(al)
        chk('marker error unit %d [m]' % (i + 1),
            abs(R['marker_signed'][i]), bnd)
    ge = cons['gaperr']
    for i in range(B['n'] - 1):
        bnd = float(ge[i]) if isinstance(ge, list) else float(ge)
        chk('gap error pair %d [m]' % (i + 2), abs(R['gap_signed'][i]), bnd)
    t3 = dict(rows=rows, n_rows=len(rows),
              n_inside=sum(1 for r in rows if r['inside']))
    t3['pass'] = bool(t3['n_inside'] == t3['n_rows'])
    log('  T3 pass: %s (%d of %d inside)' % (t3['pass'], t3['n_inside'],
                                             t3['n_rows']))
    t4, t5 = _port_checks(log)
    t6 = _hifi_check(B, log)
    out.update(T1=t1, T2=t2, T3=t3, T4=t4, T5=t5, T6=t6,
               all_pass=bool(t1['pass'] and t2['pass'] and t3['pass']),
               port_checks_pass=bool(t4['pass'] and t5['pass']
                                     and t6['pass']),
               run_proposed=slim(R), run_nofilter=slim(Rn))
    log('  bridge: acceptance tests T1-T3 all pass %s; port checks T4-T6 '
        'pass %s' % (out['all_pass'], out['port_checks_pass']))
    finish('bridge', out, D, log, t0)
    return out['all_pass']


# =================================================================== base
def _mpc_frozen(log):
    p = os.path.join(OUT, 'mpc.json')
    if not os.path.exists(p):
        log('  mpc.json not found: the MPC rows are omitted (run part mpc '
            'first)')
        return None
    return json.load(open(p))


def part_base(args):
    log = Log('base')
    t0 = time.time()
    log('=== part base ===')
    D = load_design(log)
    B = sb.setup(D['cfg'])
    out = header('base', D, B, t0)
    out['constants_verified'] = _verify_constants(log)
    v0h = sb.doubled_mismatch(B)
    T_end = B['T_xi'] + 60.0
    # ---- PID retuning (protocol of run_b2_ext.part_mpc, PID part)
    cands = []
    for K in PID_GRID:
        g = dict(sb.K_PID, **K)
        S = sb.run(B, 'pid', gains=g, v0=v0h, T_end=T_end)
        cands.append(dict(K=K, score=score(S),
                          train=row_of('pid train', S), train_run=slim(S)))
        log('  pid grid %s -> score %s | disp=%s marker=%.3f gap=%.3f '
            'filt=%.2f clip0=%.2f contact=%s'
            % (K, cands[-1]['score'], fmt(S['disp']), S['marker_abs_max'],
               S['gap_abs_max'], max(S['t_barrier']), max(S['t_clip0']),
               S['contact_time']))
    best = min(cands, key=lambda c: tuple(c['score']))
    K_best = dict(sb.K_PID, **best['K'])
    log('  pid best %s score %s' % (best['K'], best['score']))
    S0 = sb.run(B, 'pid', gains=dict(sb.K_PID), v0=v0h, T_end=T_end)
    out['pid_retune'] = dict(
        protocol=dict(
            grid='PID_GRID of run_b2_ext.py (27 gain triples, Imax of '
                 'sim_root2.K_PID)',
            training='ideal plant, doubled-mismatch takeover (speed '
                     'increments doubled, head speed unchanged), filter on',
            heldout='ideal plant, nominal takeover',
            objective=score.__doc__),
        v0_training=v0h, grid=PID_GRID, candidates=cands,
        best_K=best['K'], best_score=best['score'],
        original_K=dict(sb.K_PID), original_score=score(S0),
        original_train=row_of('pid hand-tuned train', S0))
    # ---- rows
    specs = [
        ('proposed', 'proposed', {}),
        ('nofilter', 'nofilter', {}),
        ('nodev', 'nodev', {}),
        ('planonly', 'planonly', {}),
        ('single', 'single', {}),
        ('pid hand-tuned', 'pid', dict(gains=dict(sb.K_PID))),
        ('pid retuned', 'pid', dict(gains=K_best)),
        ('feedback', 'feedback', {}),
        ('cacc h_w=0.5', 'cacc', dict(h_w=0.5)),
        ('cacc h_w=0.2', 'cacc', dict(h_w=0.2)),
        ('cacc h_w=1.0', 'cacc', dict(h_w=1.0)),
        ('pid hand-tuned, filter off', 'pid',
         dict(gains=dict(sb.K_PID), apply_filter=False)),
        ('pid retuned, filter off', 'pid',
         dict(gains=K_best, apply_filter=False)),
        ('feedback, filter off', 'feedback', dict(apply_filter=False)),
        ('cacc h_w=0.5, filter off', 'cacc',
         dict(h_w=0.5, apply_filter=False)),
        ('cacc h_w=0.2, filter off', 'cacc',
         dict(h_w=0.2, apply_filter=False)),
        ('cacc h_w=1.0, filter off', 'cacc',
         dict(h_w=1.0, apply_filter=False)),
    ]
    rows = []
    runs = {}
    for name, var, kw in specs:
        R = sb.run(B, var, T_end=T_end, **kw)
        r = row_of(name, R, gains=R['gains'], h_w=R['h_w'])
        rows.append(r)
        runs[name] = slim(R)
        log(row_line(r))
    M = _mpc_frozen(log)
    if M is not None:
        same = M['design']['cfg_sha256'] == out['design']['cfg_sha256']
        out['mpc_source'] = dict(file='studies/mpc.json',
                                 same_design=bool(same))
        for tag in ('literal', 'extended'):
            P = M['protocols'][tag]
            W = P['best']['W']
            pln = P['best']['planner']
            for stop in ('detector', 'latch'):
                R = sb.run(B, 'mpc', W=W, mpc_planner=pln, mpc_stop=stop,
                           T_end=B['T_xi'] + T_END_MPC)
                name = 'mpc %s grid, stop %s' % (tag, stop)
                r = row_of(name, R, W=W, planner=pln, mpc_stop=stop,
                           runtime_solve_ms_mean=R['runtime_solve_ms_mean'])
                rows.append(r)
                runs[name] = slim(R)
                log(row_line(r))
    out['rows'] = rows
    out['runs'] = runs
    # ---- additional: the laws with plans on the doubled-mismatch takeover
    rows2 = []
    for name, var, kw in (
            ('proposed', 'proposed', {}),
            ('nofilter', 'nofilter', {}),
            ('nodev', 'nodev', {}),
            ('single', 'single', {}),
            ('pid hand-tuned', 'pid', dict(gains=dict(sb.K_PID))),
            ('pid retuned', 'pid', dict(gains=K_best))):
        R = sb.run(B, var, v0=v0h, T_end=T_end, **kw)
        r = row_of(name + ', doubled mismatch', R, gains=R['gains'],
                   v0=v0h, tset=R['tset'])
        rows2.append(r)
        log(row_line(r))
    out['rows_doubled_mismatch'] = rows2
    out['columns'] = dict(
        disp='stopping-time dispersion, last stop minus first stop [s] '
             '(None: not every unit stopped, or contact)',
        marker='largest |marker error| [m]', gap='largest |terminal gap '
        'error| [m]', gmin='smallest clearance margin g over the run and '
        'the pairs [m] (intersample minimum)',
        u_abs_max='largest |applied command| before the latch [m/s^2]',
        filt='time in which the filter lowered the applied command of the '
             'worst follower, fallback steps included [s]',
        filt_raw='time with min{u_nom, u_cbf} (or the fallback) below '
                 'u_nom before the input clip, worst follower [s]',
        infeas='time with u_cbf < -k, summed over the followers [s]',
        contact='contact instant [s] (None: no contact)',
        clip0='time in which the clip at 0 acted (filter output above 0), '
              'worst unit [s]',
        nom_pos='time with u_nom > 0, worst unit [s]',
        clipk='time with the applied command at -k, worst unit [s]',
        at0='time with the applied command at 0 before the latch, worst '
            'unit [s]')
    finish('base', out, D, log, t0)


# ================================================================== plant
def part_plant(args):
    log = Log('plant')
    t0 = time.time()
    log('=== part plant ===')
    D = load_design(log)
    B = sb.setup(D['cfg'])
    out = header('plant', D, B, t0)
    out['constants_verified'] = _verify_constants(log)
    out['hifi_default'] = dict(sb.HIFI_DEFAULT)
    out['grade_signs'] = list(sb.GRADE_SIGNS)
    out['actuator_limits'] = dict(lower='-derate * amax_i', upper=sb.U_ACT)
    rows = []
    for name, ov in PLANT_ROWS + PLANT2_ROWS + PLANT_EXTRA_ROWS:
        rs = []
        for sd in SEEDS:
            R = sb.run(B, 'proposed', plant='hifi', hifi=ov, seed=sd)
            rec = dict(
                seed=sd, disp=R['disp'], marker=R['marker_abs_max'],
                gap=R['gap_abs_max'], marker_signed=R['marker_signed'],
                gap_signed=R['gap_signed'], tstop=R['tstop'], T_f=R['T_f'],
                stop_order=R['stop_order'],
                gmin=R['gmin_min'], gmin_pair=R['gmin_pair'],
                gmin_pairs=R['gmin'], hmin=R['hmin_min'],
                filt=max(R['t_barrier']), infeas=float(sum(R['t_infeas'])),
                clip0=max(R['t_clip0']), clip0_units=R['t_clip0'],
                nom_pos=max(R['t_nom_pos']), clipk=max(R['t_clipk']),
                at0=max(R['t_at0']),
                u_min=R['u_min'], u_max=R['u_max'],
                u_realized_abs_max=max(R['u_realized_abs_max']),
                jerk_pk=max(R['jerk_pk']),
                stopped=R['all_stopped'], contact=R['contact_time'],
                buffer_lost=R['buffer_lost'], tset=R['tset'],
                rule_units=R['rule_units'],
                pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
                pass_gap=R['pass_gap'], runtime_s=R['runtime_s'])
            rec['joint'] = joint(R)
            rs.append(rec)
        dd = [r['disp'] for r in rs if r['disp'] is not None]
        row = dict(
            name=name, overrides=ov, n=len(rs),
            disp_med=float(np.median(dd)) if dd else None,
            disp_min=min(dd) if dd else None,
            disp_max=max(dd) if dd else None,
            marker_max=max(r['marker'] for r in rs),
            marker_med=float(np.median([r['marker'] for r in rs])),
            gap_max=max(r['gap'] for r in rs),
            gap_med=float(np.median([r['gap'] for r in rs])),
            gmin_min=min(r['gmin'] for r in rs),
            filt_runs=sum(1 for r in rs if r['filt'] > 0),
            filt_max=max(r['filt'] for r in rs),
            infeas_runs=sum(1 for r in rs if r['infeas'] > 0),
            clip0_runs=sum(1 for r in rs if r['clip0'] > 0),
            clip0_med=float(np.median([r['clip0'] for r in rs])),
            clip0_max=max(r['clip0'] for r in rs),
            T_f_max=max([r['T_f'] for r in rs if r['T_f'] is not None]
                        or [None]),
            contacts=sum(1 for r in rs if r['contact'] is not None),
            buffer_lost_runs=sum(1 for r in rs if r['buffer_lost']),
            stopped=sum(1 for r in rs if r['stopped']),
            pass_disp=sum(1 for r in rs if r['pass_disp']),
            pass_marker=sum(1 for r in rs if r['pass_marker']),
            pass_gap=sum(1 for r in rs if r['pass_gap']),
            joint_pass=sum(1 for r in rs if r['joint']),
            seeds=rs)
        rows.append(row)
        log('  %-32s joint %2d/%d  disp med/max %s/%s marker %.2f gap %.2f '
            'gmin %.2f filt %d clip0 med %.1f s contacts %d stopped %d '
            '(pass d/m/g %d/%d/%d)'
            % (name, row['joint_pass'], row['n'],
               fmt(row['disp_med'], '%.3f'),
               fmt(row['disp_max'], '%.3f'), row['marker_max'],
               row['gap_max'], row['gmin_min'], row['filt_runs'],
               row['clip0_med'], row['contacts'], row['stopped'],
               row['pass_disp'], row['pass_marker'], row['pass_gap']))
    nspec = len(PLANT_ROWS) + len(PLANT2_ROWS)
    out['rows'] = rows[:nspec]
    out['rows_extra'] = rows[nspec:]
    out['rows_extra_note'] = ('additional rows, not in run_b2_ext.py: they '
                              'separate the resistance feedforward and '
                              'the lead')
    out['seeds'] = list(SEEDS)
    finish('plant', out, D, log, t0)


# =================================================================== loss
def _loss_row(R):
    k = int(np.argmin(R['gmin']))
    tb = [x for x in R['t_buffer_lost'] if x is not None]
    aborted = list(R['override_units'])
    completed = bool(R['all_stopped'] and not aborted)
    return dict(
        variant=R['variant'], t_loss=R['t_loss'],
        emergency_unit=R['emergency_unit'], T_wd=R['T_wd'],
        buffer_lost=R['buffer_lost'],
        t_buffer_lost=min(tb) if tb else None,
        buffer_lost_pairs=[i + 2 for i, x in enumerate(R['t_buffer_lost'])
                           if x is not None],
        t_buffer_lost_pairs=R['t_buffer_lost'],
        contact=R['contact_time'] is not None,
        t_contact=R['contact_time'], contact_pair=R['contact_pair'],
        gmin=R['gmin_min'], gmin_pair=k + 2, t_gmin=R['t_gmin'][k],
        gmin_pairs=R['gmin'],
        all_stopped=R['all_stopped'], n_stopped=R['n_stopped'],
        override_units=aborted, watchdog_units=R['watchdog_units'],
        t_watchdog=R['t_watchdog'],
        arrival_completed=completed,
        disp=R['disp'] if completed else None,
        marker=R['marker_abs_max'] if completed else None,
        gap=R['gap_abs_max'] if completed else None,
        pass_all=bool(completed and R['pass_all']),
        tstop=R['tstop'], T_f=R['T_f'], stop_order=R['stop_order'],
        rule_units=R['rule_units'],
        filt=max(R['t_barrier']), infeas=float(sum(R['t_infeas'])),
        clip0=max(R['t_clip0']), clipk=max(R['t_clipk']),
        at0=max(R['t_at0']),
        u_min=R['u_min'],
        state_at_end=dict(t=R['t_end_sim'], marker_signed=R['marker_signed'],
                          gap_signed=R['gap_signed'], v=R['v_final'],
                          mode=R['mode_final']))


def part_loss(args):
    log = Log('loss')
    t0 = time.time()
    log('=== part loss ===')
    D = load_design(log)
    B = sb.setup(D['cfg'])
    out = header('loss', D, B, t0)
    rows = []
    runs = []
    T_end = B['T_xi'] + 60.0
    jobs = [('proposed', tl, em, wd) for em in (None, 3) for tl in T_LOSS
            for wd in T_WD]
    jobs += [('feedback', tl, None, wd) for tl in (25.0, 50.0)
             for wd in T_WD]
    for var, tl, em, wd in jobs:
        R = sb.run(B, var, scenario='loss', t_loss=tl, emergency_unit=em,
                   T_wd=wd, T_end=T_end)
        r = _loss_row(R)
        rows.append(r)
        runs.append(slim(R))
        log('  %-8s t_loss=%4.0f em=%-4s T_wd=%-4s buffer lost=%s (%s) '
            'contact=%s gmin=%.3f (pair %d, t=%.2f) stopped=%s (%d) '
            'completed=%s disp=%s marker=%s gap=%s filt=%.2f infeas=%.2f'
            % (var, tl, em, wd, r['buffer_lost'],
               fmt(r['t_buffer_lost'], '%.3f', '-'),
               fmt(r['t_contact'], '%.3f', '-'), r['gmin'], r['gmin_pair'],
               r['t_gmin'], r['all_stopped'], r['n_stopped'],
               r['arrival_completed'], fmt(r['disp']),
               fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'), r['filt'],
               r['infeas']))
    # state of the nominal run at the loss instants (speeds, clearances)
    Rn = sb.run(B, 'proposed', record=True, rec_dt=1.0, T_end=T_end)
    st = []
    for tl in T_LOSS:
        j = int(round(tl))
        s = Rn['rec']['s'][:, j]
        v = Rn['rec']['v'][:, j]
        st.append(dict(t=float(Rn['rec']['t'][j]),
                       v=[float(x) for x in v],
                       clearance=[float(s[i - 1] - s[i] - B['ell'])
                                  for i in range(1, B['n'])]))
    out.update(rows=rows, runs=runs, nominal_state_at_loss=st,
               t_loss=list(T_LOSS), T_wd=list(T_WD), emergency_unit=3)
    finish('loss', out, D, log, t0)


# ================================================================== clock
def _clock_row(R, pattern, D_ms, cons, seed=None):
    al = cons['align']
    ge = cons['gaperr']
    n = len(R['marker_signed'])
    inside = bool(
        R['all_stopped'] and R['disp'] <= cons['settling_bound']
        and R['T_f'] <= cons['T_f_bar']
        and all(abs(R['marker_signed'][i])
                <= (al[i] if isinstance(al, list) else al)
                for i in range(n))
        and all(abs(R['gap_signed'][i])
                <= (ge[i] if isinstance(ge, list) else ge)
                for i in range(n - 1)))
    return dict(
        bound_ms=D_ms, pattern=pattern, seed=seed,
        offsets_ms=[int(round(1e3 * x)) for x in R['clock_offsets']],
        disp=R['disp'], marker=R['marker_abs_max'], gap=R['gap_abs_max'],
        marker_signed=R['marker_signed'], gap_signed=R['gap_signed'],
        tstop=R['tstop'], T_f=R['T_f'], stop_order=R['stop_order'],
        gmin=R['gmin_min'], gmin_pair=R['gmin_pair'], hmin=R['hmin_min'],
        mmin=min(R['mmin']),
        filt=max(R['t_barrier']), infeas=float(sum(R['t_infeas'])),
        clip0=max(R['t_clip0']), clip0_sum=float(sum(R['t_clip0'])),
        nom_pos=max(R['t_nom_pos']), clipk=max(R['t_clipk']),
        at0=max(R['t_at0']),
        u_min=R['u_min'], u_max=R['u_max'],
        rule_units=R['rule_units'], tset=R['tset'],
        all_stopped=R['all_stopped'], contact=R['contact_time'],
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], pass_all=R['pass_all'],
        inside_certified_bounds=inside)


def part_clock(args):
    log = Log('clock')
    t0 = time.time()
    log('=== part clock ===')
    D = load_design(log)
    B = sb.setup(D['cfg'])
    cons = D['certificate']['consts']
    out = header('clock', D, B, t0)
    n = B['n']
    R0 = sb.run(B, 'proposed')
    out['nominal'] = _clock_row(dict(R0, clock_offsets=[0.0] * n), 'none',
                                0, cons)
    rows = []
    summary = []
    for D_ms in CLOCK_MS:
        d = D_ms * 1e-3
        pats = [('alternating', [d * (1 if i % 2 == 0 else -1)
                                 for i in range(n)], None),
                ('head late', [-d] + [0.0] * (n - 1), None),
                ('tail late', [0.0] * (n - 1) + [-d], None)]
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            off = rng.integers(-D_ms, D_ms + 1, size=n)
            pats.append(('random', [1e-3 * int(x) for x in off], sd))
        rr = []
        for pat, off, sd in pats:
            R = sb.run(B, 'proposed', clock_offsets=off)
            r = _clock_row(R, pat, D_ms, cons, sd)
            rows.append(r)
            rr.append(r)
            if pat != 'random':
                log('  %3d ms %-12s disp=%.4f marker=%.4f gap=%.4f '
                    'gmin=%.3f mmin=%.3f filt=%.3f clip0=%.3f clipk=%.3f '
                    'rule=%s pass=%s inside bounds=%s'
                    % (D_ms, pat, r['disp'], r['marker'], r['gap'],
                       r['gmin'], r['mmin'], r['filt'], r['clip0'],
                       r['clipk'], r['rule_units'], r['pass_all'],
                       r['inside_certified_bounds']))
        rnd = [r for r in rr if r['pattern'] == 'random']
        sm = dict(
            bound_ms=D_ms, n_runs=len(rr), n_random=len(rnd),
            random=dict(
                disp_max=max(r['disp'] for r in rnd),
                disp_med=float(np.median([r['disp'] for r in rnd])),
                marker_max=max(r['marker'] for r in rnd),
                gap_max=max(r['gap'] for r in rnd),
                gmin_min=min(r['gmin'] for r in rnd),
                mmin_min=min(r['mmin'] for r in rnd),
                filt_max=max(r['filt'] for r in rnd),
                clip0_max=max(r['clip0'] for r in rnd),
                clipk_max=max(r['clipk'] for r in rnd),
                rule_runs=sum(1 for r in rnd if r['rule_units']),
                pass_all=sum(1 for r in rnd if r['pass_all']),
                inside_certified_bounds=sum(
                    1 for r in rnd if r['inside_certified_bounds'])),
            all_runs=dict(
                disp_max=max(r['disp'] for r in rr),
                marker_max=max(r['marker'] for r in rr),
                gap_max=max(r['gap'] for r in rr),
                gmin_min=min(r['gmin'] for r in rr),
                mmin_min=min(r['mmin'] for r in rr),
                filt_max=max(r['filt'] for r in rr),
                infeas_max=max(r['infeas'] for r in rr),
                clip0_max=max(r['clip0'] for r in rr),
                clipk_max=max(r['clipk'] for r in rr),
                u_max=max(r['u_max'] for r in rr),
                u_min=min(r['u_min'] for r in rr),
                all_stopped=sum(1 for r in rr if r['all_stopped']),
                contacts=sum(1 for r in rr if r['contact'] is not None),
                pass_all=sum(1 for r in rr if r['pass_all']),
                inside_certified_bounds=sum(
                    1 for r in rr if r['inside_certified_bounds'])))
        summary.append(sm)
        log('  %3d ms random (20): disp max %.4f med %.4f marker max %.4f '
            'gap max %.4f gmin min %.3f filt max %.3f clip0 max %.3f pass '
            '%d/20 inside bounds %d/20'
            % (D_ms, sm['random']['disp_max'], sm['random']['disp_med'],
               sm['random']['marker_max'], sm['random']['gap_max'],
               sm['random']['gmin_min'], sm['random']['filt_max'],
               sm['random']['clip0_max'], sm['random']['pass_all'],
               sm['random']['inside_certified_bounds']))
    out.update(rows=rows, summary=summary, bounds_ms=list(CLOCK_MS),
               seeds=list(SEEDS),
               convention='local time of unit i = t + Delta_i; a late '
                          'clock has Delta_i < 0 (the unit processes its '
                          'planned jumps and its braking late)')
    finish('clock', out, D, log, t0)


# ==================================================================== mpc
_WB = {}


def _mpc_job(job):
    """One MPC run in a worker process (the plan is rebuilt from cfg)."""
    key, cfg, kw = job
    h = cfg_hash(cfg)
    if h not in _WB:
        _WB[h] = sb.setup(cfg)
    B = _WB[h]
    R = sb.run(B, 'mpc', T_end=B['T_xi'] + T_END_MPC, **kw)
    return key, slim(R)


def _pool_map(jobs, workers, log, label):
    res = {}
    t0 = time.time()
    if workers <= 1:
        for j, job in enumerate(jobs):
            k, R = _mpc_job(job)
            res[k] = R
            if (j + 1) % 10 == 0:
                log('    %s %d/%d (%.0f s)'
                    % (label, j + 1, len(jobs), time.time() - t0))
        return res
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_mpc_job, job) for job in jobs]
        for j, f in enumerate(as_completed(futs)):
            k, R = f.result()
            res[k] = R
            if (j + 1) % 20 == 0:
                log('    %s %d/%d (%.0f s)'
                    % (label, j + 1, len(jobs), time.time() - t0))
    return res


def _mrow(R):
    return dict(
        disp=R['disp'], marker=R['marker_abs_max'], gap=R['gap_abs_max'],
        marker_signed=R['marker_signed'], gap_signed=R['gap_signed'],
        tstop=R['tstop'], T_f=R['T_f'], stop_order=R['stop_order'],
        gmin=R['gmin_min'], gmin_pair=R['gmin_pair'], hmin=R['hmin_min'],
        u_min=R['u_min'], u_max=R['u_max'], u_abs_max=R['u_abs_max'],
        filt=max(R['t_barrier']), filt_raw=max(R['t_barrier_raw']),
        infeas=float(sum(R['t_infeas'])), clip0=max(R['t_clip0']),
        nom_pos=max(R['t_nom_pos']), clipk=max(R['t_clipk']),
        at0=max(R['t_at0']), at0_units=R['t_at0'],
        contact_time=R['contact_time'], all_stopped=R['all_stopped'],
        buffer_lost=R['buffer_lost'],
        marker_abs_max=R['marker_abs_max'], gap_abs_max=R['gap_abs_max'],
        pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
        pass_gap=R['pass_gap'], joint=joint(R),
        n_solver_not_converged=R['n_solver_not_converged'],
        runtime_solve_ms_mean=R['runtime_solve_ms_mean'],
        runtime_solve_ms_max=R['runtime_solve_ms_max'],
        runtime_s=R['runtime_s'])


def part_mpc(args):
    log = Log('mpc')
    t0 = time.time()
    log('=== part mpc ===')
    D = load_design(log)
    cfg = D['cfg']
    B = sb.setup(cfg)
    out = header('mpc', D, B, t0)
    out['constants_verified'] = _verify_constants(log)
    workers = args.get('workers', 8)
    v0h = sb.doubled_mismatch(B)
    jobs = []
    for dv in MPC_DV:
        pln = dict(tau_c=sb.TAU_C, dv_max=dv)
        for iw, W in enumerate(MPC_GRID):
            jobs.append((('ideal', dv, iw), cfg,
                         dict(W=W, mpc_planner=pln, plant='ideal', v0=v0h)))
            jobs.append((('hifi', dv, iw), cfg,
                         dict(W=W, mpc_planner=pln, plant='hifi', seed=1)))
    log('  training: %d runs on %d workers' % (len(jobs), workers))
    res = _pool_map(jobs, workers, log, 'training')
    cands = []
    for dv in MPC_DV:
        for iw, W in enumerate(MPC_GRID):
            a = res[('ideal', dv, iw)]
            h1 = res[('hifi', dv, iw)]
            sa, sh1 = score(a), score(h1)
            cands.append(dict(W=W, planner=dict(tau_c=sb.TAU_C, dv_max=dv),
                              score=[sa[0] + sh1[0], sa[1] + sh1[1]],
                              score_ideal=sa, score_hifi=sh1,
                              train_ideal=_mrow(a), train_hifi=_mrow(h1)))
            log('  mpc dv=%.1f %s -> score [%g, %.4f] | ideal disp=%s '
                'marker=%.2f gap=%.2f filt=%.1f | hifi disp=%s marker=%.2f '
                'gap=%.2f contact=%s'
                % (dv, W, cands[-1]['score'][0], cands[-1]['score'][1],
                   fmt(a['disp'], '%.3f'), a['marker_abs_max'],
                   a['gap_abs_max'], max(a['t_barrier']),
                   fmt(h1['disp'], '%.3f'), h1['marker_abs_max'],
                   h1['gap_abs_max'], h1['contact_time']))
    protocols = {}
    for tag, dvs in (('literal', (MPC_DV[0],)), ('extended', MPC_DV)):
        cs = [c for c in cands if c['planner']['dv_max'] in dvs]
        best = min(cs, key=lambda c: tuple(c['score']))
        W = best['W']
        pln = best['planner']
        hj = [(('held', 'ideal', 'detector'), cfg,
               dict(W=W, mpc_planner=pln, plant='ideal',
                    mpc_stop='detector')),
              (('held', 'ideal', 'latch'), cfg,
               dict(W=W, mpc_planner=pln, plant='ideal', mpc_stop='latch'))]
        for sd in range(2, 22):
            hj.append((('held', 'hifi', sd), cfg,
                       dict(W=W, mpc_planner=pln, plant='hifi', seed=sd)))
        hr = _pool_map(hj, workers, log, 'held-out ' + tag)
        ideal = hr[('held', 'ideal', 'detector')]
        latch = hr[('held', 'ideal', 'latch')]
        hifi = [dict(_mrow(hr[('held', 'hifi', sd)]), seed=sd)
                for sd in range(2, 22)]
        dd = [h['disp'] for h in hifi if h['disp'] is not None]
        n_free = sum(1 for c in cs if c['score'][0] < 1e9)
        n_clean = sum(1 for c in cs if c['score'][0] == 0)
        protocols[tag] = dict(
            dv_max_values=list(dvs), n_candidates=len(cs),
            n_contact_free_complete=n_free, n_without_violation=n_clean,
            best=dict(W=W, planner=pln, score=best['score'],
                      score_ideal=best['score_ideal'],
                      score_hifi=best['score_hifi'],
                      train_ideal=best['train_ideal'],
                      train_hifi=best['train_hifi']),
            heldout_ideal=_mrow(ideal), heldout_ideal_run=ideal,
            heldout_ideal_latch=_mrow(latch), heldout_ideal_latch_run=latch,
            heldout_hifi=hifi,
            hifi_joint_pass=sum(1 for h in hifi if h['joint']),
            hifi_stopped=sum(1 for h in hifi if h['all_stopped']),
            hifi_contacts=sum(1 for h in hifi
                              if h['contact_time'] is not None),
            hifi_pass_disp=sum(1 for h in hifi if h['pass_disp']),
            hifi_pass_marker=sum(1 for h in hifi if h['pass_marker']),
            hifi_pass_gap=sum(1 for h in hifi if h['pass_gap']),
            hifi_disp_med=float(np.median(dd)) if dd else None,
            hifi_disp_max=max(dd) if dd else None,
            hifi_marker_max=max(h['marker'] for h in hifi),
            hifi_gap_max=max(h['gap'] for h in hifi),
            hifi_gmin_min=min(h['gmin'] for h in hifi),
            hifi_T_f_max=max([h['T_f'] for h in hifi
                              if h['T_f'] is not None] or [None]))
        P = protocols[tag]
        log('  %s grid: best W %s planner %s score %s (%d candidates, %d '
            'complete without contact, %d without violation)'
            % (tag, W, pln, best['score'], len(cs), n_free, n_clean))
        log('    held-out ideal (stop detector): disp=%s marker=%.3f '
            'gap=%.3f gmin=%.3f filt=%.2f T_f=%s solve %.2f ms'
            % (fmt(ideal['disp']), ideal['marker_abs_max'],
               ideal['gap_abs_max'], ideal['gmin_min'],
               max(ideal['t_barrier']), fmt(ideal['T_f'], '%.3f'),
               ideal['runtime_solve_ms_mean']))
        log('    held-out ideal (zero-speed latch only): stopped=%s (%d of '
            '5) disp=%s' % (latch['all_stopped'], latch['n_stopped'],
                            fmt(latch['disp'])))
        log('    held-out perturbed plant, seeds 2-21: joint %d/20, stopped '
            '%d, contacts %d, disp med/max %s/%s, marker max %.3f, gap max '
            '%.3f, gmin min %.3f, T_f max %s'
            % (P['hifi_joint_pass'], P['hifi_stopped'], P['hifi_contacts'],
               fmt(P['hifi_disp_med'], '%.3f'),
               fmt(P['hifi_disp_max'], '%.3f'), P['hifi_marker_max'],
               P['hifi_gap_max'], P['hifi_gmin_min'],
               fmt(P['hifi_T_f_max'], '%.3f')))
    # ---- the proposed law on the held-out seeds (default perturbed plant)
    ps = []
    for sd in range(2, 22):
        R = sb.run(B, 'proposed', plant='hifi', seed=sd)
        ps.append(dict(seed=sd, disp=R['disp'], marker=R['marker_abs_max'],
                       gap=R['gap_abs_max'], gmin=R['gmin_min'],
                       T_f=R['T_f'], all_stopped=R['all_stopped'],
                       contact_time=R['contact_time'],
                       pass_disp=R['pass_disp'],
                       pass_marker=R['pass_marker'],
                       pass_gap=R['pass_gap'], joint=joint(R)))
    out['proposed_on_heldout_seeds'] = dict(
        plant='default perturbed plant, no compensation', seeds=ps,
        joint_pass=sum(1 for r in ps if r['joint']),
        pass_disp=sum(1 for r in ps if r['pass_disp']),
        pass_marker=sum(1 for r in ps if r['pass_marker']),
        pass_gap=sum(1 for r in ps if r['pass_gap']),
        disp_med=float(np.median([r['disp'] for r in ps])),
        disp_max=max(r['disp'] for r in ps),
        marker_max=max(r['marker'] for r in ps),
        gap_max=max(r['gap'] for r in ps),
        gmin_min=min(r['gmin'] for r in ps),
        T_f_max=max(r['T_f'] for r in ps))
    log('  proposed law on the held-out seeds 2-21 (default perturbed '
        'plant): joint %d/20, disp max %.3f, marker max %.3f, gap max '
        '%.3f, T_f max %.3f'
        % (out['proposed_on_heldout_seeds']['joint_pass'],
           out['proposed_on_heldout_seeds']['disp_max'],
           out['proposed_on_heldout_seeds']['marker_max'],
           out['proposed_on_heldout_seeds']['gap_max'],
           out['proposed_on_heldout_seeds']['T_f_max']))
    out.update(
        protocol=dict(
            T_mpc=sb.T_MPC, N=sb.N_H, sig=dict(sb.MPC_SIG),
            w_u=sb.MPC_DEFAULT_W['w_u'], grid_size=len(MPC_GRID),
            dv_max_values=list(MPC_DV), tau_c=sb.TAU_C,
            input_bounds=[-B['k'], 0.0],
            stop_detector=dict(ideal=sb.V_STOP_MPC,
                               hifi=sb.HIFI_DEFAULT['v_stop']),
            T_end=B['T_xi'] + T_END_MPC,
            training=['ideal plant, doubled-mismatch takeover',
                      'perturbed plant default, seed 1, nominal takeover'],
            heldout=['ideal plant, nominal takeover',
                     'perturbed plant default, seeds 2-21'],
            objective=score.__doc__,
            candidate_score='sum of the two training scores, component '
                            'by component; best = lexicographic minimum, '
                            'first in grid order on ties'),
        v0_training=v0h, grid=MPC_GRID, candidates=cands,
        protocols=protocols)
    finish('mpc', out, D, log, t0)


# ================================================================= fbtune
# common gains of the retuning of 'feedback' and 'cacc': the scan of the
# design rule (0.05, 0.06, ..., 0.20 1/s) and four larger values
FB_LAMBDAS = tuple(round(0.05 + 0.01 * j, 2) for j in range(16)) \
    + (0.25, 0.30, 0.40, 0.50)
FB_CONTROLLERS = (('feedback', 'feedback', None),
                  ('cacc h_w=0.2', 'cacc', 0.2),
                  ('cacc h_w=0.5', 'cacc', 0.5),
                  ('cacc h_w=1.0', 'cacc', 1.0))
FB_PLANT_ROWS = (('default', {}),
                 ('grade+resistance ff, no lead',
                  dict(grade_ff=1.0, res_ff=1.0)))


def with_gain(B, lam):
    """The simulator inputs B with the common gain lam of the critically
    damped family (beta = lam^2, gamma = 2 lam) for every unit, the head
    included.  The design gain returns the floats of B itself."""
    lam = float(lam)
    if lam == B['lam']:
        return dict(B)
    return dict(B, lam=lam, beta=lam * lam, gamma=2.0 * lam)


def _same(a, b, skip=('runtime_s',)):
    """Equality of two records after the conversion of dump(), without the
    run-time fields."""
    def cl(o):
        if isinstance(o, dict):
            return {k: cl(v) for k, v in o.items() if k not in skip}
        if isinstance(o, list):
            return [cl(x) for x in o]
        return o
    a = json.loads(json.dumps(_py(a)))
    b = json.loads(json.dumps(_py(b)))
    return cl(a) == cl(b)


def _plant_row(B, variant, name, ov, **kw):
    """One row of the perturbed plant (seeds SEEDS) with the fields of
    part_plant (the code of part_plant, for any variant)."""
    rs = []
    for sd in SEEDS:
        R = sb.run(B, variant, plant='hifi', hifi=ov, seed=sd, **kw)
        rec = dict(
            seed=sd, disp=R['disp'], marker=R['marker_abs_max'],
            gap=R['gap_abs_max'], marker_signed=R['marker_signed'],
            gap_signed=R['gap_signed'], tstop=R['tstop'], T_f=R['T_f'],
            stop_order=R['stop_order'],
            gmin=R['gmin_min'], gmin_pair=R['gmin_pair'],
            gmin_pairs=R['gmin'], hmin=R['hmin_min'],
            filt=max(R['t_barrier']), infeas=float(sum(R['t_infeas'])),
            clip0=max(R['t_clip0']), clip0_units=R['t_clip0'],
            nom_pos=max(R['t_nom_pos']), clipk=max(R['t_clipk']),
            at0=max(R['t_at0']),
            u_min=R['u_min'], u_max=R['u_max'],
            u_realized_abs_max=max(R['u_realized_abs_max']),
            jerk_pk=max(R['jerk_pk']),
            stopped=R['all_stopped'], contact=R['contact_time'],
            buffer_lost=R['buffer_lost'], tset=R['tset'],
            rule_units=R['rule_units'],
            pass_disp=R['pass_disp'], pass_marker=R['pass_marker'],
            pass_gap=R['pass_gap'], runtime_s=R['runtime_s'])
        rec['joint'] = joint(R)
        rs.append(rec)
    dd = [r['disp'] for r in rs if r['disp'] is not None]
    return dict(
        name=name, overrides=ov, n=len(rs),
        disp_med=float(np.median(dd)) if dd else None,
        disp_min=min(dd) if dd else None,
        disp_max=max(dd) if dd else None,
        marker_max=max(r['marker'] for r in rs),
        marker_med=float(np.median([r['marker'] for r in rs])),
        gap_max=max(r['gap'] for r in rs),
        gap_med=float(np.median([r['gap'] for r in rs])),
        gmin_min=min(r['gmin'] for r in rs),
        filt_runs=sum(1 for r in rs if r['filt'] > 0),
        filt_max=max(r['filt'] for r in rs),
        infeas_runs=sum(1 for r in rs if r['infeas'] > 0),
        clip0_runs=sum(1 for r in rs if r['clip0'] > 0),
        clip0_med=float(np.median([r['clip0'] for r in rs])),
        clip0_max=max(r['clip0'] for r in rs),
        T_f_max=max([r['T_f'] for r in rs if r['T_f'] is not None]
                    or [None]),
        contacts=sum(1 for r in rs if r['contact'] is not None),
        buffer_lost_runs=sum(1 for r in rs if r['buffer_lost']),
        stopped=sum(1 for r in rs if r['stopped']),
        pass_disp=sum(1 for r in rs if r['pass_disp']),
        pass_marker=sum(1 for r in rs if r['pass_marker']),
        pass_gap=sum(1 for r in rs if r['pass_gap']),
        joint_pass=sum(1 for r in rs if r['joint']),
        seeds=rs)


def part_fbtune(args):
    log = Log('fbtune')
    t0 = time.time()
    log('=== part fbtune ===')
    D = load_design(log)
    B = sb.setup(D['cfg'])
    out = header('fbtune', D, B, t0)
    v0h = sb.doubled_mismatch(B)
    T_end = B['T_xi'] + 60.0
    lam0 = float(B['lam'])
    assert lam0 in FB_LAMBDAS
    checks = {}

    # ---- retuning of the four controllers over the common gain
    ctrl = {}
    for name, var, hw in FB_CONTROLLERS:
        kw = {} if hw is None else dict(h_w=hw)
        cands = []
        for lam in FB_LAMBDAS:
            Bg = with_gain(B, lam)
            S = sb.run(Bg, var, v0=v0h, T_end=T_end, **kw)
            H = sb.run(Bg, var, T_end=T_end, **kw)
            cands.append(dict(
                lam=lam, beta=Bg['beta'], gamma=Bg['gamma'],
                score=score(S),
                train=row_of(name + ' train', S, h_w=S['h_w'], lam=lam),
                heldout=row_of(name + ' held-out', H, h_w=H['h_w'],
                               lam=lam),
                train_run=slim(S), heldout_run=slim(H)))
            c = cands[-1]
            log('  %-13s lam=%.2f score %-28s | train disp=%s marker=%.3f '
                'gap=%.3f at0=%.2f contact=%s | held-out disp=%s '
                'marker=%.3f gap=%.3f gmin=%.3f at0=%.2f filt=%.2f '
                'contact=%s pass=%s'
                % (name, lam, c['score'], fmt(S['disp']),
                   S['marker_abs_max'], S['gap_abs_max'], max(S['t_at0']),
                   fmt(S['contact_time'], '%.3f', '-'), fmt(H['disp']),
                   H['marker_abs_max'], H['gap_abs_max'], H['gmin_min'],
                   max(H['t_at0']), max(H['t_barrier']),
                   fmt(H['contact_time'], '%.3f', '-'), H['pass_all']))
        best = min(cands, key=lambda c: tuple(c['score']))
        des = [c for c in cands if c['lam'] == lam0][0]
        ok = [c['lam'] for c in cands if c['score'][0] == 0]
        okh = [c['lam'] for c in cands if c['heldout']['pass_all']]
        ctrl[name] = dict(
            variant=var, h_w=hw, candidates=cands,
            best_lam=best['lam'], best_score=best['score'],
            best_train=best['train'], best_heldout=best['heldout'],
            design_lam=lam0, design_score=des['score'],
            design_train=des['train'], design_heldout=des['heldout'],
            lams_without_violation_train=ok,
            lams_pass_heldout=okh,
            lams_rejected=[c['lam'] for c in cands
                           if c['score'][0] >= 1e9])
        log('  %-13s best lam=%.2f score %s; design lam=%.2f score %s; '
            'training without violation: %s; held-out pass: %s'
            % (name, best['lam'], best['score'], lam0, des['score'], ok,
               okh))
    out['controllers'] = ctrl
    out['protocol'] = dict(
        family='critically damped S2 feedback with one common gain lam '
               'for every unit, the head included: beta = lam^2, gamma = '
               '2 lam; every other constant of the benchmark unchanged '
               '(alpha, detector, filter, input set, channel)',
        grid=list(FB_LAMBDAS),
        training='ideal plant, doubled-mismatch takeover (speed '
                 'increments doubled, head speed unchanged), filter on',
        heldout='ideal plant, nominal takeover, filter on',
        objective=score.__doc__,
        selection='lexicographic minimum of the training score, first in '
                  'grid order on ties')
    out['v0_training'] = v0h

    # ---- check: the rows of base.json with the design gain
    try:
        BJ = json.load(open(os.path.join(OUT, 'base.json')))
        rb = {r['name']: r for r in BJ['rows']}
        for name, var, hw in FB_CONTROLLERS:
            a = dict(ctrl[name]['design_heldout'])
            b = dict(rb[name])
            for k_ in ('name', 'lam', 'gains'):
                a.pop(k_, None)
                b.pop(k_, None)
            checks['base.json row %s' % name] = _same(a, b)
    except Exception as ex:                           # pragma: no cover
        checks['base.json'] = 'not checked (%s)' % ex

    # ---- feedback closure: loss of every link, both gains
    lam1 = ctrl['feedback']['best_lam']
    gains = [('design', lam0), ('retuned', lam1)]
    loss_rows = []
    loss_runs = []
    for tag, lam in gains:
        Bg = with_gain(B, lam)
        for tl in T_LOSS:
            R = sb.run(Bg, 'feedback', scenario='loss', t_loss=tl,
                       emergency_unit=None, T_wd=None, T_end=T_end)
            r = _loss_row(R)
            r.update(gain=tag, lam=lam)
            loss_rows.append(r)
            loss_runs.append(dict(slim(R), gain=tag, lam=lam))
            log('  feedback %-8s lam=%.2f t_loss=%4.0f buffer lost=%s (%s) '
                'contact=%s gmin=%.3f (pair %d) stopped=%s completed=%s '
                'disp=%s marker=%s gap=%s pass=%s filt=%.2f infeas=%.2f'
                % (tag, lam, tl, r['buffer_lost'],
                   fmt(r['t_buffer_lost'], '%.3f', '-'),
                   fmt(r['t_contact'], '%.3f', '-'), r['gmin'],
                   r['gmin_pair'], r['all_stopped'],
                   r['arrival_completed'], fmt(r['disp']),
                   fmt(r['marker'], '%.3f'), fmt(r['gap'], '%.3f'),
                   r['pass_all'], r['filt'], r['infeas']))
    out['loss'] = dict(rows=loss_rows, runs=loss_runs,
                       t_loss=list(T_LOSS), gains=dict(gains),
                       watchdog=None, emergency_unit=None)
    try:
        LJ = json.load(open(os.path.join(OUT, 'loss.json')))
        for ra in LJ['rows']:
            if ra['variant'] == 'feedback' and ra['T_wd'] is None:
                mine = [r for r in loss_rows if r['gain'] == 'design'
                        and r['t_loss'] == ra['t_loss']][0]
                a = {k_: v_ for k_, v_ in mine.items()
                     if k_ not in ('gain', 'lam')}
                checks['loss.json feedback t_loss=%g' % ra['t_loss']] = \
                    _same(a, ra)
    except Exception as ex:                           # pragma: no cover
        checks['loss.json'] = 'not checked (%s)' % ex

    # ---- feedback closure: perturbed plant, both gains
    plant_rows = []
    for tag, lam in gains:
        Bg = with_gain(B, lam)
        for name, ov in FB_PLANT_ROWS:
            row = _plant_row(Bg, 'feedback', name, ov)
            row.update(gain=tag, lam=lam, variant='feedback')
            plant_rows.append(row)
            log('  feedback %-8s lam=%.2f %-30s joint %2d/%d  disp med/max '
                '%s/%s marker %.2f gap %.2f gmin %.2f filt %d clip0 med '
                '%.1f s contacts %d stopped %d (pass d/m/g %d/%d/%d)'
                % (tag, lam, name, row['joint_pass'], row['n'],
                   fmt(row['disp_med'], '%.3f'),
                   fmt(row['disp_max'], '%.3f'), row['marker_max'],
                   row['gap_max'], row['gmin_min'], row['filt_runs'],
                   row['clip0_med'], row['contacts'], row['stopped'],
                   row['pass_disp'], row['pass_marker'], row['pass_gap']))
    out['plant'] = dict(rows=plant_rows, seeds=list(SEEDS),
                        hifi_default=dict(sb.HIFI_DEFAULT),
                        grade_signs=list(sb.GRADE_SIGNS))
    try:
        PJ = json.load(open(os.path.join(OUT, 'plant.json')))
        ref = [r for r in PJ['rows'] if r['name'] == 'default'][0]
        mine = _plant_row(B, 'proposed', 'default', {})
        checks['plant.json row default (proposed)'] = _same(mine, ref)
    except Exception as ex:                           # pragma: no cover
        checks['plant.json'] = 'not checked (%s)' % ex

    # ---- additional: the two scenarios for every gain of the grid, so
    # that the dependence on the gain is on record (summaries only)
    keys_l = ('t_loss', 'buffer_lost', 't_buffer_lost', 'contact',
              't_contact', 'contact_pair', 'gmin', 'gmin_pair',
              'all_stopped', 'arrival_completed', 'disp', 'marker', 'gap',
              'pass_all', 'T_f', 'rule_units', 'filt', 'infeas', 'at0')
    scan_l = []
    scan_p = []
    for lam in FB_LAMBDAS:
        Bg = with_gain(B, lam)
        rl = []
        for tl in T_LOSS:
            R = sb.run(Bg, 'feedback', scenario='loss', t_loss=tl,
                       emergency_unit=None, T_wd=None, T_end=T_end)
            r = _loss_row(R)
            rl.append({k_: r[k_] for k_ in keys_l})
        scan_l.append(dict(
            lam=lam, rows=rl,
            n_contact=sum(1 for r in rl if r['contact']),
            n_buffer_lost=sum(1 for r in rl if r['buffer_lost']),
            n_completed=sum(1 for r in rl if r['arrival_completed']),
            n_pass=sum(1 for r in rl if r['pass_all']),
            gmin_min=min(r['gmin'] for r in rl)))
        rp = {}
        for name, ov in FB_PLANT_ROWS:
            row = _plant_row(Bg, 'feedback', name, ov)
            row.pop('seeds')
            rp[name] = row
        scan_p.append(dict(lam=lam, rows=rp))
        log('  scan lam=%.2f loss: contact %d, buffer lost %d, completed '
            '%d, pass %d of %d, gmin %.3f | plant default: joint %d/%d '
            'contacts %d marker %.2f gap %.2f | compensated: joint %d/%d '
            'contacts %d marker %.2f gap %.2f'
            % (lam, scan_l[-1]['n_contact'], scan_l[-1]['n_buffer_lost'],
               scan_l[-1]['n_completed'], scan_l[-1]['n_pass'],
               len(T_LOSS), scan_l[-1]['gmin_min'],
               rp['default']['joint_pass'], rp['default']['n'],
               rp['default']['contacts'], rp['default']['marker_max'],
               rp['default']['gap_max'],
               rp[FB_PLANT_ROWS[1][0]]['joint_pass'],
               rp[FB_PLANT_ROWS[1][0]]['n'],
               rp[FB_PLANT_ROWS[1][0]]['contacts'],
               rp[FB_PLANT_ROWS[1][0]]['marker_max'],
               rp[FB_PLANT_ROWS[1][0]]['gap_max']))
    out['loss_scan'] = dict(
        note='additional to the specification: loss of every link at the '
             'six instants for every gain of the grid, feedback closure, '
             'no watchdog, no emergency braking (summaries)',
        rows=scan_l)
    out['plant_scan'] = dict(
        note='additional to the specification: the two rows of the '
             'perturbed plant for every gain of the grid, feedback '
             'closure, seeds 1 to 20 (summaries without the records of '
             'the runs)',
        rows=scan_p)
    for tag, lam in gains:
        a = [r for r in scan_l if r['lam'] == lam][0]['rows']
        b = [{k_: r[k_] for k_ in keys_l} for r in loss_rows
             if r['gain'] == tag]
        checks['loss scan equals the loss rows (%s)' % tag] = _same(a, b)
        a = [r for r in scan_p if r['lam'] == lam][0]['rows']
        for r in plant_rows:
            if r['gain'] == tag:
                b = {k_: v_ for k_, v_ in r.items()
                     if k_ not in ('seeds', 'gain', 'lam', 'variant')}
                checks['plant scan equals the plant rows (%s, %s)'
                       % (tag, r['name'])] = _same(a[r['name']], b)

    checks['all'] = all(v is True for v in checks.values())
    out['checks'] = checks
    log('  checks against the archived records: %s' % checks)
    finish('fbtune', out, D, log, t0)
    return checks['all']


PARTS = dict(bridge=part_bridge, base=part_base, plant=part_plant,
             loss=part_loss, clock=part_clock, mpc=part_mpc,
             fbtune=part_fbtune)

if __name__ == '__main__':
    argv = sys.argv[1:]
    opts = {}
    if '--workers' in argv:
        j = argv.index('--workers')
        opts['workers'] = int(argv[j + 1])
        argv = argv[:j] + argv[j + 2:]
    if not argv:
        print(__doc__)
        sys.exit(0)
    if argv == ['all']:
        argv = ['bridge', 'mpc', 'base', 'plant', 'loss', 'clock']
    for a in argv:
        if a not in PARTS:
            raise SystemExit('unknown part %s' % a)
    for a in argv:
        ok = PARTS[a](opts)
        if a == 'bridge' and ok is False:
            raise SystemExit('bridge: acceptance tests failed; the studies '
                             'are not run')
