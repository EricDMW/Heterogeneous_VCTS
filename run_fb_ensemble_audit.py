#!/usr/bin/env python3
r"""run_fb_ensemble_audit.py -- audit of the ensemble runs of run_fb_validate.py
(data/new_d19/fbref/ensemble.json) in which an observation of a certified
(admitted) takeover state lies outside its certified bound.

For every admitted state with a failed bound row the audit records
  * the takeover state and the failed rows (from ensemble.json);
  * the head's speed error at takeover, eps_1(0) = v^in - v_1(0), as the exact
    rational of the certificate and as the double the simulator computes
    (10.45 - v_1(0) in IEEE arithmetic), against the detection threshold
    eps_det: the certificate decides the head's detector sample at t = 0
    with the exact value, the simulator with the double;
  * the certificate of the state as drawn (design rule re-run at the design
    gain, as in the ensemble) and of the states with the head speed nudged
    by +-1e-9 m/s (same reference): admission, the head's lower command
    envelope before T_xi, the switch deadline;
  * the runs of the ensemble (zoh 1 kHz, constant 19 ms and the state's
    random schedule) re-simulated: the head's first commands, its S2 switch
    instant, and every bound row against the certificate of the state as
    drawn and against the certificates of the nudged states;
  * runs of the nudged states against their own certificates.

Evidence class: certificate (floating point, certify_fb) + simulation
(sim_hr, FB law) of the audited states.  Output:
data/new_d19/fbref/ensemble_audit.json, log logs_ext/fbref/ensemble_audit.log.
"""
import json
import os
import sys
import time
from collections import OrderedDict
from decimal import Decimal

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
if CODE not in sys.path:
    sys.path.insert(0, CODE)
import numpy as np                                        # noqa: E402
import run_fb_validate as V                               # noqa: E402

OUT = os.path.join(V.OUT, 'ensemble_audit.json')
LOG = os.path.join(V.LOGDIR, 'ensemble_audit.log')
NUDGE = ('1e-9', '-1e-9')


def log(msg=''):
    print(msg, flush=True)
    os.makedirs(V.LOGDIR, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def cert_of(plan, c0, v0, e0):
    cfg = V.cert_cfg(plan=plan, c0=list(c0), v0=list(v0), e0=list(e0))
    ev = V.cf.evaluate(cfg, keep_curves=True)
    return cfg, ev, V.cert_pack(ev, cfg)


def cert_row(ev, cp):
    return OrderedDict(
        admitted=bool(ev['admitted']), failed=list(ev['failed']),
        eps0_head=float(ev['consts']['eps0'][0]),
        head_lower_envelope_before_T_xi=cp['bounds']['u_min_before_T_xi'][0],
        u_min_before_T_xi=cp['bounds']['u_min_before_T_xi'],
        switch_deadline_head=cp['bounds']['switch_deadline'][0],
        tlow_head=cp['consts']['tlow_sw'][0],
        C7b_slack=cp['conds']['C7b']['slack'], C7a_slack=cp['conds']['C7a']['slack'])


def run_against(S, D, cps):
    """one zoh run of the FB law and its bound rows against every
    certificate of cps (name -> cert pack)."""
    R, T, k, _ = V.run_rows(S, None, 'zoh', D=D)
    out = OrderedDict(
        head_first_commands=[float(x) for x in T['u'][0, :4]],
        head_eps_first=[float(x) for x in T['eps'][0, :3]],
        tset=[float(x) for x in k['tset']], tact=[float(x) for x in k['tact']],
        u_min_pre_unit=[float(x) for x in k['u_min_pre_unit']],
        settling=k['settling'], marker_abs_max=k['marker_abs_max'], gap_abs_max=k['gap_abs_max'],
        filter_modifications=k['filter_modifications'], n_pos_nom=k['n_pos_nom'],
        fb_law_ok=k['fb_law']['ok'], against=OrderedDict())
    for nm, cp in cps.items():
        bt = V.bounds_rows(cp, k, T)
        fr = OrderedDict((r_, dict(bound=bt['rows'][r_].get('bound'),
                                   observed=bt['rows'][r_].get('observed')))
                         for r_ in bt['failed'])
        out['against'][nm] = OrderedDict(all_ok=bt['all_ok'], failed=bt['failed'],
                                         n_checked=bt['n_checked'], n_rows=bt['n_rows'],
                                         failed_rows=fr)
    return out


def _job_rejected(job):
    """certificate of one rejected ensemble state (its reshaped reference):
    the failing conditions with their slack and binding cell."""
    idx, plan, c0, v0, e0, inc = job
    cfg = V.cert_cfg(plan=plan, c0=list(c0), v0=list(v0), e0=list(e0))
    ev = V.cf.evaluate(cfg)
    fails = OrderedDict()
    for nm in ev['failed']:
        c = ev['conds'][nm]
        fails[nm] = OrderedDict(slack=c['slack'], at=c.get('at'),
                                per_pair=c.get('per_pair'), per_unit=c.get('per_unit'))
    K = ev['consts']
    return V._py(OrderedDict(
        index=idx, increments=inc, v1=v0[0], e0=[float(V.cq.F(x)) for x in e0],
        eps0=K['eps0'], Lam1=K['Lam1'], tlow_sw=K['tlow_sw'], tset=K['tset'],
        a_r0=K['a_r0'], failed=ev['failed'], conds=fails,
        u_max_before_T_xi=ev['bounds']['u_max_before_T_xi'],
        u_min_before_T_xi=ev['bounds']['u_min_before_T_xi']))


def rejected_section(E):
    rej = [r for r in E['rows'] if not r['certificate']['admitted']]
    jobs = [(r['index'], r['rule']['plan'], r['c0'], r['v0'], r['e0'], r['increments'])
            for r in rej]
    res = V.pool_map(_job_rejected, jobs, None)
    by = OrderedDict()
    for nm in ('C7a', 'C7b', 'C8b', 'C8c', 'C9a', 'C3'):
        sel = [x for x in res if nm in x['failed']]
        if not sel:
            continue
        at_t = [x['conds'][nm]['at']['t'] for x in sel if x['conds'][nm].get('at')]
        at_u = [x['conds'][nm]['at']['unit'] for x in sel if x['conds'][nm].get('at')]
        by[nm] = OrderedDict(
            n=len(sel), slack=V._rng(x['conds'][nm]['slack'] for x in sel),
            binding_time_s=V._rng(at_t), binding_units=sorted(set(at_u)),
            n_binding_before_0p5s=sum(1 for t in at_t if t < 0.5),
            first_increment=V._rng(x['increments'][0] for x in sel),
            last_increment=V._rng(x['increments'][3] for x in sel),
            head_speed=V._rng(float(x['v1']) for x in sel),
            largest_takeover_error=V._rng(max(x['e0'][1:]) for x in sel),
            tlow_2=V._rng(x['tlow_sw'][1] for x in sel))
    return OrderedDict(n=len(rej), by_condition=by, states=res)


def main():
    t0 = time.time()
    E = V.load('ensemble.json')
    assert E is not None, 'ensemble.json missing'
    eps_det = float(V.cert_cfg()['eps_det'])
    v_in = float(V.cq.F(V.cert_cfg()['plan']['v_xi']))
    flagged = []
    for r in E['rows']:
        if not r['certificate']['admitted']:
            continue
        bad = OrderedDict((rn, x['bounds_failed']) for rn, x in r['runs'].items()
                          if not x['bounds_ok'])
        if bad:
            flagged.append((r, bad))
    # every accepted state on the head's detection threshold
    thr = [dict(index=r['index'], v1=r['v0'][0], admitted=r['certificate']['admitted'])
           for r in E['rows']
           if V.cq.F(r['v0'][0]) - V.cq.F(E['design_plan']['v_xi']) == V.cq.F('0.015')]
    log('==== run_fb_ensemble_audit.py %s: %d admitted states with failed bound rows; states '
        'with |eps_1(0)| = eps_det exactly: %s ====' % (time.strftime('%Y-%m-%d %H:%M:%S'),
                                                        len(flagged), thr))
    states = []
    for r, bad in flagged:
        plan = r['rule']['plan']
        c0, v0, e0 = r['c0'], r['v0'], r['e0']
        eps_exact = V.cq.F(E['design_plan']['v_xi']) - V.cq.F(v0[0])
        eps_double = v_in - float(v0[0])
        cfg, ev, cp = cert_of(plan, c0, v0, e0)
        certs = OrderedDict(as_drawn=cp)
        crows = OrderedDict(as_drawn=cert_row(ev, cp))
        nudged_v0 = OrderedDict()
        for dv in NUDGE:
            v0n = [str(Decimal(v0[0]) + Decimal(dv))] + list(v0[1:])
            nudged_v0[dv] = v0n
            _, evn, cpn = cert_of(plan, c0, v0n, e0)
            certs['v1%+g' % float(dv)] = cpn
            crows['v1%+g' % float(dv)] = cert_row(evn, cpn)
        S = V.setup_of(cfg)
        N = V.n_gen(S)
        Drand = V.rcv.random_schedule(np.random.default_rng(V.SEED_ENS_SCHED + r['index']), N, 19)
        runs = OrderedDict()
        for rn, D in (('constant', None), ('random', Drand)):
            runs[rn] = run_against(S, D, certs)
        nudged_runs = OrderedDict()
        for dv, v0n in nudged_v0.items():
            cfgn = V.cert_cfg(plan=plan, c0=list(c0), v0=list(v0n), e0=list(e0))
            Sn = V.setup_of(cfgn)
            key = 'v1%+g' % float(dv)
            nudged_runs[key] = OrderedDict(
                (rn, run_against(Sn, D, OrderedDict([(key, certs[key])])))
                for rn, D in (('constant', None), ('random', Drand)))
        st = OrderedDict(
            index=r['index'], c0=c0, v0=v0, e0=e0, failed_rows_in_ensemble=bad,
            eps_1_0=OrderedDict(exact=str(eps_exact), double=eps_double,
                                double_repr='%.20g' % eps_double,
                                excess_over_eps_det_double=abs(eps_double) - eps_det,
                                exact_equals_minus_eps_det=bool(eps_exact == -V.cq.F('0.015')),
                                v_in_double='%.20g' % v_in, v1_double='%.20g' % float(v0[0])),
            certificates=crows, runs_as_drawn=runs, runs_nudged=nudged_runs,
            reading=('the certificate evaluates the state with exact rationals: eps_1(0) = '
                     '-eps_det, so the head\'s detector switches at its activation sample t = 0 '
                     'and no S1 braking term enters the head\'s lower envelope; the simulator '
                     'computes eps_1(0) = 10.45 - v_1(0) in double arithmetic, which exceeds '
                     'eps_det in magnitude by the rounding excess, so the head brakes with the S1 '
                     'term -alpha for one detector period (1 ms) before it switches; the '
                     'certificate of the state with v_1(0) nudged up by 1e-9 m/s contains that '
                     'S1 term (head lower envelope a_r0 - alpha and tolerance terms)'))
        states.append(st)
        for nm, c in crows.items():
            log('  state %d %-9s cert admitted %s failed %s, head lower envelope %.6f, switch '
                'deadline %.6f s, C7b slack %.6f' % (r['index'], nm, c['admitted'], c['failed'],
                                                     c['head_lower_envelope_before_T_xi'],
                                                     c['switch_deadline_head'], c['C7b_slack']))
        log('  state %d eps_1(0): exact %s, double %s (|eps| - eps_det = %.3g)'
            % (r['index'], st['eps_1_0']['exact'], st['eps_1_0']['double_repr'],
               st['eps_1_0']['excess_over_eps_det_double']))
        for rn, x in runs.items():
            log('  state %d run %-8s head u %s, head tset %.4f s; rows: %s'
                % (r['index'], rn, ['%.6f' % z for z in x['head_first_commands'][:3]], x['tset'][0],
                   {nm: (a['all_ok'], a['failed']) for nm, a in x['against'].items()}))
        for key, rr in nudged_runs.items():
            for rn, x in rr.items():
                log('  state %d nudged %s run %-8s head u %s, head tset %.4f s; rows vs own '
                    'certificate: %s' % (r['index'], key, rn,
                                         ['%.6f' % z for z in x['head_first_commands'][:3]],
                                         x['tset'][0], {nm: (a['all_ok'], a['failed'])
                                                        for nm, a in x['against'].items()}))
    t1 = time.time()
    rejected = rejected_section(E)
    for nm, g in rejected['by_condition'].items():
        log('  rejected by %-4s n=%3d slack %s binding t %s s units %s (before 0.5 s: %d); '
            'first increment %s, last increment %s, head speed %s, largest e_i(0) %s, tlow_2 %s'
            % (nm, g['n'], g['slack'], g['binding_time_s'], g['binding_units'],
               g['n_binding_before_0p5s'], g['first_increment'], g['last_increment'],
               g['head_speed'], g['largest_takeover_error'], g['tlow_2']))
    rejected['runtime_s'] = time.time() - t1
    out = OrderedDict(
        meta=V.meta('ensemble', script_audit='code/run_fb_ensemble_audit.py',
                    audit_sha256=V.sha256(os.path.abspath(__file__)),
                    evidence_class_audit='certificate (floating point, certify_fb) + simulation '
                                         '(sim_hr, FB law) of the audited states'),
        item='audit of the ensemble (ensemble.json): (1) the admitted states with an '
             'observation outside its certified bound; (2) the failing conditions of the '
             'rejected states with their binding cells (certificate re-evaluated on the '
             'reshaped reference of each state)',
        n_admitted=E['aggregate']['n_admitted'], n_flagged=len(flagged),
        states_on_head_detection_threshold=thr, states=states, rejected=rejected,
        runtime_s=time.time() - t0)
    V.dump('ensemble_audit.json', out)
    log('  wrote data/new_d19/fbref/ensemble_audit.json (%.1f s)' % (time.time() - t0))


if __name__ == '__main__':
    main()
