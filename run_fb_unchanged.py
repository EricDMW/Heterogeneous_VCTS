#!/usr/bin/env python3
"""run_fb_unchanged.py -- zero-age lower bound on the closure time of the
feedback-only laws of the FB design (followers regulate c_i - d_s,i from
takeover by COAST, S1, S2 with the received command as feedforward; the head
tracks a reference) under the input set [-U^-, U] of the FB benchmark, for
the benchmark takeover state (closures c_i(0) - d_s,i = 43.5 m, head at its
slot 6 m).

Port of run_hr_unchanged.py and run_caseD_unchanged.py (both unchanged; the
latter hard-codes the 5 m gaps of the schedule benchmark in its assertions,
so its model is re-implemented here statement for statement with the FB
inputs read from certify_fb.config() and design_fb.json).

Model (nominal, zero age).  With zero information age the forwarded command
is the predecessor's current command, so the telescoping identity reads
    u_i(t) = a_r(t) + S_i(t),   S_i(t) := sum_{k<=i} f_k(t),
with f_k the feedback part of pair k (0 in COAST, alpha sat(eps/phi) in S1,
beta e + gamma eps in S2, beta = lam^2, gamma = 2 lam); the pair errors obey
COAST: eps' = 0; S1: eps' = -alpha sat(eps/phi); S2: e'' + 2 lam e' + lam^2 e
= 0, which do not involve the reference.  Activation of unit i >= 2 at the
S1 detection instant of unit i-1 (zero age), detection sampled every T_c =
1 ms from activation, switch to S2 at the first sample with |eps_i| <=
eps_det.  The pair errors, the feedback sums S_i and the closure instant are
the same for every reference.  The zero-age realization satisfies the
communication contract (age in [0, dbar], FIFO), so every certificate of
these laws must cover it.

Constraint.  With d(t) := -a_r(t), every command lies in [-U^-, U] iff
    max_i S_i(t) - U <= d(t) <= U^- + min_i S_i(t),
the reference is nonaccelerating (d >= 0), and it must still move at the
handoff speed of the design when the closure ends: int_0^{T_cl} d <= v^in -
v^hand (v^in = 10.45 m/s, v^hand = 1 m/s).  T_cl is the persistent entry of
every pair error, the head pair included, into the handoff box (0.05 m,
0.005 m/s).  Reference classes:
  A  constant approach speed: d = 0;
  B  constant deceleration from takeover (the smallest admissible constant);
  C  any nonaccelerating reference (the most favorable one): d(t) =
     max(0, max_i S_i(t) - U), the pointwise smallest admissible
     deceleration; every admissible d is at least d_C pointwise, so its
     speed loss is at least as large.
For each class: the largest common gain lam* for which the constraint can be
met (bisection to a relative width of 1e-6 between the gains of a
geometric scan of 121 gains in [1e-4, 0.5]) and the closure time there;
T_cl_lower (the smallest closure time at the infeasible gain next to lam*)
bounds the closure time of every admissible gain from below when T_cl is
nonincreasing in lam and the feasible set is down-closed (both checked on
the scan).

Evidence class: nominal computation in floating point (closed-form pair
responses on a time grid of 1 ms on [0, 5] s and 400001 points up to
5 + 14/lam).  Output: data/new_d19/fbref/unchanged_law_fb.json, log
logs_ext/fbref/unchanged_fb.log.
Usage: python run_fb_unchanged.py
"""
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict

import numpy as np

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src', 'certification'))
import certify_fb as cf                                      # noqa: E402
import certify_plan as cq                                    # noqa: E402

OUT = os.path.join(cf.DATA_FB, 'unchanged_law_fb.json')
LOGDIR = os.path.join(HERE, 'logs_ext', 'fbref')
LOG = os.path.join(LOGDIR, 'unchanged_fb.log')
REF = os.path.join(cf.DATA_FB, 'reference.json')

DREC = json.load(open(cf.DESIGN_FB, encoding='utf-8'))
cfg = cf.config()
N = int(cfg['n'])
U = float(cfg['U'])
K = float(cfg['Uminus'])
AMAX = [float(x) for x in cfg['amax']]
ALPHA = float(cfg['alpha'])
EPS_DET = float(cfg['eps_det'])
PHI = float(cfg['phi'])
T_C = float(cfg['T_c'])
EPS_E = float(cfg['eps_e'])
EPS_V = float(cfg['eps_v'])
V_XI = float(cq.F(cfg['plan']['v_xi']))                      # v^in of the reference
V_C1_MIN = float(cq.F(DREC['fixed_choices']['v_hand']))      # handoff speed of the design
V0 = [float(cq.F(x)) for x in cfg['v0']]
C0 = [float(x) for x in cfg['c0']]
D_S = [float(x) for x in cfg['d_s']]
LAM_DESIGN = float(DREC['selected']['lam'])
assert N == 5 and K == 1.2 and U == 0.02 and ALPHA == 0.35 and V_XI == 10.45
assert V_C1_MIN == 1.0 and min(AMAX) >= K                     # U_i^- = U^- for every unit
assert C0 == [50.0] * 4 and D_S == [6.0, 6.5, 6.5, 6.5, 6.5]
assert EPS_DET - PHI > ALPHA * T_C          # saturation stays engaged in S1

# takeover pair errors of the FB laws: head at its slot, followers at c_i(0)
E0 = [0.0] + [round(C0[i - 1] - D_S[i], 12) for i in range(1, N)]
EPS0 = [round(V_XI - V0[0], 12)] + [round(V0[i - 1] - V0[i], 12) for i in range(1, N)]
assert E0 == [0.0, 43.5, 43.5, 43.5, 43.5]
assert [float(cq.F(x)) for x in cfg['e0']] == E0              # the evaluator's e0
DV_MAX = V_XI - V_C1_MIN                      # admissible speed loss of the reference
BOX = (EPS_E, EPS_V)


def log(msg=''):
    print(msg, flush=True)
    os.makedirs(LOGDIR, exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def sha256(path):
    with open(path, 'rb') as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def switches():
    """Zero-age activation and S1 detection instants and switch states
    (statement-for-statement copy of run_caseD_unchanged.switches)."""
    out = []
    t_act = 0.0
    for i in range(N):
        e_a = E0[i] + EPS0[i] * t_act                  # COAST: eps constant
        x_a = EPS0[i]
        if abs(x_a) <= EPS_DET:
            m = 0
        else:
            # S1: eps moves toward 0 at rate alpha (saturation engaged)
            m = math.ceil((abs(x_a) - EPS_DET) / (ALPHA * T_C) - 1e-9)
            while abs(abs(x_a) - ALPHA * m * T_C) > EPS_DET + 1e-15:
                m += 1
        s = -1.0 if x_a < 0 else 1.0
        tau = m * T_C
        x_sw = x_a - s * ALPHA * tau
        e_sw = e_a + x_a * tau - s * ALPHA * tau * tau / 2.0
        assert abs(x_sw) <= EPS_DET + 1e-12 and (m == 0 or abs(x_sw) >= PHI)
        out.append(dict(t_act=t_act, t_set=t_act + tau, e_act=e_a, eps_act=x_a,
                        e_sw=e_sw, eps_sw=x_sw, s1_sign=s, s1_samples=m))
        t_act = t_act + tau                            # flag at zero age
    return out


SW = switches()
T_SW_LAST = SW[-1]['t_set']


def grid(lam):
    """time grid: 1 ms on [0, 5] s (S1 stages), then 400000 points up to
    t_end = 5 + 14/lam (as run_caseD_unchanged)."""
    t_end = 5.0 + 14.0 / lam
    return np.concatenate([np.arange(0.0, 5.0, 1e-3),
                           np.linspace(5.0, t_end, 400001)])


def pair_signals(lam, t):
    """e_k, eps_k and f_k of every pair on the grid t (zero age)."""
    e = np.empty((N, t.size))
    x = np.empty((N, t.size))
    f = np.empty((N, t.size))
    for k, s in enumerate(SW):
        ta, ts = s['t_act'], s['t_set']
        c = t < ta                                     # COAST
        e[k, c] = E0[k] + EPS0[k] * t[c]
        x[k, c] = EPS0[k]
        f[k, c] = 0.0
        c1 = (t >= ta) & (t < ts)                      # S1
        tau = t[c1] - ta
        x[k, c1] = s['eps_act'] - s['s1_sign'] * ALPHA * tau
        e[k, c1] = s['e_act'] + s['eps_act'] * tau - s['s1_sign'] * ALPHA * tau ** 2 / 2
        f[k, c1] = ALPHA * np.clip(x[k, c1] / PHI, -1.0, 1.0)
        c2 = t >= ts                                   # S2, free response
        r = t[c2] - ts
        ex = np.exp(-lam * r)
        es, xs = s['e_sw'], s['eps_sw']
        e[k, c2] = (es * (1 + lam * r) + xs * r) * ex
        x[k, c2] = (xs * (1 - lam * r) - es * lam ** 2 * r) * ex
        f[k, c2] = lam ** 2 * e[k, c2] + 2 * lam * x[k, c2]
    return e, x, f


def evaluate(lam):
    t = grid(lam)
    e, x, f = pair_signals(lam, t)
    S = np.cumsum(f, axis=0)                           # S_i, i = 1..n
    out_box = (np.abs(e) > BOX[0]) | (np.abs(x) > BOX[1])
    anyout = out_box.any(axis=0)
    j = int(np.nonzero(anyout)[0][-1])                 # last grid sample outside
    assert j + 1 < t.size, 'closure beyond the grid'
    T_lo, T_hi = float(t[j]), float(t[j + 1])          # last exit in [T_lo, T_hi]
    w = t <= T_hi
    Smax = S[:, w].max(axis=0)
    Smin = S[:, w].min(axis=0)
    tw = t[w]
    # class A: d = 0
    A_ok = bool(Smax.max() - U <= 0.0 and Smin.min() >= -K)
    # class B: the smallest admissible constant deceleration from takeover
    d_B = float(max(0.0, (Smax - U).max()))
    B_lower_ok = bool(d_B <= K + Smin.min())
    B_loss = d_B * T_hi
    B_ok = bool(B_lower_ok and B_loss <= DV_MAX)
    # class C: pointwise smallest admissible nonaccelerating deceleration
    d_C = np.maximum(Smax - U, 0.0)
    spread_C = float((d_C - Smin).max())               # d(t) - min_i S_i(t) <= U^-
    loss_C = float(np.sum(0.5 * (d_C[1:] + d_C[:-1]) * np.diff(tw)))
    iSp = int(np.argmax(Smax))
    iSm = int(np.argmin(Smin))
    iC = int(np.argmax(d_C - Smin))
    # per-pair entry into the box (last exit of each pair)
    entry_pairs = []
    for k in range(N):
        jj = np.nonzero(out_box[k])[0]
        entry_pairs.append(float(t[jj[-1] + 1]) if jj.size else 0.0)
    return OrderedDict(
        lam=lam, T_cl=T_hi, T_cl_lo=T_lo, entry_pairs=entry_pairs,
        Smax=float(Smax.max()), Smax_at=float(tw[iSp]),
        Smin=float(Smin.min()), Smin_at=float(tw[iSm]),
        A_ok=A_ok,
        B_ok=B_ok, B_lower_ok=B_lower_ok, B_d=d_B, B_loss=B_loss,
        B_speed_at_T=V_XI - B_loss,
        C_ok=bool(spread_C <= K and loss_C <= DV_MAX),
        C_spread=spread_C, C_spread_at=float(tw[iC]), C_loss=loss_C,
        C_d_max=float(d_C.max()), C_speed_at_T=V_XI - loss_C,
        C_binding_if_infeasible=None if (spread_C <= K and loss_C <= DV_MAX) else
        ('lower input bound' if spread_C > K else 'speed loss'))


def bisect(cls, lo, hi, rel=1e-6):
    """largest feasible lam in [lo, hi] (feasible at lo, infeasible at hi)."""
    r_lo, r_hi = evaluate(lo), evaluate(hi)
    assert r_lo[cls + '_ok'] and not r_hi[cls + '_ok']
    while hi - lo > rel * hi:
        mid = math.sqrt(lo * hi)
        r = evaluate(mid)
        if r[cls + '_ok']:
            lo, r_lo = mid, r
        else:
            hi, r_hi = mid, r
    return r_lo, r_hi


def main():
    t0 = time.time()
    lams = [float(x) for x in np.geomspace(1e-4, 0.5, 121)]
    rows = [evaluate(l) for l in lams]
    Tcl = [r['T_cl'] for r in rows]
    mono_T = all(Tcl[j + 1] <= Tcl[j] * (1 + 1e-9) for j in range(len(Tcl) - 1))
    res = OrderedDict()
    for cls in ('A', 'B', 'C'):
        ok = [r[cls + '_ok'] for r in rows]
        feas = [r['lam'] for r in rows if r[cls + '_ok']]
        down_closed = all(ok[j] or not any(ok[j + 1:]) for j in range(len(ok)))
        if not feas:
            res[cls] = dict(feasible_on_scan=False, down_closed=down_closed,
                            lam_scan=[lams[0], lams[-1]])
            continue
        jmax = max(j for j in range(len(ok)) if ok[j])
        if jmax + 1 >= len(rows):
            res[cls] = dict(feasible_on_scan=True, down_closed=down_closed, edge=True,
                            lam_star='>= %g (scan edge)' % lams[-1],
                            T_cl_lower=rows[-1]['T_cl_lo'])
            continue
        r_lo, r_hi = bisect(cls, lams[jmax], lams[jmax + 1])
        res[cls] = OrderedDict(
            feasible_on_scan=True, down_closed=down_closed,
            lam_star=r_lo['lam'], lam_infeasible=r_hi['lam'],
            T_cl_at_lam_star=r_lo['T_cl'],
            T_cl_lower=r_hi['T_cl_lo'],
            at_lam_star=r_lo, at_lam_infeasible=r_hi)
        if cls == 'C':
            res[cls]['binding'] = r_hi['C_binding_if_infeasible']
            res[cls]['speed_at_T_cl'] = r_lo['C_speed_at_T']
        elif cls == 'B':
            res[cls]['binding'] = 'lower input bound' if not r_hi['B_lower_ok'] else 'speed loss'
            res[cls]['speed_at_T_cl'] = r_lo['B_speed_at_T']
        else:
            res[cls]['binding'] = ('upper input bound (traction limit U)'
                                   if r_hi['Smax'] - U > 0.0 else 'lower input bound')
            res[cls]['speed_at_T_cl'] = V_XI
    # the design gain and the gains of the design scan
    scan_gains = [float(x) for x in DREC['gains']]
    at_gains = OrderedDict(('%g' % g, evaluate(g)) for g in scan_gains)
    Rd = at_gains['%g' % LAM_DESIGN]
    sel = DREC['selected']
    cert = sel['certificate_20ms']
    ref = json.load(open(REF, encoding='utf-8')) if os.path.exists(REF) else None
    comp = OrderedDict(
        design_gain=LAM_DESIGN,
        certified_entry_deadline_s=cert['bounds']['entry_deadline'],
        certified_T_hold_s=cert['bounds']['T_hold'],
        design_T_xi_s=sel['T_xi'], certified_T_f_bar_s=cert['bounds']['T_f_bar'],
        observed_entry_reference_run_s=None if ref is None else ref['kpis']['entry_obs'],
        observed_entry_note=None if ref is None else
        'persistent entry of every pair error into the handoff box in the reference run '
        '(sim_hr cont, h = 1e-4 s, constant transport 19 ms; data/new_d19/fbref/reference.json)',
        zero_age_T_cl_at_design_gain_s=Rd['T_cl'],
        zero_age_class_C_feasible_at_design_gain=Rd['C_ok'],
        zero_age_class_C_spread_at_design_gain=Rd['C_spread'],
        zero_age_class_C_speed_loss_at_design_gain=Rd['C_loss'],
        lam_star_C=res['C'].get('lam_star'),
        T_cl_lower_C_s=res['C'].get('T_cl_lower'),
        ratio_certified_entry_to_T_cl_lower=(cert['bounds']['entry_deadline']
                                             / res['C']['T_cl_lower'])
        if isinstance(res['C'].get('T_cl_lower'), float) else None,
        excess_certified_entry_over_T_cl_lower_s=(cert['bounds']['entry_deadline']
                                                  - res['C']['T_cl_lower'])
        if isinstance(res['C'].get('T_cl_lower'), float) else None,
        excess_certified_entry_over_zero_age_T_cl_at_design_gain_s=(
            cert['bounds']['entry_deadline'] - Rd['T_cl']),
        design_scan=DREC['gains'],
        design_scan_certified_at_20ms=[r['lam'] for r in DREC['scan'] if r['admitted']],
        design_scan_failed=OrderedDict(('%g' % r['lam'], r['failed']) for r in DREC['scan']
                                       if not r['admitted']))
    out = OrderedDict(
        study='feedback-only laws of the FB design under the input set [-U^-, U] = [-1.2, 0.02] '
              'm/s^2: zero-age nominal pair dynamics of the FB benchmark takeover state '
              '(closures 43.5 m, head at its slot); largest common gain for which some '
              'nonaccelerating reference keeps every nominal command in the input set and '
              'still moves at the handoff speed 1 m/s when every pair has entered the '
              'handoff box, and the closure time there; comparison with the certified design',
        evidence_class='nominal computation in floating point (closed-form pair responses on a '
                       'time grid of 1 ms on [0, 5] s and 400001 points up to 5 + 14/lam; '
                       'bisection on lam to a relative width of 1e-6)',
        script='code/run_fb_unchanged.py',
        templates='code/run_hr_unchanged.py, code/run_caseD_unchanged.py (not modified; the '
                  'model is re-implemented statement for statement with the FB inputs)',
        code_sha256=OrderedDict(run_fb_unchanged=sha256(os.path.abspath(__file__)),
                                certify_fb=sha256(cf.__file__),
                                design_fb_json=sha256(cf.DESIGN_FB)),
        python=sys.version.split()[0], numpy=np.__version__, platform=platform.platform(),
        inputs=OrderedDict(n=N, U=U, Uminus=K, alpha=ALPHA, eps_det=EPS_DET, phi=PHI, T_c=T_C,
                           eps_e=EPS_E, eps_v=EPS_V, v_in=V_XI, v_hand=V_C1_MIN, v0=V0,
                           c0=C0, d_s=D_S, e0=E0, eps0=EPS0, closure_m=E0[1],
                           max_speed_loss=DV_MAX, gains='beta = lam^2, gamma = 2 lam (common)'),
        switches=SW, last_detection=T_SW_LAST,
        classes=OrderedDict(A='constant approach speed (d = 0)',
                            B='constant deceleration from takeover (smallest admissible)',
                            C='any nonaccelerating reference: pointwise smallest deceleration '
                              'd(t) = max(0, max_i S_i(t) - U)'),
        results=res,
        checks=OrderedDict(T_cl_nonincreasing_in_lam=mono_T,
                           down_closed=OrderedDict((c, res[c]['down_closed']) for c in res)),
        at_design_scan_gains=at_gains,
        comparison_with_certified_design=comp,
        certificate=dict(
            evaluated=False,
            reason='the zero-age realization satisfies the communication contract, so the '
                   'nominal limits bound every certified design of these laws; the '
                   'certificate of the FB design (certify_fb) is evaluated in '
                   'design_fb.json and interval_fb.json'),
        scan=rows,
        runtime_s=time.time() - t0)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    os.replace(tmp, OUT)
    log('==== run_fb_unchanged.py %s ====' % time.strftime('%Y-%m-%d %H:%M:%S'))
    log('inputs: U %g, U^- %g, closures %s m, eps0 %s, box %s, max speed loss %g m/s'
        % (U, K, E0[1:], EPS0, BOX, DV_MAX))
    log('zero-age switch instants %s s' % [round(s['t_set'], 3) for s in SW])
    for cls in ('A', 'B', 'C'):
        r = res[cls]
        if r.get('feasible_on_scan') and not r.get('edge'):
            log('%s: lam* = %.6g (infeasible at %.6g), T_cl(lam*) = %.3f s, T_cl >= %.3f s, '
                'binding %s, speed at T_cl %.4f m/s'
                % (cls, r['lam_star'], r['lam_infeasible'], r['T_cl_at_lam_star'],
                   r['T_cl_lower'], r.get('binding'), r.get('speed_at_T_cl', float('nan'))))
        else:
            log('%s: %s' % (cls, json.dumps(r, default=str)))
    for g, r in at_gains.items():
        log('lam %s: T_cl %.3f s; class C feasible %s (spread %.4f <= %g, loss %.4f <= %g); '
            'class B %s (d %.4f, loss %.3f); class A %s; S max %.4f at %.2f s, S min %.4f'
            % (g, r['T_cl'], r['C_ok'], r['C_spread'], K, r['C_loss'], DV_MAX, r['B_ok'],
               r['B_d'], r['B_loss'], r['A_ok'], r['Smax'], r['Smax_at'], r['Smin']))
    log('comparison: %s' % json.dumps(comp, default=str))
    log('T_cl nonincreasing: %s; down-closed: %s' % (mono_T, out['checks']['down_closed']))
    log('wrote %s (%.1f s)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
