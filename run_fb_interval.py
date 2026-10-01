#!/usr/bin/env python3
r"""run_fb_interval.py -- interval verification of the dispatch certificate
of the arrival with feedback-only followers and a shaped head reference (FB,
2026-09-29): src/certification/interval_fb.py against the floating-point
evaluator src/certification/certify_fb.py.  Adapted from run_hr_interval.py
(which stays unchanged for the schedule version).

Configuration: certify_fb.config() (the selected design of
data/new_d19/fbref/design_fb.json: shaped head reference, lam = 0.065,
V^hnd = 1e-2, lags (41, 41), U = 0.02, U^- = 1.2, handoff box (0.05 m,
0.005 m/s), parking gaps 6.5 m (head slot 6 m), cells of 10 ms, one-sided
switch box with the earliest S2 switch instants of the FB refinement), at the
ages 20 ms (design age bound), the age boundary of the design record (24 ms)
and one millisecond above it (25 ms, first rejected age, binding C7a).  Every
age is the exact rational k/1000 s in the enclosure; the floating-point
evaluator runs at the nearest double.

Parts
  boundary  floating-point age boundary on the 1 ms grid
            (certify_fb.age_boundary_ms, bisection, kmax 120 ms as in the
            design record) and its agreement with design_fb.json
  ages      at each age: the enclosure of every condition (lower and upper
            end of its slack, verified / refuted / undecided), the enclosed
            constants and bounds, the entry layer, the floating-point
            cross-check of interval_fb (every slack and constant against its
            interval, the planned pulse bounds cell by cell, the entry
            deadlines and command tails, the margins in units of the rounding
            estimate) and, for the FB refinement, the float earliest switch
            instants tlow_k against their intervals and the cells on which
            the float and the interval activation masks of the S2 terms differ

Evidence class: interval arithmetic (outward-rounded double intervals, Arb
balls; exact rationals for the design data) enclosing the floating-point
certificate.

Writes data/new_d19/fbref/interval_fb.json and logs_ext/fbref/interval_fb.log.
Modifies no other file.
"""
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr

import numpy as np

CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
sys.path.insert(0, CODE)
sys.dont_write_bytecode = True
import certify_fb as fb                                   # noqa: E402
import interval_fb as ih                                  # noqa: E402

OUT_DIR = os.path.join(CODE, 'data', 'new_d19', 'fbref')
OUT = os.path.join(OUT_DIR, 'interval_fb.json')
LOG_DIR = os.path.join(CODE, 'logs_ext', 'fbref')
LOG = os.path.join(LOG_DIR, 'interval_fb.log')
DESIGN = fb.DESIGN_FB

KMAX = 120                          # search edge of the design record


def design_record():
    return json.load(open(DESIGN, encoding='utf-8'))


def design_ages():
    """20 ms, the recorded age boundary and one millisecond above it."""
    B = design_record()['selected']['age_boundary']['k_ms']
    return (20, int(B), int(B) + 1)


AGES_MS = design_ages()


def log(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(s + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()


def clean(o):
    """json-ready copy: numpy scalars and arrays to Python, non-finite
    floats to strings, private keys (leading underscore) dropped."""
    if isinstance(o, dict):
        return OrderedDict((str(k), clean(v)) for k, v in o.items()
                           if not str(k).startswith('_'))
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        x = float(o)
        if math.isfinite(x):
            return x
        return 'inf' if x > 0 else ('-inf' if x < 0 else 'nan')
    if isinstance(o, Fr):
        return str(o)
    return o


def cfg_at(ms):
    return fb.config(dbar=float(Fr(ms, 1000)))


def fmt(x):
    return '%+.12e' % x if isinstance(x, float) and math.isfinite(x) else str(x)


def part_boundary(rec):
    t0 = time.time()
    log('== floating-point age boundary (certify_fb.age_boundary_ms, 1 ms grid, kmax %d) =='
        % KMAX)
    cfg = fb.config()
    b = fb.age_boundary_ms(cfg, kmax=KMAX)
    D = design_record()['selected']['age_boundary']
    same = bool(b['k_ms'] == D['k_ms'] and b.get('failed_next') == D.get('failed_next'))
    log('   boundary %s ms; failing at +1 ms: %s; coarse check below: %s; equal to the design '
        'record (%s ms, %s): %s' % (b['k_ms'], b.get('failed_next'),
                                    b.get('coarse_below_all_admitted'), D['k_ms'],
                                    D.get('failed_next'), same))
    rec['float_boundary'] = dict(k_ms=b['k_ms'], failed_next=b.get('failed_next'),
                                 slacks_at_k=b.get('slacks_at_k'),
                                 slacks_at_k1=b.get('slacks_at_k1'),
                                 coarse_below_all_admitted=b.get('coarse_below_all_admitted'),
                                 equal_to_design_record=same,
                                 design_record=dict(k_ms=D['k_ms'],
                                                    failed_next=D.get('failed_next')),
                                 note='bisection with dbar = k * 1e-3 (double), kmax %d' % KMAX,
                                 runtime_s=time.time() - t0)


def fb_refinement_check(enc, fres):
    """the FB refinement in both evaluators: float tlow_k against the
    interval tlow_k, and the 10 ms cells on which the activation masks of
    the S2 terms differ (float: cell end >= tlow_k; interval: cell end >=
    lower end of tlow_k)."""
    tl_f = [float(x) for x in fres['consts']['tlow_sw']]
    tl_i = enc['consts']['tlow_sw']
    tg1 = np.asarray(enc['_cells']['tg1'], float)
    rows = []
    for k, (f, (lo, hi)) in enumerate(zip(tl_f, tl_i)):
        mf = tg1 >= f
        mi = tg1 >= lo
        only_i = int(np.sum(mi & ~mf))
        only_f = int(np.sum(mf & ~mi))
        rows.append(OrderedDict(
            unit=k + 1, float=f, interval=[lo, hi], inside=bool(lo <= f <= hi),
            width=hi - lo, cells_kept_only_by_interval=only_i,
            cells_kept_only_by_float=only_f,
            first_kept_cell_end_float=float(tg1[mf][0]) if mf.any() else None,
            first_kept_cell_end_interval=float(tg1[mi][0]) if mi.any() else None))
    return OrderedDict(
        switch_lower=bool(fres['consts'].get('switch_lower')), rows=rows,
        all_inside=all(r['inside'] for r in rows),
        masks_identical=all(r['cells_kept_only_by_interval'] == 0
                            and r['cells_kept_only_by_float'] == 0 for r in rows),
        interval_never_drops_a_float_term=all(r['cells_kept_only_by_float'] == 0
                                              for r in rows),
        note='tlow_k: earliest S2 switch instant of unit k (FB refinement); the S2 terms '
             'of layer k are dropped on the cells whose end lies strictly below it')


def part_age(rec, ms):
    t0 = time.time()
    cfg = cfg_at(ms)
    log('== age %d ms (exact %s s) ==' % (ms, Fr(ms, 1000)))
    enc = ih.enclose(cfg, dbar=Fr(ms, 1000), entry=True, keep=True)
    t1 = time.time()
    sc = ih.selfcheck(cfg, enc)
    log('   self-check (independent of the float evaluator, %.1f s): ok %s; chain vs Arb %s; '
        'window extrema vs Fractions %d tests, %d failures; pulse cells vs Arb %d tests, '
        '%d failures' % (time.time() - t1, sc['ok'], sc['chain_vs_arb']['ok'],
                         sc['window_extrema_vs_fractions']['tests'],
                         sc['window_extrema_vs_fractions']['failures'],
                         sc['pulse_cells_vs_arb']['tests'], sc['pulse_cells_vs_arb']['failures']))
    t2 = time.time()
    fres = fb.evaluate(dict(cfg, dbar=float(Fr(ms, 1000))), keep_curves=True)
    t_float = time.time() - t2
    fc = ih.float_crosscheck(cfg, enc, fres=fres)
    fc['float_runtime_s'] = t_float
    fbr = fb_refinement_check(enc, fres)
    log('   enclosure %.1f s, float evaluator %.1f s; closed %s; modal branches '
        'undecided %d' % (enc['timing']['total_s'], t_float, enc['closed'],
                          enc['consts']['modal_undecided_branches']))
    log('   %-11s %-6s %-21s %-21s %-10s %-21s %-7s %s'
        % ('condition', 'sense', 'slack lo', 'slack hi', 'status', 'float slack', 'inside',
           'excursion (ulp)'))
    for nm, c in enc['conds'].items():
        f = fc['conds'][nm]
        log('   %-11s %-6s %-21s %-21s %-10s %-21s %-7s %s'
            % (nm, c['sense'], fmt(c['lo']), fmt(c['hi']), c['status'],
               fmt(f['float_slack']) if f['float_slack'] is not None else 'n/a',
               f['inside'], '' if not f['excursion'] else '%.3g (%.1f ulp)'
               % (f['excursion'], f['excursion_ulps'])))
    K = enc['consts']
    B = enc['bounds']
    log('   FB refinement: tlow float %s' % [round(r['float'], 12) for r in fbr['rows']])
    log('                  tlow interval %s' % [[round(a, 12) for a in r['interval']]
                                                for r in fbr['rows']])
    log('                  float inside %s; activation masks identical %s; cells kept only '
        'by the interval %s, only by the float %s'
        % (fbr['all_inside'], fbr['masks_identical'],
           [r['cells_kept_only_by_interval'] for r in fbr['rows']],
           [r['cells_kept_only_by_float'] for r in fbr['rows']]))
    log('   pulses: P_e %s' % [[round(a, 12), round(b, 12)] for a, b in K['P_e'][1:]])
    log('           P_eps %s' % [[round(a, 12), round(b, 12)] for a, b in K['P_eps'][1:]])
    log('           P_f %s' % [[round(a, 12), round(b, 12)] for a, b in K['P_f'][1:]])
    log('           A1 %s A2 %s (exact)' % (K['A1_exact'][1:], K['A2_exact'][1:]))
    log('           cells with activation decided exactly: %s'
        % [p['activation_decided_exactly'] for p in K['pulses']])
    log('   bounds: dispersion %s  T_f %s  marker max %s  gap max %s'
        % (B['dispersion'], B['T_f_bar'], B['marker_max'], B['gap_max']))
    log('           clearance floors %s' % B['clearance_floor'])
    log('           u_max before T_xi %s' % B['u_max_before_T_xi'])
    if 'entry' in enc:
        e = enc['entry']
        log('   entry: deadline grid indices (ms, lower/upper) %s; T_ent_bar %s; T_hold %s; Vhnd max %s'
            % ([[r.get('k_low'), r.get('k_star')] for r in e['rows']], e['T_ent_bar'],
               e['T_hold'], e['Vhnd_max']))
    cc = fc['constants']
    log('   float constants checked %d, outside their interval %d (beyond 1e-13 rel: %d, '
        'max %.1f ulp)' % (cc['checked'], cc['outside'], cc['outside_beyond_1e13_rel'],
                           cc['max_excursion_ulps']))
    for o in cc['items_outside']:
        log('      %-28s float %.17g  interval [%.17g, %.17g]  excursion %.3g (%.1f ulp)'
            % (o['name'], o['float'], o['interval'][0], o['interval'][1], o['excursion'],
               o['excursion_ulps']))
    if 'pulse_cells' in fc:
        pc = fc['pulse_cells']
        log('   float pulse bounds inside the cell enclosures: %s (%d pair/bound rows)'
            % (pc['all_inside'], len(pc['rows'])))
    if 'entry' in fc:
        er = fc['entry']
        log('   float entry deadlines %s s, equal to the enclosure: %s'
            % (er['float_entry_deadlines_s'], er['deadlines_equal']))
        vt = er.get('V_tails')
        if vt:
            log('   float entry deadlines recomputed with the exact A^tail: %s s'
                % vt['float_entry_deadlines_with_exact_Atail_s'])
            for tag in ('float_as_evaluated', 'float_with_exact_Atail',
                        'float_with_exact_Atail_and_shifts'):
                log('   V_j tails, %s:' % tag)
                for r in vt[tag]:
                    log('      V%d inside %.4f; above %d pts (%d beyond 1e-12 rel; max rel %.2e '
                        'at %.3f s, median %.1e); below %d pts (%d beyond 1e-12 rel; max rel '
                        '%.2e at %.3f s)'
                        % (r['j'], r['inside_fraction'], r['above_points'],
                           r['above_points_beyond_1e12_rel'], r['max_rel_above'],
                           r['at_above_s'], r['median_rel_above'], r['below_points'],
                           r['below_points_beyond_1e12_rel'], r['max_rel_below'],
                           r['at_below_s']))
            log('   A^tail: float grid points beyond rounding %d (at ties hi_J = T_k: %d; below '
                'the exact tail: %d); float lag shifts exceeding the exact ones (V: %s; '
                'A^tail: %s)'
                % (vt['Atail_float_points_beyond_rounding'],
                   vt['Atail_float_points_beyond_rounding_at_ties'],
                   vt['Atail_float_points_beyond_rounding_below_exact'],
                   vt['float_shift_excess_V'], vt['float_shift_excess_Atail']))
    rm = fc['rounding_margins']
    log('   margins in units of the rounding estimate (N = %d ops, u = 2^-53):' % ih.N_OPS)
    for nm, r in rm.items():
        log('      %-24s slack %.4g  estimate %.3g  ratio %.3g  >= 1e3: %s'
            % (nm, r['slack'], r['rounding_estimate'], r['ratio'], r['at_least_1e3']))
    row = OrderedDict(
        dbar_exact=str(Fr(ms, 1000)), dbar_float=float(Fr(ms, 1000)),
        closed=enc['closed'], verified_all=enc['verified_all'],
        refuted=enc['refuted'], undecided=enc['undecided'],
        conds=OrderedDict((nm, OrderedDict(list(c.items()) + [
            ('float_slack', fc['conds'][nm]['float_slack']),
            ('float_ok', fc['conds'][nm]['float_ok']),
            ('float_inside', fc['conds'][nm]['inside']),
            ('float_excursion', fc['conds'][nm]['excursion']),
            ('float_excursion_ulps', fc['conds'][nm]['excursion_ulps']),
            ('float_inside_within_1e13_rel', fc['conds'][nm]['inside_within_1e13_rel']),
            ('status_agrees_with_float', fc['conds'][nm]['status_agrees_with_float'])]))
            for nm, c in enc['conds'].items()),
        float_nonpos=OrderedDict(
            (k_, fres['conds']['NONPOS'].get(k_)) for k_ in
            ('ok', 'slack', 'before_T_xi', 'before_T_xi_at', 'pre_receipt', 'post_receipt')),
        bounds=enc['bounds'], consts=enc['consts'], entry=enc.get('entry'),
        fb_refinement=fbr,
        float_check=OrderedDict((k, v) for k, v in fc.items() if k != 'conds'),
        selfcheck=sc, timing=enc['timing'], runtime_s=time.time() - t0)
    row['discrepancies'] = discrepancies(enc, fc, fbr)
    for d in row['discrepancies']:
        log('   discrepancy [%s] %s' % (d['item'], d['finding']))
    rec['ages'][str(ms)] = row
    log('   verified all: %s; refuted: %s; undecided: %s (%.1f s)'
        % (enc['verified_all'], enc['refuted'], enc['undecided'], time.time() - t0))
    return enc, fc


def discrepancies(enc, fc, fbr):
    """every difference between the floating-point evaluator and the
    enclosure, with the mechanism established by the cross-check."""
    out = []
    for nm, f in fc['conds'].items():
        if f['inside'] is False:
            c = enc['conds'][nm]
            side = 'above' if f['excursion'] > 0 else 'below'
            if nm == 'C0b':
                why = ('the exact slack is the rational %s; the float evaluator computes '
                       'T_xi - (t_J + 2(n-1) dbar) from rounded times' % c.get('exact'))
            elif nm == 'C5':
                vt = fc.get('entry', {}).get('V_tails', {})
                fs = vt.get('float_with_exact_Atail_and_shifts', [])
                why = ('the float command tails at T_xi - dbar exceed the enclosure '
                       '(conservative side); float lag shifts ceil((B + dbar)/dtg - 1e-12) '
                       'one grid step longer than the exact ones for B = %s s; with the '
                       'exact A^tail and exact shifts the float tails exceed the enclosure by '
                       'at most %.2g relative (ledger guard 1e-15, rounded constants)'
                       % (sorted(vt.get('float_shift_excess_V', {}).keys()) or 'none',
                          max([r['max_rel_above'] for r in fs] or [float('nan')])))
            else:
                why = 'rounding of the float evaluator'
            out.append(dict(item=nm, finding='float slack %.17g %s the interval [%.17g, %.17g] '
                            'by %.3g (%.1f ulp); %s' % (f['float_slack'], side, c['lo'], c['hi'],
                                                        abs(f['excursion']), f['excursion_ulps'],
                                                        why)))
    cc = fc['constants']
    if cc['outside']:
        out.append(dict(item='constants', finding='%d of %d float constants outside their '
                        'interval, max %.1f ulp (beyond 1e-13 relative: %d); items: %s'
                        % (cc['outside'], cc['checked'], cc['max_excursion_ulps'],
                           cc['outside_beyond_1e13_rel'],
                           ', '.join(o['name'] for o in cc['items_outside']))))
    if not fbr['all_inside']:
        out.append(dict(item='tlow_sw', finding='float earliest switch instants outside their '
                        'intervals: %s' % [(r['unit'], r['float'], r['interval'])
                                           for r in fbr['rows'] if not r['inside']]))
    if not fbr['masks_identical']:
        out.append(dict(item='activation masks', finding=(
            'cells on which only the interval evaluator keeps the S2 terms of a layer (cell '
            'end in [tlow.lo, tlow_float)): %s; cells on which only the float evaluator keeps '
            'them: %s' % ([r['cells_kept_only_by_interval'] for r in fbr['rows']],
                          [r['cells_kept_only_by_float'] for r in fbr['rows']]))))
    er = fc.get('entry', {})
    vt = er.get('V_tails')
    if vt:
        fa, fe = vt['float_as_evaluated'], vt['float_with_exact_Atail']
        nb = [r['below_points_beyond_1e12_rel'] for r in fa]
        if any(nb):
            out.append(dict(item='entry V_j tails (below)', finding=(
                'float tails V_1..V_%d below the enclosure by more than 1e-12 relative at %s '
                'grid points (max relative %.3g): the float planned L1 tail differs from the '
                'exact one beyond rounding at %d grid points, %d of them at ties hi_J = T_k '
                '(float below exact at %d): the float compares rounded values and omits the '
                'jump at such a tie, i.e. uses hi_J > T there, which is also a valid bound '
                'because every pulse is supported in [lo_J, hi_J); recomputed with the exact '
                'A^tail, the float tails are below the enclosure beyond 1e-12 relative at %s '
                'points (remaining below points are rounding, max relative %.2g)'
                % (len(fa), nb, max(r['max_rel_below'] for r in fa),
                   vt['Atail_float_points_beyond_rounding'],
                   vt['Atail_float_points_beyond_rounding_at_ties'],
                   vt['Atail_float_points_beyond_rounding_below_exact'],
                   [r['below_points_beyond_1e12_rel'] for r in fe],
                   max(r['max_rel_below'] for r in fe)))))
        na = [r['above_points_beyond_1e12_rel'] for r in fa]
        fs = vt['float_with_exact_Atail_and_shifts']
        if any(na):
            out.append(dict(item='entry V_j tails (above)', finding=(
                'float tails above the enclosure by more than 1e-12 relative at %s grid points '
                '(max relative %.3g, conservative side): the float lag shift '
                'ceil((B + dbar)/dtg - 1e-12) is one grid step longer than the exact '
                'ceil((B + dbar)/(1 ms)) for B = %s s; recomputed with the exact A^tail and '
                'the exact shifts, the float tails are above beyond 1e-12 relative at %s points '
                'and below at %s points (max relative above %.2g, below %.2g: guards 1e-15 and '
                'rounded constants)'
                % (na, max(r['max_rel_above'] for r in fa),
                   sorted(vt['float_shift_excess_V'].keys()) or 'none',
                   [r['above_points_beyond_1e12_rel'] for r in fs],
                   [r['below_points_beyond_1e12_rel'] for r in fs],
                   max(r['max_rel_above'] for r in fs), max(r['max_rel_below'] for r in fs)))))
        if not er.get('deadlines_equal'):
            Tf = fc['float_bounds']['entry_deadline']
            Ti = enc['bounds'].get('entry_deadline')
            same_T = Ti is not None and Ti[0] <= Tf <= Ti[1]
            out.append(dict(item='entry deadlines', finding=(
                'float entry deadlines %s s against the exactly identified grid indices %s ms; '
                'recomputed with the exact A^tail the float deadlines are %s s; float T_ent_bar %.17g %s the '
                'enclosure %s, float C3 slack inside its interval: %s'
                % (er['float_entry_deadlines_s'], er['interval_entry_idx_ms'],
                   vt['float_entry_deadlines_with_exact_Atail_s'], Tf,
                   'inside' if same_T else 'OUTSIDE', Ti, fc['conds']['C3']['inside']))))
    return out


def summary(rec):
    A = rec['ages']
    s = OrderedDict()
    for ms in AGES_MS:
        a = A.get(str(ms))
        if a is None:
            continue
        s['%d_ms' % ms] = OrderedDict(
            verified_all=a['verified_all'], refuted=a['refuted'], undecided=a['undecided'],
            float_admitted=a['float_check']['float_admitted'],
            float_failed=a['float_check']['float_failed'],
            float_slacks_outside=[nm for nm, c in a['conds'].items()
                                  if c['float_inside'] is False],
            float_slacks_outside_beyond_1e13_rel=[
                nm for nm, c in a['conds'].items()
                if c['float_inside_within_1e13_rel'] is False],
            status_disagrees_with_float=[nm for nm, c in a['conds'].items()
                                         if c['status_agrees_with_float'] is False],
            float_constants_outside=a['float_check']['constants']['outside'],
            float_constants_outside_beyond_1e13_rel=a['float_check']['constants'][
                'outside_beyond_1e13_rel'],
            pulse_cells_all_inside=a['float_check'].get('pulse_cells', {}).get('all_inside'),
            entry_deadlines_equal=a['float_check'].get('entry', {}).get('deadlines_equal'),
            selfcheck_ok=a['selfcheck']['ok'],
            tlow_float_inside=a['fb_refinement']['all_inside'],
            activation_masks_identical=a['fb_refinement']['masks_identical'],
            smallest_verified_slacks=OrderedDict(
                sorted(((nm, c['lo']) for nm, c in a['conds'].items()
                        if c['status'] == 'verified' and isinstance(c['lo'], float)),
                       key=lambda z: z[1])[:5]),
            refuted_slack_intervals=OrderedDict((nm, [a['conds'][nm]['lo'], a['conds'][nm]['hi']])
                                                for nm in a['refuted']),
            runtime_s=a['runtime_s'])
    a20 = A.get('20')
    if a20:
        b = a20['bounds']
        s['certified_bounds_20ms'] = OrderedDict(
            dispersion_s=b['dispersion'], T_f_bar_s=b['T_f_bar'],
            marker_max_m=b['marker_max'], gap_max_m=b['gap_max'],
            clearance_floor_m=b['clearance_floor'],
            clearance_floor_min_m=b['clearance_floor_min'],
            entry_deadline_s=dict(interval=b.get('entry_deadline'),
                                  float=a20['float_check']['float_bounds']['entry_deadline']),
            T_hold_s=dict(interval=b.get('T_hold'),
                          float=a20['float_check']['float_bounds']['T_hold']),
            velocity_floor_m_s=b['velocity_floor'],
            u_max_before_T_xi=b['u_max_before_T_xi'],
            u_min_before_T_xi=b['u_min_before_T_xi'])
    if 'float_boundary' in rec:
        k = rec['float_boundary']['k_ms']
        s['age_boundary'] = OrderedDict(
            float_boundary_ms=k, float_failed_next=rec['float_boundary']['failed_next'],
            equal_to_design_record=rec['float_boundary']['equal_to_design_record'],
            interval_verified_at_boundary=A.get(str(k), {}).get('verified_all'),
            interval_refuted_at_boundary_plus_1=A.get(str(k + 1), {}).get('refuted'))
    rec['summary'] = s


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = sys.argv[1:] or ['boundary', 'ages']
    t0 = time.time()
    rec = OrderedDict()
    import flint
    cfg0 = fb.config()
    rec['meta'] = OrderedDict(
        item='interval verification of the FB dispatch certificate (feedback-only followers, '
             'shaped head reference)',
        evidence_class='interval arithmetic (outward-rounded double intervals with one-ulp '
                       'steps, Arb balls, exact rationals of the design data) enclosing the '
                       'floating-point certificate certify_fb.evaluate',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        driver='code/run_fb_interval.py', driver_sha256=sha(os.path.abspath(__file__)),
        template='code/run_hr_interval.py (not modified)',
        enclosure='code/src/certification/interval_fb.py',
        enclosure_sha256=sha(ih.__file__),
        evaluator='code/src/certification/certify_fb.py',
        evaluator_sha256=sha(fb.__file__),
        design_record='code/data/new_d19/fbref/design_fb.json',
        design_record_sha256=sha(DESIGN),
        entry_method='code/run_witness_caseB_entry.py (closed-form helpers imported unchanged)',
        python=sys.version.split()[0], numpy=np.__version__,
        python_flint=flint.__version__, platform=platform.platform(),
        cfg=cfg0, ages_ms=list(AGES_MS),
        design=OrderedDict(lam=cfg0['lam'], U=cfg0['U'], Uminus=cfg0['Uminus'],
                           vhnd=cfg0['vhnd'], lags=cfg0['lags'], d_s=cfg0['d_s'],
                           e0=cfg0['e0'], one_sided=cfg0['one_sided'],
                           switch_lower=cfg0['switch_lower'],
                           T_xi=cfg0['plan']['T_xi'], n_pieces=len(cfg0['plan']['pieces'])),
        method=(
            'exact rationals for the design values and plan data (tightest double '
            'intervals); scalars as double intervals with one-ulp outward rounding '
            '(interval_profile.IV); exponentials by Arb balls (python-flint, 120 bits; '
            'entry layer 200 bits) stepped one ulp outward; cell functions on the '
            'evaluator\'s 10 ms cells as outward-rounded double arrays; the planned '
            'pulse bounds of Lemma 7.1 enclosed on every cell for every jump (no '
            'Lipschitz margin needed); closed forms branch by branch, undecided '
            'branches as the union; the entry layer on the exact grid k/1000 s '
            '(run_witness_caseB_entry.py method); every enclosure two-sided; FB '
            'refinement: the earliest S2 switch instants tlow_k as intervals, the S2 '
            'terms of layer k dropped only on the cells whose end lies below the lower '
            'end of tlow_k'),
        status_rule=('verified: lower end of the slack > 0 (strict tests) or >= 0; '
                     'refuted: upper end <= 0 (strict) or < 0; otherwise undecided'),
        not_enclosed='NONPOS (report-only in certify_fb; its float values are recorded)')
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('\n')
    log('==== run_fb_interval.py %s parts %s ====' % (rec['meta']['generated'], parts))
    log('   design lam %s, T_xi %s, %d reference pieces, ages %s ms'
        % (cfg0['lam'], cfg0['plan']['T_xi'], len(cfg0['plan']['pieces']), list(AGES_MS)))
    rec['ages'] = OrderedDict()
    if 'boundary' in parts:
        part_boundary(rec)
    if 'ages' in parts:
        for ms in AGES_MS:
            part_age(rec, ms)
    summary(rec)
    rec['runtime_s'] = time.time() - t0
    tmp = OUT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(clean(rec), f, indent=1, allow_nan=False)
    os.replace(tmp, OUT)
    log('   summary: %s' % json.dumps(clean(rec['summary'])))
    log('   wrote %s (%.1f s)' % (OUT, time.time() - t0))


if __name__ == '__main__':
    main()
