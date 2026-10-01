#!/usr/bin/env python3
"""run_caseC_scan2.py -- revised Case C certificate program (2026-09-26,
authors' requirements: takeover clearances about 50 m, settling about
1 s, relaxed handoff box, synchronized stop by about 100 s).

Parts (data under data/new_d19/caseC/ only)
  regress    certify_profile.regress (regression.json), refine_check
             (refine_check.json) and interval_profile.regress
             (interval_regression.json)
  scan       certify_profile.scan of default_grid() (scan.json; resumable)
  frontier   frontier.json, frontier.md and scan_summary.json from scan.json:
             per c0 the smallest admitted T_f_bar and its design, for
             T_f_bar <= 100 s the largest admitted c0, the binding
             conditions, and the recommendation
  recommend  recommend.json: the recommended design(s) with the certificate
             at 20 ms, the per-condition age crossings, the 1 ms boundary
             and the interval enclosures
  all        regress frontier recommend (scan is run separately)

Usage: python run_caseC_scan2.py [part ...]
"""
import json
import os
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
import numpy as np                                      # noqa: E402
import certify_profile as cp                            # noqa: E402
import interval_profile as ip                           # noqa: E402

OUT = cp.DATA_C
COND = list(cp.COND_NAMES)
T_F_TARGET = 100.0
T_F_MAX = 105.0
ADM_MIN = 0.2          # required admission slack [m]
AUTH_MIN = 0.02        # required authority slack [m/s^2]
C0_GOAL = 46.0         # smallest clearance that counts as "about 50 m"


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(cp._py(obj), open(os.path.join(OUT, name), 'w'), indent=1)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------- regress
def part_regress():
    rep = cp.regress()
    dump('regression.json', rep)
    rc = cp.refine_check()
    dump('refine_check.json', rc)
    ri = ip.regress(verbose=False)
    dump('interval_regression.json', ri)
    print('regression ok %s; refine check ok %s; interval regression '
          'verified %s inside %s' % (rep['ok'], rc['ok'], ri['verified'],
                                     ri['all_inside']))


# ------------------------------------------------------------- scan
def part_scan():
    cp.scan()


# ------------------------------------------------------------- frontier
def c0mean(c0):
    return float(np.mean(c0))


def c0label(c0):
    c0 = [float(x) for x in c0]
    if max(c0) - min(c0) < 1e-9:
        return '%g' % c0[0]
    return '(' + ', '.join('%g' % x for x in c0) + ')'


def pref_key(r):
    """Preference order of the authors: largest clearance, largest a_d,
    smallest tolerances (eps_e, then eps_v), then the smaller pad, the gain
    closest to Case B, alpha = 0.35 first, the earliest handoff."""
    return (-c0mean(r['c0']), -r['a_d'], r['eps_e'], r['eps_v'], r['vhnd'],
            abs(r['lam'] - 0.055), -r['alpha'], r['T_xi'])


def meets_slacks(r):
    return (r['slack']['H4'] is not None and r['slack']['H4'] >= ADM_MIN
            and r['slack']['H1'] is not None and r['slack']['H1'] >= AUTH_MIN)


def brief(r):
    if r is None:
        return None
    return dict(c0=r['c0'], lam=r['lam'], alpha=r['alpha'], eps_e=r['eps_e'],
                eps_v=r['eps_v'], a_d=r['a_d'], vhnd=r['vhnd'], T_xi=r['T_xi'],
                t_c=r['t_c'], T_ent_bar=r['T_ent_bar'], T_f_bar=r['T_f_bar'],
                admitted=r['admitted'], failed=r['failed'],
                binding=r['binding'], binding_ratio=r['binding_ratio'],
                slack=r['slack'], margins=r['margins'],
                auth_slack=r['auth_slack'],
                auth_slack_nopre=r['auth_slack_nopre'],
                dem_parts_last=r['dem_parts_last'], seg_dem=r['seg_dem'],
                vflr_time_min=r['vflr_time_min'], gmin=r['gmin'],
                disp=r['disp'], align=r['align'], gaperr_max=r['gaperr_max'],
                Vhnd_max=r['Vhnd_max'])


def count_by(rows, key, fmt=lambda x: x):
    out = {}
    for r in rows:
        k = fmt(key(r))
        out.setdefault(k, [0, 0])
        out[k][1] += 1
        if r['admitted']:
            out[k][0] += 1
    return {str(k): dict(admitted=v[0], total=v[1]) for k, v in sorted(out.items())}


def fail_counts(rows):
    out = {}
    for r in rows:
        if r['admitted']:
            continue
        for f in r['failed']:
            out[f] = out.get(f, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def worst_violation(r):
    """largest normalized violation among the failed conditions (slack
    divided by the Case B slack at 20 ms, H6 by the profile-scaled one)."""
    sB = cp.caseB_slacks()
    w = 0.0; wk = None
    for k in r['failed']:
        s = r['slack'].get(k)
        if s is None:
            return None, k
        ref = sB[k] if k != 'H6' else sB[k] / 10.8
        if k == 'P_seg' or ref is None or ref <= 0:
            ref = 1.0
        v = -s / ref
        if v > w:
            w = v; wk = k
    return w, wk


def part_frontier():
    S = load('scan.json')
    if S is None:
        raise SystemExit('scan.json missing; run scan first')
    rows = S['rows']
    grid = S['grid']
    adm = [r for r in rows if r['admitted']]
    req = [r for r in rows if c0mean(r['c0']) >= 42.0]
    ext = [r for r in rows if c0mean(r['c0']) < 42.0]
    F = dict(n_rows=len(rows), n_admitted=len(adm), partial=S.get('partial'),
             runtime_s=S.get('runtime_s'), grid=grid,
             n_rows_requested_axis=len(req),
             n_admitted_requested_axis=sum(1 for r in req if r['admitted']),
             n_rows_extension=len(ext),
             n_admitted_extension=sum(1 for r in ext if r['admitted']))
    F['admitted_by'] = dict(
        c0=count_by(rows, lambda r: c0mean(r['c0']), lambda x: '%g' % x),
        lam=count_by(rows, lambda r: r['lam']),
        alpha=count_by(rows, lambda r: r['alpha']),
        eps_e=count_by(rows, lambda r: r['eps_e']),
        eps_v=count_by(rows, lambda r: r['eps_v']),
        a_d=count_by(rows, lambda r: r['a_d']),
        vhnd=count_by(rows, lambda r: r['vhnd']),
        T_xi=count_by(rows, lambda r: r['T_xi']))
    F['fail_counts_all'] = fail_counts(rows)
    F['fail_counts_requested_axis'] = fail_counts(req)
    F['fail_counts_requested_by_c0'] = {
        c0label(c0): fail_counts([r for r in req if list(r['c0']) == list(c0)])
        for c0 in grid['c0'] if c0mean(c0) >= 42.0}
    F['fail_counts_requested_by_lam'] = {
        '%g' % lam: fail_counts([r for r in req if r['lam'] == lam])
        for lam in grid['lam']}
    # binding conditions among the admitted rows
    bc = {}
    for r in adm:
        bc[r['binding']] = bc.get(r['binding'], 0) + 1
    F['binding_counts_admitted'] = dict(sorted(bc.items(), key=lambda kv: -kv[1]))
    # per c0: smallest admitted T_f_bar and the design that attains it
    per_c0 = {}
    for c0 in grid['c0']:
        sub = [r for r in adm if list(r['c0']) == list(c0)]
        lab = c0label(c0)
        if not sub:
            # the nearest miss: fewest failed conditions, then the smallest
            # worst normalized violation
            cand = [r for r in rows if list(r['c0']) == list(c0)]
            scored = []
            for r in cand:
                w, wk = worst_violation(r)
                scored.append((len(r['failed']), w if w is not None else np.inf, r))
            scored.sort(key=lambda t: (t[0], t[1]))
            near = scored[0][2] if scored else None
            per_c0[lab] = dict(c0=list(c0), n_admitted=0, min_T_f_bar=None,
                               design=None, nearest_miss=brief(near),
                               nearest_miss_worst=worst_violation(near)[1]
                               if near else None)
            continue
        tmin = min(r['T_f_bar'] for r in sub)
        best = sorted([r for r in sub if abs(r['T_f_bar'] - tmin) < 1e-9],
                      key=pref_key)[0]
        ok100 = [r for r in sub if r['T_f_bar'] <= T_F_TARGET]
        ok100s = [r for r in ok100 if meets_slacks(r)]
        per_c0[lab] = dict(
            c0=list(c0), n_admitted=len(sub), min_T_f_bar=tmin,
            design=brief(best),
            n_admitted_100=len(ok100), n_admitted_100_with_slacks=len(ok100s),
            best_100=brief(sorted(ok100, key=pref_key)[0]) if ok100 else None,
            best_100_with_slacks=brief(sorted(ok100s, key=pref_key)[0])
            if ok100s else None,
            largest_a_d_100=max(r['a_d'] for r in ok100) if ok100 else None,
            smallest_lam_100=min(r['lam'] for r in ok100) if ok100 else None)
    F['per_c0'] = per_c0
    # largest admitted c0 for T_f_bar <= 100 and <= 105 (with and without
    # the slack requirements)
    def largest(rows_, tf, slacks):
        sub = [r for r in rows_ if r['admitted'] and r['T_f_bar'] <= tf
               and (meets_slacks(r) if slacks else True)]
        if not sub:
            return None
        cmax = max(c0mean(r['c0']) for r in sub)
        top = [r for r in sub if abs(c0mean(r['c0']) - cmax) < 1e-9]
        return brief(sorted(top, key=pref_key)[0])
    F['largest_c0_100'] = largest(rows, T_F_TARGET, False)
    F['largest_c0_100_with_slacks'] = largest(rows, T_F_TARGET, True)
    F['largest_c0_105'] = largest(rows, T_F_MAX, False)
    F['largest_c0_105_with_slacks'] = largest(rows, T_F_MAX, True)
    # recommendation
    rec = F['largest_c0_100_with_slacks']
    goal_met = rec is not None and c0mean(rec['c0']) >= C0_GOAL
    F['recommendation'] = dict(
        rule='largest c0 with T_f_bar <= 100 s, then largest a_d, then '
             'smallest (eps_e, eps_v), then smallest V^hnd, then lam '
             'closest to 0.055, then alpha = 0.35, then smallest T_xi; '
             'admission slack >= 0.2 m and authority slack >= 0.02 m/s^2',
        goal_c0_ge_46_met=goal_met, recommended=rec,
        best_at_100=F['largest_c0_100_with_slacks'],
        best_at_105=F['largest_c0_105_with_slacks'])
    # what blocks the requested axis: per requested c0 and lam, the
    # failing conditions in the row with the fewest failures
    block = {}
    for c0 in grid['c0']:
        if c0mean(c0) < 42.0:
            continue
        for lam in grid['lam']:
            cand = [r for r in req if list(r['c0']) == list(c0) and r['lam'] == lam]
            if not cand:
                continue
            adm_c = [r for r in cand if r['admitted']]
            if adm_c:
                b = sorted(adm_c, key=pref_key)[0]
                block['%s|%g' % (c0label(c0), lam)] = dict(admitted=True, design=brief(b))
                continue
            scored = sorted(cand, key=lambda r: (len(r['failed']),
                                                 worst_violation(r)[0] or np.inf))
            r = scored[0]
            block['%s|%g' % (c0label(c0), lam)] = dict(
                admitted=False, fewest_failed=r['failed'],
                min_slacks={k: min(x['slack'][k] for x in cand
                                   if x['slack'][k] is not None)
                            for k in r['failed'] if k != 'P_seg'},
                max_slacks={k: max(x['slack'][k] for x in cand
                                   if x['slack'][k] is not None)
                            for k in r['failed'] if k != 'P_seg'},
                row=brief(r))
    F['requested_axis_blockers'] = block
    # reporting: rows that H1 alone rejects and that the H1 slack without
    # the fourth candidate would accept
    n_h1_only = sum(1 for r in rows if r['failed'] == ['H1'])
    n_h1_nopre = sum(1 for r in rows if r['failed'] == ['H1']
                     and r['auth_slack_nopre'] is not None
                     and min(r['auth_slack_nopre']) >= 0)
    F['H1_only_rejections'] = dict(rows_failing_only_H1=n_h1_only,
                                   of_which_pass_without_pre_candidate=n_h1_nopre)
    dump('frontier.json', F)
    dump('scan_summary.json', dict(
        n_rows=F['n_rows'], n_admitted=F['n_admitted'],
        admitted_by=F['admitted_by'], fail_counts_all=F['fail_counts_all'],
        fail_counts_requested_axis=F['fail_counts_requested_axis'],
        binding_counts_admitted=F['binding_counts_admitted']))
    open(os.path.join(OUT, 'frontier.md'), 'w', encoding='utf-8').write(
        markdown(F))
    print('frontier: %d rows, %d admitted; requested axis %d/%d; '
          'largest c0 at 100 s: %s; goal met: %s'
          % (F['n_rows'], F['n_admitted'], F['n_admitted_requested_axis'],
             F['n_rows_requested_axis'],
             None if rec is None else c0label(rec['c0']), goal_met))
    return F


def fmt_design(d, with_slacks=True):
    if d is None:
        return 'none'
    s = ('c0 %s m, lam %g, alpha %g, eps_e %g m, eps_v %g m/s, a_d %g m/s^2, '
         'V^hnd %g m/s^2, T_xi %g s (t_c %.1f s): T_ent_bar %.2f s, '
         'T_f_bar %.2f s' % (c0label(d['c0']), d['lam'], d['alpha'], d['eps_e'],
                             d['eps_v'], d['a_d'], d['vhnd'], d['T_xi'],
                             d['t_c'], d['T_ent_bar'], d['T_f_bar']))
    if with_slacks:
        s += ('; slacks H1 %.3f, H3 %.2f, H4 %.3f, H5 %.3f, H6 %.3f, C_hnd %.1e, '
              'C_al %.3f, C_gap %.3f, C_sync %.3f; binding %s'
              % (d['slack']['H1'], d['slack']['H3'], d['slack']['H4'],
                 d['slack']['H5'], d['slack']['H6'], d['slack']['C_hnd'],
                 d['slack']['C_al'], d['slack']['C_gap'], d['slack']['C_sync'],
                 d['binding']))
    return s


def markdown(F):
    L = []
    L.append('# Case C frontier (revised scan, 2026-09-26)\n')
    L.append('Source: `scan.json` (%d rows, %d admitted; %s). Evaluator '
             '`certify_profile.py` with the refinements (R1) and (R2) '
             '(`evaluator_notes.md`). Fixed: t_d = 10 s, v_c1 = 1 m/s, dbar = '
             '20 ms, gaps (6, 5, 5, 5, 5) m, mismatches of Case B, a_b = 1, '
             'kappa = 1, s_m = 1 m.\n'
             % (F['n_rows'], F['n_admitted'],
                'partial file' if F.get('partial') else 'complete'))
    g = F['grid']
    L.append('Axes: c0 %s m; lam %s; alpha %s; eps_e %s m; eps_v %s m/s; a_d '
             '%s m/s^2; V^hnd %s m/s^2; T_xi %s s.\n'
             % (', '.join(c0label(c) for c in g['c0']), g['lam'], g['alpha'],
                g['eps_e'], g['eps_v'], g['a_d'], g['vhnd'], g['T_xi']))
    L.append('The requested axis (clearances 42 m and more) has %d rows, of '
             'which %d are admitted. The extension (28 to 40 m) has %d rows, '
             'of which %d are admitted.\n'
             % (F['n_rows_requested_axis'], F['n_admitted_requested_axis'],
                F['n_rows_extension'], F['n_admitted_extension']))
    L.append('## 1. Smallest admitted completion bound per clearance\n')
    L.append('| c0 [m] | admitted rows | min T_f_bar [s] | design attaining it | rows with T_f_bar <= 100 s (with slacks) | best design at 100 s with slacks |')
    L.append('|---|---|---|---|---|---|')
    for lab, p in F['per_c0'].items():
        if p['n_admitted'] == 0:
            nm = p['nearest_miss']
            L.append('| %s | 0 | none | nearest miss: %s | 0 | none |'
                     % (lab, 'none' if nm is None else
                        ('%s; failed %s' % (fmt_design(nm, False), nm['failed']))))
        else:
            L.append('| %s | %d | %.2f | %s | %d (%d) | %s |'
                     % (lab, p['n_admitted'], p['min_T_f_bar'],
                        fmt_design(p['design']), p['n_admitted_100'],
                        p['n_admitted_100_with_slacks'],
                        fmt_design(p['best_100_with_slacks'])))
    L.append('')
    L.append('## 2. Largest admitted clearance per completion bound\n')
    for key, lab in (('largest_c0_100', 'T_f_bar <= 100 s'),
                     ('largest_c0_100_with_slacks',
                      'T_f_bar <= 100 s, admission slack >= 0.2 m, authority slack >= 0.02 m/s^2'),
                     ('largest_c0_105', 'T_f_bar <= 105 s'),
                     ('largest_c0_105_with_slacks',
                      'T_f_bar <= 105 s, admission slack >= 0.2 m, authority slack >= 0.02 m/s^2')):
        L.append('* %s: %s' % (lab, fmt_design(F[key])))
    L.append('')
    L.append('## 3. What rejects the requested axis\n')
    L.append('Failing conditions over the %d rows of the requested axis: %s.\n'
             % (F['n_rows_requested_axis'],
                ', '.join('%s %d' % kv for kv in F['fail_counts_requested_axis'].items())))
    L.append('| c0 [m] | failing conditions (rows) |')
    L.append('|---|---|')
    for lab, fc in F['fail_counts_requested_by_c0'].items():
        L.append('| %s | %s |' % (lab, ', '.join('%s %d' % kv for kv in fc.items())))
    L.append('')
    L.append('| lam | failing conditions on the requested axis (rows) |')
    L.append('|---|---|')
    for lab, fc in F['fail_counts_requested_by_lam'].items():
        L.append('| %s | %s |' % (lab, ', '.join('%s %d' % kv for kv in fc.items())))
    L.append('')
    L.append('Per requested clearance and gain: the row with the fewest failed '
             'conditions, its failed set and the range of the failed slacks over '
             'all rows of that (c0, lam).\n')
    L.append('| c0 [m] | lam | fewest failed | slack range of the failed conditions (min .. max over the (c0, lam) rows) | that row |')
    L.append('|---|---|---|---|---|')
    for key, b in F['requested_axis_blockers'].items():
        lab, lam = key.split('|')
        if b['admitted']:
            L.append('| %s | %s | admitted | | %s |' % (lab, lam, fmt_design(b['design'])))
        else:
            rng = ', '.join('%s %.3f .. %.3f' % (k, b['min_slacks'][k], b['max_slacks'][k])
                            for k in b['min_slacks'])
            L.append('| %s | %s | %s | %s | %s |'
                     % (lab, lam, b['fewest_failed'], rng, fmt_design(b['row'], False)))
    L.append('')
    L.append('## 4. Counts\n')
    for ax, d in F['admitted_by'].items():
        L.append('* by %s: %s' % (ax, '; '.join('%s: %d of %d' % (k, v['admitted'], v['total'])
                                                 for k, v in d.items())))
    L.append('* failing conditions over all rejected rows: %s'
             % ', '.join('%s %d' % kv for kv in F['fail_counts_all'].items()))
    L.append('* binding condition among the admitted rows (smallest slack ratio to Case B at 20 ms): %s'
             % ', '.join('%s %d' % kv for kv in F['binding_counts_admitted'].items()))
    L.append('* rows rejected by H1 alone: %d, of which %d would pass without the '
             'candidate (a_b + F^b_(i-1)) + fbar_i (reporting only)'
             % (F['H1_only_rejections']['rows_failing_only_H1'],
                F['H1_only_rejections']['of_which_pass_without_pre_candidate']))
    L.append('')
    L.append('## 5. Recommendation\n')
    R = F['recommendation']
    L.append('Rule: %s.\n' % R['rule'])
    L.append('Goal (c0 >= 46 m with T_f_bar <= 100 s) met: %s.\n' % R['goal_c0_ge_46_met'])
    L.append('* recommended (best at 100 s): %s' % fmt_design(R['best_at_100']))
    L.append('* best at <= 105 s: %s' % fmt_design(R['best_at_105']))
    L.append('')
    return '\n'.join(L)


# ------------------------------------------------------------- recommend
def design_cfg(d):
    return cp.config(c0=tuple(d['c0']), lam=d['lam'], alpha=d['alpha'],
                     eps_e=d['eps_e'], eps_v=d['eps_v'], a_d=d['a_d'],
                     vhnd=d['vhnd'], T_xi=d['T_xi'], t_d=10.0, v_c1=1.0,
                     dbar=0.02, refine=True)


def full_record(d, tag):
    cfg = design_cfg(d)
    t0 = time.time()
    ev = cp.evaluate(cfg)
    ce = cp.age_ceilings(cfg, verbose=False)
    bd = cp.age_boundary_ms(cfg, kmax=200)
    enc20 = ip.check_enclosure(cfg)
    encb = ip.check_enclosure(dict(cfg, dbar=bd['k_ms'] * 1e-3)) \
        if bd['k_ms'] > 0 else None
    # per-condition 1 ms boundaries (largest k with the condition holding);
    # one evaluation memo for all conditions
    per = {}
    memo = {}

    def res_at(k):
        if k not in memo:
            memo[k] = cp.evaluate(dict(cfg, dbar=k * 1e-3))
        return memo[k]

    for nm in COND:
        if nm == 'P_seg':
            continue

        def ok(k, nm=nm):
            r = res_at(k)
            return bool(r['closed'] and r['conds'][nm]['ok'])
        if not ok(1):
            per[nm] = 0; continue
        if ok(200):
            per[nm] = 200; continue
        lo, hi = 1, 200
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        per[nm] = lo
    rec = dict(tag=tag, cfg={k: v for k, v in cfg.items() if k != 'G'},
               certificate=ev, ceilings=ce, boundary=bd,
               cond_boundary_ms=per,
               interval_20ms=dict(verified=enc20['enclosure']['verified'],
                                  all_inside=enc20['all_inside'],
                                  conds=enc20['enclosure']['conds'],
                                  values=enc20['enclosure']['values']),
               interval_boundary=None if encb is None else dict(
                   dbar=bd['k_ms'] * 1e-3,
                   verified=encb['enclosure']['verified'],
                   all_inside=encb['all_inside'],
                   conds=encb['enclosure']['conds']),
               runtime_s=time.time() - t0)
    print('  %s: admitted %s, T_f_bar %.2f, boundary %d ms (fails %s), '
          'interval verified %s (%.0f s)'
          % (tag, ev['admitted'], ev['consts']['T_f_bar'], bd['k_ms'],
             bd['failed_next'], enc20['enclosure']['verified'], rec['runtime_s']),
          flush=True)
    return rec


def part_recommend():
    F = load('frontier.json')
    if F is None:
        F = part_frontier()
    R = F['recommendation']
    picks = []
    seen = set()
    for tag, key in (('recommended_100', 'best_at_100'),
                     ('best_105', 'best_at_105')):
        d = R.get(key)
        if d is None:
            continue
        k = json.dumps(cp._py(dict(c0=d['c0'], lam=d['lam'], alpha=d['alpha'],
                                   eps_e=d['eps_e'], eps_v=d['eps_v'],
                                   a_d=d['a_d'], vhnd=d['vhnd'], T_xi=d['T_xi'])))
        if k in seen:
            continue
        seen.add(k)
        picks.append((tag, d))
    # also the largest-clearance design without the slack requirements at
    # 100 s and at 105 s when they differ
    for tag, key in (('largest_c0_100_no_slack_rule', 'largest_c0_100'),
                     ('largest_c0_105_no_slack_rule', 'largest_c0_105')):
        d = F.get(key)
        if d is None:
            continue
        k = json.dumps(cp._py(dict(c0=d['c0'], lam=d['lam'], alpha=d['alpha'],
                                   eps_e=d['eps_e'], eps_v=d['eps_v'],
                                   a_d=d['a_d'], vhnd=d['vhnd'], T_xi=d['T_xi'])))
        if k in seen:
            continue
        seen.add(k)
        picks.append((tag, d))
    out = dict(rule=R['rule'], goal_c0_ge_46_met=R['goal_c0_ge_46_met'],
               designs=[])
    for tag, d in picks:
        out['designs'].append(full_record(d, tag))
    dump('recommend.json', out)
    return out


if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    os.makedirs(OUT, exist_ok=True)
    for a in args:
        if a == 'all':
            part_regress(); part_frontier(); part_recommend()
        elif a == 'regress':
            part_regress()
        elif a == 'scan':
            part_scan()
        elif a == 'frontier':
            part_frontier()
        elif a == 'recommend':
            part_recommend()
        else:
            raise SystemExit('unknown part %s' % a)
