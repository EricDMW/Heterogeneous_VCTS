#!/usr/bin/env python3
"""run_caseC_ext.py -- late-handoff extension of the revised Case C scan
(authors' clarification of 2026-09-26: clearances about 50 m first, the
running time may exceed 100 s, dispersion requirement 1.0 s).

Parts (data under data/new_d19/caseC/ only)
  scan      certify_profile.scan of grid_ext() -> scan_ext.json (resumable;
            scan.json is not touched)
  scan_low  the same late-handoff grid at the clearances 40 and 44 m
            (gains 0.055 to 0.07) -> scan_ext_low.json, so that the
            frontier T_f_bar(c0) has no gap between 36 and 46 m
  compact   scan.json (376 MB) streamed once into scan_compact.json (the
            per-row scalars; the C_sync slack recomputed for d_req = 1.0)
  frontier  frontier.md, frontier.json, scan_summary.json rebuilt from
            scan_compact.json, scan_ext.json and scan_ext_low.json:
            T_f_bar against c0, the slacks, the selection
  select    design.json (selected design with its full certificate
            record) from the selection of the frontier
  all       compact frontier select (the scans are run separately)

Usage: python run_caseC_ext.py [part ...]
"""
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
CODE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
import numpy as np                                      # noqa: E402
import certify_profile as cp                            # noqa: E402
import interval_profile as ip                           # noqa: E402
import run_caseC_scan2 as rs                            # noqa: E402

OUT = cp.DATA_C
COND = list(cp.COND_NAMES)
D_REQ = 1.0            # dispersion requirement of Case C [s]
ADM_MIN = 0.15         # preferred admission slack [m]
AUTH_MIN = 0.02        # preferred authority slack [m/s^2]
H3_RESERVE = 0.5       # entry reserve of the 1 s handoff refinement [s]
LAM_REF = 0.055        # Case B gain (tie-break)
DBAR = 0.020
FIXED_C = dict(t_d=10.0, v_c1=1.0, dbar=DBAR, d_s=[6.0, 5.0, 5.0, 5.0, 5.0],
               eps0=[-0.30, 0.25, 0.18, 0.12, 0.08], refine=True, d_req=D_REQ)
SCAN_EXT_LOW_PATH = os.path.join(OUT, 'scan_ext_low.json')
SEL_LABELS = ('50', '48', '(47, 50, 52, 54)')   # 48 to 50 m of the rule
ABOUT50 = 46.0                                  # smallest mean clearance


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    json.dump(cp._py(obj), open(os.path.join(OUT, name), 'w'), indent=1)


def load(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


c0label = rs.c0label
c0mean = rs.c0mean


# ------------------------------------------------------------- grids
def grid_ext_low():
    """Late-handoff grid at the clearances 40 and 44 m (between the
    admitted 36 m of scan.json and the 46 m of grid_ext): gains 0.055 to
    0.07, T_xi 105 to 130 s, otherwise as grid_ext()."""
    g = cp.grid_ext()
    g['c0'] = [(40.0,) * 4, (44.0,) * 4]
    g['lam'] = [0.055, 0.06, 0.065, 0.07]
    g['T_xi'] = [float(x) for x in range(105, 131, 5)]
    return g


# ------------------------------------------------------------- scans
def part_scan():
    cp.scan(grid=cp.grid_ext(), out_path=cp.SCAN_EXT_PATH)


def part_scan_low():
    cp.scan(grid=grid_ext_low(), out_path=SCAN_EXT_LOW_PATH)


# ------------------------------------------------------------- compact
COMPACT_KEYS = ('c0', 'lam', 'alpha', 'eps_e', 'eps_v', 'a_d', 'vhnd', 'T_xi',
                't_c', 'closed', 'admitted', 'failed', 'binding',
                'binding_ratio', 'slack', 'T_ent_bar', 'T_f_bar', 'T_sw_bar',
                'vflr_time_min', 'gmin', 'disp', 'align', 'gaperr_max',
                'Vhnd_max')


def compact_row(r, d_req_row):
    """Per-row scalars.  The C_sync slack is recomputed for D_REQ (the
    rows of scan.json were evaluated with the Case B requirement 0.9 s;
    C_sync never fails, so the admitted flag is unchanged)."""
    c = {k: r.get(k) for k in COMPACT_KEYS}
    c['d_req_row'] = d_req_row
    if c.get('slack') is not None and r.get('disp') is not None:
        c['slack'] = dict(c['slack'])
        c['slack_C_sync_row'] = c['slack'].get('C_sync')
        c['slack']['C_sync'] = D_REQ - float(r['disp'])
    c['min_margin'] = (min(r['margins']) if r.get('margins') else None)
    c['min_auth'] = (min(r['auth_slack']) if r.get('auth_slack') else None)
    c['min_auth_nopre'] = (min(r['auth_slack_nopre'])
                           if r.get('auth_slack_nopre') else None)
    c['seg_dem_max'] = (max(r['seg_dem']) if r.get('seg_dem') else None)
    c['Tent'] = r.get('Tent')
    return c


def stream_rows(path):
    """Yield the rows of a scan file one at a time (the file is read as
    text; only one row is decoded at a time)."""
    txt = open(path, encoding='utf-8').read()
    dec = json.JSONDecoder()
    i = txt.index('"grid": ') + len('"grid": ')
    grid, _ = dec.raw_decode(txt, i)
    i = txt.index('"rows": [') + len('"rows": [')
    n = len(txt)
    yield ('grid', grid)
    while True:
        while i < n and txt[i] in ' \n\r\t,':
            i += 1
        if txt[i] == ']':
            break
        obj, i = dec.raw_decode(txt, i)
        yield ('row', obj)


def part_compact():
    t0 = time.time()
    rows = []
    grid = None
    nfail_sync = 0
    for kind, obj in stream_rows(cp.SCAN_PATH):
        if kind == 'grid':
            grid = obj
            continue
        c = compact_row(obj, 0.9)
        if c['closed'] and c['slack_C_sync_row'] is not None \
                and c['slack_C_sync_row'] < 0:
            nfail_sync += 1
        rows.append(c)
    out = dict(source='scan.json', grid=grid, n_rows=len(rows),
               n_admitted=sum(1 for r in rows if r['admitted']),
               d_req_rows=0.9, d_req=D_REQ,
               n_rows_failing_C_sync_at_0p9=nfail_sync,
               min_C_sync_slack_at_0p9=min(r['slack_C_sync_row'] for r in rows
                                           if r['closed']),
               note='per-row scalars of scan.json; slack.C_sync recomputed '
                    'as d_req - disp with d_req = %g (the rows were '
                    'evaluated with 0.9 s; slack_C_sync_row keeps the '
                    'original value); Tent kept' % D_REQ,
               runtime_s=time.time() - t0, rows=rows)
    dump('scan_compact.json', out)
    print('compact: %d rows, %d admitted, C_sync failures at 0.9 s: %d, '
          'min C_sync slack %.4f (%.0f s)'
          % (len(rows), out['n_admitted'], nfail_sync,
             out['min_C_sync_slack_at_0p9'], out['runtime_s']))
    return out


# ------------------------------------------------------------- frontier
def load_all():
    """The merged row table: scan_compact.json (rows evaluated at 0.9 s,
    C_sync slack recomputed), scan_ext.json and scan_ext_low.json (rows
    evaluated at d_req = 1.0)."""
    S = load('scan_compact.json')
    if S is None:
        S = part_compact()
    rows = []
    for r in S['rows']:
        r = dict(r); r['source'] = 'scan'
        rows.append(r)
    grids = dict(scan=S['grid'])
    for name, tag in (('scan_ext.json', 'ext'), ('scan_ext_low.json', 'ext_low')):
        E = load(name)
        if E is None:
            continue
        grids[tag] = E['grid']
        grids[tag + '_partial'] = E.get('partial')
        for r in E['rows']:
            c = compact_row(r, r.get('d_req', D_REQ))
            c['source'] = tag
            rows.append(c)
    return rows, grids


def meets(r, adm=ADM_MIN, auth=AUTH_MIN):
    s = r['slack']
    return (r['admitted'] and s['H4'] is not None and s['H4'] >= adm
            and s['H1'] is not None and s['H1'] >= auth)


def meets_but_H3(r, adm=ADM_MIN, auth=AUTH_MIN):
    """Rows that hold every condition except the entry deadline (H3) with
    the preferred slacks: they are admitted at a later handoff than the
    grid holds (T_xi_star of the 1 s refinement), because T_ent_bar does
    not depend on T_xi and C_hnd and P_seg hold for every later T_xi."""
    s = r['slack']
    return (r['closed'] and not r['admitted'] and set(r['failed']) == {'H3'}
            and s['H4'] is not None and s['H4'] >= adm
            and s['H1'] is not None and s['H1'] >= auth)


def txi_star(r):
    """Handoff instant of the 1 s refinement: the smallest integer T with
    T - T_ent_bar - dbar >= H3_RESERVE (the conditions that depend on T_xi
    are H3, C_hnd and P_seg, all of which hold for every later T_xi once
    they hold; the other conditions do not depend on T_xi)."""
    if r['T_ent_bar'] is None or not np.isfinite(r['T_ent_bar']):
        return None
    return int(math.ceil(r['T_ent_bar'] + DBAR + H3_RESERVE - 1e-9))


def tf_star(r):
    t = txi_star(r)
    return None if t is None else t + (r['T_f_bar'] - r['T_xi'])


def rank_key(r):
    """Authors' order: smallest completion bound (at the 1 s resolution of
    the handoff refinement), largest a_d, smallest box (eps_e, then
    eps_v), then the larger clearance, the gain nearest Case B, alpha =
    0.35 first, the smaller pad, the earlier scan handoff."""
    return (txi_star(r), -r['a_d'], r['eps_e'], r['eps_v'], -c0mean(r['c0']),
            abs(r['lam'] - LAM_REF), -r['alpha'], r['vhnd'], r['T_xi'])


def brief(r):
    if r is None:
        return None
    b = {k: r.get(k) for k in ('c0', 'lam', 'alpha', 'eps_e', 'eps_v', 'a_d',
                               'vhnd', 'T_xi', 't_c', 'T_ent_bar', 'T_f_bar',
                               'admitted', 'failed', 'binding',
                               'binding_ratio', 'slack', 'min_margin',
                               'min_auth', 'min_auth_nopre', 'seg_dem_max',
                               'vflr_time_min', 'gmin', 'disp', 'align',
                               'gaperr_max', 'Vhnd_max', 'source')}
    b['T_xi_star'] = txi_star(r)
    b['T_f_bar_star'] = tf_star(r)
    return b


def fmt(d, slacks=True):
    if d is None:
        return 'none'
    def g(x, fm='%.2f'):
        return 'none' if x is None or not np.isfinite(x) else fm % x
    s = ('c0 %s m, lam %g, alpha %g, eps_e %g m, eps_v %g m/s, a_d %g m/s^2, '
         'T_xi %g s (t_c %s s): T_ent_bar %s s, T_f_bar %s s'
         % (c0label(d['c0']), d['lam'], d['alpha'], d['eps_e'], d['eps_v'],
            d['a_d'], d['T_xi'], g(d['t_c'], '%.1f'), g(d['T_ent_bar']),
            g(d['T_f_bar'])))
    if d.get('T_xi_star') is not None:
        s += ' (1 s refinement: T_xi %d s, T_f_bar %.2f s)' % (d['T_xi_star'],
                                                               d['T_f_bar_star'])
    if slacks and d['slack']['H1'] is not None:
        s += ('; slacks H1 %.3f, H3 %.2f, H4 %.3f, H5 %.3f, H6 %.3f, C_hnd '
              '%.1e, C_al %.3f, C_gap %.3f, C_sync %.3f'
              % (d['slack']['H1'], d['slack']['H3'], d['slack']['H4'],
                 d['slack']['H5'], d['slack']['H6'], d['slack']['C_hnd'],
                 d['slack']['C_al'], d['slack']['C_gap'], d['slack']['C_sync']))
    return s


def _first(rows, key=rank_key):
    return sorted(rows, key=key)[0] if rows else None


def _fmin(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return min(xs) if xs else None


def _fmax(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return max(xs) if xs else None


def c0_order(labels):
    def key(lab):
        if lab.startswith('('):
            return (float(np.mean([float(x) for x in lab.strip('()').split(',')])), 1)
        return (float(lab), 0)
    return sorted(labels, key=key)


def part_frontier():
    rows, grids = load_all()
    adm = [r for r in rows if r['admitted']]
    labels = c0_order(sorted(set(c0label(r['c0']) for r in rows)))
    F = dict(sources=dict(scan=dict(n_rows=sum(1 for r in rows if r['source'] == 'scan'),
                                    grid=grids.get('scan'), d_req_rows=0.9),
                          ext=dict(n_rows=sum(1 for r in rows if r['source'] == 'ext'),
                                   grid=grids.get('ext'),
                                   partial=grids.get('ext_partial'), d_req_rows=D_REQ),
                          ext_low=dict(n_rows=sum(1 for r in rows if r['source'] == 'ext_low'),
                                       grid=grids.get('ext_low'),
                                       partial=grids.get('ext_low_partial'),
                                       d_req_rows=D_REQ)),
             d_req=D_REQ, adm_min=ADM_MIN, auth_min=AUTH_MIN,
             h3_reserve=H3_RESERVE, n_rows=len(rows), n_admitted=len(adm))
    # ---- per clearance: the frontier of T_f_bar
    per = {}
    for lab in labels:
        sub = [r for r in rows if c0label(r['c0']) == lab]
        a = [r for r in sub if r['admitted']]
        s = [r for r in a if meets(r)]
        bg = [r for r in sub if meets_but_H3(r)] if not s else []
        ent = dict(c0=list(sub[0]['c0']), n_rows=len(sub), n_admitted=len(a),
                   n_admitted_with_slacks=len(s),
                   sources=sorted(set(r['source'] for r in sub)),
                   lam_admitted=sorted(set(r['lam'] for r in a)),
                   T_xi_admitted_min=_fmin([r['T_xi'] for r in a]),
                   T_ent_bar_min_admitted=_fmin([r['T_ent_bar'] for r in a]),
                   T_ent_bar_min_all=_fmin([r['T_ent_bar'] for r in sub]),
                   max_H4_admitted=_fmax([r['slack']['H4'] for r in a]),
                   max_H1_admitted=_fmax([r['slack']['H1'] for r in a]),
                   a_d_admitted_max=_fmax([r['a_d'] for r in a]),
                   a_d_admitted_with_slacks_max=_fmax([r['a_d'] for r in s]),
                   best_any=brief(_first(a)),
                   best_with_slacks=brief(_first(s)),
                   n_with_slacks_beyond_grid=len(bg),
                   best_with_slacks_beyond_grid=brief(_first(bg)),
                   min_T_f_bar_any=_fmin([r['T_f_bar'] for r in a]),
                   min_T_f_bar_with_slacks=_fmin([r['T_f_bar'] for r in s]),
                   min_T_f_bar_star_any=_fmin([tf_star(r) for r in a]),
                   min_T_f_bar_star_with_slacks=_fmin([tf_star(r) for r in s]))
        if not a:
            scored = sorted(sub, key=lambda r: (len(r['failed']),
                                                rs.worst_violation(r)[0] or np.inf))
            ent['nearest_miss'] = brief(scored[0]) if scored else None
        per[lab] = ent
    F['per_c0'] = per
    # ---- per (c0, lam): the trade between gain and handoff
    trade = {}
    for lab in labels:
        for lam in sorted(set(r['lam'] for r in rows if c0label(r['c0']) == lab)):
            sub = [r for r in rows if c0label(r['c0']) == lab and r['lam'] == lam]
            a = [r for r in sub if r['admitted']]
            s = [r for r in a if meets(r)]
            trade['%s|%g' % (lab, lam)] = dict(
                c0=lab, lam=lam, n_rows=len(sub), n_admitted=len(a),
                n_with_slacks=len(s),
                T_xi_range=[min(r['T_xi'] for r in sub), max(r['T_xi'] for r in sub)],
                T_ent_bar_min=_fmin([r['T_ent_bar'] for r in sub]),
                T_ent_bar_min_admitted=_fmin([r['T_ent_bar'] for r in a]),
                max_H4=_fmax([r['slack']['H4'] for r in sub]),
                max_H1=_fmax([r['slack']['H1'] for r in sub]),
                max_H5=_fmax([r['slack']['H5'] for r in sub]),
                min_T_xi_admitted=_fmin([r['T_xi'] for r in a]),
                min_T_xi_with_slacks=_fmin([r['T_xi'] for r in s]),
                best_with_slacks=brief(_first(s)),
                best_any=brief(_first(a)),
                n_slack_ok_beyond_grid=sum(1 for r in sub if meets_but_H3(r)),
                T_ent_bar_min_slack_ok_beyond_grid=_fmin(
                    [r['T_ent_bar'] for r in sub if meets_but_H3(r)]),
                fail_counts=rs.fail_counts(sub))
    F['trade'] = trade
    # ---- counts of the extension
    ext = [r for r in rows if r['source'] in ('ext', 'ext_low')]
    F['ext_counts'] = dict(
        n_rows=len(ext), n_admitted=sum(1 for r in ext if r['admitted']),
        by=dict(c0=rs.count_by(ext, lambda r: c0label(r['c0'])),
                lam=rs.count_by(ext, lambda r: r['lam']),
                alpha=rs.count_by(ext, lambda r: r['alpha']),
                eps_e=rs.count_by(ext, lambda r: r['eps_e']),
                eps_v=rs.count_by(ext, lambda r: r['eps_v']),
                a_d=rs.count_by(ext, lambda r: r['a_d']),
                T_xi=rs.count_by(ext, lambda r: r['T_xi'])),
        fail_counts=rs.fail_counts(ext),
        binding_counts={k: v for k, v in sorted(
            ((b, sum(1 for r in ext if r['admitted'] and r['binding'] == b))
             for b in set(r['binding'] for r in ext if r['admitted'])),
            key=lambda kv: -kv[1])})
    # ---- selection
    tiers = []
    tierA = [r for r in adm if c0label(r['c0']) in SEL_LABELS and meets(r)]
    tiers.append(('A: 48 to 50 m with admission slack >= %g m and authority '
                  'slack >= %g m/s^2' % (ADM_MIN, AUTH_MIN), tierA))
    tierB = [r for r in adm if c0mean(r['c0']) >= ABOUT50 and meets(r)
             and c0label(r['c0']) not in SEL_LABELS]
    tiers.append(('B: other clearances of 46 m or more with the slacks', tierB))
    tierC = [r for r in adm if c0mean(r['c0']) >= ABOUT50 and not meets(r)]
    tiers.append(('C: 46 m or more with thinner slacks', tierC))
    tierD = [r for r in adm if c0mean(r['c0']) < ABOUT50]
    tiers.append(('D: below 46 m', tierD))
    sel = None
    tier_used = None
    for name, t in tiers:
        if t:
            sel = _first(t); tier_used = name
            break
    alt = {}
    for lab in labels:
        s = [r for r in adm if c0label(r['c0']) == lab and meets(r)]
        a = [r for r in adm if c0label(r['c0']) == lab]
        alt[lab] = dict(with_slacks=brief(_first(s)), any=brief(_first(a)))
    # the ranked list of tier A (top 12 distinct designs; the rows of one
    # design at the several scan handoffs differ only in the H3, C_hnd
    # and P_seg slacks) for the notes
    ranked = []; seen = set()
    for r in sorted(tierA, key=rank_key):
        key = (tuple(r['c0']), r['lam'], r['alpha'], r['eps_e'], r['eps_v'],
               r['a_d'], r['vhnd'])
        if key in seen:
            continue
        seen.add(key); ranked.append(brief(r))
        if len(ranked) >= 12:
            break
    # the largest a_d admitted with slacks at 48 and 50 m and its best row
    F['selection'] = dict(
        rule=('tier A: c0 in {48, 50, (47, 50, 52, 54)} m, admitted at 20 ms, '
              'admission slack >= %g m, authority slack >= %g m/s^2; then '
              'tier B (other c0 >= 46 m with the slacks), C (c0 >= 46 m, '
              'thinner slacks), D (c0 < 46 m).  Inside a tier: smallest '
              'T_xi of the 1 s refinement (smallest integer T_xi with '
              'T_xi - T_ent_bar - dbar >= %g s), then largest a_d, then '
              'smallest eps_e, then smallest eps_v, then the larger '
              'clearance, the gain nearest %g, alpha = 0.35 first, the '
              'smaller pad, the earlier scan handoff'
              % (ADM_MIN, AUTH_MIN, H3_RESERVE, LAM_REF)),
        tiers={name: len(t) for name, t in tiers}, tier_used=tier_used,
        selected=brief(sel), ranked_tier_A=ranked, per_c0_best=alt)
    dump('frontier.json', F)
    dump('scan_summary.json', dict(
        n_rows=F['n_rows'], n_admitted=F['n_admitted'], per_c0={
            lab: dict(n_rows=p['n_rows'], n_admitted=p['n_admitted'],
                      n_with_slacks=p['n_admitted_with_slacks'],
                      min_T_f_bar_any=p['min_T_f_bar_any'],
                      min_T_f_bar_with_slacks=p['min_T_f_bar_with_slacks'])
            for lab, p in per.items()},
        ext_counts=F['ext_counts'], selection=dict(
            tier_used=tier_used, selected=F['selection']['selected'])))
    open(os.path.join(OUT, 'frontier.md'), 'w', encoding='utf-8').write(markdown(F))
    print('frontier: %d rows, %d admitted; tier %s; selected: %s'
          % (len(rows), len(adm), tier_used, fmt(F['selection']['selected'])))
    return F


def markdown(F):
    L = []
    L.append('# Case C frontier (revised scan and late-handoff extension, 2026-09-26)\n')
    src = F['sources']
    L.append('Sources: `scan.json` (%d rows, compacted in `scan_compact.json`; '
             'handoff 88 to 103 s, gains 0.06 to 0.09, clearances 28 to 52 m, '
             'evaluated with the Case B dispersion requirement 0.9 s; the '
             'C_sync slack is recomputed here for the Case C requirement %g s, '
             'C_sync fails in no row of either requirement), `scan_ext.json` '
             '(%d rows%s; handoff 105 to 150 s, gains 0.05 to 0.065, '
             'clearances 46 to 52 m and the two Case B patterns, V^hnd = '
             '1e-2 m/s^2, dispersion requirement %g s) and `scan_ext_low.json` '
             '(%d rows%s; the same late handoffs at 40 and 44 m, gains 0.055 '
             'to 0.07, handoff 105 to 130 s). Evaluator `certify_profile.py` '
             'with the refinements (R1) and (R2) (`evaluator_notes.md`). '
             'Fixed: t_d = 10 s, v_c1 = 1 m/s, dbar = 20 ms, gaps (6, 5, 5, 5, '
             '5) m, mismatches of Case B, a_b = 1, kappa = 1, s_m = 1 m.\n'
             % (src['scan']['n_rows'], F['d_req'], src['ext']['n_rows'],
                ' (partial file)' if src['ext'].get('partial') else '',
                F['d_req'], src['ext_low']['n_rows'],
                ' (partial file)' if src['ext_low'].get('partial') else ''))
    g = src['ext'].get('grid') or {}
    if g:
        L.append('Axes of `scan_ext.json`: c0 %s m; lam %s; alpha %s; eps_e %s m; '
                 'eps_v %s m/s; a_d %s m/s^2; V^hnd %s m/s^2; T_xi %s s.\n'
                 % (', '.join(c0label(c) for c in g['c0']), g['lam'], g['alpha'],
                    g['eps_e'], g['eps_v'], g['a_d'], g['vhnd'], g['T_xi']))
    L.append('The gains above 0.065 are not in the extension because (H1) and '
             '(H4), which do not depend on T_xi, fail at every clearance of '
             '46 m or more for lam >= 0.07 in `scan.json`; the gains 0.05 and '
             '0.055 are not in `scan.json` because their entry deadlines lie '
             'beyond 103 s at every clearance of the requested axis.\n')
    L.append('"With slacks" means admission slack (H4) >= %g m and authority '
             'slack (H1) >= %g m/s^2. "1 s refinement" means the smallest '
             'integer handoff T_xi with T_xi - T_ent_bar - dbar >= %g s, '
             'available for every admitted row because T_ent_bar and '
             'T_f_bar - T_xi do not depend on T_xi and the T_xi-dependent '
             'conditions (H3, C_hnd, P_seg) hold for every later handoff '
             'once they hold.\n' % (F['adm_min'], F['auth_min'], F['h3_reserve']))
    L.append('## 1. Frontier: smallest certified completion bound per clearance\n')
    L.append('| c0 [m] | rows | admitted (with slacks) | admitted gains | min T_xi admitted [s] | min T_ent_bar admitted [s] | min T_f_bar any [s] (1 s refinement) | min T_f_bar with slacks [s] (1 s refinement) | max H4, H1 slack among admitted | largest a_d admitted (with slacks) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|')
    for lab, p in F['per_c0'].items():
        def f(x, fm='%.2f'):
            return 'none' if x is None else fm % x
        L.append('| %s | %d | %d (%d) | %s | %s | %s | %s (%s) | %s (%s) | %s, %s | %s (%s) |'
                 % (lab, p['n_rows'], p['n_admitted'], p['n_admitted_with_slacks'],
                    ', '.join('%g' % x for x in p['lam_admitted']) or 'none',
                    f(p['T_xi_admitted_min'], '%g'), f(p['T_ent_bar_min_admitted']),
                    f(p['min_T_f_bar_any']), f(p['min_T_f_bar_star_any']),
                    f(p['min_T_f_bar_with_slacks']), f(p['min_T_f_bar_star_with_slacks']),
                    f(p['max_H4_admitted'], '%.3f'), f(p['max_H1_admitted'], '%.3f'),
                    f(p['a_d_admitted_max'], '%g'),
                    f(p['a_d_admitted_with_slacks_max'], '%g')))
    L.append('')
    L.append('Designs attaining the frontier (best row per clearance in the '
             'selection order; slacks at 20 ms):\n')
    for lab, p in F['per_c0'].items():
        if p['n_admitted'] == 0:
            nm = p.get('nearest_miss')
            L.append('* %s m: none admitted; nearest miss %s; failed %s'
                     % (lab, fmt(nm, False), nm['failed'] if nm else None))
            continue
        L.append('* %s m, with slacks: %s' % (lab, fmt(p['best_with_slacks'])))
        if p['best_with_slacks'] is None and p.get('best_with_slacks_beyond_grid'):
            L.append('* %s m, with slacks beyond the handoff grid (every '
                     'condition but H3 holds in %d rows; admitted at the '
                     'refined handoff): %s'
                     % (lab, p['n_with_slacks_beyond_grid'],
                        fmt(p['best_with_slacks_beyond_grid'])))
        if p['best_any'] and p['best_with_slacks'] and \
                rank_key(p['best_any']) != rank_key(p['best_with_slacks']):
            L.append('* %s m, any slacks: %s' % (lab, fmt(p['best_any'])))
        elif p['best_with_slacks'] is None:
            L.append('* %s m, any slacks: %s' % (lab, fmt(p['best_any'])))
    L.append('')
    L.append('## 2. Gain against handoff per clearance\n')
    L.append('Per (c0, lam): the smallest certified entry deadline T_ent_bar '
             'over the boxes, rates and S1 rates, the largest admission and '
             'authority slacks, the smallest admitted T_xi (with slacks in '
             'parentheses) and the failing conditions.\n')
    L.append('| c0 [m] | lam | rows | T_xi range [s] | min T_ent_bar [s] | max H4 [m] | max H1 [m/s^2] | max H5 [m] | admitted (with slacks) | min T_xi admitted (with slacks) | rows with the slacks failing H3 alone (min T_ent_bar) | failing conditions |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key, t in F['trade'].items():
        def f(x, fm='%.2f'):
            return 'none' if x is None else fm % x
        L.append('| %s | %g | %d | %g to %g | %s | %s | %s | %s | %d (%d) | %s (%s) | %d (%s) | %s |'
                 % (t['c0'], t['lam'], t['n_rows'], t['T_xi_range'][0],
                    t['T_xi_range'][1], f(t['T_ent_bar_min']), f(t['max_H4'], '%+.3f'),
                    f(t['max_H1'], '%+.3f'), f(t['max_H5'], '%+.3f'),
                    t['n_admitted'], t['n_with_slacks'],
                    f(t['min_T_xi_admitted'], '%g'), f(t['min_T_xi_with_slacks'], '%g'),
                    t['n_slack_ok_beyond_grid'], f(t['T_ent_bar_min_slack_ok_beyond_grid']),
                    ', '.join('%s %d' % kv for kv in t['fail_counts'].items()) or 'none'))
    L.append('')
    L.append('## 3. Counts of the extension\n')
    E = F['ext_counts']
    L.append('%d rows, %d admitted.\n' % (E['n_rows'], E['n_admitted']))
    for ax, d in E['by'].items():
        L.append('* by %s: %s' % (ax, '; '.join('%s: %d of %d' % (k, v['admitted'], v['total'])
                                                 for k, v in d.items())))
    L.append('* failing conditions over the rejected rows: %s'
             % ', '.join('%s %d' % kv for kv in E['fail_counts'].items()))
    L.append('* binding condition among the admitted rows (smallest slack ratio to Case B at 20 ms): %s'
             % ', '.join('%s %d' % kv for kv in E['binding_counts'].items()))
    L.append('')
    L.append('## 4. Selection\n')
    S = F['selection']
    L.append('Rule: %s.\n' % S['rule'])
    L.append('Tier sizes: %s. Tier used: %s.\n'
             % ('; '.join('%s: %d' % kv for kv in S['tiers'].items()), S['tier_used']))
    L.append('Selected: %s\n' % fmt(S['selected']))
    if S['ranked_tier_A']:
        L.append('Ranked tier A (first %d):\n' % len(S['ranked_tier_A']))
        L.append('| rank | c0 [m] | lam | alpha | eps_e [m] | eps_v [m/s] | a_d [m/s^2] | T_ent_bar [s] | T_xi (1 s) [s] | T_f_bar (1 s) [s] | H1 | H4 | H5 | H6 | C_al | C_hnd |')
        L.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
        for j, d in enumerate(S['ranked_tier_A']):
            s = d['slack']
            L.append('| %d | %s | %g | %g | %g | %g | %g | %.2f | %d | %.2f | %.3f | %.3f | %.3f | %.3f | %.3f | %.1e |'
                     % (j + 1, c0label(d['c0']), d['lam'], d['alpha'], d['eps_e'],
                        d['eps_v'], d['a_d'], d['T_ent_bar'], d['T_xi_star'],
                        d['T_f_bar_star'], s['H1'], s['H4'], s['H5'], s['H6'],
                        s['C_al'], s['C_hnd']))
        L.append('')
    L.append('Best admitted row per clearance (with slacks, and with any slacks '
             'when different):\n')
    for lab, d in S['per_c0_best'].items():
        L.append('* %s m: with slacks %s' % (lab, fmt(d['with_slacks'])))
        if d['any'] is not None and (d['with_slacks'] is None or
                                     rank_key(d['any']) != rank_key(d['with_slacks'])):
            L.append('* %s m: any slacks %s' % (lab, fmt(d['any'])))
    L.append('')
    return '\n'.join(L)


# ------------------------------------------------------------- select
def sel_cfg(d, T_xi=None, dbar=DBAR):
    """certify_profile configuration of a frontier row."""
    return cp.config(**dict(FIXED_C, c0=tuple(float(x) for x in d['c0']),
                            lam=float(d['lam']), alpha=float(d['alpha']),
                            eps_e=float(d['eps_e']), eps_v=float(d['eps_v']),
                            a_d=float(d['a_d']), vhnd=float(d['vhnd']),
                            T_xi=float(d['T_xi'] if T_xi is None else T_xi),
                            dbar=float(dbar)))


def run_cfg(ccfg, name='caseC'):
    """run_caseC design dictionary (PROVISIONAL keys) of a certificate
    configuration."""
    import run_caseC as rcc
    cfg = dict(rcc.PROVISIONAL, name=name)
    cfg.update(dict(
        v_xi=float(ccfg['v_xi']), a_b=float(ccfg['a_b']), T_xi=float(ccfg['T_xi']),
        t_d=float(ccfg['t_d']), a_d=float(ccfg['a_d']), v_c1=float(ccfg['v_c1']),
        alpha=float(ccfg['alpha']), lam=float(ccfg['lam']), phi=float(ccfg['phi']),
        eps_det=float(ccfg['eps_det']), eps_e=float(ccfg['eps_e']),
        eps_v=float(ccfg['eps_v']), dbar=float(ccfg['dbar']), n=int(ccfg['n']),
        s_m=float(ccfg['s_m']), amax=[float(x) for x in ccfg['amax']],
        d_s=[float(x) for x in ccfg['d_s']], c0=[float(x) for x in ccfg['c0']],
        eps0=[float(x) for x in ccfg['eps0']], e0_head=float(ccfg['e0_head']),
        vhnd=float(ccfg['vhnd']), d_req=float(ccfg['sync_tol']),
        refine=bool(ccfg['refine']), T_END=None))
    return cfg


def cond_boundary_ms(cfg, kmax=450):
    """Per condition the largest integer k with the condition holding at
    dbar = k ms (bisection on the 1 ms grid; neighbours verified)."""
    memo = {}

    def ev(k):
        if k not in memo:
            memo[k] = cp.evaluate(dict(cfg, dbar=k * 1e-3))
        return memo[k]

    def ok(nm, k):
        r = ev(k)
        return bool(r['closed'] and r['conds'][nm]['ok'])
    out = {}
    for nm in COND:
        if not ok(nm, 1):
            out[nm] = dict(k_ms=0, note='fails at 1 ms'); continue
        if ok(nm, kmax):
            out[nm] = dict(k_ms=kmax, note='holds at kmax'); continue
        lo, hi = 1, kmax
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(nm, mid):
                lo = mid
            else:
                hi = mid
        out[nm] = dict(k_ms=lo, verified=bool(ok(nm, lo) and not ok(nm, lo + 1)),
                       slack_at_k=ev(lo)['conds'][nm]['slack'],
                       slack_at_k1=ev(lo + 1)['conds'][nm]['slack'])
    return out


def refine_txi(d, want_boundary=21, tmax_extra=6):
    """The 1 s handoff refinement of a frontier row: the smallest integer
    T_xi >= T_xi_star(d) such that the certificate is admitted at 20 ms
    and the whole-certificate 1 ms boundary is at least want_boundary ms
    (a latency reserve of 1 ms above the contract); the table of every
    integer tried is returned."""
    t0 = txi_star(d)
    table = []
    chosen = None
    for T in range(t0, t0 + tmax_extra + 1):
        cfg = sel_cfg(d, T_xi=T)
        ev = cp.evaluate(cfg)
        bd = cp.age_boundary_ms(cfg, kmax=120) if ev['admitted'] else None
        row = dict(T_xi=T, admitted=ev['admitted'], failed=ev['failed'],
                   H3_slack=ev['conds']['H3']['slack'],
                   C_hnd_slack=ev['conds']['C_hnd']['slack'],
                   T_f_bar=ev['consts']['T_f_bar'] if ev['closed'] else None,
                   boundary_ms=None if bd is None else bd['k_ms'],
                   boundary_fails_next=None if bd is None else bd['failed_next'])
        table.append(row)
        print('  T_xi %d: admitted %s H3 %.3f boundary %s (fails %s)'
              % (T, ev['admitted'], row['H3_slack'] or -1, row['boundary_ms'],
                 row['boundary_fails_next']), flush=True)
        if ev['admitted'] and bd['k_ms'] >= want_boundary and chosen is None:
            chosen = T
            break
    if chosen is None:
        chosen = next((r['T_xi'] for r in table if r['admitted']), t0)
    return chosen, table


def full_record(d, T_xi):
    cfg = sel_cfg(d, T_xi=T_xi)
    t0 = time.time()
    ev = cp.evaluate(cfg)
    assert ev['admitted'], ev['failed']
    # bisection crossings of every condition, the premise P_seg included
    # (its slack min{t_d - T_sw_bar - (n-1) dbar, T_xi - t_c - (n-1) dbar}
    # is nonincreasing in dbar as well)
    ce = cp.age_ceilings(cfg, names=list(cp.COND_NAMES), verbose=False)
    bd = cp.age_boundary_ms(cfg, kmax=450)
    cb = cond_boundary_ms(cfg)
    enc = ip.check_enclosure(cfg)
    enc_k = ip.check_enclosure(dict(cfg, dbar=bd['k_ms'] * 1e-3)) \
        if bd['k_ms'] > 0 else None
    rec = dict(
        cert_cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        certificate=ev, ceilings=ce, dcert_num=min(ce.values()),
        binding=min(ce, key=ce.get), boundary_ms=bd, cond_boundary_ms=cb,
        interval_at_20ms=dict(verified=enc['enclosure']['verified'],
                              conds=enc['enclosure']['conds'],
                              values=enc['enclosure'].get('values'),
                              fp_inside=enc['all_inside']),
        interval_at_boundary=None if enc_k is None else dict(
            dbar=bd['k_ms'] * 1e-3, verified=enc_k['enclosure']['verified'],
            conds=enc_k['enclosure']['conds'], fp_inside=enc_k['all_inside']),
        runtime_s=time.time() - t0)
    print('  record: admitted %s, T_f_bar %.3f, boundary %d ms (fails %s), '
          'ceilings min %s %.4f, interval verified %s inside %s (%.0f s)'
          % (ev['admitted'], ev['consts']['T_f_bar'], bd['k_ms'], bd['failed_next'],
             rec['binding'], rec['dcert_num'], enc['enclosure']['verified'],
             enc['all_inside'], rec['runtime_s']), flush=True)
    return rec


def part_select():
    F = load('frontier.json')
    if F is None:
        F = part_frontier()
    S = F['selection']
    d = S['selected']
    if d is None:
        raise SystemExit('no admitted design')
    print('[select] %s' % fmt(d), flush=True)
    T_xi, table = refine_txi(d)
    rec = full_record(d, T_xi)
    cfg = run_cfg(rec['cert_cfg'])
    # the runners-up: at every clearance of the frontier the best design
    # with the slacks (or beyond the grid) and the best design without the
    # slack rule, each evaluated at its own 1 s refinement (evaluate at
    # 20 ms, 1 ms boundary; the handoff is raised by whole seconds, at
    # most six, until the design is admitted at 21 ms)
    runners = {}
    for lab in c0_order(list(F['per_c0'])):
        b = dict(S['per_c0_best'].get(lab, {}))
        p = F['per_c0'].get(lab, {})
        if b.get('with_slacks') is None and p.get('best_with_slacks_beyond_grid'):
            b['with_slacks_beyond_grid'] = p['best_with_slacks_beyond_grid']
        for kind in ('with_slacks', 'with_slacks_beyond_grid', 'any'):
            r = b.get(kind)
            if r is None or (kind == 'any' and b.get('with_slacks') is not None
                             and rank_key(r) == rank_key(b['with_slacks'])):
                continue
            if rank_key(r) == rank_key(d):
                runners['%s|%s' % (lab, kind)] = dict(row=r, same_as_selected=True)
                continue
            T, tab = refine_txi(r)
            cfg_r = sel_cfg(r, T_xi=T)
            ev_r = cp.evaluate(cfg_r)
            bd_r = cp.age_boundary_ms(cfg_r, kmax=120)
            runners['%s|%s' % (lab, kind)] = dict(
                row=r, T_xi=T, refinement=tab, admitted=ev_r['admitted'],
                T_ent_bar=ev_r['consts']['T_ent_bar'], T_f_bar=ev_r['consts']['T_f_bar'],
                slack={nm: ev_r['conds'][nm]['slack'] for nm in COND},
                boundary_ms=bd_r['k_ms'], boundary_fails_next=bd_r['failed_next'])
            print('  runner-up %s %s: T_xi %d, T_f_bar %.2f, boundary %d ms'
                  % (lab, kind, T, ev_r['consts']['T_f_bar'], bd_r['k_ms']), flush=True)
    out = dict(cfg=cfg, selected=dict(
        c0=cfg['c0'], lam=cfg['lam'], alpha=cfg['alpha'], eps_e=cfg['eps_e'],
        eps_v=cfg['eps_v'], vhnd=cfg['vhnd'], a_d=cfg['a_d'], t_d=cfg['t_d'],
        v_c1=cfg['v_c1'], T_xi=cfg['T_xi'], dbar=cfg['dbar'], d_req=cfg['d_req'],
        t_c=rec['certificate']['case']['t_c'],
        tau_r=rec['certificate']['reference']['tau_r']),
        selection_rule=S['rule'], tier_used=S['tier_used'], tiers=S['tiers'],
        scan_row=d, handoff_refinement=dict(T_xi_star=txi_star(d), chosen=T_xi,
                                            reserve_s=H3_RESERVE,
                                            want_boundary_ms=21, table=table),
        ranked_tier_A=S['ranked_tier_A'], runners_up=runners,
        frontier_per_c0={lab: dict(n_admitted=p['n_admitted'],
                                   n_with_slacks=p['n_admitted_with_slacks'],
                                   min_T_f_bar_any=p['min_T_f_bar_any'],
                                   min_T_f_bar_star_any=p['min_T_f_bar_star_any'],
                                   min_T_f_bar_with_slacks=p['min_T_f_bar_with_slacks'],
                                   min_T_f_bar_star_with_slacks=p['min_T_f_bar_star_with_slacks'],
                                   best_with_slacks=p['best_with_slacks'],
                                   best_any=p['best_any'])
                         for lab, p in F['per_c0'].items()},
        caseB_slacks_20ms=cp.caseB_slacks())
    out.update(rec)
    dump('design.json', out)
    print('  -> design.json: %s' % json.dumps(out['selected']), flush=True)
    return out


PARTS = dict(scan=part_scan, scan_low=part_scan_low, compact=part_compact,
             frontier=part_frontier, select=part_select)

if __name__ == '__main__':
    args = sys.argv[1:] or ['all']
    os.makedirs(OUT, exist_ok=True)
    for a in args:
        if a == 'all':
            for p in ('compact', 'frontier', 'select'):
                PARTS[p]()
        elif a in PARTS:
            PARTS[a]()
        else:
            raise SystemExit('unknown part %s' % a)
