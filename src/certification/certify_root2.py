#!/usr/bin/env python3
"""certify_root2.py -- parameterized driver over the FROZEN dispatch
evaluator certify_v30.py (root revision, benchmark B2 program).

The frozen evaluator must never be edited.  This driver loads its
byte-identical source, substitutes ONLY the featured design constants
(the algorithm, every budget formula, and the assembly code are the
frozen bytes), and executes it in a fresh namespace:

  * mode='quick' truncates the source before the ceiling-bisection
    section (marker line) -- returning the forward pass R, admission
    margins, floors, authority, event and velocity-floor checks plus
    the grid entry estimate; used by the design search.
  * mode='full' runs the entire evaluator (ceilings, certified entry,
    emergency study) and returns the machine-readable `vals` dict the
    frozen code emits (v30_values.json written to cwd).

Validation: evaluate() with no overrides reproduces the archived
register v30_values.json exactly (checked by validate_featured()).
"""
import io
import json
import os
import re
import contextlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FROZEN = os.path.join(HERE, 'certify_v30.py')
QUICK_MARK = ('# full ceiling set (geometry frozen; bisection on '
              'monotone feasible sets)')

# exact assignment lines of the frozen parameter block -> template
_SUBS = {
    'n':     (r'n       = 5', 'n       = {v}'),
    'T_c':   (r'T_c     = 0\.001', 'T_c     = {v}'),
    'alpha': (r'alpha_g = np\.full\(n, 0\.30\)',
              'alpha_g = np.full(n, {v})'),
    'beta':  (r'beta    = 0\.005625', 'beta    = {v}'),
    'gamma': (r'gamma   = 0\.15', 'gamma   = {v}'),
    'lam':   (r'lam     = 0\.075', 'lam     = {v}'),
    'a_b':   (r'a_b     = 0\.70', 'a_b     = {v}'),
    'v_xi':  (r'v_xi    = 10\.0', 'v_xi    = {v}'),
    'eps_e': (r'eps_e   = 0\.20', 'eps_e   = {v}'),
    'eps_v': (r'eps_v   = 0\.02', 'eps_v   = {v}'),
    's_m':   (r's_m     = 3\.0', 's_m     = {v}'),
    'g_star': (r'g_star  = 4\.2', 'g_star  = {v}'),
    'T_xi':  (r'T_xi    = 180\.0', 'T_xi    = {v}'),
    'ALIGN_TOL': (r'ALIGN_TOL = 3\.0', 'ALIGN_TOL = {v}'),
    'amax':  (r'amax = np\.array\(\[1\.60, 1\.48, 1\.38, 1\.29, 1\.21\]\)',
              'amax = np.array({v})'),
    'e0':    (r'e0   = np\.array\(\[-0\.5, 40\.0, 40\.0, 40\.0, 40\.0\]\)',
              'e0   = np.array({v})'),
    'eps0':  (r'eps0 = np\.array\(\[-0\.05, 0\.025, 0\.025, 0\.025, '
              r'0\.025\]\)',
              'eps0 = np.array({v})'),
    'dbar':  (r'\ndbar = 0\.06\n', '\ndbar = {v}\n'),
    # detector band and boundary layer (message-period / detector
    # study, root revision): exact frozen assignment lines
    'eps_det': (r'eps_det = 0\.015', 'eps_det = {v}'),
    'phi':   (r'phi     = 0\.005', 'phi     = {v}'),
}


def _source(overrides, mode):
    src = open(FROZEN, encoding='utf-8').read()
    for key, val in overrides.items():
        pat, tpl = _SUBS[key]
        if isinstance(val, (list, tuple, np.ndarray)):
            rep = tpl.format(v=repr(list(val)))
        elif isinstance(val, int):
            rep = tpl.format(v=repr(val))
        else:
            rep = tpl.format(v=repr(float(val)))
        new, cnt = re.subn(pat, rep.replace('\\', r'\\'), src, count=1)
        assert cnt == 1, 'substitution failed for %s' % key
        src = new
    if mode == 'quick':
        idx = src.find(QUICK_MARK)
        assert idx > 0, 'quick-mode marker not found'
        src = src[:idx]
    return src


def evaluate(overrides=None, mode='full', quiet=True, workdir=None):
    """Run the frozen evaluator with substituted constants.

    Returns the exec namespace G; in full mode G['vals'] is the
    machine-readable certificate; in quick mode use G['R'],
    G['h0'], G['g_init'], G['rows'], G['gmin'], G['Tent_bar'], etc.
    """
    src = _source(overrides or {}, mode)
    G = {'__name__': 'certify_root2_run'}
    cwd = os.getcwd()
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
    try:
        buf = io.StringIO()
        ctx = contextlib.redirect_stdout(buf) if quiet \
            else contextlib.nullcontext()
        with ctx:
            exec(compile(src, 'certify_v30.py[root2]', 'exec'), G)
        G['_stdout'] = buf.getvalue() if quiet else ''
    finally:
        os.chdir(cwd)
    return G


def quick_summary(G):
    """Compact design-search summary from a quick-mode namespace."""
    R, h0, n = G['R'], G['h0'], G['n']
    NUMPAD = G['NUMPAD']
    margins = [h0[i] - R['pairs'][i]['thr'] - NUMPAD for i in range(1, n)]
    rows, gmin = G['floors'](R, G['g_init'], h0)
    auth = [float(R['dem'][i]) - float(G['Umin'][i]) for i in range(n)]
    return dict(
        d_s=[float(x) for x in G['d_s'][1:]],
        margins=[float(m) for m in margins],
        adm_ok=min(margins) > 0,
        gmin=float(gmin), clr_ok=gmin > 0,
        auth_slack=[-a for a in auth], auth_ok=bool(R['auth_ok']),
        event_ok=bool(R['event_ok']), vfloor=float(R['vfloor']),
        disp=float(R['disp']), late=float(R['late']),
        sched=float(R['sched']), align=float(R['align']),
        gaperr=float(np.max(R['gaperr'])),
        T_f=float(R['T_f']), Tent_bar_grid=float(G['Tent_bar']),
        h0=[float(x) for x in h0[1:]],
        anet=[float(x) for x in (G['a_b'] - R['Fb'][1:n])])


def validate_featured():
    ref = json.load(open(os.path.join(HERE, '..', '..', 'data',
                                      'v30_values.json')))
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for f in ('v30_values.json',):
            pass
        G = evaluate({}, mode='full', workdir=td)
        new = json.loads(json.dumps(G['vals']))
    bad = []

    def cmp(a, b, path):
        if isinstance(a, dict):
            for k in a:
                cmp(a[k], b.get(k, None), path + '/' + str(k))
        elif isinstance(a, list):
            for j, (x, y) in enumerate(zip(a, b)):
                cmp(x, y, path + '[%d]' % j)
        elif isinstance(a, float) and isinstance(b, float):
            if not (abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))):
                bad.append((path, a, b))
        elif a != b:
            bad.append((path, a, b))
    cmp(ref, new, '')
    return bad


if __name__ == '__main__':
    bad = validate_featured()
    if bad:
        print('VALIDATION FAILED (%d fields):' % len(bad))
        for p, a, b in bad[:20]:
            print(' ', p, a, b)
    else:
        print('certify_root2: featured reproduction EXACT '
              '(all register fields match)')
