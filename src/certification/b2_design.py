#!/usr/bin/env python3
"""b2_design.py -- benchmark-B2 certificate driver (root revision).

Runs the FROZEN evaluator machinery (via certify_root2's source
loader) with the B2 design constants, but performs its own geometry
assembly so that the parking gaps can be sized PER PAIR by inverting
the admission inequality: admission margin is affine (slope one) in
d_s[i] at fixed closure error, so the minimal certifiable gap is
well defined; a short fixed-point iteration absorbs the weak budget
feedback through the closure error e0 = c0 - d_s.

Everything numerical reported for B2 comes from this driver (class C/D
evidence in the paper's taxonomy); the featured Case-A chain remains
the frozen digest-pinned stack.
"""
import io
import contextlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_root2 as cr

ENTRY_START = '# R3-2: CERTIFIED entry deadlines via the tail-variation'
ENTRY_END = 'Tent_cert, Vhnd, _Vfns = certified_entry(R, dbar)'


def _namespace(overrides):
    """Exec the frozen defs prefix + the certified-entry block with the
    given constant overrides; no featured assembly runs."""
    full = cr._source(overrides, mode='full')
    quick = cr._source(overrides, mode='quick')
    # defs prefix = everything before the featured assembly line
    cut = quick.find('\ndbar = ')
    prefix = quick[:cut + 1]
    i0 = full.find(ENTRY_START)
    i1 = full.find(ENTRY_END)
    assert i0 > 0 and i1 > i0, 'entry-block markers not found'
    entry_block = full[i0:i1]
    G = {'__name__': 'b2_design_run'}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(prefix, 'certify_v30.py[b2-defs]', 'exec'), G)
        exec(compile(entry_block, 'certify_v30.py[b2-entry]', 'exec'), G)
    return G


def assemble(G, d_s, c0, dbar):
    """Geometry + full condition evaluation at per-pair gaps d_s
    (index 0 = head slot) and physical initial gaps c0 (pairs 2..n)."""
    n = G['n']; s_m = G['s_m']; v_xi = G['v_xi']
    eps0 = G['eps0']; b = G['b']
    e0 = np.array([G['e0'][0]] + [c0[i - 1] - d_s[i]
                                  for i in range(1, n)])
    G['e0'][:] = e0                       # budgets read the global
    v = np.zeros(n); vp = v_xi
    for i in range(n):
        v[i] = vp - eps0[i]; vp = v[i]
    g_init = np.array([0.0] + [c0[i - 1] - s_m for i in range(1, n)])
    h0 = np.array([0.0] + [g_init[i] + v[i - 1] / b[i - 1]
                           - v[i] / b[i] for i in range(1, n)])
    R = G['forward_pass'](dbar)
    out = dict(R=R, e0=e0, v0=v, g_init=g_init, h0=h0, d_s=d_s)
    if R is None:
        out['closed'] = False
        return out
    out['closed'] = True
    NUMPAD = G['NUMPAD']
    out['margins'] = np.array([h0[i] - R['pairs'][i]['thr'] - NUMPAD
                               for i in range(1, n)])
    rows, gmin = G['floors'](R, g_init, h0)
    out['rows'] = rows; out['gmin'] = float(gmin)
    out['auth_ok'] = bool(R['auth_ok'])
    out['auth_slack'] = [float(G['Umin'][i] - R['dem'][i])
                         for i in range(n)]
    out['event_ok'] = bool(R['event_ok'])
    out['vfloor'] = float(R['vfloor'])
    out['ramp_ok'] = (n - 1) * dbar < R['vfloor'] / float(
        np.max(G['Umin']))
    for k in ('disp', 'late', 'sched', 'align', 'T_f', 'T_brk'):
        out[k] = float(R[k])
    out['gaperr'] = np.array(R['gaperr'], float)
    return out


def entry_certify(G, R, dbar):
    Tent_c, Vhnd, _ = G['certified_entry'](R, dbar)
    Tent_bar = float(np.max(R['tset'] + Tent_c))
    return dict(Tent_cert=[float(x) for x in Tent_c],
                Tent_bar=Tent_bar, T_hold=Tent_bar + dbar,
                hold_ok=Tent_bar + dbar <= G['T_xi'],
                Vhnd_max=float(np.max(Vhnd)),
                hnd_ok=float(np.max(Vhnd)) <= G['VHND_PAD'])


def size_gaps(G, c0, dbar, target=0.15, floor_pad=0.0, iters=8):
    """Per-pair minimal certifiable gaps: smallest d_s with padded
    admission margin >= target, floored by the Corollary-1 arrest
    bound.  Fixed point over the closure-error feedback."""
    n = G['n']; s_m = G['s_m']; v_xi = G['v_xi']
    amax = G['amax']; head_slot = 6.0
    arrest = np.array([0.0] + [
        s_m + 0.5 * v_xi ** 2 * max(1 / amax[i] - 1 / amax[i - 1], 0.0)
        + v_xi * dbar + floor_pad for i in range(1, n)])
    d_s = np.maximum(arrest, s_m + 1.0)
    d_s[0] = head_slot
    hist = []
    for it in range(iters):
        A = assemble(G, d_s.copy(), c0, dbar)
        need = target - A['margins']          # slope-one in d_s
        newd = d_s.copy()
        newd[1:] = np.maximum(arrest[1:], d_s[1:] + need)
        hist.append((d_s.copy(), A['margins'].copy()))
        if np.max(np.abs(newd - d_s)) < 5e-3:
            d_s = newd
            break
        d_s = newd
    A = assemble(G, d_s.copy(), c0, dbar)
    A['arrest_floor'] = arrest
    A['at_floor'] = [bool(d_s[i] <= arrest[i] + 1e-6)
                     for i in range(1, len(d_s))]
    return d_s, A
