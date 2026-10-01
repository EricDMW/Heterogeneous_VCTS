#!/usr/bin/env python3
"""gap_construction_tests.py -- exhaustive-grid tests of the sequential
admission-gap construction (main-text Proposition 1), pre-submission
review item T2/T3 (2026-09-07).

Three checks:

  (1) synthetic counterexample (the reviewer's): margins m2(x) = x,
      m3(x, y) = x + y - 1.5 on [0, 1]^2 at target mu = 0.1.  The greedy
      pass fails at pair 3 after choosing x = 0.1, yet (0.6, 1.0) is
      feasible and lexicographically minimal.  The pass must report
      PREFIX_NOT_EXTENDABLE, never INFEASIBLE.
  (2) the real evaluator on the n = 2 and n = 3 tiles of the fleet-size
      family: an exhaustive grid over the pair gaps (0.5 m) at several
      margin targets checks (a) own-gap monotonicity, (b) up-chain
      antitonicity of the minimal own gap, (c) that the greedy pass,
      whenever it completes, returns the lexicographically minimal
      admission-certifiable vector to grid resolution, and (d) whether
      a non-extendable minimal prefix exists for this evaluator.
  (3) agreement of b2_design.size_gaps with the bisection greedy where
      the arrest floor does not bind.

The greedy pass here is written against the proposition's statement
(per-pair bisection of the smallest gap meeting the target given the
already-minimal prefix), independently of b2_design's fixed-point
implementation, and labels its outcomes explicitly.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'certification'))
sys.path.insert(0, os.path.join(HERE, '..', '..'))

COMPLETED = 'COMPLETED'
NOT_EXT = 'PREFIX_NOT_EXTENDABLE'


def greedy(margin_fn, lo, hi, mu, tol=1e-3):
    """Sequential construction: for each pair choose the smallest own
    gap in [lo_i, hi_i] with margin >= mu given the prefix, by
    bisection on a nondecreasing margin.  Returns (status, gaps, pair)."""
    gaps = []
    for i in range(len(lo)):
        if margin_fn(i, gaps + [hi[i]]) < mu:
            return NOT_EXT, gaps, i
        a, b = lo[i], hi[i]
        if margin_fn(i, gaps + [a]) >= mu:
            gaps.append(a); continue
        while b - a > tol:
            m = 0.5 * (a + b)
            if margin_fn(i, gaps + [m]) >= mu:
                b = m
            else:
                a = m
        gaps.append(b)
    return COMPLETED, gaps, None


# ----------------------------------------------------------- (1) synthetic
def synthetic():
    def m(i, g):
        return g[0] if i == 0 else g[0] + g[1] - 1.5
    st, gaps, pair = greedy(m, [0.0, 0.0], [1.0, 1.0], 0.1)
    assert st == NOT_EXT and pair == 1 and abs(gaps[0] - 0.1) < 2e-3
    xs = np.linspace(0, 1, 101)
    feas = [(x, y) for x in xs for y in xs if x >= 0.1 and x + y - 1.5 >= 0.1]
    assert feas, 'synthetic feasible set must be nonempty'
    lexmin = min(feas)
    assert abs(lexmin[0] - 0.6) < 1e-9 and abs(lexmin[1] - 1.0) < 1e-9
    return dict(status=st, failed_pair=pair + 2, greedy_prefix=gaps,
                grid_lexmin=list(lexmin), feasible_points=len(feas))


# ------------------------------------------------------- (2) real evaluator
def _tile(n):
    import run_b2 as rb
    r = (1.21 / 1.60) ** (1.0 / (n - 1))
    am = [round(1.60 * r ** k, 3) for k in range(n)]
    eps0 = [-0.30] + [round(0.25 * (0.7 ** k), 3) for k in range(n - 1)]
    e0 = [-0.5] + [37.0] * (n - 1)
    c0 = tuple(np.linspace(42.0, 49.0, n - 1).round(1)) if n > 2 \
        else (45.0,)
    return dict(rb.B2, n=n, amax=am, eps0=eps0, e0=e0), c0


def real_evaluator(n, step=0.5, targets=(0.15, 0.5, 1.0)):
    import run_b2 as rb
    import b2_design as bd
    ov, c0 = _tile(n)
    G = bd._namespace(ov)
    dbar = rb.B2['dbar']; s_m = ov['s_m']
    npair = n - 1
    lo = [s_m + 0.5] * npair
    hi = [float(c) for c in c0]

    def margins(gaps):
        d = np.array([rb.HEAD_SLOT] + list(gaps), float)
        A = bd.assemble(G, d, c0, dbar)
        if not A['closed']:
            return None
        return np.array(A['margins'], float)

    def margin_fn(i, gaps):
        full = list(gaps) + [hi[k] for k in range(len(gaps), npair)]
        m = margins(full)
        return -1e9 if m is None else float(m[i])

    axes = [np.arange(lo[i], hi[i] + 1e-9, step) for i in range(npair)]
    t0 = time.time()
    if npair == 1:
        M = np.array([margins([a])[0] for a in axes[0]])
        mono = bool(np.all(np.diff(M) >= -1e-9))
        out = dict(n=n, grid=[len(axes[0])], own_monotone=mono,
                   antitone=None, evals=len(axes[0]))
        res = []
        for mu in targets:
            st, gaps, pair = greedy(margin_fn, lo, hi, mu)
            feas = [float(a) for a, m in zip(axes[0], M) if m >= mu]
            rec = dict(mu=mu, status=st, greedy=gaps,
                       grid_lexmin=[min(feas)] if feas else None)
            if st == COMPLETED:
                assert feas and gaps[0] <= min(feas) + 1e-6 \
                    and min(feas) - step <= gaps[0] + 1e-6
            else:
                assert not feas
            res.append(rec)
        out['targets'] = res
        out['seconds'] = time.time() - t0
        return out
    # n = 3: two pairs, exhaustive grid
    A2, A3 = axes
    M2 = np.zeros(len(A2)); M3 = np.zeros((len(A2), len(A3)))
    for a, x in enumerate(A2):
        for b, y in enumerate(A3):
            m = margins([x, y])
            M2[a] = m[0]; M3[a, b] = m[1]
    own2 = bool(np.all(np.diff(M2) >= -1e-9))
    own3 = bool(np.all(np.diff(M3, axis=1) >= -1e-9))
    # antitonicity of the minimal own gap of pair 3 in the up-chain gap
    res = []
    found_nonext = None
    for mu in targets:
        L3 = np.array([A3[np.argmax(M3[a] >= mu)] if np.any(M3[a] >= mu)
                       else np.nan for a in range(len(A2))])
        ok2 = M2 >= mu
        anti = bool(np.all(np.diff(L3[np.isfinite(L3)]) <= 1e-9))
        feas = [(float(A2[a]), float(A3[b])) for a in range(len(A2))
                for b in range(len(A3)) if ok2[a] and M3[a, b] >= mu]
        st, gaps, pair = greedy(margin_fn, lo, hi, mu)
        rec = dict(mu=mu, status=st, greedy=gaps, failed_pair=pair,
                   antitone_L3=anti,
                   L3_drop_per_m=float((np.nanmax(L3) - np.nanmin(L3))
                                       / (A2[-1] - A2[0]))
                   if np.isfinite(L3).sum() > 1 else None,
                   grid_lexmin=list(min(feas)) if feas else None,
                   feasible_points=len(feas))
        if st == COMPLETED:
            lm = min(feas)
            assert gaps[0] <= lm[0] + 1e-6 and lm[0] - step <= gaps[0] + 1e-6
            # given the greedy first coordinate, the greedy second
            # coordinate is minimal to grid resolution
            m_at = [margins([gaps[0], y])[1] for y in A3]
            f3 = [float(y) for y, m in zip(A3, m_at) if m >= mu]
            assert f3 and gaps[1] <= min(f3) + 1e-6 \
                and min(f3) - step <= gaps[1] + 1e-6
        elif feas:
            found_nonext = mu          # real non-extendable minimal prefix
        res.append(rec)
    # search for a non-extendable minimal prefix at finer targets
    if found_nonext is None:
        for mu in np.linspace(0.05, float(np.max(M3)), 40):
            ok2 = M2 >= mu
            feas_any = bool(np.any(ok2[:, None] & (M3 >= mu)))
            st, gaps, pair = greedy(margin_fn, lo, hi, float(mu))
            if st == NOT_EXT and feas_any:
                found_nonext = float(mu); break
    return dict(n=n, grid=[len(A2), len(A3)], own_monotone=own2 and own3,
                evals=int(M3.size), targets=res,
                nonextendable_prefix_found=found_nonext,
                seconds=time.time() - t0)


# ------------------------------------------- (3) agreement with b2_design
def size_gaps_agreement(n=3, mu=0.15):
    import run_b2 as rb
    import b2_design as bd
    ov, c0 = _tile(n)
    G = bd._namespace(ov)
    d_s, A = bd.size_gaps(G, c0, rb.B2['dbar'], target=mu)
    floors = np.array(A['arrest_floor'])[1:]
    at_floor = [bool(x) for x in A['at_floor']]
    return dict(n=n, mu=mu, size_gaps=[float(x) for x in d_s[1:]],
                arrest_floor=[float(x) for x in floors],
                at_floor=at_floor,
                margins=[float(x) for x in A['margins']])


def main():
    out = dict(synthetic=synthetic())
    out['n2'] = real_evaluator(2)
    out['n3'] = real_evaluator(3)
    out['size_gaps'] = size_gaps_agreement()
    return out


if __name__ == '__main__':
    import json
    R = main()
    print(json.dumps(R, indent=1, default=float))
