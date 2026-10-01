#!/usr/bin/env python3
"""certify_plan.py -- dispatch-certificate evaluator of Case D
(deceleration-only arrival with planned references, Assumption 1'',
Proposition D1 of data/new_d19/caseD/theory_caseD.md, 2026-09-27).

Every unit follows its own planned reference (s*_i, v*_i, a*_i); the pair
coordinates are taken relative to the plan and the laws of the main text
act on the deviation commands u~_i = u_i - a*_i.  The certificate is the
conjunction of the conditions of Section 7.3 of the theory note:

  P1       initial speed 0 < v^xi <= v_1(0)            (slack in m/s)
  P2       planned rates -k <= a*_i <= -a_min < 0     (slack in m/s^2)
  P3       common crawl before T_xi, exact closure    (slack in s)
  P4       positive crawl speed v_c1 > 0              (slack in m/s)
           ((P5) is reported with P3; one row per premise, so that no
           slack mixes units)
  P_S1    braking S1 stages: eps_k(0) + Lambda1_k <= 0 (one-sided box)
  H7       sampled event (frozen)
  H3D      hold: T_ent_bar + (n-1) dbar <= T_xi
  H6D      velocity floor v*_i(t_{j+1}) - sum_{k<=i} P_k(t_j) > 0
  C_hnd    handoff pad (frozen)
  H2       headroom a_b - F^b_{i-1} > 0
  H1D_i    authority, upper:  F^+_i(t_P) <= a_P on every piece
  H1D_ii   authority, lower:  a_P + F^-_i(t_P) <= k on every piece
  H1D_iii  authority, braking: a_b + F^b_i <= k and F^b_i <= a_b
  H4D_i/ii/iii  admission with the planned CBF residual mu*_i
  H5D_i    clearance floor g_min^(i) > 0 (direct identity g = g* + e)
  H5D_ii   barrier floor in acquisition
  C_sync   settling time sum_k q_k <= 1.0 s
  C_al     marker bound <= 3 m
  C_gap    terminal gap bound <= 1 m

Implementation (Section 7.2 of the theory note):
  Step A   acquisition chain: a statement-for-statement copy of the frozen
           forward pass of certify_v30.py, executed on the frozen namespace
           (b2_design._namespace, constant overrides only) with the frozen
           helpers corner_stats, D_quad, DB_quad;
  Step B   braking recursion without the ramp charge a_b dbar in W_i;
           ramp_charge=True restores the frozen W_i, and then Steps A and B
           reproduce every field of the frozen forward_pass bit for bit
           (regress());
  Step C   entry recursion: certify_profile._entry with ff=None and the
           frozen lag grids, bit-identical to the frozen certified_entry;
  Step D   plan: exact rational arithmetic (fractions.Fraction) for the
           planned states, breakpoints, closure residuals, markers and the
           piecewise-quadratic minima of g*_i, h*_i and mu*_i;
  Step E   envelopes F^+-_i at the left endpoint of every piece, the
           velocity-floor envelopes P_k on the 1 ms grid, by the closed-form
           signed tail supremum S(p, q; R) (certify_profile._tail_sup_signed).

The frozen evaluator certify_v30.py and certify_profile.py are imported
read-only and never edited.  The evaluation is floating point except the
plan quantities of Step D, which are exact rationals rounded to double.
interval_plan.py encloses the closed-form conditions with outward
intervals.

Reduction.  Case D has no exact reduction to Case A, B or C: the input set
is [-k, 0] (a constant reference violates (P2) and the braking-only set),
the ramp is local (W_i without a_b dbar, no V^rise, no (H8), received
braking envelope a_b + F^b_{i-1} only), the hold has (n-1) dbar, the
authority is sign- and time-resolved and the admission uses mu*_i.  The
regression therefore checks (regress()): (R1) Steps A-B with ramp_charge
reproduce the frozen forward pass bit for bit (Cases A, B and the Case D
constants); (R2) Step C reproduces the frozen certified_entry; (R3) with a
constant plan (a* = 0 before T_xi, premise checks bypassed) and the Case B
takeover state, h*(t) + h~(0) = h(0) and mu* = kappa h*, so the phase-1 and
phase-2 admission margins equal the frozen ones and the phase-3 margin
equals the frozen one up to the V^rise term and the received braking
envelope, which are reported; (R4) brute-force checks of every closed form
(plan_bruteforce()); (R5, round 5) the round-4 design of the canonical
family (data/new_d19/caseD_round4/design.json) is reproduced field by field
(regress_round4()); (R6, round 5) the fan family: exact closure, crawl-speed
and marker residuals, premises, decreasing main rates, joins, float closure,
and the crawl rate and completion bound of an independent construction
(regress_fan()).

Plan families (plan()): 'canonical' (round 4, staggered onsets at a common
main rate), 'fan' (round 5: common onset, unit-specific main rates gentler
along the train, joins onto a common crawl; round 6: optionally with
ramped main phases, a symmetric staircase of the planned deceleration on a
planning grid, regressions R7 and R8) and 'constant' (regression).

API
  config(**kw)            Case D constants updated by kw
  plan(cfg)               Plan object (exact planned references, offsets)
  evaluate(cfg)           dict(conds, consts, admitted, failed, plan, ...)
  age_boundary_ms(cfg)    largest admitted age bound on the 1 ms grid
  cond_boundaries_ms(cfg) per-condition boundaries on the 1 ms grid
  regress()               (R1)-(R3)
  plan_bruteforce(cfg)    (R4)
"""
import json
import math
import os
import sys
import time
from collections import OrderedDict
from fractions import Fraction as Fr

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2_design as bd                                   # noqa: E402
import certify_profile as cp                             # noqa: E402

CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_D = os.path.join(CODE, 'data', 'new_d19', 'caseD')

# frozen constants of certify_v30.py without an override hook
FIXED = dict(U=1.6, ell=25.0, kappa=1.0, v_c=1.2)

CASE_D = dict(
    n=5, amax=[1.60, 1.48, 1.38, 1.29, 1.21], k=1.2, a_b=1.0,
    alpha=0.35, lam=0.2, s_m=1.0, eps_det=0.015, phi=0.005, T_c=0.001,
    dbar=0.020, eps_e=0.15, eps_v=0.02, vhnd=1e-2,
    d_s=[6.0, 5.0, 5.0, 5.0, 5.0], c0=[50.0, 50.0, 50.0, 50.0],
    v0=[10.47, 10.55, 10.67, 10.85, 11.10],
    plan=dict(family='canonical', v_xi='10.45', a_c='0.03', a_d='1.15',
              t_d1='0.1', T_xi='29.5', v_c1='1.0'),
    sync_tol=1.0, align_tol=3.0, gap_tol=1.0, numpad=0.01,
    one_sided=True, windowed=False, ramp_charge=False, grid_dt=0.001)

COND_NAMES = ('P1', 'P2', 'P3', 'P4', 'P_S1', 'H7','H3D', 'H6D', 'C_hnd', 'H2',
              'H1D_i', 'H1D_ii', 'H1D_iii', 'H4D_i', 'H4D_ii', 'H4D_iii',
              'H5D_i', 'H5D_ii', 'C_sync', 'C_al', 'C_gap')


def config(**kw):
    """Case D constants updated by kw (plan entries merged)."""
    cfg = json.loads(json.dumps(CASE_D))
    pl = kw.pop('plan', None)
    cfg.update(kw)
    if pl is not None:
        if pl.get('family') == 'fan' and cfg['plan'].get('family') != 'fan':
            # a fan plan replaces the default (canonical) entries
            cfg['plan'] = dict(pl)
        else:
            cfg['plan'] = dict(cfg['plan'], **pl)
    return cfg


def F(x):
    """exact rational of a decimal input (str, int, float via repr)."""
    if isinstance(x, Fr):
        return x
    if isinstance(x, int):
        return Fr(x)
    return Fr(str(x))


# ===================================================================== plan
class Plan:
    """Planned references of Assumption 1'' with exact rational data.

    units[i] = [(t0, a), ...]: the pieces of a*_i on [0, T_xi) (t0
    increasing, first t0 = 0); from T_xi every plan brakes at -a_b and is
    continued beyond its planned stop.  States are exact rationals."""

    def __init__(self, family, params, v_xi, s0, units, T_xi, a_b, d_s, ell,
                 C, extra=None):
        self.family = family
        self.params = params
        self.v_xi = v_xi
        self.s0 = list(s0)
        self.units = [list(u) for u in units]
        self.T_xi = T_xi
        self.a_b = a_b
        self.d_s = list(d_s)
        self.ell = ell
        self.C = list(C)
        self.n = len(units)
        self.extra = extra or {}
        self.segs = []
        for i in range(self.n):
            pcs = self.units[i] + [(T_xi, -a_b)]
            s, v = self.s0[i], v_xi
            segs = []
            for j, (t0, a) in enumerate(pcs):
                t1 = pcs[j + 1][0] if j + 1 < len(pcs) else None
                segs.append((t0, t1, a, s, v))
                if t1 is not None:
                    L = t1 - t0
                    assert L > 0, 'pieces must be strictly increasing'
                    s = s + v * L + a * L * L / 2
                    v = v + a * L
            self.segs.append(segs)
        self._fsegs = [np.array([[float(t0), float(a), float(s), float(v)]
                                 for (t0, t1, a, s, v) in segs])
                       for segs in self.segs]
        self.v_c1 = self.state(0, T_xi)[1]
        self.tau_star = T_xi + self.v_c1 / a_b
        Delta = []
        acc = Fr(0)
        for i in range(self.n):
            acc += self.d_s[i] + ell
            Delta.append(acc)
        self.Delta = Delta
        self.s_ref_stop = self.state(0, self.tau_star)[0] + Delta[0]
        self.markers = [self.s_ref_stop - Delta[i] for i in range(self.n)]

    # ---- exact evaluation
    def state(self, i, t):
        """exact (s*, v*, a*) of unit i at time t >= 0 (right-continuous)."""
        t = F(t) if not isinstance(t, Fr) else t
        for (t0, t1, a, s, v) in reversed(self.segs[i]):
            if t >= t0:
                tau = t - t0
                return s + v * tau + a * tau * tau / 2, v + a * tau, a
        raise ValueError('t < 0')

    def acc_pieces(self, i):
        """pieces of a*_i on [0, T_xi): list of (t0, t1, a) exact."""
        pcs = self.units[i]
        out = []
        for j, (t0, a) in enumerate(pcs):
            t1 = pcs[j + 1][0] if j + 1 < len(pcs) else self.T_xi
            out.append((t0, t1, a))
        return out

    def breakpoints(self, i):
        return [t0 for (t0, a) in self.units[i][1:]]

    # ---- float evaluation (vectorized)
    def state_f(self, i, t):
        t = np.asarray(t, float)
        S = self._fsegs[i]
        j = np.searchsorted(S[:, 0], t, side='right') - 1
        j = np.clip(j, 0, len(S) - 1)
        tau = t - S[j, 0]
        a = S[j, 1]
        return S[j, 2] + S[j, 3] * tau + 0.5 * a * tau * tau, S[j, 3] + a * tau, a

    # ---- premises
    def premises(self, k, v1_0):
        """(P1)-(P5), closure residuals and marker residuals (exact)."""
        n = self.n
        rep = {}
        rates = [a for i in range(n) for (t0, a) in self.units[i]]
        a_min = min(-a for a in rates)
        a_max = max(-a for a in rates)
        rep['a_min'] = a_min
        rep['a_max'] = a_max
        rep['P2'] = bool(a_min > 0 and a_max <= k)
        rep['P1'] = bool(0 < self.v_xi <= v1_0)
        # t^c_max: the last planned jump before T_xi of any unit
        tcm = max(t0 for i in range(n) for (t0, a) in self.units[i])
        rep['t_c_max'] = tcm
        last_rate = set(self.units[i][-1][1] for i in range(n))
        common = len(last_rate) == 1
        res_gap = []; res_v = []
        for i in range(1, n):
            sp, vp, _ = self.state(i - 1, tcm)
            sf, vf, _ = self.state(i, tcm)
            res_gap.append(sp - sf - (self.d_s[i] + self.ell))
            res_v.append(vp - vf)
        rep['closure_residual'] = res_gap
        rep['crawl_speed_residual'] = res_v
        rep['P3'] = bool(common and all(r == 0 for r in res_gap)
                         and all(r == 0 for r in res_v) and tcm < self.T_xi)
        rep['P4'] = bool(self.v_c1 > 0)
        mres = [self.state(i, self.tau_star)[0] - self.markers[i]
                for i in range(n)]
        rep['marker_residual'] = mres
        # (P5): eps*_i = v*_{i-1} - v*_i <= 0 at every breakpoint (piecewise
        # linear on [0, T_xi]; after T_xi zero relative motion under (P3))
        p5 = True; eps_min = Fr(0)
        for i in range(1, n):
            ts = sorted(set([Fr(0), self.T_xi] + self.breakpoints(i - 1)
                            + self.breakpoints(i)))
            for t in ts:
                e = self.state(i - 1, t)[1] - self.state(i, t)[1]
                eps_min = min(eps_min, e)
                if e > 0:
                    p5 = False
        rep['P5'] = p5
        rep['eps_star_min'] = eps_min
        return rep

    def table(self):
        """json-ready plan table (floats and exact strings)."""
        out = dict(family=self.family,
                   params={k: str(v) for k, v in self.params.items()},
                   T_xi=float(self.T_xi), v_xi=float(self.v_xi),
                   v_c1=float(self.v_c1), tau_star=float(self.tau_star),
                   s_ref_stop=float(self.s_ref_stop),
                   markers=[float(x) for x in self.markers],
                   s0=[float(x) for x in self.s0], units=[])
        for i in range(self.n):
            pcs = []
            for (t0, t1, a, s, v) in self.segs[i]:
                pcs.append(dict(t0=float(t0), t0_exact=str(t0),
                                a=float(a), s0=float(s), v0=float(v)))
            out['units'].append(pcs)
        for k, v in self.extra.items():
            if isinstance(v, (list, tuple)):
                out[k] = [float(x) for x in v]
            else:
                out[k] = float(v)
        return out


def takeover_positions(d_s, c0, ell):
    """s_0(0) = 0 convention: s_1(0) = -(d_s1 + ell) (e_1(0) = 0),
    s_i(0) = s_{i-1}(0) - ell - c_i(0)."""
    s0 = [-(d_s[0] + ell)]
    for i in range(1, len(d_s)):
        s0.append(s0[-1] - ell - c0[i - 1])
    return s0


def plan(cfg):
    """Plan object of cfg['plan'] (exact rationals).

    family 'canonical' (theory Sec. 2.2, base rate = crawl rate a_c):
      params v_xi, a_c, a_d, t_d1, T_xi, v_c1;  Delta_v = v_xi - v_c1
      - a_c T_xi (the crawl line reaches v_c1 at T_xi), D = Delta_v/(a_d -
      a_c), t^d_i = t^d_{i-1} + C_i/Delta_v, t^c_i = t^d_i + D.
    family 'constant' (regression only): a* = 0 on [0, T_xi), v* = v_xi;
      params v_xi, T_xi; optional 's0' (list) overrides the initial
      planned positions.
    family 'fan' (round 5): common onset, unit-specific main rates, joins
      onto a common crawl line; see fan_plan.
    """
    p = cfg['plan']
    n = int(cfg['n'])
    ell = F(FIXED['ell'])
    d_s = [F(x) for x in cfg['d_s']]
    c0 = [F(x) for x in cfg['c0']]
    C = [None] + [c0[i - 1] - d_s[i] for i in range(1, n)]
    a_b = F(cfg['a_b'])
    fam = p.get('family', 'canonical')
    if fam == 'canonical':
        v_xi = F(p['v_xi']); a_c = F(p['a_c']); a_d = F(p['a_d'])
        t_d1 = F(p['t_d1']); T_xi = F(p['T_xi']); v_c1 = F(p['v_c1'])
        dv = v_xi - v_c1 - a_c * T_xi
        if not (dv > 0 and a_d > a_c > 0):
            raise ValueError('canonical plan: need Delta_v > 0 and a_d > a_c > 0')
        D = dv / (a_d - a_c)
        td = [t_d1]
        for i in range(1, n):
            td.append(td[-1] + C[i] / dv)
        tc = [t + D for t in td]
        if not tc[-1] < T_xi:
            raise ValueError('canonical plan: t^c_n = %.4f >= T_xi' % float(tc[-1]))
        s0 = takeover_positions(d_s, c0, ell)
        units = [[(Fr(0), -a_c), (td[i], -a_d), (tc[i], -a_c)]
                 for i in range(n)]
        if t_d1 == 0:
            units = [[(Fr(0), -a_d), (tc[0], -a_c)]] + units[1:]
        return Plan('canonical', dict(v_xi=v_xi, a_c=a_c, a_d=a_d, t_d1=t_d1,
                                      T_xi=T_xi, v_c1=v_c1),
                    v_xi, s0, units, T_xi, a_b, d_s, ell, C,
                    extra=dict(Delta_v=dv, D=D, t_d=td, t_c=tc,
                               offsets=[td[i] - td[i - 1] for i in range(1, n)]))
    if fam == 'constant':
        v_xi = F(p['v_xi']); T_xi = F(p['T_xi'])
        if p.get('s0') is not None:
            s0 = [F(x) for x in p['s0']]
        else:
            s0 = takeover_positions(d_s, c0, ell)
        units = [[(Fr(0), Fr(0))] for i in range(n)]
        return Plan('constant', dict(v_xi=v_xi, T_xi=T_xi), v_xi, s0, units,
                    T_xi, a_b, d_s, ell, C)
    if fam == 'fan':
        return fan_plan(p, n, ell, d_s, c0, C, a_b)
    if fam == 'shaped':
        return shaped_plan(p, n, ell, d_s, c0, C, a_b)
    raise ValueError('unknown plan family %s' % fam)


def shaped_plan(p, n, ell, d_s, c0, C, a_b):
    """family 'shaped' (feedback-only followers, 2026-09-29): the head's
    reference acceleration is the list p['pieces'] = [(t, a), ...] of exact
    pieces on [0, T_xi) (first t = 0, strictly increasing, every a <= 0);
    every unit plans the same accelerations, and the planned positions keep
    the parking gaps (s0 = -Delta_i), so the planned clearances equal d_s,
    every schedule vanishes, and the pair errors of the followers start at
    e_i(0) = c_i(0) - d_{s,i} (evaluator key e0).
      params: v_xi (initial reference speed v_in), T_xi, pieces."""
    v_in = F(p['v_xi'])
    T_xi = F(p['T_xi'])
    pcs = [(F(t), F(a)) for t, a in p['pieces']]
    if not pcs or pcs[0][0] != 0:
        raise ValueError('shaped plan: the first piece must start at 0')
    if any(pcs[j][0] >= pcs[j + 1][0] for j in range(len(pcs) - 1)):
        raise ValueError('shaped plan: piece starts must increase')
    if not pcs[-1][0] < T_xi:
        raise ValueError('shaped plan: every piece must start before T_xi')
    if any(a > 0 for t, a in pcs):
        raise ValueError('shaped plan: the reference must not accelerate')
    s0 = takeover_positions(d_s, d_s[1:], ell)      # planned clearances = d_s
    units = [list(pcs) for _ in range(n)]
    return Plan('shaped', dict(v_xi=v_in, T_xi=T_xi), v_in, s0, units, T_xi,
                a_b, d_s, ell, C, extra=dict(a_crawl=-pcs[-1][1],
                                             n_pieces=len(pcs)))


def fan_plan(p, n, ell, d_s, c0, C, a_b):
    """family 'fan' (theory Sec. 2.2, round 5): common onset t_0 after an
    approach at the pre-braking rate a_0; unit i brakes at its own constant
    main rate a_i = a_c + G/L_i on [t_0, t^c_i), t^c_i = t_0 + L_i, and then
    follows the common crawl line at the crawl rate a_c; every plan brakes
    at -a_b from T_xi.  G is the common speed drop below the crawl line,
    and the closure equations G (L_i - L_{i-1})/2 = C_i give
    L_i = L_{i-1} + 2 C_i/G.
      params: v_xi, t_0, L_1, T_xi; G, or delta (equal closures, G = 2C/delta);
              a_c or a_head (the other one follows from a_head = a_c + G/L_1;
              both given: they must agree); a_0 (default: a_c).
    With t_0 = 0 the approach piece is dropped.

    Ramped main phases (round 6, optional): T_p (planning period) together
    with ramp (the fraction rho of every main duration, 0 < rho <= 1/2, which
    gives the ramp steps N_i = floor(rho L_i/T_p + 1/2), at most
    floor(L_i/(2 T_p))) or with ramp_steps (the list N_i).  The planned
    deceleration above the crawl rate, r_i(tau) = -a*_i - a_c on the main
    phase tau in [0, L_i), is then a symmetric staircase: it rises over the
    ramp time T^r_i = N_i T_p in N_i steps of T_p to its peak
    r_i = G/(L_i - T^r_i), holds the peak on [T^r_i, L_i - T^r_i) and falls
    back in N_i steps; step m = 0, ..., N_i - 1 of the rising ramp has the
    value r_i (2m + 1)/(2 N_i), the mean of the linear ramp over the step,
    and the falling ramp mirrors it.  The staircase is symmetric about
    L_i/2 and has the integral G, so the speed excess over the crawl line
    falls from G to 0 with the area G L_i/2, and the closure equations
    L_i = L_{i-1} + 2 C_i/G of the fan with constant rates hold unchanged.
    a_head is the peak rate of the head, a_head = a_c + G/(L_1 - T^r_1), and
    rates lists the peak rates a_c + r_i.  With N_i = 0 for every unit the
    plan is the fan with constant rates."""
    v_xi = F(p['v_xi']); t_0 = F(p['t_0']); L_1 = F(p['L_1'])
    T_xi = F(p['T_xi'])
    if p.get('G') is not None:
        G = F(p['G'])
    else:
        if len(set(C[1:])) != 1:
            raise ValueError('fan plan: delta needs equal closures; give G')
        G = 2 * C[1] / F(p['delta'])
    if not (G > 0 and L_1 > 0 and t_0 >= 0):
        raise ValueError('fan plan: need G > 0, L_1 > 0 and t_0 >= 0')
    L = [L_1]
    for i in range(1, n):
        L.append(L[-1] + 2 * C[i] / G)
    # ramp steps of the main phases (none: constant main rates)
    T_p = None if p.get('T_p') is None else F(p['T_p'])
    if p.get('ramp_steps') is not None:
        Nr = [int(x) for x in p['ramp_steps']]
        if len(Nr) != n or T_p is None:
            raise ValueError('fan plan: ramp_steps needs one entry per unit and T_p')
        rho = None
    elif p.get('ramp') is not None and F(p['ramp']) > 0:
        rho = F(p['ramp'])
        if T_p is None or not (T_p > 0 and rho <= Fr(1, 2)):
            raise ValueError('fan plan: ramp needs T_p > 0 and rho <= 1/2')
        Nr = [min((rho * Li / T_p + Fr(1, 2)).__floor__(),
                  (Li / (2 * T_p)).__floor__()) for Li in L]
    else:
        Nr = [0] * n
        rho = None
    if any(N < 0 for N in Nr) or any(2 * N * (T_p or 0) > Li
                                     for N, Li in zip(Nr, L)):
        raise ValueError('fan plan: the two ramps of a main phase overlap')
    Tr = [N * T_p if N else Fr(0) for N in Nr]
    has_c = p.get('a_c') is not None
    has_h = p.get('a_head') is not None
    if has_c:
        a_c = F(p['a_c'])
        a_head = a_c + G / (L_1 - Tr[0])
        if has_h and F(p['a_head']) != a_head:
            raise ValueError('fan plan: a_head != a_c + G/(L_1 - T^r_1)')
    elif has_h:
        a_head = F(p['a_head'])
        a_c = a_head - G / (L_1 - Tr[0])
    else:
        raise ValueError('fan plan: give a_c or a_head')
    a_0 = a_c if p.get('a_0') in (None, 'a_c') else F(p['a_0'])
    if not (a_c > 0 and a_0 > 0):
        raise ValueError('fan plan: need a_c > 0 and a_0 > 0 (a_c = %.6f)'
                         % float(a_c))
    peaks = [G / (Li - Ti) for Li, Ti in zip(L, Tr)]
    rates = [a_c + r for r in peaks]
    t_c = [t_0 + Li for Li in L]
    if not t_c[-1] < T_xi:
        raise ValueError('fan plan: t^c_n = %.4f >= T_xi' % float(t_c[-1]))
    s0 = takeover_positions(d_s, c0, ell)
    units = []
    for i in range(n):
        pcs = [(Fr(0), -a_0)] if t_0 > 0 else []
        N = Nr[i]
        w = [Fr(2 * m + 1, 2 * N) for m in range(N)]
        t = t_0
        for m in range(N):
            pcs.append((t, -(a_c + peaks[i] * w[m])))
            t += T_p
        if L[i] - 2 * Tr[i] > 0:
            pcs.append((t, -rates[i]))
            t += L[i] - 2 * Tr[i]
        for m in reversed(range(N)):
            pcs.append((t, -(a_c + peaks[i] * w[m])))
            t += T_p
        assert t == t_c[i]
        pcs.append((t_c[i], -a_c))
        units.append(pcs)
    v_on = v_xi - a_0 * t_0
    extra = dict(G=G, a_c=a_c, a_0=a_0, a_head=a_head, t_0=t_0,
                 rates=rates, L=L, t_c=t_c,
                 offsets=[L[i] - L[i - 1] for i in range(1, n)],
                 v_onset=v_on,
                 v_join=[v_on - G - a_c * Li for Li in L])
    if any(Nr):
        extra.update(T_p=T_p, ramp_steps=Nr, T_r=Tr,
                     plateau_start=[t_0 + Ti for Ti in Tr],
                     plateau_end=[tc - Ti for tc, Ti in zip(t_c, Tr)])
        if rho is not None:
            extra['ramp'] = rho
    return Plan('fan', dict(v_xi=v_xi, t_0=t_0, L_1=L_1, G=G, a_c=a_c,
                            a_0=a_0, a_head=a_head, T_xi=T_xi),
                v_xi, s0, units, T_xi, a_b, d_s, ell, C, extra=extra)


# ---------------------------------------------- piecewise-quadratic minima
def _quad_min(c0, c1, c2, L):
    """exact min over [0, L] of c0 + c1 tau + c2 tau^2 and its argmin."""
    best = (c0, Fr(0))
    vL = c0 + c1 * L + c2 * L * L
    if vL < best[0]:
        best = (vL, L)
    if c2 > 0:
        tv = -c1 / (2 * c2)
        if 0 < tv < L:
            val = c0 + c1 * tv + c2 * tv * tv
            if val < best[0]:
                best = (val, tv)
    return best


def pair_minima(P, c, lo, hi, b, kappa, s_m):
    """exact minima of g*_c, h*_c, mu*_c, eps*_c over the closed window
    [lo, hi] (pre-T_xi pieces are closed at their right ends: left
    limits included).  Pair c: units c-1 (predecessor) and c (code
    indices).  Returns dict name -> (value, argmin) as Fractions."""
    bps = {lo, hi}
    for u in (c - 1, c):
        for (t0, a) in P.units[u]:
            if lo < t0 < hi:
                bps.add(t0)
    if hi > P.T_xi:
        raise ValueError('window beyond T_xi')
    ts = sorted(bps)
    out = dict(g=None, h=None, mu=None, eps=None)
    for t0, t1 in zip(ts[:-1], ts[1:]):
        sp, vp, A = P.state(c - 1, t0)
        sf, vf, B = P.state(c, t0)
        g0 = sp - sf - P.ell - s_m
        e0 = vp - vf
        h0 = g0 + vp / b[c - 1] - vf / b[c]
        L = t1 - t0
        r = A / b[c - 1] - B / b[c]
        cands = dict(g=(g0, e0, (A - B) / 2),
                     h=(h0, e0 + r, (A - B) / 2),
                     mu=(kappa * h0 + e0 + r, kappa * (e0 + r) + (A - B),
                         kappa * (A - B) / 2),
                     eps=(e0, A - B, Fr(0)))
        for k, (q0, q1, q2) in cands.items():
            val, tau = _quad_min(q0, q1, q2, L)
            if out[k] is None or val < out[k][0]:
                out[k] = (val, t0 + tau)
    return out


# ================================================================ namespace
_NS = OrderedDict()
_NS_MAX = 8


def _overrides(cfg):
    lam = float(cfg['lam'])
    n = int(cfg['n'])
    v_xi = float(F(cfg['plan']['v_xi']))
    return dict(n=n, v_xi=v_xi, a_b=float(cfg['a_b']),
                alpha=float(cfg['alpha']), lam=lam, beta=lam * lam,
                gamma=2 * lam, s_m=float(cfg['s_m']),
                eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
                eps0=[float(x) for x in eps0_of(cfg)],
                e0=[0.0] * n,
                amax=[float(x) for x in cfg['amax']],
                T_c=float(cfg['T_c']), eps_det=float(cfg['eps_det']),
                phi=float(cfg['phi']), g_star=4.2,
                ALIGN_TOL=float(cfg['align_tol']))


def eps0_of(cfg):
    """takeover pair errors relative to the plan (float):
    eps_1(0) = v^xi - v_1(0), eps_i(0) = v_{i-1}(0) - v_i(0)."""
    v0 = [F(x) for x in cfg['v0']]
    v_xi = F(cfg['plan']['v_xi'])
    e = [v_xi - v0[0]] + [v0[i - 1] - v0[i] for i in range(1, len(v0))]
    return [float(x) for x in e]


def namespace(cfg):
    """Frozen-definition namespace (cached)."""
    ov = _overrides(cfg)
    vhnd = float(cfg['vhnd'])
    key = tuple((k, tuple(v) if isinstance(v, list) else v)
                for k, v in sorted(ov.items())) + (('vhnd', vhnd),)
    if key in _NS:
        _NS.move_to_end(key)
        return _NS[key]
    full = dict(ov, T_xi=float(F(cfg['plan']['T_xi'])),
                dbar=float(cfg['dbar']))
    G = bd._namespace(full)
    G['VHND_PAD'] = vhnd
    _NS[key] = G
    while len(_NS) > _NS_MAX:
        _NS.popitem(last=False)
    return G


# ============================================= Steps A and B (frozen copy)
def chain(G, e0, eps0, dbar, ramp_charge=False):
    """Statement-for-statement copy of certify_v30.forward_pass (the
    featured globals replaced by the arguments e0, eps0; VHND_PAD, a_b,
    v_xi, eps_e, eps_v, T_c, eps_det, alpha_g and the modal constants from
    the frozen namespace G).  ramp_charge=False drops a_b from W_i
    (Case D, local ramp).  Returns a record dict (closed = False when the
    frozen divergence or headroom test fails; the acquisition fields are
    returned in either case)."""
    n = G['n']; alpha_g = G['alpha_g']; T_c = G['T_c']
    eps_det = G['eps_det']; beta = G['beta']; gamma = G['gamma']
    b = G['b']; a_b = G['a_b']; v_xi = G['v_xi']
    eps_e = G['eps_e']; eps_v = G['eps_v']; VHND_PAD = G['VHND_PAD']
    corner_stats = G['corner_stats']; D_quad = G['D_quad']
    DB_quad = G['DB_quad']
    GAIN_X = G['GAIN_X']; GAIN_F = G['GAIN_F']; GAIN_E = G['GAIN_E']
    TV_KIMP = G['TV_KIMP']; XFREE_PK = G['XFREE_PK']
    EFREE_PK = G['EFREE_PK']; IE_BOX = G['IE_BOX']; IX_BOX = G['IX_BOX']
    L1_GE = G['L1_GE']; L1_GX = G['L1_GX']
    e0 = np.asarray(e0, float); eps0 = np.asarray(eps0, float)

    Lam1 = np.zeros(n); Lam2 = np.zeros(n); taub = np.zeros(n)
    tact = np.zeros(n); tset = np.zeros(n); E1 = np.zeros(n)
    ebar = np.zeros(n); fbar = np.zeros(n); fsw = np.zeros(n)
    tvS2 = np.zeros(n); x2tot = np.zeros(n); Tk = np.zeros(n)
    Dq = np.zeros(n); DBq = np.zeros(n)
    sup_e = np.zeros(n); sup_x = np.zeros(n)

    for k in range(n):
        Lam1[k] = dbar*np.sum(Tk[:k])
        m_k     = abs(eps0[k]) + Lam1[k]
        taub[k] = m_k/alpha_g[k] + T_c
        tact[k] = 0.0 if k == 0 else tset[k-1] + dbar
        tset[k] = tact[k] + taub[k]
        Lk_eps  = alpha_g[k] + np.sum(Tk[:k])
        E1[k]   = m_k*tact[k] \
                  + max(m_k**2 - eps_det**2, 0.0)/(2.0*alpha_g[k]) \
                  + (eps_det + Lk_eps*T_c)*T_c
        ebar[k] = abs(e0[k]) + E1[k]
        st      = corner_stats(ebar[k], eps_det)
        Lam2[k] = dbar*np.sum(tvS2[:k])
        x2tot[k] = st['sup_x'] + GAIN_X*Lam2[k]
        fbar[k]  = st['sup_f'] + GAIN_F*Lam2[k]
        tvS2[k]  = st['tv_f'] + TV_KIMP*Lam2[k]
        fsw[k]   = beta*ebar[k] + gamma*eps_det
        Tk[k]    = 2.0*alpha_g[k] + fsw[k] + tvS2[k]
        Dq[k]    = D_quad(ebar[k], eps_det, b[k])
        DBq[k]   = DB_quad(b[k])
        sup_e[k] = st['sup_e']; sup_x[k] = st['sup_x']

    Fv = np.array([np.sum(fbar[:i]) for i in range(n)])

    f_xi = beta*eps_e + gamma*eps_v
    T_brk = (v_xi + eps_v)/a_b + (n-1)*dbar + 1.0
    diverged = False
    for _ in range(30):
        W = np.zeros(n); epsb = np.zeros(n); eb = np.zeros(n)
        fb = np.zeros(n); Th = np.zeros(n)
        IE = np.zeros(n); IX = np.zeros(n); IF = np.zeros(n)
        Fcum = np.zeros(n)
        for k in range(1, n):
            if ramp_charge:
                W[k] = dbar*(VHND_PAD + a_b + f_xi
                             + np.sum(Th[1:k]) + np.sum(Fcum[1:k]))
            else:
                W[k] = dbar*(VHND_PAD + f_xi
                             + np.sum(Th[1:k]) + np.sum(Fcum[1:k]))
            epsb[k] = XFREE_PK + GAIN_X*W[k]
            eb[k]   = EFREE_PK + GAIN_E*W[k]
            fb[k]   = beta*eb[k] + gamma*epsb[k]
            Fcum[k] = f_xi + np.sum(fb[1:k+1])
            IE[k]   = IE_BOX + L1_GE*W[k]
            IX[k]   = IX_BOX + L1_GX*W[k]
            IF[k]   = beta*IE[k] + gamma*IX[k]
            Th[k]   = beta*IX[k] + gamma*(W[k] + IF[k])
        epsb_max = np.max(epsb)
        if not np.isfinite(epsb_max) or epsb_max > 50.0:
            diverged = True; break
        T_new = (v_xi + eps_v)/a_b + (n-1)*(dbar + epsb_max/a_b)
        if abs(T_new - T_brk) < 1e-10:
            break
        T_brk = T_new
    closed = not (diverged or np.min(a_b - Fcum[:n-1]) <= 0)
    Fb = np.array([f_xi + np.sum(fb[1:i]) for i in range(n)])
    hops = epsb[1:]/(a_b - Fb[1:n])
    disp = np.sum(hops)
    late = np.sum(epsb[1:]/a_b)
    sched = eps_v/a_b + np.sum(hops)
    gaperr = np.array([EFREE_PK + GAIN_E*W[i]
                       + epsb[i]**2/(2*(a_b - Fb[i]))
                       for i in range(1, n)])
    Leps = np.max(alpha_g) + np.sum(Tk)
    return dict(closed=bool(closed), diverged=bool(diverged),
                Lam1=Lam1, Lam2=Lam2, taub=taub, tact=tact, tset=tset,
                E1=E1, ebar=ebar, fbar=fbar, fsw=fsw, tvS2=tvS2,
                x2tot=x2tot, Tk=Tk, Dq=Dq, DBq=DBq, F=Fv, sup_e=sup_e,
                sup_x=sup_x, W=W, epsb=epsb, eb=eb, fb=fb, Th=Th,
                Fcum=Fcum, Fb=Fb, IE=IE, IX=IX, IF=IF, f_xi=f_xi,
                T_brk=T_brk, hops=hops, disp=disp, late=late, sched=sched,
                gaperr=gaperr, Leps=Leps)


# ======================================================== Step E helpers
def _S(p, q, R, lam):
    """sup_{r >= R} (p + q r) e^{-lam r} (signed, closed form)."""
    return cp._tail_sup_signed(p, q, R, lam)


def _Swin(p, q, R1, R2, lam):
    """sup_{r in [R1, R2]} (p + q r) e^{-lam r} (closed form)."""
    g1 = (p + q * R1) * math.exp(-lam * R1)
    g2 = (p + q * R2) * math.exp(-lam * R2)
    out = max(g1, g2)
    if q > 0.0:
        ts = (q - lam * p) / (lam * q)
        if R1 < ts < R2:
            out = max(out, (q / lam) * math.exp(-lam * ts))
    return out


def box_corners(e0k, E1k, eps_det, one_sided):
    if one_sided:
        return [(e, x) for e in (e0k - E1k, e0k) for x in (-eps_det, 0.0)]
    return [(e, x) for e in (e0k - E1k, e0k + E1k)
            for x in (-eps_det, eps_det)]


def _f_pq(lam, e, x):
    return lam * lam * e + 2.0 * lam * x, -lam * lam * (x + lam * e)


def _x_pq(lam, e, x):
    return x, -lam * (x + lam * e)


def Tf(lam, corners, R, sign=+1):
    """T^{f+}_k(R) (sign=+1) or T^{f-}_k(R) (sign=-1): max over the corners
    of sup_{r >= R} (+-f_free)(r); R scalar or array."""
    R = np.asarray(R, float)
    out = np.zeros_like(R)
    for (e, x) in corners:
        p, q = _f_pq(lam, e, x)
        out = np.maximum(out, _S(sign * p, sign * q, R, lam))
    return out


def Tf_win(lam, corners, R1, R2, sign=+1):
    out = 0.0
    for (e, x) in corners:
        p, q = _f_pq(lam, e, x)
        out = max(out, _Swin(sign * p, sign * q, R1, R2, lam))
    return out


def EnvX(lam, corners, R):
    """max over the corners of sup_{r >= R} eps_free(r) (without Lambda2)."""
    R = np.asarray(R, float)
    out = np.zeros_like(R)
    for (e, x) in corners:
        p, q = _x_pq(lam, e, x)
        out = np.maximum(out, _S(p, q, R, lam))
    return out


# ================================================================ evaluate
def _cond(ok, slack, value=None, threshold=None, sense=None, **kw):
    d = dict(ok=bool(ok), slack=None if slack is None else float(slack),
             value=None if value is None else float(value),
             threshold=None if threshold is None else float(threshold),
             sense=sense)
    d.update(kw)
    return d


def evaluate(cfg, want_entry=True, P=None):
    """Full evaluation of one configuration."""
    cfg = config(**cfg) if 'plan' not in cfg or 'n' not in cfg else cfg
    for kf, vf in FIXED.items():
        if kf in cfg and abs(float(cfg[kf]) - vf) > 0:
            raise ValueError('%s is frozen at %g' % (kf, vf))
    t_start = time.time()
    n = int(cfg['n']); dbar = float(cfg['dbar'])
    P = P or plan(cfg)
    G = namespace(cfg)
    lam = G['lam']; beta = G['beta']; gamma = G['gamma']
    alpha_g = G['alpha_g']; b_f = G['b']; Pi = G['Pi']; eta = G['eta']
    kappa = G['kappa']; a_b = G['a_b']; eps_e = G['eps_e']
    eps_v = G['eps_v']; eps_det = G['eps_det']; phi = G['phi']
    T_c = G['T_c']; GAIN_E = G['GAIN_E']; A_run = G['A_run']
    kk = float(cfg['k']); s_m = float(cfg['s_m'])
    NUMPAD = float(cfg['numpad'])
    T_xi_F = P.T_xi; T_xi = float(T_xi_F)
    v_c1 = float(P.v_c1)
    v0 = [float(F(x)) for x in cfg['v0']]
    eps0 = np.array(eps0_of(cfg))
    # takeover pair errors relative to the plan: zero under (P1); the key
    # 'e0' is used only by the constant-plan regression (R3)
    e0 = np.zeros(n) if cfg.get('e0') is None else np.array(cfg['e0'], float)
    vt0 = np.array([float(F(cfg['v0'][i]) - P.v_xi) for i in range(n)])
    d_s = [float(x) for x in cfg['d_s']]
    c0 = [float(x) for x in cfg['c0']]
    # exact slopes for the plan quantities
    b_F = [F(a) / F(FIXED['v_c']) for a in cfg['amax']]
    kap_F = F(FIXED['kappa']); s_m_F = F(cfg['s_m'])

    R = chain(G, e0, eps0, dbar, ramp_charge=bool(cfg['ramp_charge']))
    res = dict(case='D', dbar=dbar, T_xi=T_xi, closed=R['closed'])
    conds = OrderedDict()

    # ------------------------------------------------ plan premises
    prem = P.premises(F(cfg['k']), F(cfg['v0'][0]))
    tcm = float(prem['t_c_max'])
    # one row per premise of Assumption 1'' (exact truth values from
    # Plan.premises; the slacks are in one unit each)
    s_p1 = float(min(P.v_xi, F(cfg['v0'][0]) - P.v_xi))
    s_p2 = float(min(prem['a_min'], F(cfg['k']) - prem['a_max']))
    conds['P1'] = _cond(prem['P1'], s_p1, float(P.v_xi),
                        float(F(cfg['v0'][0])), '<=', unit='m/s')
    conds['P2'] = _cond(prem['P2'], s_p2, float(prem['a_min']), 0.0, '>',
                        unit='m/s^2', a_min=float(prem['a_min']),
                        a_max=float(prem['a_max']))
    conds['P3'] = _cond(prem['P3'], float(T_xi_F - prem['t_c_max']), tcm,
                        T_xi, '<', unit='s', P5=prem['P5'])
    conds['P4'] = _cond(prem['P4'], v_c1, v_c1, 0.0, '>', unit='m/s')

    # ------------------------------------------------ Step A derived
    Lam1 = R['Lam1']; Lam2 = R['Lam2']; tset = R['tset']; tact = R['tact']
    E1 = R['E1']; ebar = R['ebar']
    ps1_vals = eps0 + Lam1
    ps1_slack = -float(np.max(ps1_vals))
    one_sided = cfg.get('one_sided')
    if one_sided is None:
        one_sided = ps1_slack >= 0
    one_sided = bool(one_sided)
    conds['P_S1'] = _cond(ps1_slack >= 0 or not one_sided, ps1_slack,
                          float(np.max(ps1_vals)), 0.0, '<=',
                          required=one_sided)
    # H7 (frozen)
    Leps = float(R['Leps'])
    ev_ok = (eps_det - phi >= Leps * T_c) and (eps_det > Leps * T_c)
    conds['H7'] = _cond(ev_ok, eps_det - phi - Leps * T_c, Leps * T_c,
                        eps_det - phi, '<=')

    if not R['closed']:
        for nm in COND_NAMES:
            if nm not in conds:
                conds[nm] = _cond(False, None)
        conds['H2'] = _cond(False, float(np.min(a_b - R['Fb'][1:n])))
        res.update(conds=conds, admitted=False,
                   failed=[k for k, c in conds.items() if not c['ok']],
                   note='braking recursion diverged or headroom <= 0',
                   plan=P.table())
        return res

    # ------------------------------------------------ Step C entry
    if want_entry:
        Tent, V, _ = cp._entry(G, R, Lam2, dbar, ff=None,
                               lags=cp.LAG_FROZEN)
        T_ent_bar = float(np.max(tset + Tent))
        NT = G['NT']; dtg = G['dtg']
        ixh = min(NT - 1, int((T_xi - dbar) / dtg))
        Vhnd = np.array([0.0] + [float(V[i][ixh]) for i in range(1, n)])
        vh = float(np.max(Vhnd))
        conds['H3D'] = _cond(T_ent_bar + (n - 1) * dbar <= T_xi,
                             T_xi - T_ent_bar - (n - 1) * dbar,
                             T_ent_bar + (n - 1) * dbar, T_xi, '<=')
        conds['C_hnd'] = _cond(vh <= G['VHND_PAD'], G['VHND_PAD'] - vh, vh,
                               G['VHND_PAD'], '<=')
    else:
        Tent = None; T_ent_bar = None; Vhnd = None
        conds['H3D'] = _cond(False, None)
        conds['C_hnd'] = _cond(False, None)

    # ------------------------------------------------ Step E envelopes
    corners = [box_corners(e0[k], E1[k], eps_det, one_sided) for k in range(n)]
    sig_p = np.array([1.0 if eps0[k] + Lam1[k] > eps_det else 0.0
                      for k in range(n)])
    sig_m = np.array([1.0 if eps0[k] - Lam1[k] < -eps_det else 0.0
                      for k in range(n)])
    Lam2h = Lam2.copy(); Lam2h[0] = 0.0

    # certified lower bounds of the switch instants (windowed refinement)
    Tk = R['Tk']
    tset_lo = np.zeros(n)
    for k in range(n):
        Lk = alpha_g[k] + np.sum(Tk[:k])
        dk = max(abs(eps0[k]) - (Lam1[k] if k else 0.0) - eps_det, 0.0) / Lk
        tset_lo[k] = (tset_lo[k - 1] if k else 0.0) + dk

    def Fpm(i, t, t_b=None):
        """(F^+_i(t), F^-_i(t), parts); with t_b the windowed piece bound."""
        fp = 0.0; fm = 0.0; parts = []
        s1p = 0.0; s1m = 0.0
        for k in range(i + 1):
            Rk = max(t - (i - k) * dbar - tset[k], 0.0)
            chi = 1.0 if t < tset[k] + (i - k) * dbar else 0.0
            if t_b is not None and cfg.get('windowed'):
                R2 = t_b - tset_lo[k]
                if R2 <= 0.0:
                    tp = tm = 0.0
                else:
                    tp = Tf_win(lam, corners[k], Rk, max(R2, Rk), +1)
                    tm = Tf_win(lam, corners[k], Rk, max(R2, Rk), -1)
                    tp = max(tp, 0.0); tm = max(tm, 0.0)
            else:
                tp = float(Tf(lam, corners[k], Rk, +1))
                tm = float(Tf(lam, corners[k], Rk, -1))
            forced = gamma * Lam2h[k]
            fp += tp + forced; fm += tm + forced
            s1p = max(s1p, alpha_g[k] * sig_p[k] * chi)
            s1m = max(s1m, alpha_g[k] * sig_m[k] * chi)
            parts.append(dict(k=k + 1, R=Rk, chi=chi, Tf_plus=tp, Tf_minus=tm,
                              forced=forced))
        return fp + s1p, fm + s1m, dict(layers=parts, s1_plus=s1p,
                                        s1_minus=s1m)

    auth = []
    up_min = (np.inf, None); lo_min = (np.inf, None)
    for i in range(n):
        for (t0, t1, a) in P.acc_pieces(i):
            aP = float(-a)
            fp, fm, parts = Fpm(i, float(t0), float(t1))
            su = aP - fp; sl = kk - aP - fm
            row = dict(unit=i + 1, t0=float(t0), t1=float(t1), a_P=aP,
                       F_plus=fp, F_minus=fm, slack_upper=su,
                       slack_lower=sl, s1_plus=parts['s1_plus'],
                       s1_minus=parts['s1_minus'])
            auth.append(row)
            if su < up_min[0]:
                up_min = (su, (i + 1, float(t0)))
            if sl < lo_min[0]:
                lo_min = (sl, (i + 1, float(t0)))
    conds['H1D_i'] = _cond(up_min[0] >= 0, up_min[0], None, 0.0, '>=',
                           at=up_min[1])
    conds['H1D_ii'] = _cond(lo_min[0] >= 0, lo_min[0], None, 0.0, '>=',
                            at=lo_min[1])
    Fcum = R['Fcum']; Fb = R['Fb']
    s3 = [min(kk - a_b - Fcum[c], a_b - Fcum[c]) for c in range(1, n)]
    conds['H1D_iii'] = _cond(min(s3) >= 0, min(s3), float(a_b + np.max(Fcum[1:n])),
                             kk, '<=')
    anet = a_b - Fb[1:n]
    conds['H2'] = _cond(float(np.min(anet)) > 0, float(np.min(anet)),
                        float(np.min(anet)), 0.0, '>')

    # ------------------------------------------------ H6D velocity floor
    dt = float(cfg['grid_dt'])
    J = int(math.ceil(T_xi / dt - 1e-12))
    tj = np.arange(J) * dt
    tj1 = np.minimum((np.arange(J) + 1) * dt, T_xi)
    Pk = []
    for k in range(n):
        Rg = np.maximum(tj - tset[k], 0.0)
        env = EnvX(lam, corners[k], Rg) + Lam2[k]
        if not one_sided:
            M1k = abs(eps0[k]) + Lam1[k]
            pre = max(max(eps0[k] + Lam1[k], 0.0), M1k * sig_p[k],
                      float(EnvX(lam, corners[k], 0.0)) + Lam2[k])
            env = np.where(tj < tset[k], pre, env)
        Pk.append(env)
    Pcum = np.cumsum(np.array(Pk), axis=0)
    vfl = np.inf; vfl_at = None
    for i in range(n):
        vs = P.state_f(i, tj1)[1]
        cur = vs - Pcum[i]
        j = int(np.argmin(cur))
        if cur[j] < vfl:
            vfl = float(cur[j]); vfl_at = (i + 1, float(tj[j]))
    vfl_const = v_c1 - float(Pcum[n - 1][0])
    conds['H6D'] = _cond(vfl > 0, vfl, vfl, 0.0, '>', at=vfl_at,
                         constant_floor=vfl_const)

    # ------------------------------------------------ budgets, admission
    H1 = np.zeros(n); Ht = np.zeros(n); hm1 = np.zeros(n); hm2 = np.zeros(n)
    Ph2 = np.zeros(n); hmb = np.zeros(n); hmb_s2 = np.zeros(n)
    hmb_rule = np.zeros(n)
    x2tot = R['x2tot']; fbar = R['fbar']; Fv = R['F']
    W = R['W']; epsb = R['epsb']; IX = R['IX']; IF = R['IF']
    Dq = R['Dq']; DBq = R['DBq']; fb = R['fb']; eb = R['eb']
    pairs = {}
    adm = {'H4D_i': [], 'H4D_ii': [], 'H4D_iii': []}
    clr = []; bfl = []
    for c in range(1, n):
        V1 = np.sum(np.abs(eps0[:c]) + Lam1[:c] + x2tot[:c])
        H1[c] = E1[c] + (alpha_g[c]/b_f[c])*R['taub'][c] + V1*abs(Pi[c]) \
            + Lam1[c]/b_f[c]
        V2 = 2.0*np.sum(x2tot[:c])
        Ht[c] = Dq[c] + V2*abs(Pi[c]) + Lam2[c]*(1.0/b_f[c] + DBq[c])
        M1 = abs(eps0[c]) + Lam1[c]
        hm1_coast = (b_f[c]*M1 + (Fv[c] + A_run[c-1])*eta[c])/(kappa*b_f[c])
        hm1_s1 = (alpha_g[c] + b_f[c]*M1 + Fv[c]*eta[c])/(kappa*b_f[c])
        hm1[c] = max(hm1_coast, hm1_s1)
        hm2[c] = (fbar[c] + b_f[c]*x2tot[c] + Fv[c]*eta[c])/(kappa*b_f[c])
        bmin = min(b_f[c], b_f[c-1])
        Vbrk = v_c1 + c*eps_v
        Ph2[c] = IX[c] + epsb[c]/bmin + epsb[c]**2/(2.0*anet[c-1]) \
            + IF[c]/b_f[c] + Vbrk*max(Pi[c], 0.0) + W[c]*(1.0/b_f[c] + DBq[c])
        Ubrk = a_b + Fb[c]
        hmb_s2[c] = (fb[c] + b_f[c]*epsb[c] + Ubrk*eta[c])/(kappa*b_f[c])
        hmb_rule[c] = max(0.0, b_f[c]*epsb[c]
                          + (b_f[c]/b_f[c-1])*(a_b + Fb[c]) - a_b)/(kappa*b_f[c])
        hmb[c] = max(hmb_s2[c], hmb_rule[c])
        ht0 = e0[c] + eps0[c]/b_f[c] + vt0[c-1]*Pi[c]
        # plan minima (exact rationals)
        w1 = pair_minima(P, c, Fr(0), min(Fr(float(tset[c])), T_xi_F),
                         b_F, kap_F, s_m_F)
        w2 = pair_minima(P, c, Fr(0), T_xi_F, b_F, kap_F, s_m_F)
        mu1 = float(w1['mu'][0]); mu2 = float(w2['mu'][0])
        hstar_T = P.state(c - 1, T_xi_F)[0] - P.state(c, T_xi_F)[0] \
            - P.ell - s_m_F + P.state(c - 1, T_xi_F)[1] / b_F[c - 1] \
            - P.state(c, T_xi_F)[1] / b_F[c]
        hstar_T = float(hstar_T)
        s_i = ht0 - H1[c] + mu1/float(kap_F) - hm1[c] - NUMPAD
        s_ii = ht0 - H1[c] - Ht[c] + mu2/float(kap_F) - hm2[c] - NUMPAD
        hb = hstar_T + ht0 - H1[c] - Ht[c] - Ph2[c]
        s_iii = hb - hmb[c] - NUMPAD
        adm['H4D_i'].append(s_i); adm['H4D_ii'].append(s_ii)
        adm['H4D_iii'].append(s_iii)
        # clearance floor (Sec. 6.7, general form without (P5))
        e2bar = R['sup_e'][c] + GAIN_E*Lam2[c]
        gmin1 = float(w1['g'][0]) + e0[c] - E1[c]
        gmin2 = float(w2['g'][0]) - e2bar
        gmin3 = (d_s[c] - s_m) - eb[c] - epsb[c]**2/(2.0*a_b)
        clr.append(min(gmin1, gmin2, gmin3))
        # barrier floors in acquisition
        hf1 = float(w1['h'][0]) + ht0 - H1[c]
        hf2 = float(w2['h'][0]) + ht0 - H1[c] - Ht[c]
        bfl.append(min(hf1, hf2))
        sigma = b_f[c]*min(s_i + NUMPAD, s_ii + NUMPAD, s_iii + NUMPAD)
        pairs[c + 1] = dict(
            h_tilde0=ht0, H1=H1[c], Ht=Ht[c], hm1=hm1[c], hm2=hm2[c],
            Phi2D=Ph2[c], hmbD=hmb[c], hmb_s2=hmb_s2[c], hmb_rule=hmb_rule[c],
            Ubrk=Ubrk, V1=V1, V2=V2, Vbrk=Vbrk,
            mu1=mu1, mu1_at=float(w1['mu'][1]), mu2=mu2,
            mu2_at=float(w2['mu'][1]), hstar_min1=float(w1['h'][0]),
            hstar_min2=float(w2['h'][0]), hstar_min2_at=float(w2['h'][1]),
            gstar_min1=float(w1['g'][0]), gstar_min2=float(w2['g'][0]),
            eps_star_min=float(w2['eps'][0]), hstar_T_xi=hstar_T,
            h0=float(c0[c-1] - s_m + v0[c-1]/b_f[c-1] - v0[c]/b_f[c]),
            hstar0=float(P.state(c - 1, 0)[0] - P.state(c, 0)[0] - P.ell
                         - s_m_F + P.v_xi / b_F[c - 1] - P.v_xi / b_F[c]),
            slack_i=s_i, slack_ii=s_ii, slack_iii=s_iii, h_b=hb,
            gmin_acq=gmin1, gmin_s2=gmin2, gmin_brk=gmin3,
            e2bar=e2bar, hfloor_acq=hf1, hfloor_s2=hf2, sigma=sigma)
    for nm in ('H4D_i', 'H4D_ii', 'H4D_iii'):
        v = adm[nm]
        conds[nm] = _cond(min(v) > 0, min(v), None, 0.0, '>',
                          per_pair=[float(x) for x in v])
    conds['H5D_i'] = _cond(min(clr) > 0, min(clr), min(clr), 0.0, '>',
                           per_pair=[float(x) for x in clr])
    conds['H5D_ii'] = _cond(min(bfl) >= 0, min(bfl), min(bfl), 0.0, '>=',
                            per_pair=[float(x) for x in bfl])

    # ------------------------------------------------ terminal
    disp = float(R['disp'])
    gaperr = R['gaperr']
    align = eps_e + v_c1*eps_v/a_b + eps_v**2/(2*a_b) + float(np.sum(gaperr))
    conds['C_sync'] = _cond(disp <= cfg['sync_tol'], cfg['sync_tol'] - disp,
                            disp, cfg['sync_tol'], '<=')
    conds['C_al'] = _cond(align <= cfg['align_tol'], cfg['align_tol'] - align,
                          align, cfg['align_tol'], '<=')
    ge = float(np.max(gaperr))
    conds['C_gap'] = _cond(ge <= cfg['gap_tol'], cfg['gap_tol'] - ge, ge,
                           cfg['gap_tol'], '<=')
    epsb_max = float(np.max(epsb[1:]))
    T_f_bar = T_xi + (v_c1 + eps_v)/a_b + (n - 1)*epsb_max/a_b
    sched = eps_v/a_b + disp

    # order the table as COND_NAMES
    conds = OrderedDict((k, conds[k]) for k in COND_NAMES)
    admitted = all(c['ok'] for c in conds.values())
    consts = dict(
        eps0=eps0.tolist(), vt0=vt0.tolist(), one_sided=one_sided,
        Lam1=Lam1.tolist(), Lam2=Lam2.tolist(), taub=R['taub'].tolist(),
        tact=tact.tolist(), tset=tset.tolist(), tset_lo=tset_lo.tolist(),
        T_sw_bar=float(np.max(tset)), E1=E1.tolist(), ebar=ebar.tolist(),
        fsw=R['fsw'].tolist(), Tk=R['Tk'].tolist(), x2tot=x2tot.tolist(),
        fbar=fbar.tolist(), tvS2=R['tvS2'].tolist(), F=Fv.tolist(),
        W=W[1:].tolist(), epsb=epsb[1:].tolist(), eb=eb[1:].tolist(),
        fb=fb[1:].tolist(), Theta=R['Th'][1:].tolist(),
        Fb_units=Fcum.tolist(), f_xi=float(R['f_xi']),
        anet=anet.tolist(), q=R['hops'].tolist(), settling_bound=disp,
        late=float(R['late']), sched=sched, align=align,
        gaperr=gaperr.tolist(), T_f_bar=T_f_bar, Leps=Leps,
        Tent=None if Tent is None else [float(x) for x in Tent],
        T_ent_bar=T_ent_bar,
        T_hold=None if T_ent_bar is None else T_ent_bar + (n - 1)*dbar,
        Vhnd=None if Vhnd is None else Vhnd[1:].tolist(),
        Vhnd_max=None if Vhnd is None else float(np.max(Vhnd)),
        pairs=pairs, authority=auth,
        F_plus_0=[Fpm(i, 0.0)[0] for i in range(n)],
        F_minus_0=[Fpm(i, 0.0)[1] for i in range(n)],
        P_k_0=[float(Pk[k][0]) for k in range(n)],
        P_k_Txi=[float(Pk[k][-1]) for k in range(n)],
        vfloor_min=vfl, vfloor_at=vfl_at, vfloor_const=vfl_const,
        v_c1=v_c1, tau_star=float(P.tau_star),
        stop_offset_bound=sched,
        plan_premises=dict(
            a_min=float(prem['a_min']), a_max=float(prem['a_max']),
            t_c_max=tcm, P1=prem['P1'], P2=prem['P2'], P3=prem['P3'],
            P4=prem['P4'], P5=prem['P5'],
            closure_residual=[str(x) for x in prem['closure_residual']],
            crawl_speed_residual=[str(x) for x in prem['crawl_speed_residual']],
            marker_residual=[str(x) for x in prem['marker_residual']],
            eps_star_min=float(prem['eps_star_min'])),
        NUMPAD=NUMPAD, VHND_PAD=float(G['VHND_PAD']))
    res.update(conds=conds, admitted=bool(admitted),
               failed=[k for k, c in conds.items() if not c['ok']],
               consts=consts, plan=P.table(), one_sided=one_sided,
               runtime_s=time.time() - t_start)
    return res


# ======================================================= age boundaries
def age_boundary_ms(cfg, kmax=200, verbose=False, memo=None):
    """Largest integer k with the certificate admitted at dbar = k ms
    (bisection; the neighbours k and k+1 are evaluated), with the failing
    conditions at k+1.  Down-closedness in dbar: theory Sec. 6.10."""
    cfg = config(**cfg) if 'n' not in cfg else cfg
    memo = {} if memo is None else memo
    P = plan(cfg)

    def ev(k):
        if k not in memo:
            memo[k] = evaluate(dict(cfg, dbar=k * 1e-3), P=P)
        return memo[k]

    if not ev(1)['admitted']:
        return dict(k_ms=0, failed_next=ev(1)['failed'])
    lo, hi = 1, kmax
    if ev(hi)['admitted']:
        return dict(k_ms=hi, note='admitted at kmax')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ev(mid)['admitted']:
            lo = mid
        else:
            hi = mid
    r0 = ev(lo); r1 = ev(lo + 1)
    assert r0['admitted'] and not r1['admitted']
    out = dict(k_ms=lo, admitted_at=lo * 1e-3, failed_next=r1['failed'],
               slacks_at_k={nm: r0['conds'][nm]['slack'] for nm in COND_NAMES},
               slacks_at_k1={nm: r1['conds'][nm]['slack'] for nm in COND_NAMES})
    if verbose:
        print('  boundary: %d ms admitted, %d ms fails %s'
              % (lo, lo + 1, r1['failed']), flush=True)
    return out


def cond_boundaries_ms(cfg, kmax=200, memo=None, names=None):
    """Per-condition largest k (ms) at which the condition holds."""
    cfg = config(**cfg) if 'n' not in cfg else cfg
    memo = {} if memo is None else memo
    P = plan(cfg)

    def ev(k):
        if k not in memo:
            memo[k] = evaluate(dict(cfg, dbar=k * 1e-3), P=P)
        return memo[k]

    out = {}
    for nm in (names or COND_NAMES):
        def ok(k):
            return bool(ev(k)['conds'][nm]['ok'])
        if not ok(1):
            out[nm] = dict(k_ms=0); continue
        if ok(kmax):
            out[nm] = dict(k_ms=kmax, note='holds at kmax'); continue
        lo, hi = 1, kmax
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if ok(mid):
                lo = mid
            else:
                hi = mid
        out[nm] = dict(k_ms=lo, slack_at_k=ev(lo)['conds'][nm]['slack'],
                       slack_at_k1=ev(lo + 1)['conds'][nm]['slack'])
    return out


# ============================================================ regression
def _bitcmp(a, b, path, bad):
    if isinstance(a, dict):
        for k in a:
            _bitcmp(a[k], b[k], path + '/' + str(k), bad)
    elif isinstance(a, np.ndarray):
        if a.shape != np.asarray(b).shape or not np.array_equal(a, np.asarray(b)):
            bad.append((path, a.tolist(), np.asarray(b).tolist()))
    elif isinstance(a, float) or isinstance(a, np.floating):
        if not (a == b):
            bad.append((path, float(a), float(b)))


FIELDS_R1 = ('Lam1', 'Lam2', 'taub', 'tact', 'tset', 'E1', 'ebar', 'fbar',
             'fsw', 'tvS2', 'x2tot', 'Tk', 'Dq', 'DBq', 'F', 'W', 'epsb',
             'eb', 'fb', 'Th', 'Fb', 'Fcum', 'IE', 'IX', 'IF', 'f_xi',
             'T_brk', 'hops', 'disp', 'late', 'sched', 'gaperr', 'Leps')


def regress(verbose=True):
    """(R1) Steps A-B with ramp_charge reproduce the frozen forward pass
    bit for bit; (R2) Step C reproduces the frozen certified_entry;
    (R3) constant-plan reduction of the admission margins (Case B)."""
    out = {}
    # ---- (R1)+(R2) on Case B, Case A and the Case D constants
    cases = {}
    cfgB = cp.config()
    GB = cp.namespace(cfgB)
    geoB = cp.geometry(GB, cfgB)
    cases['caseB'] = (GB, geoB['e0'], np.array(GB['eps0']), float(cfgB['dbar']))
    A = json.load(open(os.path.join(HERE, 'v30_values.json')))
    cfgA = cp.config(v_xi=10.0, a_b=0.70, alpha=0.30, lam=0.075, s_m=3.0,
                     T_xi=180.0, dbar=0.06, eps_e=0.20, eps_v=0.02,
                     eps0=[-0.05, 0.025, 0.025, 0.025, 0.025],
                     e0=[-0.5, 40.0, 40.0, 40.0, 40.0], c0=None,
                     d_s=[0.0] + list(A['d_s']), sync_tol=1.0,
                     align_tol=3.0, gap_tol=1.0, vhnd=1e-3)
    GA = cp.namespace(cfgA)
    geoA = cp.geometry(GA, cfgA)
    cases['caseA'] = (GA, geoA['e0'], np.array(GA['eps0']), float(cfgA['dbar']))
    cfgD = config()
    GD = namespace(cfgD)
    cases['caseD_constants'] = (GD, np.zeros(5), np.array(eps0_of(cfgD)),
                                float(cfgD['dbar']))
    r1 = {}
    for name, (G, e0, eps0, dbar) in cases.items():
        G['e0'][:] = e0
        Rf = G['forward_pass'](dbar)
        Rc = chain(G, e0, eps0, dbar, ramp_charge=True)
        bad = []
        if Rf is None:
            r1[name] = dict(frozen_closed=False, copy_closed=Rc['closed'])
            continue
        for k in FIELDS_R1:
            _bitcmp(Rf[k], Rc[k], name + '/' + k, bad)
        # (R2) entry
        Tf_, Vh_, Vfr = G['certified_entry'](Rf, dbar)
        Tc_, Vc_, _ = cp._entry(G, Rc, Rc['Lam2'], dbar, ff=None,
                                lags=cp.LAG_FROZEN)
        bad_e = []
        _bitcmp(np.asarray(Tf_), np.asarray(Tc_), name + '/Tent', bad_e)
        for j in range(len(Vfr)):
            _bitcmp(np.asarray(Vfr[j]), np.asarray(Vc_[j]),
                    name + '/V%d' % j, bad_e)
        r1[name] = dict(fields=len(FIELDS_R1), bit_identical=len(bad) == 0,
                        mismatches=bad[:5], entry_bit_identical=len(bad_e) == 0,
                        entry_mismatches=[x[0] for x in bad_e[:5]])
        if verbose:
            print('(R1/R2) %-16s forward pass %s, entry %s'
                  % (name, 'BIT-IDENTICAL' if not bad else 'MISMATCH %d' % len(bad),
                     'BIT-IDENTICAL' if not bad_e else 'MISMATCH'), flush=True)
    out['R1_R2'] = r1
    # also: without the ramp charge the acquisition fields are unchanged and
    # W differs exactly by dbar a_b in the first braking pair
    G, e0, eps0, dbar = cases['caseD_constants']
    R0 = chain(G, e0, eps0, dbar, ramp_charge=False)
    R1c = chain(G, e0, eps0, dbar, ramp_charge=True)
    acq_same = all(np.array_equal(R0[k], R1c[k]) for k in
                   ('Lam1', 'Lam2', 'taub', 'tset', 'E1', 'ebar', 'fbar',
                    'tvS2', 'x2tot', 'Tk', 'F'))
    out['ramp_charge_effect'] = dict(
        acquisition_fields_identical=bool(acq_same),
        W1_difference=float(R1c['W'][1] - R0['W'][1]),
        dbar_times_ab=float(dbar * G['a_b']))
    # ---- (R3) constant-plan reduction on the Case B takeover state: the
    # plan is the Case B reference spacing at constant speed v^xi, so the
    # takeover pair errors relative to the plan are the Case B errors e0
    RB = cp.evaluate(cfgB)
    KB = RB['consts']
    GBn = cp.namespace(cfgB)
    geo = cp.geometry(GBn, cfgB)
    n = 5
    ell = F(FIXED['ell']); d_sB = [F(x) for x in cfgB['d_s']]
    Delta = []; acc = Fr(0)
    for i in range(n):
        acc += d_sB[i] + ell; Delta.append(acc)
    s0 = [-Delta[i] for i in range(n)]
    v_xiB = F(str(cfgB['v_xi']))
    v0B = []; vp = v_xiB
    for x in cfgB['eps0']:
        vp = vp - F(x); v0B.append(vp)
    cfgR = config(v0=[str(x) for x in v0B], e0=[float(x) for x in geo['e0']],
                  d_s=cfgB['d_s'], c0=list(cfgB['c0']), lam=cfgB['lam'],
                  alpha=cfgB['alpha'], eps_e=cfgB['eps_e'],
                  eps_v=cfgB['eps_v'], vhnd=cfgB['vhnd'], dbar=cfgB['dbar'],
                  a_b=cfgB['a_b'], ramp_charge=True, one_sided=False,
                  plan=dict(family='constant', v_xi=str(cfgB['v_xi']),
                            T_xi=str(cfgB['T_xi']), s0=[str(x) for x in s0]))
    assert eps0_of(cfgR) == [float(x) for x in cfgB['eps0']]
    rR = evaluate(cfgR)
    KR = rR['consts']
    rows = []
    bad3 = []
    for c in range(1, n):
        pr = KB['pairs'][str(c)]
        pd = KR['pairs'][c + 1]
        for nm in ('H1', 'Ht', 'hm1', 'hm2', 'hmb_rule'):
            if not (pd[nm] == pr[nm]):
                bad3.append(('pair%d/%s' % (c + 1, nm), pd[nm], pr[nm]))
        h0B = float(geo['h0'][c])
        rows.append(dict(
            pair=c + 1, hstar=pd['hstar_min2'], mu_star=pd['mu2'],
            mu_equals_kappa_hstar=abs(pd['mu2'] - pd['hstar_min2']) < 1e-12,
            h_star_plus_htilde0=pd['hstar0'] + pd['h_tilde0'], h0_frozen=h0B,
            h0_diff=pd['hstar0'] + pd['h_tilde0'] - h0B,
            margin1_caseD=pd['slack_i'] + KR['NUMPAD'],
            margin1_frozen=h0B - pr['thr1'],
            margin2_caseD=pd['slack_ii'] + KR['NUMPAD'],
            margin2_frozen=h0B - pr['thr2'],
            margin3_caseD=pd['slack_iii'] + KR['NUMPAD'],
            margin3_frozen=h0B - pr['thr3'],
            Phi2D=pd['Phi2D'], Ph2_frozen=pr['Ph2'],
            Vrise_term_frozen=pr['Vrise'] * max(-float(GBn['Pi'][c]), 0.0),
            Phi2D_minus_frozen_plus_Vrise=pd['Phi2D'] - (
                pr['Ph2'] - pr['Vrise'] * max(-float(GBn['Pi'][c]), 0.0)),
            hmbD=pd['hmbD'], hmb_frozen=pr['hmb'],
            Ubrk_caseD=pd['Ubrk'], Ubrk_frozen=pr['Ubrk']))
    out['R3_constant_plan'] = dict(
        rows=rows, budget_mismatches=bad3,
        max_h0_diff=max(abs(r['h0_diff']) for r in rows),
        max_margin1_diff=max(abs(r['margin1_caseD'] - r['margin1_frozen']) for r in rows),
        max_margin2_diff=max(abs(r['margin2_caseD'] - r['margin2_frozen']) for r in rows),
        note='with a constant plan and the Case B takeover state the Case D '
             'assembly reproduces the frozen budgets H1, H_tail, h_min,1, '
             'h_min,2 and the RULE branch of h_min,b bit for bit, and the '
             'phase-1 and phase-2 admission margins up to rounding (h* + h~(0) '
             '= h(0), mu* = kappa h*); phase 3 differs by construction (local '
             'ramp: no V^rise term, received braking envelope a_b + F^b_{i-1} '
             'only; with ramp_charge the W_i are the frozen ones), so H4D(iii) '
             'has no exact reduction; the differences are listed per pair')
    ok_r1 = all(v.get('bit_identical', False) and v.get('entry_bit_identical', False)
                for v in r1.values())
    ok_r3 = (out['R3_constant_plan']['max_h0_diff'] < 1e-12
             and out['R3_constant_plan']['max_margin1_diff'] < 1e-12
             and out['R3_constant_plan']['max_margin2_diff'] < 1e-12
             and not bad3
             and all(abs(r['Phi2D_minus_frozen_plus_Vrise']) < 1e-12
                     for r in rows)
             and all(r['mu_equals_kappa_hstar'] for r in rows))
    out['R5_round4_design'] = regress_round4(verbose=verbose)
    out['R6_fan_plan'] = regress_fan(verbose=verbose)
    out['R7_round5_design'] = regress_round5(verbose=verbose)
    out['R8_ramped_fan'] = regress_ramp(verbose=verbose)
    out['ok'] = bool(ok_r1 and ok_r3 and acq_same
                     and out['R5_round4_design'].get('ok', False)
                     and out['R6_fan_plan']['ok']
                     and out['R7_round5_design'].get('ok', False)
                     and out['R8_ramped_fan']['ok'])
    if verbose:
        print('(R3) constant plan: max |h*+h~(0)-h(0)| = %.2e, margin1 diff %.2e, '
              'margin2 diff %.2e' % (out['R3_constant_plan']['max_h0_diff'],
                                     out['R3_constant_plan']['max_margin1_diff'],
                                     out['R3_constant_plan']['max_margin2_diff']))
        print('regression ok:', out['ok'])
    return out


ROUND4_DESIGN = os.path.join(CODE, 'data', 'new_d19', 'caseD_round4',
                             'design.json')


def _jsoncmp(a, b, path, bad):
    """exact comparison of json-like structures (floats bit for bit)."""
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            bad.append((path, 'keys')); return
        for k in a:
            _jsoncmp(a[k], b[k], path + '/' + str(k), bad)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            bad.append((path, 'length')); return
        for j, (x, y) in enumerate(zip(a, b)):
            _jsoncmp(x, y, '%s[%d]' % (path, j), bad)
    elif not (a == b or (a != a and b != b)):
        bad.append((path, a, b))


def regress_round4(verbose=True):
    """(R5) the round-4 design (canonical family, data/new_d19/caseD_round4/
    design.json) evaluated by this module reproduces the archived round-4
    certificate field by field (every condition, slack and constant; the
    run time excluded)."""
    if not os.path.exists(ROUND4_DESIGN):
        return dict(ok=False, note='archived round-4 design not found')
    d = json.load(open(ROUND4_DESIGN))
    r = evaluate(d['cfg'])
    a = json.loads(json.dumps(cp._py({k: v for k, v in r.items()
                                      if k != 'runtime_s'})))
    b = {k: v for k, v in d['certificate'].items() if k != 'runtime_s'}
    bad = []
    _jsoncmp(a, b, '', bad)
    out = dict(ok=len(bad) == 0, family=d['cfg']['plan']['family'],
               fields_compared='certificate (conds, consts, plan, admitted, '
                               'failed, one_sided)',
               mismatches=[str(x) for x in bad[:5]],
               T_f_bar=r['consts']['T_f_bar'],
               T_f_bar_archived=d['certificate']['consts']['T_f_bar'])
    if verbose:
        print('(R5) round-4 design:', 'BIT-IDENTICAL' if out['ok']
              else 'MISMATCH %s' % out['mismatches'], flush=True)
    return out


# row 9 of the round-5 candidate study (fan, head 0.3 m/s^2, 10 s approach,
# Case D constants of round 4): the exact crawl rate and the completion bound
# of the independent construction of the round-5 synthesis
FAN_R6 = dict(plan=dict(family='fan', v_xi='10.45', t_0='10', L_1='30.312',
                        delta='10.135', a_head='0.3', T_xi='80.853'),
              lam=0.11, vhnd=1e-3, a_c='60101/8533670',
              T_f_bar=81.9902978993945)


def regress_fan(verbose=True):
    """(R6) fan family: exact closure, crawl-speed and marker residuals, the
    premises (P1) to (P5), strictly decreasing main rates a_i = a_c + G/L_i,
    joins onto the crawl line, a float integration of the closure, and the
    crawl rate and completion bound of the independent construction of the
    candidate study (row 9)."""
    cfg = config(lam=FAN_R6['lam'], vhnd=FAN_R6['vhnd'], plan=FAN_R6['plan'])
    P = plan(cfg)
    ex = P.extra
    prem = P.premises(F(cfg['k']), F(cfg['v0'][0]))
    n = P.n
    rates_ok = all(ex['rates'][i] < ex['rates'][i - 1] for i in range(1, n)) and \
        all(ex['rates'][i] == ex['a_c'] + ex['G'] / ex['L'][i] for i in range(n))
    # unit i reaches the crawl line at t^c_i, where the head already crawls
    joins_ok = all(P.state(i, ex['t_c'][i])[1] == ex['v_join'][i]
                   and P.state(0, ex['t_c'][i])[1] == ex['v_join'][i]
                   for i in range(n))
    above = all(P.state(i, t)[1] > P.state(i - 1, t)[1]
                for i in range(1, n) for t in (ex['t_0'] + Fr(1, 1000),
                                               ex['t_c'][i - 1]))
    t = np.linspace(0.0, float(P.tau_star), 400001)
    dt = t[1] - t[0]
    V = np.array([P.state_f(i, t)[1] for i in range(n)])
    clos = [float(np.trapz(V[i] - V[i - 1], dx=dt)) for i in range(1, n)]
    clos_err = max(abs(clos[i - 1] - float(P.C[i])) for i in range(1, n))
    r = evaluate(cfg)
    out = dict(
        closure_residual=[str(x) for x in prem['closure_residual']],
        crawl_speed_residual=[str(x) for x in prem['crawl_speed_residual']],
        marker_residual=[str(x) for x in prem['marker_residual']],
        premises=dict(P1=prem['P1'], P2=prem['P2'], P3=prem['P3'],
                      P4=prem['P4'], P5=prem['P5']),
        rates=[str(x) for x in ex['rates']], rates_decreasing=rates_ok,
        joins_on_crawl_line=joins_ok, followers_above_predecessors=above,
        closure_float=clos, closure_float_err=clos_err,
        a_c=str(ex['a_c']), a_c_pinned=FAN_R6['a_c'],
        T_f_bar=r['consts']['T_f_bar'], T_f_bar_pinned=FAN_R6['T_f_bar'],
        admitted=r['admitted'])
    out['ok'] = bool(all(x == '0' for x in out['closure_residual'] +
                         out['crawl_speed_residual'] + out['marker_residual'])
                     and all(out['premises'].values()) and rates_ok and joins_ok
                     and above and clos_err < 1e-6
                     and out['a_c'] == FAN_R6['a_c']
                     and out['T_f_bar'] == FAN_R6['T_f_bar'] and r['admitted'])
    if verbose:
        print('(R6) fan plan: residuals exact %s, rates decreasing %s, joins %s, '
              'closure float err %.1e, pinned a_c and T_f_bar %s, ok %s'
              % (all(x == '0' for x in out['closure_residual']), rates_ok,
                 joins_ok, clos_err, out['a_c'] == FAN_R6['a_c'] and
                 out['T_f_bar'] == FAN_R6['T_f_bar'], out['ok']), flush=True)
    return out


ROUND5_DESIGN = os.path.join(CODE, 'data', 'new_d19', 'caseD_round5',
                             'design.json')


def regress_round5(verbose=True):
    """(R7) the round-5 design (fan with constant rates, data/new_d19/
    caseD_round5/design.json) evaluated by this module reproduces the
    archived round-5 certificate field by field (the run time excluded)."""
    if not os.path.exists(ROUND5_DESIGN):
        return dict(ok=False, note='archived round-5 design not found')
    d = json.load(open(ROUND5_DESIGN))
    r = evaluate(d['cfg'])
    a = json.loads(json.dumps(cp._py({k: v for k, v in r.items()
                                      if k != 'runtime_s'})))
    b = {k: v for k, v in d['certificate'].items() if k != 'runtime_s'}
    bad = []
    _jsoncmp(a, b, '', bad)
    out = dict(ok=len(bad) == 0, family=d['cfg']['plan']['family'],
               fields_compared='certificate (conds, consts, plan, admitted, '
                               'failed, one_sided)',
               mismatches=[str(x) for x in bad[:5]],
               T_f_bar=r['consts']['T_f_bar'],
               T_f_bar_archived=d['certificate']['consts']['T_f_bar'])
    if verbose:
        print('(R7) round-5 design:', 'BIT-IDENTICAL' if out['ok']
              else 'MISMATCH %s' % out['mismatches'], flush=True)
    return out


# a ramped fan on the Case D constants (round 6): ramp fraction 0.35 on the
# planning grid of 250 ms, head peak rate 0.45 m/s^2
FAN_R8 = dict(plan=dict(family='fan', v_xi='10.45', t_0='10', L_1='30.886',
                        delta='10.064', a_head='0.45', T_xi='86.142',
                        ramp='0.35', T_p='0.25'),
              lam=0.07, vhnd=1e-3, eps_e=0.05, eps_v=0.005)


def regress_ramp(verbose=True):
    """(R8) ramped fan plans: exact closure, crawl-speed and marker
    residuals; the premises (P1) to (P5); for every unit the staircase of
    the main phase is symmetric about the middle of the phase, lies on the
    planning grid, and removes exactly the drop G; the speed excess over
    the crawl line has the exact area G L_i/2; the peak rates decrease
    along the train; every jump lies on the 1 ms grid; a float integration
    of the closure; and ramp = 0 returns the fan with constant rates."""
    cfg = config(lam=FAN_R8['lam'], vhnd=FAN_R8['vhnd'], eps_e=FAN_R8['eps_e'],
                 eps_v=FAN_R8['eps_v'], plan=FAN_R8['plan'])
    P = plan(cfg)
    ex = P.extra
    n = P.n
    prem = P.premises(F(cfg['k']), F(cfg['v0'][0]))
    a_c = ex['a_c']; G = ex['G']; t_0 = ex['t_0']
    sym = True; drop = True; area = True; grid = True; onp = True
    for i in range(n):
        main = [(t, -a - a_c) for (t, a) in P.units[i] if t_0 <= t < ex['t_c'][i]]
        ends = [t for (t, r_) in main[1:]] + [ex['t_c'][i]]
        lens = [e_ - t for (t, r_), e_ in zip(main, ends)]
        vals = [r_ for (t, r_) in main]
        sym = sym and vals == vals[::-1] and lens == lens[::-1]
        drop = drop and sum(r_ * L_ for r_, L_ in zip(vals, lens)) == G
        # exact area of the speed excess E(tau) = G - int_0^tau r
        E = G; A = Fr(0)
        for r_, L_ in zip(vals, lens):
            A += E * L_ - r_ * L_ * L_ / 2
            E -= r_ * L_
        area = area and E == 0 and A == G * ex['L'][i] / 2
        grid = grid and all((t * 1000).denominator == 1 for (t, a) in P.units[i])
        N = ex['ramp_steps'][i]
        onp = onp and all(lens[m] == ex['T_p'] for m in range(N)) \
            and all(lens[-1 - m] == ex['T_p'] for m in range(N))
    rates_ok = all(ex['rates'][i] < ex['rates'][i - 1] for i in range(1, n))
    t = np.linspace(0.0, float(P.tau_star), 400001)
    dt = t[1] - t[0]
    V = np.array([P.state_f(i, t)[1] for i in range(n)])
    clos = [float(np.trapz(V[i] - V[i - 1], dx=dt)) for i in range(1, n)]
    clos_err = max(abs(clos[i - 1] - float(P.C[i])) for i in range(1, n))
    # ramp = 0: the fan with constant rates (same function, no ramp keys)
    pl0 = {k: v for k, v in FAN_R8['plan'].items() if k not in ('ramp', 'T_p')}
    P0 = plan(config(lam=FAN_R8['lam'], vhnd=FAN_R8['vhnd'], plan=pl0))
    Pz = plan(config(lam=FAN_R8['lam'], vhnd=FAN_R8['vhnd'],
                     plan=dict(pl0, ramp='0', T_p='0.25')))
    const_ok = all(P0.units[i] == Pz.units[i] and len(P0.units[i]) == 3
                   for i in range(n))
    r = evaluate(cfg)
    out = dict(
        closure_residual=[str(x) for x in prem['closure_residual']],
        crawl_speed_residual=[str(x) for x in prem['crawl_speed_residual']],
        marker_residual=[str(x) for x in prem['marker_residual']],
        premises=dict(P1=prem['P1'], P2=prem['P2'], P3=prem['P3'],
                      P4=prem['P4'], P5=prem['P5']),
        ramp_steps=list(ex['ramp_steps']), pieces=[len(u) for u in P.units],
        staircase_symmetric=bool(sym), drop_exact=bool(drop),
        area_exact=bool(area), jumps_on_grid=bool(grid),
        ramp_steps_on_planning_grid=bool(onp), peak_rates_decreasing=rates_ok,
        closure_float=clos, closure_float_err=clos_err,
        constant_rates_without_ramp=bool(const_ok),
        admitted=r['admitted'], T_f_bar=r['consts']['T_f_bar'])
    out['ok'] = bool(all(x == '0' for x in out['closure_residual'] +
                         out['crawl_speed_residual'] + out['marker_residual'])
                     and all(out['premises'].values()) and sym and drop and area
                     and grid and onp and rates_ok and clos_err < 1e-6
                     and const_ok and r['admitted'])
    if verbose:
        print('(R8) ramped fan: residuals exact %s, symmetric %s, drop %s, area '
              '%s, grid %s, closure float err %.1e, ok %s'
              % (all(x == '0' for x in out['closure_residual']), sym, drop, area,
                 grid, clos_err, out['ok']), flush=True)
    return out


# ============================================= brute-force checks (R4)
def plan_bruteforce(cfg=None, seed=20260927, n_box=9, verbose=True):
    """Brute-force checks of the Case D closed forms for cfg:
    (a) exact minima of g*, h*, mu*, eps* against a dense float grid;
    (b) plan: planned stops at the markers and closure residuals exact,
        float integration of v* against C_i;
    (c) T^{f+-}_k(R) and Env^+_k(R) against a dense grid in r and a dense
        grid of switch states inside the box (corner property);
    (d) F^+-_i(t_P) against a random sampling of layer states: every sum of
        admissible layer values lies below the bound;
    (e) monotonicity of F^+-_i and P_k on a grid."""
    cfg = config(**(cfg or {})) if (cfg is None or 'n' not in cfg) else cfg
    rng = np.random.default_rng(seed)
    P = plan(cfg)
    r = evaluate(cfg)
    K = r['consts']
    n = int(cfg['n'])
    lam = float(cfg['lam'])
    b = [float(a) / FIXED['v_c'] for a in cfg['amax']]
    b_F = [F(a) / F(FIXED['v_c']) for a in cfg['amax']]
    s_m = float(cfg['s_m'])
    out = {}
    # (a)
    worst = dict(g=0.0, h=0.0, mu=0.0, eps=0.0)
    below = True
    T_xi = float(P.T_xi)
    for c in range(1, n):
        for (loF, hiF) in ((Fr(0), min(Fr(K['tset'][c]), P.T_xi)),
                           (Fr(0), P.T_xi)):
            w = pair_minima(P, c, loF, hiF, b_F, Fr(1), F(cfg['s_m']))
            lo, hi = float(loF), float(hiF)
            t = np.linspace(lo, hi, 400001)
            # left limits at the right end: evaluate slightly inside
            t[-1] = hi - 1e-12
            sp, vp, A = P.state_f(c - 1, t)
            sf, vf, B = P.state_f(c, t)
            g = sp - sf - FIXED['ell'] - s_m
            e = vp - vf
            h = g + vp / b[c - 1] - vf / b[c]
            mu = e + A / b[c - 1] - B / b[c] + h
            for nm, arr in (('g', g), ('h', h), ('mu', mu), ('eps', e)):
                cf = float(w[nm][0]); gm = float(np.min(arr))
                worst[nm] = max(worst[nm], gm - cf)
                if cf > gm + 1e-9:
                    below = False
    out['a_minima'] = dict(grid_minus_closed_form_max=worst,
                           closed_form_not_above_grid=below,
                           note='grid step about 7e-5 s; the closed form is '
                                'the exact minimum, the grid value may exceed '
                                'it by the resolution')
    # (b)
    t = np.linspace(0.0, float(P.tau_star), 1000001)
    dt = t[1] - t[0]
    V = np.array([P.state_f(i, t)[1] for i in range(n)])
    clos = [float(np.trapz(V[i] - V[i - 1], dx=dt)) for i in range(1, n)]
    out['b_plan'] = dict(
        closure_numeric=clos, C=[float(x) for x in P.C[1:]],
        closure_numeric_err=max(abs(clos[i - 1] - float(P.C[i]))
                                for i in range(1, n)),
        closure_residual_exact=K['plan_premises']['closure_residual'],
        marker_residual_exact=K['plan_premises']['marker_residual'],
        stop_speeds=[float(P.state(i, P.tau_star)[1]) for i in range(n)])
    # (c)
    E1 = K['E1']; eps_det = float(cfg['eps_det'])
    worst_c = dict(Tfp=0.0, Tfm=0.0, Env=0.0)
    ok_c = True
    for k in range(n):
        corners = box_corners(0.0, E1[k], eps_det, True)
        es = np.linspace(-E1[k], 0.0, n_box); xs = np.linspace(-eps_det, 0.0, n_box)
        for R in (0.0, 1.0, 5.0, 12.0, 25.0):
            rr = R + np.linspace(0.0, 150.0 / lam, 60001)
            bp = 0.0; bm = 0.0; bx = 0.0
            for e in es:
                for x in xs:
                    p, q = _f_pq(lam, e, x)
                    f = (p + q * rr) * np.exp(-lam * rr)
                    pe, qe = _x_pq(lam, e, x)
                    xx = (pe + qe * rr) * np.exp(-lam * rr)
                    bp = max(bp, float(np.max(f))); bm = max(bm, float(np.max(-f)))
                    bx = max(bx, float(np.max(xx)))
            cp_ = float(Tf(lam, corners, R, +1)); cm_ = float(Tf(lam, corners, R, -1))
            cx_ = float(EnvX(lam, corners, R))
            for nm, br, cf in (('Tfp', bp, cp_), ('Tfm', bm, cm_), ('Env', bx, cx_)):
                worst_c[nm] = max(worst_c[nm], br - cf)
                if br > cf + 1e-12:
                    ok_c = False
    out['c_envelopes'] = dict(brute_minus_closed_form_max=worst_c,
                              closed_form_dominates=ok_c,
                              box_grid=n_box)
    # (d) random layer sums on every piece: evaluation time t in the piece,
    # receive times s_k in [t - (i-k) dbar, t], actual switch instants in
    # [tset_lo_k, tset_k], switch states in the one-sided box, forced parts
    # in [-gamma Lambda2_k, gamma Lambda2_k]; at most one S1 layer (-alpha)
    lam_ = lam; gamma = 2 * lam; dbar = float(cfg['dbar'])
    alpha = float(cfg['alpha'])
    tset = np.array(K['tset']); tlo = np.array(K['tset_lo'])
    Lam2 = np.array(K['Lam2']); Lam2[0] = 0.0
    worst_p = -np.inf; worst_m = -np.inf; ok_d = True
    for row in K['authority']:
        i = row['unit'] - 1
        for trial in range(300):
            t = rng.uniform(row['t0'], row['t1'])
            if trial < 20:
                t = row['t0']
            tot = 0.0; s1 = False
            for kk in range(i + 1):
                s_k = t - rng.uniform(0.0, (i - kk) * dbar)
                tsw = rng.uniform(tlo[kk], tset[kk])
                rclk = s_k - tsw
                if rclk < 0:
                    if not s1 and rng.uniform() < 0.5:
                        tot += -alpha; s1 = True      # S1, braking sign
                    continue                            # COAST
                e = rng.uniform(-E1[kk], 0.0); x = rng.uniform(-eps_det, 0.0)
                if rng.uniform() < 0.3:
                    e = rng.choice([-E1[kk], 0.0]); x = rng.choice([-eps_det, 0.0])
                p, q = _f_pq(lam_, e, x)
                fr = (p + q * rclk) * math.exp(-lam_ * rclk)
                tot += fr + rng.uniform(-1, 1) * gamma * Lam2[kk]
            worst_p = max(worst_p, tot - row['F_plus'])
            worst_m = max(worst_m, -tot - row['F_minus'])
            if tot > row['F_plus'] + 1e-15 or -tot > row['F_minus'] + 1e-15:
                ok_d = False
    out['d_layer_sums'] = dict(max_sample_minus_Fplus=float(worst_p),
                               max_neg_sample_minus_Fminus=float(worst_m),
                               bound_holds=ok_d)
    # (e) monotonicity of the envelopes on a grid
    mono = True
    for i in range(n):
        vals = []
        for tt in np.linspace(0.0, T_xi, 301):
            fp = 0.0
            for kk in range(i + 1):
                Rk = max(tt - (i - kk) * dbar - tset[kk], 0.0)
                corners = box_corners(0.0, E1[kk], eps_det, True)
                fp += float(Tf(lam, corners, Rk, +1)) + gamma * Lam2[kk]
            vals.append(fp)
        if np.any(np.diff(vals) > 1e-15):
            mono = False
    out['e_monotone_Fplus'] = mono
    out['ok'] = bool(below and ok_c and ok_d and mono
                     and out['b_plan']['closure_numeric_err'] < 1e-6
                     and all(x == '0' for x in out['b_plan']['closure_residual_exact'])
                     and all(x == '0' for x in out['b_plan']['marker_residual_exact']))
    if verbose:
        print('(R4) brute force:', json.dumps(cp._py(dict(
            a=out['a_minima']['closed_form_not_above_grid'],
            a_worst=out['a_minima']['grid_minus_closed_form_max'],
            b=out['b_plan']['closure_numeric_err'],
            c=out['c_envelopes']['closed_form_dominates'],
            c_worst=out['c_envelopes']['brute_minus_closed_form_max'],
            d=out['d_layer_sums'], e=mono, ok=out['ok']))), flush=True)
    return out


def summary(res):
    K = res.get('consts', {})
    return dict(admitted=res['admitted'], failed=res['failed'],
                slacks={k: c['slack'] for k, c in res['conds'].items()},
                T_f_bar=K.get('T_f_bar'), T_ent_bar=K.get('T_ent_bar'),
                settling_bound=K.get('settling_bound'),
                align=K.get('align'), gaperr_max=None if not K else max(K['gaperr']))


if __name__ == '__main__':
    args = sys.argv[1:] or ['eval']
    for a in args:
        if a == 'eval':
            r = evaluate(config())
            print(json.dumps(cp._py(summary(r)), indent=1))
        elif a == 'regress':
            rep = regress()
            os.makedirs(DATA_D, exist_ok=True)
            json.dump(cp._py(rep), open(os.path.join(DATA_D, 'regression.json'),
                                        'w'), indent=1)
        elif a == 'brute':
            dp = os.path.join(DATA_D, 'design.json')
            cfg = json.load(open(dp))['cfg'] if os.path.exists(dp) else None
            rep = plan_bruteforce(cfg)
            rep['cfg'] = cfg
            json.dump(cp._py(rep), open(os.path.join(DATA_D, 'bruteforce.json'),
                                        'w'), indent=1)
        else:
            raise SystemExit('unknown part %s' % a)
