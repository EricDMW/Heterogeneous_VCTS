#!/usr/bin/env python3
"""certify_profile.py -- dispatch-certificate evaluator under Assumption 1'
(Case C, 2026-09-26): piecewise-constant reference acceleration with one
deceleration segment during acquisition.

Reference (s_0(0) = 0):
    v_r(t) = v^xi                  on [0, t_d),
             v^xi - a_d (t - t_d)  on [t_d, t_c),   t_c = t_d + (v^xi - v_c1)/a_d,
             v_c1                  on [t_c, T_xi),
             v_c1 - a_b (t - T_xi) on [T_xi, tau_r), tau_r = T_xi + v_c1/a_b,
             0                     afterwards,
    s_ref,stop = s_0(T_xi) + v_c1^2/(2 a_b),  markers s_ref,stop - Delta_i.
The head law carries a_r(t) as feedforward before T_xi and brakes open
loop afterwards; the followers are unchanged.

The module executes the FROZEN chain of certify_v30.py through
b2_design._namespace (constant overrides only) and reuses every frozen
quantity whose derivation is unchanged under Assumption 1':
  * the acquisition chain up to the switch deadlines (Lambda1, tau_bar,
    t_bar, t_set deadlines, E1, e_bar, the corner statistics, T_k, L_eps),
  * the braking recursion (W, eps_b, e_b, f_b, Theta, F^b, F^cum, q_k,
    dispersion, late, schedule, gap errors),
  * the budget H1, the thresholds h_min,1 and the RULE branch of h_min,b,
  * the free-tail functionals and the helpers of the entry recursion.
It re-derives (Sections (i)-(viii) of the design note):
  * the S2-window ledger Lambda2 with the feedforward charge 2 a_d dbar for
    every pair i >= 2, and the dependent S2 envelopes (x2tot, fbar, tvS2),
    the sums F_i, the pre-ramp envelopes P^pre and the received-command
    envelopes (Ubrk);
  * H_tail with the Pi-term of the reference deceleration, Phi2 with
    V^brk at v_c1, h_min,2 with the received-command envelope F_i + a_d,
    h_min,b with the re-derived Ubrk; the authority demands with + a_d on
    the acquisition branch;
  * the time-resolved velocity floor (H6) and the handoff floor v_flr,c
    used by H8;
  * the clearance floors with v_c1 in the braking row and a crawl row;
  * the alignment and completion bounds at v_c1;
  * the entry recursion with the feedforward jumps of the head layer.
With a_d = 0 and v_c1 = v^xi every re-derived quantity equals the frozen
one bit for bit (regression against b2_results.json 'cert' and
v30_values.json; see regress()).

Refinements of 2026-09-26 (cfg['refine'], default True; Sections (R1),
(R2) of the design note; both are inactive without a segment):
  (R1) time-resolved authority on the deceleration segment: the
       feedforward a_d is added to the certified feedback envelopes of
       the layers at the times at which they are received, not to the
       whole-acquisition peak demand; needs the premise
       t_d >= T_sw_bar + (n-1) dbar, which enters P_seg;
  (R2) one-sided velocity floor: the upper bound on sum_k eps_k(t) that
       the floor v_i = v_r - sum_k eps_k needs is taken as the signed
       supremum of the free response over the asymmetric switch-state
       box e^sw in [e_k(0) - E1_k, e_k(0) + E1_k], |eps^sw| <= eps_det,
       instead of the supremum of |eps_k| over the symmetric box.
Design inputs of cfg: eps_e, eps_v (handoff box), vhnd (handoff pad
V^hnd, frozen constant VHND_PAD of certify_v30.py, set in the executed
namespace), alpha (S1 rate).

The evaluation is floating point (IEEE double).  interval_profile.py
(when present) encloses the closed-form conditions of a configuration
with outward-rounded intervals and Arb exponentials.

API
  evaluate(cfg)        -> dict (conds, consts, admitted, ...)
  core(cfg)            -> the T_xi-independent part (expensive)
  finalize(C, T_xi)    -> evaluate() result from a core
  age_ceilings(cfg)    -> per-condition age-bound crossings (bisection)
  age_boundary_ms(cfg) -> largest admitted age bound on the 1 ms grid
  reference(cfg)       -> profile constants and s_0, v_r, a_r callables
  scan(grid)           -> grid evaluation, writes data/new_d19/caseC/scan.json
  regress()            -> regression against the archived certificates
"""
import json
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import b2_design as bd                                   # noqa: E402

CODE = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_C = os.path.join(CODE, 'data', 'new_d19', 'caseC')
SCAN_PATH = os.path.join(DATA_C, 'scan.json')

# frozen constants of certify_v30.py without an override hook
FIXED = dict(U=1.6, ell=25.0, kappa=1.0, v_c=1.2)

# Case B (primary benchmark) constants; Case C changes t_d, a_d, v_c1,
# T_xi, c0, lam
CASE_B = dict(n=5, v_xi=10.8, a_b=1.0, alpha=0.35, lam=0.055, s_m=1.0,
              dbar=0.02, eps_e=0.15, eps_v=0.02,
              eps0=[-0.30, 0.25, 0.18, 0.12, 0.08],
              d_s=[6.0, 5.0, 5.0, 5.0, 5.0],
              amax=[1.60, 1.48, 1.38, 1.29, 1.21],
              e0_head=-0.5, T_c=0.001, eps_det=0.015, phi=0.005,
              c0=(42.0, 45.0, 47.0, 49.0), T_xi=165.0,
              t_d=10.0, a_d=0.0, v_c1=None,
              vhnd=1e-3, refine=True,
              sync_tol=0.9, align_tol=3.0, gap_tol=1.0)

COND_NAMES = ('H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8',
              'C_hnd', 'C_sync', 'C_al', 'C_gap', 'P_seg')
# run_b2 / certify_v30 short names of the condition families
SHORT = dict(auth='H1', ledger='H2', hold='H3', adm='H4', clr='H5',
             vel='H6', event='H7', ramp='H8', hnd='C_hnd', sync='C_sync',
             align='C_al', gap='C_gap')


# ----------------------------------------------------------------- config
def config(**kw):
    """Case B constants updated by kw; v_c1 = None means v_c1 = v^xi.
    d_req (stop-time dispersion requirement of C_sync, s) is an alias of
    sync_tol: when given and not None it sets sync_tol (Case C of
    2026-09-26 uses d_req = 1.0; the default 0.9 is that of Case B)."""
    cfg = dict(CASE_B)
    cfg.update(kw)
    if cfg.get('v_c1') is None:
        cfg['v_c1'] = cfg['v_xi']
    if cfg.get('d_req') is not None:
        cfg['sync_tol'] = float(cfg['d_req'])
    cfg['d_req'] = float(cfg['sync_tol'])
    return cfg


def _segment(cfg):
    """(has_segment, t_d, t_c, a_d, v_c1); a_d = 0 or v_c1 = v^xi means
    Assumption 1 (no segment)."""
    a_d = float(cfg.get('a_d', 0.0) or 0.0)
    v_c1 = cfg.get('v_c1')
    v_c1 = float(cfg['v_xi']) if v_c1 is None else float(v_c1)
    if a_d <= 0.0 or v_c1 >= float(cfg['v_xi']):
        return False, None, None, 0.0, float(cfg['v_xi'])
    t_d = float(cfg['t_d'])
    t_c = t_d + (float(cfg['v_xi']) - v_c1) / a_d
    return True, t_d, t_c, a_d, v_c1


def reference(cfg):
    """Profile constants and callables s0(t), vr(t), ar(t) of Assumption 1'
    (s_0(0) = 0), plus the markers s_ref,stop - Delta_i."""
    seg, t_d, t_c, a_d, v_c1 = _segment(cfg)
    v_xi = float(cfg['v_xi']); a_b = float(cfg['a_b'])
    T_xi = float(cfg['T_xi'])
    if not seg:
        t_d = t_c = T_xi
    tau_r = T_xi + v_c1 / a_b
    s_td = v_xi * t_d
    s_tc = s_td + v_xi * (t_c - t_d) - 0.5 * a_d * (t_c - t_d) ** 2
    s_Txi = s_tc + v_c1 * (T_xi - t_c)
    s_stop = s_Txi + v_c1 ** 2 / (2.0 * a_b)

    def vr(t):
        t = np.asarray(t, float)
        return np.where(t < t_d, v_xi,
               np.where(t < t_c, v_xi - a_d * (t - t_d),
               np.where(t < T_xi, v_c1,
               np.where(t < tau_r, v_c1 - a_b * (t - T_xi), 0.0))))

    def ar(t):
        t = np.asarray(t, float)
        return np.where(t < t_d, 0.0,
               np.where(t < t_c, -a_d,
               np.where(t < T_xi, 0.0,
               np.where(t < tau_r, -a_b, 0.0))))

    def s0(t):
        t = np.asarray(t, float)
        d1 = np.clip(t - t_d, 0.0, t_c - t_d)
        d3 = np.clip(t - T_xi, 0.0, tau_r - T_xi)
        return (v_xi * np.minimum(t, t_d)
                + v_xi * d1 - 0.5 * a_d * d1 ** 2
                + v_c1 * np.clip(t - t_c, 0.0, T_xi - t_c)
                + v_c1 * d3 - 0.5 * a_b * d3 ** 2)

    d_s = np.array(cfg['d_s'], float)
    Delta = np.cumsum(d_s + FIXED['ell'])
    return dict(segment=seg, t_d=t_d, t_c=t_c, a_d=a_d, v_c1=v_c1,
                T_xi=T_xi, tau_r=tau_r, s0_td=s_td, s0_tc=s_tc,
                s0_Txi=s_Txi, s_ref_stop=s_stop,
                markers=(s_stop - Delta).tolist(), s0=s0, vr=vr, ar=ar)


# -------------------------------------------------------------- namespace
_NS = OrderedDict()          # bounded cache of executed frozen namespaces
_NS_MAX = 32                 # each namespace holds about 10 MB of grids


def _overrides(cfg):
    lam = float(cfg['lam'])
    n = int(cfg['n'])
    ov = dict(n=n, v_xi=float(cfg['v_xi']), a_b=float(cfg['a_b']),
              alpha=float(cfg['alpha']), lam=lam, beta=lam * lam,
              gamma=2 * lam, s_m=float(cfg['s_m']),
              eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
              eps0=[float(x) for x in cfg['eps0']],
              e0=[float(cfg['e0_head'])] + [37.0] * (n - 1),
              amax=[float(x) for x in cfg['amax']],
              T_c=float(cfg['T_c']), eps_det=float(cfg['eps_det']),
              phi=float(cfg['phi']), g_star=4.2,
              ALIGN_TOL=float(cfg['align_tol']))
    return ov


def namespace(cfg):
    """Frozen-definition namespace for cfg (cached; T_xi and dbar are
    evaluation arguments, not namespace constants)."""
    for k, v in FIXED.items():
        if k in cfg and abs(float(cfg[k]) - v) > 0:
            raise ValueError('%s is frozen at %g in certify_v30.py' % (k, v))
    ov = _overrides(cfg)
    vhnd = float(cfg.get('vhnd', 1e-3))
    key = tuple((k, tuple(v) if isinstance(v, list) else v)
                for k, v in sorted(ov.items())) + (('vhnd', vhnd),)
    if key in _NS:
        _NS.move_to_end(key)
        return _NS[key]
    full = dict(ov, T_xi=float(cfg['T_xi']), dbar=float(cfg['dbar']))
    G = bd._namespace(full)
    # The handoff pad V^hnd is the frozen module constant VHND_PAD, read at
    # call time by forward_pass (W_i = dbar (V^hnd + a_b + ...)) and by the
    # C_hnd test; no derived constant of the executed prefix uses it, so
    # setting it in the namespace is the exact analogue of an override.
    G['VHND_PAD'] = vhnd
    _NS[key] = G
    while len(_NS) > _NS_MAX:
        _NS.popitem(last=False)
    return G


def geometry(G, cfg):
    """Takeover geometry.  With cfg['c0'] (physical initial gaps of pairs
    2..n) the arithmetic of b2_design.assemble is used; with cfg['e0']
    (closure errors, head included) that of the frozen featured block."""
    n = G['n']; s_m = G['s_m']; v_xi = G['v_xi']
    eps0 = G['eps0']; b = G['b']
    d_s = np.array(cfg['d_s'], float)
    if cfg.get('c0') is not None:
        c0 = np.array(cfg['c0'], float)
        e0 = np.array([float(cfg['e0_head'])]
                      + [c0[i - 1] - d_s[i] for i in range(1, n)])
        g_init = np.array([0.0] + [c0[i - 1] - s_m for i in range(1, n)])
    else:
        e0 = np.array(cfg['e0'], float)
        c0 = None
        g_init = np.array([0.0] + [d_s[i] - s_m + e0[i]
                                   for i in range(1, n)])
    G['e0'][:] = e0                       # the frozen budgets read the global
    v = np.zeros(n); vp = v_xi
    for i in range(n):
        v[i] = vp - eps0[i]; vp = v[i]
    h0 = np.array([0.0] + [g_init[i] + v[i - 1] / b[i - 1] - v[i] / b[i]
                           for i in range(1, n)])
    return dict(d_s=d_s, c0=c0, e0=e0, v0=v, g_init=g_init, h0=h0)


# ------------------------------------------------------- entry recursion
LAG_FROZEN = (15, 20)      # frozen lag grids: B in {0,2,..,28}, A in {0,..,38}
LAG_SEGMENT = (41, 41)     # with a segment: both grids extended to 80 s


_ENTRY_CACHE = OrderedDict()   # (eps_e, eps_v)-independent entry curves
_ENTRY_CACHE_MAX = 2           # about 30 MB per entry


def _entry_key(G, R, Lam2f, dbar, ff, lags):
    """Every input that _entry_curves reads: the frozen-namespace constants
    of the modal family and the S1 rate, the acquisition-chain values and
    the ledger; the handoff box (eps_e, eps_v) and the pad are not among
    them, so the curves are shared across the tolerance grid."""
    return (float(G['lam']), tuple(float(x) for x in G['alpha_g']),
            float(G['eps_det']), float(dbar),
            tuple(float(x) for x in R['ebar']),
            tuple(float(x) for x in R['tset']),
            tuple(float(x) for x in R['tact']),
            tuple(float(x) for x in R['fsw']),
            tuple(float(x) for x in Lam2f), ff, lags)


def _entry(G, R, Lam2f, dbar, ff=None, lags=None):
    """certified_entry of certify_v30.py with the S2-window ledger Lam2f
    (feedforward charge included) and, for ff = (a_d, t_d, t_c), the two
    feedforward jumps of the head layer in the tail-variation recursion.
    With Lam2f = R['Lam2'] and ff = None the result is bit-identical to
    the frozen function.  lags = (nB, nA) sets the fixed lag grids
    {0, 2, ..., 2(nB-1)} and {0, 2, ..., 2(nA-1)} of the forced tails
    (every lag gives a valid bound; the minimum over a larger grid is
    still a bound).  Returns (Tent lags, V list, SxL list).  The entry
    envelopes totE_i, totX_i and the tail variations V_j do not depend on
    the handoff box; _entry_curves computes and caches them, and this
    function applies the box test (totE_i <= eps_e) & (totX_i <= eps_v)."""
    lags = lags or (LAG_FROZEN if ff is None else LAG_SEGMENT)
    cur = _entry_curves(G, R, Lam2f, dbar, ff, lags)
    n = G['n']; TGRID = G['TGRID']
    eps_e = G['eps_e']; eps_v = G['eps_v']
    Tent_c = np.zeros(n)
    for i in range(n):
        ok = (cur['totE'][i] <= eps_e) & (cur['totX'][i] <= eps_v)
        if ok.any():
            m0 = int(np.argmax(ok))
            assert ok[m0:].all(), "entry envelopes must be nonincreasing"
            Tent_c[i] = TGRID[m0]
        else:
            Tent_c[i] = np.inf
    return Tent_c, cur['V'], cur['SxL']


def _entry_curves(G, R, Lam2f, dbar, ff, lags):
    """Tail variations V_j, corner tail sups (SeL, SxL) and the entry
    envelopes (totE_i, totX_i) of the recursion (cached)."""
    key = _entry_key(G, R, Lam2f, dbar, ff, lags)
    if key in _ENTRY_CACHE:
        _ENTRY_CACHE.move_to_end(key)
        return _ENTRY_CACHE[key]
    nB, nA = lags
    n = G['n']; NT = G['NT']; TGRID = G['TGRID']; dtg = G['dtg']
    alpha_g = G['alpha_g']; gamma = G['gamma']; lam = G['lam']
    EULER = G['EULER']
    _corner_tails = G['_corner_tails']; _shift_ceil = G['_shift_ceil']
    _Psi = G['_Psi']; L1FP_X = G['L1FP_X']; _tail_sup = G['_tail_sup']
    Wtot = Lam2f + 1e-15
    SeL, SxL, FTL = [], [], []
    for k in range(n):
        Se, Sx, FT = _corner_tails(R['ebar'][k])
        SeL.append(Se); SxL.append(Sx); FTL.append(FT)
    V = [np.zeros(NT)]
    Bset = [2.0 * m for m in range(nB)]
    for j in range(1, n + 1):
        jj = j - 1
        Vs = _shift_ceil(V[jj], dbar)
        sh = R['tset'][jj]
        idx = np.maximum(np.floor((TGRID - sh) / dtg).astype(int), 0)
        idx = np.minimum(idx, NT - 1)
        ft = FTL[jj][idx]
        jmp = np.where(TGRID <= R['tact'][jj] + 1e-12, alpha_g[jj], 0.0) \
            + np.where(TGRID <= R['tset'][jj] + 1e-12,
                       alpha_g[jj] + R['fsw'][jj], 0.0)
        if jj == 0 and ff is not None:
            a_d, t_d, t_c = ff
            # feedforward steps -a_d at t_d and +a_d at t_c of the head
            # command: TV a_d each on every tail that contains them
            jmp = jmp + np.where(TGRID <= t_d + 1e-12, a_d, 0.0) \
                + np.where(TGRID <= t_c + 1e-12, a_d, 0.0)
        if jj == 0:
            forc = np.zeros(NT)
        else:
            forc = gamma * dbar * Vs
            ct = np.full(NT, np.inf)
            for B in Bset:
                ct = np.minimum(ct, _Psi(B) * Wtot[jj]
                                + L1FP_X * dbar * _shift_ceil(V[jj], B + dbar))
            forc = forc + ct
        V.append(Vs + ft + jmp + forc)
    totEL = []; totXL = []
    Aset = [2.0 * m for m in range(nA)]
    for i in range(n):
        totE = SeL[i].copy(); totX = SxL[i].copy()
        if i >= 1:
            fbE = np.full(NT, np.inf); fbX = np.full(NT, np.inf)
            for A in Aset:
                Vsh = dbar * _shift_ceil(V[i], A + dbar)
                kE = float(_tail_sup(0.0, 1.0, np.array([A]))[0])
                kX = float(_tail_sup(1.0, -lam, np.array([A]))[0])
                fbE = np.minimum(fbE, kE * Wtot[i] + (1.0 / (lam * EULER)) * Vsh)
                fbX = np.minimum(fbX, kX * Wtot[i] + 1.0 * Vsh)
            totE = totE + fbE; totX = totX + fbX
        totEL.append(totE); totXL.append(totX)
    cur = dict(V=V, SeL=SeL, SxL=SxL, totE=totEL, totX=totXL)
    _ENTRY_CACHE[key] = cur
    while len(_ENTRY_CACHE) > _ENTRY_CACHE_MAX:
        _ENTRY_CACHE.popitem(last=False)
    return cur


# ------------------------------------------------ refinement helpers
def _tail_sup_signed(p, q, T, lam):
    """sup_{r >= T} (p + q r) e^{-lam r} exactly (vectorized in T >= 0).
    The limit at infinity is zero, so the supremum is at least zero; for
    q > 0 the interior stationary point ts = (q - lam p)/(lam q) is a
    maximum with value (q/lam) e^{-lam ts}, for q < 0 it is a minimum."""
    T = np.asarray(T, float)
    gT = (p + q * T) * np.exp(-lam * T)
    out = np.maximum(gT, 0.0)
    if q > 0.0:
        ts = (q - lam * p) / (lam * q)
        gs = (q / lam) * np.exp(-lam * max(ts, 0.0))
        out = np.where(T < ts, np.maximum(out, gs), out)
    return out


def _corner_fsup(G, ebar_k, T):
    """max over the four corners of the symmetric switch-state box of
    sup_{r >= T} |f_free(r)|, f_free = beta e_free + gamma eps_free =
    lam^2 e^sw (1 - lam r) e^{-lam r} + lam eps^sw (2 - lam r) e^{-lam r}
    (the frozen modal form (p + q r) e^{-lam r} with p = beta e^sw +
    gamma eps^sw and q = -lam^2 (eps^sw + lam e^sw))."""
    lam = G['lam']; beta = G['beta']; gamma = G['gamma']
    eps_det = G['eps_det']; _tail_sup = G['_tail_sup']
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e0c, x0c = se * ebar_k, sx * eps_det
            c = x0c + lam * e0c
            out = np.maximum(out, _tail_sup(beta * e0c + gamma * x0c,
                                            -lam ** 2 * c, T))
    return out


def _corner_xsup_signed(G, e_lo, e_hi, T):
    """max over the four corners of the asymmetric switch-state box
    e^sw in [e_lo, e_hi], eps^sw in {-eps_det, +eps_det} of
    sup_{r >= T} eps_free(r), eps_free = (eps^sw - lam c r) e^{-lam r},
    c = eps^sw + lam e^sw.  eps_free is linear in the switch state, so the
    corners attain the supremum over the box at every r."""
    lam = G['lam']; eps_det = G['eps_det']
    T = np.asarray(T, float)
    out = np.zeros_like(T)
    for e0c in (e_lo, e_hi):
        for x0c in (eps_det, -eps_det):
            c = x0c + lam * e0c
            out = np.maximum(out, _tail_sup_signed(x0c, -lam * c, T, lam))
    return out


_ENV_CACHE = OrderedDict()     # grid curves of (R1) and (R2) per pair
_ENV_CACHE_MAX = 4


def _env_curves(G, ebar, e0, E1):
    """Per pair k on the 1 ms grid: Fsup_k(T) = corner max of
    sup_{r >= T} |f_free,k(r)| (symmetric box, (R1)) and Sxp_k(T) = corner
    max of sup_{r >= T} eps_free,k(r) (asymmetric box, (R2)).  Both depend
    only on the modal constants and the switch-state boxes (cached)."""
    key = (float(G['lam']), float(G['eps_det']), tuple(float(x) for x in ebar),
           tuple(float(x) for x in e0), tuple(float(x) for x in E1))
    if key in _ENV_CACHE:
        _ENV_CACHE.move_to_end(key)
        return _ENV_CACHE[key]
    TGRID = G['TGRID']; n = G['n']
    cur = dict(Fsup=[_corner_fsup(G, ebar[k], TGRID) for k in range(n)],
               Sxp=[_corner_xsup_signed(G, e0[k] - E1[k], e0[k] + E1[k], TGRID)
                    for k in range(n)])
    _ENV_CACHE[key] = cur
    while len(_ENV_CACHE) > _ENV_CACHE_MAX:
        _ENV_CACHE.popitem(last=False)
    return cur


# --------------------------------------------------------------- core
def core(cfg, want_entry=True):
    """Everything that does not depend on T_xi: frozen forward pass,
    re-derived S2-window chain, budgets, thresholds, floors, authority,
    entry lags and tail variations, velocity-floor curves."""
    cfg = config(**cfg)
    G = namespace(cfg)
    dbar = float(cfg['dbar'])
    n = G['n']; b = G['b']; Pi = G['Pi']; eta = G['eta']
    alpha_g = G['alpha_g']; A_run = G['A_run']; kappa = G['kappa']
    a_b = G['a_b']; v_xi = G['v_xi']; eps_v = G['eps_v']; eps_e = G['eps_e']
    eps_det = G['eps_det']; eps0 = G['eps0']; Umin = G['Umin']
    NUMPAD = G['NUMPAD']; T_c = G['T_c']; phi = G['phi']
    GAIN_X = G['GAIN_X']; GAIN_F = G['GAIN_F']; TV_KIMP = G['TV_KIMP']
    seg, t_d, t_c, a_d, v_c1 = _segment(cfg)
    dv = v_xi - v_c1                       # reference speed decrease
    refine = bool(cfg.get('refine', True)) and seg   # (R1), (R2) need a segment
    gamma = G['gamma']; lam = G['lam']

    geo = geometry(G, cfg)
    g_init = geo['g_init']; h0 = geo['h0']
    R = G['forward_pass'](dbar)
    out = dict(cfg=cfg, G=G, geo=geo, seg=seg, t_d=t_d, t_c=t_c, a_d=a_d,
               v_c1=v_c1, dbar=dbar, closed=R is not None, R=R,
               refine=refine)
    if R is None:
        return out

    # ---- S2-window chain with the feedforward charge (pre-values frozen)
    Lam1 = R['Lam1']; ebar = R['ebar']; tset = R['tset']; tact = R['tact']
    Lam2p = R['Lam2']
    Lam2f = np.zeros(n); x2f = np.zeros(n); fbf = np.zeros(n)
    tvf = np.zeros(n)
    for k in range(n):
        st = G['corner_stats'](ebar[k], eps_det)
        s = np.sum(tvf[:k])
        if seg and k >= 1:
            s = s + 2.0 * a_d
        Lam2f[k] = dbar * s
        x2f[k] = st['sup_x'] + GAIN_X * Lam2f[k]
        fbf[k] = st['sup_f'] + GAIN_F * Lam2f[k]
        tvf[k] = st['tv_f'] + TV_KIMP * Lam2f[k]
    Ff = np.array([np.sum(fbf[:i]) for i in range(n)])
    fb = R['fb']; Fb = R['Fb']; Fcum = R['Fcum']; f_xi = R['f_xi']
    W = R['W']; epsb = R['epsb']; IX = R['IX']; IF = R['IF']
    Dq = R['Dq']; DBq = R['DBq']
    Pp = np.zeros(n)
    for u in range(1, n):
        Pp[u] = fb[u] + max(Ff[u], Pp[u - 1])

    # ---- budgets, thresholds, margins per pair (code index i = pair i+1)
    pairs = {}
    for i in range(1, n):
        P = R['pairs'][i]
        H1 = P['H1']; hm1 = P['hm1']
        V2 = 2.0 * np.sum(x2f[:i])
        Ht = Dq[i] + V2 * abs(Pi[i]) + Lam2f[i] * (1.0 / b[i] + DBq[i])
        dPi = 0.0
        if dv > 0.0 and Pi[i] > 0.0:
            dPi = dv * Pi[i]                # (vii) telescoped Pi-term
            Ht = Ht + dPi
        Vbrk = v_c1 + i * eps_v
        Vrise = (i - 1) * dbar * Pp[i - 1]
        bmin = min(b[i], b[i - 1])
        anet = a_b - Fb[i]
        Ph2 = IX[i] + epsb[i] / bmin + epsb[i] ** 2 / (2.0 * anet) + IF[i] / b[i] \
            + Vbrk * max(Pi[i], 0.0) + Vrise * max(-Pi[i], 0.0) \
            + W[i] * (1.0 / b[i] + DBq[i])
        Fi = Ff[i] + a_d if seg else Ff[i]
        hm2 = (fbf[i] + b[i] * x2f[i] + Fi * eta[i]) / (kappa * b[i])
        dem_prev = Ff[i - 1] + max(A_run[i - 1], fbf[i - 1])
        Ubrk = max(dem_prev, Pp[i - 1], a_b + Fb[i],
                   (a_b + Fb[i - 1]) + fbf[i - 1])
        hmb_s2 = (fb[i] + b[i] * epsb[i] + Ubrk * eta[i]) / (kappa * b[i])
        hmb_rule = P['hmb_rule']
        hmb = max(hmb_s2, hmb_rule)
        thr1 = H1 + hm1
        thr2 = H1 + Ht + hm2
        thr3 = H1 + Ht + Ph2 + hmb
        thr = max(thr1, thr2, thr3)
        uh_acq = h0[i] - H1
        uh = uh_acq - Ht
        bf = uh - Ph2
        pairs[i] = dict(H1=H1, Ht=Ht, Ht_Pi_term=dPi, Ph2=Ph2, hm1=hm1,
                        hm2=hm2, hmb=hmb, hmb_s2=hmb_s2, hmb_rule=hmb_rule,
                        thr1=thr1, thr2=thr2, thr3=thr3, thr=thr,
                        h0=h0[i], margin=h0[i] - thr,
                        padded=h0[i] - thr - NUMPAD,
                        sigma=kappa * b[i] * (h0[i] - thr),
                        V1=P['V1'], V2=V2, Vbrk=Vbrk, Vrise=Vrise,
                        Ubrk=Ubrk, dem_prev=dem_prev, anet=anet,
                        uh_acq=uh_acq, uh=uh, bf=bf)
    margins = np.array([pairs[i]['padded'] for i in range(1, n)])

    # ---- authority (H1)
    T_sw_bar = float(np.max(tset))
    TGRID = G['TGRID']; dtg = G['dtg']; NT = G['NT']
    seg_dem = None; seg_dem_cf = None; seg_parts = None; seg_premise = None
    if refine:
        # (R1) time-resolved demand on the deceleration segment.  For
        # t in [t_d, t_c + (i-1) dbar) unit i may hold the feedforward
        # -a_d; layer k <= i of its command is the S2 feedback of pair k
        # received at theta_{k+1:i}(t) in [t - (i-k) dbar, t] (FIFO
        # composition of i-k receive operators, Proposition 1(b)).  Under
        # the premise t_d >= T_sw_bar + (n-1) dbar every such receive time
        # lies after the switch deadline of pair k, so
        #   |f_k| <= max_corners sup_{r >= t - (i-k) dbar - tset_max,k}
        #            |f_free,k(r)| + gamma Lambda2_k,
        # and the envelope is nonincreasing in t: the supremum over the
        # segment window is its value at t = t_d.  The condition value is
        # the rigorous per-cell bound of the 1 ms grid cell that contains
        # t_d (floor index, argument <= true argument); the closed form
        # at t_d is reported beside it.
        seg_premise = t_d - T_sw_bar - (n - 1) * dbar
        FsupL = _env_curves(G, ebar, geo['e0'], R['E1'])['Fsup']
        seg_dem = np.zeros(n); seg_dem_cf = np.zeros(n); seg_parts = {}
        j_d = min(NT - 1, int(np.floor(t_d / dtg)))
        for i in range(n):
            tot_g = a_d; tot_c = a_d; parts = []
            for k in range(i + 1):
                Rk = t_d - (i - k) * dbar - tset[k]
                Rk_cell = TGRID[j_d] - (i - k) * dbar - tset[k]
                jk = min(NT - 1, max(0, int(np.floor(Rk_cell / dtg))))
                fs_g = float(FsupL[k][jk])
                fs_c = float(_corner_fsup(G, ebar[k], np.array([max(Rk, 0.0)]))[0])
                forced = GAIN_F * Lam2f[k]
                tot_g += fs_g + forced; tot_c += fs_c + forced
                parts.append(dict(k=k + 1, R=float(Rk), free_cell=fs_g,
                                  free_cf=fs_c, forced=float(forced)))
            seg_dem[i] = tot_g; seg_dem_cf[i] = tot_c; seg_parts[i] = parts
    dem = np.zeros(n); dem_parts = {}
    for i in range(n):
        acq = Ff[i] + max(A_run[i], fbf[i])
        if seg and not refine:
            acq = acq + a_d                 # frozen (H1) with the feedforward
        brk = a_b + (Fcum[i] if i >= 1 else f_xi)
        pre_neg = (a_b + Fb[i]) + fbf[i] if i >= 1 else 0.0
        cands = [acq, brk, Pp[i], pre_neg]
        if refine:
            cands.append(float(seg_dem[i]))  # (R1) segment window
        dem[i] = max(cands)
        dem_parts[i] = tuple(cands)
    auth_slack = Umin - dem
    # reporting only: the slack without the fourth candidate
    # (a_b + F^b_{i-1}) + fbar_i, which the main text calls conservative
    # slack (after the ramp receipt the own feedback is bounded by f^b_i)
    dem_nopre = np.array([max(c for j, c in enumerate(dem_parts[i]) if j != 3)
                          for i in range(n)])
    auth_slack_nopre = Umin - dem_nopre

    # ---- fleet bounds at v_c1
    gaperr = np.array(R['gaperr'], float)
    align = eps_e + v_c1 * eps_v / a_b + eps_v ** 2 / (2 * a_b) + np.sum(gaperr)
    epsb_max = float(np.max(epsb[1:]))
    T_brk = (v_c1 + eps_v) / a_b + (n - 1) * (dbar + epsb_max / a_b)
    T_f_off = (v_c1 + eps_v) / a_b + (n - 1) * np.max(epsb[1:]) / a_b

    # ---- entry recursion (with feedforward jumps) and tail sups
    Tent = None; V = None; SxL = None
    if want_entry:
        Tent, V, SxL = _entry(G, R, Lam2f, dbar,
                              ff=(a_d, t_d, t_c) if seg else None)
    else:
        SxL = [G['_corner_tails'](ebar[k])[1] for k in range(n)]

    # ---- velocity floor: pre-switch bound and time-resolved curve
    # env_k_sym: sup |eps_k| envelope (symmetric switch-state box; used for
    # the crawl row, which needs an upper bound on the predecessor speed,
    # and for H6 without the refinement).  env_k: the envelope that H6
    # uses; with (R2) it is the signed supremum of eps_k over the
    # asymmetric switch-state box e^sw_k in [e_k(0) - E1_k, e_k(0) + E1_k],
    # |eps^sw_k| <= eps_det, plus the forced part Lambda2_k (GAIN_X = 1).
    # v_i(t) = v_r(t) - sum_{k<=i} eps_k(t) needs only an upper bound on
    # each eps_k, and a follower that closes a certified gap (e_lo > 0)
    # has eps_free,k(r) = eps^sw (1 - lam r) e^{-lam r} - e^sw lam^2 r
    # e^{-lam r} <= eps_det, so the envelope is at most eps_det + Lambda2_k
    # after the switch deadline instead of about ebar_k lam/e.
    E1 = R['E1']; e0g = geo['e0']
    SxpL = _env_curves(G, ebar, e0g, E1)['Sxp'] if refine else None
    env = np.zeros(NT); env_sym = np.zeros(NT)
    env_k = []; env_k_sym = []
    for k in range(n):
        idx = np.maximum(np.floor((TGRID - tset[k]) / dtg).astype(int), 0)
        idx = np.minimum(idx, NT - 1)
        ek_sym = SxL[k][idx] + Lam2f[k]
        if refine:
            ek = SxpL[k][idx] + Lam2f[k]
        else:
            ek = ek_sym
        env_k.append(ek); env_k_sym.append(ek_sym)
        env = env + ek; env_sym = env_sym + ek_sym
    if seg:
        vr_grid = np.where(TGRID < t_d, v_xi,
                           np.where(TGRID < t_c, v_xi - a_d * (TGRID - t_d),
                                    v_c1))
    else:
        vr_grid = np.full(NT, v_xi)
    # rigorous on each grid cell [t_j, t_{j+1}]: v_r nonincreasing, env
    # nonincreasing  ->  v_flr >= v_r(t_{j+1}) - env(t_j)
    vr_next = np.concatenate([vr_grid[1:], vr_grid[-1:]])
    flr_curve = vr_next - env
    j_sw = int(np.ceil(T_sw_bar / dtg - 1e-12))
    # Coverage of the partition: the cells j_sw .. j_xi-1 cover
    # [j_sw dtg, j_xi dtg].  The sub-cell [T_sw_bar, j_sw dtg) is covered by
    # the pre-switch bound vfloor_pre below, provided t_d >= j_sw dtg; the
    # last cell covers [j_xi dtg, T_xi] because v_r = v_c1 and env is
    # nonincreasing there (checked in finalize()).  The flag enters (H6).
    flr_cover_ok = (not seg) or (t_d + 1e-12 >= j_sw * dtg)
    # Pre-switch bound (Proposition 1(d) with v_r = v^xi):
    #   vfloor_pre = v^xi - max_i sum_{k<=i} max{M1_k, x2f_k},
    # M1_k = |eps0_k| + Lam1_k, with the band x2f of the S2-window chain
    # (feedforward ledger Lam2f included) in place of the pre-segment band
    # R['x2tot'] of the frozen pass.  It is valid on [0, t_d): for t < t_d
    # every receive time satisfies theta(t) <= t < t_d, so the forwarded
    # feedforward layer a_r(theta_{2:i}(t)) is zero and every signal
    # coincides on [0, t_d) with the acquisition continuation under
    # Assumption 1 (pre-segment constants), and x2f >= R['x2tot'] makes the
    # bound only smaller.  The S2-window band is used, and not the smaller
    # pre-segment one, so that vfloor_pre is nonincreasing in dbar together
    # with the time-resolved curve: on the sliver [T_sw_bar(d1), T_sw_bar(d2))
    # the time-resolved bound at d1 is at least v^xi - sum_k x2f_k(d1) >=
    # vfloor_pre(d1) >= vfloor_pre(d2), which the Lemma 2 extension needs.
    # Without a segment x2f == R['x2tot'] and vfloor_pre == R['vfloor']
    # (regress()).
    supx_pre = np.maximum(np.abs(eps0) + Lam1, x2f)
    vfloor_pre = float(v_xi - np.max(np.cumsum(supx_pre)))

    def env_at(t, k=None, sym=False):
        j = min(NT - 1, max(0, int(np.floor(t / dtg))))
        E, Ek = (env_sym, env_k_sym) if sym else (env, env_k)
        return float(E[j]) if k is None else float(Ek[k][j])

    # ---- clearance floors (rows) with v_c1 in braking and a crawl row
    rows = {}; gmin = np.inf
    for i in range(1, n):
        P = pairs[i]; uh = P['uh']; bf = P['bf']
        row = dict(uh_acq=P['uh_acq'], uh=uh, bf=bf, gTf=bf,
                   g_init=g_init[i], acq_spd=None, brk_spd=None,
                   crawl_spd=None, V_acq=None, V_brk=None, V_crawl=None)
        if Pi[i] > 0:
            # acquisition-row envelope over the whole acquisition phase
            # [0, T_xi]: M1_k = |eps0_k| + Lam1_k plus the S2 bands x2f of the
            # S2-window chain (enlarged ledger), because the S2 windows
            # contain the two feedforward steps.  H1 above keeps the frozen
            # pre-segment V1 because its window ends before t_d.
            V1_seg = float(np.sum(np.abs(eps0[:i]) + Lam1[:i] + x2f[:i]))
            vs_a = v_xi + V1_seg
            vs_b = v_c1 + eps_v + np.sum(epsb[1:i])
            acq_spd = uh - Pi[i] * vs_a
            brk_spd = bf - Pi[i] * vs_b
            cands = [g_init[i], acq_spd, brk_spd]
            row.update(acq_spd=acq_spd, brk_spd=brk_spd, V_acq=vs_a,
                       V_brk=vs_b)
            if seg:
                # upper bound on v_{i-1}: needs sup(-eps_k), symmetric env
                vs_c = v_c1 + sum(env_at(t_c, k, sym=True) for k in range(i))
                crawl_spd = uh - Pi[i] * vs_c
                cands.append(crawl_spd)
                row.update(crawl_spd=crawl_spd, V_crawl=vs_c)
            gspd = min(cands)
            row['gspd'] = gspd
            gmin = min(gmin, gspd, bf, bf)
        else:
            row['gspd'] = None
            gmin = min(gmin, g_init[i], bf)
        rows[i] = row

    out.update(Lam1=Lam1, Lam2_pre=Lam2p, Lam2=Lam2f, x2tot_pre=R['x2tot'],
               x2tot=x2f, fbar_pre=R['fbar'], fbar=fbf, tvS2_pre=R['tvS2'],
               tvS2=tvf, F_pre=R['F'], F=Ff, Pp=Pp, Tk=R['Tk'],
               taub=R['taub'], tact=tact, tset=tset, T_sw_bar=T_sw_bar,
               E1=R['E1'], ebar=ebar, fsw=R['fsw'],
               W=W, epsb=epsb, eb=R['eb'], fb=fb, Th=R['Th'], Fb=Fb,
               Fcum=Fcum, f_xi=f_xi, anet=a_b - Fb[1:n], hops=R['hops'],
               IE=R['IE'], IX=IX, IF=IF, Dq=Dq, DBq=DBq,
               disp=float(R['disp']), late=float(R['late']),
               sched=float(R['sched']), align=float(align), gaperr=gaperr,
               T_brk=float(T_brk), T_f_off=float(T_f_off),
               pairs=pairs, margins=margins, rows=rows, gmin=float(gmin),
               dem=dem, dem_parts=dem_parts, auth_slack=auth_slack,
               dem_nopre=dem_nopre, auth_slack_nopre=auth_slack_nopre,
               Umin=Umin, Leps=float(R['Leps']),
               event_ok=bool(R['event_ok']),
               event_slack=float(eps_det - phi - R['Leps'] * T_c),
               vfloor_pre=vfloor_pre, vfloor_pre_frozen=float(R['vfloor']),
               env=env, env_k=env_k,
               env_sym=env_sym, env_k_sym=env_k_sym,
               vr_grid=vr_grid, flr_curve=flr_curve, j_sw=j_sw,
               flr_cover_ok=bool(flr_cover_ok),
               env_at=env_at, Tent=Tent, V=V,
               vflr_c=float(v_c1 - n * eps_v),
               seg_dem=seg_dem, seg_dem_cf=seg_dem_cf, seg_parts=seg_parts,
               seg_premise=seg_premise,
               ff_charges=dict(Lam2_add=2.0 * a_d * dbar if seg else 0.0,
                               hm2_add=[float(a_d * eta[i] / (kappa * b[i]))
                                        if seg else 0.0 for i in range(1, n)],
                               Ht_add=[pairs[i]['Ht_Pi_term'] for i in range(1, n)],
                               auth_add=(a_d if seg and not refine else 0.0)))
    return out


# ----------------------------------------------------------- finalize
def _cond(ok, slack, value, threshold, sense):
    return dict(ok=bool(ok), slack=None if slack is None else float(slack),
                value=None if value is None else float(value),
                threshold=None if threshold is None else float(threshold),
                sense=sense)


def finalize(C, T_xi=None):
    """Condition table and constants for the handoff instant T_xi."""
    cfg = C['cfg']
    T_xi = float(cfg['T_xi'] if T_xi is None else T_xi)
    G = C['G']; n = G['n']; dbar = C['dbar']
    Umin = G['Umin']; VHND_PAD = G['VHND_PAD']
    seg, t_d, t_c, a_d, v_c1 = C['seg'], C['t_d'], C['t_c'], C['a_d'], C['v_c1']
    res = dict(case=dict(T_xi=T_xi, dbar=dbar, t_d=t_d, t_c=t_c, a_d=a_d,
                         v_c1=v_c1, segment=seg, v_xi=G['v_xi'],
                         lam=G['lam'], alpha=float(cfg['alpha']),
                         eps_e=float(cfg['eps_e']), eps_v=float(cfg['eps_v']),
                         vhnd=float(VHND_PAD), refine=bool(C.get('refine')),
                         sync_tol=float(cfg['sync_tol']),
                         c0=None if C['geo']['c0'] is None
                         else C['geo']['c0'].tolist(),
                         e0=C['geo']['e0'].tolist(),
                         d_s=C['geo']['d_s'].tolist()),
               closed=C['closed'])
    conds = {}
    if not C['closed']:
        conds['H2'] = _cond(False, None, None, 0.0, '>')
        for nm in COND_NAMES:
            if nm != 'H2':
                conds[nm] = _cond(False, None, None, None, None)
        res.update(conds=conds, admitted=False, failed=['H2'],
                   note='braking recursion diverged or headroom <= 0')
        return res
    R = C['R']
    # H2 headroom
    anet_min = float(np.min(C['anet']))
    conds['H2'] = _cond(anet_min > 0, anet_min, anet_min, 0.0, '>')
    # H1 authority (frozen tolerance 1e-9)
    s = float(np.min(C['auth_slack']))
    conds['H1'] = _cond(s >= -1e-9, s, float(np.max(C['dem'] - Umin)),
                        0.0, '<=')
    # H4 admission (padded by NUMPAD)
    s = float(np.min(C['margins']))
    conds['H4'] = _cond(s > 0, s, s, 0.0, '>')
    # H5 clearance
    conds['H5'] = _cond(C['gmin'] > 0, C['gmin'], C['gmin'], 0.0, '>')
    # H7 sampled event
    conds['H7'] = _cond(C['event_ok'], C['event_slack'],
                        C['Leps'] * G['T_c'], G['eps_det'] - G['phi'], '<=')
    # H6 velocity floor: pre-switch bound and time-resolved minimum
    dtg = G['dtg']; NT = G['NT']
    j_xi = min(NT - 1, int(np.floor(T_xi / dtg)))
    j_sw = C['j_sw']
    if j_xi > j_sw:
        seg_curve = C['flr_curve'][j_sw:j_xi]
        jm = int(np.argmin(seg_curve))
        vflr_time = float(seg_curve[jm]); t_min = float(G['TGRID'][j_sw + jm])
    else:
        vflr_time = np.inf; t_min = None
    vflr = min(C['vfloor_pre'], vflr_time)
    # decimated record of the time-resolved floor for the figures (segment
    # profiles only): the per-cell bound of every 0.1 s grid cell from
    # j_sw to j_xi - 1, of the cell of the minimum and of the last cell;
    # the pre-switch bound vfloor_pre applies on [0, T_sw_bar)
    vflr_curve = None
    if seg and j_xi > j_sw:
        step = max(1, int(round(0.1 / dtg)))
        js = sorted(set(range(j_sw, j_xi, step)) | {j_sw + jm, j_xi - 1})
        vflr_curve = dict(t=[float(G['TGRID'][j]) for j in js],
                          v=[float(C['flr_curve'][j]) for j in js],
                          step_s=float(step * dtg))
    # the partition must cover [T_sw_bar, T_xi] (see core(): the pre-switch
    # bound covers [T_sw_bar, j_sw dtg) when t_d >= j_sw dtg, and the last
    # cell covers [j_xi dtg, T_xi] when v_r has reached v_c1 by j_xi dtg)
    cover_ok = C.get('flr_cover_ok', True) and \
        ((not C['seg']) or j_xi * dtg + 1e-12 >= C['t_c'])
    conds['H6'] = _cond(vflr > 0 and cover_ok, vflr, vflr, 0.0, '>')
    # H8 ramp before stop with the handoff floor v_flr,c = v_c1 - n eps_v
    vflr_c = C['vflr_c']
    lhs = (n - 1) * dbar * float(np.max(Umin))
    conds['H8'] = _cond(lhs < vflr_c, vflr_c - lhs, lhs, vflr_c, '<')
    # H3 hold and C_hnd handoff pad
    if C['Tent'] is not None:
        Tent = C['Tent']
        Tent_bar = float(np.max(C['tset'] + Tent))
        conds['H3'] = _cond(Tent_bar + dbar <= T_xi, T_xi - Tent_bar - dbar,
                            Tent_bar + dbar, T_xi, '<=')
        ixh = min(NT - 1, int((T_xi - dbar) / dtg))
        Vhnd = np.zeros(n)
        for i in range(1, n):
            Vhnd[i] = float(C['V'][i][ixh])
        vh = float(np.max(Vhnd))
        conds['C_hnd'] = _cond(vh <= VHND_PAD, VHND_PAD - vh, vh, VHND_PAD,
                               '<=')
    else:
        Tent = None; Tent_bar = None; Vhnd = None
        conds['H3'] = _cond(False, None, None, T_xi, '<=')
        conds['C_hnd'] = _cond(False, None, None, VHND_PAD, '<=')
    # terminal conditions
    conds['C_sync'] = _cond(C['disp'] <= cfg['sync_tol'],
                            cfg['sync_tol'] - C['disp'], C['disp'],
                            cfg['sync_tol'], '<=')
    conds['C_al'] = _cond(C['align'] <= cfg['align_tol'],
                          cfg['align_tol'] - C['align'], C['align'],
                          cfg['align_tol'], '<=')
    ge = float(np.max(C['gaperr']))
    conds['C_gap'] = _cond(ge <= cfg['gap_tol'], cfg['gap_tol'] - ge, ge,
                           cfg['gap_tol'], '<=')
    # premise of Assumption 1': segment after the last switch deadline and
    # before the handoff, chained delays included; with (R1) the segment
    # must also start after every chained receive time of the last switch,
    # t_d >= T_sw_bar + (n-1) dbar
    if seg:
        s1 = t_d - C['T_sw_bar']
        if C.get('refine'):
            s1 = float(C['seg_premise'])
        s2 = T_xi - (t_c + (n - 1) * dbar)
        conds['P_seg'] = _cond(s1 >= 0 and s2 > 0, min(s1, s2),
                               t_c + (n - 1) * dbar, T_xi, '<')
    else:
        conds['P_seg'] = _cond(True, np.inf, None, None, None)
    admitted = all(c['ok'] for c in conds.values())
    T_f = T_xi + C['T_f_off']
    consts = dict(
        t_bar=C['tact'].tolist(), tau_bar=C['taub'].tolist(),
        tset=C['tset'].tolist(), T_sw_bar=C['T_sw_bar'],
        Lam1=C['Lam1'].tolist(), Lam2_pre=C['Lam2_pre'].tolist(),
        Lam2=C['Lam2'].tolist(), E1=C['E1'].tolist(), ebar=C['ebar'].tolist(),
        fsw=C['fsw'].tolist(), Tk=C['Tk'].tolist(),
        x2tot_pre=C['x2tot_pre'].tolist(), x2tot=C['x2tot'].tolist(),
        fbar_pre=C['fbar_pre'].tolist(), fbar=C['fbar'].tolist(),
        tvS2_pre=C['tvS2_pre'].tolist(), tvS2=C['tvS2'].tolist(),
        F=C['F'].tolist(), Ppre=C['Pp'].tolist(),
        W=C['W'][1:].tolist(), epsb=C['epsb'][1:].tolist(),
        eb=C['eb'][1:].tolist(), fb=C['fb'][1:].tolist(),
        Theta=C['Th'][1:].tolist(), Fb=C['Fb'].tolist(),
        Fcum=C['Fcum'][1:].tolist(), f_xi=float(C['f_xi']),
        anet=np.asarray(C['anet']).tolist(), q=np.asarray(C['hops']).tolist(),
        disp=C['disp'], late=C['late'], sched=C['sched'], align=C['align'],
        gaperr=C['gaperr'].tolist(), T_brk=C['T_brk'], T_f_bar=float(T_f),
        pairs={str(i): {k: (None if v is None else float(v))
                        for k, v in C['pairs'][i].items()}
               for i in C['pairs']},
        margins=C['margins'].tolist(),
        floors={str(i): {k: (None if v is None else float(v))
                         for k, v in C['rows'][i].items()}
                for i in C['rows']},
        gmin=C['gmin'], dem=C['dem'].tolist(),
        dem_parts={str(i): [float(x) for x in C['dem_parts'][i]]
                   for i in C['dem_parts']},
        Umin=np.asarray(Umin).tolist(), auth_slack=C['auth_slack'].tolist(),
        dem_nopre=C['dem_nopre'].tolist(),
        auth_slack_nopre=C['auth_slack_nopre'].tolist(),
        Leps=C['Leps'], LepsTc=C['Leps'] * G['T_c'],
        vfloor_pre=C['vfloor_pre'], vfloor_pre_frozen=C['vfloor_pre_frozen'],
        vflr_time_min=vflr_time,
        vflr_time_argmin=t_min, vflr=float(vflr), vflr_c=vflr_c,
        vflr_curve=vflr_curve,
        env_T_sw=[float(C['env_at'](C['T_sw_bar'], k)) for k in range(n)],
        env_t_c=[float(C['env_at'](t_c, k)) for k in range(n)] if seg else None,
        env_sym_t_c=[float(C['env_at'](t_c, k, sym=True)) for k in range(n)]
        if seg else None,
        env_kind='one-sided (R2)' if C.get('refine') else 'symmetric',
        seg_dem=None if C['seg_dem'] is None else C['seg_dem'].tolist(),
        seg_dem_cf=None if C['seg_dem_cf'] is None else C['seg_dem_cf'].tolist(),
        seg_parts=None if C['seg_parts'] is None else
        {str(i + 1): C['seg_parts'][i] for i in C['seg_parts']},
        seg_premise=C['seg_premise'],
        Tent=None if Tent is None else [float(x) for x in Tent],
        T_ent_bar=Tent_bar,
        T_hold=None if Tent_bar is None else Tent_bar + dbar,
        Vhnd=None if Vhnd is None else Vhnd[1:].tolist(),
        Vhnd_max=None if Vhnd is None else float(np.max(Vhnd)),
        ff_charges=C['ff_charges'],
        h0=C['geo']['h0'][1:].tolist(), g_init=C['geo']['g_init'][1:].tolist(),
        v0=C['geo']['v0'].tolist(), e0=C['geo']['e0'].tolist(),
        NUMPAD=G['NUMPAD'], VHND_PAD=VHND_PAD)
    ref = reference(dict(cfg, T_xi=T_xi))
    res.update(conds=conds, admitted=bool(admitted),
               failed=[k for k, c in conds.items() if not c['ok']],
               consts=consts, refine=bool(C.get('refine')),
               reference={k: ref[k] for k in ('t_d', 't_c', 'a_d', 'v_c1',
                                              'T_xi', 'tau_r', 's0_Txi',
                                              's_ref_stop', 'markers')})
    return res


def evaluate(cfg, want_entry=True):
    """Full evaluation of one configuration (dict with every Table I
    condition as {ok, slack, value, threshold}, the internal constants
    and admitted = all ok)."""
    cfg = config(**cfg)
    C = core(cfg, want_entry=want_entry)
    return finalize(C, cfg['T_xi'])


# ------------------------------------------------------ age-bound search
def age_ceilings(cfg, names=None, lo=0.002, hi=0.45, tol=2e-4,
                 round_mid=6, verbose=False):
    """Per-condition feasible endpoints by bisection in the age bound with
    the design frozen (procedure of run_b2.part_cert: lo = 0.002,
    mid rounded to 6 decimals; certify_v30.ceil_of: lo = 0.004,
    round_mid = None).  A condition that fails at lo reports 0.0 and one
    that holds at hi reports hi."""
    cfg = config(**cfg)
    names = names or [k for k in COND_NAMES if k != 'P_seg']
    memo = {}

    def ev(d):
        if d not in memo:
            memo[d] = evaluate(dict(cfg, dbar=d))
        return memo[d]

    def cond(nm, d):
        r = ev(d)
        if not r['closed']:
            return False
        return r['conds'][nm]['ok']

    ceils = {}
    for nm in names:
        a, bnd = lo, hi
        if not cond(nm, a):
            ceils[nm] = 0.0; continue
        if cond(nm, bnd):
            ceils[nm] = bnd; continue
        while bnd - a > tol:
            mid = 0.5 * (a + bnd)
            if round_mid is not None:
                mid = round(mid, round_mid)
            if cond(nm, mid):
                a = mid
            else:
                bnd = mid
        ceils[nm] = a
        if verbose:
            print('  %-6s ceiling %.6f' % (nm, a), flush=True)
    return ceils


def age_boundary_ms(cfg, kmax=450, verbose=False):
    """Largest integer k with the certificate admitted at dbar = k ms
    (bisection on the 1 ms grid; the neighbours k and k+1 are verified),
    with the failing conditions at k+1."""
    cfg = config(**cfg)
    memo = {}

    def adm(k):
        if k not in memo:
            memo[k] = evaluate(dict(cfg, dbar=k * 1e-3))
        return memo[k]['admitted']

    if not adm(1):
        return dict(k_ms=0, admitted_at=None, failed_next=memo[1]['failed'])
    lo, hi = 1, kmax
    if adm(hi):
        return dict(k_ms=hi, admitted_at=hi * 1e-3, failed_next=None,
                    note='admitted at kmax')
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if adm(mid):
            lo = mid
        else:
            hi = mid
    assert adm(lo) and not adm(lo + 1)
    r1 = memo[lo + 1]
    out = dict(k_ms=lo, admitted_at=lo * 1e-3, failed_next=r1['failed'],
               slacks_at_k={nm: memo[lo]['conds'][nm]['slack']
                            for nm in COND_NAMES},
               slacks_at_k1={nm: r1['conds'][nm]['slack']
                             for nm in COND_NAMES})
    if verbose:
        print('  boundary: %d ms admitted, %d ms fails %s'
              % (lo, lo + 1, r1['failed']), flush=True)
    return out


# ------------------------------------------------------------- scan
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


def grid_prev():
    """Grid of the first Case C scan (2026-09-26, data/new_d19/caseC_prev):
    equal takeover clearances 6..30 m plus the Case B pattern scaled, lam,
    a_d, T_xi, Case B tolerances."""
    c0 = [(float(c),) * 4 for c in range(6, 31)]
    for s in np.arange(0.15, 0.651, 0.05):
        c0.append(tuple(round(float(x) * float(s), 3)
                        for x in (42.0, 45.0, 47.0, 49.0)))
    return dict(c0=c0, lam=[0.055, 0.065, 0.075, 0.085, 0.10],
                alpha=[0.35], eps_e=[0.15], eps_v=[0.02], vhnd=[1e-3],
                a_d=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40],
                T_xi=[70.0, 75.0, 80.0, 85.0, 90.0, 95.0],
                fixed=dict(t_d=10.0, v_c1=1.0, dbar=0.020,
                           d_s=[6.0, 5.0, 5.0, 5.0, 5.0],
                           eps0=[-0.30, 0.25, 0.18, 0.12, 0.08]))


# axes of the revised scan (authors' requirements of 2026-09-26)
SCAN_C0_REQUESTED = ([(float(c),) * 4 for c in (44, 46, 48, 50, 52)]
                     + [(42.0, 45.0, 47.0, 49.0), (47.0, 50.0, 52.0, 54.0)])
SCAN_C0_EXTENSION = [(float(c),) * 4 for c in (28, 32, 36, 40)]


def default_grid(extension=True):
    """Revised Case C grid: takeover clearances about 50 m (the requested
    axis) plus, with extension, the equal clearances 28 to 40 m that locate
    the admitted frontier when the requested axis is rejected; lam, alpha,
    the handoff box (eps_e, eps_v), a_d, the handoff pad V^hnd and T_xi;
    t_d = 10 s, v_c1 = 1 m/s, dbar = 20 ms, gaps and mismatches of Case B."""
    c0 = list(SCAN_C0_REQUESTED) + (list(SCAN_C0_EXTENSION) if extension
                                     else [])
    return dict(c0=c0, lam=[0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09],
                alpha=[0.30, 0.35], eps_e=[0.15, 0.2, 0.3, 0.4, 0.5],
                eps_v=[0.02, 0.03, 0.05], a_d=[0.2, 0.3, 0.4, 0.5, 0.6],
                vhnd=[1e-3, 1e-2],
                T_xi=[88.0, 90.0, 92.0, 94.0, 96.0, 98.0, 100.0, 103.0],
                fixed=dict(t_d=10.0, v_c1=1.0, dbar=0.020,
                           d_s=[6.0, 5.0, 5.0, 5.0, 5.0],
                           eps0=[-0.30, 0.25, 0.18, 0.12, 0.08],
                           refine=True))


# extension of 2026-09-26 (authors' clarification: the running time may
# exceed 100 s; clearances about 50 m first, then the smallest T_f_bar)
SCAN_EXT_PATH = os.path.join(DATA_C, 'scan_ext.json')
SCAN_C0_EXT2 = ([(float(c),) * 4 for c in (46, 48, 50, 52)]
                + [(42.0, 45.0, 47.0, 49.0), (47.0, 50.0, 52.0, 54.0)])


def grid_ext():
    """Late-handoff extension of the revised grid: the requested
    clearances (46, 48, 50, 52 m and the two Case B patterns), the gains
    0.05 to 0.065 (the entry deadline at 50 m is 113.4 s at lam = 0.06 and
    the admission and authority slacks grow as lam decreases), alpha 0.30
    and 0.35, the same boxes and rates, V^hnd = 1e-2 only (1e-3 fails
    C_hnd in every row of scan.json) and T_xi from 105 to 150 s in 5 s
    steps; dispersion requirement d_req = 1.0 s."""
    return dict(c0=list(SCAN_C0_EXT2), lam=[0.05, 0.055, 0.06, 0.065],
                alpha=[0.30, 0.35], eps_e=[0.15, 0.2, 0.3, 0.4, 0.5],
                eps_v=[0.02, 0.03, 0.05], a_d=[0.2, 0.3, 0.4, 0.5, 0.6],
                vhnd=[1e-2],
                T_xi=[float(x) for x in range(105, 151, 5)],
                fixed=dict(t_d=10.0, v_c1=1.0, dbar=0.020,
                           d_s=[6.0, 5.0, 5.0, 5.0, 5.0],
                           eps0=[-0.30, 0.25, 0.18, 0.12, 0.08],
                           refine=True, d_req=1.0))


# Case B slacks at 20 ms for the normalized-slack ranking (H6 is scaled
# by the profile speed, H8 and P_seg are excluded)
_SB = {}


def caseB_slacks():
    if not _SB:
        r = evaluate(config())
        _SB.update({k: r['conds'][k]['slack'] for k in COND_NAMES})
    return _SB


def binding(res):
    """For an admitted row, the condition with the smallest slack relative
    to the Case B slack at 20 ms (H6 scaled by v_c1/v^xi; H8 and P_seg
    excluded), with the ratio; for a rejected row, the failed list."""
    if not res['admitted']:
        return dict(binding=None, ratio=None, failed=res['failed'])
    sB = caseB_slacks()
    best = None
    for k in COND_NAMES:
        if k in ('H8', 'P_seg'):
            continue
        s = res['conds'][k]['slack']
        ref = sB[k]
        if k == 'H6':
            ref = sB[k] * res['case']['v_c1'] / res['case']['v_xi']
        if s is None or ref is None or ref <= 0:
            continue
        ratio = s / ref
        if best is None or ratio < best[1]:
            best = (k, ratio)
    return dict(binding=best[0], ratio=best[1], failed=[])


def _row(res, cfg):
    K = res['consts'] if res['closed'] else {}
    B = binding(res) if res['closed'] else dict(binding=None, ratio=None,
                                                failed=res['failed'])
    dp = K.get('dem_parts')
    return dict(c0=list(cfg['c0']), lam=cfg['lam'], alpha=cfg['alpha'],
                eps_e=cfg['eps_e'], eps_v=cfg['eps_v'], a_d=cfg['a_d'],
                vhnd=cfg['vhnd'], T_xi=res['case']['T_xi'],
                d_req=float(cfg['sync_tol']),
                t_c=res['case']['t_c'], closed=res['closed'],
                admitted=res['admitted'], failed=res['failed'],
                binding=B['binding'], binding_ratio=B['ratio'],
                slack={nm: res['conds'][nm]['slack'] for nm in COND_NAMES},
                T_ent_bar=K.get('T_ent_bar'), T_hold=K.get('T_hold'),
                T_f_bar=K.get('T_f_bar'), T_sw_bar=K.get('T_sw_bar'),
                vflr_time_min=K.get('vflr_time_min'),
                vflr_time_argmin=K.get('vflr_time_argmin'),
                margins=K.get('margins'), gmin=K.get('gmin'),
                auth_slack=K.get('auth_slack'),
                auth_slack_nopre=K.get('auth_slack_nopre'),
                dem_parts_last=None if dp is None else dp[str(len(dp) - 1)],
                seg_dem=K.get('seg_dem'), disp=K.get('disp'),
                align=K.get('align'),
                gaperr_max=None if not K else float(np.max(K['gaperr'])),
                Vhnd_max=K.get('Vhnd_max'), Tent=K.get('Tent'))


def _combo_key(c0, lam, alpha, a_d):
    return (tuple(float(x) for x in c0), float(lam), float(alpha), float(a_d))


def scan(grid=None, out_path=SCAN_PATH, verbose=True, checkpoint=20,
         resume=True):
    """Evaluate the grid and write out_path.  Loop order: (lam, alpha)
    outermost (frozen namespaces), then (c0, a_d) (entry curves and
    envelope curves cached per group), then (eps_e, eps_v, vhnd) (cheap
    forward pass and box test), finalized for every T_xi.  With resume,
    the rows of a partial file at out_path are kept and their completed
    (c0, lam, alpha, a_d) groups are skipped."""
    grid = grid or default_grid()
    fixed = dict(grid.get('fixed', {}))
    inner = [(e, v, h) for e in grid['eps_e'] for v in grid['eps_v']
             for h in grid['vhnd']]
    per_group = len(inner) * len(grid['T_xi'])
    rows = []
    done = set()
    if resume and os.path.exists(out_path):
        old = json.load(open(out_path))
        same = all(old.get('grid', {}).get(k) == _py(grid[k])
                   for k in ('T_xi', 'eps_e', 'eps_v', 'vhnd'))
        if same:
            cnt = {}
            for r in old['rows']:
                key = _combo_key(r['c0'], r['lam'], r['alpha'], r['a_d'])
                cnt[key] = cnt.get(key, 0) + 1
            done = {k for k, v in cnt.items() if v == per_group}
            rows = [r for r in old['rows']
                    if _combo_key(r['c0'], r['lam'], r['alpha'], r['a_d'])
                    in done]
            if verbose:
                print('  resume: %d groups (%d rows) kept' % (len(done),
                                                               len(rows)),
                      flush=True)
    t0 = time.time()
    groups = [(lam, alpha, c0, a_d) for lam in grid['lam']
              for alpha in grid['alpha'] for c0 in grid['c0']
              for a_d in grid['a_d']
              if _combo_key(c0, lam, alpha, a_d) not in done]
    Tmax = max(grid['T_xi'])
    for m, (lam, alpha, c0, a_d) in enumerate(groups):
        for eps_e, eps_v, vhnd in inner:
            cfg = config(**dict(fixed, c0=tuple(c0), lam=float(lam),
                                alpha=float(alpha), a_d=float(a_d),
                                eps_e=float(eps_e), eps_v=float(eps_v),
                                vhnd=float(vhnd), T_xi=float(Tmax)))
            try:
                C = core(cfg)
                for T_xi in grid['T_xi']:
                    rows.append(_row(finalize(C, float(T_xi)), cfg))
            except Exception as ex:                 # pragma: no cover
                for T_xi in grid['T_xi']:
                    rows.append(dict(c0=list(c0), lam=lam, alpha=alpha,
                                     eps_e=eps_e, eps_v=eps_v, a_d=a_d,
                                     vhnd=vhnd, T_xi=T_xi, closed=False,
                                     admitted=False,
                                     failed=['error:%s' % ex]))
        if verbose and (m + 1) % 10 == 0:
            nadm = sum(1 for r in rows if r['admitted'])
            print('  scan %d/%d groups, %d rows, %d admitted (%.0f s)'
                  % (m + 1, len(groups), len(rows), nadm, time.time() - t0),
                  flush=True)
        if checkpoint and (m + 1) % checkpoint == 0:
            _write_scan(rows, grid, out_path, time.time() - t0, partial=True)
    _write_scan(rows, grid, out_path, time.time() - t0, partial=False)
    return rows


def _write_scan(rows, grid, out_path, runtime, partial):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g = {k: v for k, v in grid.items()}
    json.dump(_py(dict(grid=g, n_rows=len(rows),
                       n_admitted=sum(1 for r in rows if r['admitted']),
                       partial=partial, runtime_s=runtime,
                       columns='slack: per-condition slack in the units of '
                               'the condition (m, s, m/s, m/s^2); '
                               'admitted = all conditions and the premise '
                               'P_seg hold; binding: condition with the '
                               'smallest slack ratio to Case B at 20 ms '
                               '(admitted rows); auth_slack_nopre: '
                               'reporting only, H1 slack without the '
                               'candidate (a_b + F^b_{i-1}) + fbar_i; '
                               'dem_parts_last: H1 candidates of the last '
                               'unit (acq, brk, P^pre, pre_neg[, seg])',
                       rows=rows)),
              open(out_path, 'w'), indent=1)


# ------------------------------------------------------------- regression
def _cmp(a, b, path, bad, rtol=0.0):
    if isinstance(a, dict):
        for k in a:
            _cmp(a[k], b.get(k) if isinstance(b, dict) else None,
                 path + '/' + str(k), bad, rtol)
    elif isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)) or len(a) != len(b):
            bad.append((path, a, b)); return
        for j, (x, y) in enumerate(zip(a, b)):
            _cmp(x, y, path + '[%d]' % j, bad, rtol)
    elif isinstance(a, float) and isinstance(b, (float, int)):
        if a == b:                       # equal values, infinities included
            return
        if not (abs(a - b) <= rtol * max(1.0, abs(a), abs(b))):
            bad.append((path, a, b))
    elif a != b:
        bad.append((path, a, b))


def regress(verbose=True):
    """Case B (b2_results.json 'cert') and Case A (v30_values.json) must be
    reproduced exactly (bit for bit; ceilings included)."""
    out = {}
    # ---- Case B
    B = json.load(open(os.path.join(CODE, 'data', 'new_d19',
                                    'b2_results.json')))['cert']
    cfg = config()
    r = evaluate(cfg)
    K = r['consts']
    got = dict(disp=K['disp'], late=K['late'], sched=K['sched'],
               align=K['align'], gaperr=K['gaperr'][1:], T_f=K['T_f_bar'],
               T_brk=K['T_brk'], taub=K['tau_bar'], tset=K['tset'],
               E1=K['E1'], epsb=K['epsb'], anet=K['anet'], hops=K['q'],
               Leps=K['Leps'], margins=K['margins'], gmin=K['gmin'],
               auth_slack=K['auth_slack'], vfloor=K['vflr'],
               h0=K['h0'], v0=K['v0'], e0=K['e0'],
               entry=dict(Tent_cert=K['Tent'], Tent_bar=K['T_ent_bar'],
                          T_hold=K['T_hold'], hold_ok=r['conds']['H3']['ok'],
                          Vhnd_max=K['Vhnd_max'],
                          hnd_ok=r['conds']['C_hnd']['ok']),
               conds={k: r['conds'][v]['ok'] for k, v in SHORT.items()
                      if k != 'ledger'})
    ref = {k: B[k] for k in got}
    bad = []
    _cmp(ref, got, 'B', bad)
    names = ['sync', 'align', 'gap', 'adm', 'auth', 'clr', 'event', 'vel',
             'ramp', 'hold', 'hnd']
    t0 = time.time()
    ce = age_ceilings(cfg, names=[SHORT[k] for k in names], lo=0.002,
                      hi=0.45, tol=2e-4, round_mid=6)
    ce_short = {k: ce[SHORT[k]] for k in names}
    _cmp(B['ceils'], ce_short, 'B/ceils', bad)
    out['caseB'] = dict(fields=len(bad) == 0, bad=bad[:40], ceils=ce_short,
                        ceils_ref=B['ceils'], ceil_runtime_s=time.time() - t0,
                        n_fields_compared=_count(ref) + len(names))
    if verbose:
        print('Case B regression: %s (%d mismatches; ceilings %s)'
              % ('EXACT' if not bad else 'MISMATCH', len(bad),
                 'exact' if ce_short == B['ceils'] else 'differ'), flush=True)
        for p, x, y in bad[:20]:
            print('   ', p, x, y)
    # ---- Case A (frozen featured design)
    A = json.load(open(os.path.join(HERE, 'v30_values.json')))
    cfgA = config(v_xi=10.0, a_b=0.70, alpha=0.30, lam=0.075, s_m=3.0,
                  T_xi=180.0, dbar=0.06, eps_e=0.20, eps_v=0.02,
                  eps0=[-0.05, 0.025, 0.025, 0.025, 0.025],
                  e0=[-0.5, 40.0, 40.0, 40.0, 40.0], c0=None,
                  d_s=[0.0] + list(A['d_s']), sync_tol=1.0, align_tol=3.0,
                  gap_tol=1.0)
    rA = evaluate(cfgA)
    KA = rA['consts']
    gotA = dict(disp=KA['disp'], late=KA['late'], sched=KA['sched'],
                align=KA['align'], gaperr=KA['gaperr'], T_f=KA['T_f_bar'],
                T_brk=KA['T_brk'], Tk=KA['Tk'], Lam1=KA['Lam1'],
                Lam2=KA['Lam2'], taub=KA['tau_bar'], tset=KA['tset'],
                E1=KA['E1'], ebar=KA['ebar'], fbar=KA['fbar'],
                tvS2=KA['tvS2'], F=KA['F'], x2tot=KA['x2tot'], W=KA['W'],
                epsb=KA['epsb'], fb=KA['fb'], Fb=KA['Fb'][1:],
                anet=KA['anet'], Ppre=KA['Ppre'], dem=KA['dem'],
                Umin=KA['Umin'],
                dem_parts=KA['dem_parts'], Leps=KA['Leps'],
                LepsTc=KA['LepsTc'], vfloor=KA['vflr'],
                Tent_cert=KA['Tent'], Tent_bar=KA['T_ent_bar'],
                T_hold=KA['T_hold'], Vhnd_max=KA['Vhnd_max'],
                Vhnd_list=[0.0] + KA['Vhnd'], h0=KA['h0'],
                gmin_all=KA['gmin'],
                pairs={i: dict(h0=KA['pairs'][i]['h0'], H1=KA['pairs'][i]['H1'],
                               Ht=KA['pairs'][i]['Ht'], Ph2=KA['pairs'][i]['Ph2'],
                               hm1=KA['pairs'][i]['hm1'], hm2=KA['pairs'][i]['hm2'],
                               hmb=KA['pairs'][i]['hmb'],
                               hmb_s2=KA['pairs'][i]['hmb_s2'],
                               hmb_rule=KA['pairs'][i]['hmb_rule'],
                               thr=KA['pairs'][i]['thr'],
                               margin=KA['pairs'][i]['margin'],
                               padded=KA['pairs'][i]['padded'],
                               uh=KA['pairs'][i]['uh'], bf=KA['pairs'][i]['bf'],
                               gspd=KA['floors'][i]['gspd'],
                               gTf=None if KA['floors'][i]['gspd'] is None
                               else KA['floors'][i]['gTf'],
                               Vrise=KA['pairs'][i]['Vrise'],
                               Ubrk=KA['pairs'][i]['Ubrk'],
                               g_init=KA['floors'][i]['g_init'])
                       for i in KA['pairs']})
    refA = {k: A[k] for k in gotA}
    badA = []
    _cmp(refA, gotA, 'A', badA)
    namesA = ['sync', 'align', 'gap', 'adm', 'auth', 'clr', 'event', 'hold',
              'hnd', 'vel', 'ramp', 'ledger']
    t0 = time.time()
    ceA = age_ceilings(cfgA, names=[SHORT[k] for k in namesA], lo=0.004,
                       hi=0.45, tol=2e-4, round_mid=None)
    ceA_short = {k: ceA[SHORT[k]] for k in namesA}
    _cmp(A['ceils'], ceA_short, 'A/ceils', badA)
    out['caseA'] = dict(fields=len(badA) == 0, bad=badA[:40], ceils=ceA_short,
                        ceils_ref=A['ceils'], ceil_runtime_s=time.time() - t0,
                        n_fields_compared=_count(refA) + len(namesA))
    if verbose:
        print('Case A regression: %s (%d mismatches; ceilings %s)'
              % ('EXACT' if not badA else 'MISMATCH', len(badA),
                 'exact' if ceA_short == A['ceils'] else 'differ'), flush=True)
        for p, x, y in badA[:20]:
            print('   ', p, x, y)
    out['ok'] = bool(not bad and not badA)
    return out


def _count(o):
    if isinstance(o, dict):
        return sum(_count(v) for v in o.values())
    if isinstance(o, (list, tuple)):
        return sum(_count(v) for v in o)
    return 1


def _prev_design_cfg():
    """cfg of the first Case C design (data/new_d19/caseC_prev/design.json)."""
    p = os.path.join(CODE, 'data', 'new_d19', 'caseC_prev', 'design.json')
    d = json.load(open(p))['cfg']
    return config(n=int(d['n']), v_xi=float(d['v_xi']), a_b=float(d['a_b']),
                  alpha=float(d['alpha']), lam=float(d['lam']),
                  s_m=float(d['s_m']), dbar=float(d['dbar']),
                  eps_e=float(d['eps_e']), eps_v=float(d['eps_v']),
                  eps0=[float(x) for x in d['eps0']],
                  d_s=[float(x) for x in d['d_s']],
                  amax=[float(x) for x in d['amax']],
                  e0_head=float(d['e0_head']), T_c=0.001,
                  eps_det=float(d['eps_det']), phi=float(d['phi']),
                  c0=tuple(float(x) for x in d['c0']), T_xi=float(d['T_xi']),
                  t_d=float(d['t_d']), a_d=float(d['a_d']),
                  v_c1=float(d['v_c1']))


def _diff_results(a, b):
    """Fields of two evaluate() results that differ (conds and consts)."""
    bad = []
    _cmp(dict(conds=a['conds'], consts=a['consts'], admitted=a['admitted']),
         dict(conds=b['conds'], consts=b['consts'], admitted=b['admitted']),
         '', bad)
    return bad


def refine_check(verbose=True):
    """Properties of the refinements (R1), (R2).
    (a) Case B and Case A: refine on and off give identical results (no
        segment, so neither refinement is active).
    (b) The first Case C design (caseC_prev/design.json): refine off
        reproduces the archived certificate record (conds and consts of
        design.json 'record'), refine on admits it, its H1 demand is not
        larger and its velocity floor not smaller.
    (c) A set of segment configurations: per unit, the refined demand is
        at most the unrefined demand (implication of the frozen (H1)
        with a_d) and the refined floor is at least the unrefined floor
        (implication of the symmetric (H6)); the (R1) closed form at t_d
        is at most the grid-cell value; the (R2) envelope after the switch
        deadline is at most eps_det + Lambda2_k (task bound)."""
    out = {}
    # (a)
    same = {}
    for name, cfg in (('caseB', config()),
                      ('caseA', config(v_xi=10.0, a_b=0.70, alpha=0.30,
                                       lam=0.075, s_m=3.0, T_xi=180.0,
                                       dbar=0.06, eps_e=0.20, eps_v=0.02,
                                       eps0=[-0.05, 0.025, 0.025, 0.025, 0.025],
                                       e0=[-0.5, 40.0, 40.0, 40.0, 40.0],
                                       c0=None,
                                       d_s=[0.0] + list(json.load(open(
                                           os.path.join(HERE, 'v30_values.json')))['d_s']),
                                       sync_tol=1.0, align_tol=3.0, gap_tol=1.0))):
        r1 = evaluate(dict(cfg, refine=True))
        r0 = evaluate(dict(cfg, refine=False))
        bad = _diff_results(r0, r1)
        same[name] = dict(identical=len(bad) == 0, n_diff=len(bad),
                          examples=bad[:5])
    out['no_segment_identical'] = same
    # (b)
    cfgC = _prev_design_cfg()
    prev = json.load(open(os.path.join(CODE, 'data', 'new_d19', 'caseC_prev',
                                       'design.json')))
    rec = prev.get('record') or prev.get('certificate') or {}
    r0 = evaluate(dict(cfgC, refine=False))
    r1 = evaluate(dict(cfgC, refine=True))
    badC = []
    if rec and 'conds' in rec:
        _cmp({k: rec['conds'][k]['slack'] for k in rec['conds']},
             {k: r0['conds'][k]['slack'] for k in rec['conds']}, 'conds', badC)
        for k in ('dem', 'margins', 'gmin', 'vflr', 'T_ent_bar', 'T_f_bar',
                  'Vhnd_max', 'Lam2', 'disp', 'align'):
            if k in rec.get('consts', {}):
                _cmp(rec['consts'][k], r0['consts'][k], 'consts/' + k, badC)
    out['prev_design'] = dict(
        cfg={k: v for k, v in cfgC.items() if k != 'G'},
        archived_record_found=bool(rec and 'conds' in rec),
        refine_off_matches_archive=len(badC) == 0, mismatches=badC[:10],
        admitted_off=r0['admitted'], admitted_on=r1['admitted'],
        slack_off={k: r0['conds'][k]['slack'] for k in COND_NAMES},
        slack_on={k: r1['conds'][k]['slack'] for k in COND_NAMES},
        dem_off=r0['consts']['dem'], dem_on=r1['consts']['dem'],
        seg_dem_on=r1['consts']['seg_dem'],
        vflr_off=r0['consts']['vflr'], vflr_on=r1['consts']['vflr'],
        dem_not_larger=bool(np.all(np.array(r1['consts']['dem'])
                                   <= np.array(r0['consts']['dem']) + 1e-15)),
        vflr_not_smaller=bool(r1['consts']['vflr'] >= r0['consts']['vflr'] - 1e-15))
    # (c)
    tests = []
    cases = [dict(c0=(50.0,) * 4, lam=0.07, a_d=0.4, T_xi=96.0, eps_e=0.3, eps_v=0.03),
             dict(c0=(46.0,) * 4, lam=0.06, a_d=0.2, T_xi=100.0, eps_e=0.15, eps_v=0.02),
             dict(c0=(30.0,) * 4, lam=0.09, a_d=0.6, T_xi=90.0, eps_e=0.5, eps_v=0.05,
                  alpha=0.30, vhnd=1e-2),
             dict(c0=(18.0,) * 4, lam=0.10, a_d=0.15, T_xi=95.0, eps_e=0.15, eps_v=0.02),
             dict(c0=(7.0,) * 4, lam=0.055, a_d=0.4, T_xi=95.0, eps_e=0.15, eps_v=0.02),
             dict(c0=(42.0, 45.0, 47.0, 49.0), lam=0.065, a_d=0.3, T_xi=98.0,
                  eps_e=0.4, eps_v=0.05)]
    for kw in cases:
        cfg = config(**dict(kw, t_d=10.0, v_c1=1.0, dbar=0.02))
        r1 = evaluate(dict(cfg, refine=True))
        r0 = evaluate(dict(cfg, refine=False))
        K1, K0 = r1['consts'], r0['consts']
        G = namespace(cfg)
        eps_det = G['eps_det']
        # (R2) envelope at t_c against the simple bound eps_det + Lam2_k,
        # which holds for the pairs with a certified closure
        # (e_k(0) - E1_k > 0); the head pair (e_1(0) = -0.5 m) is not one
        # and keeps the rigorous signed corner supremum
        env_tc = K1['env_t_c']; Lam2 = K1['Lam2']
        e0v = K1['e0']; E1v = K1['E1']
        closure = [e0v[k] - E1v[k] > 0 for k in range(len(env_tc))]
        task_bound_ok = all(env_tc[k] <= eps_det + Lam2[k] + 1e-12
                            for k in range(len(env_tc)) if closure[k])
        tests.append(dict(
            cfg=kw, admitted_on=r1['admitted'], admitted_off=r0['admitted'],
            failed_on=r1['failed'], failed_off=r0['failed'],
            dem_on=K1['dem'], dem_off=K0['dem'],
            dem_not_larger=bool(np.all(np.array(K1['dem']) <= np.array(K0['dem']) + 1e-15)),
            seg_dem_grid=K1['seg_dem'], seg_dem_closed_form=K1['seg_dem_cf'],
            closed_form_not_larger=bool(np.all(np.array(K1['seg_dem_cf'])
                                               <= np.array(K1['seg_dem']) + 1e-15)),
            vflr_on=K1['vflr'], vflr_off=K0['vflr'],
            vflr_not_smaller=bool(K1['vflr'] >= K0['vflr'] - 1e-15),
            env_t_c_on=env_tc, env_t_c_sym=K1['env_sym_t_c'],
            env_task_bound=[eps_det + x for x in Lam2],
            certified_closure=closure,
            env_within_task_bound=task_bound_ok,
            H1_on=r1['conds']['H1']['slack'], H1_off=r0['conds']['H1']['slack'],
            H6_on=r1['conds']['H6']['slack'], H6_off=r0['conds']['H6']['slack'],
            P_seg_on=r1['conds']['P_seg']['slack'],
            P_seg_off=r0['conds']['P_seg']['slack']))
    out['segment_cases'] = tests
    out['ok'] = bool(all(v['identical'] for v in same.values())
                     and out['prev_design']['refine_off_matches_archive']
                     and out['prev_design']['admitted_on']
                     and out['prev_design']['dem_not_larger']
                     and out['prev_design']['vflr_not_smaller']
                     and all(t['dem_not_larger'] and t['vflr_not_smaller']
                             and t['closed_form_not_larger']
                             and t['env_within_task_bound'] for t in tests))
    if verbose:
        print('refine check: no-segment identical %s; previous design: '
              'archive match %s, admitted on %s (H1 %.4f -> %.4f, H6 %.4f -> '
              '%.4f); segment cases ok %s; all ok %s'
              % ({k: v['identical'] for k, v in same.items()},
                 out['prev_design']['refine_off_matches_archive'],
                 out['prev_design']['admitted_on'],
                 out['prev_design']['slack_off']['H1'],
                 out['prev_design']['slack_on']['H1'],
                 out['prev_design']['slack_off']['H6'],
                 out['prev_design']['slack_on']['H6'],
                 all(t['dem_not_larger'] and t['vflr_not_smaller'] for t in tests),
                 out['ok']), flush=True)
        for t in tests:
            print('   c0 %s lam %.3f a_d %.2f: admitted off %s on %s; H1 %+.4f -> '
                  '%+.4f; H6 %+.4f -> %+.4f; failed on %s'
                  % (t['cfg']['c0'][0], t['cfg']['lam'], t['cfg']['a_d'],
                     t['admitted_off'], t['admitted_on'], t['H1_off'], t['H1_on'],
                     t['H6_off'], t['H6_on'], t['failed_on']))
    return out


# ------------------------------------------------------------- CLI
if __name__ == '__main__':
    args = sys.argv[1:] or ['regress']
    os.makedirs(DATA_C, exist_ok=True)
    for a in args:
        if a == 'regress':
            rep = regress()
            json.dump(_py(rep), open(os.path.join(DATA_C, 'regression.json'),
                                     'w'), indent=1)
            print('regression ok:', rep['ok'])
        elif a == 'scan':
            scan()
        elif a == 'scan_requested':
            scan(grid=default_grid(extension=False))
        elif a == 'refine':
            rep = refine_check()
            json.dump(_py(rep), open(os.path.join(DATA_C, 'refine_check.json'),
                                     'w'), indent=1)
            print('refine check ok:', rep['ok'])
        elif a.startswith('eval:'):
            kv = dict(x.split('=') for x in a[5:].split(','))
            cfg = config(**{k: (float(v) if k not in ('c0',) else
                                (float(v),) * 4) for k, v in kv.items()})
            r = evaluate(cfg)
            print(json.dumps(_py(dict(admitted=r['admitted'],
                                      failed=r['failed'],
                                      conds=r['conds'])), indent=1))
        else:
            raise SystemExit('unknown part %s' % a)
