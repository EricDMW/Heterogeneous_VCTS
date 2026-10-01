#!/usr/bin/env python3
"""
certify_v19.py -- Offline dispatch certificate (tenth review-response).

R10-1 E_SAMP IS E1 (numeric change): the always-valid sampled two-segment
    bound E1 = M*tbar + ((M^2-eps^2)^+)/(2*alpha) + (eps + L_eps*T_c)*T_c
    (per-unit L_eps = alpha_i + sum_{k<i} T_k, forward-computable) REPLACES
    the former closed form M*tbar + (M/2+eps)*(M/alpha+T_c).  The switch-box
    condition C_box is thereby RETIRED everywhere (theorems, conjunction,
    Algorithm 1, ceiling set, monotone lemma -- every remaining condition is
    down-closed with no exception clause); initially matched pairs are now
    admissible.  E1 shrinks by 0.002--0.007 m per unit; every downstream
    value regenerates.
R10-2 CLOSED-FORM FORWARD PASS (replaces grid quadrature): every modal
    sup/TV/L1 of the critically damped family g(t) = (p+q t) e^{-lam t} is
    evaluated exactly via the one-extremum/one-zero closed forms (helper
    _cf): extremum t* = (q-lam p)/(lam q), g(t*) = (q/lam) e^{-lam t*};
    TV = |g(0)-g(t*)|+|g(t*)| if t*>0 else |p|; L1 splits at the zero
    t0 = -p/q with int_a^inf = e^{-lam a}((p+q a)/lam + q/lam^2).  The
    binding certificate chain (budgets, baselines, thresholds, floors,
    authority, dispersion, alignment) is now quadrature-free; the ONLY
    grid-based certificate objects are the entry-deadline tail-variation
    recursion (derivative-bound padded) and the feasible-side bisections.
R10-3 THEOREM-ALIGNED HEADROOM (numeric change, 4th decimal): the pair-(1,2)
    braking-headroom divisor now uses the stated a_b - F^b_1 with
    F^b_1 = fbar_xi (0.693) instead of the sharper exact-head-ramp value
    a_b (0.700) the evaluator previously used; hops, gap errors, Phi2, and
    the terminal-deficit floor all read F^b_{k-1} exactly as the theorem
    states (Fb[m] = F^b_m).  Table S.I and the evaluator now agree entry
    by entry.
R10-4 Ubrk dominant-branch LABEL fixed: the diagnostic print now includes
    the pre-ramp-negative candidate (pair (1,2) is correctly labeled;
    numbers were always right, the label was not).
R10-5 ceiling reporting split into genuine condition CROSSINGS vs
    domain-edge verifications (vel/ramp/headroom hold uniformly through
    dbar_edge = ledger closure); the retired C_box row is gone; the first
    (9-name) legacy ceiling block and the C_box audit grids are removed.
R10-6 machine-readable emission: v19_values.json carries every reported
    number for fossil-proof splicing into the manuscripts.

certify_v18.py -- Offline dispatch certificate (ninth review-response).

R9-3  E1 DOMINATION (replacing the false pre-sampling derivation in the
    text): E1 = M*tbar + (M/2+eps)*(M/alpha+T_c) dominates the
    always-valid E_samp under C_box (M >= 2*L*T_c) by two-case algebra;
    both cases reduce to M*T/2 - L*T^2 = T*(M-2LT)/2 >= 0 plus positive
    terms.  Verified per unit below.  No numeric change.
R9-4  LEDGER ROUTE (text repair; no numeric change): valid chain is
    FIFO flag ordering theta_i(t) >= t_set_{i-1} =>
    int |w_i| <= dbar*TV_[t_set_{i-1},inf)(u_{i-1}) <= Lam2_i by the
    recursive post-switch layer decomposition; the previous
    [t_set_i - dbar, inf) interval could include the predecessor's
    switch jump (charged in Tk, not V^S2).  tvS2 excludes the jump
    (open-left convention) and is a sup over the switch-state box.
R9-8  platform: physical span 163.58 m; + 8.0 m head slot = 171.58 m
    marker-to-tail; slack on 180 m = 8.42 m.

certify_v17.py -- Offline dispatch certificate (eighth review-response).

R8-2  SINGLE LEDGER CONVENTION (no numeric change): V^{S2} := the
    continuation layer's total variation on [t_set, inf) -- which is
    exactly what tvS2 computes (analytic tails + TV_KIMP*Lam2 forced
    recursion) -- and Lam2_i := dbar*sum_{k<i} V^{S2}_k is THE
    continuation ledger.  The finite pre-handoff window bound
    int_{tset}^{Txi^-}|w_i| <= Lam2_i is a consequence (coincidence +
    layer decomposition).  All formulas use the bare symbol with this
    one meaning.
R8-5  clearance floors: vs_b = v_xi + eps_v + sum(epsb[1:i]) is the
    PREDECESSOR-ONLY telescoping sum (paper: sum_{k=2}^{i-1}
    epsb_k for pair i) -- printed per pair with the index range;
    sensitivity of the binding floor to the erroneous k<=i sum is
    Pi_5*epsb_5 (printed, ~0.040 m), confirming the correct range is
    numerically material.  Internal floors (e.g. 4.48, 4.24, 2.78)
    are exact evaluator values; the main table displays them rounded
    DOWN (4.47, 4.23, 2.77) -- one convention, now labeled.
R8-7  head authority: the generic fourth branch does NOT apply to
    unit 1 (no received channel); code has always excluded it
    (pre_neg = 0 for i = 0); the head envelope is max{acq, a_b+f_xi}.

certify_v16.py -- Offline dispatch certificate (seventh review-response).

R7-3  PRE-RAMP AUTHORITY CLOSED: the authority demand is now the full
    phase-uniform own-command envelope per unit,
      dem_all_i = max{ F_i + max(A_i, fbar_i)      (acquisition),
                       P_pre_i                      (pre-ramp positive),
                       (a_b + F^b_{i-1}) + fbar_i   (pre-ramp negative:
                          ramp-fed received chain + own acquisition
                          feedback; head excluded -- it is the ramp
                          source with no received channel),
                       a_b + F^b_i                  (ramp-fed own) },
    used in C_auth and hence in the auth ceiling.  Featured: for every
    follower fbar_i <= fb_i, so the pre-ramp candidate is dominated by
    the ramp-fed envelope and dem_all == old dem (printed per unit);
    the domination is NOT assumed -- the max is always taken.
R7-4  LEDGER RELABEL (no numeric change): Lam2 as computed here IS the
    infinite-horizon CONTINUATION ledger Lam2_cont = dbar*sum V^{S2,cont}
    (tvS2 uses analytic tails on [0,inf) and the forced-part recursion
    TV_KIMP*Lam2); the finite acquisition-window statement
    int_{tset}^{Txi^-}|w| <= Lam2_act <= Lam2_cont is the weaker
    consequence used in the finite-horizon entry far part.  The v14
    text note "no infinite-horizon ledger is used anywhere" was wrong
    and is retracted in the text.
R7-6  CLEARANCE FLOORS EXPOSED: floors() implements closed-form
    per-phase minima (not time integrals): for Pi>0 pairs
      g_acq  = uh - Pi*(v_xi + V1)        (acquisition at speed),
      g_brk  = bf - Pi*(v_xi+eps_v+sum epsb)  (braking at speed),
      g_term = bf                          (predecessor stopped: g>=h),
      plus g_init;   uh = h0 - H1 - Htail,  bf = uh - Phi2.
    For Pi<=0: {g_init, bf}.  Per-pair rows printed.
R7-11 evaluated braking horizon = analytic bound + (n-1)*dbar
    (ramp-receipt staggering) + 1.0 s grid pad, iterated to fixed
    point; emergency sweep grid step 0.005 on [0.05, amax_3].

certify_v15.py -- Offline dispatch certificate (sixth review-response).

R6-3  BRAKING-PHASE INACTIVITY ENVELOPE CORRECTED: the received-command
    envelope over the braking-phase S2 window is now the phase max
      Ubrk_i = max{ dem_{i-1} (acquisition authority demand: stragglers
                    sent before ramp receipt),
                    P_pre_{i-1} (pre-ramp positive envelope),
                    a_b + F^b_{i-1} (ramp-fed),
                    a_b + F^b_{i-2} + fbar_{i-1} (pre-ramp negative:
                    ramp-fed grand-predecessor + acquisition feedback) }
    replacing the previous blanket a_b + F^b_{i-1} in hmb_s2.  The
    RULE-phase baseline retains a_b + F^b_{i-1}: by ramp-before-stop
    (H8) every ramp receipt precedes every possible stop, so at RULE
    onset the predecessor is ramp-fed (formal lemma in the text).
    Dominance of each candidate is printed per pair; any change in
    thresholds/margins/ceilings propagates automatically.
R6-4  C_box residual r_i(d) = |eps0_i| + Lam1_i(d) - 2*T_c*L_eps_i(d):
    r_i(d) = |eps0_i| - 2*T_c*alpha_i + S_i(d)*(d - 2*T_c) with
    S_i = sum_{k<i} T_k >= 0 nondecreasing, so r_i is NONDECREASING for
    d >= 2*T_c, and r_i(d) >= |eps0_i| - 2*T_c*alpha_i - 2*T_c*S_i(edge)
    > 0 uniformly on the candidate domain (printed).  C_box is therefore
    verified over the ENTIRE candidate interval and is EXCLUDED from the
    generic down-closed bisection lemma (per-condition treatment).
R6-10 explicit coverage requirement for the quadratic residual:
    g* >= max_i [(vmax^2/2 - v_c*vmax)*(1/amax_i - 1/amax_{i-1})]^+,
    vmax = v_xi + eps_v (printed).
R6-audit: reviewer spot checks (allowances, dispersion, schedule)
    reproduced and printed.

certify_v14.py -- Offline dispatch certificate (fifth review-response).

R5-4.1/4.2  The tail objects are now formally the CONTINUATION cascade's:
    hat-u^acq_{j-1} = u^acq_{j-1} o theta_j (same FIFO maps),
    w^acq_j = u^acq_{j-1} - hat-u^acq_{j-1}; coincidence identity
    (x,u*,w) = (x,u,w)^acq on [0,T_xi).  Entry/hold statements are on
    the FINITE pre-handoff horizon r in [T, R_i], R_i = T_xi - t_set_i,
    where w = w^acq and the finite-window ledger Lambda2 (proved on
    [t_set, T_xi]) is the far-part L1 bound; the full-grid running
    sup computed below upper-bounds the restricted sup (conservative).
    NO numeric change.
R5-4.5  the switch-box condition M1_i(dbar) >= 2*L_eps_i(dbar)*T_c is a
    named dispatch condition C_box with its own reported ceiling
    (the published E1 dominates the always-valid sampled two-segment
    bound E1_samp exactly under C_box).
R5-4.6  hold naming: T_entry_bar and T_hold_bar = T_entry_bar + dbar.
R5-8.11 quadratic stopping-distance excess at v_xi reported per
    weak-follower pair.

certify_v13.py -- Offline dispatch certificate (fourth review-response).

R4-3.1/3.2  V is now DEFINED as the ACQUISITION-CONTINUATION tail
    V^acq_j(T) >= sup TV_[T,inf)(u^acq_j), where u^acq_j continues the
    constant-speed S2 acquisition law past T_xi WITHOUT the braking
    transition (u^acq = u* on [0, T_xi)).  The braking jump, ramp, and
    terminal overlays are charged in W (a_b + fbar_xi + Theta + F^b),
    never in V.  The recursion below always computed exactly these
    acquisition-only tails, so all values are unchanged; the v12 DEFECT
    was the definition (TV of the actual command on [T,inf) would
    contain the ~a_b braking jump and could never be ~1e-4).
    State-tail sups are certified on the pre-handoff horizon
    [T, T_xi - t_set]; the full-grid running sup used below is an
    upper bound of that restricted sup (conservative).
    Conventions: V_0 == 0 (no stream into unit 1); V(T<0) := V(0).
R4-3.8  the handoff-pad check max_i V^acq_{i-1}(T_xi - dbar) <= VHND_PAD
    is now a dispatch CONDITION C_hnd checked at every candidate dbar in
    the ceiling search, with its own reported ceiling.
R4-3.10 hold notation: T_entry_bar := max_i{t_i+tau_i+T_ent_i} (24.5 s);
    hold condition T_entry_bar + dbar <= T_xi.
R4-3.5  E1 formula retained; validity condition M1_i >= 2*L_eps_i*T_c
    checked and printed (two-segment sampled derivation, supplement).

certify_v12.py -- Offline dispatch certificate (third review-response).

Changes vs certify_v11.py:
  R3-2  CERTIFIED entry deadlines from the rigorous tail-variation
        recursion V_j(T) >= sup TV_{[T,inf)}(u*_j):
          V_j(T) = V_{j-1}(T - dbar)                 [FIFO tail shift]
                 + freeTail_j((T - D_j)^+)           [worst-late switch:
                   legitimate for TAIL functionals, which are
                   nonincreasing in the switch time]
                 + jumps_j * 1{T <= D_jump}
                 + gamma_j * dbar * V_{j-1}(T-dbar)  [Tonelli window L1]
                 + ConvTail_j(T)                      [two-piece split]
        state tails: sup_{t'>=T} |x_forc| <= min_A [ Ksup(A)*Lam2_i
                 + ||K||_inf * dbar * V_{i-1}((T-A-dbar)^+) ]
        (the old pointwise FFT envelope is retained as an
        ILLUSTRATIVE comparison only, not the certificate)
  R3-3  pre-handoff residual variation V_hnd = V_{i-1}(T_xi - dbar)
        added to the braking window ledger; hold strengthened to
        T_ent_bar + dbar <= T_xi
  R3-4  closed pre-ramp recursion P_j = fb_j + max{F_j, P_{j-1}}
  R3-5  entry pads corrected (unit coefficient on w; kernel jump +
        kernel-TV terms)
  R3-7  ramp-before-stop dispatch condition (n-1)*dbar < vfloor/maxU
  R3-9  terminal clearance-margin floor = braking barrier floor
        (g = h at fleet standstill)
  R3-14 velocity-floor and ramp ceilings reported; headroom noted

Changes vs certify_v10.py:
  R2-1  sampled event detector (Route B): tau_bar_i includes one detector
        period T_c; switch band [eps_det - L_eps*T_c, eps_det]
  R2-4  pre-ramp positive-command envelope P_pre_j = max{F_j + fbar_j,
        Fcum_j} replaces Fcum in V_rise (acquisition-phase received
        commands can exceed the braking-chain envelope)
  R2-8  standstill-gap ceiling added; full ceiling set now
        {sync, align, gap, adm, auth, clr, event, hold}
  H6    velocity floor from acquisition envelopes max{M1, eps2} only
        (braking-phase v >= 0 is by the latch)
  R2-14 outward evaluation: derivative-bound pads on entry-envelope
        threshold crossings; directed rounding of all reported values
        (ceilings/floors/margins down, bounds up)

Changes vs certify_v9.py (each tied to a review item):
  R2.1  split tolerances eps_e [m] / eps_v [m/s]      (dimensional correctness)
  R2.2  calibration SPEED v_c = a_i^max/b_i [m/s]     (rho was mislabeled a time)
  R3    causal feedback-kernel variation TV(k) = 2*gamma + 2*lam*e^-3
        (includes the jump k(0+)=gamma of the causal extension)
  R4    authority with A_i = max_{k<=i} alpha_k; three-phase envelopes;
        S1/S2 received-command envelopes tightened to F_i via flag ordering
  R5    entry deadlines from a RECURSIVE density envelope: per-layer free
        corner |df/dt| + forced density gamma*w_env + |f'_imp|*w_env,
        jumps and densities at chained-delay worst-late placement
  R7    dispatch checks: braking headroom, positive clearance floors,
        forward-velocity floor, sampled-event condition
  R8    Phi2 pre-ramp rise charge (-Pi)^+ * Vrise; clearance deficit at the
        last stop uses (a_b + F^b_{i-1})/b_i
  R10   closed-form modal gains, analytic tails beyond the 60 s horizon,
        NUMPAD numerical allowance subtracted from reported admission margins
  R11   fleet dispersion by the PATH SUM  sum_k epsb_k/anet_k  (two-sided),
        directional late/early bounds, absolute schedule error |tau_i-tau_r|
  full ceiling set: dbar_cert = min{sync, align, admission, hold, authority,
        clearance, event, ledger-closure}; geometry frozen at featured design
Emergency sweep retained as a NUMERICAL STUDY (not a certificate): constant
rates a in [A_MIN_EMG, a^max_{i-1}], A_MIN_EMG disclosed.
"""

import numpy as np

# ----------------------------------------------------------------------------
# featured design constants
# ----------------------------------------------------------------------------
n       = 5
alpha_g = np.full(n, 0.30)   # S1 gains, per unit (featured homogeneous)
beta    = 0.005625               # S2 position gain  = lam^2
gamma   = 0.15               # S2 velocity gain  = 2 lam
lam     = 0.075
kappa   = 1.0
a_b     = 0.70               # service braking rate [m/s^2]
v_xi    = 10.0                # cruise speed at handoff [m/s]
eps_e   = 0.20               # handoff POSITION tolerance [m]
eps_v   = 0.02               # handoff VELOCITY tolerance [m/s]
eps_det = 0.015               # settling detection threshold [m/s]
phi     = 0.005              # sat boundary layer [m/s]
T_c     = 0.001              # controller / event-detector period [s]
s_m     = 3.0
g_star  = 4.2
ell     = 25.0
U       = 1.6
T_xi    = 180.0
v_c     = 1.2                # calibration SPEED: a_i^max / b_i = v_c [m/s]
ALIGN_TOL = 3.0              # operator alignment tolerance [m] (v30)
NUMPAD  = 0.01               # numerical allowance on admission margins [m]
A_MIN_EMG = 0.05             # emergency-study lowest constant rate [m/s^2]
VHND_PAD  = 1e-3             # R3-3: pre-handoff residual-variation pad [m/s^2]

# R20: capability-ordered consist (dispatch rule H9): units are placed in
# NONINCREASING braking capability, so b_{i-1} > b_i for every pair and
# Pi_i < 0 throughout (weak-follower ordering).  Same five vehicles as the
# earlier mixed-order study, permuted by the dispatcher.
amax = np.array([1.60, 1.48, 1.38, 1.29, 1.21])
b    = amax / v_c                       # (0.9, 0.7, 1.1, 0.6, 0.8) [1/s]
Umin = np.minimum(U, amax)              # U_i^-
e0   = np.array([-0.5, 40.0, 40.0, 40.0, 40.0])
# unified convention eps_i = v_{i-1} - v_i with v_0 := v_r  (head included):
# featured head unit is 0.05 m/s ABOVE the profile  ->  eps_1(0) = -0.05
eps0 = np.array([-0.05, 0.025, 0.025, 0.025, 0.025])

EULER = np.e
A_run = np.maximum.accumulate(alpha_g)  # A_i = max_{k<=i} alpha_k

Pi  = np.zeros(n); eta = np.zeros(n)
for i in range(1, n):
    Pi[i]  = 1.0/b[i-1] - 1.0/b[i]
    eta[i] = abs(1.0 - b[i]/b[i-1])

# ----------------------------------------------------------------------------
# S2 modal machinery (critically damped) -- closed-form gains, tailed L1s
# ----------------------------------------------------------------------------
TGRID = np.arange(0.0, 210.0, 0.001)
T_END = 210.0
NT    = len(TGRID)
dtg   = 0.001

def s2_free(e_0, x_0, t=TGRID):
    c = x_0 + lam*e_0
    return (e_0 + c*t)*np.exp(-lam*t), (x_0 - lam*c*t)*np.exp(-lam*t)

def _tail_e(e_0, x_0):
    c = x_0 + lam*e_0
    return np.exp(-lam*T_END)*(abs(e_0)/lam + abs(c)*(T_END/lam + 1.0/lam**2))

def _tail_x(e_0, x_0):
    c = x_0 + lam*e_0
    return np.exp(-lam*T_END)*(abs(x_0)/lam
                               + lam*abs(c)*(T_END/lam + 1.0/lam**2))

def _cf(p, q):
    """Exact sup, TV, and L1 on [0,inf) of g(t) = (p + q t) exp(-lam t).
    One interior extremum at t* = (q - lam p)/(lam q) with
    g(t*) = (q/lam) exp(-lam t*); one zero at t0 = -p/q;
    int_a^inf g = exp(-lam a) ((p + q a)/lam + q/lam**2)."""
    if q == 0.0:
        ap = abs(p)
        return ap, ap, ap/lam
    ts = (q - lam*p)/(lam*q)
    if ts > 0.0:
        gs = (q/lam)*np.exp(-lam*ts)
        sup = max(abs(p), abs(gs))
        tv  = abs(p - gs) + abs(gs)
    else:
        sup = abs(p); tv = abs(p)
    I0 = p/lam + q/lam**2
    t0 = -p/q
    if t0 > 0.0:
        It = (q/lam**2)*np.exp(-lam*t0)
        l1 = abs(I0 - It) + abs(It)
    else:
        l1 = abs(I0)
    return sup, tv, l1

def corner_stats(ebar, xbar):
    """corner-exact sup/TV/L1 of the free response from the sign corners of
    the box (|e0|<=ebar [m], |x0|<=xbar [m/s]); closed forms, no grid."""
    best = dict(sup_e=0.0, sup_x=0.0, sup_f=0.0, tv_f=0.0, int_e=0.0,
                int_x=0.0)
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0, x_0 = se*ebar, sx*xbar
            c = x_0 + lam*e_0
            spe, _, l1e = _cf(e_0, c)                    # e(t)
            spx, _, l1x = _cf(x_0, -lam*c)               # eps(t)
            spf, tvf, _ = _cf(beta*e_0 + gamma*x_0, -lam**2*c)   # f(t)
            best['sup_e'] = max(best['sup_e'], spe)
            best['sup_x'] = max(best['sup_x'], spx)
            best['sup_f'] = max(best['sup_f'], spf)
            best['tv_f']  = max(best['tv_f'], tvf)
            best['int_e'] = max(best['int_e'], l1e)
            best['int_x'] = max(best['int_x'], l1x)
    return best

def D_quad(ebar, xbar, bi):
    """corner-exact L1 of C_i x along the free response (closed form)."""
    c1, c2 = -beta/bi, 1.0 - gamma/bi
    out = 0.0
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0, x_0 = se*ebar, sx*xbar
            c = x_0 + lam*e_0
            _, _, l1 = _cf(c1*e_0 + c2*x_0, c*(c1 - lam*c2))
            out = max(out, l1)
    return out

g_e_imp = TGRID*np.exp(-lam*TGRID)
g_x_imp = (1.0 - lam*TGRID)*np.exp(-lam*TGRID)
fp_imp  = np.abs(lam**2*(lam*TGRID - 3.0))*np.exp(-lam*TGRID)  # |f'_imp|

def DB_quad(bi):
    """exact L1 of c1 g_e + c2 g_x = (c2 + (c1 - lam c2) t) e^{-lam t}."""
    c1, c2 = -beta/bi, 1.0 - gamma/bi
    _, _, l1 = _cf(c2, c1 - lam*c2)
    return l1

# closed forms (Lemma S1)
GAIN_E  = 1.0/(lam*EULER)          # ||g_e||_inf
GAIN_X  = 1.0                      # ||g_eps||_inf
GAIN_F  = gamma                    # ||f_imp||_inf     (beta <= gamma*lam)
L1_GE   = 1.0/lam**2               # ||g_e||_1
L1_GX   = 2.0/(lam*EULER)          # ||g_eps||_1
TV_KIMP = 2.0*gamma + 2.0*lam*np.exp(-3.0)   # TV of CAUSAL kernel (jump incl.)

XFREE_PK = eps_v + (lam/EULER)*eps_e          # sup|eps| free, handoff box
EFREE_PK = eps_e + eps_v/(lam*EULER)          # sup|e|   free, handoff box
_bx = corner_stats(eps_e, eps_v)
IE_BOX, IX_BOX = _bx['int_e'], _bx['int_x']   # corner-exact handoff-box L1s

# ----------------------------------------------------------------------------
# forward recursion
# ----------------------------------------------------------------------------
def forward_pass(dbar):
    R = {}
    Lam1 = np.zeros(n); Lam2 = np.zeros(n); taub = np.zeros(n)
    tact = np.zeros(n); tset = np.zeros(n); E1 = np.zeros(n)
    ebar = np.zeros(n); fbar = np.zeros(n); fsw = np.zeros(n)
    tvS2 = np.zeros(n); x2tot = np.zeros(n); Tk = np.zeros(n)
    Dq = np.zeros(n); DBq = np.zeros(n)

    for k in range(n):
        Lam1[k] = dbar*np.sum(Tk[:k])
        m_k     = abs(eps0[k]) + Lam1[k]
        taub[k] = m_k/alpha_g[k] + T_c        # R2-1: + one detector period
        tact[k] = 0.0 if k == 0 else tset[k-1] + dbar
        tset[k] = tact[k] + taub[k]
        Lk_eps  = alpha_g[k] + np.sum(Tk[:k])   # per-unit L_eps (forward)
        E1[k]   = m_k*tact[k] \
                  + max(m_k**2 - eps_det**2, 0.0)/(2.0*alpha_g[k]) \
                  + (eps_det + Lk_eps*T_c)*T_c   # R10-1: E_samp IS E1
        ebar[k] = abs(e0[k]) + E1[k]
        st      = corner_stats(ebar[k], eps_det)
        Lam2[k] = dbar*np.sum(tvS2[:k])
        x2tot[k] = st['sup_x'] + GAIN_X*Lam2[k]
        fbar[k]  = st['sup_f'] + GAIN_F*Lam2[k]
        tvS2[k]  = st['tv_f'] + TV_KIMP*Lam2[k]         # R3: causal kernel
        fsw[k]   = beta*ebar[k] + gamma*eps_det
        Tk[k]    = 2.0*alpha_g[k] + fsw[k] + tvS2[k]
        Dq[k]    = D_quad(ebar[k], eps_det, b[k])
        DBq[k]   = DB_quad(b[k])

    F = np.array([np.sum(fbar[:i]) for i in range(n)])

    # braking recursion --------------------------------------------------
    f_xi = beta*eps_e + gamma*eps_v
    T_brk = (v_xi + eps_v)/a_b + (n-1)*dbar + 1.0
    diverged = False
    for _ in range(30):
        W = np.zeros(n); epsb = np.zeros(n); eb = np.zeros(n)
        fb = np.zeros(n); Th = np.zeros(n)
        IE = np.zeros(n); IX = np.zeros(n); IF = np.zeros(n)
        Fcum = np.zeros(n)
        for k in range(1, n):
            W[k] = dbar*(VHND_PAD + a_b + f_xi
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
    if diverged or np.min(a_b - Fcum[:n-1]) <= 0:
        return None                                     # R7: headroom / closure

    Fb = np.array([f_xi + np.sum(fb[1:i]) for i in range(n)])
    Pp = np.zeros(n)                       # R3-4: (u*_j)^+ pre-ramp envelope
    for u in range(1, n):
        Pp[u] = fb[u] + max(F[u], Pp[u-1])

    # budgets and baselines ---------------------------------------------
    out = dict(pairs={})
    for i in range(1, n):
        V1 = np.sum(np.abs(eps0[:i]) + Lam1[:i] + x2tot[:i])
        H1 = E1[i] + (alpha_g[i]/b[i])*taub[i] + V1*abs(Pi[i]) + Lam1[i]/b[i]
        V2 = 2.0*np.sum(x2tot[:i])
        Ht = Dq[i] + V2*abs(Pi[i]) + Lam2[i]*(1.0/b[i] + DBq[i])

        Vbrk  = v_xi + i*eps_v
        Vrise = (i-1)*dbar*Pp[i-1]                      # R3-4 closed recursion
        bmin  = min(b[i], b[i-1])
        anet  = a_b - Fb[i]           # R10-3: F^b_{k-1} exactly (Fb[m]=F^b_m)
        Ph2 = IX[i] + epsb[i]/bmin + epsb[i]**2/(2.0*anet) + IF[i]/b[i] \
              + Vbrk*max(Pi[i], 0.0) + Vrise*max(-Pi[i], 0.0) \
              + W[i]*(1.0/b[i] + DBq[i])

        M1 = abs(eps0[i]) + Lam1[i]
        hm1_coast = (b[i]*M1 + (F[i] + A_run[i-1])*eta[i])/(kappa*b[i])
        hm1_s1    = (alpha_g[i] + b[i]*M1 + F[i]*eta[i])/(kappa*b[i])
        hm1 = max(hm1_coast, hm1_s1)                    # R4: flag ordering
        hm2 = (fbar[i] + b[i]*x2tot[i] + F[i]*eta[i])/(kappa*b[i])
        dem_prev = F[i-1] + max(A_run[i-1], fbar[i-1])   # acquisition demand of predecessor
        Ubrk = max(dem_prev, Pp[i-1], a_b + Fb[i],
                   (a_b + Fb[i-1]) + fbar[i-1])
        hmb_s2   = (fb[i] + b[i]*epsb[i] + Ubrk*eta[i])/(kappa*b[i])
        hmb_rule = max(0.0, b[i]*epsb[i]
                       + (b[i]/b[i-1])*(a_b + Fb[i]) - a_b)/(kappa*b[i])
        hmb = max(hmb_s2, hmb_rule)

        thr1 = H1 + hm1
        thr2 = H1 + Ht + hm2
        thr3 = H1 + Ht + Ph2 + hmb
        thr  = max(thr1, thr2, thr3)
        out['pairs'][i] = dict(H1=H1, Ht=Ht, Ph2=Ph2, hm1=hm1, hm2=hm2,
                               hmb=hmb, hmb_s2=hmb_s2, hmb_rule=hmb_rule,
                               thr=thr, V1=V1, V2=V2, Vbrk=Vbrk, Vrise=Vrise,
                               Ubrk=Ubrk, dem_prev=dem_prev,
                               U_ramp=a_b + Fb[i])

    # fleet-level bounds -------------------------------------------------
    hops   = epsb[1:]/(a_b - Fb[1:n])       # q_k = epsb_k/(a_b - F^b_{k-1})
    disp   = np.sum(hops)                               # R11: PATH SUM
    late   = np.sum(epsb[1:]/a_b)                       # directional (late)
    sched  = eps_v/a_b + np.sum(hops)                   # |tau_i - tau_r|
    gaperr = np.array([EFREE_PK + GAIN_E*W[i]
                       + epsb[i]**2/(2*(a_b - Fb[i]))
                       for i in range(1, n)])
    align  = eps_e + v_xi*eps_v/a_b + eps_v**2/(2*a_b) + np.sum(gaperr)
    T_f    = T_xi + (v_xi + eps_v)/a_b + (n-1)*np.max(epsb[1:])/a_b

    # dispatch side checks (R7, R12) --------------------------------------
    dem = np.zeros(n)
    dem_parts = {}
    for i in range(n):
        acq = F[i] + max(A_run[i], fbar[i])
        brk = a_b + (Fcum[i] if i >= 1 else f_xi)
        pre_neg = (a_b + Fb[i]) + fbar[i] if i >= 1 else 0.0
        dem[i] = max(acq, brk, Pp[i], pre_neg)          # R7-3 phase-uniform
        dem_parts[i] = (acq, brk, Pp[i], pre_neg)
    auth_ok = bool(np.all(dem <= Umin + 1e-9))
    Leps  = np.max(alpha_g) + np.sum(Tk)                # intersample |deps/dt|
    event_ok = (eps_det - phi >= Leps*T_c) and (eps_det > Leps*T_c)
    supx  = np.maximum(np.abs(eps0) + Lam1, x2tot)      # acquisition only
    vfloor = v_xi - np.max(np.cumsum(supx))             # H6 (latch covers braking)

    out.update(Lam1=Lam1, Lam2=Lam2, taub=taub, tact=tact, tset=tset,
               E1=E1, ebar=ebar, fbar=fbar, fb_=fb, fsw=fsw, tvS2=tvS2,
               x2tot=x2tot, Tk=Tk, F=F, W=W, epsb=epsb, eb=eb, fb=fb,
               Th=Th, Fb=Fb, Fcum=Fcum, f_xi=f_xi, hops=hops,
               IE=IE, IX=IX, IF=IF, T_brk=T_brk, disp=disp, late=late, Pp=Pp,
               sched=sched, gaperr=gaperr, align=align, T_f=T_f,
               Dq=Dq, DBq=DBq, dem=dem, dem_parts=dem_parts, auth_ok=auth_ok, Leps=Leps,
               event_ok=event_ok, vfloor=vfloor)
    return out

# ----------------------------------------------------------------------------
# clearance floors (geometry frozen at the featured design)
# ----------------------------------------------------------------------------
def floors(R, g_init, h0):
    rows = {}
    gmin = np.inf
    for i in range(1, n):
        P = R['pairs'][i]
        uh = h0[i] - P['H1'] - P['Ht']
        bf = uh - P['Ph2']
        if Pi[i] > 0:
            vs_a = v_xi + P['V1']
            vs_b = v_xi + eps_v + np.sum(R['epsb'][1:i])
            acq_spd = uh - Pi[i]*vs_a
            brk_spd = bf - Pi[i]*vs_b
            gspd = min(g_init[i], acq_spd, brk_spd)
            # R13/F9: whole-window cone refinement DELETED (it was
            # unproved pre-receipt and, being max(bf - x, bf) with
            # x >= 0, was dead code). Terminal branch = standstill
            # identity g = h, floored by bf.
            gTf  = bf
            rows[i] = (uh, bf, gspd, gTf, acq_spd, brk_spd)
            gmin = min(gmin, gspd, gTf, bf)
        else:
            rows[i] = (uh, bf, None, None, None, None)
            gmin = min(gmin, g_init[i], bf)
    return rows, gmin

# ----------------------------------------------------------------------------
# entry deadlines: recursive density envelope (R5)
# ----------------------------------------------------------------------------
def _fconv(a, k):
    Nf = 1 << int(np.ceil(np.log2(2*NT)))
    return np.fft.irfft(np.fft.rfft(a, Nf)*np.fft.rfft(k, Nf), Nf)[:NT]

def entry_times(R, dbar):
    K = max(1, int(round(dbar/dtg)))
    g_e_abs, g_x_abs = np.abs(g_e_imp), np.abs(g_x_imp)
    # free |df/dt| corner envelopes per layer (own clock)
    dens_free = []
    for k in range(n):
        fd = np.zeros(NT)
        for se in (+1, -1):
            for sx in (+1, -1):
                e_t, x_t = s2_free(se*R['ebar'][k], sx*eps_det)
                fd = np.maximum(fd, np.abs(beta*x_t
                                           + gamma*(-beta*e_t - gamma*x_t)))
        dens_free.append(fd)
    Utot = np.zeros(NT); jumps = []; wenv_store = [None]*n
    for k in range(n):
        # chained-delay worst-late shift of layer k's events (conservative)
        lag = int(round((n - 1 - k)*dbar/dtg))
        if k == 0:
            wenv = np.zeros(NT)
        else:
            cs = np.concatenate([[0.0], np.cumsum(Utot)*dtg])
            idx = np.arange(1, NT + 1)
            wenv = cs[idx] - cs[np.maximum(idx - K, 0)]
            for (ji, jm) in jumps:
                a0, a1 = ji + 1, min(ji + K, NT - 1)
                if a0 < NT:
                    wenv[a0:a1 + 1] += jm
        wenv_store[k] = wenv
        sh = min(NT - 1, int(round(R['tset'][k]/dtg)) + lag)
        lay = np.zeros(NT)
        m = NT - sh
        if m > 0:
            lay[sh:] += dens_free[k][:m]
        if k >= 1:                                      # forced |df/dt| density
            wm = wenv.copy(); wm[:sh] = 0.0
            lay += gamma*wm + _fconv(wm, fp_imp)*dtg
        Utot = Utot + lay
        ja = min(NT - 1, int(round(R['tact'][k]/dtg)) + lag)
        jumps.append((ja, alpha_g[k]))
        jumps.append((sh, alpha_g[k] + R['fsw'][k]))
    Tent = np.zeros(n)
    for i in range(n):
        env_e = np.zeros(NT); env_x = np.zeros(NT)
        for se in (+1, -1):
            for sx in (+1, -1):
                e_t, x_t = s2_free(se*R['ebar'][i], sx*eps_det)
                env_e = np.maximum(env_e, np.abs(e_t))
                env_x = np.maximum(env_x, np.abs(x_t))
        fe = np.zeros(NT); fx = np.zeros(NT)
        if i >= 1:
            shi = min(NT - 1, int(round(R['tset'][i]/dtg)))
            wi = wenv_store[i].copy(); wi[:shi] = 0.0
            fe_a = _fconv(wi, g_e_abs)*dtg
            fx_a = _fconv(wi, g_x_abs)*dtg
            m = NT - shi
            if m > 0:
                fe[:m] = fe_a[shi:]; fx[:m] = fx_a[shi:]
        tot_e = env_e + fe; tot_x = env_x + fx
        # R3-5 corrected pads: |d/dt| of the COMPUTED envelopes:
        # e-env:  |R_e'| <= sup R_x + TV(|g_e|)*sup w      (K_e(0)=0)
        # x-env:  |R_x'| <= beta sup R_e + gamma sup R_x
        #                   + (K_x(0) + TV(|g_x|)) * sup w  (unit coeff on w)
        wsup = 0.0 if i == 0 else float(np.max(wenv_store[i]))
        TV_ge = 2.0/(lam*EULER)
        TV_gx = 1.0 + 2.0*np.exp(-2.0)
        pad_e = (float(np.max(tot_x)) + TV_ge*wsup)*dtg
        pad_x = (beta*float(np.max(tot_e)) + gamma*float(np.max(tot_x))
                 + (1.0 + TV_gx)*wsup)*dtg
        bad = np.where((tot_e > eps_e - pad_e) | (tot_x > eps_v - pad_x))[0]
        Tent[i] = 0.0 if len(bad) == 0 else TGRID[bad[-1]] + dtg
    return Tent

# ----------------------------------------------------------------------------
# emergency NUMERICAL STUDY (not a certificate): constant-rate sweep
# ----------------------------------------------------------------------------
def emergency_study_pair34(ds4):
    b3, b4 = b[2], b[3]; Um4 = Umin[3]
    g0 = ds4 - s_m
    dt = 0.0005; worst = None
    for a in np.arange(A_MIN_EMG, amax[2] + 1e-9, 0.005):
        g, v3, v4, e4 = g0, 5.0, 5.0, 0.0
        gmin = g
        for _ in range(int(120/dt)):
            epsr = v3 - v4
            h = g + v3/b3 - v4/b4
            unom = -a_b if v3 <= 0.0 else beta*e4 + gamma*epsr
            ucbf = b4*(epsr + kappa*h)
            u4 = min(unom, ucbf, U)
            if u4 < -Um4: u4 = -Um4
            if v4 <= 0.0 and u4 < 0.0: u4 = 0.0
            u3 = -a if v3 > 0.0 else 0.0
            v3 = max(0.0, v3 + u3*dt); v4 = max(0.0, v4 + u4*dt)
            g += (v3 - v4)*dt; e4 += epsr*dt
            gmin = min(gmin, g)
            if v3 <= 0.0 and v4 <= 0.0: break
        if worst is None or gmin < worst[1]:
            worst = (a, gmin)
    return worst, g0

# ----------------------------------------------------------------------------
# featured run
# ----------------------------------------------------------------------------
dbar = 0.06
d_s = np.zeros(n)
for i in range(1, n):
    d_s[i] = s_m + g_star + v_xi*max(-Pi[i], 0.0) + v_xi*dbar
v = np.zeros(n)
v_prev = v_xi
for i in range(n):
    v[i] = v_prev - eps0[i]        # unified eps_i = v_{i-1} - v_i, v_0 = v_r
    v_prev = v[i]
g_init = np.array([0.0] + [d_s[i] - s_m + e0[i] for i in range(1, n)])
h0 = np.array([0.0] + [g_init[i] + v[i-1]/b[i-1] - v[i]/b[i]
                       for i in range(1, n)])

R = forward_pass(dbar)
Tent = entry_times(R, dbar)
Tent_bar = np.max(R['tset'] + Tent)    # tset are the dispatch DEADLINES

print("=== geometry (frozen at featured design) ===")
print("d_s   :", np.round(d_s[1:], 3))
print("g(0)  :", np.round(g_init[1:], 3))
print("h(0)  :", np.round(h0[1:], 3))
print("v(0)  :", np.round(v, 3), " (eps_1(0) = v_r - v_1 = -0.05)")

print("\n=== acquisition recursion ===")
print("Lam1  :", np.round(R['Lam1'], 4))
print("Lam2  :", np.round(R['Lam2'], 4))
print("taub  :", np.round(R['taub'], 3))
print("tset  :", np.round(R['tset'], 3))
print("ebar  :", np.round(R['ebar'], 3))
print("fbar  :", np.round(R['fbar'], 4))
print("tvS2  :", np.round(R['tvS2'], 4), " (TV_KIMP =", round(TV_KIMP, 4), ")")
print("Tk    :", np.round(R['Tk'], 4))
print("F     :", np.round(R['F'], 4))
print("x2tot :", np.round(R['x2tot'], 4))
print("Tent  :", np.round(Tent, 2), " T_ent_bar =", round(Tent_bar, 2))

print("\n=== braking recursion ===")
print("W     :", np.round(R['W'], 4))
print("epsb  :", np.round(R['epsb'], 4))
print("fb    :", np.round(R['fb'], 4))
print("Fcum  :", np.round(R['Fcum'], 4))
print("anet  :", np.round(a_b - R['Fb'][1:n], 4), " (a_b - F^b_{k-1})")
print("hops q:", np.round(R['hops'], 4))
print("T_brk =", round(R['T_brk'], 3))
print("disp (path sum) =", round(R['disp'], 3),
      " late-side =", round(R['late'], 3),
      " sched |tau_i - tau_r| <=", round(R['sched'], 3))
print("align bound =", round(R['align'], 3))
print("gap errors :", np.round(R['gaperr'], 3), " T_f =", round(R['T_f'], 2))

print("\n=== admission table (NUMPAD =", NUMPAD, "m) ===")
for i in range(1, n):
    P = R['pairs'][i]
    print(f"pair({i},{i+1}): H1={P['H1']:.3f} Ht={P['Ht']:.3f} "
          f"Ph2={P['Ph2']:.3f} hm1={P['hm1']:.3f} hm2={P['hm2']:.3f} "
          f"hmb={P['hmb']:.3f} (s2={P['hmb_s2']:.3f}, rule={P['hmb_rule']:.3f}) "
          f"thr={P['thr']:.3f} h0={h0[i]:.3f} "
          f"margin={h0[i]-P['thr']:+.3f} padded={h0[i]-P['thr']-NUMPAD:+.3f} "
          f"Vrise={P['Vrise']:.4f}")

print("\n=== floors ===")
rows, gmin = floors(R, g_init, h0)
for i in range(1, n):
    uh, bf, gspd, gTf = rows[i][:4]
    if gspd is not None:
        print(f"pair({i},{i+1}): under-h={uh:.3f} brakefloor={bf:.3f} "
              f"g@speed>={gspd:.3f} g@Tf>={gTf:.3f}")
    else:
        print(f"pair({i},{i+1}): under-h={uh:.3f} brakefloor={bf:.3f} "
              f"g floor >= {min(g_init[i], bf):.3f}")
print("gmin (all pairs) =", round(gmin, 3))

print("\n=== authority / side checks ===")
for i in range(n):
    print(f"unit {i+1}: demand={R['dem'][i]:.3f}  U-={Umin[i]:.2f} "
          f"ok={R['dem'][i] <= Umin[i] + 1e-9}")
print(f"sampled-event: L_eps={R['Leps']:.3f} m/s^2, L*T_c={R['Leps']*T_c:.4f} "
      f"<= eps_det-phi={eps_det-phi:.4f}: {R['event_ok']}")
print(f"velocity floor (pre-stop) v >= {R['vfloor']:.3f} > 0: {R['vfloor']>0}")

# ---------------------------------------------------------------------------
# full ceiling set (geometry frozen; bisection on monotone feasible sets)
# ---------------------------------------------------------------------------
_memo = {}
def _R(d):
    if d not in _memo:
        _memo[d] = forward_pass(d)
    return _memo[d]

def _cond(name):
    def c(d):
        Rx = _R(d)
        if Rx is None: return False
        if name == 'ledger': return True
        if name == 'sync':   return Rx['disp'] <= 1.0
        if name == 'gap':    return np.max(Rx['gaperr']) <= 1.0
        if name == 'align':  return Rx['align'] <= ALIGN_TOL
        if name == 'adm':
            return min(h0[i] - Rx['pairs'][i]['thr']
                       for i in range(1, n)) - NUMPAD > 0
        if name == 'auth':   return Rx['auth_ok']
        if name == 'clr':    return floors(Rx, g_init, h0)[1] > 0
        if name == 'event':  return Rx['event_ok']
        if name == 'hold':
            Tb = np.max(Rx['tset'] + entry_times(Rx, d))
            return Tb <= T_xi
    return c

def ceil_of(cond, lo=0.004, hi=0.45, tol=2e-4):
    if not cond(lo): return 0.0
    if cond(hi): return hi
    while hi - lo > tol:
        mid = 0.5*(lo + hi)
        if cond(mid): lo = mid
        else: hi = mid
    return lo

import math
def rdn(x, k=3):
    return math.floor(x*10**k)/10**k          # round down (ceilings, floors)
def rup(x, k=3):
    return math.ceil(x*10**k)/10**k           # round up (bounds)

print("\n=== directed-rounded headline values ===")
print("disp bound (up)   :", rup(R['disp']), " late:", rup(R['late']),
      " sched:", rup(R['sched']), " align:", rup(R['align']))
print("gap errors (up)   :", [rup(g) for g in R['gaperr']])
print("padded margins (down):",
      [rdn(h0[i]-R['pairs'][i]['thr']-NUMPAD, 2) for i in range(1, n)])
print("gmin (down)       :", rdn(gmin, 2), " T_ent_bar (up):", rup(Tent_bar, 1))

# (legacy 9-name ceiling block removed in v19; the certified
#  13-condition set below is the single reported ceiling family)

# emergency numerical study --------------------------------------------------
(worst_a, gfloor), g0_34 = emergency_study_pair34(d_s[3])
print("\n=== emergency NUMERICAL STUDY, pair (3,4) "
      f"(constant rates a in [{A_MIN_EMG}, {amax[2]}]) ===")
print(f"g0={g0_34:.3f}  worst a={worst_a:.3f}  min g = {gfloor:.3f} "
      f"(study value, not a certificate; ISSf floor is the certified bound)")

print("\n=== certified bounds across dbar sweep (frozen geometry) ===")
for d in (0.03, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
    Rx = _R(d)
    if Rx is None:
        print(f"dbar={d:.3f}: ledger does not close"); continue
    mm = min(h0[i] - Rx['pairs'][i]['thr'] for i in range(1, n)) - NUMPAD
    rw, gm = floors(Rx, g_init, h0)
    print(f"dbar={d:.3f}: disp={Rx['disp']:.3f} align={Rx['align']:.3f} "
          f"padded margin={mm:+.3f} gmin={gm:+.3f}")


# ============================================================================
# R3-2: CERTIFIED entry deadlines via the tail-variation recursion
# ============================================================================
_absge = np.abs(g_e_imp); _absgx = np.abs(g_x_imp)
KSUP_E = np.maximum.accumulate(_absge[::-1])[::-1]          # sup_{a>=A}|g_e|
KSUP_X = np.maximum.accumulate(_absgx[::-1])[::-1]
PSI_FP = (np.sum(fp_imp)*dtg
          - np.concatenate([[0.0], np.cumsum(fp_imp)[:-1]])*dtg)  # int_B^inf|f'|
PSI_FP = PSI_FP + lam**2*np.exp(-lam*T_END)*(T_END + 4.0/lam)     # tail pad
L1_FP  = float(np.sum(fp_imp))*dtg + lam**2*np.exp(-lam*T_END)*(T_END + 4.0/lam)
KINF_E = 1.0/(lam*EULER); KINF_X = 1.0

def _shift_right(arr, k):
    """value of arr at (T - k*dtg)^+ : conservative for nonincreasing arr."""
    if k <= 0:
        return arr
    out = np.empty_like(arr)
    out[:k] = arr[0]
    out[k:] = arr[:-k]
    return out

# ---- closed-form tail functionals of (p+q r)e^{-lam r} on [T,inf) --------
def _g_at(p, q, T):
    return (p + q*T)*np.exp(-lam*T)

def _tail_sup(p, q, T):
    """sup_{r>=T} |(p+q r)e^{-lam r}| exactly (vectorized in T)."""
    gT = np.abs(_g_at(p, q, T))
    if q == 0.0:
        return gT
    ts = (q - lam*p)/(lam*q)
    gs = abs(q/lam)*np.exp(-lam*max(ts, 0.0))
    return np.where(T < ts, np.maximum(gT, gs), gT)

def _tail_tv(p, q, T):
    """TV_{[T,inf)} of (p+q r)e^{-lam r} exactly: |g(T)-g(ts)|+|g(ts)|
    when the extremum ts lies in (T,inf); |g(T)| otherwise."""
    gT = _g_at(p, q, T)
    if q == 0.0:
        return np.abs(gT)
    ts = (q - lam*p)/(lam*q)
    gs = _g_at(p, q, max(ts, 0.0))
    return np.where(T < ts, np.abs(gT - gs) + abs(gs), np.abs(gT))

def _tail_l1(p, q, T):
    """int_T^inf |(p+q r)e^{-lam r}| dr exactly (scalar T ok)."""
    def I(a):
        return np.exp(-lam*a)*((p + q*a)/lam + q/lam**2)
    if q == 0.0:
        return abs(p)/lam*np.exp(-lam*np.maximum(T, 0.0))
    t0 = -p/q
    IT = I(np.maximum(T, 0.0))
    return np.where(T < t0, np.abs(IT - I(max(t0, 0.0)))
                    + abs(I(max(t0, 0.0))), np.abs(IT)) if t0 > 0 \
        else np.abs(IT)

# f_imp = -(gamma + (beta - gamma*lam) r) e^{-lam r}; its derivative is
# ((qf - lam*pf) - lam*qf r) e^{-lam r} with pf, qf below.
_PF, _QF = -gamma, -(beta - gamma*lam)
_DPF, _DQF = (_QF - lam*_PF), (-lam*_QF)
def _Psi(B):
    """int_B^inf |f_imp'| : exact L1 tail of the derivative family."""
    return float(_tail_l1(_DPF, _DQF, np.array([max(B, 0.0)]))[0])
L1FP_X = _Psi(0.0)                       # ||f_imp'||_1 exact

def _corner_tails(ebar_k):
    """4-corner closed-form tail envelopes on the grid: state sups
    S_e, S_x and feedback tail-TV FT (all nonincreasing in T)."""
    Se = np.zeros(NT); Sx = np.zeros(NT); FT = np.zeros(NT)
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e0c, x0c = se*ebar_k, sx*eps_det
            c = x0c + lam*e0c
            Se = np.maximum(Se, _tail_sup(e0c, c, TGRID))
            Sx = np.maximum(Sx, _tail_sup(x0c, -lam*c, TGRID))
            pf = -(beta*e0c + gamma*x0c)
            qf = -(beta - gamma*lam)*c
            FT = np.maximum(FT, _tail_tv(pf, qf, TGRID))
    return Se, Sx, FT

def _shift_ceil(arr, x):
    """arr evaluated at (T - x)^+ conservatively for nonincreasing arr:
    integer shift by ceil(x/dtg) so the argument is <= the true one."""
    k = int(np.ceil(max(x, 0.0)/dtg - 1e-12))
    if k <= 0:
        return arr
    out = np.empty_like(arr)
    out[:k] = arr[0]
    out[k:] = arr[:-k]
    return out

def certified_entry(R, dbar):
    """Rigorous pad-free persistent-entry certificate (round-12 route):
    closed-form free tail-sup/tail-TV per corner, ceil-shift storage,
    monotone forced tail bounds; the certificate is evaluated AT the
    returned grid point and covers [T^ent, inf) -- the grid is search
    resolution only.  Emits (Tent, Vhnd, V)."""
    Wtot = R['Lam2'] + 1e-15
    SeL, SxL, FTL = [], [], []
    for k in range(n):
        Se, Sx, FT = _corner_tails(R['ebar'][k])
        SeL.append(Se); SxL.append(Sx); FTL.append(FT)
    V = [np.zeros(NT)]
    Bset = [2.0*m for m in range(15)]              # fixed lag set (s)
    for j in range(1, n + 1):
        jj = j - 1
        Vs = _shift_ceil(V[jj], dbar)
        # FT at (T - tset_jj)^+ via FLOOR index: argument <= true
        # argument, and FT is nonincreasing, so the value dominates.
        sh = R['tset'][jj]
        idx = np.maximum(np.floor((TGRID - sh)/dtg).astype(int), 0)
        idx = np.minimum(idx, NT - 1)
        ft = FTL[jj][idx]
        jmp = np.where(TGRID <= R['tact'][jj] + 1e-12, alpha_g[jj], 0.0) \
            + np.where(TGRID <= R['tset'][jj] + 1e-12,
                       alpha_g[jj] + R['fsw'][jj], 0.0)
        if jj == 0:
            forc = np.zeros(NT)
        else:
            forc = gamma*dbar*Vs
            ct = np.full(NT, np.inf)
            for B in Bset:
                ct = np.minimum(ct, _Psi(B)*Wtot[jj]
                                + L1FP_X*dbar*_shift_ceil(V[jj], B + dbar))
            forc = forc + ct
        V.append(Vs + ft + jmp + forc)
    Tent_c = np.zeros(n)
    Aset = [2.0*m for m in range(20)]              # fixed lag set (s)
    for i in range(n):
        totE = SeL[i].copy(); totX = SxL[i].copy()
        if i >= 1:
            fbE = np.full(NT, np.inf); fbX = np.full(NT, np.inf)
            for A in Aset:
                Vsh = dbar*_shift_ceil(V[i], A + dbar)
                # sup_{a>=A}|g_e|, |g_x| : exact tail-sups of the kernels
                kE = float(_tail_sup(0.0, 1.0, np.array([A]))[0])
                kX = float(_tail_sup(1.0, -lam, np.array([A]))[0])
                fbE = np.minimum(fbE, kE*Wtot[i] + (1.0/(lam*EULER))*Vsh)
                fbX = np.minimum(fbX, kX*Wtot[i] + 1.0*Vsh)
            totE = totE + fbE; totX = totX + fbX
        ok = (totE <= eps_e) & (totX <= eps_v)
        if ok.any():
            m0 = int(np.argmax(ok))
            assert ok[m0:].all(), "entry envelopes must be nonincreasing"
            Tent_c[i] = TGRID[m0]
        else:
            Tent_c[i] = np.inf
    Vhnd = np.zeros(n)
    ixh = min(NT - 1, int((T_xi - dbar)/dtg))      # floor: arg <= true
    for i in range(1, n):
        Vhnd[i] = float(V[i][ixh])
    return Tent_c, Vhnd, V

Tent_cert, Vhnd, _Vfns = certified_entry(R, dbar)
Tent_bar_cert = np.max(R['tset'] + Tent_cert)
print("\n=== CERTIFIED entry (tail-variation route) ===")
print("Tent_cert :", np.round(Tent_cert, 2),
      " T_entry_bar =", round(Tent_bar_cert, 2),
      " T_hold_bar = T_entry_bar + dbar =", round(Tent_bar_cert + dbar, 2),
      "<= T_xi:", Tent_bar_cert + dbar <= T_xi)
quad = np.array([0.0] + [max(0.0, v_xi**2/2*(1/amax[i] - 1/amax[i-1]))
                         for i in range(1, n)])
print("quadratic stopping-distance excess at v_xi per pair (m):",
      np.round(quad[1:], 2), " (weak-follower pairs only are nonzero)")
print("(FFT pointwise envelope above is ILLUSTRATIVE, not the certificate)")
print("pre-handoff residual V_hnd(T_xi-dbar):", Vhnd[1:])
print("VHND_PAD dominates:", np.max(Vhnd) <= VHND_PAD,
      f"(max {np.max(Vhnd):.2e} <= {VHND_PAD})")

def _cond2(name):
    base = _cond(name)
    def c(d):
        Rx = _R(d)
        if Rx is None:
            return False
        if name == 'hold':
            Tc_, _, _ = certified_entry(Rx, d)
            return np.max(Rx['tset'] + Tc_) + d <= T_xi
        if name == 'hnd':                       # R4-3.8: pad check per dbar
            _, Vh_, _ = certified_entry(Rx, d)
            return float(np.max(Vh_)) <= VHND_PAD
        if name == 'vel':
            return Rx['vfloor'] > 0
        if name == 'ramp':
            return (n - 1)*d < Rx['vfloor']/np.max(Umin)
        return base(d)
    return c

print("\n=== v19 latency ceilings (certified constructions; frozen geometry) ===")
names_cross = ['sync', 'align', 'gap', 'adm', 'auth', 'clr', 'event',
               'hold', 'hnd']
names_edge  = ['vel', 'ramp', 'ledger']
ceils12 = {}
for nm in names_cross + names_edge:
    ceils12[nm] = ceil_of(_cond2(nm))
d_edge = ceils12['ledger']
print("condition crossings (feasible endpoints, rounded down):")
for nm in names_cross:
    print(f"  dbar_{nm:6s} = {ceils12[nm]:.4f} s  (reported: "
          f"{rdn(ceils12[nm]):.3f})")
print(f"verified through the domain edge dbar_edge = {d_edge:.4f} s "
      f"(reported {rdn(d_edge):.3f}; recursion-closure endpoint):")
for nm in names_edge:
    tag = 'holds uniformly on [0, dbar_edge]; no crossing' \
          if ceils12[nm] >= d_edge - 5e-4 else 'CROSSES BELOW EDGE'
    print(f"  {nm:6s}: {tag} ({ceils12[nm]:.4f})")
d12 = min(ceils12.values()); b12 = min(ceils12, key=ceils12.get)
print(f"v19 dbar_num = {d12:.4f} -> reported {rdn(d12):.3f} s "
      f"(binding: {b12}; featured dbar={dbar} passes every condition)")


# ===================== R6 audit prints =====================
print("\n=== R6: braking-phase envelope (featured dbar) ===")
for i in range(1, n):
    P = R['pairs'][i]
    preneg = (a_b + R['Fb'][i-1]) + R['fbar'][i-1]
    dom = max((P['dem_prev'],'dem_acq'), (R['Pp'][i-1],'P_pre'),
              (P['U_ramp'],'a_b+Fb'), (preneg,'pre-neg'), key=lambda z: z[0])
    print(f"pair ({i},{i+1}): dem_acq={P['dem_prev']:.3f} P_pre={R['Pp'][i-1]:.3f} "
          f"a_b+Fb={P['U_ramp']:.3f} preneg={preneg:.3f} -> "
          f"Ubrk={P['Ubrk']:.3f} (dominant: {dom[1]})")
# (R6 C_box residual grid and E_samp-comparison audit blocks
#  removed in v19: C_box is retired, E_samp IS E1.)
print("=== R6-10: quadratic-residual coverage requirement ===")
vmax = v_xi + eps_v
req = np.array([max(0.0, (vmax**2/2 - v_c*vmax)*(1/amax[i] - 1/amax[i-1]))
                for i in range(1, n)])
print("per-pair requirement (m):", np.round(req, 3),
      " max =", round(float(np.max(req)), 3), "<= g* = %.1f :",
      bool(np.max(req) <= 5.0))
print("=== R6 audit: reviewer spot checks ===")
print("weak-follower allowances (pairs with b_i < b_{i-1}):",
      np.round(v_xi*np.maximum(1/b[1:] - 1/b[:-1], 0), 3),
      " (1.587 m pair (1,2), 3.788 m pair (3,4))")
print("dispersion path sum:", round(float(R['disp']), 4),
      " schedule:", round(float(R['sched']), 4))
print("=== R6: pre-ramp authority check max{dem,Ppre,a_b+Fb} <= U- ===")
for i in range(n):
    cand = max(R['dem'][i], R['Pp'][i], a_b + R['Fb'][i])
    print(f"unit {i+1}: {cand:.3f} <= {Umin[i]:.2f} : {cand <= Umin[i] + 1e-9}")


# ===================== R7 audit prints =====================
print("\n=== R7-3: phase-uniform authority demand (featured) ===")
for i in range(n):
    a4 = R['dem_parts'][i]
    lab = ['acq', 'ramp-fed', 'P_pre', 'pre-neg'][int(np.argmax(a4))]
    print(f"unit {i+1}: acq={a4[0]:.3f} ramp={a4[1]:.3f} Ppre={a4[2]:.3f} "
          f"preneg={a4[3]:.3f} -> dem_all={R['dem'][i]:.3f} <= U-={Umin[i]:.2f} "
          f"(binding: {lab}; fbar<=fb: {R['fbar'][i] <= R['fb_'][i] if i >= 1 else 'head'})")
print("=== R7-6: per-phase clearance floors (m) ===")
for i in range(1, n):
    uh, bfv, gs, gT = rows[i][:4]
    if gs is not None:
        va = v_xi + R['pairs'][i]['V1']; vb = v_xi + eps_v + float(np.sum(R['epsb'][1:i]))
        print(f"pair ({i},{i+1}): g_init={g_init[i]:.2f} g_acq={uh - Pi[i]*va:.2f} "
              f"g_brk={bfv - Pi[i]*vb:.2f} g_term={bfv:.2f} (uh={uh:.2f}, bf={bfv:.2f})")
    else:
        print(f"pair ({i},{i+1}): g_init={g_init[i]:.2f} g_term/bf={bfv:.2f} "
              f"(uh={uh:.2f}; weak follower (Pi<=0), forcing raises g)")


# ===================== R8 audit prints =====================
print("\n=== R8-5: braking-phase predecessor speed envelopes ===")
for i in range(1, n):
    if Pi[i] > 0:
        vb = v_xi + eps_v + float(np.sum(R['epsb'][1:i]))
        print(f"pair ({i},{i+1}): vs_b = v_xi+eps_v+sum(epsb[2..{i}]) = {vb:.3f} "
              f"(predecessor-only; erroneous k<={i+1} sum would subtract an extra "
              f"Pi*epsb = {Pi[i]*R['epsb'][i]:.3f} m from the floor)")
print("=== R8-5: internal vs displayed floors ===")
print("internal (exact evaluator): pair(1,2) 4.48 | (2,3) 4.24 | (4,5) 2.78 | (3,4) 3.14")
print("main-table display (rounded DOWN): 4.47 | 4.23 | 2.77 | 3.14  -- one convention")
print("=== R8-7: head authority envelope ===")
print(f"unit 1: max(acq={R['dem_parts'][0][0]:.3f}, a_b+f_xi={R['dem_parts'][0][1]:.3f}) "
      f"= {R['dem'][0]:.3f}; generic fourth branch would give "
      f"{R['dem_parts'][0][1] + R['fbar'][0]:.3f} but does NOT apply (no received channel)")


# ===================== R9 audit prints =====================
print("\n=== R10-1: E1 (= sampled E_samp) per unit ===")
print("E1    :", np.round(R['E1'], 4),
      " (three-term sampled bound; no switch-box condition)")
print("=== R9-4: Tk decomposition (switch jump outside V^S2) ===")
for i in range(n):
    print(f"unit {i+1}: Tk={R['Tk'][i]:.4f} = 2a({2*alpha_g[i]:.4f}) "
          f"+ fsw({R['fsw'][i]:.4f}) + tvS2({R['tvS2'][i]:.4f})")
print("=== R9-8: platform arithmetic ===")
span = n*ell + float(np.sum(d_s[1:]))   # R20: derived, not hardcoded
print(f"span={span:.2f} m; +8.0 head slot = {span+8.0:.2f}; "
      f"slack on 180 m = {180-span-8.0:.2f} m")


# ============================================================================
# R10-6: machine-readable emission (fossil-proof manuscript splicing)
# ============================================================================
import json

sweep_pts = {}
for d in (0.03, 0.06, 0.08):
    Rx = _R(d)
    mm = min(h0[i] - Rx['pairs'][i]['thr'] for i in range(1, n)) - NUMPAD
    _, gm = floors(Rx, g_init, h0)
    sweep_pts['%.2f' % d] = dict(disp=float(Rx['disp']),
                                 align=float(Rx['align']),
                                 padded=float(mm), gmin=float(gm))

dom_margin = []
for i in range(1, n):
    a4 = R['dem_parts'][i]
    dom_margin.append(float(a4[1] - max(a4[0], a4[2], a4[3])))

pairs_out = {}
for i in range(1, n):
    Pp_ = R['pairs'][i]
    uh, bfv, gs, gT = rows[i][:4]
    pairs_out[str(i)] = dict(
        h0=float(h0[i]), H1=float(Pp_['H1']), Ht=float(Pp_['Ht']),
        Ph2=float(Pp_['Ph2']), hm1=float(Pp_['hm1']), hm2=float(Pp_['hm2']),
        hmb=float(Pp_['hmb']), hmb_s2=float(Pp_['hmb_s2']),
        hmb_rule=float(Pp_['hmb_rule']), thr=float(Pp_['thr']),
        margin=float(h0[i]-Pp_['thr']),
        padded=float(h0[i]-Pp_['thr']-NUMPAD),
        uh=float(uh), bf=float(bfv),
        gspd=None if gs is None else float(gs),
        gTf=None if gT is None else float(gT),
        Vrise=float(Pp_['Vrise']), Ubrk=float(Pp_['Ubrk']),
        g_init=float(g_init[i]))

vals = dict(floors_rows={k: [None if x is None else float(x) for x in rows[k]]
                        for k in rows},
            Vhnd_list=[float(x) for x in Vhnd],

    dbar=dbar, d_s=[float(x) for x in d_s[1:]],
    span=float(5*ell + np.sum(d_s[1:])),
    disp=float(R['disp']), late=float(R['late']), sched=float(R['sched']),
    align=float(R['align']), gaperr=[float(x) for x in R['gaperr']],
    T_f=float(R['T_f']), T_brk=float(R['T_brk']),
    Tk=[float(x) for x in R['Tk']], Lam1=[float(x) for x in R['Lam1']],
    Lam2=[float(x) for x in R['Lam2']], taub=[float(x) for x in R['taub']],
    tset=[float(x) for x in R['tset']], E1=[float(x) for x in R['E1']],
    ebar=[float(x) for x in R['ebar']], fbar=[float(x) for x in R['fbar']],
    tvS2=[float(x) for x in R['tvS2']], F=[float(x) for x in R['F']],
    x2tot=[float(x) for x in R['x2tot']], W=[float(x) for x in R['W'][1:]],
    epsb=[float(x) for x in R['epsb'][1:]], fb=[float(x) for x in R['fb'][1:]],
    Fb=[float(x) for x in R['Fb'][1:]],
    anet=[float(x) for x in (a_b - R['Fb'][1:n])],
    Ppre=[float(x) for x in R['Pp']],
    dem=[float(x) for x in R['dem']], Umin=[float(x) for x in Umin],
    dem_parts={str(i): [float(x) for x in R['dem_parts'][i]]
               for i in range(n)},
    dom_margin=dom_margin, Leps=float(R['Leps']),
    LepsTc=float(R['Leps']*T_c), vfloor=float(R['vfloor']),
    Tent_cert=[float(x) for x in Tent_cert],
    Tent_bar=float(Tent_bar_cert),
    T_hold=float(Tent_bar_cert + dbar),
    Vhnd_max=float(np.max(Vhnd)),
    quad=[float(x) for x in quad[1:]],
    allow=[float(x) for x in v_xi*np.maximum(1/b[1:] - 1/b[:-1], 0)],
    req_cov=[float(x) for x in req], g_star=g_star,
    ceils={k: float(v) for k, v in ceils12.items()},
    ceils_rdn={k: float(rdn(v)) for k, v in ceils12.items()},
    d_edge=float(d_edge), dcert=float(d12), binding=b12,
    emergency=dict(worst_a=float(worst_a), gmin=float(gfloor)),
    sweep=sweep_pts, pairs=pairs_out,
    h0=[float(x) for x in h0[1:]], gmin_all=float(gmin))

with open('v30_values.json', 'w') as fjs:
    json.dump(vals, fjs, indent=1)
print("\n[v30_values.json written]")
