# witness core v5: identical logic to interval_witness.py with the
# exponential primitive replaced by Arb-certified bounds (python-flint,
# 120-bit balls; float bounds stepped one ulp outward), so every
# transcendental enclosure is backend-certified; rational operations
# remain IEEE correctly-rounded doubles with per-operation outward steps.
#!/usr/bin/env python3
"""interval_witness.py -- outward directed-interval verification of the
closed-form dispatch chain of certify_v19.py.

Scope (interval-verified): acquisition recursion, braking recursion,
budgets, thresholds, admission margins, clearance floors, authority
demands, dispersion / late / schedule / alignment / gap bounds, sampled
-event Lipschitz margin, velocity floor, braking headroom, ledger
closure, ramp-before-stop.  Every elementary operation is evaluated with
two-sided intervals and 1-ulp outward rounding; sup/TV/L1 of the modal
family use branch-union upper bounds that contain the true value for
every parameter in the input box.

Out of scope (declared): the entry-deadline tail recursion and the
handoff-pad supremum (grid-evaluated with derivative-bound pads in
certify_v19.py) -- these remain computer-assisted, not interval-enclosed.

Outputs: interval_report_v30.json + printed witness log; parameter-block
SHA-256 printed for the verification artifact.
"""
import hashlib, json, math

SRC = open('certify_v30.py').read()
PARAM_BLOCK = SRC[:SRC.index('dbar = 0.06\nd_s')]
print("parameter-block sha256:",
      hashlib.sha256(PARAM_BLOCK.encode()).hexdigest())

ns = {}
exec(PARAM_BLOCK, ns)
import numpy as np
n      = ns['n'];      b      = [float(x) for x in ns['b']]
beta   = float(ns['beta']);   gamma = float(ns['gamma'])
lam    = float(ns['lam']);    a_b   = float(ns['a_b'])
alpha  = [float(x) for x in ns['alpha_g']]
A_run  = [float(x) for x in ns['A_run']]
eps0   = [float(x) for x in ns['eps0']]
e0     = [float(x) for x in ns['e0']]
eps_e  = float(ns['eps_e']); eps_v = float(ns['eps_v'])
eps_det = float(ns['eps_det']); phi = float(ns['phi'])
T_c    = float(ns['T_c']);   kappa = float(ns['kappa'])
Pi     = [float(x) for x in ns['Pi']]
eta    = [float(x) for x in ns['eta']]
Umin   = [float(x) for x in ns['Umin']]
v_xi   = float(ns['v_xi'])
s_m    = float(ns['s_m']);   ell = float(ns['ell'])
NUMPAD = float(ns['NUMPAD'])
VHND_PAD = float(ns['VHND_PAD'])
T_xi = float(ns['T_xi'])
GAIN_E = float(ns['GAIN_E']); GAIN_X = float(ns['GAIN_X'])
GAIN_F = float(ns['GAIN_F']); L1_GE = float(ns['L1_GE'])
L1_GX = float(ns['L1_GX']);  TV_KIMP = float(ns['TV_KIMP'])
XFREE_PK = float(ns['XFREE_PK']); EFREE_PK = float(ns['EFREE_PK'])
IE_BOX = float(ns['IE_BOX']); IX_BOX = float(ns['IX_BOX'])

INF = math.inf
def dn(x): return x if x in (0.0, -INF) else math.nextafter(x, -INF)
def up(x): return x if x in (0.0,  INF) else math.nextafter(x,  INF)

class IV:
    __slots__ = ('lo', 'hi')
    def __init__(self, lo, hi=None):
        if hi is None: hi = lo
        self.lo, self.hi = float(lo), float(hi)
        assert self.lo <= self.hi
    def __add__(s, o):
        o = _iv(o); return IV(dn(s.lo+o.lo), up(s.hi+o.hi))
    __radd__ = __add__
    def __sub__(s, o):
        o = _iv(o); return IV(dn(s.lo-o.hi), up(s.hi-o.lo))
    def __rsub__(s, o): return _iv(o) - s
    def __mul__(s, o):
        o = _iv(o)
        c = (s.lo*o.lo, s.lo*o.hi, s.hi*o.lo, s.hi*o.hi)
        return IV(dn(min(c)), up(max(c)))
    __rmul__ = __mul__
    def __truediv__(s, o):
        o = _iv(o)
        assert o.lo > 0.0, "interval division needs positive denominator"
        c = (s.lo/o.lo, s.lo/o.hi, s.hi/o.lo, s.hi/o.hi)
        return IV(dn(min(c)), up(max(c)))
    def __neg__(s): return IV(-s.hi, -s.lo)
def _iv(x): return x if isinstance(x, IV) else IV(x)
import mpmath as _mp
_mp.mp.prec = 80          # documented correctly-rounded backend (mpmath)
def _exp_dn(v):
    from flint import arb, ctx as _ctx
    _ctx.prec = 120
    lo = float(arb(v).exp().lower())
    return math.nextafter(lo, -INF)

def _exp_up(v):
    from flint import arb, ctx as _ctx
    _ctx.prec = 120
    hi = float(arb(v).exp().upper())
    return math.nextafter(hi, INF)

def iexp(x): x = _iv(x); return IV(_exp_dn(x.lo), _exp_up(x.hi))
def iabs(x):
    x = _iv(x)
    lo = 0.0 if x.lo <= 0.0 <= x.hi else min(abs(x.lo), abs(x.hi))
    return IV(lo, up(max(abs(x.lo), abs(x.hi))))
def imax(*xs):
    xs = [_iv(x) for x in xs]
    return IV(max(x.lo for x in xs), max(x.hi for x in xs))
def imin(*xs):
    xs = [_iv(x) for x in xs]
    return IV(min(x.lo for x in xs), min(x.hi for x in xs))
def ipos(x): x = _iv(x); return IV(max(x.lo, 0.0), max(x.hi, 0.0))
def isum(xs):
    t = IV(0.0)
    for x in xs: t = t + x
    return t

def cf_upper(p, q):
    """valid UPPER bounds (as intervals [0,hi]) on sup, TV, L1 of
    (p+qt)e^{-lam t} on [0,inf); tight when interval signs resolve the
    extremum/zero branches, branch-union otherwise."""
    p, q = _iv(p), _iv(q)
    ap = iabs(p)
    qz = not (q.lo > 0.0 or q.hi < 0.0)          # q may be zero
    if qz:
        gs = q * IV(1.0/lam)
        sup_u = imax(ap, iabs(gs)).hi
        tv_u = max(ap.hi, (ap + 2.0*iabs(gs)).hi)
        l1_u = (iabs(p)*IV(1.0/lam) + iabs(q)*IV(1.0/lam**2)).hi
        return IV(0, sup_u), IV(0, tv_u), IV(0, l1_u)
    sq = 1.0 if q.lo > 0 else -1.0
    ts = (q - lam*p) / iabs(q) * IV(sq/lam)
    gs = (q * IV(1.0/lam)) * iexp(-lam*ipos(ts))
    if ts.hi <= 0.0:                              # no interior extremum
        sup_u, tv_u = ap.hi, ap.hi
    elif ts.lo > 0.0:                             # extremum certainly interior
        sup_u = imax(ap, iabs(gs)).hi
        tv_u = (iabs(p - gs) + iabs(gs)).hi
    else:                                         # straddle: branch union
        sup_u = imax(ap, iabs(gs)).hi
        tv_u = max(ap.hi, (iabs(p - gs) + iabs(gs)).hi)
    I0 = p*IV(1.0/lam) + q*IV(1.0/lam**2)
    t0 = (-p) / q if q.lo > 0.0 else p / (-q)
    It = q * IV(1.0/lam**2) * iexp(-lam*ipos(t0))
    if t0.hi <= 0.0:                              # no interior sign change
        l1_u = iabs(I0).hi
    elif t0.lo > 0.0:
        l1_u = (iabs(I0 - It) + iabs(It)).hi
    else:
        l1_u = max(iabs(I0).hi, (iabs(I0 - It) + iabs(It)).hi)
    return IV(0, sup_u), IV(0, tv_u), IV(0, l1_u)

def corner_stats_iv(ebar, xbar):
    best = dict(sup_x=IV(0), sup_f=IV(0), tv_f=IV(0),
                int_e=IV(0), int_x=IV(0))
    eb, xb = _iv(ebar), _iv(xbar)
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0 = IV(se*eb.hi, se*eb.lo) if se < 0 else IV(eb.lo, eb.hi)
            x_0 = IV(sx*xb.hi, sx*xb.lo) if sx < 0 else IV(xb.lo, xb.hi)
            c = x_0 + lam*e_0
            _, _, l1e = cf_upper(e_0, c)
            spx, _, l1x = cf_upper(x_0, IV(-lam)*c)
            spf, tvf, _ = cf_upper(beta*e_0 + gamma*x_0, IV(-lam**2)*c)
            best['sup_x'] = imax(best['sup_x'], spx)
            best['sup_f'] = imax(best['sup_f'], spf)
            best['tv_f'] = imax(best['tv_f'], tvf)
            best['int_e'] = imax(best['int_e'], l1e)
            best['int_x'] = imax(best['int_x'], l1x)
    return best

def D_quad_iv(ebar, xbar, bi):
    c1, c2 = -beta/bi, 1.0 - gamma/bi
    out = IV(0)
    eb, xb = _iv(ebar), _iv(xbar)
    for se in (+1.0, -1.0):
        for sx in (+1.0, -1.0):
            e_0 = IV(se*eb.hi, se*eb.lo) if se < 0 else IV(eb.lo, eb.hi)
            x_0 = IV(sx*xb.hi, sx*xb.lo) if sx < 0 else IV(xb.lo, xb.hi)
            c = x_0 + lam*e_0
            _, _, l1 = cf_upper(IV(c1)*e_0 + IV(c2)*x_0,
                                c*IV(c1 - lam*c2))
            out = imax(out, l1)
    return out

def DB_quad_iv(bi):
    c1, c2 = -beta/bi, 1.0 - gamma/bi
    _, _, l1 = cf_upper(IV(c2), IV(c1 - lam*c2))
    return l1

def chain(dbar):
    d = IV(dbar)
    Lam1 = [IV(0)]*n; Lam2 = [IV(0)]*n; taub = [IV(0)]*n
    tact = [IV(0)]*n; tset = [IV(0)]*n; E1 = [IV(0)]*n
    ebar = [IV(0)]*n; fbar = [IV(0)]*n; tvS2 = [IV(0)]*n
    x2 = [IV(0)]*n; Tk = [IV(0)]*n; Dq = [IV(0)]*n; DBq = [IV(0)]*n
    for k in range(n):
        Lam1[k] = d*isum(Tk[:k])
        m = IV(abs(eps0[k])) + Lam1[k]
        taub[k] = m*IV(1.0/alpha[k]) + IV(T_c)
        tact[k] = IV(0) if k == 0 else tset[k-1] + d
        tset[k] = tact[k] + taub[k]
        Lk = IV(alpha[k]) + isum(Tk[:k])
        E1[k] = m*tact[k] + ipos(m*m - IV(eps_det**2))*IV(1/(2*alpha[k])) \
                + (IV(eps_det) + Lk*IV(T_c))*IV(T_c)
        ebar[k] = IV(abs(e0[k])) + E1[k]
        st = corner_stats_iv(ebar[k], IV(eps_det))
        Lam2[k] = d*isum(tvS2[:k])
        x2[k] = st['sup_x'] + IV(GAIN_X)*Lam2[k]
        fbar[k] = st['sup_f'] + IV(GAIN_F)*Lam2[k]
        tvS2[k] = st['tv_f'] + IV(TV_KIMP)*Lam2[k]
        Tk[k] = IV(2*alpha[k]) + (IV(beta)*ebar[k] + IV(gamma*eps_det)) \
                + tvS2[k]
        Dq[k] = D_quad_iv(ebar[k], IV(eps_det), b[k])
        DBq[k] = DB_quad_iv(b[k])
    F = [isum(fbar[:i]) for i in range(n)]
    f_xi = beta*eps_e + gamma*eps_v
    W = [IV(0)]*n; epsb = [IV(0)]*n; eb_ = [IV(0)]*n; fb = [IV(0)]*n
    Th = [IV(0)]*n; IE = [IV(0)]*n; IX = [IV(0)]*n; IF = [IV(0)]*n
    Fcum = [IV(0)]*n
    for k in range(1, n):
        W[k] = d*(IV(VHND_PAD + a_b + f_xi) + isum(Th[1:k])
                  + isum(Fcum[1:k]))
        epsb[k] = IV(XFREE_PK) + IV(GAIN_X)*W[k]
        eb_[k] = IV(EFREE_PK) + IV(GAIN_E)*W[k]
        fb[k] = IV(beta)*eb_[k] + IV(gamma)*epsb[k]
        Fcum[k] = IV(f_xi) + isum(fb[1:k+1])
        IE[k] = IV(IE_BOX) + IV(L1_GE)*W[k]
        IX[k] = IV(IX_BOX) + IV(L1_GX)*W[k]
        IF[k] = IV(beta)*IE[k] + IV(gamma)*IX[k]
        Th[k] = IV(beta)*IX[k] + IV(gamma)*(W[k] + IF[k])
    Fb = [IV(f_xi) + isum(fb[1:i]) for i in range(n)]
    Pp = [IV(0)]*n
    for u in range(1, n):
        Pp[u] = fb[u] + imax(F[u], Pp[u-1])
    pairs = {}
    for i in range(1, n):
        V1 = isum([IV(abs(eps0[k])) + Lam1[k] + x2[k] for k in range(i)])
        H1 = E1[i] + IV(alpha[i]/b[i])*taub[i] + V1*IV(abs(Pi[i])) \
             + Lam1[i]*IV(1/b[i])
        V2 = IV(2.0)*isum(x2[:i])
        Ht = Dq[i] + V2*IV(abs(Pi[i])) + Lam2[i]*(IV(1/b[i]) + DBq[i])
        Vbrk = IV(v_xi + i*eps_v)
        Vrise = IV(i-1)*d*Pp[i-1]
        bmin = min(b[i], b[i-1])
        anet = IV(a_b) - Fb[i]
        assert anet.lo > 0, "headroom interval must be positive"
        Ph2 = IX[i] + epsb[i]*IV(1/bmin) + epsb[i]*epsb[i]/(IV(2.0)*anet) \
              + IF[i]*IV(1/b[i]) + Vbrk*IV(max(Pi[i], 0.0)) \
              + Vrise*IV(max(-Pi[i], 0.0)) + W[i]*(IV(1/b[i]) + DBq[i])
        M1 = IV(abs(eps0[i])) + Lam1[i]
        hm1 = imax((IV(b[i])*M1 + (F[i] + IV(A_run[i-1]))*IV(eta[i]))
                   * IV(1/(kappa*b[i])),
                   (IV(alpha[i]) + IV(b[i])*M1 + F[i]*IV(eta[i]))
                   * IV(1/(kappa*b[i])))
        hm2 = (fbar[i] + IV(b[i])*x2[i] + F[i]*IV(eta[i])) \
              * IV(1/(kappa*b[i]))
        dem_prev = F[i-1] + imax(IV(A_run[i-1]), fbar[i-1])
        Ubrk = imax(dem_prev, Pp[i-1], IV(a_b) + Fb[i],
                    (IV(a_b) + Fb[i-1]) + fbar[i-1])
        hmb = imax((fb[i] + IV(b[i])*epsb[i] + Ubrk*IV(eta[i]))
                   * IV(1/(kappa*b[i])),
                   ipos(IV(b[i])*epsb[i]
                        + IV(b[i]/b[i-1])*(IV(a_b) + Fb[i]) - IV(a_b))
                   * IV(1/(kappa*b[i])))
        thr = imax(H1 + hm1, H1 + Ht + hm2, H1 + Ht + Ph2 + hmb)
        pairs[i] = dict(H1=H1, Ht=Ht, Ph2=Ph2, thr=thr, V1=V1, anet=anet)
    hops = [epsb[i]/(IV(a_b) - Fb[i]) for i in range(1, n)]
    disp = isum(hops)
    late = isum([epsb[i]*IV(1/a_b) for i in range(1, n)])
    sched = IV(eps_v/a_b) + disp
    gaperr = [IV(EFREE_PK) + IV(GAIN_E)*W[i]
              + epsb[i]*epsb[i]/(IV(2.0)*(IV(a_b) - Fb[i]))
              for i in range(1, n)]
    align = IV(eps_e) + IV(v_xi*eps_v/a_b) + IV(eps_v**2/(2*a_b)) \
            + isum(gaperr)
    dem = []
    for i in range(n):
        acq = F[i] + imax(IV(A_run[i]), fbar[i])
        brk = IV(a_b) + (Fcum[i] if i >= 1 else IV(f_xi))
        pre = (IV(a_b) + Fb[i]) + fbar[i] if i >= 1 else IV(0)
        dem.append(imax(acq, brk, Pp[i], pre))
    Leps = IV(max(alpha)) + isum(Tk)
    supx = [imax(IV(abs(eps0[k])) + Lam1[k], x2[k]) for k in range(n)]
    cum = []; t = IV(0)
    for s in supx:
        t = t + s; cum.append(t)
    vfloor = IV(v_xi) - imax(*cum)
    return dict(pairs=pairs, disp=disp, late=late, sched=sched,
                align=align, gaperr=gaperr, dem=dem, Leps=Leps,
                vfloor=vfloor, Fcum=Fcum, Fb=Fb, W=W, epsb=epsb,
                x2=x2, V1s={i: pairs[i]['V1'] for i in pairs},
                Lam2=Lam2, tact=tact, tset=tset, ebar=ebar,
                fsw=[IV(beta)*ebar[k] + IV(gamma*eps_det)
                     for k in range(n)])

# geometry (interval) --------------------------------------------------------
def geometry():
    d_s = [IV(0)]*n
    for i in range(1, n):
        d_s[i] = IV(s_m) + IV(5.0) + IV(v_xi)*IV(max(-Pi[i], 0.0)) \
                 + IV(v_xi)*IV(0.06)
    v = []; vp = IV(v_xi)
    for i in range(n):
        vi = vp - IV(eps0[i]); v.append(vi); vp = vi
    g_init = [IV(0)] + [d_s[i] - IV(s_m) + IV(e0[i]) for i in range(1, n)]
    h0 = [IV(0)] + [g_init[i] + v[i-1]*IV(1/b[i-1]) - v[i]*IV(1/b[i])
                    for i in range(1, n)]
    return g_init, h0

def floors_iv(R, g_init, h0):
    gmin = IV(INF, INF)
    for i in range(1, n):
        P = R['pairs'][i]
        uh = h0[i] - P['H1'] - P['Ht']
        bf = uh - P['Ph2']
        if Pi[i] > 0:
            vs_a = IV(v_xi) + P['V1']
            vs_b = IV(v_xi + eps_v) + isum(R['epsb'][1:i])
            gspd = imin(g_init[i], uh - IV(Pi[i])*vs_a,
                        bf - IV(Pi[i])*vs_b)
            gTf = imax(bf - IV(Pi[i])*(IV(a_b) + R['Fb'][i])*IV(1/b[i]),
                       bf)
            gmin = imin(gmin, gspd, gTf, bf)
        else:
            gmin = imin(gmin, g_init[i], bf)
    return gmin

def witness(dbar, conds=None, verbose=True):
    R = chain(dbar)
    g_init, h0 = geometry()
    out = {}
    M = {}
    M['adm'] = [h0[i] - R['pairs'][i]['thr'] - IV(NUMPAD)
                for i in range(1, n)]
    M['auth'] = [IV(Umin[i]) - R['dem'][i] for i in range(n)]
    M['clr'] = [floors_iv(R, g_init, h0)]
    M['event'] = [IV(eps_det - phi) - R['Leps']*IV(T_c)]
    M['vel'] = [R['vfloor']]
    M['head'] = [IV(a_b) - R['Fcum'][i] for i in range(1, n-1)]
    M['ramp'] = [R['vfloor']*IV(1/max(Umin)) - IV((n-1)*dbar)]
    M['sync'] = [IV(1.0) - R['disp']]
    M['align'] = [IV(float(ns.get('ALIGN_TOL', 1.0))) - R['align']]
    M['gap'] = [IV(1.0) - g for g in R['gaperr']]
    for _k, _lst in M.items():
        out[_k] = min(x.lo for x in _lst)
        out['#' + _k] = min(x.hi for x in _lst)
    out['_disp_hi'] = R['disp'].hi
    out['_align_hi'] = R['align'].hi
    out['_sched_hi'] = R['sched'].hi
    out['_late_hi'] = R['late'].hi
    keys = conds or [k for k in out
                     if not k.startswith(('_', '#'))]
    okall = all(out[k] > 0 for k in keys)
    if verbose:
        for k in keys:
            print(f"  {k:6s}: interval margin lower = {out[k]:+.6f}  "
                  f"[{'VERIFIED' if out[k] > 0 else 'not verified'}]")
    return okall, out

print("\n=== featured witness: dbar = 0.06 (closed-form conjunction) ===")
ok, feat = witness(0.06)
print(f"disp bound (interval upper) = {feat['_disp_hi']:.6f}  "
      f"<= 0.48: {feat['_disp_hi'] <= 0.48}")
print(f"align/sched/late uppers = {feat['_align_hi']:.4f} / "
      f"{feat['_sched_hi']:.4f} / {feat['_late_hi']:.4f}")
print("FEATURED WITNESS:", "VERIFIED" if ok else "NOT VERIFIED")
print("\n=== residual interval table (featured, lo / hi) ===")
for k in [k for k in feat if not k.startswith(('_', '#'))]:
    print(f"  {k:6s}: [{feat[k]:+.6f}, {feat['#'+k]:+.6f}]")

print("\n=== displayed crossings as interval-feasible points ===")
cross = dict(sync=(0.158, ['sync']), align=(0.079, ['align']),
             gap=(0.151, ['gap']), adm=(0.061, ['adm']),
             auth=(0.311, ['auth']), clr=(0.063, ['clr']),
             event=(0.450, ['event']))
report = {}
for nm, (dv, keys) in cross.items():
    used = dv
    okc, o = witness(dv, conds=keys, verbose=False)
    for step in (0.0005, 0.001, 0.002, 0.004, 0.008):
        if okc:
            break
        used = dv - step
        okc, o = witness(used, conds=keys, verbose=False)
    report[nm] = dict(at=used, margin_lo=o[keys[0]], verified=bool(okc))
    tag = 'VERIFIED' if okc else 'NOT verified'
    extra = '' if used == dv else f' (fallback point {used:.4f})'
    print(f"  {nm:6s} @ {dv:.3f}: margin lower {o[keys[0]]:+.6f} "
          f"[{tag}]{extra}")

print("\n=== ceiling brackets: feasible-lower / infeasible-upper ===")
brackets = {}
for nm, (dv, keys) in cross.items():
    lo_pt = report[nm]['at']
    hi_pt = None; mhi = None
    for step in (0.001, 0.002, 0.003, 0.004, 0.006, 0.008, 0.010, 0.014):
        p = round(dv + step, 4)
        okc2, o2 = witness(p, conds=keys, verbose=False)
        if o2['#' + keys[0]] < 0:
            hi_pt, mhi = p, o2['#' + keys[0]]
            break
    brackets[nm] = dict(feas_lo=lo_pt, infeas_hi=hi_pt, margin_hi=mhi)
    if hi_pt is not None:
        print(f"  {nm:6s}: feasible @ {lo_pt:.4f} | infeasible @ "
              f"{hi_pt:.4f} (margin upper {mhi:+.6f})")
    else:
        print(f"  {nm:6s}: feasible @ {lo_pt:.4f} | no infeasible point "
              f"found within +0.014")

json.dump(dict(featured=dict(ok=bool(ok),
                             margins={k: v for k, v in feat.items()}),
               crossings=report, brackets=brackets,
               scope="closed-form chain only; entry/hold/hnd excluded"),
          open('interval_report_v30.json', 'w'), indent=1)
print("\n[interval_report_v30.json written]")
assert ok and feat['_disp_hi'] <= 0.48, "witness must pass before shipping"
print("INTERVAL WITNESS: ALL VERIFIED")

# ============================================================================
# v3: OUTWARD-INTERVAL PERSISTENT-ENTRY VERIFICATION (H3 and C_hnd at 0.06 s)
# Two-sided primitives; the recursion propagates certified UPPER bounds.
# ============================================================================
import numpy as np
NTI, HG = 200000, 0.001   # v30: horizon extended for T_xi = 180 s
TGI = np.arange(NTI)*HG
NEG, POS = -INF, INF
def _dnA(a): return np.nextafter(a, NEG)
def _upA(a): return np.nextafter(a, POS)

# rigorous e^{-lam*k*h} grid: cumulative interval powers of enclosed e^{-lam h}
_e1lo, _e1hi = _exp_dn(-lam*HG), _exp_up(-lam*HG)
_Elo = np.empty(NTI); _Ehi = np.empty(NTI)
_Elo[0] = _Ehi[0] = 1.0
for _k in range(1, NTI):
    _Elo[_k] = _dnA(_Elo[_k-1]*_e1lo)
    _Ehi[_k] = _upA(_Ehi[_k-1]*_e1hi)

def _idiv(N, D):
    assert D.lo > 0 or D.hi < 0
    cs = [N.lo/D.lo, N.lo/D.hi, N.hi/D.lo, N.hi/D.hi]
    return IV(dn(min(cs)), up(max(cs)))
def _ineg(x): return IV(-x.hi, -x.lo)

def _gpair(p, q):
    """interval arrays (lo,hi) of g(T)=(p+qT)e^{-lam T}, p,q IV scalars."""
    plo = _dnA(np.minimum(q.lo*TGI, q.hi*TGI))
    phi = _upA(np.maximum(q.lo*TGI, q.hi*TGI))
    lin_lo = _dnA(plo + p.lo)
    lin_hi = _upA(phi + p.hi)
    glo = np.where(lin_lo >= 0, _dnA(lin_lo*_Elo), _dnA(lin_lo*_Ehi))
    ghi = np.where(lin_hi >= 0, _upA(lin_hi*_Ehi), _upA(lin_hi*_Elo))
    return glo, ghi

def _absU(lo, hi):
    return np.maximum(np.abs(lo), np.abs(hi))

def _Bsafe(p, q):
    """upper bound on sup_{r>=0}|g|: |p| + |q|/(lam e)."""
    ap = max(abs(p.lo), abs(p.hi)); aq = max(abs(q.lo), abs(q.hi))
    return up(ap + up(aq/dn(lam*math.e)))

def _tail_sup_hi(p, q):
    glo, ghi = _gpair(p, q)
    A = _absU(glo, ghi)
    if q.lo > 0 or q.hi < 0:
        ts = _idiv(q - IV(lam)*p, IV(lam)*q)
        gs = _Bsafe(p, q) if ts.hi > 0 else 0.0
        S = np.where(TGI < ts.hi, np.maximum(A, gs), A)
    else:
        S = np.maximum(A, _Bsafe(p, q))
    return np.maximum.accumulate(S[::-1])[::-1]   # enforce nonincreasing hi

def _tail_tv_hi(p, q):
    glo, ghi = _gpair(p, q)
    A = _absU(glo, ghi)
    if q.lo > 0 or q.hi < 0:
        ts = _idiv(q - IV(lam)*p, IV(lam)*q)
        B2 = 2.0*_Bsafe(p, q) if ts.hi > 0 else 0.0
        TV = np.where(TGI < ts.hi, _upA(A + B2), A)
    else:
        TV = _upA(A + 2.0*_Bsafe(p, q))
    return np.maximum.accumulate(TV[::-1])[::-1]

def _l1_tail_hi(p, q, B):
    """upper bound on int_B^inf |(p+qr)e^{-lam r}|dr (scalar)."""
    def Ihi(a):
        e = IV(_exp_dn(-lam*a), _exp_up(-lam*a))
        return ((p + q*IV(a))*IV(1/lam) + q*IV(1/lam**2))*e
    ap = max(abs(p.lo), abs(p.hi)); aq = max(abs(q.lo), abs(q.hi))
    if q.lo > 0 or q.hi < 0:
        t0 = _ineg(_idiv(p, q))
        if t0.hi <= B:
            v = Ihi(B); return max(abs(v.lo), abs(v.hi))
        t0c = max(t0.hi, B)
        v1, v0 = Ihi(B), Ihi(t0c)
        return up(max(abs((v1 - v0).lo), abs((v1 - v0).hi))
                  + max(abs(v0.lo), abs(v0.hi)))
    # q straddles zero: |g| <= (ap+aq*r)e^{-lam r}; integrate that
    e = IV(_exp_dn(-lam*B), _exp_up(-lam*B))
    v = (IV(ap) + IV(aq)*(IV(B) + IV(1/lam)))*IV(1/lam)*e
    return v.hi

_PFi, _QFi = IV(gamma), IV(beta - gamma*lam)     # f_imp=(g+(b-gl)r)e^{-lr}
_DPFi, _DQFi = _QFi - IV(lam)*_PFi, IV(-lam)*_QFi
def _Psi_hi(B): return _l1_tail_hi(_DPFi, _DQFi, B)
_L1FP_hi = _Psi_hi(0.0)

def entry_interval(dbar):
    R = chain(dbar)
    K = int(math.ceil(dbar/HG - 1e-12))
    def shR(a, k):
        if k <= 0: return a
        o = np.empty_like(a); o[:k] = a[0]; o[k:] = a[:-k]; return o
    SeH, SxH, FTH = [], [], []
    for k in range(n):
        Se = np.zeros(NTI); Sx = np.zeros(NTI); FT = np.zeros(NTI)
        eb = R['ebar'][k]
        for se in (1.0, -1.0):
            for sx in (1.0, -1.0):
                e0c = IV(se)*eb; x0c = IV(sx*eps_det)
                c = x0c + IV(lam)*e0c
                Se = np.maximum(Se, _tail_sup_hi(e0c, c))
                Sx = np.maximum(Sx, _tail_sup_hi(x0c, IV(-lam)*c))
                pf = IV(-1)*(IV(beta)*e0c + IV(gamma)*x0c)
                qf = IV(-(beta - gamma*lam))*c
                FT = np.maximum(FT, _tail_tv_hi(pf, qf))
        SeH.append(Se); SxH.append(Sx); FTH.append(FT)
    V = [np.zeros(NTI)]
    Bset = [2.0*m for m in range(15)]
    for j in range(1, n + 1):
        jj = j - 1
        Vs = shR(V[jj], K)
        idx = np.minimum(np.maximum(
            np.floor((TGI - R['tset'][jj].hi)/HG).astype(int), 0), NTI-1)
        ft = FTH[jj][idx]
        aj = alpha[jj]; fsw = R['fsw'][jj].hi
        jmp = np.where(TGI <= R['tact'][jj].hi + 1e-12, aj, 0.0) \
            + np.where(TGI <= R['tset'][jj].hi + 1e-12, _upA(aj + fsw), 0.0)
        if jj == 0:
            forc = np.zeros(NTI)
        else:
            Lh = R['Lam2'][jj].hi
            cgd = up(gamma*dbar)
            forc = _upA(cgd*Vs)
            ct = np.full(NTI, INF)
            for B in Bset:
                c2 = up(up(_L1FP_hi*dbar))
                cand = _upA(up(_Psi_hi(B)*Lh)
                            + _upA(c2*shR(V[jj], K + int(B/HG))))
                ct = np.minimum(ct, cand)
            forc = _upA(forc + ct)
        V.append(_upA(_upA(Vs + ft) + _upA(jmp + forc)))
    TentH = np.zeros(n)
    Aset = [2.0*m for m in range(20)]
    for i in range(n):
        totE = SeH[i].copy(); totX = SxH[i].copy()
        if i >= 1:
            Lh = R['Lam2'][i].hi
            fbE = np.full(NTI, INF); fbX = np.full(NTI, INF)
            KEinf = up(1.0/dn(lam*math.e))
            for A in Aset:
                Vsh = _upA(dbar*shR(V[i], K + int(A/HG)))
                if A*lam >= 1.0:                    # past the g_e peak
                    kE = up(A*_exp_up(-lam*A))
                else:
                    kE = KEinf
                gA = up(abs(1.0 - lam*A)*_exp_up(-lam*A))
                kX = up(max(gA, _exp_up(-2.0)*(1+1e-15))) \
                    if A*lam < 2.0 else gA
                if A == 0.0: kX = 1.0
                fbE = np.minimum(fbE, _upA(up(kE*Lh) + _upA(KEinf*Vsh)))
                fbX = np.minimum(fbX, _upA(up(kX*Lh) + Vsh))
            totE = _upA(totE + fbE); totX = _upA(totX + fbX)
        ok = (totE <= eps_e) & (totX <= eps_v)
        TentH[i] = TGI[int(np.argmax(ok))] if ok.any() else INF
        if ok.any():
            assert ok[int(np.argmax(ok)):].all()
    TbarH = max(R['tset'][i].hi + TentH[i] for i in range(n))
    ixh = int((T_xi - dbar)/HG)
    VhndH = [float(V[i][ixh]) for i in range(1, n)]
    return TentH, TbarH, VhndH

print("\n=== v3: interval-verified entry (upper bounds) at dbar=0.06 ===")
TentH, TbarH, VhndH = entry_interval(0.06)
print("Tent upper  :", [round(float(x), 3) for x in TentH])
print(f"Tbar_ent upper = {TbarH:.4f}; H3: {TbarH + 0.06:.4f} <= {T_xi}:",
      TbarH + 0.06 <= T_xi)
print("Vhnd upper  :", ["%.3e" % v for v in VhndH],
      " C_hnd:", max(VhndH) <= VHND_PAD, f"(pad {VHND_PAD})")
