#!/usr/bin/env python3
"""sim_expm_nominal.py -- independent exact propagation of the nominal
arrival system Sigma^N (filter removed) for the audit cross-check of the
canonical simulator sim_cont.py (R2-B02, 2026-09-25).

Shares no code with sim_cont.py.  Between two consecutive message
instants t_k = k T_m every mode law of Sec. III-A is affine in the
augmented state

    X = (s_0, v_0, s_1..s_n, v_1..v_n, uhat_1..uhat_n, 1)

(S1 with the saturation engaged is u = uhat + alpha sgn(eps); S2 is
uhat + beta e + gamma eps; RULE and BRAKE are -a_b; LATCH freezes the
unit), and all discrete events except zero-speed instants lie on the
T_m grid when T_c = T_m and the transport delay is a multiple of T_m.
The state is therefore propagated exactly by X(t_k + tau) =
expm(M tau) X(t_k); zero-speed instants inside an interval are found by
Brent's method on tau -> v_i(t_k + tau).  Event priorities, send and
delivery conventions are those of the main text (LATCH > RULE > packet
batch > detector sample; RULE on exact predecessor standstill; LATCH at
the first zero speed at or after the S2 switch; head BRAKE at T_xi).

The S1 saturation is assumed engaged (|eps_i| >= phi on every S1
interval, which condition (H7) guarantees); the run reports any grid
instant at which it is not (n_s1_unsaturated).  The margins and
clearances are evaluated at the grid instants (both one-sided limits).
"""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

COAST, S1, S2, RULE, LATCH, BRAKE = range(6)


def run_nominal(P, d_s, v0, e0, transport, T_m=1e-3, T_END=None):
    n = int(P['n'])
    ell, s_m, a_b = P['ell'], P['s_m'], P['a_b']
    v_xi, T_xi = P['v_xi'], P['T_xi']
    alpha, beta, gamma, phi = P['alpha'], P['beta'], P['gamma'], P['phi']
    eps_det, kappa = P['eps_det'], P['kappa']
    amax = np.array(P['amax'], float)
    b = amax / P['v_c']
    T_END = P['T_END'] if T_END is None else T_END
    Ntr = int(round(transport / T_m))
    N = int(round(T_END / T_m))
    k_xi = int(round(T_xi / T_m))
    tau_r = T_xi + v_xi / a_b
    d_s = np.asarray(d_s, float)

    DIM = 3 + 3 * n
    S0, V0 = 0, 1
    iS = [2 + i for i in range(n)]
    iV = [2 + n + i for i in range(n)]
    iU = [2 + 2 * n + i for i in range(n)]
    iC = 2 + 3 * n

    X = np.zeros(DIM)
    X[S0] = 0.0
    X[V0] = v_xi
    X[iS] = -np.cumsum(ell + d_s + np.asarray(e0, float))
    X[iV] = np.asarray(v0, float)
    X[iC] = 1.0

    mode = [S1] + [COAST] * (n - 1)
    kact = [0] + [-1] * (n - 1)
    flag = [False] * n
    sig = [0.0] * n
    sent = np.zeros((n, N + 1))
    fsent = np.zeros((n, N + 1), bool)
    tstop = np.full(n, np.nan)
    tact = np.full(n, np.nan)
    tact[0] = 0.0
    tset = np.full(n, np.nan)
    trule = np.full(n, np.nan)
    gmin = np.full(n - 1, np.inf)
    mmin = np.full(n - 1, np.inf)
    upk = np.zeros(n)
    e_hnd = np.full(n, np.nan)
    eps_hnd = np.full(n, np.nan)
    n_unsat = 0
    cache = {}          # mode configuration -> generator M (kept alive)
    ecache = {}         # id(M) -> expm(M T_m)

    def matrix(md, sg):
        key = (tuple(md), tuple(sg))
        M = cache.get(key)
        if M is not None:
            return M
        M = np.zeros((DIM, DIM))
        M[S0, V0] = 1.0                       # reference: v_0 constant before T_xi
        for i in range(n):
            if md[i] == LATCH:
                continue
            M[iS[i], iV[i]] = 1.0
            r = iV[i]
            if md[i] in (RULE, BRAKE):
                M[r, iC] = -a_b
                continue
            if i >= 1:
                M[r, iU[i]] = 1.0             # feedforward of the received command
            ps, pv = (S0, V0) if i == 0 else (iS[i - 1], iV[i - 1])
            if md[i] == S1:
                M[r, iC] += alpha * sg[i]
            elif md[i] == S2:
                M[r, ps] += beta
                M[r, iS[i]] -= beta
                M[r, iC] -= beta * (ell + d_s[i])
                M[r, pv] += gamma
                M[r, iV[i]] -= gamma
        cache[key] = M
        return M

    def eps_of(Z, i):
        return (Z[V0] if i == 0 else Z[iV[i - 1]]) - Z[iV[i]]

    def margins(Z, M, md):
        u = M @ Z
        out = []
        for i in range(1, n):
            if md[i] == LATCH:
                continue
            g = Z[iS[i - 1]] - Z[iS[i]] - ell - s_m
            hb = g + Z[iV[i - 1]] / b[i - 1] - Z[iV[i]] / b[i]
            ucbf = b[i] * (eps_of(Z, i) + Z[iU[i]] / b[i - 1] + kappa * hb)
            out.append((i, ucbf - u[iV[i]]))
        return out

    M_prev = None
    for k in range(N):
        t = k * T_m
        if k == k_xi:
            for i in range(n):
                ps = X[S0] if i == 0 else X[iS[i - 1]]
                e_hnd[i] = ps - X[iS[i]] - ell - d_s[i]
                eps_hnd[i] = eps_of(X, i)
        for i in range(1, n):
            gmin[i - 1] = min(gmin[i - 1], X[iS[i - 1]] - X[iS[i]] - ell - s_m)
        # left limits of the margins at this instant
        if M_prev is not None:
            for i, m in margins(X, M_prev, mode):
                mmin[i - 1] = min(mmin[i - 1], m)
        # events in priority order
        for i in range(n):
            if mode[i] in (S2, RULE, BRAKE) and X[iV[i]] <= 0.0:
                mode[i] = LATCH
                tstop[i] = t
                X[iV[i]] = 0.0
            elif mode[i] == S2 and i >= 1 and mode[i - 1] == LATCH:
                mode[i] = RULE
                trule[i] = t
            if i == 0 and k == k_xi and mode[0] != LATCH:
                mode[0] = BRAKE
            if i >= 1 and k >= Ntr:
                X[iU[i]] = sent[i - 1, k - Ntr]
                flag[i] = flag[i] or bool(fsent[i - 1, k - Ntr])
                if mode[i] == COAST and flag[i]:
                    mode[i] = S1
                    kact[i] = k
                    tact[i] = t
            if mode[i] == S1 and abs(eps_of(X, i)) <= eps_det:
                mode[i] = S2
                tset[i] = t
            if mode[i] == S1:
                ep = eps_of(X, i)
                if abs(ep) < phi:
                    n_unsat += 1
                sig[i] = 1.0 if ep > 0 else -1.0
        M = matrix(mode, sig)
        u = M @ X
        for i in range(n):
            ui = 0.0 if mode[i] == LATCH else u[iV[i]]
            upk[i] = max(upk[i], abs(ui))
            sent[i, k] = ui
            fsent[i, k] = mode[i] >= S2
        for i, m in margins(X, M, mode):
            mmin[i - 1] = min(mmin[i - 1], m)
        # exact propagation over [t_k, t_{k+1}] with localized zero speeds
        tc, rem = t, T_m
        while rem > 0.0:
            Mc = matrix(mode, sig)
            if rem == T_m:
                E = ecache.get(id(Mc))
                if E is None:
                    E = expm(Mc * T_m)
                    ecache[id(Mc)] = E
                Xn = E @ X
            else:
                Xn = expm(Mc * rem) @ X
            cand = []
            for i in range(n):
                if mode[i] in (S2, RULE, BRAKE) and X[iV[i]] > 0.0 >= Xn[iV[i]]:
                    f = (lambda tau, i=i, Mc=Mc: (expm(Mc * tau) @ X)[iV[i]])
                    cand.append((brentq(f, 0.0, rem, xtol=1e-15, rtol=1e-15,
                                        maxiter=200), i))
            if not cand:
                X = Xn
                break
            tau, j = min(cand)
            X = expm(Mc * tau) @ X
            tc += tau
            rem -= tau
            X[iV[j]] = 0.0
            mode[j] = LATCH
            tstop[j] = tc
            if j + 1 < n and mode[j + 1] == S2:
                mode[j + 1] = RULE
                trule[j + 1] = tc
        M_prev = matrix(mode, sig)
        if t > tau_r + 2 * T_m and all(md == LATCH for md in mode):
            break
    s = X[iS].copy()
    mark_ref = v_xi * T_xi + v_xi ** 2 / (2 * a_b)
    marks = mark_ref - np.cumsum(ell + d_s)
    egap = (np.concatenate(([mark_ref], s[:-1])) - s - ell) - d_s
    return dict(tstop=tstop, tact=tact, tset=tset, trule=trule, s=s,
                marker_signed=s - marks, egap=egap,
                disp=float(np.nanmax(tstop) - np.nanmin(tstop)),
                align=float(np.max(np.abs(s - marks))),
                gaperr=float(np.max(np.abs(egap))),
                gmin_grid=gmin, mmin_grid=mmin, upk_grid=upk, e_hnd=e_hnd,
                eps_hnd=eps_hnd, n_s1_unsaturated=n_unsat,
                n_mode_configurations=len(cache))
