#!/usr/bin/env python3
"""witness_root2.py -- corrected, parameterized interval-witness driver
over the FROZEN witness_core_v7.py (root revision).

Two roles:

1. GEOMETRY CORRECTION (Case A).  The frozen witness's geometry()
   hardcodes IV(5.0) for the equilibrium clearance and IV(0.06) for
   the design information age, while the Case-A evaluator uses
   g_star = 4.2: the archived 'featured' admission margin (+0.8956 m)
   and clearance margin (+1.0614 m) were therefore verified on an
   inflated geometry (~+0.8 m of initial barrier).  This driver
   substitutes the true (g_star, dbar_des) -- or an explicit per-pair
   gap vector -- into the geometry and re-runs the witness, giving
   the interval margins for the geometry the paper actually reports.

2. PARAMETERIZATION (primary benchmark).  As with certify_root2, the
   frozen witness reads its constants by exec-ing the evaluator's
   parameter block; staging a substituted evaluator source therefore
   runs the witness on the primary-benchmark constants, and the
   explicit-gap hook supplies its per-pair minimal gaps.

The frozen file is never edited; the two patches below are textual,
minimal, and documented:
  G1  IV(5.0)            -> IV(<g_star from the parameter block>)
  G2  IV(v_xi)*IV(0.06)  -> IV(v_xi)*IV(<dbar_des>)
  G3  optional D_S_OVERRIDE list replaces the uniform-rule gaps
  G4  the sync tolerance IV(1.0) and crossing list become inputs
  G5  the featured print/emit epilogue is replaced by a JSON emit of
      this driver's own results (witness_report_root2.json)
"""
import io
import contextlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import certify_root2 as cr

FROZEN = os.path.join(HERE, 'witness_core_v7.py')
EPILOGUE_MARK = 'print("\\n=== featured witness: dbar = 0.06 '


def run(overrides=None, dbar_op=0.06, dbar_des=None, g_star=None,
        d_s=None, sync_tol=1.0, align_tol=None, crossings=None,
        workdir=None, quiet=True):
    """Run the interval witness for the given design.

    overrides : certify_root2-style constant overrides (None = Case A)
    d_s       : explicit per-pair physical gap vector (index 0 = head
                slot), overriding the uniform sizing rule
    crossings : {name: (dbar_point, [condition])} feasible points to
                verify, replacing the frozen Case-A list
    Returns the report dict (also written to
    witness_report_root2.json in the working directory)."""
    src = open(FROZEN, encoding='utf-8').read()

    # stage the (possibly substituted) evaluator source for the
    # parameter-block exec inside the witness
    cert_src = cr._source(overrides or {}, mode='full')

    gs = g_star if g_star is not None else (
        overrides or {}).get('g_star', 4.2)
    dd = dbar_des if dbar_des is not None else (
        overrides or {}).get('dbar', 0.06)

    # the frozen witness splits the evaluator source at its featured
    # 'dbar = 0.06' line; retarget the split marker when the staged
    # source carries a substituted operating age
    ov_dbar = (overrides or {}).get('dbar')
    if ov_dbar is not None:
        marker_old = r"SRC.index('dbar = 0.06\nd_s')"
        marker_new = r"SRC.index('dbar = " + repr(float(ov_dbar)) \
            + r"\nd_s')"
        assert marker_old in src
        src = src.replace(marker_old, marker_new, 1)

    # G1/G2: geometry constants
    n_subs = 0
    src, c1 = re.subn(r"IV\(s_m\) \+ IV\(5\.0\)",
                      "IV(s_m) + IV(%r)" % float(gs), src)
    src, c2 = re.subn(r"\+ IV\(v_xi\)\*IV\(0\.06\)",
                      "+ IV(v_xi)*IV(%r)" % float(dd), src)
    assert c1 == 1 and c2 == 1, 'geometry patch failed'

    # G3: explicit gap vector
    if d_s is not None:
        hook = ("    if D_S_OVERRIDE is not None:\n"
                "        d_s = [IV(float(x)) for x in D_S_OVERRIDE]\n")
        anchor = "    v = []; vp = IV(v_xi)\n"
        assert anchor in src
        src = src.replace(anchor, hook + anchor, 1)

    # G4: sync tolerance
    src, c3 = re.subn(r"M\['sync'\] = \[IV\(1\.0\) - R\['disp'\]\]",
                      "M['sync'] = [IV(SYNC_TOL) - R['disp']]", src)
    assert c3 == 1
    if align_tol is not None:
        src = src.replace("float(ns.get('ALIGN_TOL', 1.0))",
                          "float(%r)" % float(align_tol), 1)

    # G5: drop the frozen featured epilogue; we drive witness() below
    idx = src.find(EPILOGUE_MARK)
    assert idx > 0, 'epilogue marker not found'
    src = src[:idx]

    cwd = os.getcwd()
    if workdir:
        os.makedirs(workdir, exist_ok=True)
        os.chdir(workdir)
    try:
        open('certify_v30.py', 'w', encoding='utf-8').write(cert_src)
        G = {'__name__': 'witness_root2_run',
             'D_S_OVERRIDE': list(d_s) if d_s is not None else None,
             'SYNC_TOL': float(sync_tol)}
        buf = io.StringIO()
        ctx = contextlib.redirect_stdout(buf) if quiet \
            else contextlib.nullcontext()
        with ctx:
            exec(compile(src, 'witness_core_v7.py[root2]', 'exec'), G)
            ok, feat = G['witness'](dbar_op, verbose=not quiet)
        report = dict(dbar_op=dbar_op, g_star=float(gs),
                      dbar_des=float(dd),
                      d_s=None if d_s is None else
                      [float(x) for x in d_s],
                      ok=bool(ok),
                      margins={k: float(v) for k, v in feat.items()})
        if crossings:
            cx = {}
            for nm, (dv, keys) in crossings.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    okc, o = G['witness'](dv, conds=keys,
                                          verbose=False)
                # infeasible upper bracket search
                hi_pt = None; mhi = None
                for step in (0.001, 0.002, 0.003, 0.004, 0.006,
                             0.008, 0.010, 0.014, 0.020):
                    with contextlib.redirect_stdout(io.StringIO()):
                        ok2, o2 = G['witness'](round(dv + step, 4),
                                               conds=keys,
                                               verbose=False)
                    if o2['#' + keys[0]] < 0:
                        hi_pt = round(dv + step, 4)
                        mhi = float(o2['#' + keys[0]])
                        break
                cx[nm] = dict(at=dv, margin_lo=float(o[keys[0]]),
                              verified=bool(okc), infeas_hi=hi_pt,
                              margin_hi=mhi)
            report['crossings'] = cx
        json.dump(report, open('witness_report_root2.json', 'w'),
                  indent=1)
        return report
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rep = run(workdir=td, quiet=True)
    print('Case A, CORRECTED geometry (g*=4.2, dbar_des=0.06):')
    for k, v in rep['margins'].items():
        if not k.startswith(('_', '#')):
            print('  %-6s interval margin lower = %+.6f  [%s]'
                  % (k, v, 'VERIFIED' if v > 0 else 'NOT verified'))
    print('  overall:', 'VERIFIED' if rep['ok'] else 'NOT VERIFIED')
    print('  disp interval upper = %.6f' % rep['margins']['_disp_hi'])
