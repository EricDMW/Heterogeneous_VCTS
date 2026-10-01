#!/usr/bin/env python3
"""run_caseD_runtime.py -- run time of the dispatch evaluator of the planned
arrival, with a recorded environment.

Measured, each in a fresh Python process so that no cache of an earlier
evaluation is reused:
  forward pass   one evaluation of the certificate at the operating age
                 bound (plan in exact rational arithmetic, forward pass,
                 entry recursion, envelopes, conditions), five repetitions;
  search         the search of the largest certified age bound on the
                 millisecond grid, 1 ms to 450 ms (bisection, the two
                 neighbours of the boundary evaluated), one repetition;
  interval       the interval enclosure of the closed-form conditions at the
                 operating age bound (interval_plan.check_enclosure), one
                 repetition.
The processor model, the operating system, the library versions, the load
of the machine before and after the timing, and the CPU time beside the wall
time are recorded.

Usage:  python run_caseD_runtime.py
Writes: data/new_d19/caseD/runtime.json
"""
import json
import os
import platform
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(CODE, 'data', 'new_d19', 'caseD')
OUT = os.path.join(DATA, 'runtime.json')

CHILD = r'''
import json, os, sys, time
CODE = %r
sys.path.insert(0, os.path.join(CODE, 'src', 'certification'))
t_imp = time.perf_counter()
import certify_plan as cq
import interval_plan as ip
t_imp = time.perf_counter() - t_imp
design = json.load(open(os.path.join(CODE, 'data', 'new_d19', 'caseD', 'design.json')))
cfg = design['cfg']
mode = sys.argv[1]
t0 = time.perf_counter(); c0 = time.process_time()
if mode == 'pass':
    r = cq.evaluate(cfg)
    out = dict(admitted=r['admitted'], T_f_bar=r['consts']['T_f_bar'])
elif mode == 'search':
    memo = {}
    b = cq.age_boundary_ms(cfg, kmax=450, memo=memo)
    out = dict(boundary_ms=b['k_ms'], failed_next=b['failed_next'], evaluations=len(memo))
elif mode == 'interval':
    e = ip.check_enclosure(cfg)
    out = dict(verified=e['verified'], all_inside=e['all_inside'])
out.update(wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0, import_s=t_imp)
print(json.dumps(out))
''' % CODE


def _ps(cmd):
    try:
        r = subprocess.run(['powershell.exe', '-NoProfile', '-Command', cmd],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip()
    except Exception as ex:               # noqa: BLE001
        return 'unavailable: %s' % ex


def machine():
    cpu = _ps("Get-CimInstance Win32_Processor | Select-Object Name,"
              "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed | "
              "ConvertTo-Json -Compress")
    osx = _ps("Get-CimInstance Win32_OperatingSystem | Select-Object "
              "Caption,Version,BuildNumber,TotalVisibleMemorySize | "
              "ConvertTo-Json -Compress")
    out = {}
    for k, v in (('cpu', cpu), ('os', osx)):
        try:
            out[k] = json.loads(v)
        except Exception:                 # noqa: BLE001
            out[k] = v
    out.update(platform=platform.platform(), machine=platform.machine(),
               logical_cpus_os=os.cpu_count())
    return out


def versions():
    import numpy as np
    out = dict(python=platform.python_version(), numpy=np.__version__)
    for mod in ('scipy', 'flint', 'numba'):
        try:
            out['python-flint' if mod == 'flint' else mod] = __import__(mod).__version__
        except Exception:                 # noqa: BLE001
            out[mod] = None
    return out


def load(tag):
    snap = dict(tag=tag, time=time.strftime('%Y-%m-%d %H:%M:%S'))
    try:
        import psutil
        snap['cpu_percent_1s'] = psutil.cpu_percent(interval=1.0)
        me = os.getpid()
        others = []
        for p in psutil.process_iter(['pid', 'name']):
            nm = (p.info.get('name') or '').lower()
            if p.info['pid'] != me and any(s in nm for s in ('python', 'pdflatex')):
                try:
                    c = p.cpu_percent(interval=0.2)
                except Exception:         # noqa: BLE001
                    c = None
                others.append(dict(pid=p.info['pid'], name=nm, cpu_percent=c))
        snap['other_python_or_latex_processes'] = others
    except Exception as ex:               # noqa: BLE001
        snap['psutil'] = 'unavailable: %s' % ex
    return snap


def child(mode):
    r = subprocess.run([sys.executable, '-c', CHILD, mode], capture_output=True,
                       text=True, cwd=CODE)
    if r.returncode != 0:
        raise SystemExit('child %s failed:\n%s' % (mode, r.stderr[-2000:]))
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    doc = dict(item='run time of the dispatch evaluator of the planned arrival',
               generated=time.strftime('%Y-%m-%d %H:%M:%S'),
               script='code/run_caseD_runtime.py', machine=machine(),
               versions=versions(), load_before=load('before'))
    design = json.load(open(os.path.join(DATA, 'design.json')))
    doc['design_revision'] = design.get('revision')
    passes = [child('pass') for _ in range(5)]
    assert all(p['admitted'] for p in passes)
    assert all(abs(p['T_f_bar'] - design['certificate']['consts']['T_f_bar']) < 1e-12
               for p in passes)
    walls = sorted(p['wall_s'] for p in passes)
    doc['forward_pass'] = dict(runs=passes, wall_median_s=walls[2], wall_min_s=walls[0],
                               wall_max_s=walls[-1],
                               cpu_over_wall=[p['cpu_s'] / p['wall_s'] for p in passes])
    s = child('search')
    assert s['boundary_ms'] == design['boundary_ms']['k_ms']
    doc['search'] = s
    doc['interval'] = child('interval')
    assert doc['interval']['verified']
    doc['load_after'] = load('after')
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1)
    print('forward pass: median %.2f s (min %.2f, max %.2f), cpu/wall %s'
          % (walls[2], walls[0], walls[-1],
             ['%.2f' % x for x in doc['forward_pass']['cpu_over_wall']]))
    print('search: %.1f s, %d evaluations, boundary %d ms'
          % (s['wall_s'], s['evaluations'], s['boundary_ms']))
    print('interval enclosure: %.1f s' % doc['interval']['wall_s'])
    print('load before %s %%, after %s %%' % (doc['load_before'].get('cpu_percent_1s'),
                                             doc['load_after'].get('cpu_percent_1s')))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
