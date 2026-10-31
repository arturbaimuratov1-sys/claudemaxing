"""Short report of a Kaggle run in kit/runs (run from the kit folder): metrics of every run,
own kernel times and shares, the profiler's top kernels, the checks. Lines are kept short."""

import glob
import json
import textwrap


def out(text):
    print(textwrap.fill(str(text), 72, subsequent_indent="    "))


for f in sorted(glob.glob("runs/*/metrics.json")):
    m = json.load(open(f))
    out(f"{f}: " + ", ".join(f"{k}={v}" for k, v in m.items()
                             if isinstance(v, (int, float)) and not isinstance(v, bool)))
    own = m.get("own_kernel_ms") or {}
    if own:
        loop = m.get("timed_loop_s") or 0
        tot = sum(own.values())
        swiglu = sum(v for k, v in own.items() if k.startswith("silu_mul"))
        out("own ms: " + ", ".join(f"{k} {v:.0f}" for k, v in sorted(own.items(),
                                                                     key=lambda kv: -kv[1])))
        if loop:
            out(f"share {100 * tot / 1000 / loop:.2f}%, without SwiGLU "
                f"{100 * (tot - swiglu) / 1000 / loop:.2f}%")
try:
    p = json.load(open("runs/solution_profile/profile.json"))
    out(f"profile: total GPU {p['total_gpu_us'] / 1e6:.2f} s over {p.get('active_steps')} steps")
    for row in p["kernels"][:25]:
        name = str(row.get("name", ""))[:48]
        out(f"  {row.get('time_us', 0) / 1e3:9.1f} ms  x{row.get('count', '')}  {name}")
except Exception as e:  # noqa: BLE001
    out(f"no profile: {e}")
try:
    checks = json.load(open("runs/checks.json"))
    items = checks.get("checks", checks) if isinstance(checks, dict) else checks
    for c in items if isinstance(items, list) else [items]:
        out(json.dumps(c, ensure_ascii=False)[:600])
except Exception as e:  # noqa: BLE001
    out(f"no checks: {e}")
