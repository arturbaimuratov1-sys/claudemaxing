#!/bin/bash
# One measured Kaggle T4 run of the checked-out commit of this repository (cm/) with the
# challenge kit's harness: tests_local, run_task.py --repeats 1 --profile, check_rules.py.
# Notebook cell: cd /kaggle/working && rm -rf cm && git clone -q <repo> cm &&
#                git -C cm checkout -q <COMMIT> && bash cm/kaggle_run.sh
set -e
cd /kaggle/working
git -C cm log -1 --format='commit %H %s'
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv
python -m pip install -q "soup-cli[train]==0.77.0" "transformers==5.16.1" "peft==0.20.0" \
    "trl==0.29.1" "bitsandbytes==0.50.2" "accelerate==1.14.0" pytest 2>&1 | tail -2
rm -rf kit && git clone -q https://github.com/MakazhanAlpamys/soup-kernel-challenge kit
rm -rf kit/solution kit/tests_local && cp -r cm/solution cm/tests_local kit/
cd kit
python task/prepare_data.py 2>&1 | tail -1
CUDA_VISIBLE_DEVICES=0 python -m pytest tests_local -q -p no:cacheprovider 2>&1 | tail -2
python harness/run_task.py --repeats 1 --profile 2>&1 | tail -25
python harness/check_rules.py 2>&1 | tail -16
python - <<'EOF'
import glob, json
for f in sorted(glob.glob("runs/*/metrics.json")):
    m = json.load(open(f))
    print(f, {k: v for k, v in m.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
    own = m.get("own_kernel_ms") or {}
    if own:
        loop = m.get("timed_loop_s") or 0
        tot = sum(own.values())
        swiglu = sum(v for k, v in own.items() if k.startswith("silu_mul"))
        print("  own kernels ms:", {k: round(v, 1) for k, v in sorted(own.items(), key=lambda kv: -kv[1])})
        if loop:
            print(f"  share {100 * tot / 1000 / loop:.2f}%  without SwiGLU {100 * (tot - swiglu) / 1000 / loop:.2f}%")
try:
    p = json.load(open("runs/solution_profile/profile.json"))
    print("profile total GPU s", round(p["total_gpu_us"] / 1e6, 2))
    for row in p["kernels"][:12]:
        print("  ", {k: row[k] for k in row if k in ("name", "time_us", "count")})
except Exception as e:  # noqa: BLE001
    print("no profile:", e)
EOF
