"""Smallest |w| of every RMSNorm weight of the cached Qwen2.5-1.5B-Instruct (CPU only; not part
of the solution)."""
import glob
import os

from safetensors import safe_open

files = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/*/*.safetensors"))
mins = []
for f in files:
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.endswith("norm.weight"):
                w = s.get_tensor(k).float()
                mins.append((w.abs().min().item(), k))
mins.sort()
print(len(mins), "norms; smallest |w|:", mins[:5])
print("below 2^-10:", sum(m < 2 ** -10 for m, _ in mins))
counts = []
for f in files:
    with safe_open(f, "pt") as s:
        for k in s.keys():
            if k.endswith("norm.weight"):
                counts.append(int((s.get_tensor(k).float().abs() < 2 ** -10).sum()))
print("tiny-|w| columns per norm: max", max(counts), "total", sum(counts), "norms over 64:",
      sum(c > 64 for c in counts))
