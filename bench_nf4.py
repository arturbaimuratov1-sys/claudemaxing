"""NF4 dequantization speed: ours against bitsandbytes on this GPU (not part of the solution)."""
import sys
import time

import torch

sys.path.insert(0, ".")
import bitsandbytes as bnb  # noqa: E402

from solution import nf4  # noqa: E402

for n, k in ((8960, 1536), (1536, 8960), (1536, 1536), (256, 1536)):
    w = (torch.randn(n, k, device="cuda") * 0.05).half()
    packed, state = bnb.functional.quantize_4bit(w, blocksize=64, quant_type="nf4",
                                                 compress_statistics=True)
    res = {}
    for name, f in (("bnb", lambda: bnb.functional.dequantize_4bit(packed, state)),
                    ("ours", lambda: nf4.dequantize_nf4(packed, state))):
        f()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(50):
            f()
        torch.cuda.synchronize()
        res[name] = (time.perf_counter() - t) / 50 * 1e6
    print(f"{n}x{k}: bnb {res['bnb']:.1f} us, ours {res['ours']:.1f} us")
