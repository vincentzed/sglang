"""Prove the green-ctx arm really co-resides: trace one green_a16_b132 region
and report how much of side A's kernel time overlaps side B's kernel time."""

import json
import sys

import torch

sys.path.insert(0, "scratch/m4_green_ctx_overlap")
from bench_step1 import Workload

from sglang.srt.multiplex.green_ctx import split_device_green_ctx_by_sm_count

torch.cuda.set_device(0)
wl = Workload(16384, 81920, 0)
default = torch.cuda.current_stream()
(ga, gb), res = split_device_green_ctx_by_sm_count(0, [16, 132])
print("granted:", [r.sm.smCount for r in res])


def green():
    ga.wait_stream(default)
    gb.wait_stream(default)
    with torch.cuda.stream(ga):
        wl.side_a()
    with torch.cuda.stream(gb):
        wl.side_b()
    default.wait_stream(ga)
    default.wait_stream(gb)


for _ in range(3):
    green()
torch.cuda.synchronize()

with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
    green()
    torch.cuda.synchronize()

trace_path = "scratch/m4_green_ctx_overlap/green_a16.trace.json"
prof.export_chrome_trace(trace_path)

events = json.load(open(trace_path))["traceEvents"]
kernels = [e for e in events if e.get("cat") == "kernel"]
streams = {}
for e in kernels:
    streams.setdefault(e["args"]["stream"], []).append((e["ts"], e["ts"] + e["dur"]))
print(
    {
        s: (len(v), f"{sum(b - a for a, b in v) / 1e3:.3f} ms")
        for s, v in streams.items()
    }
)

sids = sorted(streams, key=lambda s: sum(b - a for a, b in streams[s]))
a_sid, b_sid = sids[-2], sids[-1]  # side A: smaller busy time; side B: larger


def overlap(iv_a, iv_b):
    tot = 0.0
    for a0, a1 in iv_a:
        for b0, b1 in iv_b:
            tot += max(0.0, min(a1, b1) - max(a0, b0))
    return tot


a_busy = sum(b - a for a, b in streams[a_sid])
ov = overlap(streams[a_sid], streams[b_sid])
print(
    f"side A stream {a_sid}: busy {a_busy/1e3:.3f} ms; "
    f"overlapped with side B stream {b_sid}: {ov/1e3:.3f} ms ({100*ov/a_busy:.1f}%)"
)
