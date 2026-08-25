#!/usr/bin/env python3
"""
One-click checker for the LLM op interview bank.

Usage:
    python run_all.py             # check your implementations (my_* functions)
    python run_all.py --solution  # sanity-check the bundled reference solutions
    python run_all.py --quick     # skip triton problems

Comm problems need torch.distributed, they show up as SKIP here.
Run them with torchrun, e.g.:

    torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py
    torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py --solution

All terminal output of this script is pure ASCII.
"""
import importlib
import sys

COMPUTE = [
    ("compute.c01_softmax", "stable softmax"),
    ("compute.c02_norm", "layernorm + rmsnorm"),
    ("compute.c03_rope", "rotary position embedding"),
    ("compute.c04_attention", "multi-head attention, causal + GQA"),
    ("compute.c05_kv_cache", "attention with kv cache"),
    ("compute.c06_swiglu", "swiglu feed-forward"),
    ("compute.c07_cross_entropy", "stable cross entropy"),
    ("compute.c08_sampling", "top-k / top-p sampling"),
    ("compute.c09_quant", "int8 / int4-group quantize"),
    ("compute.c10_autograd", "custom autograd.Function"),
    ("compute.c11_triton_softmax", "[triton] fused softmax"),
    ("compute.c12_triton_flash_attn", "[triton] flash attention"),
]

COMM = [
    ("comm.c01_ring_allreduce", "ring all-reduce via send/recv"),
    ("comm.c02_gather_scatter", "ring all-gather + reduce-scatter"),
    ("comm.c03_all_to_all", "all-to-all, EP dispatch"),
    ("comm.c04_busbw", "algbw / busbw calculator, no dist needed"),
    ("comm.c05_tp_parallel", "column/row parallel linear, TP"),
    ("comm.c06_overlap", "async comm + compute overlap"),
]


def run_one(name, use_mine):
    try:
        mod = importlib.import_module(name)
    except Exception as e:
        return "ERROR", "import failed: %s: %s" % (type(e).__name__, e)
    try:
        ok, msg = mod.check(use_mine=use_mine)
    except NotImplementedError:
        return "TODO", "my_* not implemented yet"
    except Exception as e:
        return "ERROR", "%s: %s" % (type(e).__name__, e)
    if ok is None:
        return "SKIP", msg
    return ("PASS" if ok else "FAIL"), msg


def main():
    use_mine = "--solution" not in sys.argv
    mode = "your implementations" if use_mine else "bundled reference solutions"
    print("=" * 72)
    print("LLM Op Interview Bank  [%s]" % mode)
    print("=" * 72)
    counts = {"PASS": 0, "FAIL": 0, "TODO": 0, "SKIP": 0, "ERROR": 0}
    for title, table in (("compute", COMPUTE), ("comm", COMM)):
        print("--- %s ---" % title)
        for name, desc in table:
            status, msg = run_one(name, use_mine)
            counts[status] += 1
            print("%-7s %-36s %s" % (status, name.split(".")[-1] + ":", msg))
        print()
    print("Summary: %d PASS / %d FAIL / %d TODO / %d SKIP / %d ERROR" % (
        counts["PASS"], counts["FAIL"], counts["TODO"], counts["SKIP"],
        counts["ERROR"]))
    print("Comm problems need torchrun, see README.md for exact commands.")
    return 0 if counts["FAIL"] == 0 and counts["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
