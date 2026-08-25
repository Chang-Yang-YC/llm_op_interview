#!/usr/bin/env python3
"""
One-shot comm checker: init process group once, run all comm problems.

Usage (from the project root):
    torchrun --nproc_per_node=4 comm/run_comm.py             # check my_*
    torchrun --nproc_per_node=4 comm/run_comm.py --solution  # check refs

All terminal output is pure ASCII.
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

MODS = [
    "comm.c01_ring_allreduce",
    "comm.c02_gather_scatter",
    "comm.c03_all_to_all",
    "comm.c04_busbw",
    "comm.c05_tp_parallel",
    "comm.c06_overlap",
]


def main():
    use_mine = "--solution" not in sys.argv
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    n = dist.get_world_size()
    if dist.get_rank() == 0:
        print("=" * 60)
        print("Comm problems, world size %d, mode %s"
              % (n, "my_*" if use_mine else "references"))
        print("=" * 60)
    for name in MODS:
        mod = importlib.import_module(name)
        try:
            ok, msg = mod.check(use_mine=use_mine)
            if ok is None:
                status = "SKIP"
            else:
                status = "PASS" if ok else "FAIL"
        except NotImplementedError:
            status, msg = "TODO", "my_* not implemented yet"
        except Exception as e:
            status, msg = "ERROR", "%s: %s" % (type(e).__name__, e)
        if dist.get_rank() == 0:
            print("%-7s %-22s %s" % (status, name.split(".")[-1], msg),
                  flush=True)
        dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
