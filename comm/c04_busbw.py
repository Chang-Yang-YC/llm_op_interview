"""
题目 c04: algbw / busbw 计算器（无需分布式环境）
===================================================
面试题（口算/白板高频）:
    8 卡做 1GB 的 allreduce，nccl-tests 报 time=10ms，
    求 algbw 和 busbw。若 NVLink 单向理论带宽 450GB/s，
    这个效率是多少？

考点:
    1. algbw = size / time —— 应用层视角。
    2. busbw = algbw * 修正系数 —— 硬件链路实际承载的流量:
       allreduce:          2*(n-1)/n     （reduce-scatter + allgather）
       allgather /
       reduce-scatter /
       all-to-all:         (n-1)/n
       broadcast / reduce: (n-1)/n （按 NCCL 定义）
    3. 为什么要 busbw: 不同集合通信算法的"有效吞吐"只有在
       换算到链路流量后才可比，也才能和硬件峰值对比。
    4. 追问: nccl-tests 的 out-of-place 与 in-place 差别；
       为什么小包看延迟（latency bound）大包看带宽；
       双工链路上 ring 为什么能同时用满收发。

上面例题答案: algbw=100GB/s, busbw=100*14/8=175GB/s,
效率 175/450 ~ 39%（提示: 跨机就远低于此，NVLink 单机应 >80%）。
"""
import torch  # noqa: F401  (仅为保持与其他题目一致的 import 风格)


def my_algbw(size_bytes: int, time_sec: float) -> float:
    # TODO
    raise NotImplementedError


def my_busbw(size_bytes: int, n_ranks: int, time_sec: float,
             op: str = "allreduce") -> float:
    """op in {allreduce, allgather, reduce_scatter, alltoall}."""
    # TODO: 先算 algbw 再乘正确的修正系数
    raise NotImplementedError


def ref_algbw(size_bytes, time_sec):
    return size_bytes / time_sec


def ref_busbw(size_bytes, n_ranks, time_sec, op="allreduce"):
    factors = {
        "allreduce": 2.0 * (n_ranks - 1) / n_ranks,
        "allgather": (n_ranks - 1) / n_ranks,
        "reduce_scatter": (n_ranks - 1) / n_ranks,
        "alltoall": (n_ranks - 1) / n_ranks,
    }
    return ref_algbw(size_bytes, time_sec) * factors[op]


def check(use_mine: bool = True):
    GB, MS = 1e9, 1e-3
    cases = [
        # (size, n, t, op, expected_busbw)
        (1 * GB, 8, 10 * MS, "allreduce", 175 * GB),       # 175 GB/s
        (1 * GB, 8, 10 * MS, "allgather", 87.5 * GB),
        (2 * GB, 4, 25 * MS, "reduce_scatter", 60 * GB),
    ]
    errs = []
    for size, n, t, op, want in cases:
        got = (my_busbw(size, n, t, op) if use_mine
               else ref_busbw(size, n, t, op))
        errs.append(abs(got - want) / want)
    a_got = my_algbw(1 * GB, 10 * MS) if use_mine else ref_algbw(1 * GB, 10 * MS)
    errs.append(abs(a_got - 100 * GB) / (100 * GB))
    err = max(errs)
    return err < 1e-6, "max rel err %.3e (busbw factors correct?)" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
