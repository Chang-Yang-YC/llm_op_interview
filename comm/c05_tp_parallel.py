"""
题目 c05: 张量并行 —— Column/Row Parallel Linear
====================================================
面试题:
    实现 Megatron 风格的 column-parallel 和 row-parallel Linear，
    并用它们组成一个 TP 版 MLP:
        x -> [ColumnParallel(gate/up)] -> SwiGLU -> [RowParallel(down)] -> y
    证明整个 MLP 前向只需 1 次 all-reduce。

考点:
    1. Column parallel: 按输出维切 W (Hout/n, Hin)，
       y_local = x @ W_local^T，无通信；输出天然按列切分。
    2. Row parallel: 按输入维切 W (Hout, Hin/n)，
       输入也按列切分，y = allreduce(x_local @ W_local^T)。
    3. 组合技巧: Column 的分片输出直接喂给 Row 的分片输入，
       中间的 SwiGLU 是逐元素操作可以分片做 => 一个 MLP 块
       前向 1 次 all-reduce、反向 1 次，这就是 Megatron 的核心设计。
    4. 追问:
       - TP=8 时 allreduce 走 NVLink；TP 跨机为什么性能崩。
       - 序列并行 (SP) 把这 1 次 allreduce 拆成 rs + ag，
         目的是让 LN/Dropout 的激活也按序列维切。
       - TP 与 EP/CP 组合时的通信拓扑问题。

运行: torchrun --nproc_per_node=2 comm/c05_tp_parallel.py [--solution]
（nproc 取 2 或 4 均可）
"""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F

HINT = "run: torchrun --nproc_per_node=2 comm/c05_tp_parallel.py"


def my_col_parallel_linear(x: torch.Tensor, w_local: torch.Tensor) -> torch.Tensor:
    """w_local: 本卡的列分片 (Hout/n, Hin)。无通信。"""
    # TODO
    raise NotImplementedError


def my_row_parallel_linear(x_local: torch.Tensor, w_local: torch.Tensor,
                           group=None) -> torch.Tensor:
    """x_local: (..., Hin/n), w_local: (Hout, Hin/n)。需要 all-reduce。"""
    # TODO
    raise NotImplementedError


def ref_col_parallel_linear(x, w_local):
    return x @ w_local.t()


def ref_row_parallel_linear(x_local, w_local, group=None):
    y = x_local @ w_local.t()
    dist.all_reduce(y, group=group)
    return y


def _shard(w, n, r, dim):
    return w.tensor_split(n, dim=dim)[r].contiguous()


def check(use_mine: bool = True):
    if not (dist.is_available() and dist.is_initialized()):
        return None, HINT
    n = dist.get_world_size()
    r = dist.get_rank()
    dev = torch.device("cuda", r % max(1, torch.cuda.device_count()))
    torch.manual_seed(3)
    B, T, H, Hffn = 2, 16, 256, 512
    x = torch.randn(B, T, H, device=dev)
    w_gate = torch.randn(Hffn, H, device=dev)
    w_up = torch.randn(Hffn, H, device=dev)
    w_down = torch.randn(H, Hffn, device=dev)

    # 参考: 单卡完整 MLP
    ref_y = F.linear(F.silu(x @ w_gate.t()) * (x @ w_up.t()), w_down)

    col = my_col_parallel_linear if use_mine else ref_col_parallel_linear
    row = my_row_parallel_linear if use_mine else ref_row_parallel_linear

    g_local = col(x, _shard(w_gate, n, r, 0))
    u_local = col(x, _shard(w_up, n, r, 0))
    act_local = F.silu(g_local) * u_local              # 分片逐元素，无通信
    y = row(act_local, _shard(w_down, n, r, 1))        # 唯一一次 all-reduce
    dist.barrier()
    err = (y - ref_y).abs().max().item()
    rel = err / ref_y.abs().max().item()  # 输出量级大，必须用相对误差
    return rel < 1e-5, ("TP=%d, MLP with 1 allreduce, rel err %.3e" % (n, rel))


def _main():
    import sys
    use_mine = "--solution" not in sys.argv
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    ok, msg = check(use_mine=use_mine)
    if dist.get_rank() == 0:
        print(("PASS" if ok else "FAIL"), msg)
    dist.destroy_process_group()


if __name__ == "__main__":
    _main()
