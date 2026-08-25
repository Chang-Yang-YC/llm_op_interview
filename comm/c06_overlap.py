"""
题目 c06: 通信与计算 overlap（async_op + 独立 stream）
=========================================================
面试题:
    给定输入 x 和权重 w，实现:
      1) 发起 x 的异步 all-reduce（不阻塞）；
      2) 与此同时计算 z = x_local @ w（用 x 的本地副本，不依赖 allreduce 结果）；
      3) 等待 all-reduce 完成，返回 (x_reduced, z)。

考点:
    1. async_op=True 只是把 kernel 入队到 NCCL 自己的 stream，
       CPU 立即返回 handle；真正并行靠的是两条 CUDA stream。
       追问: 为什么 NCCL 不直接用默认 stream（防止阻塞其他 op +
       独立优先级）。
    2. DDP 的 overlap: 反向梯度按 bucket 分块，每个 bucket 就绪
       立刻发起 allreduce，与前面层的反向计算并行。
       追问: bucket 大小怎么权衡（太小=>launch/延迟占比高，
       太大=>overlap 窗口小）。
    3. 验证 overlap 是否真实发生: nsys 里看 NCCL kernel 与
       GEMM kernel 在不同 stream 上时间交叠（这是排障基本功）。
    4. 追问: TP 的 allreduce 为什么难 overlap（在关键路径上）；
       通信分解成小块流水（如 ring 化 TP、PipeSwitch 思路）。

运行: torchrun --nproc_per_node=2 comm/c06_overlap.py [--solution]
"""
import os
import torch
import torch.distributed as dist

HINT = "run: torchrun --nproc_per_node=2 comm/c06_overlap.py"


def my_overlap_allreduce_matmul(x: torch.Tensor, w: torch.Tensor):
    """x: (M, K) 每卡不同; w: (K, N) 每卡相同。
    返回 (x_reduced, z)，z = x @ w 用未归约的本地 x 计算。"""
    # TODO: 用 dist.all_reduce(..., async_op=True)
    raise NotImplementedError


def ref_overlap_allreduce_matmul(x, w):
    x_buf = x.clone()
    handle = dist.all_reduce(x_buf, async_op=True)
    z = x @ w
    handle.wait()
    return x_buf, z


def check(use_mine: bool = True):
    if not (dist.is_available() and dist.is_initialized()):
        return None, HINT
    n = dist.get_world_size()
    r = dist.get_rank()
    dev = torch.device("cuda", r % max(1, torch.cuda.device_count()))
    torch.manual_seed(5)
    M, K, N = 1024, 1024, 1024
    x = torch.randn(M, K, device=dev) + r
    w = torch.randn(K, N, device=dev)  # 各卡相同（同 seed，未加 r）

    # ground truth
    x_ref = x.clone()
    dist.all_reduce(x_ref)
    z_ref = x @ w

    fn = my_overlap_allreduce_matmul if use_mine else ref_overlap_allreduce_matmul
    x_out, z_out = fn(x, w)
    torch.cuda.synchronize()
    dist.barrier()
    err_x = (x_out - x_ref).abs().max().item()
    err_z = (z_out - z_ref).abs().max().item()
    ok = err_x < 1e-4 and err_z < 1e-4
    return ok, "n=%d, allreduce err %.2e, matmul err %.2e" % (n, err_x, err_z)


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
