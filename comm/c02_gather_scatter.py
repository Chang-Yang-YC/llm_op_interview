"""
题目 c02: Ring All-Gather / Reduce-Scatter 与 allreduce 分解
==============================================================
面试题:
    1) 手写 ring all-gather（输入本地分片，输出完整张量）
       和 ring reduce-scatter（输入完整张量，输出属于本卡的分片）。
    2) 验证: allreduce = reduce-scatter + allgather。

考点:
    1. ring all-gather: n-1 步，每步把"刚收到的 chunk"转发给右邻居
       （第 0 步转发自己的分片）。
    2. reduce-scatter 与 c01 的 phase 1 完全相同。
    3. ZeRO 的通信骨架（高频追问）:
       - ZeRO-1/2 梯度同步用 reduce-scatter: 每卡只保留自己
         optimizer state 对应的那份梯度分片；
       - ZeRO-3 前向/反向还要按需 all-gather 参数，用完即丢；
       - Megatron 序列并行把 TP 的 allreduce 拆成
         reduce-scatter + allgather: 总量不变，但 LayerNorm/Dropout
         也能按序列维切分激活。
    4. busbw 系数: allgather / reduce-scatter 都是 (n-1)/n * data。

运行: torchrun --nproc_per_node=4 comm/c02_gather_scatter.py [--solution]
"""
import os
import torch
import torch.distributed as dist

HINT = "run: torchrun --nproc_per_node=4 comm/c02_gather_scatter.py"


def my_ring_all_gather(local: torch.Tensor, group=None) -> torch.Tensor:
    """输入本卡分片 (data/n, ...)，返回拼好的完整张量。"""
    # TODO
    raise NotImplementedError


def my_ring_reduce_scatter(t: torch.Tensor, group=None) -> torch.Tensor:
    """输入完整张量 t，返回 reduce 后属于本卡的分片。"""
    # TODO
    raise NotImplementedError


def ring_all_gather(local, group=None):
    n = dist.get_world_size(group)
    r = dist.get_rank(group)
    have = [None] * n
    have[r] = local
    cur = local
    for step in range(n - 1):
        recv_idx = (r - step - 1) % n
        recv_buf = torch.empty_like(cur)
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, cur, (r + 1) % n, group=group),
            dist.P2POp(dist.irecv, recv_buf, (r - 1 + n) % n, group=group),
        ])
        for op in ops:
            op.wait()
        have[recv_idx] = recv_buf
        cur = recv_buf
    return torch.cat(have, dim=0)


def ring_reduce_scatter(t, group=None):
    n = dist.get_world_size(group)
    r = dist.get_rank(group)
    chunks = list(t.tensor_split(n))
    # 与 allreduce 的 phase 1 相同：结束后 rank r 持有块 r
    for step in range(n - 1):
        send_idx = (r - step - 1) % n
        recv_idx = (r - step - 2) % n
        recv_buf = torch.empty_like(chunks[recv_idx])
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_idx], (r + 1) % n, group=group),
            dist.P2POp(dist.irecv, recv_buf, (r - 1 + n) % n, group=group),
        ])
        for op in ops:
            op.wait()
        chunks[recv_idx].add_(recv_buf)
    return chunks[r].clone()


def check(use_mine: bool = True):
    if not (dist.is_available() and dist.is_initialized()):
        return None, HINT
    n = dist.get_world_size()
    r = dist.get_rank()
    dev = torch.device("cuda", r % max(1, torch.cuda.device_count()))
    torch.manual_seed(7)
    x = torch.randn(512 * n, 32, device=dev) + r
    local = x.tensor_split(n)[r].clone()

    # --- all-gather 对拍 dist.all_gather ---
    gathered_ref = [torch.empty_like(local) for _ in range(n)]
    dist.all_gather(gathered_ref, local)
    gt_gather = torch.cat(gathered_ref, dim=0)
    fn_g = my_ring_all_gather if use_mine else ring_all_gather
    err_g = (fn_g(local) - gt_gather).abs().max().item()

    # --- reduce-scatter 对拍 dist.reduce_scatter_tensor ---
    out_rs_ref = torch.empty_like(local)
    dist.reduce_scatter_tensor(out_rs_ref, x)
    fn_rs = my_ring_reduce_scatter if use_mine else ring_reduce_scatter
    err_rs = (fn_rs(x.clone()) - out_rs_ref).abs().max().item()

    # --- 组合等价性: rs + ag == allreduce ---
    shard = ring_reduce_scatter(x.clone())
    composed = ring_all_gather(shard)
    ar = x.clone()
    dist.all_reduce(ar)
    err_comp = (composed - ar).abs().max().item()
    dist.barrier()

    ok = err_g < 1e-5 and err_rs < 1e-4 and err_comp < 1e-4
    return ok, ("n=%d gather err %.2e, rs err %.2e, rs+ag==ar err %.2e"
                % (n, err_g, err_rs, err_comp))


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
