"""
题目 c03: All-to-All（EP dispatch 的最小模型）
================================================
面试题:
    用 send/recv 手写 all-to-all: 每卡把输入按 rank 数切成 n 块，
    第 i 块发给 rank i；同时从所有 rank 收取属于自己的块，
    按来源 rank 顺序拼好返回。

考点:
    1. all-to-all 的语义: 转置式通信 —— 输出[i] = 来自 rank i 的
       第 r 块。与 allgather（每卡都拿到全量）本质不同，
       all-to-all 总数据量守恒。
    2. MoE 的骨架（高频追问）:
       - dispatch: token embeddings 按路由结果 all-to-all 到
         expert 所在 rank；combine: 结果再 all-to-all 回来。
       - 真实 token 数不均 => 变长 all-to-all
         （dist.all_to_all_single 的分块 sizes 参数 / DeepEP 的
         low-latency 模式用 SM 内直发）。
       - 通信量: 每层 MoE 两次 all-to-all，各 ~data 量级，
         EP 越大跨机流量越大 => 为什么要 EP 内用 NVLink。
    3. busbw 系数: all-to-all 是 (n-1)/n * data。

运行: torchrun --nproc_per_node=4 comm/c03_all_to_all.py [--solution]
"""
import os
import torch
import torch.distributed as dist

HINT = "run: torchrun --nproc_per_node=4 comm/c03_all_to_all.py"


def my_all_to_all(t: torch.Tensor, group=None) -> torch.Tensor:
    """t: (n*chunk, ...)，切 n 块分发给各 rank，返回按来源拼好的结果。"""
    # TODO
    raise NotImplementedError


def ring_all_to_all(t, group=None):
    n = dist.get_world_size(group)
    r = dist.get_rank(group)
    blocks = list(t.tensor_split(n))
    out = [None] * n
    out[r] = blocks[r].clone()
    for step in range(1, n):
        dst = (r + step) % n          # 把自己的第 dst 块发给 rank dst
        src = (r - step + n) % n      # 从 rank src 收它的第 r 块
        recv_buf = torch.empty_like(blocks[r])
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, blocks[dst].contiguous(), dst, group=group),
            dist.P2POp(dist.irecv, recv_buf, src, group=group),
        ])
        for op in ops:
            op.wait()
        out[src] = recv_buf
    return torch.cat(out, dim=0)


def check(use_mine: bool = True):
    if not (dist.is_available() and dist.is_initialized()):
        return None, HINT
    n = dist.get_world_size()
    r = dist.get_rank()
    dev = torch.device("cuda", r % max(1, torch.cuda.device_count()))
    torch.manual_seed(11)
    # 让每个元素带上 (rank, index) 信息，便于人眼验证语义
    x = torch.arange(256 * n, device=dev, dtype=torch.float32) \
        .reshape(256 * n, 1) + r * 1e6
    ref_out = torch.empty_like(x)
    dist.all_to_all_single(ref_out, x)  # 单个张量，内部按 dim0 切 n 块
    fn = my_all_to_all if use_mine else ring_all_to_all
    out = fn(x.clone())
    dist.barrier()
    err = (out - ref_out).abs().max().item()
    return err < 1e-3, "n=%d, max abs err %.3e vs dist.all_to_all_single" % (n, err)


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
