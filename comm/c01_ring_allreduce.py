"""
题目 c01: 手写 Ring All-Reduce
================================
面试题:
    用 dist.isend / dist.irecv 在 ring 拓扑上手写 all-reduce(SUM)，
    与 dist.all_reduce 对齐。

考点:
    1. 两阶段分解: 前 n-1 步做 reduce-scatter（每步把自己负责的
       chunk 发给右邻居并累加收到的 chunk），后 n-1 步做
       all-gather 转发。共 2(n-1) 步，每步每卡发送 data/n。
    2. 带宽公式（必背）: 总传输量 2*(n-1)/n * data，
       busbw = data * 2(n-1)/n / time。面试常给 nccl-tests 输出
       让你判断是否打满 NVLink。
    3. 细节: chunk 不等长用 tensor_split；isend 的张量在 wait
       之前不能被改写（所以发送块与接收块必须不同）；
       in-place 累加用 add_。
    4. 一个隐蔽的坑: 逐条调用 isend/irecv（unbatched P2P）会触发
       每对 rank 懒创建 2-rank 通信子，ring 的首发顺序构成循环依赖
       => 死锁。正确姿势是 dist.batch_isend_irecv + P2POp，
       torch 会按全局一致的顺序建立通信子。
    5. 追问: ring vs tree 的延迟/带宽权衡（NCCL 双算法 + NVLS）；
       为什么 allreduce 天然适合与反向计算 overlap（bucket 化）。

运行: torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py
自测参考实现: 加 --solution
"""
import chunk
import os
import torch
import torch.distributed as dist

HINT = "run: torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py"


def my_ring_allreduce(t: torch.Tensor, group=None) -> torch.Tensor:
    # """in-place ring all-reduce(SUM)，返回 t。"""
    # # TODO
    # raise NotImplementedError
    worldsize = dist.get_world_size(group)
    rank_id = dist.get_rank(group)
    chunks = list(t.tensor_split(worldsize))

    for step in range(worldsize - 1):
        send_idx = (rank_id - step - 1 + worldsize) % worldsize
        recv_idx = (rank_id - step + worldsize - 2) % worldsize
        recv_buf = torch.empty_like(chunks[recv_idx])
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_idx], (rank_id + 1) % worldsize, group=group),
            dist.P2POp(dist.irecv, recv_buf, (rank_id - 1 + worldsize) % worldsize, group=group),
        ])
        for op in ops:
            op.wait()
        chunks[recv_idx].add_(recv_buf)
    for step in range(worldsize - 1):
        send_idx = (rank_id - step + worldsize) % worldsize
        recv_idx = (rank_id - step + worldsize - 1) % worldsize
        recv_buf = torch.empty_like(chunks[recv_idx])
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_idx], (rank_id + 1) % worldsize, group=group),
            dist.P2POp(dist.irecv, recv_buf, (rank_id - 1 + worldsize) % worldsize, group=group),
        ])
        for op in ops:
            op.wait()
        chunks[recv_idx].copy_(recv_buf)
    return t



def ring_allreduce(t, group=None):
    n = dist.get_world_size(group)
    r = dist.get_rank(group)
    chunks = list(t.tensor_split(n))
    # phase 1: reduce-scatter
    # 索引规则: 第 step 步发送块 (r-step-1)%n，接收块 (r-step-2)%n 并原地累加；
    # n-1 步后 rank r 恰好持有归约完成的块 r（若用 (r-step)/(r-step-1)，
    # 归属会错位成块 (r+1)%n，phase 2 就会转发未归约的数据）。
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
    # phase 2: all-gather
    for step in range(n - 1):
        send_idx = (r - step) % n
        recv_idx = (r - step - 1) % n
        recv_buf = torch.empty_like(chunks[recv_idx])
        ops = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, chunks[send_idx], (r + 1) % n, group=group),
            dist.P2POp(dist.irecv, recv_buf, (r - 1 + n) % n, group=group),
        ])
        for op in ops:
            op.wait()
        # 必须 copy_ 回原张量的视图，保持 in-place 语义；
        # 只重绑定列表元素不会写回 t（常见 bug）
        chunks[recv_idx].copy_(recv_buf)
    return t


def check(use_mine: bool = True):
    if not (dist.is_available() and dist.is_initialized()):
        return None, HINT
    n = dist.get_world_size()
    r = dist.get_rank()
    dev = torch.device("cuda", r % max(1, torch.cuda.device_count()))
    torch.manual_seed(123)
    x = torch.randn(1024 * n, 64, device=dev) + r
    ref = x.clone()
    dist.all_reduce(ref, op=dist.ReduceOp.SUM)
    fn = my_ring_allreduce if use_mine else ring_allreduce
    y = fn(x.clone())
    dist.barrier()
    err = (y - ref).abs().max().item()
    return err < 1e-4, "n=%d, max abs err %.3e vs dist.all_reduce" % (n, err)


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
