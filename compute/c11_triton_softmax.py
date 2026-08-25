"""
题目 c11: [Triton] Fused Softmax
==================================
面试题:
    用 Triton 写 fused row-softmax（每行一个 program），
    并解释它比 PyTorch 原生 softmax 快在哪。

考点:
    1. kernel 结构: pid 对应行号；一行装进一个 BLOCK
       （next_power_of_2(n_cols)），mask 处理非对齐；
       行内做 max -> exp -> sum -> 归一化，一次 kernel 完成，
       中间结果全部留在寄存器。
    2. 为什么快: 原生 softmax 是 3 个 kernel（max/sum/div 的
       变体），每个都要全量读写一遍 HBM；fused 后只有 1 读 1 写，
       memory bound 算子直接 ~3x。
    3. 追问: 一行太长装不下寄存器怎么办？（two-pass / online 技巧，
       衔接 flash attention）；num_warps 怎么选；
       tl.load 的 other=-inf 在 softmax 里为什么安全。

前置: pip install triton（随 torch 官方 wheel 自带）。
"""
import torch

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _softmax_kernel(out_ptr, in_ptr, n_cols,
                        BLOCK: tl.constexpr):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        mask = cols < n_cols
        x = tl.load(in_ptr + row * n_cols + cols, mask=mask, other=-float("inf"))
        x = x - tl.max(x, axis=0)
        e = tl.exp(x)
        e = e / tl.sum(e, axis=0)
        tl.store(out_ptr + row * n_cols + cols, e, mask=mask)


def my_triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """x: (rows, cols)，返回同形状 softmax 结果。"""
    # TODO: 计算 grid、BLOCK、num_warps 并启动 kernel
    raise NotImplementedError


def ref_triton_softmax(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(cols)
    num_warps = 4 if BLOCK >= 2048 else (2 if BLOCK >= 512 else 1)
    _softmax_kernel[(rows,)](out, x, cols, BLOCK=BLOCK, num_warps=num_warps)
    return out


def check(use_mine: bool = True):
    if not HAS_TRITON:
        return None, "triton not installed"
    if not torch.cuda.is_available():
        return None, "needs CUDA"
    torch.manual_seed(0)
    x = torch.randn(1024, 4096, device="cuda", dtype=torch.float32)
    ref = torch.softmax(x, dim=-1)
    out = my_triton_softmax(x) if use_mine else ref_triton_softmax(x)
    err = (out - ref).abs().max().item()
    return err < 1e-5, "max abs err %.3e" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL") if ok is not None else "SKIP", msg)
