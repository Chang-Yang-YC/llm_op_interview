"""
题目 c01: 数值稳定 Softmax (numerically stable softmax)
=========================================================
面试题:
    不用 torch.softmax / F.softmax，实现 softmax(x, dim)。
    输入可能含很大的正数（如 60.0），fp32 下 exp(60) 不溢出，
    但 fp16 下 exp(11.1) 就溢出 —— 你的实现必须稳定。

考点:
    1. 减最大值技巧: exp(x-c)/sum(exp(x-c)) 与原式恒等
       （分子分母同乘 exp(-c) 抵消），取 c=max(x) 后指数最大为 0。
    2. Roofline 分析: B*N 个 fp32 元素，读+写 = 8*B*N 字节，
       FLOPs 约 3*B*N (减法/exp/求和/除法)，算术强度 ~0.4 FLOP/byte，
       远低于 GPU 拐点 => memory bound。优化方向永远是和相邻
       elementwise 融合（见 c11 triton 版）。
    3. 追问: online softmax / flash attention 里的 rescale 系数
       exp(m_old - m_new) 是怎么来的？（提示: 分块更新 max 时，
       历史累积量要乘上这个修正因子）

复杂度: O(B*N) 时间, O(B*N) 额外空间。
"""
import torch
import torch.nn.functional as F


def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # TODO: 实现稳定 softmax，不允许调用 torch.softmax / F.softmax
    raise NotImplementedError


def ref_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp(x - m)
    return e / e.sum(dim=dim, keepdim=True)


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(8, 4096, device=device) * 60.0  # 大幅值，测稳定性
    ref = F.softmax(x, dim=-1)
    out = my_softmax(x, dim=-1) if use_mine else ref_softmax(x, dim=-1)
    if not torch.isfinite(out).all():
        return False, "non-finite output, missing max-shift trick?"
    err = (out - ref).abs().max().item()
    return err < 1e-6, "max abs err %.3e" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
