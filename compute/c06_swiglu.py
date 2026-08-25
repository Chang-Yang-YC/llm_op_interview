"""
题目 c06: SwiGLU 前馈
======================
面试题:
    实现 SwiGLU(gate, up) = silu(gate) * up。

考点:
    1. 公式: FFN_SwiGLU(x) = down( silu(xW_gate) * (xW_up) )。
       LLaMA/Qwen 的 FFN 中间维 H_ffn 通常取 ~8/3*H（再对齐到 256），
       因为三个矩阵比 ReLU FFN 的两个矩阵多 50% 参数。
    2. FLOPs 口算（高频追问）: gate + up 两个投影 2*B*T*H*H_ffn*2，
       down 投影 2*B*T*H_ffn*H，合计 6*B*T*H*H_ffn。
    3. 优化: silu、逐元素乘、以及和 up/gate 线性层的融合。
       这是典型的 memory bound elementwise 链 —— 中间结果全写回
       HBM 再读回是浪费，生产实现（如 vLLM/SGLang 的
       mul_and_silu）融合成单 kernel。追问: 为什么激活函数值得融合
       而 GEMM 一般不融合（compute bound vs memory bound）。

复杂度: O(B*T*H_ffn)。
"""
import torch
import torch.nn.functional as F


def my_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    # TODO: silu(gate) * up，不要用现成的 SwiGLU 封装
    raise NotImplementedError


def ref_swiglu(gate, up):
    return F.silu(gate) * up


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = torch.randn(4, 128, 4096, device=device)
    up = torch.randn(4, 128, 4096, device=device)
    ref = ref_swiglu(gate, up)
    out = my_swiglu(gate, up) if use_mine else ref
    err = (out - ref).abs().max().item()
    return err < 1e-6, "max abs err %.3e" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
