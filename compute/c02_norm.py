"""
题目 c02: LayerNorm + RMSNorm
==============================
面试题:
    分别实现 LayerNorm（带 affine）和 RMSNorm，对最后一维归一化。

考点:
    1. torch.var 默认 unbiased=True（贝塞尔修正 n-1），
       归一化必须用 unbiased=False，否则和 F.layer_norm 对不上。
       这是最常见的翻车点。
    2. RMSNorm = x / sqrt(mean(x^2) + eps) * w，省掉求均值与减均值，
       少一遍归约 => 访存 bound 的 kernel 上快 ~10-15%。
       LLaMA / Qwen 全部用 RMSNorm。
    3. eps 的位置: sqrt(var + eps)，不是 sqrt(var) + eps。
    4. 追问: pre-norm vs post-norm 对训练稳定性的影响；
       DeepNorm / QK-norm 是什么动机。

复杂度: O(B*T*H) 时间, O(B*T*H) 空间（可做到 O(1) 额外空间的融合 kernel）。
"""
import torch
import torch.nn.functional as F


def my_layernorm(x: torch.Tensor, weight: torch.Tensor,
                 bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    # TODO: 对最后一维做 layernorm，x: (..., H), weight/bias: (H,)
    raise NotImplementedError


def my_rmsnorm(x: torch.Tensor, weight: torch.Tensor,
               eps: float = 1e-6) -> torch.Tensor:
    # TODO: 对最后一维做 rmsnorm
    raise NotImplementedError


def ref_layernorm(x, weight, bias, eps=1e-5):
    mean = x.mean(-1, keepdim=True)
    var = x.var(-1, unbiased=False, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


def ref_rmsnorm(x, weight, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4, 128, 1024, device=device)
    w = torch.randn(1024, device=device)
    b = torch.randn(1024, device=device)

    ln_ref = F.layer_norm(x, (1024,), w, b, 1e-5)
    ln_out = my_layernorm(x, w, b) if use_mine else ref_layernorm(x, w, b)
    err_ln = (ln_out - ln_ref).abs().max().item()

    # rmsnorm 无官方对照，用两种等价写法互验
    rms_a = my_rmsnorm(x, w) if use_mine else ref_rmsnorm(x, w)
    rms_b = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    err_rms = (rms_a - rms_b).abs().max().item()

    ok = err_ln < 1e-5 and err_rms < 1e-5
    return ok, "ln err %.3e, rms err %.3e" % (err_ln, err_rms)


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
