"""
题目 c04: Multi-Head Attention（causal mask + GQA）
====================================================
面试题:
    实现 scaled dot-product attention:
    q: (B, nH, T, d), k/v: (B, nKV, T, d)，nH 是 nKV 的整数倍（GQA）。
    支持 causal mask。

考点:
    1. GQA: k/v 的 head 要 repeat 到 nH 份再算（repeat_kv），
       追问 KV cache 显存节省比例 nKV/nH，MQA 是 nKV=1 的极端。
    2. causal mask 的实现: 上三角置 -inf 再过 softmax。
       追问: 全 -inf 行 softmax 会出 nan（什么时候出现？
       padding + causal 同时作用时），安全写法是什么。
    3. 为什么除 sqrt(d): 点积方差随 d 线性增长，缩放使 logits 方差
       回到 ~1，避免 softmax 进入饱和区梯度消失。
    4. 追问: 这个朴素实现的显存瓶颈 —— scores 矩阵 O(T^2) 要物化，
       FlashAttention 如何用 online softmax 避免（衔接 c12）。

复杂度: O(B*nH*T^2*d) 计算, O(B*nH*T^2) scores 显存。
"""
import math
import torch
import torch.nn.functional as F


def my_repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, nKV, T, d) -> (B, nKV*n_rep, T, d)"""
    # TODO
    raise NotImplementedError


def my_mha(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
           causal: bool = True) -> torch.Tensor:
    # TODO: 完整实现 GQA + causal attention
    raise NotImplementedError


def ref_repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return x[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(
        x.size(0), x.size(1) * n_rep, x.size(2), x.size(3))


def ref_mha(q, k, v, causal=True):
    k, v = ref_repeat_kv(k, q.size(1) // k.size(1)), \
        ref_repeat_kv(v, q.size(1) // v.size(1))
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
    if causal:
        T = q.size(-2)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool,
                                     device=q.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    p = torch.softmax(scores, dim=-1)
    return p @ v


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, nH, nKV, T, d = 2, 8, 2, 32, 64
    q = torch.randn(B, nH, T, d, device=device)
    k = torch.randn(B, nKV, T, d, device=device)
    v = torch.randn(B, nKV, T, d, device=device)

    # 官方 SDPA 做 ground truth（先手动 repeat_kv）
    kk = ref_repeat_kv(k, nH // nKV).contiguous()
    vv = ref_repeat_kv(v, nH // nKV).contiguous()
    gt = F.scaled_dot_product_attention(q, kk, vv, is_causal=True)

    out = my_mha(q, k, v) if use_mine else ref_mha(q, k, v)
    err = (out - gt).abs().max().item()
    return err < 1e-5, "max abs err %.3e (GQA %d->%d heads)" % (err, nKV, nH)


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
