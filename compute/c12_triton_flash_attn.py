"""
题目 c12: [Triton] Flash Attention（进阶）
============================================
面试题:
    用 Triton 实现分块 (tiled) 的 causal attention，
    即 FlashAttention 的核心 loop，不允许物化 T*T 的 scores 矩阵。

考点:
    1. online softmax: 块状遍历 KV，维护 running max m_i 和
       归一化因子 l_i；每来一个新块，历史累积量 acc 乘上修正系数
       exp(m_old - m_new) —— 这就是 c01 追问的 rescale 的来源。
    2. 分块参数: BLOCK_M (query 块) x BLOCK_N (kv 块)，
       causal 时第 start_m 块只需遍历到 (start_m+1)*BLOCK_M，
       计算量减半。
    3. 复杂度: 计算 O(T^2*d) 不变，但 HBM 访存从 O(T^2) 降到
       O(T^2*d/M)（M 为 SRAM 大小），这就是 IO-aware 的本质。
    4. 追问: FA2 怎么改 loop 方向减少非矩阵乘 FLOPs；
       FA3 在 Hopper 上用了什么（TMA / WGMM / 异步流水线）。

前置: triton + CUDA。形状 (B, H, T, D), D=64, fp16。
"""
import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _attn_fwd(Q, K, V, Out, sm_scale,
                  stride_qz, stride_qh, stride_qm, stride_qd,
                  stride_kz, stride_kh, stride_kn, stride_kd,
                  stride_vz, stride_vh, stride_vn, stride_vd,
                  Z, H, N_CTX,
                  BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
                  BLOCK_N: tl.constexpr, IS_CAUSAL: tl.constexpr):
        start_m = tl.program_id(0)
        off_hz = tl.program_id(1)
        off_z = off_hz // H
        off_h = off_hz % H
        q_base = Q + off_z * stride_qz + off_h * stride_qh
        k_base = K + off_z * stride_kz + off_h * stride_kh
        v_base = V + off_z * stride_vz + off_h * stride_vh
        o_base = Out + off_z * stride_qz + off_h * stride_qh

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_DMODEL)

        q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        k_ptrs = k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

        q_mask = offs_m[:, None] < N_CTX
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)

        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
        for start_n in range(0, hi, BLOCK_N):
            n_mask = (start_n + offs_n)[:, None] < N_CTX
            k = tl.load(k_ptrs + start_n * stride_kn, mask=n_mask, other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            if IS_CAUSAL:
                qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :],
                               0.0, float("-inf"))
            else:
                qk += tl.where((start_n + offs_n)[None, :] < N_CTX,
                               0.0, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.exp(qk - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]
            v = tl.load(v_ptrs + start_n * stride_vn, mask=n_mask, other=0.0)
            acc += tl.dot(p.to(v.dtype), v)
            m_i = m_new

        acc = acc / l_i[:, None]
        o_ptrs = o_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=q_mask)


def my_flash_attn(q, k, v, causal=True):
    """q,k,v: (B, H, T, 64) fp16。返回 attention 输出。"""
    # TODO: 组 grid、算 sm_scale、启动 _attn_fwd
    raise NotImplementedError


def ref_flash_attn(q, k, v, causal=True):
    B, H, T, D = q.shape
    out = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(T, BLOCK_M), B * H, 1)
    _attn_fwd[grid](q, k, v, out, 1.0 / math.sqrt(D),
                    q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                    k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                    v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                    B, H, T,
                    BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
                    IS_CAUSAL=causal, num_warps=4, num_stages=3)
    return out


def check(use_mine: bool = True):
    if not HAS_TRITON:
        return None, "triton not installed"
    if not torch.cuda.is_available():
        return None, "needs CUDA"
    torch.manual_seed(0)
    B, H, T, D = 2, 4, 128, 64
    q = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    gt = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    out = my_flash_attn(q, k, v) if use_mine else ref_flash_attn(q, k, v)
    err = (out.float() - gt.float()).abs().max().item()
    return err < 1e-2, "max abs err %.3e vs SDPA(fp16)" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL") if ok is not None else "SKIP", msg)
