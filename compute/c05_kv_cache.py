"""
题目 c05: KV Cache 增量解码
============================
面试题:
    实现自回归解码的单步 attention:
    给定新 token 的 q_t (B, nH, 1, d)、新算出的 k_t/v_t (B, nKV, 1, d)
    和历史缓存 past_k/past_v (B, nKV, T, d)，
    返回 (out (B, nH, 1, d), new_k, new_v)。

考点:
    1. decode 步内不需要 causal mask —— 单个 query 对所有历史位置
       都可见；mask 已经在 prefill 阶段处理过了。
    2. cache 拼接是 cat 出新张量还是写入预分配 buffer？
       （后者避免反复 realloc，vLLM 进一步用 paged block 表）
    3. 为什么 decode 是 memory bound: 每生成一个 token 要读整个
       KV cache，计算量 O(d) 而访存 O(T*d*nKV)。
       追问: MQA/GQA、KV 量化、PagedAttention 分别缓解什么。
    4. 正确性判据: 增量结果必须和"把整段序列做一次完整 attention
       取最后一个位置"逐位一致。

复杂度: 每步 O(B*nH*T*d)。
"""
import math
import torch


def my_decode_step(q_t: torch.Tensor, k_t: torch.Tensor, v_t: torch.Tensor,
                   past_k: torch.Tensor, past_v: torch.Tensor):
    """返回 (out, new_k, new_v)。GQA: nH 是 nKV 整数倍。"""
    # TODO
    raise NotImplementedError


def _repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    return x[:, :, None].expand(-1, -1, n_rep, -1, -1).reshape(
        x.size(0), x.size(1) * n_rep, x.size(2), x.size(3))


def ref_decode_step(q_t, k_t, v_t, past_k, past_v):
    k = torch.cat([past_k, k_t], dim=2)
    v = torch.cat([past_v, v_t], dim=2)
    kk = _repeat_kv(k, q_t.size(1) // k.size(1))
    vv = _repeat_kv(v, q_t.size(1) // v.size(1))
    scores = q_t @ kk.transpose(-2, -1) / math.sqrt(q_t.size(-1))
    return torch.softmax(scores, dim=-1) @ vv, k, v


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, nH, nKV, T, d = 2, 8, 2, 15, 64
    past_k = torch.randn(B, nKV, T, d, device=device)
    past_v = torch.randn(B, nKV, T, d, device=device)
    q_t = torch.randn(B, nH, 1, d, device=device)
    k_t = torch.randn(B, nKV, 1, d, device=device)
    v_t = torch.randn(B, nKV, 1, d, device=device)

    ro, rk, rv = ref_decode_step(q_t, k_t, v_t, past_k, past_v)
    mo, mk, mv = (my_decode_step(q_t, k_t, v_t, past_k, past_v)
                  if use_mine else (ro, rk, rv))
    err = max((mo - ro).abs().max().item(),
              (mk - rk).abs().max().item(),
              (mv - rv).abs().max().item())
    shape_ok = tuple(mk.shape) == (B, nKV, T + 1, d)
    return err < 1e-5 and shape_ok, \
        "max abs err %.3e, cache shape %s" % (err, tuple(mk.shape))


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
