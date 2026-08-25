"""
题目 c03: RoPE 旋转位置编码
============================
面试题:
    实现 RoPE: 给定 q, k: (B, nH, T, D) 和预计算的旋转角，
    对每个 head 的每对维度施加旋转。

考点:
    1. 两种等价实现:
       a) 复数乘法: 把 D 维看成 D/2 个复数 (x0+i*x1, x2+i*x3, ...)，
          乘以 e^{i*m*theta_j}（LLaMA 官方写法）。
       b) rotate_half: 偶数维/奇数维分开，x'*cos - x_pair*sin。
       两者必须逐位一致 —— 注意是 (x_0,x_1) 交错配对，
       不是 (前半, 后半) 配对（GPT-NeoX 交错 vs LLaMA-half 是两种约定）。
    2. 核心性质: <R(m)q, R(n)k> = q^T R(n-m) k，注意力分数只依赖
       相对位置 (m-n)。要能手推 R(m)^T R(n) = R(n-m)。
    3. theta=10000，频率 theta_j = theta^(-2j/D) 是几何级数，
       低频维编码长程相对位置。追问: NTK-aware / YaRN /
       position interpolation 各自怎么外推。

复杂度: O(B*nH*T*D)。
"""
import torch


def precompute_freqs(dim: int, max_t: int, theta: float = 10000.0):
    """返回 (max_t, dim/2) 的复数张量 e^{i*t*theta_j}。"""
    inv_freq = theta ** (-torch.arange(0, dim, 2).float() / dim)
    t = torch.arange(max_t).float()
    ang = torch.outer(t, inv_freq)  # (T, dim/2)
    return torch.polar(torch.ones_like(ang), ang)


def my_rope(q: torch.Tensor, k: torch.Tensor, freqs_cis):
    """q,k: (B, nH, T, D); freqs_cis: (T, D/2) complex。返回旋转后的 (q', k')。"""
    # TODO: 实现 RoPE（复数形式或 rotate_half 均可）
    raise NotImplementedError


def ref_rope(q, k, freqs_cis):
    def rot(x):
        xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        f = freqs_cis.to(x.device)[None, None]  # (1, 1, T, D/2)
        out = torch.view_as_real(xc * f).flatten(-2)
        return out.type_as(x)
    return rot(q), rot(k)


def rotate_half_form(x, ang):
    """交错配对的 rotate_half 等价形式，用于交叉验证。ang: (T, D/2) 实角。"""
    cos, sin = ang.cos(), ang.sin()
    x_even, x_odd = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x_even * cos - x_odd * sin,
                       x_even * sin + x_odd * cos), dim=-1)
    return out.flatten(-2)


def check(use_mine: bool = True):
    torch.manual_seed(0)
    B, nH, T, D = 2, 8, 64, 64
    q = torch.randn(B, nH, T, D)
    k = torch.randn(B, nH, T, D)
    freqs = precompute_freqs(D, T)

    rq, rk = ref_rope(q, k, freqs)
    mq, mk = my_rope(q, k, freqs) if use_mine else (rq, rk)
    err = max((mq - rq).abs().max().item(), (mk - rk).abs().max().item())

    # 参考解再与 rotate_half 形式互验（仅验证参考实现自洽）
    ang = torch.outer(torch.arange(T).float(),
                      10000.0 ** (-torch.arange(0, D, 2).float() / D))
    err2 = (rq - rotate_half_form(q.float(), ang)).abs().max().item()

    ok = err < 1e-5 and err2 < 1e-5
    return ok, "err vs ref %.3e, ref vs rotate-half %.3e" % (err, err2)


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
