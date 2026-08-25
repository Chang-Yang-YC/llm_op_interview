"""
题目 c08: Top-k / Top-p 采样
=============================
面试题:
    给定 logits (V,)，实现带 temperature 的 top-k + top-p(nucleus)
    采样，返回采到的 token id。

考点:
    1. 处理顺序: 先除 temperature，再做 top-k / top-p 截断，
       截断后必须重新归一化再 multinomial。
    2. top-p 的标准写法: 按概率降序排，cumsum，保留满足
       "cumsum - p_i < p" 的最小集合 —— 这保证 top-1 永远保留，
       即使它自身概率就超过 p。
    3. top-k 用 kth 值阈值而不是真的取前 k 个再 scatter 回去
       （并列值的处理）。
    4. 追问: temperature=0 等价 greedy；repetition penalty /
       frequency penalty 作用在 logits 的哪一步；
       min-p 采样是什么。

复杂度: O(V log V)（排序），可用 O(V) 的 topk 优化。
"""
import torch


def my_sample(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0,
              temperature: float = 1.0, generator=None) -> int:
    """logits: (V,)。返回采样 token id。top_k=0 表示不启用。"""
    # TODO
    raise NotImplementedError


def ref_sample(logits, top_k=0, top_p=1.0, temperature=1.0, generator=None):
    logits = logits.float() / temperature
    if top_k > 0:
        kth = torch.topk(logits, top_k).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p < 1.0:
        sorted_p, sorted_i = torch.sort(probs, descending=True)
        cum = sorted_p.cumsum(-1)
        drop = (cum - sorted_p) >= top_p  # 保留最小 nucleus 集合
        sorted_p = sorted_p.masked_fill(drop, 0.0)
        probs = probs.scatter(-1, sorted_i, sorted_p)
        probs = probs / probs.sum()
    return torch.multinomial(probs, 1, generator=generator).item()


def check(use_mine: bool = True):
    torch.manual_seed(0)
    V = 64
    logits = torch.randn(V) * 2.0

    # 1) greedy 路径必须一致（top_k=1 退化为 argmax）
    g_ref = ref_sample(logits.clone(), top_k=1)
    g_my = my_sample(logits.clone(), top_k=1) if use_mine else g_ref
    if g_my != g_ref:
        return False, "greedy mismatch: got %d want %d" % (g_my, g_ref)

    # 2) 随机采样: 比较经验分布的总变差距离
    N = 4000
    cfg = dict(top_k=8, top_p=0.9, temperature=0.8)
    if use_mine:
        cnt_my = torch.zeros(V)
        cnt_ref = torch.zeros(V)
        for i in range(N):
            s = i * 7919 + 13
            tok_my = my_sample(logits.clone(), **cfg,
                               generator=torch.Generator().manual_seed(s))
            tok_ref = ref_sample(logits.clone(), **cfg,
                                 generator=torch.Generator().manual_seed(s))
            cnt_my[tok_my] += 1
            cnt_ref[tok_ref] += 1
        tv = 0.5 * (cnt_my / N - cnt_ref / N).abs().sum().item()
    else:
        tv = 0.0
    return tv < 0.12, "greedy ok, total-variation %.3f (top_k=8, top_p=0.9)" % tv


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
