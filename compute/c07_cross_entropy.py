"""
题目 c07: 数值稳定交叉熵
==========================
面试题:
    实现 cross_entropy(logits (B, V), target (B,))，
    reduction='mean'，与 F.cross_entropy 对齐。

考点:
    1. 不能写成 -log(softmax(x)) 两步: softmax 里 exp 可能溢出，
       且 log 小数有精度损失。标准做法是 log-sum-exp:
       loss_i = -x_i[y] + m + log(sum(exp(x - m)))，m = max(x)。
    2. 数值上验证: x_i - logsumexp(x) == log_softmax(x)_i。
    3. 追问三连:
       - ignore_index 怎么实现？（mask 后按有效样本数取均值）
       - label smoothing 的等价改写: (1-a)*nll + a*(-mean(logp))，
         或 logits 层面的等价形式。
       - 大词表 (V=150k) 下 gather + logsumexp 的访存量，
         为什么训练里这是非 attention 部分的大头。

复杂度: O(B*V)。
"""
import torch
import torch.nn.functional as F


def my_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """logits: (B, V), target: (B,) long。返回标量 mean loss。"""
    # TODO: log-sum-exp 稳定实现
    raise NotImplementedError


def ref_cross_entropy(logits, target):
    m = logits.max(-1, keepdim=True).values
    lse = m.squeeze(-1) + (logits - m).exp().sum(-1).log()
    nll = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    return (lse - nll).mean()


def check(use_mine: bool = True):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logits = torch.randn(16, 32000, device=device) * 30.0  # 尖锐分布
    target = torch.randint(0, 32000, (16,), device=device)
    ref = F.cross_entropy(logits, target, reduction="mean")
    out = my_cross_entropy(logits, target) if use_mine \
        else ref_cross_entropy(logits, target)
    if not torch.isfinite(out):
        return False, "non-finite loss, missing log-sum-exp shift?"
    err = (out - ref).abs().item()
    return err < 1e-5, "abs err %.3e" % err


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
