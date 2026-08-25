"""
题目 c09: int8 / int4-group 量化与反量化
==========================================
面试题:
    1) 实现 per-tensor 对称 int8 量化与反量化。
    2) 实现 per-group 对称 int4 量化（group_size=128，q 用 int8 存）。

考点:
    1. 对称量化: scale = max|w| / qmax；int8 常用 [-127, 127]
       （放弃 -128 保持对称）；int4 范围 [-8, 7]。
    2. 反量化误差上界 scale/2（round-to-nearest）。
    3. 追问链:
       - 为什么权重离群值让 per-tensor 量化崩掉 => per-channel /
         per-group（SmoothQuant/AWQ 进一步把离群值搬动到激活/权重侧）。
       - fp8: e4m3 (max 448, 精度高) 用于 forward，e5m2 (max 57344,
         动态范围大) 用于梯度 —— 为什么这样搭配。
       - w8a8 vs w8a16: 激活量化为什么要 per-token dynamic scale。
       - 量化感知训练 (QAT) 里 straight-through estimator 是什么。

复杂度: O(numel)。
"""
import torch


def my_quant_int8(w: torch.Tensor):
    """per-tensor 对称量化。返回 (q int8, scale float32 标量)。"""
    # TODO
    raise NotImplementedError


def my_dequant(q: torch.Tensor, scale) -> torch.Tensor:
    # TODO
    raise NotImplementedError


def my_quant_int4_group(w: torch.Tensor, group_size: int = 128):
    """w: (out, in)，in 能被 group_size 整除。
    返回 (q int8 存 [-8,7], scales (out, in/group_size) fp32)。"""
    # TODO
    raise NotImplementedError


def ref_quant_int8(w):
    scale = w.abs().max() / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale


def ref_dequant(q, scale):
    return q.to(torch.float32) * scale


def ref_quant_int4_group(w, group_size=128):
    out_f, in_f = w.shape
    g = w.reshape(out_f, in_f // group_size, group_size)
    scale = g.abs().amax(dim=-1, keepdim=True) / 7.0
    q = torch.round(g / scale).clamp(-8, 7).to(torch.int8)
    return q.reshape(out_f, in_f), scale.squeeze(-1)


def check(use_mine: bool = True):
    torch.manual_seed(0)
    w = torch.randn(256, 512)

    q_ref, s_ref = ref_quant_int8(w)
    q_my, s_my = (my_quant_int8(w) if use_mine else (q_ref, s_ref))
    dq = my_dequant(q_my, s_my) if use_mine else ref_dequant(q_ref, s_ref)
    err8 = (dq - w).abs().max().item()
    bound8 = float(s_ref) / 2.0 + 1e-6

    qg_ref, sg_ref = ref_quant_int4_group(w)
    qg_my, sg_my = (my_quant_int4_group(w) if use_mine
                    else (qg_ref, sg_ref))
    err_scale = (sg_my.float() - sg_ref.float()).abs().max().item() \
        if use_mine else 0.0
    range_ok = bool(((qg_my >= -8) & (qg_my <= 7)).all())

    ok = err8 <= bound8 and abs(float(s_my) - float(s_ref)) < 1e-6 \
        and err_scale < 1e-6 and range_ok
    return ok, ("int8 dequant err %.3e (bound %.3e), "
                "int4 scale err %.3e, range %s" %
                (err8, bound8, err_scale, range_ok))


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
