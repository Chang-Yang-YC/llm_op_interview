"""
题目 c10: 自定义 autograd.Function
===================================
面试题:
    用 torch.autograd.Function 手写 Linear: y = x @ w.t() + b，
    前向保存张量，反向推导并实现三个梯度。

考点:
    1. 梯度推导（必须能白板推）:
       y = x W^T + b,  x: (..., Hin), W: (Hout, Hin)
       dL/dx = gy @ W                 (..., Hin)
       dL/dW = gy_flat^T @ x_flat     (Hout, Hin)
       dL/db = gy.sum(dim=0..-2)      (Hout,)
    2. save_for_backward 的语义: 只保存前向输入/输出的引用，
       autograd 会负责版本检查与内存管理；不能保存中间量后
       in-place 修改。
    3. gradcheck 用 float64 + 小形状做数值差分验证。
    4. 追问: checkpoint(重计算) 就是用 autograd.Function 在反向时
       重跑 forward；torch.compile 下自定义 Function 的限制
       （setup_context、不能混用 ctx 属性）。

复杂度: 前向 2*B*T*Hin*Hout FLOPs。
"""
import torch
import torch.nn.functional as F


class MyLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        # TODO: y = x @ w.t() + b，并保存反向所需张量
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_y):
        # TODO: 返回 (dx, dw, db)
        raise NotImplementedError


class RefLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w)
        return x @ w.t() + b

    @staticmethod
    def backward(ctx, grad_y):
        x, w = ctx.saved_tensors
        gy_flat = grad_y.reshape(-1, grad_y.size(-1))
        x_flat = x.reshape(-1, x.size(-1))
        dx = grad_y @ w
        dw = gy_flat.t() @ x_flat
        db = grad_y.reshape(-1, grad_y.size(-1)).sum(0)
        return dx, dw, db


def check(use_mine: bool = True):
    torch.manual_seed(0)
    fn = MyLinear if use_mine else RefLinear
    # 1) 数值梯度检查（float64，小形状）
    x = torch.randn(2, 4, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(6, 8, dtype=torch.float64, requires_grad=True)
    b = torch.randn(6, dtype=torch.float64, requires_grad=True)
    ok_grad = torch.autograd.gradcheck(fn.apply, (x, w, b),
                                       eps=1e-6, atol=1e-4)
    # 2) 与 F.linear 的梯度对齐
    x2 = torch.randn(3, 5, 8, requires_grad=True)
    w2 = torch.randn(6, 8, requires_grad=True)
    b2 = torch.randn(6, requires_grad=True)
    y_my = fn.apply(x2, w2, b2)
    y_ref = F.linear(x2, w2, b2)
    g = torch.randn_like(y_ref)
    y_my.backward(g, retain_graph=False)
    gx, gw, gb = x2.grad.clone(), w2.grad.clone(), b2.grad.clone()
    x2.grad, w2.grad, b2.grad = None, None, None
    y_ref.backward(g)
    err = max((gx - x2.grad).abs().max().item(),
              (gw - w2.grad).abs().max().item(),
              (gb - b2.grad).abs().max().item())
    return ok_grad and err < 1e-5, "gradcheck %s, grad err %.3e" % (
        ok_grad, err)


if __name__ == "__main__":
    import sys
    ok, msg = check(use_mine="--solution" not in sys.argv)
    print(("PASS" if ok else "FAIL"), msg)
