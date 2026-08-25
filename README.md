# LLM 算子面试题库（PyTorch / Triton / 分布式）

面向大模型 infra 面试的手写算子练习库。每道题一个文件：
题目背景、考点、`my_*` 待实现桩函数、参考实现 `ref_*`、自动对拍 `check()`。
做完一题跑一次 checker，看着 TODO 变 PASS。

## 快速开始

```bash
cd /disk1/.yc/llm_op_interview

# 检查自己写的 my_* 实现
python run_all.py

# 参考实现自检（题库本身的健康检查）
python run_all.py --solution

# 跳过 triton 题
python run_all.py --quick

# 通讯题需要 torchrun（nproc 2~8 均可）
torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py
torchrun --nproc_per_node=4 comm/c01_ring_allreduce.py --solution

# 或者一次跑完全部通讯题（只初始化一次进程组，更快）
torchrun --nproc_per_node=4 comm/run_comm.py --solution
```

## 题目清单

### compute/ —— 计算算子（单卡）

| 文件 | 内容 | 核心考点 |
| --- | --- | --- |
| c01_softmax | 数值稳定 softmax | 减最大值技巧、memory-bound 分析 |
| c02_norm | LayerNorm + RMSNorm | unbiased=False、RMSNorm 省一遍归约 |
| c03_rope | RoPE 旋转编码 | 复数形式 vs rotate_half、相对位置性质 |
| c04_attention | MHA + causal + GQA | repeat_kv、mask 写法、为何除 sqrt(d) |
| c05_kv_cache | 增量解码 attention | decode 为何 memory-bound、PagedAttention |
| c06_swiglu | SwiGLU FFN | FFN FLOPs 口算、elementwise 融合动机 |
| c07_cross_entropy | 稳定交叉熵 | log-sum-exp、ignore_index/label smoothing |
| c08_sampling | top-k / top-p 采样 | 截断后重归一化、nucleus 边界处理 |
| c09_quant | int8 / int4-group 量化 | scale 公式、per-group 动机、fp8 搭配 |
| c10_autograd | autograd.Function | 梯度推导、save_for_backward、gradcheck |
| c11_triton_softmax | Triton fused softmax | 一行一 program、3 kernel 融合为 1 |
| c12_triton_flash_attn | Triton flash attention | online softmax rescale、IO-aware |

### comm/ —— 通讯算子（多卡，torchrun）

| 文件 | 内容 | 核心考点 |
| --- | --- | --- |
| c01_ring_allreduce | 手写 ring all-reduce | 两阶段 2(n-1) 步、busbw=2(n-1)/n |
| c02_gather_scatter | ring all-gather + reduce-scatter | allreduce 分解、ZeRO/SP 通信骨架 |
| c03_all_to_all | 手写 all-to-all | MoE dispatch/combine、变长通信 |
| c04_busbw | algbw/busbw 计算器 | 各集合通信的修正系数（单机可跑） |
| c05_tp_parallel | Column/Row Parallel Linear | Megatron TP、一个 MLP 一次 allreduce |
| c06_overlap | async 通信+计算重叠 | async_op/stream、DDP bucket |

## 建议刷题顺序

1. c01 → c02 → c07 → c06（elementwise/归约基本功）
2. c03 → c04 → c05（attention 全家桶）
3. c08 → c09 → c10（解码与量化）
4. c04 的 busbw 公式 → c01 → c02 → c03（通信原语）
5. c05 → c06（并行策略映射与 overlap）
6. c11 → c12（Triton 进阶，可选）

## 练习约定

- 只改 `my_*` 函数体，不要动 `check()` 的对拍逻辑。
- 每题 docstring 里的"追问"是面试官视角的延伸问题，白板要能讲。
- 每题都要口算一遍复杂度（FLOPs / 访存量 / 算术强度），
  判断 compute-bound 还是 memory-bound —— 这往往比代码本身更被看重。
- 通讯题注意语义细节：chunk 不等长、isend 张量生命周期、in-place 规则。

## 如何加新题

复制任意一个文件，保留 `my_* / ref_* / check(use_mine)` 三件套，
在 `run_all.py` 的 COMPUTE 或 COMM 列表里加一行即可。
