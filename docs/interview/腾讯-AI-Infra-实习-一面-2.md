---
title: "腾讯 AI Infra 实习 一面 (2)"
description: "腾讯 AI Infra 实习一面面试真题，涵盖推理优化、高性能计算等方向"
pubDate: 2026-04-16
company: "腾讯"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1039
tags: ["推理优化", "高性能计算"]
---

微信 大模型算法开发 一面
1. 介绍下你实习学了什么 做过什么项目,为什么离职的
2. MoE 为什么能在参数量很大的情况下还能把训练和推理成本压住，真正难点在哪

MoE 的关键不是“参数变多了”，而是“每个 token 只激活一部分参数”。也就是说总参数量可以做得很大，但单次前向只走少数几个 expert，所以理论上计算量不会随着总参数线性增长。这个思路在大模型里很有吸引力，因为可以同时兼顾容量和成本。

但真正难的地方是路由和负载均衡。路由器如果只偏爱少数 expert，训练会很不稳定，热门 expert 被打爆，冷门 expert 学不到东西。另一个难点是通信开销，尤其多机训练时，token dispatch 和 gather 的代价不低。所以 MoE 能不能跑好，不只是模型结构问题，更是系统问题。

	
import torch
 
logits = torch.randn(4, 8)   # 4个token, 8个expert
topk_val, topk_idx = torch.topk(logits, k=2, dim=-1)
print(topk_idx)  # 每个token选择两个expert
3. MoE 里的负载均衡一般怎么做，为什么很多模型看起来 loss 正常但 expert 已经废了

最常见的方法是给路由器加辅助负载均衡损失，让 token 分配更均匀，避免所有 token 都涌向几个强 expert。还有一种做法是设置容量上限，超过容量的 token 要么被丢弃，要么走次优 expert。这样能抑制极端拥塞，但也会引入路由不连续的问题。

很多模型训练时总 loss 看起来没问题，但 expert 其实已经塌了，原因是主任务 loss 并不会直接告诉你“是不是只有两个 expert 在干活”。所以做 MoE 训练时，我会额外盯几类指标：每个 expert 的 token 占比、路由熵、溢出比例、不同层之间的激活偏差。只看 loss，很容易被骗。

4. GQA、MQA 和标准 MHA 的区别是什么，为什么线上推理里大家更关心 GQA

标准 MHA 是每个 head 都有独立的 Q、K、V，这样表达能力最完整，但 KV cache 的占用也最大。MQA 是多个 query head 共享一组 K、V，极大节省 KV cache，但表达能力会损失得比较明显。GQA 可以看成两者折中，把多个 query head 分成组，每组共享一套 K、V，所以在效果和推理成本之间比较平衡。

线上推理更关心 GQA，是因为它直接关系到长上下文服务成本。很多时候显存瓶颈不是模型权重，而是 KV cache。GQA 能让你在不明显伤效果的情况下，把 cache 压下去不少，所以工程上很实用。

5. RoPE 为什么能做位置编码，长上下文外推为什么经常失真

RoPE 的核心不是给 token 加一个绝对位置向量，而是把位置信息编码进注意力里的相对相位关系。这样做的好处是相对位置关系能自然进入注意力计算，模型更容易学到“前后依赖”而不是死记具体索引。它在中等长度上下文里很稳，也是很多大模型默认方案。

外推时经常失真，是因为模型训练时看到的相位范围有限，推理时如果把位置硬拉得很长，高频部分会扭曲，模型虽然能吃进长文本，但并不真的理解长距离关系。很多长上下文技巧，比如 NTK-aware scaling、YaRN，本质上都是在想办法减缓这种频率失配。

	
import math
 
def rope_theta(pos, dim, base=10000):
    return [pos / (base ** (2 * (i//2) / dim)) for i in range(dim)]
 
print(rope_theta(10, 8))
6. FlashAttention 为什么会快，它优化的到底是算力还是访存

FlashAttention 真正优化的重点不是减少理论计算量，而是减少 HBM 读写，把 attention 计算尽可能放在 SRAM 里分块完成。传统 attention 会显式构造大

剩余60%内容，订阅专栏后可继续查看/也可单篇购买

AI-Agent面试实战专栏
