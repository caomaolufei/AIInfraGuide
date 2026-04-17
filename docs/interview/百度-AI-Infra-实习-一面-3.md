---
title: "百度 AI Infra 实习 一面 (3)"
description: "百度 AI Infra 实习一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "百度"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1029
tags: ["推理优化", "算子优化"]
---

## 分布式训练

1. 解释 TP、PP、DP 三种并行策略的含义及具体执行流程
2. 如何根据 TP 与 PP 的通信开销进行并行策略选择？

## 量化

3. 解释 per-tensor、per-channel、group-wise 三种量化粒度的区别
4. 不同量化方法之间的精度差异，以及 group-wise 为何能进一步降低量化误差
5. 各量化方式的计算开销对比及优化手段
6. 量化过程中如何处理异常值（outlier）？
7. 分别介绍 GPTQ、AWQ、SmoothQuant 的核心思路
8. KV Cache 量化的原理与实现方式

## 注意力机制与推理优化

9. FlashAttention 的加速原理是什么？
10. FlashAttention v1 与 v2 之间的主要区别
11. FlashAttention 中 Bc 块的切分策略，以及 1-loop FlashAttention 的实现方式
12. PagedAttention 的设计思路
13. 大模型 prefill 阶段与 decoding 阶段的区别及其成因
14. FlashAttention 在 decoding 阶段存在什么问题？FlashDecoding 的改进思路

## 算子实现

15. RMSNorm 的具体实现方式
16. CUDA Kernel 优化的一般思路与方法论
17. 给定 Conv2D 输入 [C=64, W=64, H=128]、卷积核 3x3、输出 [C=128, W=64, H=128]，计算参数量与 FLOPs
18. 在 CPU 上进行算子优化有哪些方法（如 AVX-512 等）？

## C++ 基础

19. 智能指针的分类及使用场景

## 编程题

20. 实现 CUDA LayerNorm Kernel
