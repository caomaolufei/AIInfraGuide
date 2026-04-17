---
title: "快手 AI Infra 校招 (2)"
description: "快手 AI Infra 校招面试真题，涵盖训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "快手"
tier: "T1"
interviewType: "校招"
order: 2011
tags: ["训练优化", "算子优化", "高性能计算"]
---

### 基础知识

1. H100 相较于 A100 在架构层面有哪些关键改进？
2. 阐述 CUDA 中 Warp 的概念及其在执行模型中的作用。
3. 数据并行（DP）与张量并行-序列并行（TP-SP）中，计算与通信重叠的原理是什么？具体是哪些通信操作与哪些计算操作进行重叠？
4. 深入阐述 FlashAttention 的核心原理及其关键优化点。
5. 使用流水线并行（PP）和不使用 PP 时，显存峰值是否相同？为什么？
6. CUDA_DEVICE_MAX_CONNECTIONS 这一环境变量的具体含义是什么？
7. CUDA 中 launch bound 的含义是什么？H2D 和 D2H 数据传输是否可以重叠执行？

### 编程题

8. 实现 LRU 缓存。
9. 编写 Online Softmax 以及 FlashAttention 的伪代码。
