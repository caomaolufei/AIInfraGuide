---
title: "快手 AI Infra 一面"
description: "快手 AI Infra 一面面试真题，涵盖推理优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "快手"
tier: "T1"
interviewType: "未知"
round: "一面"
order: 2005
tags: ["推理优化", "算子优化", "高性能计算"]
---

### 基础知识

1. 是否了解 Prefill-Decode（PD）分离？AF 分离的目的是什么？既然已有 PD 分离，为何还需要 AF 分离？
2. FlashAttention V2 相比 V1 做了哪些改进？具体优化思路是什么？是否了解最新的 V4 版本？
3. 大模型中一层 Transformer 包含几个线性层？张量并行（TP）时如何切分？这样切分的原因是什么？有哪些思路可以优化中间的 AllReduce 操作？
4. Ray 的底层实现机制是什么？有哪些关键特性？在实际研究中是如何使用 Ray 的？
5. 列举并阐述 CUDA GEMM 的常见优化方法。

### 项目经历

6. 详细介绍项目背景、技术方案和实现细节。

### 编程题

7. LeetCode 单词接龙。
