---
title: "第6章：Attention 算子"
description: "深入理解 FlashAttention V1/V2/V3 的原理与实现，以及 Decode 阶段的 Flash-Decoding 和 PagedAttention CUDA 实现"
pubDate: 2026-04-16
category: "cuda-optimization"
order: 15
tags: ["FlashAttention", "Attention", "Flash-Decoding", "PagedAttention", "CUDA"]
---

## 本章简介

Attention 是 Transformer 的核心计算，也是 AI Infra 优化的重中之重。本章从标准 Attention 的性能问题出发，深入 FlashAttention 系列和 Decode 阶段加速技术。

**标准 Attention 的性能问题**分析朴素实现的 O(N²) 显存占用和计算/显存瓶颈。

**FlashAttention V1**详解核心思想：Tiling + Online Softmax 避免物化 N×N 矩阵，将额外显存从 O(N²) 降到 O(N)、HBM 读写从 Θ(N²) 降到 Θ(N²d²/M)（M 为 SRAM 大小），以及反向传播中的重计算策略。

**FlashAttention V2** 改进的并行策略（沿 Q 序列维度并行）和 Causal Mask 优化。

**FlashAttention-3** 利用 Hopper 架构特性（WGMMA、TMA、FP8）的异步流水线优化。

**Decode 阶段加速**覆盖 Flash-Decoding、FlashDecoding++、FlashInfer 等面向 Serving 的 Attention 引擎。

**PagedAttention CUDA 实现**解读 vLLM 中虚拟页到物理页映射在 GPU 上的实现。

**动手实验**：白板推导 FlashAttention Tiling 过程，用 Triton 实现简化版 FlashAttention。
