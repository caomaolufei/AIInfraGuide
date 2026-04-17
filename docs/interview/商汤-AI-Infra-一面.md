---
title: "商汤 AI Infra 面试"
description: "商汤 AI Infra 面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "商汤"
tier: "T2"
interviewType: "未知"
order: 3004
tags: ["算子优化"]
---

以下为商汤 AI Infra 面试考察内容，以项目深挖为主。

### 项目经历（GEMM 优化）

1. 详细描述 GEMM 优化的实现过程
2. 单个线程计算 C 矩阵 8x8 个元素的原因
3. 寄存器是否会发生溢出，如何应对
4. 项目中是否有自己的创新点
5. `#pragma unroll` 的作用与使用场景
6. 单个 tile 内执行了多少次计算
7. 与 cuTLASS 相比，自实现版本的性能对比如何

### 基础知识

8. 是否做过卷积算子优化
9. 是否了解 PPL（商汤高性能计算库）
10. Tensor Core 的工作原理与使用方式
11. 是否有低比特位 GEMM 的开发经验
