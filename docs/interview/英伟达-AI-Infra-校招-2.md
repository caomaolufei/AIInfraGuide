---
title: "英伟达 AI Infra 校招 (2)"
description: "英伟达 AI Infra 校招面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "英伟达"
tier: "T3"
interviewType: "校招"
order: 4020
tags: ["算子优化"]
---

### 基础知识

1. SIMT（Single Instruction, Multiple Threads）的含义与工作原理。
2. Occupancy 受哪些因素影响？如何进行调控？
3. Bank Conflict 的粒度是多少？
4. GEMM 分块大小的选择受哪些因素制约？
5. 使用 float4 读写全局内存为何能提升性能？
6. 一个 Block 是否可能被调度到不同的 SM 上执行？
7. 主流 GPU 型号的 Cache 容量分别是多少？
8. Warp Divergence 对性能的具体影响。
9. NVIDIA GPU 中指令级并行（ILP）的实现方式。

### 编程题

10. 实现 CUDA 矩阵转置。
11. 实现向量外积运算。
