---
title: "字节跳动 Data AI Infra 实习 一面"
description: "字节跳动 Data AI Infra 实习一面面试真题，涵盖 FP8 KV Cache、Prefill、Profiling 与 GPU 微架构"
pubDate: 2026-03-13
company: "字节跳动"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1057
tags: ["推理优化", "算子优化", "高性能计算"]
---

### 推理与量化

1. 部署推理服务时如何选择推理框架？
2. FP8 KV Cache 是怎么设置的？
3. FP8 量化在 Prefill 阶段能带来什么收益？受哪些硬件、模型和形状条件影响？

### 性能分析

4. Profile 用什么工具，怎么定位性能瓶颈？
5. 在 SM 没有用满的情况下增加矩阵乘法的大小，为什么延时会增长？
6. 除了 Compute Bound 和 Memory Bound，还有哪些类型的 Bound，分别怎么定位？

### 编程题

7. 岛屿数量。
