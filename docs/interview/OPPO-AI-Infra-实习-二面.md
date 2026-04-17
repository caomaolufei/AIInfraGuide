---
title: "OPPO AI Infra 实习 二面"
description: "OPPO AI Infra 实习二面面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "OPPO"
tier: "T1"
interviewType: "实习"
round: "二面"
order: 2029
tags: ["算子优化", "高性能计算"]
---

## 项目经历提问

1. 对项目进行深入考察与追问

## 理论基础

2. 在什么情况下可以判定算子融合（Operator Fusion）是有收益的？哪些典型场景适合进行融合？
3. CUDA Stream 的异步执行原理是什么？使用异步执行需要满足哪些前提条件（如避免内存访问冲突）？
4. 如何确定一个 CUDA kernel 最优的线程数量配置？
5. 在 CUDA 编程中，Thread、Warp、Block、SM、Grid 之间存在怎样的层级对应关系？
6. 如果在某些特殊 Shape 下使用 Shared Memory 导致了计算结果出错，应该从哪些方面进行排查和诊断？
7. 数据排布格式 NHWC 和 NCHW 各有什么特点？在模型训练和推理部署中应当如何做出选择？
8. 在哪些情况下应该考虑放弃使用 Shared Memory？比如 Bank Conflict 特别严重或直接走 L2 Cache 反而更快的场景
