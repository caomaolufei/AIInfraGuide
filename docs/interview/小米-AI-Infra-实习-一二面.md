---
title: "小米 AI Infra 实习 一二面"
description: "小米 AI Infra 实习一二面面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "小米"
tier: "T1"
interviewType: "实习"
round: "一二面"
order: 2024
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

### 一面

#### 项目经历

1. 实习期间的主要工作内容介绍。
2. 千卡规模训练项目的技术方案与实施细节。
3. 竞赛项目经历介绍。

#### 基础知识

4. Python 中深拷贝与浅拷贝的区别。
5. C++ 三种智能指针（unique_ptr、shared_ptr、weak_ptr）的特点与使用场景。
6. 写时拷贝（Copy-on-Write）的实现原理。
7. 零拷贝（Zero-Copy）的实现原理。

#### 编程题

8. 实现矩阵转置。

### 二面

#### 训练优化

9. 大模型分布式训练的完整流程及并行策略选择依据。
10. 介绍常见的分布式并行策略。
11. 张量并行（TP）为何存在按行和按列两种切分方式？各自对应的含义是什么？
12. Megatron 中序列并行（SP）的设计原理与实现方式。

#### 基础知识

13. Transformer 架构中包含哪些层和算子？
14. Encoder 与 Decoder 的结构特点及差异。
15. FlashAttention 的核心思想与实现机制。
16. Online Softmax 的计算原理。

#### 算子优化

17. CUDA 中 Block 是软件概念还是硬件概念？
18. CUDA 常见的优化方法有哪些？
19. 访存优化有哪些具体策略？
20. 计算与访存如何实现重叠（Overlap）？
21. L1 Cache 与 L2 Cache 的区别。
22. 共享内存与 L1 Cache 的关系与差异。

#### 高性能计算

23. C++ 中三种智能指针的区别，shared_ptr 存在的设计动机是什么？
24. AllReduce 操作的实现原理，有哪些常见的实现方式？
25. Ring AllReduce 的通信量分析。
26. Tree AllReduce 相较于 Ring AllReduce 有何优势？

#### 编程题

27. 合并两个有序链表。
