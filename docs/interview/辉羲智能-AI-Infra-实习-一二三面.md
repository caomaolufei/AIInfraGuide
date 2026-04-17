---
title: "辉羲智能 AI Infra 实习 一二三面"
description: "辉羲智能 AI Infra 实习一二三面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "辉羲智能"
tier: "T4"
interviewType: "实习"
round: "一二三面"
order: 5003
tags: ["算子优化"]
---

### 一面

1. 公司介绍与基本情况沟通。

### 二面

#### 算子优化

2. 什么是 bank conflict？产生的原因是什么？有哪些解决方案？
3. 编写算子时如何最大化利用缓存？如何根据 L1、L2 缓存容量进行数据分块？
4. 什么是线程束分歧（warp divergence）？它对性能有什么影响？
5. blockDim.x 与 gridDim.x 的最大值分别是多少？
6. 手写 CUDA 矩阵乘法算子（naive 版本），并说明后续优化方向及最佳分块大小的确定方法。

#### 基础知识

7. 请描述 CPU 与 GPU 各自的架构特点与设计差异。
8. 对 CUDA 中 grid、block、thread 三级层次结构的理解。
9. 常见卷积算法有哪些？各自的优缺点是什么？

#### 项目经历

10. 项目经历深入考察。

### 三面

#### 算子优化

11. 共享内存（Shared Memory）与硬件缓存（Cache）的区别是什么？
12. Tensor Core 与 CUDA Core 在矩阵乘法加速方面哪个更快？Tensor Core 的工作原理是什么？
13. 手写 CUDA Softmax 算子。

#### 基础知识

14. Transformer 架构的整体结构是怎样的？
15. Softmax 算法在深度学习中有哪些典型应用？
