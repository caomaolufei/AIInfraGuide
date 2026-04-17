---
title: "AI Infra 校招 (1)"
description: "AI Infra 校招面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "校招"
order: 2
tags: ["算子优化", "高性能计算"]
---

适用岗位：异构计算 / AI 框架研发 / 高性能计算 / 模型部署 / 算法优化 / 算子研发

## 基础知识

1. CUDA 编程中的软件层级模型（Grid、Block、Thread）是如何组织的？
2. CUDA 的内存层级结构包含哪些类型（全局内存、共享内存、寄存器等）？各自的特点是什么？
3. CPU 与 GPU 在架构设计上有哪些本质差异？GPU 为何更适合并行计算？
4. 如何理解 SM（Streaming Multiprocessor）与 SP（Streaming Processor）的关系？
5. CUDA Stream 的概念是什么？同步流与异步流有何区别？
6. GPU 的 L1/L2 缓存各自承担什么角色？
7. 使用共享内存时需要注意哪些事项（线程同步、bank conflict 等）？
8. 针对一个 CUDA kernel 进行性能优化，可以从哪些维度入手？

## 编程题

1. 实现矩阵乘法 kernel
2. 实现 Softmax 归约
3. 实现 NCHW 到 NHWC 的数据格式转换
4. 给定长度为 n 的数组（元素范围 0~256），统计每个元素出现的频次并存入另一数组
5. 将数组中奇数下标的元素移至左侧、偶数下标的元素移至右侧，要求原地操作
