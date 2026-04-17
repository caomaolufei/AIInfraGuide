---
title: "飞腾 AI Infra 实习 一面"
description: "飞腾 AI Infra 实习一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "飞腾"
tier: "T3"
interviewType: "实习"
round: "一面"
order: 4004
tags: ["推理优化", "算子优化"]
---

### 基础知识

1. 进程与线程的核心区别是什么？Cache 的层级结构及常见替换策略有哪些？
2. IEEE 浮点标准中 FP16、FP32、FP64 各自的位宽分配方式是怎样的？
3. 快速排序的执行流程是什么？堆的基本性质有哪些？拓扑排序适用于哪些场景？

### 推理与算子优化

4. 请阐述 Transformer 架构中 Decoder-only 结构的特点，包括 QKV 的生成过程以及位置编码的嵌入时机。
5. RMSNorm 的计算公式是什么？其计算访存特性如何？可以从哪些角度进行优化（如负载均衡、Double Buffer、指令替换）？
6. Softmax 的数值稳定性问题如何处理？Online Softmax 的实现原理是什么？
7. 矩阵乘法与反量化融合算子在内存方面的优化策略有哪些？
8. 稀疏矩阵 SpMV 运算中如何实现负载均衡与带宽优化？

### 项目经历

9. 项目深入考察。

### 工具与工程

10. Git 中拉取远程分支的操作方式有哪些？fetch+checkout 与 pull 的区别是什么？
