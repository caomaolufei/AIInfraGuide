---
title: "华为 AI Infra 实习 (2)"
description: "华为 AI Infra 实习面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "华为"
tier: "T1"
interviewType: "实习"
order: 2002
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

### 推理优化

1. 部署模型与推理模型时所需的参数量分别如何估算？
2. 模型推理中 Prefill 阶段与 Decoding 阶段的区别是什么？KV Cache 的大小如何计算？
3. 模型中的超参数 Temperature 的数学原理是什么？Temperature、Top-k、Top-p 三个参数的作用顺序是怎样的？
4. Softmax 函数的定义与计算过程是什么？

### 训练优化

5. DeepSpeed 框架中 ZeRO Stage 1、2、3 分别优化了什么？各阶段的区别是什么？
6. 假设需要对矩阵乘法 A x B 进行分布式计算，给定 4 张卡，请描述如何分配矩阵 A、B 的参数，并通过卡间通信实现显存节省。

### 高性能计算

7. 对通信算子有哪些了解？常见的集合通信原语有哪些？
