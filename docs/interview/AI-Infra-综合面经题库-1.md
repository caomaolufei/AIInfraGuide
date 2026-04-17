---
title: "AI Infra 综合面经题库 (1)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-16
company: "综合"
tier: "综合"
interviewType: "未知"
order: 3
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

1️⃣算法题：手撕内存池（要求支持类似new Foo[], delete []功能)，c++如何比较两个float是否相等，LRU，岛屿个数，二叉树的层序遍历，hamming weight，k-coverage intervals

2️⃣torch手撕题: LoRA adapter

3️⃣cuda手撕题：
1- 支持torch broadcast的4D tensor的elementwise mul
2- A: (1, 256),  B: (256, 128),  C: (128, 256) ，计算 (A * B) * C
3- Embedding Sparse Feature Pooling：A是 100 万个离散 ID（0~999），B是 100 万个float，计算个长度为1000的float数组C，C[i] = \sum_{j s.t. A[j] = i} B[j]

4️⃣ai infra或算法八股：
1- llm的知识蒸馏放在预训练做是否合适
2- Hopper TMA的优点，调用方式，是否需要经过L1
3- Flash Attention v2为什么外层对Q循环，Flash Decoding的combine kernel耗时占比大概是多少
4- Mooncake kv-cache centric的PD分离
5- Dit的推理框架设计思路和LLM的有什么异同
6- 分析MLA decode的计算访存比，它和seqlen、batch size是否相关
7- diffusion model的训练和推理步骤，推理num_inference_steps为40时，为什么训练的timesteps仍要设成1000
8- 介绍dLLM，如何看待它和AR的区别
9- torch.repeat 和 torch.expand的区别
10- torchrun的启动参数有哪些，如何在Linux上批量kill包含torchrun的进程
