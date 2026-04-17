---
title: "AI Infra 综合面经题库 (1)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
order: 3
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

## 基础知识

1. Hopper 架构 TMA 的优势是什么？其调用方式如何？数据传输是否需要经过 L1 缓存？
2. Flash Attention v2 中外层循环为何选择对 Q 进行遍历？Flash Decoding 的 combine kernel 耗时占比大约是多少？
3. 如何分析 MLA decode 的计算访存比？该比值与序列长度、batch size 是否存在关联？
4. Mooncake 中以 KV-Cache 为中心的 PD 分离方案的设计思路
5. DiT 推理框架的设计与 LLM 推理框架有哪些异同？
6. 大语言模型的知识蒸馏是否适合在预训练阶段进行？
7. Diffusion Model 的训练与推理步骤分别是什么？当推理 num_inference_steps 设为 40 时，为何训练的 timesteps 仍需设置为 1000？
8. 请介绍 dLLM（离散化大语言模型），它与自回归（AR）模型有何本质区别？
9. torch.repeat 与 torch.expand 在功能和内存行为上有何差异？
10. torchrun 的启动参数有哪些？在 Linux 环境下如何批量终止包含 torchrun 的进程？

## CUDA 编程题

1. 实现支持 torch broadcast 语义的 4D tensor elementwise 乘法
2. 给定 A: (1, 256), B: (256, 128), C: (128, 256)，计算 (A * B) * C
3. Embedding Sparse Feature Pooling：A 为 100 万个离散 ID（范围 0~999），B 为 100 万个 float，计算长度为 1000 的数组 C，其中 C[i] = sum of B[j] for all j where A[j] = i

## PyTorch 编程题

1. 实现 LoRA Adapter

## 算法题

1. 实现内存池（需支持类似 new Foo[] / delete[] 的功能）
2. C++ 中如何比较两个 float 是否相等？
3. 实现 LRU 缓存
4. 岛屿数量问题
5. 二叉树的层序遍历
6. 计算 Hamming Weight
7. K-Coverage Intervals 问题
