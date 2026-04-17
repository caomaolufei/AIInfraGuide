---
title: "阿里巴巴 国际 AI Infra 实习 (1)"
description: "阿里巴巴 AI Infra 实习面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "阿里巴巴"
tier: "T0"
interviewType: "实习"
order: 1013
tags: ["推理优化", "算子优化"]
---

## 基础知识

1. 阐述 FlashAttention 的核心原理，包括 Online Softmax 的实现方式。FlashAttention V1、V2、V3 之间有哪些关键改进？FlashDecoding 的原理是什么？
2. 当前推理优化的主要思路和方法有哪些？
3. 是否使用过 vLLM 或 SGLang？请说明其工作原理。
4. 什么是 Dynamic Batching？其适用场景是什么？
5. 将数据加载到 Shared Memory 相比直接从 HBM 读取为什么更快？
6. 什么是 Shared Memory 的 Bank Conflict？如何避免？

## 项目经历

7. 介绍你的实习经历及主要工作内容。
8. 针对项目内容展开深入提问。

## 编程题

9. 编写一个 Reduce 操作：先使用 Block 级实现，再使用 Warp Shuffle 进行优化，并讨论是否可以进一步优化。
