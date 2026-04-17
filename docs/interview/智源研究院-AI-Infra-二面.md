---
title: "智源研究院 AI Infra 二面"
description: "智源研究院 AI Infra 二面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "智源研究院"
tier: "T5"
interviewType: "未知"
round: "二面"
order: 6029
tags: ["推理优化", "算子优化"]
---

## 基础知识

1. 请描述 CUDA 的计算模型，包括 block 和 grid 的配置方式及其对算子性能的影响。
2. 同一 block 内不同 warp 之间如何进行数据通信？
3. 请说明 CUDA reduce 操作的实现思路。
4. 在 CUDA 中实现 softmax 时，warp 级处理与 block 级处理有何区别？
5. 静态图与动态图的概念分别是什么？动态 shape 场景下如何处理？
6. 推理框架中计算图、运行图和内存管理分别承担什么职责？如何设计一个推理框架的整体架构？
7. 常见的图优化技术有哪些？

## 算子优化

8. 请对比 FlashAttention v1 与 v2 的核心改进点。
