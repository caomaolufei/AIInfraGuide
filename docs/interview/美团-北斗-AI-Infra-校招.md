---
title: "美团 北斗 AI Infra 校招"
description: "美团 AI Infra 校招面试真题，涵盖推理优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "美团"
tier: "T1"
interviewType: "校招"
order: 2021
tags: ["推理优化", "算子优化", "高性能计算"]
---

以下为美团北斗 AI Infra 校招面试考察内容。

### 基础知识

1. 阐述 Transformer 的整体架构，与传统 RNN（循环神经网络）相比具备哪些优势
2. Transformer 中参数主要分布在哪些模块？参数量最大的部分是什么？计算量最大的部分是什么，原因是什么
3. 说明 GPU 中 CUDA Core 与 Tensor Core 的区别，列举常用 GPU 型号及其显存容量与显存带宽
4. 介绍大模型量化的基本原理，并描述量化算子的实现方式
5. 详细说明 Prefill-Decode 分离（PD 分离）的设计思路
6. 详细说明 Paged Attention 的工作机制

### 编程题

7. LeetCode 25：K 个一组翻转链表
8. 编写 CUDA 算子：实现前缀和的基础版本，并阐述可行的优化方法
9. 编写 CUDA 算子：实现 GEMM 的基础版本，并阐述可行的优化方法
