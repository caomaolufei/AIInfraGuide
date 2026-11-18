---
title: "阿里巴巴 云 AI Infra 一面"
description: "阿里巴巴云大模型推理性能优化一面面试真题，涵盖算子栈、FlashAttention、海光 HIP/DCU 与 CUDA 生态迁移"
pubDate: 2026-09-14
company: "阿里巴巴"
tier: "T0"
interviewType: "未知"
round: "一面"
order: 1055
tags: ["推理优化", "算子优化", "高性能计算"]
---

### 基础知识

1. 算子在 AI 推理的技术栈和优化栈中处于什么位置？框架层和算子层分别负责什么？

### 算子与异构算力

2. 海光的软件生态是 HIP 吗？HIP 的全称是什么？
3. 海光 DCU 和 NVIDIA GPU 在架构、软件栈和算子适配上有什么区别？
4. 将 CUDA 算子迁移到 HIP/DCU 时，除了 API 还要检查哪些层面？
5. FlashAttention 的核心实现思路是什么？项目中实际改动了哪些部分？
6. 如何评估算子优化和 AI 辅助开发带来的收益？

### 编程题

7. 给定一个大部分元素成对出现的序列，其中只有一个元素只出现一次，找出该元素的位置。
