---
title: "百度 AI Infra 实习 一面 (1)"
description: "百度 AI Infra 实习一面面试真题，涵盖推理优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "百度"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1028
tags: ["推理优化", "算子优化", "高性能计算"]
---

## 大模型推理

1. FlashAttention 的核心原理及其数学推导过程
2. FusedAttention 的优化策略有哪些？
3. RMSNorm 相比 LayerNorm 性能提升的原因是什么？
4. Llama 模型中包含多少个全连接层？
5. 描述 Llama2 的推理流程，每一层包含哪些算子？
6. 设计一种比 cudaMalloc 更灵活高效的显存分配方案（cudaAllocator）

## C++ 基础

7. C++11 引入了哪些重要新特性？
8. 智能指针的分类及各自特点
9. unique_ptr 如何保证所有权的唯一性？
10. shared_ptr 的引用计数归零后何时触发析构？
11. 类的成员函数是否可以定义为模板函数？
12. 左值与右值的定义及区别

## CUDA 编程

13. CUDA 提供了哪几种编程模型或方式？
14. Tensor Core 与 CUDA Core 的功能差异

## 编程题

15. 最长连续序列（LeetCode 128）
16. 至多包含 K 个不同字符的最长子串（LeetCode 340）
