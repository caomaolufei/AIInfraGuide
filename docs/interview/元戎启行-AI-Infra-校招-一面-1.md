---
title: "元戎启行 AI Infra 校招 一面 (1)"
description: "元戎启行 AI Infra 校招一面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "元戎启行"
tier: "T4"
interviewType: "校招"
round: "一面"
order: 5020
tags: ["算子优化"]
---

### 一面

#### 基础知识

1. 是否定义过自定义的 MLIR Dialect？
2. 项目中是否参考了 torch-mlir 的设计？
3. 是否考虑过动态图的处理问题？
4. 转换为 TOSA 和 Tensor Dialect 之后，继续 Lower 到哪些 Dialect？
5. One-shot Bufferization 与基于 Dialect 的 Bufferization 有何区别？
6. 是否了解 LLVM 中 isa 和 dyn_cast 的用法与原理？

#### 项目经历

7. 相关项目与实习经历深入讨论。

#### 编程题

8. 给定一个计算图，计算运行该计算图所需的最小内存。

### 二面

#### 基础知识

9. 静态图与动态图的区别是什么？
10. MLIR 中如何处理 In-place 操作？
11. 是否有 CUDA Kernel 开发经验？

#### 编程题

12. 实现拓扑排序算法。
