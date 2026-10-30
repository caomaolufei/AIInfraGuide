---
title: "第3章：Ascend C 算子开发"
description: "从编程模型与 Tiling 开始，掌握 Ascend C 的矢量编程、数据搬运与内存管理、矩阵编程，并通过算子工程与性能分析工具（msprof）构建完整的昇腾算子开发能力"
pubDate: 2026-10-25
category: "ascend"
order: 30
tags: ["Ascend C", "算子开发", "Tiling", "矢量编程", "GEMM"]
---

## 本章简介

如果说 aclnn 是「用现成的算子」，那 **Ascend C** 就是「自己造算子」——它是昇腾为算子开发者提供的编程语言，定位对标 CUDA，但编程范式和 CUDA 有本质区别。

**编程模型**章节先建立核心概念：核函数（Kernel Function）、Tiling（数据切分）、Global Memory 与 UB 的关系。你会在这一章理解昇腾「显式数据搬运」风格的核心原因。

**矢量编程实战**通过一个完整的 Add 算子，带你走通「Tiling → DataCopy → 矢量计算 → 写回」的完整开发流程——这是昇腾算子开发的最小闭环。

**数据搬运与内存管理**深入 DataCopy、队列（EnQue/DeQue）与多缓冲，回答「怎么让数据流高效转起来」，这是算子性能的关键。

**矩阵编程**介绍如何用 Ascend C 的矩阵接口编写 GEMM，理解 AI Cube 的 Tiling 策略，与 CUDA GEMM 优化方法论做对照。

**算子工程与性能分析**讲算子工程的工程化组织方式（CMake、op_host/op_kernel 结构）以及 msprof 分析工具，让算子不仅能写出来，还能调得快。

> 学习前提：建议先掌握第 1、2 章（达芬奇架构、存储层次、AscendCL）。有 CUDA 经验会更轻松，但请格外注意「显式 Tiling + 数据搬运」与「线程模型」的差异。
