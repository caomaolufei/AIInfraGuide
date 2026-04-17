---
title: "美团 AI Infra 一面"
description: "美团 AI Infra 一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "美团"
tier: "T1"
interviewType: "未知"
round: "一面"
order: 2018
tags: ["推理优化", "算子优化"]
---

以下为美团 AI Infra 一面及二面考察内容整理。

### 项目经历

1. 针对简历中涉及的全部项目进行深入提问，重点考察项目细节与技术实现

### 基础知识

2. Decoder-only 架构与 Encoder-only 架构各自的优势与不足
3. 概述 LLaMA 模型的主要设计特点
4. CUDA 编程模型的基本概念与层次结构
5. CUDA 内存模型包含哪些层级，各自的特点是什么
6. 使用共享内存时如何应对 bank conflict，有哪些规避策略
7. 使用寄存器时可能遇到哪些问题（如寄存器溢出），应如何处理
8. GPU 多线程与 CPU 多线程在调度机制上有何本质差异
9. 阐述 Flash Attention 的核心原理
10. 阐述 Paged Attention 的设计思路与工作机制
