---
title: "AI Infra 一面"
description: "AI Infra 一面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
round: "一面"
order: 1
tags: ["算子优化"]
---

1. 请阐述 Roofline 模型的基本原理，如何利用该模型判断一个算子是否已达到计算瓶颈？
2. 当训练或推理所使用的 GPU 卡数成倍扩展时，系统最可能在哪些环节出现瓶颈？请分析原因并给出相应的优化或缓解思路。
3. GEMM 计算是否一定属于计算瓶颈型算子？若需要对其进行优化，整体思路是什么？
4. 在性能调优过程中，如何定位瓶颈并进行检测？你通常使用哪些方法或工具？
5. 请介绍你对 Flash Attention 的理解。
6. 在 C++ 中，若数组越界写入导致其他数据结构被破坏，且现场保留了 coredump 文件，应如何排查该问题？
7. 编程题：手写包含 GQA（Grouped Query Attention）的 Attention 模块实现。
