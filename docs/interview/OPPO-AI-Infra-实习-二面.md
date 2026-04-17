---
title: "OPPO AI Infra 实习 二面"
description: "OPPO AI Infra 实习二面面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-16
company: "OPPO"
tier: "T1"
interviewType: "实习"
round: "二面"
order: 2029
tags: ["算子优化", "高性能计算"]
---

发一下问题给大家参考，攒攒人品！有面试过同岗的朋友欢迎评论区交流
项目拷打
1. 数据布局详解：NHWC vs NCHW：在训练/推理中怎么选？
2. 何时应该关闭 Shared Memory？（当出现 Bank Conflict 严重或收益不如直接访问 L2 时）
3. 特定 Shape 导致使用 Shared Memory 时结果异常如何排查
4. Thread/Warp/Block/SM/Grid 的映射关系
5. 如何确定最优线程数？
6. 异步设计：CUDA Stream 的使用前提（无内存访问重叠）
7. 算子融合决策，什么场景适合融合
