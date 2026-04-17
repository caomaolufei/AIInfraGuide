---
title: "B站 AI Infra 实习 一面"
description: "B站 AI Infra 实习一面面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-15
company: "B站"
tier: "T1"
interviewType: "实习"
round: "一面"
order: 2028
tags: ["算子优化", "高性能计算"]
---

发些面经攒攒人品
1. GPU算子优化通用方法论：profiling定性（memory/compute-bound）
2. 针对性优化（访存连续性/计算简化/block size调整）
3. 项目深挖，问得比较细，具体的优化过的部分都有问到
4. 分布式通信原语理解：all-reduce / all-gather / all-to-all 语义区分
5. 手撕CUDA编程：large array reduce sum 实现（shared memory归约 + 分层kernel设计）
6. 系统基础：进程/线程/协程概念
7. CPU调度粒度（进程级 vs 线程级公平性）
