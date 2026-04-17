---
title: "腾讯 AI Infra 实习 一面 (1)"
description: "腾讯 AI Infra 实习一面面试真题，涵盖推理优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "腾讯"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1038
tags: ["推理优化", "算子优化", "高性能计算"]
---

### 项目经历

1. 介绍项目经验及相关技术细节

### 推理优化

2. 阐述 chunked prefill 的设计动机及其解决的核心问题
3. 在大规模集群中（节点内有 NVLink，节点间部分机器有 RDMA），如何设计分布式推理方案

### 高性能计算

4. 比较 reduce-scatter 与 all-to-all 两种集合通信操作
5. 有哪些方法可以降低 launch kernel 的开销
6. CUDA 编程中 bank conflict 的产生原因及解决方案

### 编程题

7. K 个一组翻转链表
