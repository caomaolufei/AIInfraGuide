---
title: "阿里巴巴 AI Infra 二面"
description: "阿里巴巴 AI Infra 二面面试真题，涵盖 TP、CUDA Graph、Split-K 确定性、NCCL 与 PD 分离"
pubDate: 2026-09-18
company: "阿里巴巴"
tier: "T0"
interviewType: "未知"
round: "二面"
order: 1056
tags: ["推理优化", "高性能计算"]
---

### 并行与稳定性

1. TP 并行结合 CUDA Graph 时，如何排查稳定性与超时问题？
2. 如何判断 CUDA Graph 是故障的触发条件还是根本原因？怎样构造最小复现？
3. CUDA Graph 与 TP 组合时，缓冲区和同步状态需要检查什么？
4. 动态 shape 下如何选择计算路径，并设计回退策略？

### 算子与通信

5. 确定性要求会怎样影响 Router GEMM 和 Split-K 策略？Split-K 为什么可能破坏确定性？
6. Decode 阶段如何减少 CPU 与 GPU 之间的同步开销？
7. NCCL 中 Ring AllReduce 与 Tree（Double Binary Tree）AllReduce 的工作原理和适用场景分别是什么？

### PD 分离

8. Prefill 节点产生的 KV Cache 如何高效同步到 Decode 节点？
9. PD 分离后会增加哪些传输、同步和调度成本？
