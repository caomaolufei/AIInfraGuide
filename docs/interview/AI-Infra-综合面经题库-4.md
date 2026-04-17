---
title: "AI Infra 综合面经题库 (4)"
description: "AI Infra 面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
order: 6
tags: ["算子优化", "高性能计算"]
---

## 项目经历

1. 项目深入考察

## 基础知识

2. 是否有过 kernel 级别的优化经验？例如使用 CuTe DSL 或手写 CUDA 实现算子融合，请介绍具体做法
3. 在进行 kernel fusion 时，通常倾向于采用哪种方式实现？
4. 是否遇到过算子融合后性能反而下降的情况？分析其原因
5. 在编写 CUDA 程序时，是否关注过底层实现细节？例如 Hopper 架构中的 Warp Specialization 机制及其底层实现原理
6. 在部署或训练大规模模型时，是否使用过底层调试工具？当千卡规模出现 NCCL Timeout 时，通常如何定位与解决？
7. 是否针对 RL 场景下的 MoE 模型做过相关优化？
8. 在推理服务上线前，做过哪些方面的性能优化工作？
9. 是否尝试过利用 Agent 自动生成 CUDA kernel？具体是如何实现的？
