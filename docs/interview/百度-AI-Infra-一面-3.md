---
title: "百度 AI Infra 一面 (3)"
description: "百度 AI Infra 一面面试真题，涵盖训练优化、高性能计算等方向"
pubDate: 2026-04-17
company: "百度"
tier: "T0"
interviewType: "未知"
round: "一面"
order: 1020
tags: ["训练优化", "高性能计算"]
---

## 强化学习

1. Trust Region 方法与 PPO 之间的关系
2. PPO 属于 on-policy 还是 off-policy？为什么需要引入 importance sampling？
3. PPO 中 clip 机制在优势函数 A 为正值和负值时分别如何限制上下界？
4. PPO 的损失函数如何计算？广义优势估计（GAE）的计算方式及 lambda 参数对方差和偏差的影响
5. GRPO 的损失计算方式，序列级别损失如何分配到每个 token？序列级别平均与批次级别平均的区别
6. 其他 GRPO 变体（如 DAPO、GSPO、GFPO 等）的特点
7. Agentic RL 的基本概念与应用

## 分布式训练

8. 训练过程中优化器状态、梯度、模型参数各自的显存占比
9. FSDP 与 DeepSpeed ZeRO Stage 1/2/3 的对比

## 项目经历

10. 项目介绍及技术方案，如何将大语言模型与具体应用场景结合，评测方式与指标结果

## 编程题

11. 二叉树的层序遍历，并记录每个节点所在的层级（LeetCode 102 变体）
