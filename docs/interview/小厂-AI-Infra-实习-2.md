---
title: "小厂 AI Infra 实习 (2)"
description: "小厂 AI Infra 实习面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-17
company: "小厂"
tier: "T5"
interviewType: "实习"
order: 6024
tags: ["推理优化", "训练优化", "算子优化"]
---

### 基础知识

1. 预训练与 SFT 阶段在损失函数设计和数据集构建方面有何差异？
2. Transformer 架构中，Megatron 的张量并行（TP）如何对参数矩阵进行切分？MLP 中第一个线性层和第二个线性层分别采用行切分还是列切分？各自对应的通信原语是什么？
3. 简述 DeepSeek 论文中令你印象深刻的技术要点（如 FP8 训练方案）。

### 训练优化

4. 介绍预训练中的流水线并行方案，说明 1F1B 调度策略与 DualPipe 的设计原理。
5. 详细阐述大模型强化学习（RL）的完整流程，涉及哪些模型角色？PPO 与 GRPO 的核心区别是什么？
6. 在 RL 训练中，Rollout 阶段的耗时占比约为多少？Policy 模型的 MFU 大致为多少？请给出 MFU 的计算公式以及 6Nd 公式的含义。
7. RL 中的 Rollout 阶段有哪些常见优化手段（如 Rollout 量化、异步 Rollout 等）？
8. 在 RL 训练流程中，如何将预训练权重同步至推理引擎？

### 推理优化

9. 介绍 vLLM 或 SGLang 中 Continuous Batching 的工作机制。
