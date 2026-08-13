---
title: "第5章：Speculative Decoding"
description: "从 Draft-and-Verify 与拒绝采样出发，理解 Draft 模型、N-gram、Self-Draft 方案及 vLLM 落地方法"
pubDate: 2026-08-12
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "Draft-and-Verify", "N-gram", "Medusa", "EAGLE", "MTP", "vLLM"]
---

## 本章简介

普通自回归解码像一位主编逐字审稿：每写一个 Token，都要把整本模型的权重搬过 GPU 一遍。模型越大，这个“每次只定一个字”的串行过程越昂贵。**Speculative Decoding（投机解码）**换了一种分工：先让便宜的助手连续起草，再让主模型一次审一段；猜对就批量通过，猜错便从第一个错误处改写。

这不是“让小模型替大模型回答”。最终决定权始终属于 Target 模型。采用严格的 speculative sampling 时，拒绝后的残差重采样会校正 Draft 分布，使最终 Token 严格服从 Target 分布；贪心解码时，则只提交与 Target 逐位一致的最长前缀。

本章沿着一条从原理到工程的路线展开：

1. 先弄清 Draft-and-Verify 的一步为何能提交多个 Token。
2. 再用手算理解接受概率、期望接受长度和端到端加速比。
3. 比较独立 Draft 模型与无需额外权重的 N-gram 提案。
4. 理清 Medusa、EAGLE、EAGLE-2/3、MTP 等 Self-Draft 路线。
5. 最后在 vLLM 中用当前的 `--speculative-config` 跑通、观测并压测。

```mermaid
flowchart LR
    A["5.1 原理<br/>为什么正确"] --> B["5.2 Proposer<br/>由谁来猜"]
    B --> C["5.3 Self-Draft<br/>如何自己猜"]
    C --> D["5.4 系统边界<br/>何时值得"]
    D --> E["5.5 vLLM 实战<br/>如何验证"]
```

## 学习目标

完成本章后，你应该能够：

- 写出严格投机采样的接受概率与残差分布，并解释其正确性。
- 根据接受率、投机深度和 Draft/Verify 成本估算加速上限。
- 判断代码补全、开放对话、长上下文和高并发服务分别适合哪种 Proposer。
- 区分“外置小模型”“检索式 N-gram”“多头预测”“特征级自回归”和原生 MTP。
- 识别显存、算力、KV Cache、调度和采样设置对收益的共同约束。
- 使用 vLLM 当前统一配置入口部署 N-gram、Draft Model、EAGLE 与 MTP，并做基线对照。

## 本章小节

- [**5.1 核心原理**](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-核心原理)：Draft-and-Verify、严格拒绝采样、正确性推导与手算。
- [**5.2 Draft 模型与 N-gram**](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与n-gram)：独立小模型的匹配条件，以及零模型检索式提案。
- [**5.3 Self-Draft 方案**](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-self-draft方案)：Medusa、EAGLE 系列与原生 MTP 的结构和取舍。
- [**5.4 收益边界与限制**](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制)：从接受率扩展到 memory、compute、调度与 SLO。
- [**5.5 vLLM 投机解码实战**](/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战)：可核实配置、真实命令、指标观测与实验方法。

> 阅读建议：第一次阅读按顺序完成；已有使用经验的读者也不要跳过 5.1，因为“输出不变”成立需要比“Target 最后验一下”更严格的条件。
