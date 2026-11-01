---
title: "第5章：Speculative Decoding"
description: "理解投机解码的核心原理（Draft + Verify）、Self-Draft 方案（Medusa/EAGLE-2）及其收益边界与限制"
pubDate: 2026-04-16
updatedDate: 2026-10-09
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "N-gram", "Rejection Sampling"]
---

## 本章简介

自回归解码的串行依赖限制了 Decode 阶段的效率。Speculative Decoding 通过“先猜后验”，利用已知候选并行计算多个位置的条件分布，减少 Target 的串行调用轮数，同时保留因果关系。本章原理与 vLLM 中的实际配置并重。

**核心原理**详解 Speculative Sampling 框架：Draft 模型快速猜测多个 Token，Target 模型并行验证。通过接受概率与拒绝后的残差采样，证明完整输出过程保持 Target 分布。

**Draft 模型与无模型方案**讨论独立小模型的选择（大小、架构、能力匹配），以及无需额外模型的 N-gram / Suffix Decoding（基于 Prompt 与历史输出做前缀匹配），并分析 Acceptance Rate 对加速比的影响。

**Self-Draft 方案**覆盖 Medusa（多 Decoding Head 预测多 Token）、EAGLE-2 的动态 Draft Tree 与置信度，以及 EAGLE-3 的多层特征融合和训练改进，并介绍从 Token 级到 Block 级联合验证的演进。

**收益边界与限制**比较代码生成与开放对话中的接受情况，结合完整轮耗时判断实际加速；同时分析候选 KV 的保留与回退、量化后的正确性基准，以及 Continuous Batching 的调度复杂度。

**在 vLLM 中配置投机解码实战**：介绍 vLLM 对 N-gram、EAGLE、Draft 模型等投机解码方案的配置方式，并对比代码生成 vs 开放对话的 Acceptance Rate 差异。

## 本章小节

- **[5.1 核心原理](./第5章-Speculative-Decoding/5.1-投机解码核心原理.md)**：Draft + Verify、Rejection Sampling 正确性证明
- **[5.2 Draft 模型与 N-gram/Suffix 方案](./第5章-Speculative-Decoding/5.2-Draft模型与N-gram-Suffix方案.md)**：模型选择与无模型投机
- **[5.3 Self-Draft 方案](./第5章-Speculative-Decoding/5.3-Self-Draft与Draft-Tree.md)**：Medusa、EAGLE-2/3、Draft Tree
- **[5.4 收益边界与限制](./第5章-Speculative-Decoding/5.4-收益边界与限制.md)**：接受率、精度风险、调度复杂度
- **[5.5 vLLM 投机解码实战](./第5章-Speculative-Decoding/5.5-vLLM投机解码实战.md)**：配置方式与接受率实测
