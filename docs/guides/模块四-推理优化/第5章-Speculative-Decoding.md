---
title: "第5章：Speculative Decoding"
description: "理解 Draft + Verify 与保分布采样，比较独立 Draft、N-gram/Suffix、Medusa/EAGLE-2/3，并掌握收益边界与 vLLM 对照实验"
pubDate: 2026-04-16
updatedDate: 2026-10-07
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "N-gram", "Rejection Sampling"]
---

## 本章简介

自回归解码的串行瓶颈是 Decode 阶段效率低的根本原因。Speculative Decoding 通过"先猜后验"打破这一瓶颈。本章原理与 vLLM 中的实际配置并重。

**核心原理**详解 Speculative Sampling 框架：Draft 模型快速猜测多个 Token，Target 模型并行验证。通过接受规则与拒绝后的残差采样保证正确性——数学上证明最终输出服从 Target 分布；单独条件在“已接受”事件上的 Token 分布一般不同。

**Draft 模型与无模型方案**讨论独立小模型的选择（大小、架构、能力匹配），以及无需额外模型的 N-gram / Suffix Decoding（基于 Prompt 与历史输出做前缀匹配），并分析 Acceptance Rate 对加速比的影响。

**Self-Draft 与辅助草稿方案**覆盖 Medusa 的多 Decoding Head、EAGLE 的特征草稿、EAGLE-2 的动态 Draft Tree，以及 EAGLE-3 的多层特征融合与训练改进；同时区分 Tree Attention、Token 级验收和 Block 级联合验证的职责与保证。

**收益边界与限制**通过接受前缀与完整轮耗时建立性能账本，分析代码重复、开放对话、并发和长上下文下的不同机会；接受率高不保证提速，还要核对量化后的 Target 基准、缓存容量和 Continuous Batching 的调度成本。

**在 vLLM 中配置投机解码实战**：按 v0.31.0 的配置与指标定义，依次比较普通 Decode、N-gram 与独立 Draft，介绍 Suffix/EAGLE 的接入条件，并提供代码与对话的接受指标、性能和正常生成质量采集流程。

## 本章小节

- [**5.1 投机解码核心原理：先猜一段，再让大模型验收**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/)：Draft + Verify、保分布证明、首个拒绝与 KV 回退
- [**5.2 Draft 模型与无模型方案：怎样便宜地猜得更准**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/)：独立小模型、Tokenizer、N-gram/Suffix 与候选覆盖
- [**5.3 Medusa 与 EAGLE：复用大模型信息生成更好的草稿**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/)：多头、动态树、EAGLE-3、Tree Attention 与验收边界
- [**5.4 投机解码的收益边界：接受率高，为什么仍可能变慢**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/)：提交长度、轮耗时、批处理、量化与流式延迟
- [**5.5 vLLM 投机解码实战：配置、接受率与性能一起测**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/)：服务配置、code/chat 指标采集与基线对照
