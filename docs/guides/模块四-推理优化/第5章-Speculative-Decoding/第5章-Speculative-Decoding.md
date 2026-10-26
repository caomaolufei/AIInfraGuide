---
title: "第5章：Speculative Decoding"
description: "理解 Draft + Verify 与保分布采样，比较独立 Draft、N-gram/Suffix、Medusa/EAGLE/MTP，并掌握收益边界与 vLLM 对照实验"
pubDate: 2026-04-16
updatedDate: 2026-10-26
category: "inference-optimization"
order: 34
tags: ["Speculative Decoding", "投机解码", "Medusa", "EAGLE", "MTP", "N-gram"]
---

## 📖 本章简介

Decode 慢，是因为大模型每读一遍权重只写出一个 Token。投机解码换了一种分工：让便宜的 Draft 先写一段草稿，再让 Target 一次前向批量验收，猜对的直接收下，猜错的由 Target 改写。只要验收与修正的规则做对，最终输出和 Target 自己逐个生成的分布完全一样。

本章按"为什么对、怎么猜、什么时候值、怎么测"四步展开：

- **原理**：一轮 Draft + Verify 的流程、接受概率与残差分布、无损证明，以及 KV Cache 怎样回退被拒的草稿。
- **草稿来源**：独立小模型、并行 Draft、N-gram 与 Suffix Decoding；Medusa、EAGLE 系列等复用 Target 特征的方案；以及模型自带的 MTP 模块。
- **收益边界**：用"每轮提交多少 Token / 每轮花多久"建立性能账本，分析投机长度、并发、长上下文、量化和流式输出的影响。
- **实战**：在 vLLM 上用同一个 Target 依次对比普通 Decode、N-gram、独立 Draft 与 EAGLE-3，同时采集接受指标、输出内容和延迟吞吐。

> ⏰ **信息时效**：原理部分与框架无关；vLLM 相关的参数、方法名与指标基于 v0.32.0（2026-10-18 发布）。投机解码的配置项在近几个版本中变动频繁，换版本时请对照官方文档核对。

## 📑 本章小节

- [**5.1 投机解码核心原理：先猜一段，再让大模型验收**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/)：Draft + Verify 流程、保分布证明、one-hot Draft、首个拒绝与 KV 回退
- [**5.2 Draft 模型与无模型方案：怎样便宜地猜得更准**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/)：独立小模型与 PARD、Tokenizer 兼容、N-gram/Suffix，以及接受率、平均接受长度与候选覆盖的统一口径
- [**5.3 Medusa、EAGLE 与 MTP：复用大模型信息生成更好的草稿**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/)：Medusa 多头、Tree Attention、EAGLE-1/2/3、DeepSeek-V3 MTP、DFlash 与 Block Verification
- [**5.4 投机解码的收益边界：接受率高，为什么仍可能变慢**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/)：提交长度与轮耗时、投机长度的收益递减、Dynamic SD、并发、量化与流式延迟
- [**5.5 vLLM 投机解码实战：配置、接受率与性能一起测**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/)：基线、N-gram、Draft、EAGLE-3 对照，逐请求接受指标与 Benchmark
