---
title: "第3章：nano-vllm 源码解读"
description: "以约 1200 行的极简推理引擎 nano-vllm 为主线，逐模块读懂连续批处理、PagedAttention、FlashAttention、张量并行与 CUDA Graph 在真实代码里的组织方式"
pubDate: 2026-10-12
category: "inference-optimization"
order: 32
tags: ["nano-vllm", "源码解读", "推理引擎", "PagedAttention", "连续批处理", "CUDA Graph"]
---

## 📖 本章简介

第 2 章把推理引擎的核心技术逐个拆开讲了：PagedAttention 怎么管显存、Continuous Batching 怎么填满 GPU、Prefix Cache 怎么复用前缀、Chunked Prefill 怎么统一调度。这些技术点在一个真实引擎里如何拼成一条流水线，是本章要回答的问题。

本章的方法是读代码。样本是 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm)，由 Xingkai Yu（GitHub ID `GeeeekExplorer`，主页标注 DeepSeek、南京大学）开源的极简 LLM 推理引擎。项目约 1200 行 Python，覆盖了配置、调度、KV 管理、模型前向和采样的完整链路，并实现了前缀缓存、张量并行、`torch.compile` 和 CUDA Graph。它和 vLLM 的关系，类似 nanoGPT 和大规模训练框架的关系：代码量小到能一次读完，同时保留了工业系统的主干。

读完本章，你应该能对着源码目录说清楚一个请求从字符串进来、到第一个 token 输出、再到整段生成结束，中间每一层代码做了什么，并能把这些实现与 vLLM 的设计对应起来。

> ⏰ **信息时效**：本章代码分析基于 nano-vllm `main` 分支 2026-04-26 的提交（`pyproject.toml` 版本 0.2.0；截至 2026-11-01 仓库约 16.2k stars，MIT 协议），示例模型为 `Qwen/Qwen3-0.6B`。源码仍在演进，关键标识符若有出入请以仓库最新版本为准。

## 📑 本章小节

- **3.1 项目全景与四层架构**：项目定位、目录结构，以及「接口层 → 引擎层 → 调度层 → 执行层」的分层设计与端到端数据流
- **3.2 配置、入口与一次 generate 的全流程**：`Config` / `SamplingParams`、`LLM` 与 `LLMEngine` 的关系、README 示例用法，以及 `generate → add_request → step → postprocess` 主循环
- **3.3 模型骨架与核心算子**：词表并行 Embedding/LMHead、RMSNorm、RoPE、SwiGLU、并行 Linear 族与张量并行权重切分、Attention 的 KV 写入与 FlashAttention 两条路径、Sampler 温度采样、Qwen3 的组装
- **3.4 KV Cache 与 BlockManager**：`Block` 结构、PagedAttention 的分块思想、基于哈希的前缀缓存、块的分配与回收，以及 `Sequence` 的生命周期
- **3.5 Scheduler 与连续批处理**：Waiting/Running 队列、Prefill/Decode 的调度决策、Token Budget、抢占与重计算
- **3.6 ModelRunner、CUDA Graph 与性能基准**：输入张量准备、全局 Context 注入、KV Cache 显存分配、CUDA Graph 的捕获与回放，以及官方性能基准与 vLLM 对比
