---
title: "第3章：深入 vLLM 架构与源码"
description: "深入 vLLM 的整体架构、V1 引擎设计、调度器源码与关键配置调优，并横向对比 SGLang、TensorRT-LLM"
pubDate: 2026-08-12
category: "inference-optimization"
order: 32
tags: ["vLLM", "V1引擎", "Scheduler", "源码导读", "SGLang", "TensorRT-LLM"]
---

## 本章简介

vLLM 是本模块的实例主线。前两章讲的核心技术，本章落到 vLLM v0.27.1（2026-08-12）的官方架构、配置与源码中，看它们如何被组织、调度和调优；最后用统一维度对比 SGLang 与 TensorRT-LLM，帮助你建立可验证的框架选型方法。

**快速入门**从安装、离线批量推理到 OpenAI 兼容服务部署，用最短路径跑通第一个 vLLM 推理服务。

**整体架构**拆解 vLLM 的分层设计：`LLMEngine` / `AsyncLLM` 入口、`EngineCore` 执行循环、`Scheduler` 调度器、`KVCacheManager` 显存管理、`Worker` / `ModelRunner` 模型执行，理清一个请求从进入到返回的完整数据流。

**V1 引擎**通过多进程隔离让 Tokenize/Detokenize 等 CPU 任务与核心循环重叠，用统一 Token 预算表达 Prefill、Decode 与推测 token，以统一 KV block 池支撑默认 Prefix Cache，并通过 Persistent Batch 减少重复状态准备。性能提升取决于模型、硬件和负载，本章不使用脱离测试条件的统一倍数。

**调度器源码导读**深入 Waiting/Running 队列、Token Budget 分配、抢占（Preemption）与重计算（Recompute）机制，看懂 vLLM 每一步 step 到底做了什么。

**关键配置调优**讲清 `gpu-memory-utilization`、`max-num-seqs`、`max-num-batched-tokens`、`block-size` 等参数如何影响吞吐、延迟和显存，给出调参决策思路。

**框架横向对比**从模型支持、缓存、结构化输出、硬件、部署与运维成本对照 vLLM、SGLang、TensorRT-LLM，不预设性能冠军，给出统一压测和选型决策方法。

> **版本边界**：vLLM 内部模块和默认值迭代很快。本文源码路径、默认配置与命令以 v0.27.1 为准；升级时请同时核对对应 tag、发布说明和 `--help`。

## 本章小节

- [**3.1 vLLM 快速入门**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/vllm快速入门)：安装、离线批量推理、OpenAI 兼容服务部署
- [**3.2 vLLM 整体架构**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/32-vllm整体架构)：AsyncLLM、EngineCore、Scheduler、KVCacheManager、Worker/ModelRunner
- [**3.3 V1 引擎深度解析**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/33-v1引擎深度解析)：多进程架构、统一调度、Prefix Cache、Persistent Batch
- [**3.4 调度器源码导读**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/34-调度器源码导读)：Token Budget、Waiting/Running 队列、抢占与重计算
- [**3.5 关键配置调优**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/35-关键配置调优)：显存估算、批大小、Block、并行与可复现压测
- [**3.6 框架横向对比**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/36-框架横向对比)：vLLM / SGLang / TensorRT-LLM 选型与公平 benchmark

## 推荐学习顺序

第一次接触 vLLM，按 3.1 → 3.2 → 3.5 阅读，先跑通服务并学会容量估算；准备读源码时，再学习 3.3 → 3.4。3.6 适合在已有基线数据后阅读，因为框架选型必须结合实际模型、硬件、流量和 SLO。
