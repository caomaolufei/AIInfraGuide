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

**快速入门**不止跑通命令，还用离线、在线、连续批处理与 Prefix Cache 四组实验建立可复现基线，并把安装、OOM、流式缓冲和运行时退化按层排障。

**整体架构**沿真实请求对象穿过 OpenAI API、`AsyncLLM`、ZMQ IPC、`EngineCore`、`Scheduler`、`KVCacheManager`、`Worker` 与 `ModelRunner`，同时追踪 token、block table、persistent batch 和流式输出状态。

**V1 引擎**从 V0 的结构性瓶颈出发，深入多进程 IPC、统一 token 状态、Prefix Cache、Persistent Batch、乐观异步状态、batch queue、`torch.compile` 与 CUDA Graph 的 shape 调度。

**调度器源码导读**逐轮手算 running/waiting 的 budget、KV 分配与抢占回滚，并追踪 structured output、speculative decoding、grammar bitmask 和 accepted token 如何改变下一轮状态。

**关键配置调优**建立显存、调度、执行与 SLO 的参数因果图，通过单变量与二阶交互矩阵寻找 goodput、质量、成本的 Pareto 前沿，而不是寻找一组“万能参数”。

**框架横向对比**不做静态功能打勾表，而是给出同模型、同协议、同精度、同负载的公平 benchmark：覆盖 cold/warm/churn、过载恢复、质量门禁、goodput、能耗与回退演练。

> **版本边界**：vLLM 内部模块和默认值迭代很快。本文源码路径、默认配置与命令以 v0.27.1 为准；升级时请同时核对对应 tag、发布说明和 `--help`。

## 本章小节

- [**3.1 vLLM 快速入门**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/vllm快速入门)：离线/在线完整实验、指标观察与分层排障
- [**3.2 vLLM 整体架构**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/32-vllm整体架构)：请求对象、IPC、Scheduler、KV 与 ModelRunner 全链路
- [**3.3 V1 引擎深度解析**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/33-v1引擎深度解析)：V0→V1、Persistent Batch、异步流水线、compile/CUDA Graph
- [**3.4 调度器源码导读**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/34-调度器源码导读)：逐轮 budget、preemption、structured/spec decode 交互
- [**3.5 关键配置调优**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/35-关键配置调优)：参数因果图、显存手算与调参实验矩阵
- [**3.6 框架横向对比**](/AIInfraGuide/inference/模块四-推理优化/第3章-深入vllm/36-框架横向对比)：公平 benchmark、goodput、成本与回退

## 推荐学习顺序

第一次接触 vLLM，按 3.1 → 3.2 → 3.5 阅读，先跑通服务并学会容量估算；准备读源码时，再学习 3.3 → 3.4。3.6 适合在已有基线数据后阅读，因为框架选型必须结合实际模型、硬件、流量和 SLO。
