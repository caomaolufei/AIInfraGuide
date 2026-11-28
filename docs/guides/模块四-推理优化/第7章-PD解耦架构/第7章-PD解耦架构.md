---
title: "第7章：Prefill/Decode 解耦架构"
description: "理解 P/D 混合 Batching 的问题、DistServe/Splitwise 等解耦方案、KV Cache 传输、Goodput 与 SLO 感知调度，并在 vLLM 上用 NixlConnector 部署和测量"
pubDate: 2026-04-16
updatedDate: 2026-11-28
category: "inference-optimization"
order: 36
tags: ["P/D解耦", "DistServe", "Splitwise", "KV Cache传输", "Goodput", "SLO"]
---

## 📖 本章简介

Prefill 一次处理整段 Prompt，受算力限制；Decode 每步只生成一个 Token，受显存带宽限制。两者放在同一组 GPU 上，会在同一步里互相拖慢，还被迫共用同一套并行策略和硬件。P/D 解耦把两个阶段拆到不同的 GPU 上，代价是要在两组 GPU 之间搬 KV Cache，并多出一层路由和配比问题。

本章按"为什么拆、怎么拆、怎么搬、以什么为目标、拆完的代价、怎么落地"的顺序展开：

- **问题**：用算术强度解释两个阶段抢的是不同资源，用 DistServe 的测量和成本模型量化干扰。DistServe 在 13B 模型上测到，带一个 1024 Token 的 Prefill 时，小 Batch 的 Decode 步从约 10 ms 变成约 120 ms；vLLM 博客的实测里，合并部署的 P99 ITL 在 0.4 req/s 时达到 169 ms，P/D 解耦为 29 ms。
- **设计**：DistServe（OSDI'24）按每卡 Goodput 为两个阶段分别搜索并行与放置，Splitwise（ISCA'24）按阶段划分机器池并使用不同代 GPU，TaiChi 在聚合与解耦之间连续调节，MLC Microserving 用可编程路由器组合细粒度 API。
- **传输**：估算 KV 数据量与带宽需求，比较逐层与串行、推送与拉取，拆解 vLLM 的 KV Connector 抽象和 NixlConnector 的握手、租约与异构 TP。
- **目标与调度**：Goodput 的精确定义、`vllm bench` 的计算方式、SLO 怎么定，以及路由、条件解耦、实例内排序、扩缩容等 SLO 感知调度手段。
- **代价与配比**：解耦带来的新问题，$x$P$y$D 配比与每卡 Goodput 的关系，从单实例容量估算配比，以及何时该用、何时不该用。
- **实战**：在 vLLM 上部署合并部署与 1P1D 两组对照，量化干扰、比较 Goodput，并从实测数据推导配比。

> ⏰ **信息时效**：论文数据取自 DistServe（arXiv v3）、Splitwise（arXiv v2）、TaiChi（arXiv v1）、Microserving（arXiv v1）；vLLM 相关的接口、参数与命令基于 v0.32.0（2026-10-18 发布）与 `nixl==1.5.0`。P/D 解耦在 vLLM 文档中仍标注为实验性功能，换版本时请对照官方文档核对。

## 📑 本章小节

- [**7.1 混合 Batching 的问题**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/71-混合batching的问题/)：算术强度与屋脊点、DistServe 的干扰测量、成本模型、尾延迟、Chunked Prefill 的局限与资源耦合
- [**7.2 解耦架构设计**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计/)：DistServe 的 M/D/1 分析与放置算法、Splitwise 的机器池、TaiChi 的延迟转移、Microserving 的三个 API
- [**7.3 KV Cache 传输与 Connector**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kv-cache传输与connector/)：KV 数据量与带宽、逐层与串行、推送与拉取、vLLM KV Connector 接口、NixlConnector 与其他 Connector
- [**7.4 Goodput 与 SLO 感知调度**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/74-goodput与slo感知调度/)：每卡 Goodput 定义、`vllm bench --goodput`、SLO 的设定、SLO 感知调度、Goodput 的测量流程
- [**7.5 解耦架构的挑战与配比**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/75-解耦架构的挑战与配比/)：新增的系统问题、$x$P$y$D 配比公式、从单实例容量估算配比、动态调整与适用场景
- [**7.6 vLLM Disaggregated Prefill 实战**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/76-vllm-disaggregated-prefill实战/)：环境与传输路径检查、1P1D 部署、双客户端干扰实验、Goodput 对比与配比推导
