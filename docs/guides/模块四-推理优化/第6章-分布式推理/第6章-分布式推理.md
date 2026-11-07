---
title: "第6章：分布式推理与大模型部署"
description: "理解推理场景下 TP、PP、DP、EP 的存储与通信代价，掌握 vLLM 的单机与多节点部署、MoE 的 DP+EP、DCP，并用对照实验选出合适的布局"
pubDate: 2026-04-16
updatedDate: 2026-11-07
category: "inference-optimization"
order: 35
tags: ["分布式推理", "张量并行", "流水线并行", "Expert Parallel", "Ray", "vLLM"]
---

## 📖 本章简介

模型大到一张卡装不下，或者一个副本的吞吐不够时，推理就要拆到多卡、多机上。推理和训练用的是同一套并行术语，目标却不同：训练关心梯度同步和收敛，推理关心权重和 KV Cache 放不放得下，以及 TTFT、TPOT 和吞吐。

本章从"怎么切"讲到"怎么部署、怎么验证"：

- **选型**：用显存账本区分"装不下"和"不够快"两类问题，对比 TP、PP、DP、EP 各自切的是什么、要同步什么，给出选择并行度的顺序。
- **TP 与 PP**：TP 每层两次 AllReduce 的数据量和延迟、GQA 下的 KV 头复制、对 NVLink 的依赖；PP 段间传递的内容、为什么需要多个 Batch 同时在飞，以及 TP×PP 怎么组合。
- **DP 与 EP**：Dense 模型的多副本，MoE 的注意力 DP + 专家 EP、两次 All-to-All、空跑同步、负载不均与 EPLB，以及 DeepSeek-V3 的线上布局。
- **部署**：Ray 和多进程两种多节点运行时、rank 布局、MoE 的多节点 DP + EP 启动、InfiniBand 与 GPUDirect RDMA 配置；最后在一台 8 卡机上对比三种布局，并介绍用 DCP 消除长上下文的 KV 复制。

> ⏰ **信息时效**：原理部分与框架无关；vLLM 相关的参数、默认值、日志与指标基于 v0.32.0（2026-10-18 发布）的文档与源码，DeepEP 数据基于 v1.2.1。分布式相关的参数在近几个版本中变动较多，换版本时请对照官方文档核对。

## 📑 本章小节

- [**6.1 推理并行策略总览：模型装不下和服务不够快是两回事**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/61-推理并行策略总览/)：显存账本、四种切法的存储与通信对比、并行度的选择顺序与估算脚本
- [**6.2 张量并行推理：一层切成几份，每层同步两次**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/62-张量并行推理/)：列切与行切、AllReduce 的数据量与延迟模型、KV 头复制、NVLink 依赖与 Custom AllReduce
- [**6.3 流水线并行推理：按层切开，用多个 Batch 填满流水线**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/63-流水线并行推理/)：段间通信量、Decode 的 Batch 队列、层划分不均、单请求延迟与 TP×PP 组合
- [**6.4 数据并行与专家并行：多开副本，以及让 Token 去找专家**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/64-数据并行与专家并行/)：Dense DP、MoE 的 Dispatch/Combine、注意力 DP + 专家 EP、EPLB、DeepEP 与 DeepSeek-V3 部署
- [**6.5 多节点部署：用 Ray 或多进程把一个服务铺到多台机器上**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/65-多节点部署/)：Ray 集群与 Placement Group、多进程 `--nnodes`、多节点 DP + EP、NCCL 网络配置与排错
- [**6.6 vLLM 分布式部署实战：同样 8 张卡，三种切法一起测**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第6章-分布式推理/66-vllm分布式部署实战/)：TP8、TP4×PP2、TP4×DP2 的显存估算、日志核对与 Benchmark，以及 DCP
