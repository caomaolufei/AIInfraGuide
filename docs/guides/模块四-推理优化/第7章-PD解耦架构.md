---
title: "第7章：Prefill/Decode 解耦架构"
description: "理解 P/D 混合 Batching 的问题、DistServe/Splitwise 等解耦方案、KV Cache 传输、Goodput 与 SLO 感知调度"
pubDate: 2026-04-16
updatedDate: 2026-10-23
category: "inference-optimization"
order: 36
tags: ["P/D解耦", "DistServe", "Splitwise", "KV Cache传输", "Goodput", "SLO"]
---

## 📑 目录

- [本章简介](#本章简介)
- [新手怎样学习本章](#新手怎样学习本章)
  - [先接上第六章的概念](#先接上第六章的概念)
  - [遇到缩写，先用这张表对照](#遇到缩写先用这张表对照)
  - [按一条请求的六个问题逐步阅读](#按一条请求的六个问题逐步阅读)
  - [第一次不必完成所有 GPU 实验](#第一次不必完成所有-gpu-实验)
  - [带着三个问题检查自己的理解](#带着三个问题检查自己的理解)
- [本章小节](#本章小节)
- [总结](#-总结)
- [自我检验清单](#-自我检验清单)
- [参考资料](#-参考资料)

## 本章简介

Prefill 与低并发 Decode 常分别偏计算和访存密集，具体瓶颈取决于模型与工作负载；混批时还可能出现资源互扰。本章深入解耦架构的设计与挑战，并落到 vLLM 的 Disaggregated Prefill 能力。

**混合 Batching 的问题**通过时间线、教学模型和分组实验分析 Prefill 对 Decode 的干扰，区分请求平均 TPOT 与逐 Token 的 ITL；具体倍数由模型、硬件、负载和统计口径决定。

**解耦架构设计**覆盖 DistServe（OSDI'24，系统化论证 P/D 解耦）、Splitwise（ISCA'24，分配到不同 GPU 池）、TaiChi（聚合与解耦统一框架）和 MLC Microserving（跨引擎编排）。

**KV Cache 传输与 Connector**深入解耦架构最核心的工程挑战：Prefill 节点算出的 KV Cache 如何高效搬到 Decode 节点。介绍 vLLM 的 KV Connector 抽象、NIXL/NCCL 等传输后端，以及跨节点带宽对整体延迟的影响。

**Goodput 与 SLO 感知调度**强调 Raw QPS 不等于用户体验，Goodput（满足 SLO 的有效吞吐）才是真正的优化目标。

**解耦架构的挑战**包括调度器复杂度、P/D GPU 池的资源配比推导，以及何时该用/不该用解耦。

**vLLM Disaggregated Prefill 实战**：用 vLLM 的解耦部署构造混合负载，量化 P/D 互扰程度并推导给定工作负载下的 P/D GPU 池配比。

## 新手怎样学习本章

### 先接上第六章的概念

本章的 P 是 Prefill，即处理输入并建立历史 KV 的阶段；D 是 Decode，即复用历史状态逐步生成的阶段。聚合部署把两类工作放在同一引擎中，解耦部署把它们放到不同实例，再通过缓存移交连接。

注意两种切分：第六章 PP 按模型层分工，P/D 解耦按请求的计算阶段分工。P、D 各自仍需完整的相应模型计算路径，也可以在内部使用 TP/PP；D 不是只拿一份 KV 就能生成的“半个模型”。

如果 Token、KV、TTFT/TPOT、模型副本和 TP/PP 还不熟悉，可以先回顾第六章总览。阅读本章不要求先部署多节点 Ray，PD 通路的传输和编排也不等于必须由 Ray 完成。

### 遇到缩写，先用这张表对照

第一次阅读时不必记住所有英文名称。先把它们放回“用户发来一条请求，系统怎样处理”的过程里：

| 术语 | 本章里的含义 | 可以怎样理解 |
|------|--------------|--------------|
| Token | 模型实际处理的离散文本单位 | 先把文字切成模型认识的小单位，不固定等于一个汉字 |
| P / Prefill | 处理输入、建立输入对应的 KV | 先读懂题目，准备后续生成要用的历史状态 |
| D / Decode | 利用历史 KV，逐步生成后续内容 | 有了历史状态，继续一步步作答，仍需模型计算 |
| Router | 根据策略选择实例并转发、编排请求的服务组件 | 接待与派单员；不是这里所说的家用网络路由器，也不是 MoE 里选专家的 Router |
| KV Cache | Attention 复用的历史 Key/Value 状态 | 请求的计算中间结果；不是模型权重，也不是聊天原文 |
| Connector | 将引擎的缓存管理与具体传输实现连接起来的组件 | 负责接上搬运流程；不等于完整的入口调度器 |
| SLO | Service Level Objective，服务等级目标 | 一份明确的合格标准，例如本章假设的首字 ≤ 1 秒、平均出字间隔 ≤ 50 ms |
| TTFT / TPOT / ITL | 首 Token 等待 / 首 Token 后平均间隔 / 相邻 Token 间隔 | 分别观察开头等多久、平均出字多快、某一步卡多久 |
| Goodput | 在既定口径下同时达标的请求数/秒 | 及时完成且符合要求的有效工作量；不只是总完成数 |

SLO 还要说明适用请求、观测位置、统计窗口和期望通过率。“首字 ≤ 1 秒”是一条请求的时间门槛，“至少 95% 的请求同时达标”才进一步约束了整批服务表现。两者要一起写清楚，不能把示例数字当成行业统一标准。

图中的颜色只是辅助，箭头旁的文字和编号才说明含义。每张图下方都有读图步骤；也可以打开 SVG 原图放大查看。图中毫秒、字节和实例数量与对应文字算例保持一致，均不代表 GPU 实测结果。

### 按一条请求的六个问题逐步阅读

| 小节 | 先回答的问题 | 学完可以做的练习 |
|------|--------------|------------------|
| 7.1 混合 Batching | 为什么别人的长输入让我暂停出字 | 手算 TTFT、TPOT、ITL，运行互扰时间线 |
| 7.2 架构设计 | 请求、控制信息和缓存分别经过哪里 | 画 P/Router/D 路径，区分 PP 与 PD |
| 7.3 KV 传输 | 该传哪些状态、多大、何时可释放 | 算 KV 字节与带宽下界，画块生命周期 |
| 7.4 Goodput | 做完很多请求，为什么仍不合格 | 逐请求计分，算联合通过率与合格请求/s |
| 7.5 资源配比 | P、D 各需要多少实例和 GPU | 把同一流量放进计算、网络、驻留三本账 |
| 7.6 部署实战 | 怎样证明真的复用了缓存并公平比较 | 生成负载；有兼容 GPU 后再验收移交与压测 |

### 第一次不必完成所有 GPU 实验

可以先在普通 CPU 上完成时间线、带宽、Goodput、配比和数据生成练习，再读懂部署脚本。具备兼容环境后，先做一条请求的正确性验证，再增加并发，最后扩展跨机。每一步都明确自己证明了什么，避免把“脚本语法正确”当成“GPU 已成功迁移”。

文中假设的毫秒、带宽有效率、服务需求和请求记录用于说明计算方法，不是设备实测结果。部署接口继续以 vLLM v0.19.0 为基线，模型/Tokenizer 与依赖需要记录实际 revision 和版本。Goodput 的教学门槛也不是所有业务的默认标准。

### 带着三个问题检查自己的理解

一是 P 做完以后，D 凭什么可以少算输入；二是如果 D 或网络接不住，KV 会留在哪里；三是同样 GPU 预算下，多少请求真正同时满足首字、出字和质量要求。能够把这三件事解释清楚，就抓住了本章的主线。

## 本章小节

- [**7.1 混合 Batching 与 P/D 互扰：为什么出字会突然停一下**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/71-混合batching与pd互扰/)：从迭代时间线区分 Prefill/Decode 的资源竞争、排队、TPOT 与 ITL，用可运行模型理解 Chunked Prefill 与 P/D 解耦的取舍
- [**7.2 解耦架构设计：从 DistServe、Splitwise 到混合编排**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/72-解耦架构设计/)：沿请求和 KV 的生命周期比较 DistServe、Splitwise、TaiChi 与 MLC Microserving，理解解耦的资源、路由和故障边界
- [**7.3 KV Cache 传输与 Connector：搬走的不只是一个张量**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/73-kvcache传输与connector/)：核算 KV 载荷与带宽下界，理解 vLLM V1 Connector 的 Scheduler/Worker 分工、NIXL/NCCL 和缓存移交的正确性与失败处理
- [**7.4 Goodput 与 SLO 感知调度：忙得很，为什么合格请求反而少**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/74-goodput与slo感知调度/)：定义请求级 TTFT/TPOT 与联合达标 Goodput，区分分位数和逐请求判断，讨论准入、期限与长度预测及失败统计
- [**7.5 资源配比与调度挑战：P 池和 D 池到底各放几张卡**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/75-资源配比与调度挑战/)：用服务需求和排队余量推导 P/D 初始配比，区分实例数与 GPU 数，加入网络、KV 驻留、背压和动态流量约束
- [**7.6 vLLM P/D 解耦实战：从 NIXL 跑通到同预算对照**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/76-vllm-pd解耦实战/)：按 vLLM v0.19.0 配置双 GPU NixlConnector 与官方代理，生成混合负载并比较聚合、双副本和解耦的互扰、Goodput 与质量

## 📝 总结

- 解耦让阶段资源与调度可分别配置，也引入 KV 迁移和状态治理成本。
- Goodput 需要明确同一请求、统一窗口和联合 SLO 的统计规则。
- P/D 配比要一起满足计算、网络、缓存驻留与尾延迟约束。

## 🎯 自我检验清单

- 能画出请求与 KV 的生命周期及故障处理
- 能区分各系统论文、编排接口和 Connector 的层次
- 能估算迁移成本并推导工作负载相关的资源配比
- 能设计同 GPU 预算的聚合与解耦对照实验
- 能用逐请求记录解释 TTFT、TPOT、ITL 和联合 Goodput 的区别
- 能说明实例能力、GPU 数、网络带宽和 KV 驻留为何需要分别核算

## 📚 参考资料

- [DistServe — OSDI 2024](https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin)
- [vLLM v0.19.0 — Disaggregated Prefilling](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/features/disagg_prefill.md)
- [vLLM v0.19.0 — NixlConnector](https://github.com/vllm-project/vllm/blob/v0.19.0/docs/features/nixl_connector_usage.md)
