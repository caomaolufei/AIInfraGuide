---
title: "第2章：推理引擎核心技术"
description: "掌握 PagedAttention、Continuous Batching、Prefix Cache、Chunked Prefill 与 Attention 后端及图优化五大推理引擎核心技术"
pubDate: 2026-04-16
updatedDate: 2026-10-29
category: "inference-optimization"
order: 31
tags: ["PagedAttention", "Continuous Batching", "Prefix Cache", "Chunked Prefill", "FlashAttention", "CUDA Graph"]
---

## 📖 本章简介

现代 LLM 推理引擎的高性能来自多项核心技术的协同，本章逐一深入，并对照 vLLM 的具体实现。

**PagedAttention**借鉴操作系统虚拟内存分页的思想，用块表把逻辑块映射到物理块，解决 KV Cache 的内部碎片和外部碎片，大幅提升显存利用率。

**Continuous Batching**解决 Static Batching 中短请求被长请求拖累的问题：调度粒度细化到每个迭代步（Iteration-level Scheduling），请求随到随拼、完成即退，批次里不再有空转的槽位。

**Prefix Cache / RadixAttention**针对大量请求共享相同 System Prompt 的场景：vLLM 的 Prefix Cache 通过链式块哈希复用已有 KV 块，SGLang 的 RadixAttention 基于 Radix Tree 做最长前缀匹配，更擅长树状共享。vLLM V1 把前缀缓存的记账开销压到很低，并默认开启。

**Chunked Prefill 与统一 Token Budget 调度**将长 Prompt 拆成多个 Chunk 分块处理，减少 Prefill 对 Decode 请求的干扰。vLLM V1 更进一步取消 Prefill/Decode 的二分，用统一的 Token 预算调度器同时容纳两类请求。

**Attention 后端与图优化**介绍 vLLM 可插拔的 Attention 后端（FlashAttention 2/3/4、FlashInfer、FlashMLA、Triton）如何支持 Prefill/Decode 混合批次，以及 CUDA Graph 与 torch.compile 如何消除逐 Kernel 启动开销、降低 Decode 阶段的 CPU 瓶颈。

> ⏰ **信息时效**：各技术的原理依据对应论文；vLLM 的参数默认值、调度与 KV Cache 管理的源码走读、CUDA Graph 模式基于 v0.32.0（2026-10-18 发布）。

## 📑 本章小节

- [**2.1 PagedAttention：像操作系统管理内存一样管理 KV Cache**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/)：连续分配的两种碎片、Block Table 映射、Kernel 寻址、CoW 与 V1 的只读共享、抢占与重计算、`KVCacheManager` 与 `BlockPool` 源码
- [**2.2 Continuous Batching：让 GPU 不再空等的动态批处理**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/)：Static Batching 的空转槽位、迭代级调度、与 PagedAttention 的协同、V1 Scheduler 的队列与抢占
- [**2.3 Prefix Cache 与 RadixAttention：复用共享前缀的 KV**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/)：链式块哈希、LRU 与引用计数淘汰、SGLang 的 Radix Tree、vLLM APC 源码
- [**2.4 Chunked Prefill 与统一 Token Budget 调度**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/)：长 Prefill 造成的 TPOT 尖刺、切块、Token 预算装箱、`max_num_batched_tokens` 调优
- [**2.5 Attention 后端与图优化**](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/)：可插拔后端与默认选择、混合批次、CUDA Graph 与 Padding、Piecewise 与 `FULL_AND_PIECEWISE`、torch.compile 切图
