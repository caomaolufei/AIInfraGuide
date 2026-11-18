---
title: "2026 AI Infra 面试专题"
description: "围绕 2026 年 AI Infra 面试常见方向整理的考点索引，覆盖推理引擎、CUDA/Triton、量化、多卡稳定性与异构算力"
pubDate: 2026-11-18
company: "综合"
tier: "综合"
interviewType: "未知"
order: 11
tags: ["推理优化", "算子优化", "高性能计算"]
---

> 本文是 AI Infra 面试考点索引，不是某家公司的完整真题，也不代表固定题库。不同公司对同一考点的表述可能不同，回答时应结合自己的项目和实测数据。

## 1. 推理引擎与 KV Cache

题目主要集中在“瓶颈是什么、缓存怎么复用、请求怎么调度”三件事：

- PagedAttention 与 Continuous Batching 分别解决什么瓶颈？
- RadixAttention 与 vLLM Prefix Cache 是什么关系？
- FP8 KV Cache 如何配置？收益体现在容量还是 Decode 带宽？
- FP8 W8A8 在 Prefill 阶段能带来什么收益？受哪些硬件和形状条件限制？
- 长 Prompt 的 Prefill 与新请求的 Decode 如何混合调度？会不会发生抢占？
- PD 分离后，Prefill 的 KV 如何传到 Decode，传输与同步成本怎么分析？

对应教程：[PagedAttention](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) · [Continuous Batching](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) · [Prefix Cache](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) · [PD 解耦（章节导读）](/AIInfraGuide/inference/模块四-推理优化/第7章-pd解耦架构/第7章-pd解耦架构/)

## 2. CUDA、Triton 与性能分析

这类题既考 Kernel 实现，也考能否用 Profiling 证据定位瓶颈：

- CUDA GEMM 如何从正确实现推进到优化？评价指标是什么？
- 手写 MLA 时需要说明哪些张量、缓存和并行结构？
- SM 未用满时增大矩阵乘法尺寸，为什么延时会增长？
- 除 Compute Bound 和 Memory Bound 外，还有哪些瓶颈类型？
- Bank Conflict 如何产生、检测和消除？Triton 适合解决什么问题？
- 如何用 Nsight Systems、Nsight Compute 或 PyTorch Profiler 建立证据链？

对应教程：[CUDA GEMM](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化/) · [Warp 与执行模型](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型/) · [内存访问优化](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化/) · [FlashAttention V1](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解/) · [Triton](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/71-triton编程框架/) · [性能分析工具链](/AIInfraGuide/cuda/模块二-cuda编程与算子优化/第8章-性能分析工具链/)

## 3. 量化与低精度

这批面试题更关注硬件执行条件和真实收益：

- FP8 KV Cache 怎样配置，收益边界如何验证？
- AWQ 的核心原理是什么？
- W8A16 在 Forward 中怎样执行？
- 在哪些情况下 Int4 量化后反而更慢？如何检查芯片支持、实际 Kernel 与反量化开销？
- BF16、FP8、INT8/INT4 如何取舍？PTQ 与 QAT 各适合什么场景？

对应教程：[量化基础](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) · [Weight-only INT4](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) · [KV Cache 量化](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) · [FP8 与 NVFP4](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) · [量化选型与 vLLM 实战](/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/)

## 4. 多卡稳定性与通信

这组题既有线上故障排查，也有并行与通信基础：

- TP 与 CUDA Graph 同时开启后，如何排查稳定性和超时？
- CUDA Graph 是触发条件还是根本原因？如何做最小复现？
- Split-K 为什么可能破坏 Router GEMM 的确定性？
- 动态 Shape 下如何选择 Kernel 和回退路径？
- NCCL Ring AllReduce 与 Tree AllReduce 如何选择？
- PD 分离后如何分析 KV 传输与同步瓶颈？

对应教程：[集合通信原语](/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) · [张量并行与序列并行](/AIInfraGuide/distributed/模块三-分布式训练/第6章-张量并行与序列并行/) · [Attention 后端与图优化](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/)

## 5. 异构算力与推理平台

- 海光 HIP/DCU 与 CUDA 生态差异在哪里？CUDA 算子迁移要检查什么？
- 异构算力适配中，框架、算子库和运行时分别承担什么职责？
- GPU 池化与弹性扩缩容如何设计？
- 端侧内存、功耗、KV Cache 与端云协同如何权衡？

对应教程：[端侧推理基础](/AIInfraGuide/inference/模块四-推理优化/第9章-端侧推理/91-端侧推理基础/)

## 6. 回答这类问题的共同框架

1. 先固定模型、硬件、精度、Batch、输入输出长度和并发，避免用不同口径比较。
2. 区分单算子收益、端到端时延收益和线上 SLO 收益，不把 Kernel 加速比当成请求延迟下降。
3. 性能结论必须给出 Profiling 证据、正确性验证和回退条件。
4. 项目题要讲清“原来慢在哪、改了什么、为什么有效、怎么验收”。
