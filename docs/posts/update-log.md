---
title: "AIInfraGuide 文章更新日志"
description: "记录 AIInfraGuide 知识库的每一次内容更新，方便读者追踪最新变化"
pubDate: 2026-04-16
tags: ["公告", "更新日志"]
---

本文持续记录 AIInfraGuide 知识库的内容更新，按时间倒序排列，方便大家了解最新动态。

---

## 2026-08-12

### 重大更新

**[模块四 · 推理优化](/AIInfraGuide/inference)** 第 3–12 章完成：

- **深入 vLLM**：补齐整体架构、V1 引擎、调度器源码、配置调优与框架横向对比
- **核心优化与扩展**：系统讲解量化、Speculative Decoding、分布式推理和 P/D 解耦
- **生产落地**：新增结构化输出、Tool Calling、Multi-LoRA、VLM、性能分析、Kubernetes 运维、容量规划与端到端实战
- **[深入 SGLang](/AIInfraGuide/inference/模块四-推理优化/第12章-深入sglang/第12章-深入sglang)**：新增独立第 12 章，覆盖 SRT、RadixAttention、HiCache、CPU/GPU 重叠调度、结构化生成、Router 与生产调优
- 所有版本敏感的命令、源码路径与能力边界均注明核对版本；性能对比采用统一 workload 方法，不提供脱离测试条件的绝对排名

---

## 2026-07-09

### 新增内容

**模块四 · 推理优化** 第 1、2 章文章上线：

- **文章：** [1.1 LLM 推理基础](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/11-llm推理基础/) | 一篇讲透 LLM 推理的底层逻辑：Prefill/Decode 两阶段、KV Cache 显存账本、TTFT/TPOT 等性能指标，以及用 Roofline 解释为什么 Decode 是 Memory Bound

- **文章：** [2.1 PagedAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | 深入 vLLM 的核心技术 PagedAttention：借鉴虚拟内存分页思想，用 Block Table 消除 KV Cache 碎片化，并支持前缀共享与 Copy-on-Write

- **文章：** [2.2 Continuous Batching](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | 深入 Continuous Batching 的原理：Iteration-level Scheduling 让请求随到随拼、完成即退，把 GPU 利用率从三成拉到八成以上

- **文章：** [2.3 Prefix Cache 与 RadixAttention](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | 深入前缀缓存：vLLM 基于 Hash 的自动前缀缓存如何复用共享前缀的 KV 块，SGLang 的 RadixAttention 如何用 Radix Tree 做更高效的前缀共享

- **文章：** [2.4 Chunked Prefill 与统一调度](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | 深入 Chunked Prefill：把长 Prompt 的 Prefill 切块，消除它对 Decode 请求的干扰；理解 vLLM V1 如何用统一 Token 预算调度器抹平 Prefill/Decode 边界

- **文章：** [2.5 Attention 后端与图优化](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | 深入 vLLM 的可插拔 Attention 后端（FlashAttention/FlashInfer/FlashMLA/Triton）与图优化：用 CUDA Graph 和 torch.compile 消除 Decode 阶段的 CPU 启动开销

---

## 2026-07-07

### 新增内容

- **文章：**  [1.1 分布式训练总论：显存账本与五大并行策略全景](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/11-分布式训练总论/) | 手算训练显存账本，总览五大并行策略全景与选择原则

- **文章：**  [1.2 环境搭建与分布式启动](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/12-环境搭建与分布式启动/) | rank/local_rank/world_size、torchrun 启动与会合机制、NCCL 调试、最小 DDP 脚本

---

## 2026-07-06

### 新增内容

更新了分布式训练部分的教程大纲：
| 章节 | 主要内容 |
|------|----------|
| 第 1 章 分布式训练总论 | 为什么需要分布式训练、训练显存账本、五大并行策略全景、环境搭建与 DDP 启动 |
| 第 2 章 集合通信原语 | AllReduce/ReduceScatter/AllGather/All-to-All 语义、Ring 算法、通信量量化分析 |
| 第 3 章 优化器 | SGD/Adam/AdamW 演进、优化器状态显存开销分析、大 Batch 优化器 LAMB/LARS |
| 第 4 章 数据并行 | DataParallel、DistributedDataParallel、FSDP |
| 第 5 章 ZeRO 系列 | ZeRO-1/2/3 切分策略与通信代价、ZeRO-Offload/Infinity 异构内存卸载 |
| 第 6 章 张量并行与序列并行 | Megatron Column/Row Parallel Linear、通信插入位置、序列并行激活显存优化 |
| 第 7 章 流水线并行 | Bubble 问题本质、GPipe、1F1B、Interleaved 调度 |
| 第 8 章 混合精度与显存优化 | FP16/BF16/FP8 混合精度、梯度累积、Activation Checkpointing |
| 第 9 章 长序列训练与上下文并行 | Attention 的 O(s²) 困境、Ring Attention、DeepSpeed-Ulysses、Megatron Context Parallel |
| 第 10 章 MoE 并行 | Router 机制、Expert Parallelism 的 All-to-All 通信、EP×DP×TP×PP 组合与负载均衡 |
| 第 11 章 3D 并行与混合并行策略 | TP×PP×DP×EP×CP 拓扑设计、通信域映射、rank 编排、集群选型 |


---

## 2026-06-25

### 新增内容

- **文章：**  [6.1 FlashAttention V1详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/61-flashattention-v1详解) | 本文深入剖析 FlashAttention V1 的核心原理与实现细节，理解如何通过 Tiling + Online Softmax 将 Attention 的额外显存占用从 $O(N^2)$ 降至 $O(N)$，同时大幅减少 HBM 访问量，实现无精度损失的加速

- **文章：**  [6.2 FlashAttention V2详解](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/62-flashattention-v2详解) | 本本文深入剖析 FlashAttention V2 相比 V1 的核心改进：调换内外循环顺序、优化线程块内的工作分配、减少非矩阵乘运算

---

## 2026-06-06

### 新增内容

- **文章：**  [5.2 CUDA Online Softmax 实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/52-cuda-online-softmax实现) | 本文从算法推导出发，逐步实现并优化 Online Softmax 的 CUDA Kernel，这也是理解 FlashAttention 的核心前置知识

---

## 2026-06-05

### 新增内容

- **文章：**  [5.1 CUDA Softmax 朴素实现优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/51-cuda-softmax朴素实现优化) | 本文从朴素实现出发，逐步引入 Safe Softmax、Block 级并行、Warp Shuffle、向量化访存等优化手段

---

## 2026-06-02

### 新增内容

- **文章：** [4.1 CUDA GEMM算子性能优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/41-cuda-gemm算子性能优化) |本文从朴素实现出发，逐步Block-Warp-Thread三级Tiling优化、向量化访存、Bank Conflict 消除、双缓冲等优化手段，带你系统掌握 CUDA GEMM 优化的完整方法论

---

## 2026-05-18

### 新增内容

- **文章：** [3.1 CUDA Reduce算子优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/31-cuda-reduce算子优化) | Reduce（规约）是 GPU 编程中最基础、也最能体现并行思维的算子之一。本文从最朴素的实现出发，逐步引入 Warp 级原语、向量化访存、多元素处理等优化手段，每一步都有性能对比和原理解析，帮你真正搞懂"怎么写出快的 Kernel"。

---

## 2026-05-16

### 新增内容

- **模块三 · 分布式训练** 新增 [第2章：优化器](/AIInfraGuide/distributed/模块三-分布式训练/第2章-优化器) | 理解 SGD → Momentum → Adam → AdamW 的演进逻辑与内部状态组成，手算优化器显存开销（AdamW 占训练总显存 75%），掌握大 Batch 优化器 LAMB/LARS 的适用场景，为后续 ZeRO 显存优化和混合精度训练打下基础。

### 结构调整

- **模块三 · 分布式训练** 章节重新编号：原第2~7章顺延为第3~8章，新增第2章"优化器"插入总论与数据并行之间
- 同步更新 AI Infra 学习路线、README 中的模块三章节目录

---

## 2026-05-13

### 新增内容

- **文章：** [2.4 同步与原子操作](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/24-同步与原子操作) | 正确的同步机制是编写无 Bug 并行程序的基础。本文详解 CUDA 中的块内同步 `__syncthreads()`、Warp 级同步、Memory Fence，以及原子操作的使用场景、性能代价与优化技巧，帮助你在保证正确性的前提下写出高性能的并行代码。

---

## 2026-05-12

### 新增内容

- **文章：** [2.2 内存访问优化](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/22-内存访问优化) | 内存访问效率是 CUDA Kernel 性能最大的杠杆。本文深入讲解合并访问（Coalesced Access）的原理与判定方法、共享内存 Bank Conflict 的成因与 Padding 解决方案，以及向量化加载（float4/int4）提升带宽利用率的实战技巧。
- **文章：** [2.3 Occupancy 与资源分配](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/23-occupancy与资源分配) | Occupancy 衡量 SM 上实际活跃 Warp 数与理论最大值的比例，是调优 CUDA Kernel 的核心指标之一。本文讲解 Occupancy 的定义、计算方法、三大限制因素（寄存器/共享内存/Block 大小），以及为什么 Occupancy 并非越高越好——真正的目标是在延迟隐藏与资源利用之间找到平衡点。

---

## 2026-05-11

### 新增内容

- **文章：** [2.1 Warp 与执行模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/21-warp与执行模型) | 深入理解 GPU 最核心的执行单元——Warp

---
## 2026-05-09

### 新增内容

- **文章：** [CUDA 内存模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/13-cuda内存模型) | 本文详解全局内存、共享内存、寄存器、常量内存和统一内存的特性、适用场景及优化技巧
- **文章：** [第一个实用 Kernel：向量加法](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/14-第一个实用kernel) | 通过一个完整的向量加法案例，走通 CUDA 编程的全流程

---
## 2026-05-08

### 新增内容

- **文章：** [1.1 CUDA 开发环境搭建](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/11-cuda开发环境搭建)
- **文章：** [1.2 CUDA 编程模型](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/12-cuda编程模型)

---
## 2026-05-07

### 新增内容

- **文章：** [✅ CUDA编程快速入门指南 🔥](https://caomaolufei.github.io/AIInfraGuide/cuda/模块二-cuda编程与算子优化/cuda编程入门指南/)


---
## 2026-04-28

### 新增内容

- **文章：** [🔥 集群通信网络与NCCL：分布式训练的通信骨架](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/communication/collective-communication-primer/)

---
## 2026-04-27

### 新增内容

- **文章：** [5.1 NVIDIA GPU架构演进：从Volta到Blackwell](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpunvidia-gpu-evolution/)

---
## 2026-04-26

### 新增内容

- **文章：** [🔥 GPU基础知识：从硬件架构到AI计算](https://caomaolufei.github.io/AIInfraGuide/prerequisites/模块一-前置知识/gpu/gpu-basics/)

---
## 2026-04-25

### 新增内容

- **文章：** [🔥 PyTorch框架快速入门篇](/AIInfraGuide/prerequisites/模块一-前置知识/pyroch/pytorch框架入门)

---
## 2026-04-24

### 新增内容

- **文章：** [3.9 Tokenization与词嵌入](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/39-tokenization与词嵌入)


---
## 2026-04-23

### 新增内容

- **文章：** [3.8 从Transformer到LLM自回归生成深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/38-从transformer到llm自回归生成深入理解)

---

## 2026-04-21

### 新增内容

- **文章：** [3.6 LayerNorm与残差连接深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/36-layernorm与残差连接深入理解)
- **文章：** [3.7 Transformer Decoder Block完整解析](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/37-transformer-decoder-block完整解析)

---

## 2026-04-20

### 新增内容

- **文章：** [3.4 Transformer前馈网络ffn深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/34-transformer前馈网络ffn深入理解)
- **文章：** [3.5 Transformer位置编码深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/35-transformer位置编码深入理解)

---

## 2026-04-18

### 新增内容

- **文章：** [3.3 Self-Attention 机制深入理解](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/33-Self-Attention机制深入理解)

---

## 2026-04-17

### 新增内容

- **面试专区**：共收录 **180+ 场面试真题**，覆盖 **40+ 家公司**，按梯队分类组织。[在线浏览 →](https://caomaolufei.github.io/AIInfraGuide/interview)
- 涵盖公司包括字节跳动、阿里巴巴、腾讯、百度、快手、美团、蚂蚁、英伟达、MiniMax、蔚来、小鹏、理想等。
- **社区入口**：新增[wx交流群]((/AIInfraGuide/about/author))，欢迎大家加入讨论

### 内容优化

- 更新 README，补充项目介绍图片

---

## 2026-04-16

### 重大更新

- **知识库框架全面升级**：细化四大学习模块，调整页面布局结构
  - **[模块一 · 前置知识](/AIInfraGuide/prerequisites)**（6 章）：[编程语言基础](/AIInfraGuide/prerequisites/模块一-前置知识/第1章-编程语言基础)、[数学基础](/AIInfraGuide/prerequisites/模块一-前置知识/第2章-数学基础)、[Transformer 架构详解](/AIInfraGuide/prerequisites/模块一-前置知识/第3章-Transformer架构详解)、[PyTorch 框架](/AIInfraGuide/prerequisites/模块一-前置知识/第4章-PyTorch框架)、[GPU 硬件概论](/AIInfraGuide/prerequisites/模块一-前置知识/第5章-GPU硬件概论)、[集合通信基础](/AIInfraGuide/prerequisites/模块一-前置知识/第6章-集合通信基础)
  - **[模块二 · CUDA 编程与算子优化](/AIInfraGuide/cuda)**（8 章）：[CUDA 入门](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第1章-CUDA编程入门)、[性能优化基础](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第2章-CUDA性能优化基础)、Reduce 实战、GEMM 实战、Softmax 实战、[Attention 算子](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第6章-Attention算子)、[AI 编译器](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第7章-AI编译器)、[性能分析工具链](/AIInfraGuide/cuda/模块二-CUDA编程与算子优化/第8章-性能分析工具链)
  - **[模块三 · 分布式训练](/AIInfraGuide/distributed)**（7 章）：[分布式训练总论](/AIInfraGuide/distributed/模块三-分布式训练/第1章-分布式训练总论)、[数据并行](/AIInfraGuide/distributed/模块三-分布式训练/第2章-数据并行)、[ZeRO 系列](/AIInfraGuide/distributed/模块三-分布式训练/第3章-ZeRO系列)、[张量并行与序列并行](/AIInfraGuide/distributed/模块三-分布式训练/第4章-张量并行与序列并行)、[流水线并行](/AIInfraGuide/distributed/模块三-分布式训练/第5章-流水线并行)、[3D 并行与混合训练策略](/AIInfraGuide/distributed/模块三-分布式训练/第6章-其他显存优化技术)、[训练框架实战](/AIInfraGuide/distributed/模块三-分布式训练/第7章-训练框架实战)
  - **[模块四 · 推理优化](/AIInfraGuide/inference)**（12 章）：LLM 推理基础、推理引擎核心技术、深入 vLLM、量化、Speculative Decoding、分布式推理、P/D 解耦、生产级服务特性、性能分析与 Benchmark、生产部署与运维、选型与端到端实战、深入 SGLang

### 新增内容

- **Transformer 系列文章**：
  - [Transformer 架构入门](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/Transformer架构快速入门)
  - [3.1 AI Infra 工程师为什么必须懂 Transformer](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/31-ai-infra工程师为什么必须懂transformer)
  - [3.2 Transformer 全貌及代码实现](/AIInfraGuide/prerequisites/模块一-前置知识/transformer/32-transformer全貌及代码实现)
- **课程资料**：150+ 篇详细子文章上线，覆盖四大模块全部章节
- **博客文章**：新增[「关于作者」](/AIInfraGuide/about/author)介绍页

---

## 2026-04-15

### 项目上线

- AIInfraGuide 知识库正式上线
- 发布 [AI Infra 学习路线](/AIInfraGuide/guides/AI Infra学习路线)总览
- 发布[「从零理解 AI Infra」](/AIInfraGuide/blog/ai-infra-introduction)入门指引

---

## 2026-04-14

### 项目初始化

- 完成网站框架搭建（基于 Astro + Starlight）
- 初始化项目仓库，发布首次 commit
