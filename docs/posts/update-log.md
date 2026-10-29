---
title: "AIInfraGuide 文章更新日志"
description: "记录 AIInfraGuide 知识库的每一次内容更新，方便读者追踪最新变化"
pubDate: 2026-04-16
tags: ["公告", "更新日志"]
---

本文持续记录 AIInfraGuide 知识库的内容更新，按时间倒序排列，方便大家了解最新动态。

---

## 2026-10-29

### 调整内容

**模块四 · 推理优化** 第 2 章「推理引擎核心技术」全面修订：5 篇文章新增 9 张示意图，参数默认值与源码走读对齐 vLLM v0.32.0，并修正了部分过时或不准确的表述：

- **文章：** [2.1 PagedAttention：像操作系统管理内存一样管理 KV Cache](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/21-pagedattention/) | 新增 Block Table 映射图与碎片对比图；`gpu_memory_utilization` 默认值更新为 0.92；补充 V1 抢占只走重计算、不再使用 Copy-on-Write；修正 `free_blocks` 的空闲队列入队逻辑

- **文章：** [2.2 Continuous Batching：让 GPU 不再空等的动态批处理](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/22-continuous-batching/) | 新增 Static 与 Continuous Batching 时间线对比图；用空转槽位算例和 Orca 论文数据替换无出处的利用率数字；抢占伪代码改为源码中的重试循环

- **文章：** [2.3 Prefix Cache 与 RadixAttention：复用共享前缀的 KV](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/23-prefix-cache-与-radixattention/) | 新增链式块哈希图与 Radix Tree 示例图；补充 `cache_salt`、哈希种子与 RadixAttention 的引用计数、先淘汰叶子的 LRU 和按缓存命中调度

- **文章：** [2.4 Chunked Prefill 与统一 Token Budget 调度](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/24-chunked-prefill-与统一调度/) | 新增分块时间线图与 Token 预算装箱图；说明 vLLM 的块长由剩余预算决定；补充 `long_prefill_token_threshold` 与官方 `max_num_batched_tokens` 调优建议

- **文章：** [2.5 Attention 后端与图优化](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术/25-attention-后端与图优化/) | 新增 Kernel 启动与 CUDA Graph 对比图、Piecewise CUDA Graph 图；补充按 GPU 架构的默认后端；修正 CUDA Graph 的 Padding 规则与 V1 默认的 `FULL_AND_PIECEWISE` 模式

---

## 2026-10-26

### 调整内容

**模块四 · 推理优化** 第 4 章「量化」6 篇文章修订，框架相关内容对齐 vLLM v0.32.0，各篇补充信息时效说明，并重新组织章节结构：

- **文章：** [4.1 量化基础：Scale、粒度与 PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | 用"刻度有限的尺子"重写量化与反量化的讲解；位宽、矩阵轴与量化粒度拆成独立小节；校准单列一章，讲清量程选择与静态、动态校准

- **文章：** [4.2 W8A8 量化：SmoothQuant 如何处理激活离群值](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | 等价缩放补充公式与单个点积的直观推导；平滑系数一节改为讲迁移强度 α 的选择；补充离线流程、收益与适用条件，以及可动手验证的点积示例

- **文章：** [4.3 Weight-only INT4：GPTQ、AWQ 与 Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | 补充 GPTQ 的代价与效果、AWQ 用缩放代替混合精度的思路；新增 vLLM 中的 Kernel 选择、量化工具选择与排查顺序

- **文章：** [4.4 KV Cache 量化：KIVI 与 FP8 缓存](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | 新增 KV 占用估算；FP8 KV Cache 一节按 v0.32.0 重写，覆盖格式选择、Scale 校准、跳过敏感层，以及存储精度与计算精度的区别；注明旧版 `calculate_kv_scales` 参数已移除

- **文章：** [4.5 FP8 与 NVFP4/MXFP4：低比特浮点怎样工作](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | 补充 E2M1 的 8 个幅值、微块粒度的形状写法与 Scale 自身的格式；对比 NVFP4 两层 Scale 与 MXFP4 一层 Scale；新增在 vLLM v0.32.0 中的使用方式

- **文章：** [4.6 量化选型与 vLLM 实战](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | 区分量化方法、检查点格式与 Kernel 三层，补充 vLLM 各量化入口的含义；实验基准更新到 v0.32.0

**模块四 · 推理优化** 第 5 章「Speculative Decoding」5 篇文章修订，配置基准从 vLLM v0.31.0 升级到 v0.32.0，并新增 MTP 相关内容：

- **文章：** [5.1 投机解码核心原理：先猜一段，再让大模型验收](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | 重写 Draft + Verify 一轮的完整流程；新增"Draft 只给 Token、不给概率"时的验证方式

- **文章：** [5.2 Draft 模型与无模型方案：怎样便宜地猜得更准](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | 新增"与 Target 对得上比自身分数高更重要"和并行 Draft 两节；统一接受率、平均接受长度与候选覆盖三个指标的口径

- **文章：** [5.3 Medusa、EAGLE 与 MTP：复用大模型信息生成更好的草稿](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | 新增第 7 节 MTP，覆盖它从预训练目标到推理加速的转变、实际效果、在 vLLM 中的形态，以及一次前向生成整块草稿的探索；EAGLE-2 补充置信度只用于分配预算的说明

- **文章：** [5.4 投机解码的收益边界：接受率高，为什么仍可能变慢](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | 补充加速条件与投机长度收益递减的推导；新增随负载调整的动态投机长度；结构化输出纳入停止条件的讨论

- **文章：** [5.5 vLLM 投机解码实战：配置、接受率与性能一起测](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | 基准升级到 vLLM v0.32.0；新增第四组 EAGLE-3 实验与 MTP 接入条件；采集脚本改为逐请求接受指标，并补充常见现象的排查顺序与下一轮调参方向

---

## 2026-10-12

### 新增内容

**模块四 · 推理优化** 第 3 章「nano-vllm 源码解读」上线，以约 1200 行的极简推理引擎 [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) 为主线，逐模块读懂连续批处理、PagedAttention、FlashAttention、张量并行与 CUDA Graph 在真实代码里的组织方式：

- **文章：** [3.1 项目全景与四层架构](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/31-项目全景与四层架构/) | 认识 nano-vllm 的定位与目录结构，用一张四层架构图读懂从用户请求到 token 输出的完整数据流

- **文章：** [3.2 配置、入口与一次 generate 的全流程](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/32-配置入口与推理主循环/) | 从 example.py 出发拆解 Config/SamplingParams/LLM/LLMEngine，看懂 generate 主循环如何驱动 add_request、step 与 postprocess

- **文章：** [3.3 模型骨架与核心算子](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/33-模型与核心算子/) | 逐层拆解 Qwen3 实现：RMSNorm、RoPE、SwiGLU、并行 Linear 与张量并行切分、词表并行 Embedding/LMHead，以及 Attention 的 KV 写入与 FlashAttention 两条路径

- **文章：** [3.4 KV Cache 与 BlockManager](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/34-kvcache与blockmanager/) | 拆解分块 KV 管理：Block 结构、PagedAttention 的分块思想、基于 xxhash 的前缀缓存、块的分配回收，以及 Sequence 的状态机与 block_table

- **文章：** [3.5 Scheduler 与连续批处理](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/35-scheduler与连续批处理/) | 读懂调度核心：Waiting/Running 双队列、Prefill/Decode 分离调度、Token Budget 预算、chunked prefill，以及显存不足时的抢占与重计算

- **文章：** [3.6 ModelRunner、CUDA Graph 与性能基准](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第3章-nano-vllm源码解读/36-modelrunner引擎循环与性能/) | 序列张量化、全局 Context 注入、KV Cache 显存切分、CUDA Graph 的捕获与回放，以及官方性能基准与 vLLM 对比

### 调整内容

- 原《vLLM 快速入门》下沉至第 1 章，调整为 [1.2 vLLM 快速入门](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第1章-llm推理基础/12-vllm快速入门/)，作为推理基础的实操入门。

---

## 2026-10-08

### 新增内容

**模块四 · 推理优化** 第 4 章「量化」6 篇文章上线：

- **文章：** [4.1 量化基础：Scale、粒度与 PTQ/QAT](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/41-量化基础/) | 从数值映射理解量化与反量化，讲清位宽、Scale、零点、量化粒度，以及校准和 PTQ/QAT 的区别

- **文章：** [4.2 W8A8 量化：SmoothQuant 如何处理激活离群值](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/42-w8a8与smoothquant/) | 从 Activation Outlier 出发推导 SmoothQuant 的等价缩放，理解平滑系数、INT8 GEMM 与离线部署流程

- **文章：** [4.3 Weight-only INT4：GPTQ、AWQ 与 Marlin](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/43-weight-only-int4/) | 理解 GPTQ 的误差补偿、AWQ 的激活感知缩放，以及 Marlin 如何把 INT4 存储转化为推理收益

- **文章：** [4.4 KV Cache 量化：KIVI 与 FP8 缓存](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/44-kv-cache量化/) | 理解 KV Cache 量化的流式约束、KIVI 对 Key/Value 的不同分组，以及 vLLM FP8 KV Cache 的校准与收益边界

- **文章：** [4.5 FP8 与 NVFP4/MXFP4：低比特浮点怎样工作](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/45-fp8与nvfp4-mxfp4/) | 区分整数与浮点量化，理解 FP8 的范围与精度，以及 NVFP4、MXFP4 的微块缩放与硬件执行条件

- **文章：** [4.6 量化选型与 vLLM 实战](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第4章-量化/46-量化选型与vllm实战/) | 按瓶颈选择量化候选，理解 vLLM 量化格式与 Kernel 的关系，并完成 FP16 与 AWQ INT4 的吞吐和生成质量对照

---

## 2026-10-07

### 新增内容

**模块四 · 推理优化** 第 5 章「Speculative Decoding」5 篇文章上线：

- **文章：** [5.1 投机解码核心原理：先猜一段，再让大模型验收](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/51-投机解码核心原理/) | 从自回归串行瓶颈出发推导 Draft + Verify、接受概率与残差分布，证明最终输出服从 Target 分布

- **文章：** [5.2 Draft 模型与无模型方案：怎样便宜地猜得更准](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/52-draft模型与无模型方案/) | 比较独立小模型、N-gram Prompt Lookup 与 Suffix Decoding，理解 Tokenizer 兼容、历史匹配、候选覆盖与接受率的权衡

- **文章：** [5.3 Medusa 与 EAGLE：复用大模型信息生成更好的草稿](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/53-medusa与eagle/) | 讲清 Medusa 多头预测、EAGLE 特征级草稿、EAGLE-2 动态树与 EAGLE-3 训练改进，理解 Tree Attention 与路径提交

- **文章：** [5.4 投机解码的收益边界：接受率高，为什么仍可能变慢](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/54-收益边界与限制/) | 建立平均提交长度与完整轮耗时的性能账本，分析批处理、长上下文、量化、调度与流式延迟的共同约束

- **文章：** [5.5 vLLM 投机解码实战：配置、接受率与性能一起测](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第5章-speculative-decoding/55-vllm投机解码实战/) | 以 vLLM v0.31.0 为基准，依次启动普通 Decode、N-gram 与独立 Draft，采集接受指标、质量输出与延迟吞吐

---

## 2026-09-15

### 新增内容

**模块四 · 推理优化** 新增第 12 章「端侧推理」：

- **文章：** [12.1 端侧推理基础：从模型导出到异构硬件执行](https://caomaolufei.github.io/AIInfraGuide/inference/模块四-推理优化/第12章-端侧推理/121-端侧推理基础/) | 系统理解端侧推理的硬件、模型格式、运行时、量化、内存与 Benchmark，并通过 ExecuTorch/XNNPACK 走通最小部署链路

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

**模块三 · 分布式训练** 集合通信、优化器与数据并行章节文章上线：

- **文章：** [2.1 集合通信原语详解](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/21-集合通信原语详解/) | 讲清 AllReduce/ReduceScatter/AllGather/All-to-All 的输入输出语义、Ring AllReduce 算法原理，以及单次通信耗时的量化方法

- **文章：** [3.1 优化器：从 SGD 到 AdamW 与显存开销分析](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/31-优化器原理与显存开销分析/) | 梳理优化器从 SGD 到 AdamW 的演进，算清优化器状态的显存开销与 FP32 主权重的必要性，并介绍大 Batch 场景下的 LARS/LAMB

- **文章：** [4.1 数据并行详解](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/41-数据并行详解/) | 深入 DP/DDP/FSDP 三代方案的演进逻辑、梯度同步机制与通信量/显存账本的数学推导，理解 AllReduce 与参数分片的本质区别

- **文章：** [4.2 PyTorch 数据并行从原理到实战](https://caomaolufei.github.io/AIInfraGuide/distributed/模块三-分布式训练/42-pytorch-数据并行从原理到实战/) | 系统掌握 PyTorch 数据并行全栈：通信原语、DDP、FSDP 参数分片与多机多卡实战，附完整代码示例

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
  - **[模块四 · 推理优化](/AIInfraGuide/inference)**（8 章）：推理基础、[推理引擎核心技术](/AIInfraGuide/inference/模块四-推理优化/第2章-推理引擎核心技术)、推理框架、[量化](/AIInfraGuide/inference/模块四-推理优化/第4章-量化)、[Speculative Decoding](/AIInfraGuide/inference/模块四-推理优化/第5章-Speculative-Decoding)、[PD 解耦架构](/AIInfraGuide/inference/模块四-推理优化/第6章-PD解耦架构)、[性能分析与 Benchmark](/AIInfraGuide/inference/模块四-推理优化/第7章-性能分析与Benchmark)、[选型与端到端实战](/AIInfraGuide/inference/模块四-推理优化/第8章-推理优化选型与端到端实战)

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
