---
title: "AI Infra学习路线"
description: "系统梳理 AI Infra 的完整学习路线"
pubDate: 2026-03-26 10:00:00
category: "learning-path"
order: 1
tags: [AI Infra, 学习路线, CUDA, 分布式训练, 推理优化]
---

AI Infra（人工智能基础设施）是大模型时代壁垒最高、最核心的技术高地。本文从前置基础到推理部署，系统梳理 AI Infra 的完整学习路线，为每个模块列出需要掌握的知识点、推荐学习资料以及可量化的检验标准，帮助从业者建立体系化的知识树。

**作者将根据该路线编写系列文章，帮助大家系统掌握 AI Infra 技术。**

<!-- more -->

## 📑 目录

- [全景概览：三层架构](#️-全景概览三层架构)
- [第零层：前置知识](#-第零层前置知识)
- [第一层：CUDA编程与算子优化](#-第一层cuda编程与算子优化)
- [第二层：分布式训练](#️-第二层分布式训练)
- [第三层：推理与部署](#-第三层推理与部署)
- [新人破局指南](#-新人破局指南)
- [参考资料](#-参考资料)

---

## 🗺️ 全景概览：三层架构

AI Infra 的本质是 **"用系统工程释放硬件算力"** 。我们可以将其自底向上分为三个核心层级，加上一个前置知识层：

| 层级 | 名称 | 核心关注点 |
|------|------|-----------|
| 第零层 | 前置知识 | 编程语言、数学基础、Transformer 架构、PyTorch、通信拓扑 |
| 第一层 | CUDA编程与算子优化 | GPU架构、存储层次、Kernel编写、FlashAttention、AI编译器 |
| 第二层 | 分布式训练 | 数据并行、3D并行、ZeRO、混合精度 |
| 第三层 | 推理与部署 | KV Cache、PagedAttention、量化、Speculative Decoding |

所有的优化都是在 **"计算、通信、显存"** 这个不可能三角中做取舍：ZeRO 是用通信换显存；重计算（Activation Checkpointing）是用计算换显存；量化是用精度换显存和带宽。学习时始终问自己：**这个技术牺牲了什么，换取了什么？**

---

## 📖 第零层：前置知识

AI Infra 不是从零开始学的领域——它建立在编程能力、数学直觉和模型理解之上。如果把后面四层比作盖楼，这一层就是地基：地基不牢，楼层越高越晃。好消息是，你不需要成为每个方向的专家，只需要达到"够用"的水平就能继续往上走。

### 0.1 知识点

**编程语言**

- **Python** ：AI 生态的通用语言，需要熟练使用而非仅仅"会写"。重点掌握：面向对象、装饰器、生成器、多进程/多线程、性能 profiling
- **C/C++** ：CUDA 编程的宿主语言。不要求精通模板元编程，但需要理解指针、内存管理、编译链接过程。能读懂 C++ 项目代码、写简单的 C++ 函数并编译运行即可
- **Linux 基础** ：命令行操作、Shell 脚本、进程管理、环境变量配置。AI Infra 的开发和部署几乎全部在 Linux 上进行

**数学基础**

- **线性代数**：矩阵乘法、转置、分块矩阵运算、特征值分解的基本概念。理解"为什么 Transformer 的核心计算都是矩阵乘法"需要这个基础。不需要证明定理，但需要对矩阵维度变换有直觉——看到 (B, S, H) x (H, V) 能立刻知道结果是 (B, S, V)
- **基础概率论与统计**：概率分布、期望、方差、Softmax 的概率解释、交叉熵损失的含义。量化和 Speculative Decoding 的正确性证明都依赖概率论
- **微积分**（了解即可）：链式法则、梯度的含义。理解反向传播为什么能工作、混合精度训练中梯度为什么会溢出，需要这个基础

**Transformer 架构**

这是大模型时代的"通用底座"，后续每一层都在围绕它做优化——CUDA 层优化它的算子、分布式层切分它的参数、推理层加速它的生成。必须理解：

- **Self-Attention 机制**：Q、K、V 的含义与计算过程（QK^T → scale → softmax → PV），理解为什么 Attention 的计算复杂度是 O(N^2)，这是后续理解 FlashAttention 优化的前提
- **前馈网络（FFN）**：两层线性变换 + 激活函数，模型参数的大头所在
- **位置编码**：Sinusoidal、RoPE 等，理解为什么 Transformer 需要额外的位置信息
- **LayerNorm**：Pre-Norm vs Post-Norm 的区别，为什么大模型普遍用 Pre-Norm
- **完整前向过程**：能从 token embedding 开始，逐步跟踪数据在一个 Transformer Block 中的流转（Attention → Add & Norm → FFN → Add & Norm），说清每一步的输入输出维度

**PyTorch 框架**

- **核心概念**：Tensor 操作、自动微分（autograd）、Module / Parameter 的组织方式
- **训练循环**：DataLoader → forward → loss → backward → optimizer.step 的标准流程
- **模型保存与加载**：state_dict、checkpoint 的使用
- **基本调试**：用 `torch.cuda.memory_summary()` 查看显存使用、用 `torch.profiler` 做简单性能分析

**通信拓扑**

分布式训练和多卡推理都离不开高效的卡间、机间通信。这部分知识是后续理解分布式并行策略的前提——如果不知道 NVLink 和 PCIe 的带宽差了一个数量级，就无法理解"为什么张量并行不能跨机"。

- **单机内部通信**：NVLink / NVSwitch 的带宽与拓扑。NVLink 是同一机箱内 GPU 之间的专用高速通道，可以把它想象成工厂园区内的专用传送带
- **多机间通信**：InfiniBand（IB）网络、RoCE 协议。IB 则是跨城市的高速铁路，带宽比 NVLink 低但能连接远距离节点
- **集合通信原语**：AllReduce、AllGather、ReduceScatter 的含义与开销——这是分布式训练中最频繁的操作，理解它们的通信量公式是分析并行策略开销的基础
- **NCCL**：NVIDIA 集合通信库的基本用法与调优

### 0.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | Attention Is All You Need | Transformer 原始论文，必读 |
| 教程 | The Illustrated Transformer (Jay Alammar) | 图文并茂的 Transformer 入门，适合建立直觉 |
| 解读 | 琳琅阿木：图文详解LLM inference | LLM 模型架构详解，推荐配合论文阅读 |
| 教程 | PyTorch 官方教程（60 Minute Blitz） | PyTorch 快速入门 |
| 书籍 | 3Blue1Brown：线性代数的本质（视频系列） | 建立线性代数几何直觉，比教科书高效 |
| 教程 | MIT 6.S081 / Linux 命令行基础 | Linux 和系统编程基础 |
| 工具 | Andrej Karpathy：Let's build GPT from scratch | 从零手写 GPT，把 Transformer 每个模块都过一遍 |
| 官方文档 | NVIDIA NCCL 文档 | 集合通信原语与多卡编程 |
| 教程 | NVIDIA Deep Learning Performance Guide | 硬件性能瓶颈分析方法论 |

### 0.3 检验标准

这一层不需要深入钻研，但必须达到以下"够用"门槛：

- **Transformer 白板默写**：不看资料，能画出一个 Transformer Decoder Block 的完整结构（Masked Self-Attention → Add & Norm → FFN → Add & Norm），标注每一步的输入输出维度（如 (B, S, D) → (B, S, D)），并解释 Attention 中 Q、K、V 矩阵是怎么从输入通过线性投影得到的
- **维度推导**：给定一个 7B 模型的配置（hidden_dim=4096, num_heads=32, num_layers=32, vocab_size=32000），能手算出总参数量的近似值（Attention 权重 + FFN 权重 + Embedding），误差不超过 20%
- **PyTorch 训练脚本**：能独立写出一个完整的 PyTorch 训练循环（不借助 Trainer 类），包含 DataLoader、forward、loss 计算、backward、optimizer step、学习率调度、checkpoint 保存，并在 GPU 上跑通
- **C++ 基础读写**：能读懂一个简单的 CUDA kernel 的 host 端代码（malloc、memcpy、kernel launch、free），理解 CPU 和 GPU 之间的数据搬运流程
- **Linux 日常**：能在服务器上独立完成：SSH 登录、tmux 管理会话、conda/pip 管理环境、查看 GPU 状态（nvidia-smi）、用 git 管理代码、写简单的 bash 脚本批量提交任务
- **拓扑感知**：在一台 8 卡机器上执行 `nvidia-smi topo -m`，能看懂输出矩阵里的 NV12、SYS、NODE 等标记，判断哪些卡之间走 NVLink、哪些走 PCIe
- **集合通信直觉**：能画出 AllReduce、AllGather、ReduceScatter 三种操作的数据流示意图，说清每种操作的通信量公式（如 Ring AllReduce 的通信量 ≈ 2(N-1)/N × 数据量），理解为什么 NVLink 和 IB 的带宽差距直接决定了哪些并行策略能跨机

---

## 💻 第一层：CUDA编程与算子优化

这一层是连接硬件和软件的桥梁，负责把高层的数学计算翻译成 GPU 能最高效执行的机器指令。在动手写 kernel 之前，必须先理解 GPU 的硬件架构——这就像给工人写工序手册之前，得先搞清楚工厂有多少工人、车间怎么布局、货架在哪里。

### 1.1 知识点

**GPU 硬件架构**

理解 GPU 硬件是写好 CUDA kernel 的前提。你可以把一块 GPU 想象成一座**拥有数千个简单工人的超级工厂**——每个工人（CUDA Core）只会做最基本的加减乘除，但胜在人多力量大，成千上万人同时开工，吞吐量远超只有几个高级工程师（CPU 核心）的小作坊。

- **GPU 核心架构**：SM（流多处理器）、Tensor Core、CUDA Core 的区别与协作
- **主流 GPU 规格对比**：A100 / H100 / H200 的算力、显存带宽、HBM 容量
- **Memory Wall**：为什么显存带宽瓶颈往往比算力瓶颈更致命——好比一个超级快的厨师，刀工和火候都没问题，但食材传送带太慢，厨师大部分时间都在等菜上桌
- **存储层次结构**：寄存器 > 共享内存 > L1/L2 Cache > HBM > 主机内存。写 kernel 的核心就是在这个层次结构中最大化数据复用，减少对慢速存储的访问

**CUDA 编程基础**

- 编程模型：Grid / Block / Thread 层级，线程索引计算
- 内存模型：全局内存、共享内存、寄存器、常量内存的特性与用法
- 关键概念：Warp（32 个线程组成的最小调度单位，GPU 每次下达指令都是以 Warp 为单位，就像军队里以"班"为单位行动）、Bank Conflict、Coalesced Access、Occupancy
- 核心直觉："内存访问模式决定运行速度"

**常见算子实现与优化**

- Reduce：并行归约的多种实现与优化（Warp Shuffle、多级归约）
- GEMM：矩阵乘法的分块、向量化、Shared Memory Tiling、利用 Tensor Core
- Softmax：Online normalizer calculation，高效 Softmax kernel
- 算子融合：将多个小算子合并为一个 kernel，减少全局内存读写

**Attention 算子**

- FlashAttention V1/V2：Memory-aware 的精确 Attention 实现，通过 tiling 减少 HBM 访问——好比把一张大桌子上的拼图分成小块，每次只搬一小块到手边拼好再搬下一块，而不是把所有碎片一股脑倒出来占满桌面
- FlashAttention-3：在 Hopper 架构上进一步拉高利用率
- Flash-Decoding / FlashDecoding++：面向 Decode 阶段的 Attention 加速
- FlashInfer：可定制 Attention 引擎，面向 Serving 的可组合格式与异构 KV 存储
- PagedAttention CUDA Kernel：vLLM 中 PagedAttention 的底层实现

**AI 编译器**

- Triton：OpenAI 开源的 GPU 编程语言，大幅降低高效算子编写门槛
- TVM / XLA：计算图优化与代码生成
- `torch.compile`：PyTorch 2.x 的编译模式，理解 Graph Break 与性能收益

### 1.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 官方文档 | NVIDIA GPU 架构白皮书（Ampere / Hopper） | 理解 SM、Tensor Core、HBM 设计 |
| 入门教程 | 小小将：CUDA编程入门极简教程 | CUDA 零基础入门 |
| 官方文档 | NVIDIA CUDA Programming Guide | CUDA 编程权威参考 |
| Reduce | PeakCrosser：CUDA Reduce 算子优化 | Reduce 实现与优化的详尽总结 |
| GEMM | 猛猿：从啥也不会到CUDA GEMM优化 | 从基础分块到极致优化的 GEMM 教程 |
| GEMM | MegEngine Bot：CUDA 矩阵乘法终极优化指南 | 系统性的 GEMM 优化参考 |
| Softmax | Online normalizer calculation for softmax | NVIDIA 员工的 Softmax 实现论文 |
| Softmax | OneFlow：如何实现一个高效的Softmax CUDA kernel | Softmax kernel 工程实践 |
| 算子融合 | 成诚：OneFlow是如何做到世界最快深度学习框架的 | 算子融合思路与方法 |
| Attention | FlashAttention V1 Paper | Memory-aware Attention 的里程碑论文 |
| Attention | FlashAttention V2 Paper | 更好的并行与分块策略 |
| Attention | FlashAttention-3 Paper | Hopper 架构上的进一步优化 |
| Attention | Flash-Decoding 技术报告（Stanford CRFM） | Decode 阶段的 Attention 加速 |
| Attention | FlashInfer Paper + Repo | 可组合的 Attention 引擎 |
| 解读 | 猛猿：图解FlashAttention V1/V2 系列 | 适合新手入门的图文解读 |
| 解读 | 方佳瑞：深入浅出理解PagedAttention CUDA实现 | vLLM PagedAttention kernel 图文解读 |
| 编译器 | Triton 官方教程 | GPU 编程新范式 |
| 编译器 | PyTorch profiling torch.compile | 抓 Graph Break、编译收益与损耗 |
| 工具 | Nsight Systems User Guide | CPU-GPU 交互分析，判断 host 是否拖后腿 |
| 工具 | Nsight Compute Profiling Guide | Kernel 级下钻，定位 SM / Memory / Tensor Core 瓶颈 |

### 1.3 检验标准

动手是检验这一层的唯一标准，纸上谈兵不算数：

- **硬件参数直觉**：拿到一块 H100，不查资料能说出 HBM 容量（80GB）、HBM 带宽（~3.35TB/s）、L2 大小（50MB）、共享内存上限（228KB/SM）的量级，并解释为什么"显存带宽"往往比"算力"先成为瓶颈
- **带宽估算**：给定一个 AllReduce 操作的数据量（比如 2GB 梯度），能估算在 NVLink（900GB/s per GPU）vs PCIe Gen5（64GB/s）下的理论耗时差异

- **Reduce 三连**：从最朴素的全局内存原子加开始，写一个 Reduce Sum kernel；然后用共享内存 + 树形归约消除原子操作；最后用 Warp Shuffle 干掉共享内存，三个版本跑 Nsight Compute 对比 throughput，能说清每一步优化到底省在哪里
- **Bank Conflict 直觉**：手动构造一个 32x32 矩阵转置 kernel，先写一个有 32-way bank conflict 的版本，再加一列 padding 消除冲突，用 Nsight Compute 的 Shared Memory 面板验证 conflict 数从几十降到 0
- **GEMM 分块**：实现一个基于 Shared Memory Tiling 的 GEMM kernel，在 1024x1024 矩阵上与 cuBLAS 对比，达到其 50% 以上的性能即为合格——这个过程中你会真正理解"为什么访存模式决定一切"
- **FlashAttention 白板推导**：不看论文，能在白板上画出 FlashAttention 的 tiling 过程——外层循环遍历 KV 的 block，内层循环遍历 Q 的 block，每个 tile 在 SRAM 中完成 QK^T → scale → mask → softmax → PV，用 online softmax 避免两次遍历，关键是说清楚为什么 HBM 读写从 O(N^2) 降到了 O(N)
- **Triton 上手**：用 Triton 实现一个 fused Softmax kernel（参考官方教程），与 PyTorch 原生实现对比正确性和性能，体会 Triton 的 block-level 编程模型与 CUDA 的 thread-level 编程模型有何不同
- **Profiling 实战**：用 Nsight Systems 抓一次训练 iteration 的 trace，能指出 GPU idle gap 是来自 CPU 数据预处理、通信等待、还是 kernel launch overhead；用 Nsight Compute 打开一个 kernel 报告，能读懂 SOL（Speed of Light）面板判断该 kernel 是 memory bound 还是 compute bound

---

## 🏋️ 第二层：分布式训练

当模型参数量超越单卡显存极限时，分布式训练就是必经之路。这是 AI Infra 目前最活跃、最核心的区域。打个比方，训练一个千亿参数的大模型就像**抄写一本数万页的百科全书**——一个人抄到天荒地老也抄不完。数据并行是把同一本书复印多份、每人抄不同章节的内容然后汇总；张量并行是把每一页拆成几列、每人只抄自己那几列；流水线并行则是第一个人抄完第一章就传给第二个人继续，自己接着抄下一批。怎么拆、怎么传、怎么汇总，就是分布式训练要解决的核心问题。

### 2.1 知识点

**模型架构演进（分布式视角）**

第零层已经介绍了标准 Transformer 的结构，这里关注的是那些直接影响分布式切分策略和通信开销的架构变种：

- Attention 变种：MHA → MQA → GQA → MLA 的演进。MQA/GQA 通过共享 KV Head 减少 KV Cache 大小，MLA（DeepSeek V2）用低秩压缩进一步降低 KV 显存——这些变种直接影响 TP 切分方式和推理显存规划
- FFN 变种：混合专家模型 MoE（DeepSeekMoE）。MoE 的稀疏激活特性让并行策略从"切矩阵"变成"分专家"，引入 Expert Parallelism 这一新维度

**数据并行**

- DP（DataParallel）：最基础的数据并行，单进程多卡
- DDP（DistributedDataParallel）：多进程数据并行，理解 AllReduce 梯度同步
- FSDP（Fully Sharded Data Parallel）：PyTorch 原生的 ZeRO-3 实现

**模型并行（3D 并行）**

- 张量并行（TP）：将矩阵乘法沿特定维度切分到多卡，通信密集，通常限于单机
- 流水线并行（PP）：将模型不同层切分到不同机器，像流水线一样传递数据
- 序列并行（SP）：沿序列维度切分，与 TP 配合减少激活显存

**显存优化**

- ZeRO 系列（DeepSpeed）：把训练状态拆开分摊到各张卡上，好比合租房里每人只存自己那份家具，需要时再互相借用，以此腾出更多空间：
  - ZeRO-1：优化器状态切分
  - ZeRO-2：优化器状态 + 梯度切分
  - ZeRO-3：优化器状态 + 梯度 + 参数切分（用通信换显存）
- 混合精度训练：FP16 / BF16 / FP8 训练，减少显存占用与计算开销。相当于平时用草稿本（低精度）算题提高速度，只在最关键的步骤用正式答题纸（高精度）保证结果准确
- 梯度累积：在有限显存下模拟更大的有效 Batch Size
- Activation Checkpointing（重计算）：用计算换显存，只保存部分激活值，需要时重新算一遍。好比考试时不把每道草稿都留着占桌面，只记住关键中间结果，需要时重新推导一遍

**训练框架**

- Megatron-LM：张量并行与流水线并行的标杆实现
- DeepSpeed：ZeRO 系列的核心实现，丰富的训练优化工具集
- PyTorch FSDP：原生分布式训练方案

### 2.2 推荐资料

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | DeepSeek V2 技术报告 | MLA 注意力机制 |
| 论文 | DeepSeekMoE Paper | MoE 架构设计 |
| 教程 | 混合专家模型 (MoE) 详解 | MoE 入门 |
| 解读 | 苏剑林：从MHA、MQA、GQA到MLA | Attention 变种演进 |
| 论文 | Megatron-LM Paper | TP 与 PP 原理的里程碑论文 |
| 论文 | ZeRO Paper（DeepSpeed） | 显存优化的核心方法 |
| 文档 | DeepSpeed 官方文档 | ZeRO 配置与使用 |
| 文档 | PyTorch DDP / FSDP 教程 | 原生分布式训练入门 |

### 2.3 检验标准

这一层的检验核心是**算得清账、跑得通代码**：

- **显存账本**：拿到一个 7B 参数的模型（如 LLaMA-2-7B），不查资料能口算出 FP16 下参数占 ~14GB、Adam 优化器状态占 ~56GB（FP32 参数副本 + 一阶/二阶动量各 14GB），进而判断单卡 80GB 能否放下完整训练状态、是否必须上 ZeRO
- **ZeRO 拆解**：有人问你"ZeRO-2 和 ZeRO-3 到底差在哪"，你能一句话讲清：ZeRO-2 只在 backward 时按需 AllReduce 梯度，参数每卡各存一份；ZeRO-3 连参数也切了，forward/backward 都要 AllGather 拿参数、用完即弃，通信量约翻倍但每卡显存降到 1/N
- **DDP 改造**：拿到一个单卡 PyTorch 训练脚本，30 分钟内改成 DDP 多卡版本并跑通——包括 `init_process_group`、`DistributedSampler`、模型 wrap、梯度同步，不需要查太多文档就能搞定
- **3D 并行拓扑**：给一个 64 卡集群（8 节点 x 8 卡），能设计出 TP=8（机内）、PP=4（跨机）、DP=2 的并行方案，画出拓扑图标注哪些通信走 NVLink、哪些走 IB，并解释为什么 TP 不能跨机（带宽不够）
- **混合精度原理**：能回答"BF16 和 FP16 都是 16 位，为什么大模型训练更偏爱 BF16"——因为 BF16 的指数位更宽（8 位 vs 5 位），动态范围接近 FP32，不容易 overflow/underflow，大多数情况下可以不做 Loss Scaling

---

## 🚀 第三层：推理与部署

训练只是万里长征第一步。如何让模型快速、低成本地服务用户，是工业界最关心的问题。如果说训练是"教会模型知识"，那么推理就是"让模型上考场答题"——考场上最重要的是**答题速度**和**同时服务多少考生**，而且考试时把已算过的中间结果记在草稿纸上（KV Cache）能避免重复计算，大幅提速。

### 3.1 LLM 推理基础

**知识点**

- LLM 推理的两阶段：Prefill（处理输入）与 Decode（逐 token 生成）
- KV Cache：自回归生成中的"显存刺客"，理解其生命周期和碎片问题。KV Cache 就像考试时的草稿纸——把已经算过的中间结果记下来，后续生成每个新 token 时就不用从头算起，但草稿纸用多了也会占满整张桌子
- 关键性能指标：TTFT（首 token 延迟）、TPOT（每 token 延迟）、吞吐量（token/s）、P50/P95 尾延迟

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 解读 | 琳琅阿木：图文详解LLM inference——KV Cache | KV Cache 原理详解 |
| 综述 | Towards Efficient Generative LLM Serving: A Survey | CMU 的 LLM 推理综述（算法 + 系统） |

**检验标准**

把自己当成一个刚拿到推理需求的工程师，回答以下问题：

- **两阶段直觉**：同事问你"为什么 Prefill 快但 Decode 慢"，你能秒答：Prefill 一次处理整个 prompt，矩阵大、算力利用率高（compute bound）；Decode 每步只生成一个 token，矩阵退化成向量，大部分时间在等显存搬运 KV Cache（memory bound）
- **KV Cache 算账**：给定 LLaMA-2-7B（32 层、32 头、head_dim=128），上下文长度 4096，batch_size=16，FP16 存储，能手算 KV Cache 总量 = 2 × 32 × 32 × 128 × 4096 × 16 × 2B ≈ 32GB，进而判断 80GB 显卡还剩多少空间给模型参数和临时 buffer
- **链路拆解**：把一次推理请求从头到尾拆成 tokenize → prefill（GEMM 密集）→ decode loop（逐 token 生成，memory bound）→ sampling（Top-p/Top-k）→ detokenize，能指出在高并发场景下哪个环节最容易成为瓶颈（通常是 decode 阶段的 KV Cache 带宽）

### 3.2 推理引擎

**知识点**

- PagedAttention：vLLM 提出的虚拟内存分页思想管理 KV Cache，解决碎片化问题——就像操作系统把内存切成固定大小的页来管理一样，不再要求一整块连续空间，碎片问题迎刃而解
- Continuous Batching：动态组批，请求随到随处理，与传统 static batching 的差异。传统方式像旅游大巴——人齐了才发车，先到的人干等；Continuous Batching 更像网约车拼单，随到随拼、有人下车立刻补新客
- Prefix Cache / RadixAttention：复用已计算的 KV Cache，优化重复前缀场景
- Chunked Prefill：将长 prompt 分块处理，减少 prefill 对 decode 的干扰

**主流推理框架**

| 框架 | 核心特性 | 适用场景 |
|------|---------|---------|
| vLLM | PagedAttention、Continuous Batching、Prefix Cache | 通用推理服务，社区活跃 |
| SGLang | RadixAttention、cFSM 结构化输出加速 | 复杂 Agent、多轮生成、结构化输出 |
| TensorRT-LLM | Paged KV、Inflight Batching、深度硬件优化 | 追求极限性能、NVIDIA 生态 |

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | vLLM Paper（PagedAttention） | 推理引擎里程碑论文 |
| 源码 | vLLM GitHub Repo | 实战特性全集 |
| 文档 | vLLM 官方文档 | 部署与配置 |
| 解读 | 猛猿：vLLM 源码解析系列（架构/调度器/BlockManager） | 深入理解 vLLM 内部机制 |
| 论文 | SGLang Paper | RadixAttention + 前端语言 |
| 源码 | SGLang GitHub Repo | 工程实现 |
| 文档 | TensorRT-LLM 官方文档 | 工程落地导向 |
| 论文 | Orca Paper | Continuous Batching 的原始论文 |
| 解读 | 吃果冻不吐果冻皮：Continuous Batching | Continuous Batching 中文解读 |
| 解读 | DefTruth：vLLM Prefix Cache 原理图解 | Prefix Cache 万字详解 |

**检验标准**

推理引擎是工程落地的核心，检验标准偏重"会用、会选、会排查"：

- **端到端部署**：拿到一个 7B 模型，分别用 vLLM 和 SGLang 部署成 OpenAI 兼容 API，用相同的压测脚本（固定并发、输入输出长度）跑出 TTFT、TPOT、throughput 三个指标的对比表
- **Continuous Batching 原理**：能向非技术同事解释清楚——传统 Static Batching 必须等一批请求全部生成完才能接新请求，短请求被长请求拖累；Continuous Batching 允许已完成的请求随时退出、新请求随时插入，GPU 利用率可以从 30% 拉到 80%+
- **KV Cache 管理全链路**：从 KV Cache 的分配、使用、碎片化，到 PagedAttention 的虚拟页/物理页映射，再到 Prefix Cache 如何通过 hash 匹配复用已有的 KV 块，能画出完整的数据流
- **选型决策**：老板问"我们该用哪个推理框架"，你能给出结构化的回答——追求吞吐和社区生态选 vLLM；多轮对话 / Agent / 结构化输出选 SGLang（RadixAttention + cFSM）；追求极限延迟且愿意投入适配工作选 TensorRT-LLM

### 3.3 量化

**知识点**

- W8A8（SmoothQuant）：将 activation 的 outlier 难题转移到 weights，工程友好。量化的本质好比把高清照片压缩成缩略图——用更少的比特位表示权重，省下显存和带宽，代价是细节（精度）会有一定损失
- Weight-only INT4（GPTQ / AWQ）：只量化权重到 3/4-bit，减少显存和带宽占用
- KV Cache 量化（KIVI / Kitty）：对 KV Cache 进行 2-bit 量化，长上下文场景效果显著
- FP8 量化：Hopper 架构原生支持，精度与性能的平衡点
- 量化选择决策树：

```
目标：省显存？省带宽？提吞吐？
├─ 通用、工程友好 → W8A8 (SmoothQuant)
├─ 更省显存/带宽 → INT4 weight-only (AWQ/GPTQ)
└─ 长上下文/大并发 → KV Cache 量化 (KIVI)
```

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | SmoothQuant Paper | W8A8 量化的经典方案 |
| 论文 | GPTQ Paper | Weight-only PTQ 3/4-bit |
| 论文 | AWQ Paper | 基于 activation 分布的 4-bit 量化 |
| 论文 | KIVI Paper | 2-bit KV Cache 量化 |
| 论文 | Marlin Paper（FP16xINT4） | INT4 高性能 matmul kernel |
| 代码 | TensorRT-LLM 量化工具链 | FP8/INT4/AWQ/SmoothQuant 集成 |
| 代码 | vLLM 量化支持 | GPTQ/AWQ/FP8/INT8 多种接入 |

**检验标准**

量化是"看起来简单、调起来玄学"的领域，检验重点在于理解 trade-off 而非死记方案：

- **方案选择**：团队要上线一个 70B 模型但只有 2 张 A100 80GB，你能快速判断：FP16 参数就要 140GB 放不下，必须量化；W8A8 可以压到 ~70GB 但两卡还是紧张；INT4 weight-only 可以压到 ~35GB 单卡就能放下，但要评估精度损失和 kernel 效率
- **动手验证**：用 vLLM 或 TensorRT-LLM 分别加载同一模型的 FP16 和 AWQ-INT4 版本，跑同一组 benchmark prompt，输出 throughput 对比 + 几组生成结果的人工比对，能写成一个可复用的评测报告
- **反直觉理解**：别人说"量化位数越低越快"，你能解释为什么某些场景下 INT4 反而比 INT8 慢——INT4 kernel 需要额外的 unpack/dequant 计算，在小 batch size 下这个开销可能超过带宽节省；或者 INT4 的 Tensor Core 利用率不如 INT8 的 native 支持
- **故障排查**：量化后模型输出质量明显下降，你能列出排查清单——某些层的 activation outlier 特别大（需要 per-channel 或 SmoothQuant 处理）、量化校准数据不具代表性、某些特殊结构（如 MoE 的 gate）不适合低 bit 量化

### 3.4 Speculative Decoding

**知识点**

- Speculative Sampling：经典框架——用小模型（Draft）批量"猜测"多个 token，大模型（Target）一次性验证，保证分布无偏。好比让实习生先快速起草一段文字，再让资深主编一次性审阅：猜对的直接用，猜错的当场改，比主编逐字逐句从头写快得多
- Medusa：不用外部 Draft 模型，通过多个 Decoding Heads 预测多 token 再并行验证
- EAGLE-2：动态 Draft Tree，靠校准置信度更激进地产生可接受 token
- Block Verification：将 token 级验证升级为 block 级联合验证，进一步提速

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | Speculative Sampling Paper | 经典 Speculative Decoding |
| 论文 | Medusa Paper + Repo | 多头解码，免 Draft 模型 |
| 论文 | EAGLE-2 Paper | 动态 Draft Tree |
| 论文 | Block Verification Paper | Block 级联合验证 |
| 代码 | vLLM / TensorRT-LLM 的 Speculative 支持 | 工程落地参考 |
| 教程 | SGLang 结构化输出加速（cFSM） | 结构化生成场景加速 |

**检验标准**

Speculative Decoding 的精髓在于"用空间换时间、用并行换串行"，检验时重点考察对正确性和收益边界的理解：

- **正确性保证**：面试官问"Speculative Decoding 会不会改变模型的输出分布"，你能回答：不会，因为 rejection sampling 机制保证了接受的 token 严格服从 target model 的分布——Draft model 猜对了直接用，猜错了按照 target 与 draft 的概率差做修正采样，数学上等价于直接从 target model 采样
- **收益实测**：用 vLLM 的 speculative decoding 功能，分别在代码生成（高接受率场景）和开放对话（低接受率场景）上跑 benchmark，能解释为什么前者加速明显（代码 token 可预测性强，Draft 猜中率高）而后者收益有限甚至为负
- **不赚的边界**：能列出至少 3 种 speculative decoding 效果不佳的情况——高温度采样时 Draft 命中率骤降、batch size 已经很大时额外的 Draft forward pass 抢占计算资源、Draft model 与 Target model 能力差距过大导致 acceptance rate < 50%
- **工程耦合风险**：能指出 speculative decoding 与其他优化技术的冲突点——比如与量化叠加时 Draft model 的精度下降可能进一步降低接受率；与 Continuous Batching 叠加时不同请求的 speculative 长度不同，调度复杂度上升

### 3.5 系统架构：Prefill/Decode 解耦

**知识点**

- 核心问题：Prefill 与 Decode 混合 batching 造成资源耦合与互扰，导致尾延迟爆炸。这就像餐厅里让同一个厨师既负责快速出小炒（Decode 低延迟）又负责慢炖大菜（Prefill 重计算），互相拖累；解耦就是把快餐区和慢炖区分开，各配专属厨师
- DistServe（OSDI'24）：系统化论证并实现 Prefill/Decode 解耦，围绕 goodput 调度
- Splitwise（ISCA）：将 Prefill 和 Decode 分配到不同 GPU 池，优化吞吐与成本
- TaiChi（2025）：将聚合与解耦统一，面向不同 SLO 组合做最优 goodput
- Goodput：满足 SLO 的有效吞吐，区别于裸 QPS

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 论文 | DistServe（OSDI'24） | Prefill/Decode 解耦的系统化论证 |
| 论文 | Splitwise（ISCA） | Phase Splitting 设计 |
| 论文 | TaiChi（2025） | 聚合与解耦统一框架 |
| 教程 | Disaggregated Inference: 18 Months Later | 解耦为何成为默认配置 |
| 教程 | MLC microserving | 跨引擎编排的可编程 API |

**检验标准**

Prefill/Decode 解耦是系统架构层面的优化，检验重点在于理解"为什么要拆"以及"拆了之后新的问题是什么"：

- **互扰定量分析**：在一个混合 batching 的推理服务上，构造一个场景——几个超长 prompt 的 prefill 请求和大量短 decode 请求同时到达，用指标证明 decode 的 P95 TPOT 被 prefill 拖慢了 3-5 倍，这就是解耦的动机
- **Goodput 概念**：老板问"我们系统 QPS 很高啊为什么用户还在抱怨慢"，你能解释 goodput 的含义——满足 SLO（比如 TTFT < 500ms 且 TPOT < 50ms）的有效请求占比才是真正的服务质量指标，raw QPS 不等于用户体验
- **资源配比推导**：给定一个工作负载特征（平均 prompt 长度 2000 token、平均输出 500 token），能估算 prefill 和 decode 的计算量比例，进而推导出 Prefill GPU 池和 Decode GPU 池的合理配比（比如 1:3 或 1:4）
- **风险清单**：能列出解耦架构引入的新问题——KV Cache 从 Prefill 节点迁移到 Decode 节点的网络带宽压力（一个 7B 模型 2048 长度的 KV 约 4GB，IB 200Gb/s 也需要 ~160ms）、调度器的队列管理复杂度、Prefill/Decode 负载不均时某一池空转浪费资源

### 3.6 性能分析与 Benchmark

**知识点**

- 核心指标体系：QPS、TTFT、TPOT、token/s、P50/P95 尾延迟
- 性能分析工具链：
  - `torch.profiler`：PyTorch 官方 profiler，定位算子与 shape
  - Nsight Systems：CPU-GPU 交互全链路分析
  - Nsight Compute：Kernel 级性能下钻
- 压测工具：GenAI-Perf（LLM 指标一站式输出）、Triton Perf Analyzer
- 权威基准：MLPerf Inference（Datacenter），统一口径与规则
- 回归门禁：每次改动输出同一张指标对比表，退化则阻止合并

**推荐资料**

| 类型 | 资料 | 说明 |
|------|------|------|
| 工具 | torch.profiler 文档 | PyTorch 性能分析起点 |
| 工具 | GenAI-Perf | TTFT/TPOT/token throughput 一站式压测 |
| 工具 | Triton Inference Server Quickstart | 在线推理容器化基线 |
| 基准 | MLPerf Inference（Datacenter） | 权威 benchmark 入口 |
| 解读 | MLPerf Inference v5.0 LLM 任务解读 | 理解低延迟 LLM benchmark 趋势 |

**检验标准**

性能分析不是"会用工具"就行，核心是建立**可重复、可对比、可追溯**的工程体系：

- **指标全集**：每次性能评测输出的报告至少包含 6 个指标——QPS、TTFT（P50/P95）、TPOT（P50/P95）、端到端 throughput（token/s）、GPU 显存占用峰值、GPU 利用率，缺任何一个都可能遗漏瓶颈
- **Benchmark 可复现**：你的 benchmark 配置（模型、batch size、输入输出长度、并发数、硬件型号、驱动版本）全部写在一个 config 文件里，任何人拿到这个文件都能复现你的结果，两次测试之间只改一个变量
- **回归定位**：某次代码提交后 TPOT P95 退化了 15%，你能用 `git bisect` 缩小到具体 commit，再用 Nsight Systems 对比退化前后的 trace 差异——是某个 kernel 变慢了、新增了一次不必要的 sync、还是 batch 调度策略变了
- **上线门禁**：能给团队制定一套简单可执行的性能门禁规则——比如"TPOT P95 退化超过 5% 则 block merge"、"显存占用增长超过 10% 需要附上分析报告"，并集成到 CI 流程中自动执行


### 3.7 优化选型决策树

推理优化不是"把所有技术都堆上去"就完事了，而是**先定位瓶颈，再对症下药**。好比去医院看病，医生不会上来就开药，而是先问"哪里不舒服"，再做检查确认病因，最后才开处方。下面这棵决策树就是推理优化的"问诊流程"——从症状出发，逐步缩小到具体方案。

**第一步：确认主要症状**

拿到推理性能问题后，先用压测工具（GenAI-Perf 或自定义脚本）跑出基线数据，看哪个指标最不达标，然后对号入座：

```
症状是什么？
│
├─ [A] TTFT 过高（用户等首 token 太久）
├─ [B] TPOT 过高（token 一个一个蹦，体感卡顿）
├─ [C] 显存不够（OOM 或无法提高并发）
└─ [D] 尾延迟 P95/P99 失控（均值还行但毛刺严重）
```

**[A] TTFT 过高——首 token 迟迟不来**

TTFT 主要受 Prefill 阶段影响。Prefill 需要一次性处理完整个 prompt，是典型的 compute-bound 操作。定位思路：

| 子症状 | 诊断方法 | 推荐方案 |
|--------|---------|---------|
| prompt 很长（数千到数万 token） | 观察 prefill 耗时是否随 prompt 长度线性甚至超线性增长 | Chunked Prefill 分块处理，避免长 prefill 独占 GPU；升级到更高效的 GEMM kernel |
| CPU 或调度成为瓶颈 | 用 Nsight Systems 抓 trace，看 GPU idle gap 是否由 host 端引起 | 排查 tokenizer、数据预处理等 CPU 开销；考虑换用调度更成熟的推理引擎 |
| 大量请求共享相同前缀（如系统 prompt） | 统计请求的 prompt 前缀重复率 | 开启 Prefix Cache（vLLM）或 RadixAttention（SGLang），复用已计算的 KV Cache |

**[B] TPOT 过高——生成过程像挤牙膏**

Decode 阶段每步只生成一个 token，矩阵退化成向量运算，是典型的 memory-bound 操作。优化方向有三个：

| 子症状 | 诊断方法 | 推荐方案 |
|--------|---------|---------|
| KV Cache 搬运成为带宽瓶颈 | Nsight Compute 查看 Attention kernel 的 memory throughput 是否接近 HBM 带宽上限 | 使用 FlashAttention / FlashInfer 减少无效显存访问；优化 KV Cache 布局 |
| 逐 token 串行生成的本质限制 | 单请求 TPOT 已逼近理论下限，但用户仍嫌慢 | Speculative Decoding（Medusa / EAGLE-2），用"猜测+验证"打破串行瓶颈 |
| 并发不足，GPU 算力没喂饱 | GPU 利用率低于 60%，每个 batch 只有少量请求 | 开启 Continuous Batching + 提高并发请求数，把 batch 做大以提升计算密度 |

**[C] 显存不够——OOM 或并发上不去**

显存是推理场景中最稀缺的资源，被两大块占据：模型权重和 KV Cache。根据谁占大头来选择方案：

| 显存大户 | 诊断方法 | 推荐方案 |
|---------|---------|---------|
| KV Cache（长上下文或高并发场景） | 对比模型参数显存和 KV Cache 显存的比例（参考 3.1 的计算方法） | PagedAttention 解决碎片问题；KV Cache 量化（KIVI 2-bit）压缩存储 |
| 模型权重本身 | FP16 权重已经超出单卡容量 | INT4 weight-only 量化（AWQ/GPTQ）可将权重压缩到 1/4；W8A8（SmoothQuant）压缩到 1/2 且 kernel 效率更高 |

**[D] 尾延迟 P95/P99 失控——均值还行但毛刺严重**

这是最棘手的问题，因为它往往不是某个单一环节慢，而是系统层面的资源争抢和调度不当：

| 子症状 | 诊断方法 | 推荐方案 |
|--------|---------|---------|
| Prefill 和 Decode 请求互相干扰 | 在混合负载下对比纯 Decode 和混合场景的 TPOT P95，差距超过 2 倍即确认互扰 | Prefill/Decode 解耦部署（DistServe / Splitwise），让两类请求各用独立 GPU 池 |
| SLO 要求极严格，需要全局最优调度 | 单纯解耦后仍有部分请求超时，或不同请求有差异化 SLO 需求 | TaiChi 类统一调度框架，根据 SLO 优先级动态分配 Prefill/Decode 资源 |

**实际使用建议**

实践中，性能问题往往不止一个症状，可能 TTFT 和显存同时不达标。建议的操作顺序是：

1. **先解决 OOM**（显存问题），否则服务都起不来
2. **再优化 TTFT**，因为首 token 延迟直接影响用户体感
3. **然后提升 TPOT 和吞吐**，这通常需要在延迟和吞吐之间做 trade-off
4. **最后处理尾延迟**，这需要系统架构层面的改动，投入最大

---

## 🧭 新人破局指南

### 推荐学习路径

面对这么庞大的技术栈，建议采取**需求驱动**而非自底向上的学习路径：

**基础阶段（0-3个月）**

1. 完成第零层的全部检验标准：编程语言、数学基础、Transformer 架构、PyTorch 训练流程、通信拓扑基础
2. 学习 CUDA 编程基础，能写出简单的 Reduce / GEMM kernel
3. 尝试用 PyTorch DDP 将训练分布到两张卡上，观察显存和通信变化

**专项深入（3-6个月）**

1. 精读四篇里程碑论文并对照代码：
   - Megatron-LM（TP 与 PP 原理）
   - ZeRO（DeepSpeed 核心）
   - FlashAttention（Memory-aware 算法）
   - vLLM（PagedAttention、KV Cache）
2. 参与开源项目（vLLM、DeepSpeed、SGLang），贡献算子优化或功能模块
3. 掌握量化、Speculative Decoding 等推理优化技术

**工程实践（6个月以上）**

1. 在 GPU 集群上部署百亿 / 千亿参数模型，优化端到端性能
2. 建立完整的性能分析与回归体系
3. 研究 Prefill/Decode 解耦等前沿系统架构
4. 跟踪最新技术迭代（FP8、RDMA 网络优化、新架构适配等）

### 核心思维模型

学习 AI Infra 的过程中，始终牢记一个权衡思维：

| 优化技术 | 牺牲了什么 | 换取了什么 |
|---------|-----------|-----------|
| ZeRO | 通信带宽 | 显存空间 |
| Activation Checkpointing | 计算时间 | 显存空间 |
| 量化 | 精度 | 显存 + 带宽 + 吞吐 |
| Speculative Decoding | Prefill 开销 | Decode 速度 |
| Prefill/Decode 解耦 | 系统复杂度 + KV 迁移开销 | 尾延迟 + goodput |
| FlashAttention | 实现复杂度 | 显存 + 速度 |

---

## 📚 参考资料

### 论文

- **Attention Is All You Need** (Vaswani et al., 2017)：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
- **Megatron-LM** (Shoeybi et al., 2019)：[https://arxiv.org/abs/1909.08053](https://arxiv.org/abs/1909.08053)
- **ZeRO: Memory Optimizations Toward Training Trillion Parameter Models** (Rajbhandari et al., 2019)：[https://arxiv.org/abs/1910.02054](https://arxiv.org/abs/1910.02054)
- **FlashAttention V1** (Dao et al., 2022)：[https://arxiv.org/abs/2205.14135](https://arxiv.org/abs/2205.14135)
- **FlashAttention V2** (Dao, 2023)：[https://arxiv.org/abs/2307.08691](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3** (Shah et al., 2024)：[https://arxiv.org/abs/2407.08691](https://arxiv.org/abs/2407.08691)
- **Flash-Decoding** (Stanford CRFM, 2023)：[https://crfm.stanford.edu/2023/10/12/flashdecoding.html](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- **FlashInfer** (Ye et al., 2025)：[https://arxiv.org/abs/2501.01005](https://arxiv.org/abs/2501.01005)
- **vLLM / PagedAttention** (Kwon et al., 2023)：[https://arxiv.org/abs/2309.06180](https://arxiv.org/abs/2309.06180)
- **SGLang** (Zheng et al., 2023)：[https://arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- **Orca** (Yu et al., 2022)：[https://www.usenix.org/conference/osdi22/presentation/yu](https://www.usenix.org/conference/osdi22/presentation/yu)
- **DistServe** (Zhong et al., OSDI'24)：[https://arxiv.org/abs/2401.09670](https://arxiv.org/abs/2401.09670)
- **Splitwise** (Patel et al., ISCA 2024)：[https://arxiv.org/abs/2311.18677](https://arxiv.org/abs/2311.18677)
- **SmoothQuant** (Xiao et al., 2022)：[https://arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)
- **GPTQ** (Frantar et al., 2022)：[https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ** (Lin et al., 2023)：[https://arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978)
- **KIVI** (Liu et al., 2024)：[https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)
- **Speculative Sampling** (Leviathan et al., 2022 / Chen et al., 2023)：[https://arxiv.org/abs/2302.01318](https://arxiv.org/abs/2302.01318)
- **Medusa** (Cai et al., 2024)：[https://arxiv.org/abs/2401.10774](https://arxiv.org/abs/2401.10774)
- **EAGLE-2** (Li et al., 2024)：[https://arxiv.org/abs/2406.16858](https://arxiv.org/abs/2406.16858)
- **Online normalizer calculation for softmax** (Milakov & Gimelshein, 2018)：[https://arxiv.org/abs/1805.02867](https://arxiv.org/abs/1805.02867)
- **DeepSeek V2 技术报告**：[https://arxiv.org/abs/2405.04434](https://arxiv.org/abs/2405.04434)
- **DeepSeekMoE**：[https://arxiv.org/abs/2401.06066](https://arxiv.org/abs/2401.06066)
- **Towards Efficient Generative LLM Serving: A Survey** (CMU)：[https://arxiv.org/abs/2312.15234](https://arxiv.org/abs/2312.15234)

### 教程与视频

- **The Illustrated Transformer** (Jay Alammar)：[https://jalammar.github.io/illustrated-transformer/](https://jalammar.github.io/illustrated-transformer/)
- **3Blue1Brown：线性代数的本质**：[https://www.3blue1brown.com/topics/linear-algebra](https://www.3blue1brown.com/topics/linear-algebra)
- **Andrej Karpathy：Let's build GPT from scratch**：[https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

### 开源项目与工具

- **vLLM** GitHub：[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)
- **SGLang** GitHub：[https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang)
- **DeepSpeed** GitHub：[https://github.com/microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)
- **Megatron-LM** GitHub：[https://github.com/NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- **FlashAttention** GitHub：[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)
- **FlashInfer** GitHub：[https://github.com/flashinfer-ai/flashinfer](https://github.com/flashinfer-ai/flashinfer)
- **TensorRT-LLM** GitHub：[https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- **Triton** GitHub：[https://github.com/triton-lang/triton](https://github.com/triton-lang/triton)

### 官方文档

- **NVIDIA CUDA Programming Guide**：[https://docs.nvidia.com/cuda/cuda-c-programming-guide/](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- **NVIDIA NCCL 文档**：[https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- **Nsight Systems User Guide**：[https://docs.nvidia.com/nsight-systems/UserGuide/](https://docs.nvidia.com/nsight-systems/UserGuide/)
- **Nsight Compute**：[https://docs.nvidia.com/nsight-compute/ProfilingGuide/](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- **NVIDIA Deep Learning Performance Guide**：[https://docs.nvidia.com/deeplearning/performance/](https://docs.nvidia.com/deeplearning/performance/)
- **DeepSpeed 官方文档**：[https://www.deepspeed.ai/](https://www.deepspeed.ai/)
- **PyTorch 官方教程（60 Minute Blitz）**：[https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- **PyTorch DDP 教程**：[https://pytorch.org/tutorials/intermediate/ddp_tutorial.html](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- **PyTorch FSDP 教程**：[https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- **vLLM 官方文档**：[https://docs.vllm.ai/](https://docs.vllm.ai/)
- **TensorRT-LLM 官方文档**：[https://nvidia.github.io/TensorRT-LLM/](https://nvidia.github.io/TensorRT-LLM/)
- **Triton 官方教程**：[https://triton-lang.org/main/getting-started/tutorials/](https://triton-lang.org/main/getting-started/tutorials/)
- **MLPerf Inference**：[https://mlcommons.org/benchmarks/inference-datacenter/](https://mlcommons.org/benchmarks/inference-datacenter/)
- **GenAI-Perf**：[https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c%2B%2B/perf_analyzer/genai-perf/)
