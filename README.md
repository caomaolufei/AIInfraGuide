# AIInfraGuide（内容持续更新中）

> 从零开始深入理解 AI Infra 的核心全栈技术

**在线浏览：[https://caomaolufei.github.io/AIInfraGuide/](https://caomaolufei.github.io/AIInfraGuide/)**

AI Infra 正在成为大模型时代最关键的工程能力之一。本项目系统梳理从 GPU 硬件到分布式训练、从 CUDA 编程到推理优化的完整技术栈，帮助工程师构建扎实的 AI 基础设施知识体系。

<div align="center">
  <img src="https://img.shields.io/github/stars/caomaolufei/AIInfraGuide.svg?style=flat" alt="Stars" />
  <img src="https://img.shields.io/github/forks/caomaolufei/AIInfraGuide.svg?style=flat" alt="Forks" />
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" />
</div>

<img src="https://caomaolufei.github.io/AIInfraGuide/images/AIinfraGuideWeb.png" alt="AIinfraGuideWeb" style="max-width: 100%; display: block; margin: 0 auto;" />

---

<br>

## 🚀 为什么需要 AIInfraGuide

大模型时代，AI 基础设施（AI Infra）已经成为支撑训练、推理和服务的核心技术底座。然而，这个领域有一个显著的矛盾——技术迭代极快，但系统化的中文学习资料却严重匮乏。

很多工程师在学习 AI Infra 时面临相似的困境：

- CUDA 编程的入门资料散落各处，缺乏从基础到算子优化的完整路径
- 分布式训练涉及 DDP、FSDP、3D 并行等众多概念，不知道该从哪里开始
- 推理优化技术（PagedAttention、量化、Speculative Decoding）发展迅猛，难以跟上节奏
- 性能分析工具（Nsight Systems、Nsight Compute）功能强大，但上手门槛不低

AIInfraGuide 正是为了解决这些问题而创建的——一个**开源、系统、面向实践**的 AI Infra 知识库，帮助工程师构建从硬件到软件、从训练到推理的完整知识体系。

<br>

## 📖 知识库内容体系

知识库围绕 6 大核心主题，覆盖 AI Infra 工程师需要掌握的关键技术栈：

| 主题 | 涵盖内容 |
|---|---|
| **AIInfra 学习路线** | 系统化的学习路径、知识图谱、推荐资源 |
| **AIInfra 前置基础** | GPU 架构、NVIDIA 架构演进、计算机体系结构、Transformer 基础 |
| **CUDA 编程与算子优化** | CUDA 编程模型、Kernel 开发、FlashAttention、Triton |
| **分布式训练** | DDP、FSDP、ZeRO、3D 并行、DeepSpeed、Megatron-LM |
| **推理优化** | vLLM、SGLang、TensorRT-LLM、量化技术、KV Cache 优化 |
| **性能分析** | Nsight Systems/Compute、Roofline 模型、Profiling 实战 |

每篇文章都遵循「先白话后术语」的写作原则——先用通俗的语言解释"是什么、为什么需要"，再给出严谨的技术细节，确保读者既看得懂也学得对。

<br>

## 0️⃣ AIInfra 学习路线

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 0 | [从零理解 AI Infra](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra-introduction/) | AI Infra 的定义、技术栈全貌与核心组件 |
| 1 | [AI Infra 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra学习路线/) | 系统化的学习路径与知识图谱 |

<br>

## 1️⃣ AIInfra 前置基础

### GPU 与硬件

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [GPU 基础知识：从硬件架构到 AI 计算](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/gpu/gpu-basics/) | CPU vs GPU、SM 架构、显存层级、Tensor Core、CUDA 编程模型 |
| 2 | [NVIDIA GPU 架构演进：从 Volta 到 Blackwell](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/gpu/nvidia-gpu-evolution/) | V100 → A100 → H100 → B200 架构演进与关键特性 |
| 3 | 🚧 施工中，敬请期待 ..... |

### Transformer

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 3 | [Transformer 架构：快速入门篇](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/20-transformer架构入门/) | Encoder-Decoder 结构、Self-Attention 机制入门 |
| 4 | [AI Infra 工程师为什么必须懂 Transformer](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/21-ai-infra工程师为什么必须懂transformer/) | 从 Infra 视角理解 Transformer 的计算与显存特性 |
| 5 | [Transformer 全貌及代码实现](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/22-transformer全貌及代码实现/) | 完整 Transformer 架构拆解与 PyTorch 实现 |
| 6 | 🚧 施工中，敬请期待 ..... |

<br>

## 2️⃣ CUDA 编程与算子优化

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 0 | [CUDA 编程入门指南](https://caomaolufei.github.io/AIInfraGuide/guides/cuda编程与算子优化/cuda编程入门指南/) | CUDA 编程模型、线程层级、内存模型与 Kernel 编写 |
| 1 | 🚧 施工中，敬请期待 ..... |


<br>

## 3️⃣ 分布式训练（岗位少，且只有大厂有）

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [PyTorch 分布式训练：从原理到实战](https://caomaolufei.github.io/AIInfraGuide/guides/分布式训练/pytorch-distributed/) | DDP、FSDP、ZeRO、通信原语、torchrun 多机训练 |
| 2 | 🚧 施工中，敬请期待 ..... |

<br>

## 4️⃣ 推理优化

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [vLLM 快速入门](https://caomaolufei.github.io/AIInfraGuide/guides/推理优化/vllm/vllm快速入门/) | 从安装到部署你的第一个 LLM 推理服务 |
| 2 | 🚧 施工中，敬请期待 ..... |

<br>

## 5️⃣ 性能分析

> 🚧 施工中，敬请期待...

---

<br>

## 🎯 面试宝典

**面试题：**

> 🚧 施工中，敬请期待...

<br>

**投递总览：**

| 梯队 | 公司 | 主要 AI Infra 岗位 |
|------|------|------|
| **T0 大厂** | 字节跳动、阿里巴巴、腾讯、百度 | AI Infra 工程师、高性能计算研发、推理优化工程师、**分布式训练框架** |
| **T1 大厂/独角兽** | 快手、拼多多、美团、蚂蚁、OPPO、华为、蔚来 | AI Infra、高性能计算、大模型推理优化、AI 平台开发 |
| **T2 AI 独角兽** | MiniMax、阶跃星辰、智谱AI、面壁智能、月之暗面 | AI Infra、大模型算法 (偏 Infra)、推理系统开发、Agent Infra |
| **T3 芯片/硬件厂商** | 英伟达、摩尔线程、海光、寒武纪、壁仞科技、飞腾 | 算子开发、CUDA 优化、GPU 软件工程师、AI 编译器、高性能计算 |
| **T4 自动驾驶/车企** | 小鹏汽车、蔚来、理想、大疆车载、卓驭、小马智行、元戎启行 | AI Infra、高性能计算、大模型推理优化、AI 平台 |
| **T5 其他** | 科大讯飞、网易、海康威视、联想、猿辅导、好未来等 | AI Infra、高性能计算、推理引擎开发、大模型算法 |

> 注：分布式训练相关的岗位特别少，一般只有搞大模型的大厂才有，所有大家在准备的求职的过程中，可以重点关注算子优化、推理优化、推理框架这类岗位，训练相关的技术点了解即可，不用深钻。

<br>

**投递时间线：**

| 时间段 | 投递策略 |
|--------|----------|
| **3-4月** | 暑期实习提前批（字节、阿里、腾讯、快手）|
| **5-6月** | 暑期实习正式批 + 日常实习（美团、拼多多、中小厂）|
| **7-8月** | 秋招提前批（字节 AML、百度、华为）|
| **9-10月** | 秋招正式批（全面投递）|
| **11-12月** | 秋招补录 + 春招准备 |
| **次年2-4月** | 春招（HC 较少，竞争激烈）|

<br>

**按个人情况针对性投递：**

| 背景 | 推荐投递 |
|------|----------|
| **CUDA/C++ 底层强** | 英伟达、摩尔线程、壁仞、百度昆仑芯、北京智源 |
| **系统/分布式强** | 字节 AML、阿里云 PAI、百度、快手、美团 |
| **算法+工程兼备** | MiniMax、阶跃星辰、智谱、拼多多 |
| **想做推理优化** | 快手(偏推理)、猿辅导、理想、好未来、阿里云 PAI |
| **想做训练框架** | 字节豆包、百度 PaddlePaddle、华为昇腾 |
| **偏自动驾驶** | 小鹏、蔚来、大疆车载、卓驭、小马智行 |
| **求稳** | 华为、荣耀、海康、中科曙光 |

---

<br>


## 🤝 欢迎参与贡献

AIInfraGuide 是一个开源项目，欢迎通过以下方式参与共建：

- **提交 Issue**：发现错误、提出建议，或者告诉我们你希望看到的主题
- **贡献 PR**：分享你的实践经验、补充技术细节、改进现有内容
- **Star & Share**：如果觉得有帮助，请在 [GitHub](https://github.com/caomaolufei/AIInfraGuide) 上给个 Star，让更多人发现这个项目

💡 **提示**：如果你不确定从哪里开始，推荐先阅读知识库中的「AIInfra 学习路线」，它会帮你梳理一条清晰的学习路径。

让我们一起构建 AI Infra 社区的知识基础设施。

## License

MIT
