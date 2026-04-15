---
title: "欢迎来到 AIInfraGuide"
description: "AIInfraGuide 正式上线，一个开源、系统、面向实践的 AI 基础设施知识库"
pubDate: 2026-01-10
tags: ["公告", "AI Infra"]
---

## 🚀 为什么需要 AIInfraGuide

大模型时代，AI 基础设施（AI Infra）已经成为支撑训练、推理和服务的核心技术底座。然而，这个领域有一个显著的矛盾——技术迭代极快，但系统化的中文学习资料却严重匮乏。

很多工程师在学习 AI Infra 时面临相似的困境：

- CUDA 编程的入门资料散落各处，缺乏从基础到算子优化的完整路径
- 分布式训练涉及 DDP、FSDP、3D 并行等众多概念，不知道该从哪里开始
- 推理优化技术（PagedAttention、量化、Speculative Decoding）发展迅猛，难以跟上节奏
- 性能分析工具（Nsight Systems、Nsight Compute）功能强大，但上手门槛不低

AIInfraGuide 正是为了解决这些问题而创建的——一个**开源、系统、面向实践**的 AI 基础设施知识库，帮助工程师构建从硬件到软件、从训练到推理的完整知识体系。

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

## 🤝 如何参与

AIInfraGuide 是一个开源项目，欢迎通过以下方式参与共建：

- **提交 Issue**：发现错误、提出建议，或者告诉我们你希望看到的主题
- **贡献 PR**：分享你的实践经验、补充技术细节、改进现有内容
- **Star & Share**：如果觉得有帮助，请在 [GitHub](https://github.com/caomaolufei/AIInfraGuide) 上给个 Star，让更多人发现这个项目

💡 **提示**：如果你不确定从哪里开始，推荐先阅读知识库中的「AIInfra 学习路线」，它会帮你梳理一条清晰的学习路径。

让我们一起构建 AI Infra 社区的知识基础设施。
