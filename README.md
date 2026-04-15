# AIInfraGuide（内容持续更新中）

> 从零开始深入理解 AI Infra 的核心技术

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

## 🗺️ AIInfra 学习路线

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 0 | [从零理解 AI Infra](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra-introduction/) | AI Infra 的定义、技术栈全貌与核心组件 |
| 1 | [AI Infra 学习路线](https://caomaolufei.github.io/AIInfraGuide/guides/ai-infra学习路线/) | 系统化的学习路径与知识图谱 |

## 📚 一、AIInfra 前置基础

### GPU 与硬件

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [GPU 基础知识：从硬件架构到 AI 计算](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/gpu/gpu-basics/) | CPU vs GPU、SM 架构、显存层级、Tensor Core、CUDA 编程模型 |
| 2 | [NVIDIA GPU 架构演进：从 Volta 到 Blackwell](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/gpu/nvidia-gpu-evolution/) | V100 → A100 → H100 → B200 架构演进与关键特性 |

### Transformer

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 3 | [Transformer 架构：快速入门篇](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/20-transformer架构入门/) | Encoder-Decoder 结构、Self-Attention 机制入门 |
| 4 | [AI Infra 工程师为什么必须懂 Transformer](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/21-ai-infra工程师为什么必须懂transformer/) | 从 Infra 视角理解 Transformer 的计算与显存特性 |
| 5 | [Transformer 全貌及代码实现](https://caomaolufei.github.io/AIInfraGuide/guides/前置知识/transformer/22-transformer全貌及代码实现/) | 完整 Transformer 架构拆解与 PyTorch 实现 |

## ⚡ 二、CUDA 编程与算子优化

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 0 | [CUDA 编程入门指南](https://caomaolufei.github.io/AIInfraGuide/guides/cuda编程与算子优化/cuda编程入门指南/) | CUDA 编程模型、线程层级、内存模型与 Kernel 编写 |

## 🌐 三、分布式训练

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [PyTorch 分布式训练：从原理到实战](https://caomaolufei.github.io/AIInfraGuide/guides/分布式训练/pytorch-distributed/) | DDP、FSDP、ZeRO、通信原语、torchrun 多机训练 |

## 🚀 四、推理优化

| 序号 | 文章 | 说明 |
|:---:|------|------|
| 1 | [vLLM 快速入门](https://caomaolufei.github.io/AIInfraGuide/guides/推理优化/vllm/vllm快速入门/) | 从安装到部署你的第一个 LLM 推理服务 |

## 📊 五、性能分析

> 🚧 施工中，敬请期待...

---

## 🎯 面试宝典

| 梯队 | 公司 | 特点 |
|------|------|------|
| **T0 大厂** | 字节跳动、阿里巴巴、腾讯、百度 | HC 大、面经多、面试标准化 |
| **T1 大厂/独角兽** | 快手、拼多多、美团、蚂蚁、OPPO、华为、蔚来 | 部分部门HC紧张但薪资竞争力强 |
| **T2 AI 独角兽** | MiniMax、阶跃星辰、智谱AI、面壁智能、月之暗面 | 初创氛围、技术前沿、成长空间大 |
| **T3 芯片/硬件厂商** | 英伟达、摩尔线程、海光、寒武纪、壁仞科技、飞腾 | 偏底层、CUDA/编译器方向 |
| **T4 自动驾驶/车企** | 小鹏汽车、蔚来、理想、大疆车载、卓驭、小马智行、元戎启行 | AI Infra + 自动驾驶场景 |
| **T5 其他** | 科大讯飞、网易、海康威视、联想、猿辅导、好未来等 | 各有特色岗位 |

---

## 项目结构

```
AIInfraGuide/
├── docs/                    # 内容文件（Markdown）
│   ├── guides/              # 知识库文章
│   │   ├── 前置知识/
│   │   │   ├── gpu/
│   │   │   └── transformer/
│   │   ├── CUDA编程与算子优化/
│   │   ├── 分布式训练/
│   │   └── 推理优化/
│   └── posts/               # 博客文章
├── src/                     # 网页代码
├── public/                  # 静态资源
└── astro.config.mjs         # Astro 配置
```

## 🤝 如何参与

AIInfraGuide 是一个开源项目，欢迎通过以下方式参与共建：

- **提交 Issue**：发现错误、提出建议，或者告诉我们你希望看到的主题
- **贡献 PR**：分享你的实践经验、补充技术细节、改进现有内容
- **Star & Share**：如果觉得有帮助，请在 [GitHub](https://github.com/caomaolufei/AIInfraGuide) 上给个 Star，让更多人发现这个项目

💡 **提示**：如果你不确定从哪里开始，推荐先阅读知识库中的「AIInfra 学习路线」，它会帮你梳理一条清晰的学习路径。

让我们一起构建 AI Infra 社区的知识基础设施。

## License

MIT
