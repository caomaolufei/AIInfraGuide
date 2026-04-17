---
title: "快手 AI Infra 实习 一面 (3)"
description: "快手 AI Infra 实习一面面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-17
company: "快手"
tier: "T1"
interviewType: "实习"
round: "一面"
order: 2009
tags: ["推理优化", "训练优化", "算子优化"]
---

### 基础知识

1. KV Cache 压缩有哪些常见方法？
2. 分别阐述 MHA、MQA、GQA 的概念，以及 KV 广播的实现方式。Multi-head Latent Attention（MLA）与 GQA 之间的数据对应关系是怎样的？给定 hidden_size 和 RoPE 维度，MLA 对应多少个 GQA head？
3. DeepSpeed ZeRO-1/2/3 各阶段分别做了哪些优化？假设模型参数量为 N，使用 Adam 优化器，ZeRO-1 如何在 P 个 GPU 之间分配显存？不同数据类型（FP32/FP16）对显存占用有何影响？
4. SmoothQuant 的原理是什么？为什么需要进行 Smooth 操作？超参数如何确定？如何判断一个模型是否适合 SmoothQuant？若使用逐层激活值分布来判断，应关注 input channel 还是 output channel？
5. AWQ 和 GPTQ 的原理分别是什么？二者有何区别？
6. GPU 分布式通信原语有哪些？All-Gather、All-to-All 各自适用于什么场景？

### 项目经历

7. 详细介绍 KV Cache 稀疏计算的实现细节及 vLLM Triton kernel 的写法。KV Cache 稀疏计算为何不采用掩码方式？
8. 项目中为何针对不同场景选用不同的量化方法？GPTQ 和 SmoothQuant 分别适用于什么场景？
9. 蒸馏模型的具体实现流程是怎样的？使用了哪些关键技术？
