---
title: "小厂 AI Infra 实习 (1)"
description: "小厂 AI Infra 实习面试真题，涵盖推理优化等方向"
pubDate: 2026-04-17
company: "小厂"
tier: "T5"
interviewType: "实习"
order: 6023
tags: ["推理优化"]
---

### 推理优化

1. 描述 vLLM 中 scheduler 的调度流程
2. vLLM 中请求被抢占后的后续处理机制
3. 投机采样推理中草稿模型与主模型的交互流程，比较 vLLM 与 SGLang 的实现差异
4. 阐述 GPTQ 量化与 SmoothQuant 的原理
5. 介绍 DeepSeek V3 中 EPLB 的推理机制
6. MLA 在 prefill 与 decode 阶段的计算复杂度差异，以及矩阵吸收优化的原理
7. DeepSeek V3.2 相较前代有哪些创新点
8. SGLang 中多模态场景下开启 TP 时，ViT 的 image embedding 在多个进程间如何高效复用
