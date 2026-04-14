---
title: "vLLM 推理引擎详解"
description: "深入了解 vLLM 的 PagedAttention 机制和高效 LLM 推理服务架构"
pubDate: 2024-04-01
category: "inference-serving"
order: 1
tags: ["vLLM", "推理", "LLM"]
---

## vLLM 简介

vLLM 是一个高性能的 LLM 推理和服务引擎，其核心创新是 **PagedAttention** 机制，可以有效管理 KV Cache 显存。

### 为什么 KV Cache 管理很重要？

在自回归 LLM 推理中，KV Cache 会随着序列长度线性增长。传统方案为每个请求预分配最大长度的连续显存，导致严重浪费。

## PagedAttention

PagedAttention 借鉴了操作系统的虚拟内存分页机制：

- 将 KV Cache 分割成固定大小的 **Page**（块）
- 使用 **Page Table** 管理物理显存到逻辑块的映射
- 支持非连续存储，消除内存碎片

这使得 vLLM 的吞吐量比传统方案提升 **2-4 倍**。

## 快速开始

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="meta-llama/Llama-2-7b-chat-hf",
    tensor_parallel_size=2,  # 2 GPU 并行
    gpu_memory_utilization=0.9,
)

# 生成
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

outputs = llm.generate(["请解释什么是 AI Infra"], sampling_params)
print(outputs[0].outputs[0].text)
```

### 作为 API 服务部署

```bash
# 启动 OpenAI 兼容 API 服务
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --tensor-parallel-size 2 \
    --port 8000
```

## 关键特性

| 特性 | 说明 |
|------|------|
| PagedAttention | 高效 KV Cache 管理 |
| Continuous Batching | 动态批处理，提升吞吐 |
| Tensor Parallelism | 多 GPU 推理 |
| Quantization | GPTQ、AWQ、FP8 量化 |
| Prefix Caching | 共享前缀 KV Cache |

## 总结

vLLM 通过 PagedAttention 等创新机制，大幅提升了 LLM 推理效率，已成为生产环境中最受欢迎的推理引擎之一。
