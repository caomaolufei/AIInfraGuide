---
title: "NVIDIA GPU 架构演进"
description: "从 Volta 到 Blackwell，梳理 NVIDIA GPU 架构的演进历程及其对 AI 训练推理的影响"
pubDate: 2024-02-01
category: "gpu-hardware"
order: 2
tags: ["GPU", "NVIDIA", "架构"]
---

## 架构演进时间线

NVIDIA 近年来的 GPU 架构针对 AI 工作负载进行了持续优化：

### Volta (2017)
- 首次引入 **Tensor Core**，专为矩阵运算设计
- 代表产品：V100（16/32GB HBM2）
- 标志着 GPU 从通用计算向 AI 专用加速的转变

### Ampere (2020)
- 第三代 Tensor Core，支持 TF32、BF16、INT8
- 引入 **MIG**（Multi-Instance GPU）技术
- 代表产品：A100（40/80GB HBM2e）

```bash
# 查看 MIG 配置
nvidia-smi mig -lgip
```

### Hopper (2022)
- 第四代 Tensor Core，支持 FP8 训练
- 引入 **Transformer Engine**，自动混合精度
- **NVLink 4.0**：900 GB/s 互联带宽
- 代表产品：H100（80GB HBM3）

### Blackwell (2024)
- 第五代 Tensor Core
- **NVLink 5.0**：1.8 TB/s
- 支持百万级 token 上下文推理
- 代表产品：B200、GB200

## Tensor Core 的重要性

Tensor Core 是 NVIDIA GPU 加速 AI 工作负载的核心：

| 精度 | Volta | Ampere | Hopper |
|------|-------|--------|--------|
| FP16 | 125 TFLOPS | 312 TFLOPS | 989 TFLOPS |
| TF32 | - | 156 TFLOPS | 495 TFLOPS |
| FP8 | - | - | 1,979 TFLOPS |

## 总结

选择合适的 GPU 架构需要综合考虑工作负载特性、预算和生态支持。理解架构演进有助于做出更好的基础设施决策。
