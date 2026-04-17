---
title: "遂原科技 AI Infra 实习 一面"
description: "遂原科技 AI Infra 实习一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-17
company: "遂原科技"
tier: "T3"
interviewType: "实习"
round: "一面"
order: 4014
tags: ["推理优化", "算子优化"]
---

以下为遂原科技 AI Infra 实习一面考察内容。

### 项目经历

1. 实习经历深入提问
2. 项目经历深入提问

### 量化相关

3. 量化策略的选择依据：为何选用 INT8 量化，A100 与 H100 对不同量化精度的支持情况
4. 量化对象是模型权重还是 KV-Cache，scale 参数如何确定
5. 量化后是否进行过精度损失的评测

### 算子开发

6. Triton 算子的实现逻辑，包括分块等策略
7. 对比所用的官方 baseline 选择及数据类型
8. 性能提升数据的来源，动态分块策略与算子配置
9. 是否考虑过使用 CUDA 替代 Triton 进行算子开发，选择 Triton 的原因
10. 是否做过 profiling，内存吞吐等性能指标表现如何，后续有何优化思路

### 推理优化

11. Attention 模块在整个系统端到端延迟中所占比例
12. Decode 阶段属于 compute bound 还是 memory bound，KV-Cache 量化提升的是哪方面性能
13. A100 的理论显存带宽上限
