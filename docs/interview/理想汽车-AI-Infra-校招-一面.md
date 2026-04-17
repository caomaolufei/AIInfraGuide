---
title: "理想汽车 AI Infra 校招 一面"
description: "理想汽车 AI Infra 校招一面面试真题，涵盖推理优化、训练优化、算子优化等方向"
pubDate: 2026-04-17
company: "理想汽车"
tier: "T4"
interviewType: "校招"
round: "一面"
order: 5005
tags: ["推理优化", "训练优化", "算子优化"]
---

### 基础知识

1. PyTorch DDP 的原理与实现机制是什么？
2. 梳理 CV 领域的发展脉络（从 AlexNet 开始，包括架构和激活函数等方面的演进）。
3. 梳理 NLP 领域的发展路径（RNN -> LSTM -> Transformer）。
4. 对比 RNN 与 Transformer 各自的优缺点，重点说明 Transformer 的并行性优势。
5. 模型训练中有哪些并行方式？（数据并行、张量并行、流水线并行等）
6. Transformer 与 CV 的结合体现在哪里？（如 ViT 的设计思路）
7. 模型轻量化部署有哪些方法？（量化、剪枝、压缩等）
8. TensorRT 的优势与不足分别是什么？缺少算子时如何通过 Plugin 自定义算子来解决？
9. Batch Normalization 的作用是什么？为什么能加快模型收敛？

### 编程题

10. 求从左上角到右下角的最短路径（简单动态规划题）。
