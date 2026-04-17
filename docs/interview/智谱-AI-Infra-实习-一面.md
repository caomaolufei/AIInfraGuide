---
title: "智谱 AI Infra 实习 一面"
description: "智谱 AI Infra 实习一面面试真题，涵盖推理优化等方向"
pubDate: 2026-04-17
company: "智谱"
tier: "T2"
interviewType: "实习"
round: "一面"
order: 3005
tags: ["推理优化"]
---

## 基础知识

1. 请阐述 per-tensor、per-channel、per-group 三种量化粒度的区别，哪种粒度最细？
2. minmax 校准与 percentile 校准的核心差异是什么？
3. 除 minmax 和 percentile 外，还有哪些常见的校准算法？请简述 KL 散度校准和 MSE 校准的基本原理。
4. 请说明 SmoothQuant 的量化粒度及其工作原理，并与 AWQ、GPTQ 的作用流程进行对比。
5. NV FP4 的量化原理是什么？缩放因子在哪个维度上计算？存储格式如何设计？
6. 在实际部署中，常见的量化目标格式有哪些？请对比 FP8 与 NV FP4 的适用场景。

## 项目经历

7. 请介绍你的实习项目经历。

## 编程题

8. 请分别实现 minmax 校准和 percentile 校准算法。
