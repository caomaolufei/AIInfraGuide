---
title: "快手 AI Infra 实习"
description: "快手 AI Infra 实习面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-16
company: "快手"
tier: "T1"
interviewType: "实习"
order: 2006
tags: ["推理优化", "算子优化"]
---

1. 实习介绍
2. 介绍项目
3. 知不知道其他量化方法
4. 对于量化误差而言，数据应该怎样分布较好
5. 针对有异常值的情况，数据分布越均匀越好，在量化到int8,fp8,int4时都没问题，但是量化到fp4时却不是如此，为什么
6. 熟悉CUDA，描述一下如何优化GEMM，在其中计算时shared memory的大小怎么取
7. 手撕：CUDA写一个norm  input: x[N]  norm = (x – u)/o
u = sum(x[N]), o = sqrt((sum(x - u))^2/N)
