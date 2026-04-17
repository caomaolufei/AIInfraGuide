---
title: "元戎启行 AI Infra 校招 一面 (1)"
description: "元戎启行 AI Infra 校招一面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-16
company: "元戎启行"
tier: "T4"
interviewType: "校招"
round: "一面"
order: 5020
tags: ["算子优化"]
---

一面
自我介绍
经历和项目拷打
有定义新的 mlir dialect 吗
做项目时有参考 torch-mlir 吗
有考虑动态图的问题吗
转成 tosa 和 tensor 后，接着 lower 到哪些 dialect
One-shot bufferization 和基于 dialect bufferization 你有了解到吗
了解 llvm 的 isa 和 dyn_cast 吗
做题
给定一个计算图，计算运行该计算图所需的最小内存
反问
做什么的：面试官是偏图编译优化的
面评：总体还是比较满意的，但各个项目和实习都可以再做深入一点
二面
自我介绍
写过 cuda 吗
了解静态图和动态图吗，他们之间的区别是什么
mlir 中怎么处理 in-place 操作
做题：拓扑排序
反问：
做什么的：ai 编译和 cuda kernel 都做，偏车端部署
面评：经历还是很丰富的，后续可以多写写 kernel
