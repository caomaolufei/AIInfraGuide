---
title: "科大讯飞 AI Infra 校招 一面"
description: "科大讯飞 AI Infra 校招一面面试真题，涵盖AI Infra等方向"
pubDate: 2026-04-16
company: "科大讯飞"
tier: "T5"
interviewType: "校招"
round: "一面"
order: 6010
tags: []
---

整体面试还是不错的，但是没后续了，不知道是哪里出了问题
项目经历拷打
体系结构基础：
1. 浮点数：BF16 vs FP16 的区别？为什么大模型训练用BF16？
2. 内存：大端 vs 小端？主流架构是什么？（x86小端）
3. CPU设计：如果设计一个CPU，从哪几个部分考虑？（指令集、流水线五阶段）
4. OS：用户态 vs 内核态？怎么切换？（系统调用）
框架与NPU：
1. NPU开发难点和策略
2. Softmax优化：如何解决负载不均衡？
3. TP原理：Tensor Parallel切分的是什么？涉及哪些通信？
