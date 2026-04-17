---
title: "快手 AI Infra 一面"
description: "快手 AI Infra 一面面试真题，涵盖推理优化、算子优化、高性能计算等方向"
pubDate: 2026-04-16
company: "快手"
tier: "T1"
interviewType: "未知"
round: "一面"
order: 2005
tags: ["推理优化", "算子优化", "高性能计算"]
---

1. 拷打项目
2. 有没有了解过AF分离，他是为了解决什么问题，既然有PD分离了，为什么还要AF分离？
3. 有没有读过flash attention的代码，V2比起V1做了哪些改进？细聊一下他是怎么改进的。有没有了解最近的V4版本？
4. 大模型的一层有几个线性层？TP的时候怎么切的？这样子做的原因是什么？有什么思路优化中间的allreduce吗？
5. 看过ray的底层实现吗？它有什么特性，你的课题研究中是怎么使用ray的？
6. 聊一下你所找到的cuda gemm的优化方法
7. leetcode 单词接龙
