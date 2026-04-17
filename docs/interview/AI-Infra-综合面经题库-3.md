---
title: "AI Infra 综合面经题库 (3)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-16
company: "综合"
tier: "综合"
interviewType: "未知"
order: 5
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

ai infra八股：
1- 给定训练所需的Tokens，怎么估计模型训练所需的完整时间？
2- Prefill和Decode阶段各有什么优化技术？
3- 什么是Two-batch overlap，什么场景Two-batch overlap是负优化？
4- megatron-lm中通信优化怎么做？
5- 多机PD分离会有KV cache transfer开销，为什么还要做PD分离？
6- muon和AdamW的pretrain和posttrain为什么不能混用？
7- 如何看待跨SM的PD分离和AF分离？
8- cuda的global memory和shared memory访存分别需要注意什么？
9- deepseek-V3的优化点
10- deepseek-DSA和NSA，MoBA的区别
11- nccl中的通信源语有哪些？all-reduce参数更新一次参数需要几次通信？
12- 在小数据量场景使用NVSHMEM，每个GPU直接读取其他GPU的数据，在本地reduce，相比ring all-reduce的好处
13- 训练时如何设计超长序列下的并行
14- 将Ampere架构的算子适配到Hopper架构的卡上，你会对哪些地方进行升级改造？
