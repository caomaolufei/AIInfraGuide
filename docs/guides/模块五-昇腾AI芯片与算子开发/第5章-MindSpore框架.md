---
title: "第5章：MindSpore 框架"
description: "理解 MindSpore 的架构设计与图模式编译原理，掌握昇腾上的训练实战流程，并学会 PyTorch 生态与第三方框架向昇腾迁移的方法"
pubDate: 2026-10-25
category: "ascend"
order: 50
tags: ["MindSpore", "昇思", "图模式", "训练实战", "框架迁移"]
---

## 本章简介

昇腾的「亲儿子」框架是 **MindSpore（昇思）**——华为自研的 AI 框架，与昇腾硬件深度绑定。虽然 torch_npu 让 PyTorch 也能跑在昇腾上，但 MindSpore 在图编译、算子融合、分布式策略等方面与昇腾硬件配合最深。

**架构与图模式**先讲 MindSpore 的独特设计：PyNative（动态图）与 Graph（静态图）两种模式、MindIR 中间表示、以及「图编译 → 算子下发」的执行路径——理解这些，你就能看懂 MindSpore 为什么性能强、以及报错出现在哪一环。

**训练实战**给出一个从数据集到模型、再到训练器与评估的完整 MindSpore 训练流程，并介绍动态图/静态图切换对训练的影响。

**生态迁移**回答现实问题：存量 PyTorch 代码怎么迁到 MindSpore？官方提供了 PyTorch API 兼容层与迁移工具，第三方框架（PaddlePaddle、vLLM 系）在昇腾上又是什么状态？这一章给你一张清晰的迁移路线图。

> 学习前提：建议掌握前置知识模块的 PyTorch 框架基础（理解训练循环、模型定义、数据集接口），有 PyTorch 经验即可无缝对照学习。
