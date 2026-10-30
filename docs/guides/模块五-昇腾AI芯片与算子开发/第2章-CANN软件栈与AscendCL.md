---
title: "第2章：CANN 软件栈与 AscendCL"
description: "理解 CANN 异构计算架构的分层设计，掌握 AscendCL 编程模型（Device/Context/Stream/内存管理），学会 aclnn 标准算子调用与 torch_npu 迁移，走通昇腾应用开发全流程"
pubDate: 2026-10-25
category: "ascend"
order: 20
tags: ["CANN", "AscendCL", "aclnn", "torch_npu"]
---

## 本章简介

硬件（第1章）之外，昇腾最需要学习的就是它的软件栈。NVIDIA 有 CUDA，昇腾有 **CANN**——一套从编译器、运行时到算子库的完整异构计算软件栈。

**CANN 全栈架构**先帮你建立分层认知：驱动固件、运行时、算子库、图编译器、框架适配，一层层叠起来，最终把 MindSpore / PyTorch / 上层应用「翻译」成昇腾芯片上的计算。

**AscendCL 编程模型**是应用开发的主入口。我们会学习 Device / Context / Stream / 内存管理这些基础概念，写出第一个完整跑通的昇腾程序——你会发现它的流程和 CUDA 惊人地相似。

**aclnn 标准算子调用**介绍昇腾的「算子库 API」：用 `aclnnXxxGetWorkspaceSize` + `aclnnXxx` 三步完成一个算子执行，相当于把 CUDA 里 `cublas`/`cudnn` 的调用方式搬到了昇腾。

**torch_npu** 则是 AI Infra 工程师最关心的部分：如何把现有 PyTorch 模型迁移到昇腾 NPU 上训练。我们会讲清楚 `model.to('npu')` 背后发生了什么，以及遇到不支持的算子该怎么办。

> 说明：CANN 版本迭代很快（8.0 / 9.1 等），文中示例基于通用 API 形态编写，具体版本差异请以昇腾社区 CANN 文档为准。
