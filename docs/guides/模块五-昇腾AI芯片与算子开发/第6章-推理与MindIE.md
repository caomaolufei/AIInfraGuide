---
title: "第6章：推理与 MindIE"
description: "掌握昇腾推理的软硬件栈与模型转换（ATC/OM），学习 MindIE 推理引擎的核心机制与部署流程，掌握 vLLM-Ascend 适配与大模型推理，以及昇腾量化与 KV Cache 优化方法"
pubDate: 2026-10-25
category: "ascend"
order: 60
tags: ["MindIE", "推理", "vLLM-Ascend", "量化", "KV Cache"]
---

## 本章简介

训练模型是「造子弹」，推理部署才是「开枪」。昇腾生态的推理体系，是国产大模型落地的关键一环。这一章系统覆盖昇腾推理的四个层面。

**推理软硬件栈与模型转换**先建立地基：从训练模型（MindIR/ONNX）到昇腾可执行的 OM 模型，ATC 编译器做了什么，推理服务器与推理卡（Atlas 800I A2/A3 等）如何分工。

**MindIE 推理引擎**介绍昇腾自研推理引擎的核心机制：图编译优化、连续批处理（Continuous Batching）、KV Cache 管理、分布式推理，以及如何部署一个 MindIE 服务。

**vLLM-Ascend 适配**是目前大模型推理的「主流入口」：vLLM 官方社区维护的昇腾后端插件，让 Qwen 等主流模型开箱即用。我们会讲清安装、配置与迁移要点。

**量化与 KV Cache 优化**是推理性能的放大器：W8A8/AWQ/FP8 等量化方案在昇腾上的支持情况、PagedAttention 与 KV Cache 量化如何显著提升吞吐，以及长序列推理的优化手段。

> 学习前提：建议先掌握第 2 章（CANN 与 AscendCL）和第 4 章（HCCL）。对照本仓库模块四（推理优化）的通用原理学习效果更佳。
