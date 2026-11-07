---
title: "第4章：HCCL 集合通信与分布式训练"
description: "掌握 HCCL 集合通信库的通信原语与算法，理解多卡多机拓扑与 HCCL 配置，并通过 MindSpore 与 torch_npu 两条路径走通昇腾分布式训练实战"
pubDate: 2026-10-25
category: "ascend"
order: 40
tags: ["HCCL", "集合通信", "分布式训练", "AllReduce", "昇腾"]
---

## 本章简介

单卡装不下大模型，多卡协作就离不开集合通信。在 NVIDIA 生态里这个角色由 **NCCL** 扮演；在昇腾生态里，对应的就是 **HCCL（Huawei Collective Communication Library）**。

**通信原语与算法**先讲清楚 HCCL 提供什么：AllReduce、AllGather、ReduceScatter、AlltoAll 这些原语分别解决什么问题，Ring 等通信算法如何在大规模下保持高效——这些概念与 NCCL 高度一致，有分布式基础的同学会非常顺手。

**多卡多机拓扑与配置**回答「怎么连、怎么配」：单机内走 HCCS 高速互连，跨机走 RoCE，HCCL 环境变量与拓扑感知如何工作，以及常见的连接与超时问题怎么排查。

**分布式训练实战**把知识落地：分别用 MindSpore 和 torch_npu（PyTorch）两条主流路径，跑通昇腾上的数据并行训练，并介绍 3D 并行（DP/TP/PP）在昇腾上的对应配置方式。

> 学习前提：建议先完成本模块第 1 章（多卡互联）与前置知识模块的「集合通信基础」章节。HCCL 的算法与原语部分与 NCCL 几乎一一对应，可对照学习。
