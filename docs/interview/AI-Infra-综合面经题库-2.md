---
title: "AI Infra 综合面经题库 (2)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
order: 4
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

## 基础知识

1. 请介绍 Flash Attention 的核心原理及实现思路
2. CPU 按列遍历一个行优先存储的矩阵相比按行遍历，性能为何会显著下降？具体是哪个性能指标发生了劣化？
3. GPU 矩阵转置操作中使用 Shared Memory 有何优势？
4. Flow Matching 模型的预测目标是什么？如何理解以数据样本 x0 为条件的条件速度场（conditional velocity）？
5. 如何计算 QwenImage 中的 time shift？
6. Weight-Only 量化有哪些方案？实现 Weight-Only 量化 CUDA kernel 时如何优化访存？是否了解 Marlin kernel？
7. Megatron 中序列并行（SP）的实现方式是什么？
8. DeepSpeed ZeRO Stage 1 与 Stage 2 在通信量上有何差异？论文描述与代码实现之间是否存在差距？
9. 多 GPU 通信场景下 NVSHMEM 与 NVLink 有什么区别？

## PyTorch 编程题

1. 实现 Multi-Head Attention（共 3 道变体）
2. 实现 Flash Attention v1
3. 编写 Flow Matching Model 采样过程的伪代码

## 算法题

1. 快速排序
2. 寻找两个有序数组的中位数
3. 下一个排列
4. 二叉树中的最大路径和
5. Path Sum III
6. 给定若干点的数轴坐标数组和固定数量的等长线段，求线段最少需要多长才能覆盖全部点
7. 前 K 个高频字符串（词频相同时按字典序升序排列）
8. 给定初始字符串 s，每次将字符串向右旋转一位并拼接到末尾（长度每次翻倍），求无限扩展后第 N 个位置的字符
9. 两根手指置于 26 个小写字母组成的键盘上，求敲出给定字符串 s 所需的最少移动距离
