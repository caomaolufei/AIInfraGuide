---
title: "辉羲智能 AI Infra 实习 一二三面"
description: "辉羲智能 AI Infra 实习一二三面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-15
company: "辉羲智能"
tier: "T4"
interviewType: "实习"
round: "一二三面"
order: 5003
tags: ["算子优化"]
---

Time line
3.12 一面 hr面
3.13 二面 技术面
3.17 三面 技术面
3.24 oc

一面
主要就是聊聊天，介绍公司基本情况以及薪资待遇（正常来说，这不应该三面么）

二面
1.简单介绍一下你自己
2.拷打项目，几种常见卷积算法的优缺点
3.写算子时有碰到bank conflict吗？为什么会发生bank confict以及如何解决？
4.说一下CPU和GPU的架构
5.说一下你对grid，block，thread的理解
6.写算子时如何最大化地利用缓存？
迭代一次的数据尽量符合L1的大小，整个程序的数据尽量符合L2的大小。（当时我的回答）
7.你知道线程束分歧吗？（warp divergent 也叫线程束分化）
8.手撕矩阵乘算子（当时我打开vscode，他看了我写的reduce以及conv2d，于是便叫我写一个矩阵乘）
第一次技术面，有点紧张，在面试官的提示下顺利写出naive版本，然后说自己对后面的优化，以及如何确定最佳分块大小。过程中还问了blockDim.x和gridDim.x最大能开多少。
反问环节

三面
感觉和二面差不多，主要也是拷打项目
不同点：
1.共享内存和cache的区别
2.你了解Tensor core吗？它和CUDA core比加速矩阵乘谁更快？
3.你了解Transformer吗？
4.softmax算法在深度学习中的应用
5.手撕softmax算子（有了经验后，十分顺利）
反问环节

最后祝大家都能顺利找到实习
