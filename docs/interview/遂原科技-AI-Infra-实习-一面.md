---
title: "遂原科技 AI Infra 实习 一面"
description: "遂原科技 AI Infra 实习一面面试真题，涵盖推理优化、算子优化等方向"
pubDate: 2026-04-15
company: "遂原科技"
tier: "T3"
interviewType: "实习"
round: "一面"
order: 4014
tags: ["推理优化", "算子优化"]
---

发一下问题给大家参考，攒攒人品！
1.实习拷打
2.项目拷打
3.量化策略，为什么选择int8量化，A100和H100对不同量化的支持，是量化模型权重还是只是kvcache，scale如何选择，有没有测精度损失
4.triton算子实现逻辑，分块等策略
5.对比的官方baseline选择，数据类型
6.压力质疑提升数据，问attention占整个系统端到端延迟百分比
7.数据提升怎么来的，动态分块策略，算子配置等
8.有没有想过用CUDA开发算子，为什么使用
triton
9.有没有做过profile，测出来一些性能如何，比如memory吞吐，一些后续优化思路
10.decode阶段是 compute bound还是memorybound，kvcache量化提升的是什么
11.A100理论带宽上限
