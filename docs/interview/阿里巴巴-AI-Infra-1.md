---
title: "阿里巴巴 AI Infra (1)"
description: "阿里巴巴 AI Infra 面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "阿里巴巴"
tier: "T0"
interviewType: "未知"
order: 1001
tags: ["算子优化", "高性能计算"]
---

## 项目经历提问

1. 围绕项目进行深入追问

## 理论基础

2. 你是否有过利用 Agent 来自动生成 CUDA kernel 的实践？具体的实现方案是什么？
3. 在大规模模型的训练或部署过程中，有没有用过底层调试工具来排查问题？当千卡集群中出现 NCCL Timeout 时，你会如何定位故障原因并解决？
4. 如果去掉 Warp Specialization 优化，只保留 tile 分块和 Shared Memory 优化，性能下降主要会反映在哪些维度上？
5. 你有没有 kernel 层面的优化经验？比如用 CuTe DSL 或者直接手写 CUDA 来实现 fusion 类算子，请详细说一说
6. 在做 kernel fusion 时，你倾向于选择哪种技术方案？理由是什么？
7. 推理服务正式上线之前，你做过哪些性能方面的优化工作？
8. 编写 CUDA 代码时是否深入了解过底层硬件细节？例如 Hopper 架构下 Warp Specialization 的机制原理与底层实现方式
9. 是否有过对 RL 场景下 MoE 模型的优化经验？
10. 有没有做过 kernel fusion 后性能反而变差的情况？你认为原因出在哪里？
11. 通过什么方式可以判断一个 MoE 模型真正实现了专家间的分工协作，而不是简单地把 Dense 模型做了参数切分？
12. 在 RL + MoE 的训练场景下，是否碰到过 reward 信号引导路由产生偏差的问题（例如模型为了追求更高奖励而将大部分请求路由到少数几个 expert）？当时采取了什么应对措施？
