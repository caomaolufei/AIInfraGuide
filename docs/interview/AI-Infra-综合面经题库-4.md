---
title: "AI Infra 综合面经题库 (4)"
description: "AI Infra 面试真题，涵盖算子优化、高性能计算等方向"
pubDate: 2026-04-15
company: "综合"
tier: "综合"
interviewType: "未知"
order: 6
tags: ["算子优化", "高性能计算"]
---

1.项目拷打
2.你在去部署或者训练预训练或者后训练的模型时，有没有用过一些比较底层的一些训练的调试的工具，比如说千卡的话很容易就会出NCCL timeout，如果出现 NCCL timeout，一般怎么定位和解决？
3.像那种rl里面的那个MOE之类的那种的优化有去做过吗
4.看您的训练经验比较丰富，而且您上线运行的推理内容之前也进行过一些什么样的优化吗？
5.有没有做过 kernel级别的优化？比如用 CUTE DSL或者手写 CUDA去做 fusion这类算子融合优化，介绍一下
6像底层，如果你们在做.kernel fusion，倾向于用什么方式来做
7.有没有哪次你做了 fusion 结果性能反而下降的？原因是什么
8.平时写 CUDA的时候，有没有关注到底层实现细节？比如你刚提到 FA2，那再往下一层，像 Hopper架构里那个 warp specialization是什么，它底层大概是怎么实现的
9.试过用 Agent去生成cuda内核么，怎么去做的
