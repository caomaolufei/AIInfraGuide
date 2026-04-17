---
title: "字节跳动 AI Infra 实习 一面 (1)"
description: "字节跳动 AI Infra 实习一面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-15
company: "字节跳动"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1047
tags: ["算子优化"]
---

给我面没招了，发点面经攒攒人品～
1.项目拷打
2.实习拷打
3.像那种rl里面的那个MOE之类的那种的优化有去做过吗
4.看您的训练经验比较丰富，而且您上线运行的推理内容之前也进行过一些什么样的优化吗？
5.有没有做过 kernel级别的优化？比如用 CUTE DSL或者手写 CUDA去做 fusion这类算子融合优化，介绍一下
6像底层，如果你们在做.kernel fusion，倾向于用什么方式来做
7.有没有哪次你做了 fusion 结果性能反而下降的？原因是什么
8.平时写 CUDA的时候，有没有关注到底层实现细节？比如你刚提到 FA2，那再往下一层，像 Hopper架构里那个 warp specialization是什么，它底层大概是怎么实现的
9.试过用 Agent去生成cuda内核么，怎么去做的
10.如果我把 warp specialization 去掉，只保留 tile 和 shared memory 优化，大概会损失在哪？
11.怎么么判断一个 MoE 模型是真的学到了分工，而不是只是把 dense模型拆开了
12.在 RL + MoE 里，有没有遇到过 reward把 routing学坏的情况？就是模型为了拿 reward，全都走某几个 expert，这种情况你当时是怎么处理的
