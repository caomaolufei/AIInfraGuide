---
title: "字节跳动 AI Infra 实习 一面 (1)"
description: "字节跳动 AI Infra 实习一面面试真题，涵盖算子优化等方向"
pubDate: 2026-04-17
company: "字节跳动"
tier: "T0"
interviewType: "实习"
round: "一面"
order: 1047
tags: ["算子优化"]
---

## 算子优化

1. 是否有 kernel 级别的优化经验？例如使用 CUTE DSL 或手写 CUDA 实现算子融合，请具体介绍。
2. 进行 kernel fusion 时，通常倾向于采用哪种实现方式？
3. 是否遇到过 fusion 后性能反而下降的情况？原因是什么？
4. Hopper 架构中 warp specialization 的机制是什么？底层如何实现？
5. 如果去掉 warp specialization，仅保留 tiling 和 shared memory 优化，性能损失主要体现在哪些方面？

## 推理优化

6. 在 MoE 模型（如 RL 场景中）的推理优化方面是否有实践经验？
7. 如何判断 MoE 模型确实学到了专家分工，而非仅仅将 dense 模型拆分？
8. 在 RL + MoE 训练中，是否遇到过 reward 导致 routing 退化的情况（即所有请求集中到少数 expert）？如何处理？

## 项目经历

9. 请介绍你的项目经历。
10. 请介绍你的实习经历，包括上线运行的推理服务做过哪些优化。
11. 是否尝试过使用 Agent 生成 CUDA kernel？具体方案是什么？
