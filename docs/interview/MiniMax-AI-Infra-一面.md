---
title: "MiniMax AI Infra 一面"
description: "MiniMax AI Infra 一面面试真题，涵盖训练优化等方向"
pubDate: 2026-04-16
company: "MiniMax"
tier: "T2"
interviewType: "未知"
round: "一面"
order: 3006
tags: ["训练优化"]
---

1.Transformer 中 Attention 的本质是什么？从数学角度解释一下。
2.了解Agent吗？把RAG做成Agent有什么好处
3. 在 Agent 多轮对话任务中，Attention 的局限性体现在哪些方面？
4.介绍 一下SFT 的核心流程以及数据集的构建策略是怎么样的。
5. SFT 之后常见的 Post-Training（如 RLHF）还有哪些？它们之间的目的有何区别？
5.什么是 RAG？它是怎么提升生成质量的？标准RAG有什么问题与传统“检索 + 模型生成”的流程有何不同？
6.如何评估一个RAG系统是否真正 work？有哪些具体的指标或框架？
7.PPO和DPO 在大模型对齐中的主要区别是什么？DPO 训练通常有哪些注意事项？
8.是否了解或使用过 GRPO 算法？
9. 项目里的 Modular Agent 是如何实现Multi-step Planning的？
10. 项目中工具调用的调度策略是如何设计的？是否有异常 fallback策略？
11. Agent评估体系包括哪些维度？如何衡量规划能力 vs 幻觉率？
12.在微调Qwen 模型时，选择的训练阶段和 Loss 函数是如何决定的？
13. Prompt 自动推荐模块用了哪些优化策略？有没有尝试过 Prompt 压缩或 Embedding 表示的方式？
14. 场景题： 假如一个 Agent 推理链路包含 3 个工具 + 高频请求，导致系统整体延迟较高，你会如何进行工程优化？
15. 说一下LoRA的原理；LoRA完推理的时候要挂着Adaptor吗？
16手撕代码：torch写SFT的loss计算代码（注意shift right
