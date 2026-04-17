---
title: "AI Infra 综合面经题库 (3)"
description: "AI Infra 面试真题，涵盖推理优化、训练优化、算子优化、高性能计算等方向"
pubDate: 2026-04-17
company: "综合"
tier: "综合"
interviewType: "未知"
order: 5
tags: ["推理优化", "训练优化", "算子优化", "高性能计算"]
---

## 基础知识

1. CUDA Global Memory 与 Shared Memory 在访存时分别需要关注哪些问题？
2. 已知训练所需的 Token 总量，如何估算模型完成训练的总耗时？
3. Prefill 阶段与 Decode 阶段各有哪些主流优化技术？
4. Two-batch overlap 的含义是什么？在哪些场景下 Two-batch overlap 反而会成为负优化？
5. Megatron-LM 中的通信优化是如何实现的？
6. 多机 PD 分离会引入 KV Cache 传输开销，为何仍有必要进行 PD 分离？
7. Muon 优化器与 AdamW 在 Pretrain 和 Post-train 阶段为何不能混合使用？
8. 如何看待跨 SM 的 PD 分离与 AF 分离方案？
9. DeepSeek-V3 有哪些关键优化点？
10. DeepSeek-DSA、NSA 与 MoBA 之间的区别是什么？
11. NCCL 中包含哪些通信原语？执行一次 All-Reduce 参数更新需要几次通信？
12. 在小数据量场景下使用 NVSHMEM，让每个 GPU 直接读取其他 GPU 的数据并在本地进行 Reduce，相比 Ring All-Reduce 有何优势？
13. 训练超长序列时应如何设计并行策略？
14. 将 Ampere 架构上的算子迁移适配到 Hopper 架构时，哪些方面需要进行升级改造？
